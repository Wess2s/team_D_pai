"""
forkliftBehaviour_base.py
-------------------------
BehaviorScript for the ForkliftB sensor variant.
Attach to /World/forklift_b_sensor via the Behaviour Scripts panel.

Navigation workflow (navGoTo / navEnabled)
───────────────────────────────────────────
1. Assign a /World/WaypointGraph/Nodes/* prim to the `navClosestNode` relationship.
2. Set `navGoTo` (or legacy `navEnabled`) to True.
3. The forklift builds the waypoint graph, runs A*, drives through intermediate
   nodes, and finishes with a tight final_alignment().
4. navGoTo / navEnabled are set to False automatically on arrival.

Pick workflow (navGoToPick)
────────────────────────────
1. Set `navPalletToPickId` to the prim name of the target pallet (e.g. "WH_Palette_01").
   The script will find it in the stage and store it in `navPalletToPick`.
   Alternatively, set `navPalletToPick` directly to skip the search.
2. Set `navGoToPick` to True.
3. The forklift:
   a. Drives to a pre-pick position 3 m along the pallet's local Y axis (closer side).
   b. Final-aligns there (tight pos + heading, same hysteresis logic as navGoTo).
   c. Drives forward to insert the forks.
   d. Raises the forks to 0.5 m and waits.
   e. Sets navWithPallet = True, then reverses to the pre-pick position.
   f. Sets navGoToPick = False when done.

Control booleans (mutually exclusive - only one True at a time):
  navGoTo        - graph navigation to navClosestNode
  navGoToPick    - pick-up sequence
  navGotoDrop    - drop sequence (not yet implemented)
  navWithPallet  - read-only flag, set True after successful pick

forklift_b_sensor joint conventions
─────────────────────────────────────
  back_wheel_drive   angular velocity  - negative = forward, positive = backward
  back_wheel_swivel  angular position  - positive = left (when driving forward)
                                         INVERTED when reversing
  lift_joint         linear position   - higher value = forks up

Node orientation convention
────────────────────────────
  The node prim's local -X axis defines the docking heading.
  Rotate the node sphere around Z in the Stage panel to change it.
"""

from __future__ import annotations

import heapq
import math

import carb
import omni.kit.app
import omni.kit.commands
import omni.usd
from omni.kit.scripting import BehaviorScript
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


class ForkliftBehaviourBase(BehaviorScript):
    """BehaviorScript for forklift_b_sensor: graph navigation, final alignment, and pallet pick."""

    NODES_ROOT = "/World/WaypointGraph/Nodes"
    EDGES_ROOT = "/World/WaypointGraph/Edges"

    # Shared across all instances - tracks which sim frame last cleared the global
    # debug-draw buffers so multiple forklifts don't wipe each other's drawings.
    _dd_lines_cleared_frame: int  = -1
    _dd_points_cleared_frame: int = -1

    # Class-level registry of all active ProximitySensor instances keyed by sensor
    # prim path.  clear_sensors() / register_sensor() operate on a global singleton,
    # so each setup/teardown must rebuild the full list to avoid wiping other
    # forklifts' sensors.
    _active_prox_sensors: dict = {}   # prim_path -> ProximitySensor

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def on_init(self) -> None:
        self._stage = omni.usd.get_context().get_stage()

        self.root         = str(self.prim_path)
        self.body_path    = f"{self.root}/body"
        self.drive_joint  = f"{self.root}/back_wheel_joints/back_wheel_drive"
        self.swivel_joint = f"{self.root}/back_wheel_joints/back_wheel_swivel"
        self.lift_joint   = f"{self.root}/lift_joint"

        self.delta_time: float = 0.0

        # Graph navigation state
        self._nav_path: list[str]                = []
        self._nav_idx: int                       = 0
        self._aligning: bool                     = False
        self._align_dir: int                     = 1
        self._turning_at_node: bool              = False
        self._preturn_target_yaw: float          = 0.0
        self._preturn_pivot: tuple[float, float] = (0.0, 0.0)
        self._nav_reverse: bool                  = False   # reverse toward current target

        self._last_nav_idx_checked: int          = -1      # dynamic clearance rate-limit
        self._node_monitor_timer: float          = 0.0    # countdown to next ahead-scan

        # Pick state machine
        self._pick_phase: str                        = "idle"
        self._prepick_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._prepick_yaw: float                     = 0.0
        self._lift_timer: float                      = 0.0
        self._lift_target_pos: float                 = 0.0
        self._pallet_path: str                       = ""

        # Drop state machine
        self._drop_phase: str                         = "idle"
        self._predrop_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._predrop_yaw: float                      = 0.0
        self._drop_target_path: str                   = ""
        self._drop_z: float                           = 0.0
        self._drop_retreat_start: tuple[float, float] = (0.0, 0.0)
        self._drop_pallet_dist_before: float           = 0.0   # pallet↔body dist at lowering end
        self._pick_retreat_start: tuple[float, float] = (0.0, 0.0)

        # Lateral overshoot guard (used inside _align_to_pose)
        # _lat_overshoot_side: 0=normal, +1=recovering from +lateral, -1=recovering from -lateral
        # _initial_lateral_sign: sign of lateral when alignment started (to detect axis crossing)
        # _lat_last: lateral value from the previous recovery frame (divergence detection)
        self._lat_overshoot_side: int    = 0
        self._initial_lateral_sign: int  = 0
        self._lat_last: float            = 0.0

        # Crab-walk state (used in no_overshoot mode of _align_to_pose)
        # 0=approach/idle, 1=back+pos-swivel, 2=back+neg-swivel, 3=forward-to-Y0
        self._lateral_phase: int            = 0
        self._lateral_steer_sign: int       = 1
        # Heading at the moment phase 2 starts - used to detect worsening heading
        self._crab_phase2_start_heading: float = 0.0
        # Adaptive crab-walk steering: starts at pick_align_steer, decays each time
        # the walk direction reverses so the forklift converges instead of oscillating.
        # 0.0 = not yet initialised for the current alignment session.
        self._crab_steer: float   = 0.0
        self._crab_last_sign: int = 0
        # Retreat flag: set when we cross Y=0 with bad heading; cleared at along=1.0
        self._retreat_for_hdg: bool   = False
        # Drop nav: set True once we've triggered navigation toward the drop transform node
        self._drop_nav_triggered: bool = False
        # Last-seen values of navPalletToPickId / navAreaToDrop for change detection.
        # Initialised from the actual attribute in on_play to avoid false reroutes.
        self._prev_pallet_to_pick_id: str = ""
        self._prev_area_to_drop: str      = ""
        # Accumulated extra fork lowering from failed drop retries (reset on success or Play)
        self._drop_lower_offset_extra: float = 0.0
        # Locked yaw for final_alignment - chosen once at alignment start, stable for the run
        self._final_align_locked_yaw: float | None = None
        # Home navigation: True while forklift is autonomously returning to navHome
        self._going_home: bool       = False
        # USD path of navHome (stored so final backward drive knows the destination)
        self._home_path: str         = ""
        # True during the final backward drive from navClosestNode to navHome
        self._going_home_final: bool = False
        # In-place pivot: start yaw captured on first call, None when idle
        self._pivot_start_yaw: float | None = None
        # 180° flip K-turn state: None=idle, "back"=reversing+turning, "go_prev"=forward to prev node
        self._flip_phase: str | None = None
        self._flip_start_yaw: float  = 0.0
        # Previous node position captured when flip starts (None if single-node path)
        self._flip_prev_node_pos: tuple[float, float, float] | None = None
        # True while the forklift approaches the final node backward after a flip
        self._post_flip_align: bool  = False
        # Dock-picking state: set while the current pallet lives in /World/Pallets/Docking
        self._picking_from_dock: bool  = False
        self._dock_path: str           = ""
        # True once we've done the predock alignment; cleared each time pick restarts
        self._dock_at_predock: bool    = False
        # Retreat target in XY - predock + dock_retreat_extra metres, set on pick start
        self._dock_retreat_target_pos: tuple[float, float] = (0.0, 0.0)
        # navArea of the active drop target - captured when drop phase begins, used for reparenting
        self._drop_nav_area: str       = ""
        # Which attribute type matched: "navRacksPosId" | "navBufferPosId" (empty = unknown)
        self._drop_area_attr_type: str = ""
        # Body position at the start of the "picking" drive - used to measure distance driven
        self._picking_start: tuple[float, float] = (0.0, 0.0)
        # Short prim name used as log prefix on every message.
        self._name: str = self.root.split("/")[-1]

        # Proximity sensor state
        self._prox_sensor      = None
        self._proximity_blocked: bool  = False
        self._prox_clear_frames: int   = 0     # consecutive frames with no detection (hysteresis)
        self._prox_block_local_x: float = 0.0  # local X of blocking object (+= in front, -= behind)
        self._prox_block_local_y: float = 0.0  # local Y of blocking object (+= left, -= right)
        self._prox_block_axis: str      = "X"  # axis that triggered the block ("X" or "Y")
        self._prox_block_path: str      = ""   # USD path of the detected obstacle
        self._rerouted_for_pallet: bool = False  # True after reroute attempt; reset on clear
        # Crab-retreat rack guard: track start position and max allowed distance
        self._crab_retreat_start_pos: tuple[float, float] = (0.0, 0.0)
        self._crab_retreat_max_dist: float = float("inf")
        # Debug draw interface (lazy-initialised in _draw_path_highlight)
        self._debug_draw_iface             = None
        self._debug_positions_active: bool = False
        self._highlight_active: bool = False   # True while lines are being drawn
        # True while driving through the node during a node-align radius clamp correction
        self._align_returning_to_node: bool = False

        # Rack avoidance maneuver: straight clear then pivot toward nav node
        self._rack_avoid_phase: str               = ""   # "" | "clearing" | "pivoting"
        self._rack_avoid_start_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._rack_avoid_clear_dist: float = 2.0  # m to travel before post-clear action
        # What to do after clearing: "pivot" (align to next node), "resume" (continue current activity)
        self._rack_avoid_after: str        = "pivot"
        self._rack_avoid_clear_spd: float         = 0.0  # drive() speed for clearing
        self._rack_avoid_block_path: str          = ""   # USD path of the detected rack
        self._prox_sensor_path: str = f"{self.body_path}/sensors/proximity_sensor"
        # Body half-extents used for zone detection; set from BBoxCache in _setup_proximity_sensor
        self._prox_half_body_x: float = 1.5   # half body length along X (forward axis)
        self._prox_half_body_y: float = 0.6   # half body width  along Y (side axis)
        self._prox_body_sz: float     = 2.0   # full body height (for overlap_box Z extent)
        # USD traversal cache for static collision prims (racks, walls) - built once.
        # ProximitySensor only returns dynamic rigid bodies; static actors need USD traversal.
        self._static_coll_prims: list[str] = []
        self._collision_cache_built: bool   = False
        # Candidate paths from the last static scan (refreshed at _static_prox_interval)
        self._static_prox_extra: list[str] = []
        self._static_prox_timer: float     = 0.0   # countdown to next static scan
        # Full USD path of the pallet root prim currently on the forks (e.g. /World/Pallets/.../pallet_04)
        # Set when lift raises, cleared when drop completes
        self._carried_pallet_root: str = ""

        self._init_config()
        self.create_missing_attributes()
        self.stop()

        self._print(f"Initialized on {self.root}")

    def on_play(self) -> None:
        self._nav_path                = []
        self._nav_idx                 = 0
        self._aligning                = False
        self._align_dir               = 1
        self._turning_at_node         = False
        self._nav_reverse             = False
        self._last_nav_idx_checked    = -1
        self._pick_phase        = "idle"
        self._drop_phase        = "idle"
        self._node_align_label  = "prepick"  # "prepick" or "predock" - kept for per-frame log
        self._lat_overshoot_side   = 0
        self._initial_lateral_sign = 0
        self._lat_last             = 0.0
        self._lateral_phase        = 0
        self._lateral_steer_sign   = 1
        self._crab_steer           = 0.0
        self._crab_last_sign       = 0
        self._retreat_for_hdg         = False
        self._drop_nav_triggered      = False
        self._drop_lower_offset_extra = 0.0
        self._prev_pallet_to_pick_id  = ""
        self._prev_area_to_drop       = ""
        self._final_align_locked_yaw  = None
        self._going_home              = False
        self._home_path               = ""
        self._going_home_final        = False
        self._pivot_start_yaw         = None
        self._flip_phase             = None
        self._flip_start_yaw         = 0.0
        self._flip_prev_node_pos     = None
        self._post_flip_align        = False
        self._picking_from_dock      = False
        self._dock_path              = ""
        self._dock_at_predock        = False
        self._dock_retreat_target_pos = (0.0, 0.0)
        self._drop_nav_area          = ""
        self._drop_area_attr_type    = ""
        self._picking_start          = (0.0, 0.0)
        self._proximity_blocked = False
        self._setup_proximity_sensor()
        self._load_config_from_settings()
        # Initialise lift joint drive parameters.
        lift_prim = self._stage.GetPrimAtPath(self.lift_joint)
        if lift_prim and lift_prim.IsValid():
            drive = UsdPhysics.DriveAPI.Get(lift_prim, "linear")
            if drive:
                drive.GetDampingAttr().Set(1000000.0)
                drive.GetStiffnessAttr().Set(1000000.0)
                drive.GetMaxForceAttr().Set(500000.0)
        self.stop()
        # Resolve string IDs -> USD relationships, then auto-start the full sequence
        # via navGoTo (graph navigation) rather than enabling navGoToPick directly.
        # Sequence: navGoTo -> (node reached) -> navGoToPick -> (pick done) ->
        #           navGoTo -> (node reached) -> navGoToDrop -> (drop done)
        self._refresh_pallet_ref()
        self._refresh_drop_transform()
        pallet_path = self._get_pallet_prim_path()
        if pallet_path:
            self._auto_start_pick_nav(pallet_path)

        # Snapshot current string IDs so on_update change-detection doesn't fire on first frame.
        self._prev_pallet_to_pick_id = self._get_str("navPalletToPickId")
        self._prev_area_to_drop      = self._get_str("navAreaToDrop")

    def on_pause(self) -> None:
        self.stop()

    def on_stop(self) -> None:
        self._nav_path                = []
        self._nav_idx                 = 0
        self._aligning                = False
        self._turning_at_node         = False
        self._pick_phase              = "idle"
        self._drop_phase              = "idle"
        self._lat_overshoot_side      = 0
        self._initial_lateral_sign    = 0
        self._lat_last                = 0.0
        self._lateral_phase           = 0
        self._lateral_steer_sign      = 1
        self._crab_steer              = 0.0
        self._crab_last_sign          = 0
        self._retreat_for_hdg         = False
        self._align_returning_to_node = False
        self._drop_nav_triggered      = False
        self._drop_lower_offset_extra = 0.0
        self._prev_pallet_to_pick_id  = ""
        self._prev_area_to_drop       = ""
        self._final_align_locked_yaw  = None
        self._going_home              = False
        self._home_path              = ""
        self._going_home_final       = False
        self._flip_phase             = None
        self._flip_start_yaw         = 0.0
        self._flip_prev_node_pos     = None
        self._post_flip_align        = False
        self._picking_from_dock      = False
        self._dock_path              = ""
        self._dock_at_predock        = False
        self._dock_retreat_target_pos = (0.0, 0.0)
        self._drop_nav_area          = ""
        self._drop_area_attr_type    = ""
        self._picking_start          = (0.0, 0.0)
        self._proximity_blocked   = False
        self._prox_clear_frames   = 0
        self._carried_pallet_root = ""
        self._set_bool("navPathBlocked", False)
        self._set_bool("navUnreachableTarget", False)
        self._set_str("navObjectDetected", "None")
        self._set_str("navObjectDistance", "")
        prim_r = self._stage.GetPrimAtPath(self.root)
        if prim_r and prim_r.IsValid():
            attr = prim_r.GetAttribute("navFinalRoute")
            if attr and attr.IsValid():
                attr.Set("")
        self._teardown_proximity_sensor()
        # Clear debug path lines and position dots on stop
        if self._debug_draw_iface is not None:
            try:
                self._debug_draw_iface.clear_lines()
            except Exception:
                pass
            self._clear_debug_positions()
        self._debug_positions_active = False
        prim = self._stage.GetPrimAtPath(self.root)
        if prim and prim.IsValid():
            attr = prim.GetAttribute("navGoTo")
            if attr and attr.IsValid():
                attr.Set(False)
            attr = prim.GetAttribute("navGoToPick")
            if attr and attr.IsValid():
                attr.Set(False)
            attr = prim.GetAttribute("navGoToDrop")
            if attr and attr.IsValid():
                attr.Set(False)
            attr = prim.GetAttribute("navGoHome")
            if attr and attr.IsValid():
                attr.Set(False)
            attr = prim.GetAttribute("navWithPallet")
            if attr and attr.IsValid():
                attr.Set(False)
            rel = prim.GetRelationship("navPalletToPick")
            if rel:
                rel.SetTargets([])
            rel = prim.GetRelationship("navDropTransform")
            if rel:
                rel.SetTargets([])
            rel = prim.GetRelationship("navClosestNode")
            if rel:
                rel.SetTargets([])
            rel = prim.GetRelationship("navGoToNode")
            if rel:
                rel.SetTargets([])
            for str_attr in ("navPalletToPickId", "navAreaToDrop"):
                a = prim.GetAttribute(str_attr)
                if a and a.IsValid():
                    a.Set("")
        self.stop()

    def on_update(self, current_time: float, delta_time: float) -> None:
        self.delta_time = delta_time
        self._node_monitor_timer  -= delta_time
        self._static_prox_timer   -= delta_time

        self._check_proximity()
        self._check_target_changes()

        highlight = self._get_bool("navHighlightPath")
        if highlight:
            self._draw_path_highlight()
            self._highlight_active = True
        elif self._highlight_active:
            # Attribute was just turned off - clear the lines immediately
            if self._debug_draw_iface is not None:
                try:
                    self._debug_draw_iface.clear_lines()
                except Exception:
                    pass
            self._highlight_active = False

        if self._get_bool("navHighlightPrePos"):
            self._draw_debug_positions()
            self._debug_positions_active = True
        elif self._debug_positions_active:
            self._clear_debug_positions()
            self._debug_positions_active = False

        # While carrying a pallet, continuously poll navAreaToDrop so a route is
        # calculated as soon as an area is assigned - even mid-travel.
        if self._get_bool("navWithPallet"):
            self._poll_drop_area()

        # ── Global avoidance — preempts pick, drop, AND nav ───────────────────
        # If an avoidance maneuver is already running, service it before anything else.
        if self._rack_avoid_phase:
            self._rack_avoid_maneuver()
            return

        # Trigger a new avoidance on ANY detection — flip, align, pick approach, nav,
        # regardless of travel direction.  Direction filters are intentionally removed:
        # "in the moment we detect the rack, move away" with no exceptions.
        # Re-triggering mid-maneuver is blocked by the _rack_avoid_phase check above.
        # Forklift detection uses a separate behaviour handled in the nav branch below.
        is_pick_drop = self._get_bool("navGoToPick") or self._get_bool("navGoToDrop")
        if self._proximity_blocked:
            obj_type = self._get_str("navObjectDetected")
            in_front = self._prox_block_local_x >= 0

            if obj_type != "Forklift":
                _REROUTABLE = {"Pallet", "Box"}
                if obj_type not in _REROUTABLE:
                    # Fixed structure — resume after clearing so pick/drop/alignment retries.
                    after = "resume" if (is_pick_drop or self._aligning or self._turning_at_node) else "pivot"
                elif not self._rerouted_for_pallet:
                    # Dynamic obstacle (pallet/box) — resume during pick/drop; reroute during nav.
                    after = "resume" if is_pick_drop else "reroute"
                else:
                    after = None   # already rerouted: fall through

                if after is not None:
                    self._rack_avoid_phase      = "clearing"
                    self._rack_avoid_start_pos  = self._world_pos(self.body_path)
                    self._rack_avoid_clear_spd  = abs(self.move_speed) if in_front else self.move_speed
                    self._rack_avoid_clear_dist = self.proximity_separation_dist
                    self._rack_avoid_after      = after
                    self._rack_avoid_block_path = self._prox_block_path
                    direction = "backing" if in_front else "forward"
                    side      = "front"  if in_front else "back"
                    context   = "pick/drop" if is_pick_drop else "nav"
                    self._print(
                        f"Object avoid [{context}]: {direction} {self.proximity_separation_dist:.1f}m "
                        f"- {obj_type} {side} -> {after}"
                    )
                    return
                # after == None: already rerouted, fall through.

            else:
                # Forklift detected — stop and wait in all contexts (specific behaviour TBD).
                self.stop()
                return
        # ── End global avoidance ───────────────────────────────────────────────

        # A new pick order interrupts home navigation
        if self._going_home and self._get_bool("navGoToPick"):
            self._cancel_home_nav()

        if self._get_bool("navGoToPick"):
            self._update_pick()
        elif self._get_bool("navGoToDrop"):
            self._update_drop()
        elif self._nav_enabled():
            if self._going_home_final:
                self._home_final_drive()
            elif self._aligning:
                with_pallet = self._get_bool("navWithPallet")
                if not with_pallet:
                    self._refresh_pallet_ref()
                if not with_pallet and self._get_pallet_prim_path():
                    self._final_align_for_pick()
                elif with_pallet and self._has_drop_transform():
                    self._final_align_for_drop()
                else:
                    self.final_alignment()
            elif self._nav_path:
                self._follow_path()
            else:
                self._start_navigation()
        elif self._get_bool("navWithPallet"):
            pass  # _poll_drop_area (above) owns all routing while carrying
        else:
            self._check_home_nav()

    # ── Config ─────────────────────────────────────────────────────────────

    def _init_config(self) -> None:
        """Default movement, tolerance, and pick parameters."""
        self.move_speed: float     = -700.0
        self.approach_speed: float = -300.0
        self.align_speed: float    =  -150.0
        self.slow_distance: float  =    5.0


        # Intermediate node arrival (loose)
        self.node_reach_distance: float = 2.0

        # Hand-off to final alignment
        self.arrival_distance: float = 1.5

        # Final alignment tolerances (tight)
        self.final_pos_tol: float    = 0.20
        self.final_angle_tol: float  = 3.0
        self.align_hysteresis: float = 0.4
        # Heading error threshold to trigger K-turn flip before final alignment
        self.flip_hdg_threshold: float = 150.0
        # How close to the previous node go_prev must get before switching to backward approach
        self.go_prev_reach_distance: float = 0.5

        # Pre-turn parameters
        self.preturn_threshold_deg: float      = 5.0   # full-precision turn threshold (penultimate node)
        self.preturn_threshold_deg_loose: float = 30.0  # loose turn threshold for all other nodes
        # If the next segment requires more than this many degrees of turn, travel backwards.
        # While reversing, switch back to forward only when turn_needed drops below this.
        self.nav_reverse_threshold_deg: float  = 90.0
        # How often (seconds) to scan ALL remaining path nodes for obstacles.
        self.node_monitor_interval: float      = 1.0
        self.preturn_lateral_threshold: float = 0.5
        self.preturn_lookahead_dist: float    = 3.0
        # Maximum distance the forklift may drift from the pivot node during a pivot maneuver
        self.max_pivot_radius: float          = 1.0
        self.turn_gain: float       = 100.0
        self.max_turn: float        = 55.0
        # Cross-track correction applied during _align_to_pose: adds a steering
        # contribution proportional to the lateral offset from the approach axis so
        # the forklift homes onto the target X line instead of driving parallel to it.
        self.lateral_turn_gain: float = 10.0

        # Lateral overshoot guard thresholds.
        # When heading error > angle_thresh: wide band (heading still being corrected)
        # When heading error ≤ angle_thresh: tight band (heading already close, fine position)
        self.align_lateral_angle_thresh: float  = 30.0
        self.align_lateral_limit: float         = 0.6   # trigger (wide)
        self.align_lateral_recover_dist: float  = 0.5   # recover on opposite side (wide)
        self.align_lateral_tight_limit: float   = 0.2   # trigger (tight)
        self.align_lateral_tight_recover: float = 0.2   # recover on opposite side (tight)

        # Pick parameters
        self.pick_prepick_dist: float      = 3.0   # distance from pallet center to pre-pick point (m)
        self.pick_insert_offset: float     = 1.8   # fallback: stop when body this close to pallet (m)
        self.pick_insert_drive_dist: float = 1.5   # primary: stop after driving this far from prepick (m)
        self.pick_retreat_tol: float   = 0.3   # arrival tolerance when reversing to pre-pick (m)
        self.pick_lift_height: float   = 0.5   # target lift position after picking (m)
        self.pick_lift_wait: float     = 2.0   # seconds to hold lift before retreating

        # Pick pre-pick lateral crab-walk (used when no_overshoot=True in _align_to_pose)
        # At Y=0 (along≤0) with heading aligned but lateral off, the forklift backs up with
        # small steer to step sideways, then counter-steers to restore heading, then goes forward.
        self.pick_align_steer: float     = 50.0   # swivel angle for crab-walk (deg)
        self.pick_align_back_dist: float = 0.3    # how far to back up in each crab phase (m)
        # Each time the crab-walk direction reverses the steering angle is multiplied by this factor
        # so the forklift converges to the axis instead of oscillating side-to-side.
        self.crab_steer_decay: float     = 0.65
        # Crab-walk fires when heading error is within this at along=0, even if wider than angle_tol.
        # Looser than final angle_tol so the manoeuvre triggers instead of looping into retreats.
        self.crab_walk_heading_tol: float       = 15.0  # degrees - wider than angle_tol so crab-walk fires
        self.crab_walk_hdg_increase_tol: float =  5.0  # exit phase 2 early if heading worsens by this many degrees
        self.pick_align_pos_tol: float   = 0.03   # max lateral (x) error for pick alignment (m)
        self.pick_align_angle_tol: float = 0.5    # heading must be within this before crab-walk (deg)

        # Pre-pick dwell: after alignment the forklift stops, reads the pallet Z,
        # raises the forks to pallet height + offset, then proceeds to insert.
        self.pick_prepick_stop_wait: float    = 1.0    # dwell time before inserting forks (s)
        self.pick_lift_approach_offset: float = -0.25  # forks offset from pallet Z (negative = below)
        self.pick_lift_min: float             = 0.0    # floor for lift target (joint lower limit)
        self.pick_lift_pos_tol: float         = 0.02   # tolerance to consider lift "in position" (m)

        # Drop parameters
        self.drop_prepick_dist: float         = 3.5    # distance from drop center to pre-drop point (m)
        self.drop_insert_offset: float        = 1.8    # stop when body is this close to drop center (m)
        self.drop_retreat_dist: float         = 3.0    # reverse this many metres straight back after drop
        self.drop_retreat_tol: float          = 0.3    # stop when distance travelled reaches within this (m)
        self.drop_lift_approach_offset: float = 0.0    # lift to drop_Z + this before driving in
        self.drop_lower_offset: float         = -0.08  # lower to drop_Z + this after placing
        self.drop_pallet_check_dist: float    =  1.0   # after retreat: pallet within this -> still on forks (m)
        self.drop_lower_retry_increment: float =  0.1   # extra fork lowering per failed drop retry (m)
        self.drop_lift_wait: float            = 2.0    # seconds to hold lowered lift before retreating
        self.drop_prepick_stop_wait: float    = 1.0    # dwell after aligning before driving in (s)
        self.drop_align_pos_tol: float        = 0.05
        self.drop_align_angle_tol: float      = 0.5
        # If heading error vs predrop_yaw exceeds this on arrival, pivot in place first
        self.predrop_pivot_hdg_threshold: float = 60.0
        # Pivot until heading error drops below this before starting predrop_align
        self.predrop_pivot_angle_tol: float     = 20.0

        # Proximity sensor detection distances (beyond the forklift body extents)
        self.proximity_front_dist_pallet: float   = 5.0   # m front/back detection for pallets
        self.proximity_side_dist_pallet: float    = 0.8   # m side detection for pallets
        self.proximity_front_dist_rack: float     = 1.5   # m front/back detection for racks
        self.proximity_side_dist_rack: float      = 0.8   # m side detection for racks
        self.proximity_front_dist_forklift: float = 5.0   # m front/back detection for other forklifts
        self.proximity_side_dist_forklift: float  = 2.0   # m side detection specifically for forklifts
        self.proximity_front_dist_general: float  = 2.0   # m front/back for gates and unclassified objects
        self.proximity_side_dist: float           = 0.8   # m side for gates and unclassified (general)
        # How far to drive away from any detected object before resuming / rerouting
        self.proximity_separation_dist: float     = 1.0

        # Dock picking parameters
        self.dock_predock_dist: float      = 2.0   # m from dock entrance to pre-dock position
        self.dock_retreat_extra: float     = 4.5   # extra m past predock to reverse after pick
        self.dock_align_lateral_tol: float = 0.5   # m lateral tolerance for matching pallet to dock

        # In-place pivot: very slow forward speed with max steer to minimise turn radius
        self.pivot_speed: float = -50.0

        self._forward_local = Gf.Vec3d(-1.0, 0.0, 0.0)

    # ── USD setup ──────────────────────────────────────────────────────────

    def create_missing_attributes(self) -> None:
        """Create all custom USD attributes and relationships if they don't exist."""
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return

        # Navigation control booleans (all default False)
        for attr_name in ("navEnabled", "navGoTo", "navGoToPick", "navGoToDrop",
                          "navWithPallet", "navGoHome", "navPathBlocked",
                          "navUnreachableTarget", "navHighlightPath"):
            if not prim.GetAttribute(attr_name):
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
                attr.Set(False)
                self._print(f"Created attribute: {attr_name}")

        # navAvoidObjects: enable node-clearance check during route planning (default True)
        if not prim.GetAttribute("navAvoidObjects"):
            attr = prim.CreateAttribute("navAvoidObjects", Sdf.ValueTypeNames.Bool)
            attr.Set(True)
            self._print("Created attribute: navAvoidObjects")

        # navHighlightPrePos: draw green dots at prepick / predrop / predock positions
        if not prim.GetAttribute("navHighlightPrePos"):
            attr = prim.CreateAttribute("navHighlightPrePos", Sdf.ValueTypeNames.Bool)
            attr.Set(False)
            self._print("Created attribute: navHighlightPrePos")

        # Object detected by proximity sensor: "None" | "Pallet" | "Rack" | "Wall" | "Other"
        if not prim.GetAttribute("navObjectDetected"):
            attr = prim.CreateAttribute("navObjectDetected", Sdf.ValueTypeNames.String)
            attr.Set("None")
            self._print("Created attribute: navObjectDetected")

        # Local-frame distance to detected object: "[x.xx, y.yy]" or "" when clear
        if not prim.GetAttribute("navObjectDistance"):
            attr = prim.CreateAttribute("navObjectDistance", Sdf.ValueTypeNames.String)
            attr.Set("")
            self._print("Created attribute: navObjectDistance")

        # Route display string ("9, 17, 10, 11")
        if not prim.GetAttribute("navFinalRoute"):
            attr = prim.CreateAttribute("navFinalRoute", Sdf.ValueTypeNames.String)
            attr.Set("")
            self._print("Created attribute: navFinalRoute")

        # Pallet and drop-area ID strings
        if not prim.GetAttribute("navPalletToPickId"):
            attr = prim.CreateAttribute("navPalletToPickId", Sdf.ValueTypeNames.String)
            attr.Set("")
            self._print(f"Created attribute: navPalletToPickId")

        if not prim.GetAttribute("navAreaToDrop"):
            attr = prim.CreateAttribute("navAreaToDrop", Sdf.ValueTypeNames.String)
            attr.Set("")
            self._print(f"Created attribute: navAreaToDrop")

        # Relationships
        for rel_name in ("navClosestNode", "navGoToNode", "navPalletToPick", "navDropTransform", "navHome"):
            if not prim.GetRelationship(rel_name):
                prim.CreateRelationship(rel_name)
                self._print(f"Created relationship: {rel_name}")

    # ── Proximity sensor ───────────────────────────────────────────────────

    def _load_config_from_settings(self) -> None:
        """Read parameter overrides from carb settings written by the WH Settings extension.

        The extension stores values at /wh_sim/forklift/{forklift_name}/{param}.
        Any attribute present in carb settings overwrites the in-code default.
        If no setting exists for a param the default remains unchanged.
        This is called on on_init() so changes applied via the extension UI take
        effect on the next Play without needing a script reload.
        """
        try:
            s = carb.settings.get_settings()
            prefix = f"/wh_sim/forklift/{self._name}"
            applied: list[str] = []
            for attr in (
                "move_speed", "approach_speed", "align_speed", "pivot_speed",
                "slow_distance", "node_reach_distance", "arrival_distance",
                "node_monitor_interval", "turn_gain", "max_turn",
                "lateral_turn_gain", "max_pivot_radius", "final_angle_tol",
                "align_hysteresis", "flip_hdg_threshold",
                "preturn_threshold_deg", "preturn_threshold_deg_loose",
                "nav_reverse_threshold_deg",
                "pick_prepick_dist", "pick_insert_offset", "pick_lift_height",
                "pick_lift_wait", "pick_lift_approach_offset", "pick_lift_min",
                "pick_align_steer", "pick_align_angle_tol",
                "pick_prepick_stop_wait", "pick_retreat_tol",
                "crab_steer_decay", "crab_walk_heading_tol",
                "crab_walk_hdg_increase_tol", "dock_retreat_extra",
                "drop_insert_offset", "drop_retreat_dist", "drop_retreat_tol",
                "drop_lift_approach_offset", "drop_lower_offset",
                "drop_pallet_check_dist", "drop_lower_retry_increment",
                "drop_lift_wait", "drop_prepick_stop_wait",
                "drop_align_angle_tol", "predrop_pivot_hdg_threshold",
                "predrop_pivot_angle_tol",
                "proximity_front_dist_pallet", "proximity_side_dist_pallet",
                "proximity_front_dist_rack",   "proximity_side_dist_rack",
                "proximity_front_dist_forklift", "proximity_side_dist_forklift",
                "proximity_front_dist_general", "proximity_side_dist",
                "proximity_separation_dist",
            ):
                val = s.get(f"{prefix}/{attr}")
                if val is not None:
                    try:
                        setattr(self, attr, float(val))
                        applied.append(attr)
                    except (TypeError, ValueError):
                        pass
            if applied:
                carb.log_warn(
                    f"[WH Settings] {self._name}: applied {len(applied)} param(s) from extension"
                )
        except Exception as exc:
            carb.log_warn(f"[WH Settings] {self._name}: config load failed: {exc}")

    def _setup_proximity_sensor(self) -> None:
        """Enable extension, find or create sensor prim, register sensor.

        If setup_proximity_sensor.py was run first, the prim already exists as a
        proper 'def' in the root layer and is visible in the Stage panel.
        Otherwise it is created here as a temporary 'over' spec.
        """
        try:
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.sensors.physx")
            from isaacsim.sensors.physx import ProximitySensor, clear_sensors, register_sensor

            # Remove this instance's previous sensor from the class registry (if any)
            # before rebuilding the global list - do NOT clear_sensors() here as that
            # would wipe sensors belonging to other forklift instances.
            ForkliftBehaviourBase._active_prox_sensors.pop(self._prox_sensor_path, None)
            self._prox_sensor = None

            prim = self._stage.GetPrimAtPath(self._prox_sensor_path)
            if not (prim and prim.IsValid() and prim.IsDefined()):
                # Author a permanent def prim in the root layer so it is visible
                # in the Stage panel and persists across Play/Stop cycles.
                root_layer = self._stage.GetRootLayer()
                with Usd.EditContext(self._stage, root_layer):
                    # Ensure the intermediate /sensors container exists too
                    sensors_container = str(Sdf.Path(self._prox_sensor_path).GetParentPath())
                    if not self._stage.GetPrimAtPath(sensors_container).IsValid():
                        self._stage.DefinePrim(sensors_container, "Xform")
                    prim = self._stage.DefinePrim(self._prox_sensor_path, "Xform")
                if not prim or not prim.IsValid():
                    carb.log_warn(f"[PROX] ERROR: could not create prim at {self._prox_sensor_path}")
                    return
                carb.log_warn(f"[PROX] created permanent prim at {self._prox_sensor_path}")

            # Compute forklift body bounding box to size the sensor proportionally
            body_prim = self._stage.GetPrimAtPath(self.body_path)
            body_sx, body_sy, body_sz = 3.0, 1.2, 2.0  # safe fallbacks (m)
            if body_prim and body_prim.IsValid():
                try:
                    bbox_cache = UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(), ["default", "render"]
                    )
                    r = bbox_cache.ComputeWorldBound(body_prim).GetRange()
                    if not r.IsEmpty():
                        body_sx = float(r.GetMax()[0] - r.GetMin()[0])
                        body_sy = float(r.GetMax()[1] - r.GetMin()[1])
                        body_sz = float(r.GetMax()[2] - r.GetMin()[2])
                        self._prox_half_body_x = body_sx / 2.0
                        self._prox_half_body_y = body_sy / 2.0
                        self._prox_body_sz     = body_sz
                        carb.log_warn(
                            f"[PROX] body bbox:"
                            f"  X={body_sx:.2f}m  Y={body_sy:.2f}m  Z={body_sz:.2f}m"
                        )
                except Exception as exc:
                    carb.log_warn(f"[PROX] bbox failed ({exc}) - using defaults")

            # Sensor full extents sized to the largest front_dist so the
            # ProximitySensor box covers all possible detections; per-type thresholds
            # are applied later in _check_proximity when we know the object type.
            max_front = max(self.proximity_front_dist_pallet,
                            self.proximity_front_dist_rack,
                            self.proximity_front_dist_forklift,
                            self.proximity_front_dist_general)
            max_side = max(self.proximity_side_dist,
                           self.proximity_side_dist_pallet,
                           self.proximity_side_dist_rack,
                           self.proximity_side_dist_forklift)
            sx = body_sx + 2.0 * max_front
            sy = body_sy + 2.0 * max_side
            sz = body_sz

            # Set transform on the sensor prim
            xformable = UsdGeom.Xformable(prim)
            existing_ops = {op.GetOpName() for op in xformable.GetOrderedXformOps()}
            if "xformOp:translate" not in existing_ops:
                xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            else:
                prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, 0.0))
            if "xformOp:scale" not in existing_ops:
                xformable.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            else:
                prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(sx, sy, sz))

            self._prox_sensor = ProximitySensor(prim)
            # Add to class registry then rebuild the global list so sensors already
            # registered by other forklift script instances are preserved.
            ForkliftBehaviourBase._active_prox_sensors[self._prox_sensor_path] = self._prox_sensor
            clear_sensors()
            for _sensor in ForkliftBehaviourBase._active_prox_sensors.values():
                register_sensor(_sensor)
            carb.log_warn(
                f"[PROX] registered at {self._prox_sensor_path}"
                f"  box=({sx:.2f},{sy:.2f},{sz:.2f})m"
                f"  pallet front={self.proximity_front_dist_pallet}m side={self.proximity_side_dist_pallet}m"
                f"  rack front={self.proximity_front_dist_rack}m side={self.proximity_side_dist_rack}m"
                f"  forklift front={self.proximity_front_dist_forklift}m side={self.proximity_side_dist_forklift}m"
                f"  general front={self.proximity_front_dist_general}m side={self.proximity_side_dist}m"
            )

        except Exception as exc:
            carb.log_warn(f"[PROX] setup EXCEPTION: {exc}")

    def _teardown_proximity_sensor(self) -> None:
        """Remove this instance's sensor and rebuild the global registry for remaining ones."""
        if self._prox_sensor is not None:
            try:
                from isaacsim.sensors.physx import clear_sensors, register_sensor
                ForkliftBehaviourBase._active_prox_sensors.pop(self._prox_sensor_path, None)
                clear_sensors()
                for _sensor in ForkliftBehaviourBase._active_prox_sensors.values():
                    register_sensor(_sensor)
            except Exception:
                pass
            self._prox_sensor = None

    def _reinit_proximity_sensor(self) -> None:
        """Tear down and immediately recreate the proximity sensor.

        Called after a pallet reparent so the sensor's internal prim cache is
        rebuilt from scratch.  Without this, the sensor holds a reference to the
        old (now-invalid) prim path and raises RuntimeError every frame.
        """
        self._teardown_proximity_sensor()
        try:
            self._setup_proximity_sensor()
        except Exception as exc:
            carb.log_warn(f"{self._name}  proximity sensor reinit failed: {exc}")

    def _nav_moving_toward_block(self) -> bool:
        """Return True if the current nav target is in the same direction as the blocking obstacle.

        Side blocks (Y axis) never block forward/backward movement - return False.
        If there is no active path, return True (stop to be safe).
        """
        if self._prox_block_axis == "Y":
            return False   # side obstacle: forward/backward movement is safe

        if not self._nav_path or self._nav_idx >= len(self._nav_path):
            return True    # no path -> conservative stop

        target_pos = self._world_pos(self._nav_path[self._nav_idx])
        my_pos     = self._world_pos(self.body_path)
        my_yaw     = self._world_yaw(self.body_path)
        dx         = target_pos[0] - my_pos[0]
        dy         = target_pos[1] - my_pos[1]
        cos_y      = math.cos(my_yaw)
        sin_y      = math.sin(my_yaw)
        nav_local_x = dx * cos_y + dy * sin_y   # positive = target is in front

        # Moving toward the block when nav direction matches block direction
        block_forward = self._prox_block_local_x >= 0
        nav_forward   = nav_local_x >= 0
        return block_forward == nav_forward

    def _rack_avoid_maneuver(self) -> None:
        """Recover from a rack detected in the front or back threat zone.

        Phases:
          "clearing" - drive straight (no steering) until 2 m of travel from the
                       start position.  Direction is backward if the rack is in front,
                       forward if it is behind.
          "pivoting" - pivot in place (oscillate with align_speed) until the forklift
                       heading matches the direction to the current nav node.
        """
        my_pos = self._world_pos(self.body_path)

        # ── Phase 1: drive away until object is no longer detected ─────────────
        if self._rack_avoid_phase == "clearing":
            traveled = math.hypot(
                my_pos[0] - self._rack_avoid_start_pos[0],
                my_pos[1] - self._rack_avoid_start_pos[1],
            )
            # Keep driving while the object is still detected OR we haven't covered
            # the minimum separation distance yet.  The 5-frame hysteresis in
            # _check_proximity ensures _proximity_blocked is only False once the
            # object is genuinely out of range.
            if self._proximity_blocked or traveled < self._rack_avoid_clear_dist:
                self._print(
                    f"Object avoid clearing  traveled={traveled:.2f}m "
                    f"blocked={self._proximity_blocked}"
                )
                self.drive(self._rack_avoid_clear_spd, 0.0)
                return
            # Object no longer detected AND minimum distance covered — exit clearing.
            self.stop()
            if self._rack_avoid_after == "resume":
                self._rack_avoid_phase = ""
                self._print("Object avoid: object clear -> resuming")
                return
            if self._rack_avoid_after == "reroute":
                self._rack_avoid_phase = ""
                self._print("Object avoid: object clear -> rerouting")
                self._try_reroute_around_pallet()
                return
            self._rack_avoid_phase = "pivoting"
            self._print(f"Object avoid: object clear after {traveled:.1f}m -> pivoting to next node")

        # ── Phase 2: pivot toward next nav node ─────────────────────────────
        if self._rack_avoid_phase == "pivoting":
            if not self._nav_path or self._nav_idx >= len(self._nav_path):
                self._rack_avoid_phase = ""
                self._print("Rack avoid: no nav path - exiting")
                return
            my_yaw     = self._world_yaw(self.body_path)
            tgt        = self._world_pos(self._nav_path[self._nav_idx])
            dx         = tgt[0] - my_pos[0]
            dy         = tgt[1] - my_pos[1]
            target_yaw = math.atan2(dy, dx)
            heading_err = self._wrap(target_yaw - my_yaw)

            if abs(heading_err) <= math.radians(self.preturn_threshold_deg):
                self._rack_avoid_phase = ""
                self._print("Rack avoid: aligned - resuming navigation")
                return

            turn = max(-self.max_turn, min(self.max_turn, heading_err * self.turn_gain))
            # Oscillating micro-motion (same as pre-turn pivot) to allow steering
            along = (dx * math.cos(my_yaw) + dy * math.sin(my_yaw))
            if along > self.align_hysteresis:
                self._align_dir = 1
            elif along < -self.align_hysteresis:
                self._align_dir = -1
            self._print(
                f"Rack avoid pivot  err={math.degrees(heading_err):.1f}°"
            )
            self.drive(self.align_speed * self._align_dir, turn)

    def _try_reroute_around_pallet(self) -> None:
        """Replan the waypoint path excluding nodes near the detected movable obstacle.

        Called once per encounter (gated by _rerouted_for_pallet).  Blocks nodes
        within PALLET_BLOCK_RADIUS of the obstacle position, then reruns A* to the
        original goal node.  If no alternative exists the forklift waits in place.
        """
        PALLET_BLOCK_RADIUS: float = 2.0

        self._rerouted_for_pallet = True   # prevent retrying each frame

        if not self._nav_path:
            return

        goal = self._nav_path[-1]

        adjacency, positions = self._build_nav_graph()
        if not positions:
            return

        # Nodes to exclude: those near the obstacle + the current immediate target
        blocked: set[str] = set()
        if self._prox_block_path:
            obs_pos = self._world_pos(self._prox_block_path)
            blocked = {
                n for n, pos in positions.items()
                if math.hypot(pos[0] - obs_pos[0], pos[1] - obs_pos[1]) <= PALLET_BLOCK_RADIUS
            }
        if self._nav_idx < len(self._nav_path):
            blocked.add(self._nav_path[self._nav_idx])

        if goal in blocked:
            self._print("Reroute (pallet): goal node blocked - waiting")
            return

        free = {k: v for k, v in positions.items() if k not in blocked}
        if not free:
            self._print("Reroute (pallet): no free nodes - waiting")
            return

        my_pos = self._world_pos(self.body_path)
        start  = self._closest_node(my_pos, free)

        filtered_adj = {
            n: [(nb, cost) for nb, cost in neighbors if nb not in blocked]
            for n, neighbors in adjacency.items()
            if n not in blocked
        }

        new_path = self._find_path(start, goal, filtered_adj, positions)
        if not new_path:
            self._print("Reroute (pallet): no alternative path - waiting for obstacle to clear")
            return

        self._nav_path             = new_path
        self._nav_idx              = 0
        self._last_nav_idx_checked = -1

        # Apply the same reverse-vs-forward decision that node-arrival logic uses,
        # but for the very first segment of the rerouted path (current pos -> new_path[0]).
        first_pos = positions.get(new_path[0])
        if first_pos is not None:
            my_yaw      = self._world_yaw(self.body_path)
            dx_f        = first_pos[0] - my_pos[0]
            dy_f        = first_pos[1] - my_pos[1]
            angle_first = self._wrap(math.atan2(dy_f, dx_f) - my_yaw)
            self._nav_reverse = abs(angle_first) > math.radians(self.nav_reverse_threshold_deg)
            if self._nav_reverse:
                self._print(
                    f"Reroute: reversing to first node "
                    f"({math.degrees(abs(angle_first)):.1f}° > {self.nav_reverse_threshold_deg}°)"
                )
        else:
            self._nav_reverse = False

        node_nums = ", ".join(p.split("_")[-1] for p in new_path)
        prim2 = self._stage.GetPrimAtPath(self.root)
        if prim2 and prim2.IsValid():
            attr = prim2.GetAttribute("navFinalRoute")
            if attr and attr.IsValid():
                attr.Set(node_nums)
        names = " -> ".join(p.split("/")[-1] for p in new_path)
        self._print(f"Reroute (pallet blocked): {names}")

    def _direct_moving_toward_block(self, target_xy: tuple[float, float]) -> bool:
        """Like _nav_moving_toward_block but uses an explicit target (for direct drives).

        Returns True only when the forklift is heading toward the obstacle - i.e.
        the target and the block are on the same side of the forklift.
        """
        if self._prox_block_axis == "Y":
            return False  # side obstacle never blocks forward/backward movement
        my_pos = self._world_pos(self.body_path)
        my_yaw = self._world_yaw(self.body_path)
        cos_y  = math.cos(my_yaw)
        sin_y  = math.sin(my_yaw)
        dx     = target_xy[0] - my_pos[0]
        dy     = target_xy[1] - my_pos[1]
        nav_local_x = dx * cos_y + dy * sin_y  # positive = target is ahead
        block_forward = self._prox_block_local_x >= 0
        nav_forward   = nav_local_x >= 0
        return block_forward == nav_forward

    def _classify_object(self, path: str) -> str:
        """Return one of Forklift / Pallet / Box / Rack / Gate / Wall / Other based on the prim path.

        Forklift is checked first so other forklifts are never misclassified as a rack/wall.
        Gate is checked before Wall so loading-dock gates (which also contain 'dock') are not
        misclassified as walls.
        Box is cargo/crate sitting on or near a pallet (keyword 'box', e.g. 'cardbox').
        """
        lp = path.lower()
        if "forklift" in lp:
            return "Forklift"
        if any(kw in lp for kw in ("pallet", "palette", "blockpallet", "product", "crate", "wh_product")):
            return "Pallet"
        if "box" in lp:
            return "Box"
        if any(kw in lp for kw in ("rack", "shelf", "/racks/", "par", "frame")):
            return "Rack"
        if "gate" in lp:
            return "Gate"
        if any(kw in lp for kw in ("wall", "floor", "ground", "ceiling", "dock", "building", "concrete")):
            return "Wall"
        return "Other"

    def _is_carried_pallet(self, path: str) -> bool:
        """Return True if the detected prim is part of the pallet currently on the forks.

        Uses the full USD root path captured at lift time so that any child mesh
        (e.g. /…/pallet_04/blockpallet_b02) is correctly matched via startswith.
        """
        return bool(self._carried_pallet_root) and path.startswith(self._carried_pallet_root)

    def _draw_path_highlight(self) -> None:
        """Draw red debug lines from the forklift through remaining nav nodes to the goal.

        Lines are transient (one frame only) and must be redrawn each update tick.
        Always clears the line buffer first so stale lines from the previous route
        (e.g. pick path) are removed immediately when the path changes or empties.
        Requires the isaacsim.util.debug_draw extension to be enabled.
        """
        # Lazy-acquire the debug draw interface
        if self._debug_draw_iface is None:
            try:
                from isaacsim.util.debug_draw import _debug_draw as _dd
                self._debug_draw_iface = _dd.acquire_debug_draw_interface()
            except Exception as exc:
                carb.log_warn(f"[NAV] navHighlightPath: debug draw unavailable - {exc}")
                return

        dd = self._debug_draw_iface

        # Clear once per sim frame across all forklift instances so multiple forklifts
        # don't wipe each other's lines; then every active instance appends its own.
        frame = omni.kit.app.get_app().get_update_number()
        if ForkliftBehaviourBase._dd_lines_cleared_frame != frame:
            ForkliftBehaviourBase._dd_lines_cleared_frame = frame
            try:
                dd.clear_lines()
            except Exception:
                pass

        if not self._nav_path:
            return

        RED       = (1.0, 0.0, 0.0, 1.0)
        LINE_SIZE = 4.0
        Z_LIFT    = 0.1   # m above floor so lines are visible without floating too high

        # Build ordered point list: forklift -> remaining nav nodes
        fkl = self._world_pos(self.body_path)
        pts: list[carb.Float3] = [carb.Float3(fkl[0], fkl[1], fkl[2] + Z_LIFT)]
        for node_path in self._nav_path[self._nav_idx:]:
            np_ = self._world_pos(node_path)
            pts.append(carb.Float3(np_[0], np_[1], np_[2] + Z_LIFT))

        if len(pts) < 2:
            return

        starts = pts[:-1]
        ends   = pts[1:]
        n      = len(starts)
        dd.draw_lines(starts, ends, [RED] * n, [LINE_SIZE] * n)

    def _draw_debug_positions(self) -> None:
        """Draw a green dot + approach line for the currently active target position.

        Only one position is drawn at a time, matching the current action:
          • Pick approach (nav_to_node / node_align / prepick_drive / prepick_align /
            prepick_stop) -> prepick position (also covers predock).
          • Drop approach (predrop_drive / predrop_pivot / predrop_align /
            predrop_stop) -> predrop position.
          • Any other phase (picking, dropping, retreating, idle) -> nothing drawn.

        Dot  - marks the position.
        Line - extends from the dot along the approach yaw toward the target,
               showing the required alignment axis.
        """
        if self._debug_draw_iface is None:
            try:
                from isaacsim.util.debug_draw import _debug_draw as _dd
                self._debug_draw_iface = _dd.acquire_debug_draw_interface()
            except Exception as exc:
                carb.log_warn(f"[NAV] navHighlightPrePos: debug draw unavailable - {exc}")
                return

        _PICK_PHASES = {"nav_to_dock", "nav_to_node", "node_align",
                        "prepick_drive", "prepick_align", "prepick_stop"}
        _DROP_PHASES = {"predrop_drive", "predrop_pivot", "predrop_align", "predrop_stop"}

        if self._drop_phase in _DROP_PHASES:
            pos, yaw = self._predrop_pos, self._predrop_yaw
        elif self._pick_phase in _PICK_PHASES:
            pos, yaw = self._prepick_pos, self._prepick_yaw
        else:
            # No active approach - nothing to draw this frame; clear dot once per frame.
            frame = omni.kit.app.get_app().get_update_number()
            if ForkliftBehaviourBase._dd_points_cleared_frame != frame:
                ForkliftBehaviourBase._dd_points_cleared_frame = frame
                try:
                    self._debug_draw_iface.clear_points()
                except Exception:
                    pass
            return

        if pos == (0.0, 0.0, 0.0):
            return

        dd      = self._debug_draw_iface
        GREEN   = (0.0, 1.0, 0.0, 1.0)
        DOT_SZ  = 12.0
        LINE_SZ = 3.0
        Z_LIFT  = 0.1
        LINE_LEN = 2.0

        z     = pos[2] + Z_LIFT
        dot   = carb.Float3(pos[0], pos[1], z)
        end_x = pos[0] + LINE_LEN * math.cos(yaw)
        end_y = pos[1] + LINE_LEN * math.sin(yaw)
        end   = carb.Float3(end_x, end_y, z)

        # Clear points once per frame, then all active forklifts append their own dot.
        frame = omni.kit.app.get_app().get_update_number()
        if ForkliftBehaviourBase._dd_points_cleared_frame != frame:
            ForkliftBehaviourBase._dd_points_cleared_frame = frame
            try:
                dd.clear_points()
            except Exception:
                pass
        try:
            dd.draw_points([dot], [GREEN], [DOT_SZ])
        except Exception:
            pass
        try:
            dd.draw_lines([dot], [end], [GREEN], [LINE_SZ])
        except Exception:
            pass

    def _clear_debug_positions(self) -> None:
        """Remove the debug position dot and approach line drawn by _draw_debug_positions."""
        if self._debug_draw_iface is not None:
            try:
                self._debug_draw_iface.clear_points()
            except Exception:
                pass
            try:
                self._debug_draw_iface.clear_lines()
            except Exception:
                pass

    def _build_collision_prim_cache(self) -> None:
        """Traverse the stage once and cache USD paths of all collision prims to scan.

        PhysX overlap queries miss static actors (no rigid_body path) AND sleeping
        dynamic bodies (pallets at rest), so every prim with CollisionAPI is cached
        here and scanned by world-position distance each tick.
        The carried pallet is excluded at detection time by _is_carried_pallet().
        Called lazily on the first proximity scan tick.
        """
        from pxr import UsdPhysics
        _FLOOR_KW = ("floor", "ground", "terrain", "wh_floor")
        _self_root = self.root + "/"
        paths: list[str] = []
        for prim in self._stage.TraverseAll():
            path = str(prim.GetPath())
            if path == self.root or path.startswith(_self_root):
                continue
            if not UsdPhysics.CollisionAPI.Get(self._stage, prim.GetPath()):
                continue
            lp = path.lower()
            # Skip floor/terrain - but NOT pallets on a floor-named sub-path
            is_floor = any(kw in lp for kw in _FLOOR_KW)
            is_pallet = any(kw in lp for kw in ("pallet", "palette", "blockpallet",
                                                  "product", "box", "crate", "wh_product"))
            if is_floor and not is_pallet:
                continue
            if "waypointgraph" in lp or "/nodes/" in lp or "/edges/" in lp:
                continue
            paths.append(path)
        self._static_coll_prims    = paths
        self._collision_cache_built = True
        carb.log_warn(f"[PROX] Collision cache built: {len(paths)} prims (racks + pallets)")

    def _check_proximity(self) -> None:
        """Stop navigation when an obstacle enters the X or Y threat zone.

        Front/back zone (X axis): object within proximity_front_dist of the
        forklift's front or rear face AND within the forklift's body width in Y.

        Side zone (Y axis): object within proximity_side_dist of the forklift's
        left or right face AND within the forklift's body length in X.
        """
        if self._prox_sensor is None:
            return

        # Source 1: ProximitySensor - detects dynamic rigid bodies (pallets, moving objects)
        data: dict = self._prox_sensor.get_data() or {}

        # Source 2: USD-traversal static scan (racks, walls, fixed structures).
        # ProximitySensor / PhysX overlap_box returns empty rigid_body paths for
        # static physics actors, so we scan the stage's CollisionAPI cache instead.
        # Runs at ~10 Hz; results are cached between ticks.
        if self._static_prox_timer <= 0:
            self._static_prox_timer = 0.1
            if not self._collision_cache_built:
                self._build_collision_prim_cache()
            bpos     = self._world_pos(self.body_path)
            max_front = max(self.proximity_front_dist_pallet,
                            self.proximity_front_dist_rack,
                            self.proximity_front_dist_forklift)
            hx       = self._prox_half_body_x + max_front
            hy       = self._prox_half_body_y + max(self.proximity_side_dist,
                                                     self.proximity_side_dist_forklift)
            max_dist = math.hypot(hx, hy) + 1.0   # conservative reject envelope
            seen_set: set[str] = set(data)
            extras: list[str] = []
            for spath in self._static_coll_prims:
                if spath in seen_set:
                    continue
                sp = self._world_pos(spath)
                if math.hypot(sp[0] - bpos[0], sp[1] - bpos[1]) > max_dist:
                    continue
                seen_set.add(spath)
                extras.append(spath)

            # Source 2b: PhysX overlap_box for static actors (walls, gates, large racks).
            # The center-point check above misses large prims when the forklift approaches
            # an edge before the prim center enters max_dist.  This overlap query finds
            # any collider shape - including static bodies with no rigid_body - via
            # _resolve_hit_path's hit.collision fallback.
            try:
                from omni.physx import get_physx_scene_query_interface
                pq = get_physx_scene_query_interface()

                def _overlap_report(hit) -> bool:
                    p = self._resolve_hit_path(hit)
                    if p and p not in seen_set:
                        seen_set.add(p)
                        extras.append(p)
                    return True

                pq.overlap_box(
                    carb.Float3(max_dist, max_dist, self._prox_body_sz * 0.5),
                    carb.Float3(bpos[0], bpos[1], bpos[2] + self._prox_body_sz * 0.5),
                    carb.Float4(0.0, 0.0, 0.0, 1.0),
                    _overlap_report,
                    False,
                )
            except Exception:
                pass

            self._static_prox_extra = extras

        # Merge: ProximitySensor paths + cached static scan paths
        all_candidates: list[str] = list(data) + self._static_prox_extra

        if not all_candidates:
            self._prox_clear_frames += 1
            if self._proximity_blocked and self._prox_clear_frames >= 5:
                carb.log_warn("[PROX] Cleared - resuming navigation")
                self._proximity_blocked   = False
                self._prox_clear_frames   = 0
                self._set_str("navObjectDetected", "None")
                self._set_str("navObjectDistance", "")
                self._prox_block_path     = ""
                self._rerouted_for_pallet = False
            return

        my_pos = self._world_pos(self.body_path)
        my_yaw = self._world_yaw(self.body_path)
        cos_y  = math.cos(my_yaw)
        sin_y  = math.sin(my_yaw)

        # Track nearest valid candidate for continuous distance display.
        nearest_dist:  float = float("inf")
        nearest_label: str   = ""

        for path in all_candidates:
            if path.startswith(self.root):
                continue
            check_prim = self._stage.GetPrimAtPath(path)
            if not check_prim or not check_prim.IsValid():
                continue
            # Skip floor/ground plane prims by name - not by Z position, because
            # pallet prims have their origin at Z ≈ 0 and would be wrongly filtered.
            # Guard: don't skip if the path also looks like a pallet (e.g. /Pallets/Buffer/...).
            lp = path.lower()
            is_floor_name = any(kw in lp for kw in ("floor", "ground", "terrain", "wh_floor"))
            is_pallet_name = any(kw in lp for kw in ("pallet", "palette", "blockpallet", "product", "box", "crate"))
            if is_floor_name and not is_pallet_name:
                continue
            # Ignore self - any prim under our own root prim
            if path.startswith(self.root + "/") or path == self.root:
                continue
            # Ignore the pallet currently carried on the forks
            if self._is_carried_pallet(path):
                continue
            # Ignore everything inside the target pallet's USD scope: the pallet
            # mesh itself, any product boxes stacked on it, and any sub-prims.
            if self._pallet_path:
                pallet_scope = self._pallet_path.rsplit("/", 1)[0]
                if pallet_scope and path.startswith(pallet_scope + "/"):
                    continue

            obj_pos = self._world_pos(path)

            # Object offset in forklift's local frame
            dx      =  obj_pos[0] - my_pos[0]
            dy      =  obj_pos[1] - my_pos[1]
            local_x =  dx * cos_y + dy * sin_y   # along forklift length (forward=+X)
            local_y = -dx * sin_y + dy * cos_y   # along forklift width  (left=+Y)

            ax = abs(local_x)
            ay = abs(local_y)

            # Classify first so we can apply the per-type detection zone.
            obj_type = self._classify_object(path)
            if obj_type == "Forklift":
                front_dist = self.proximity_front_dist_forklift
                side_dist  = self.proximity_side_dist_forklift
            elif obj_type in ("Pallet", "Box"):
                front_dist = self.proximity_front_dist_pallet
                side_dist  = self.proximity_side_dist_pallet
            elif obj_type == "Rack":
                front_dist = self.proximity_front_dist_rack
                side_dist  = self.proximity_side_dist_rack
            elif obj_type == "Gate":
                front_dist = self.proximity_front_dist_general
                side_dist  = self.proximity_side_dist
            elif obj_type == "Wall":
                front_dist = self.proximity_front_dist_rack
                side_dist  = self.proximity_side_dist_rack * 0.5
            else:
                # Unclassified / Other
                front_dist = self.proximity_front_dist_general
                side_dist  = self.proximity_side_dist

            # Front/back zone: within body width in Y and within front_dist beyond body in X
            in_x_zone = ay <= self._prox_half_body_y + side_dist and ax <= self._prox_half_body_x + front_dist
            # Side zone: within body length in X and within side_dist beyond body in Y
            in_y_zone = ax <= self._prox_half_body_x + front_dist and ay <= self._prox_half_body_y + side_dist

            # Track nearest candidate for live distance display regardless of zone.
            dist_2d = math.hypot(local_x, local_y)
            if dist_2d < nearest_dist:
                nearest_dist  = dist_2d
                nearest_label = path.split("/")[-1] if obj_type == "Other" else obj_type

            if not (in_x_zone or in_y_zone):
                continue

            axis = "X" if in_x_zone else "Y"
            if not self._proximity_blocked:
                carb.log_warn(
                    f"[PROX] STOP [{obj_type}] - {path.split('/')[-1]}"
                    f"  local_X={local_x:.2f}m  local_Y={local_y:.2f}m"
                    f"  detected on {axis} axis"
                    f"  (X<{self._prox_half_body_x + front_dist:.1f}  Y<{self._prox_half_body_y + side_dist:.1f})"
                )
            self._prox_block_local_x = local_x
            self._prox_block_local_y = local_y
            self._prox_block_axis    = axis
            self._prox_block_path    = path
            detected_label = path.split("/")[-1] if obj_type == "Other" else obj_type
            self._set_str("navObjectDetected", detected_label)
            self._set_str("navObjectDistance", f"[{local_x:.2f}, {local_y:.2f}]")
            self._proximity_blocked = True
            self._prox_clear_frames = 0   # reset hysteresis — object is still present
            return

        # All candidates checked but none triggered proximity.
        # Still publish the nearest candidate distance for live monitoring.

        self._prox_clear_frames += 1
        if self._proximity_blocked and self._prox_clear_frames >= 5:
            carb.log_warn("[PROX] Cleared - resuming navigation")
            self._proximity_blocked   = False
            self._prox_clear_frames   = 0
            self._set_str("navObjectDetected", "None")
            self._set_str("navObjectDistance", "")
            self._prox_block_path       = ""
            self._rerouted_for_pallet   = False

    def _resolve_hit_path(self, hit) -> str:
        """Return the USD prim path for a PhysX overlap/raycast hit, or '' if it should be ignored.

        Ignores: the forklift itself and the pallet currently on the forks.
        Floor/ground filtering is handled by name keywords downstream in _check_proximity
        and _build_collision_prim_cache — NOT by Z position, because gates and walls often
        have their USD origin at Z=0 and would be incorrectly rejected by a height check.
        Dynamic actors expose hit.rigid_body; static actors (walls, gates, racks without a
        RigidBodyAPI) have hit.rigid_body == '' but always set hit.collision.
        """
        path = str(hit.rigid_body) if hit.rigid_body else (str(hit.collision) if hit.collision else "")
        if not path:
            return ""
        if path == self.root or path.startswith(self.root + "/"):
            return ""
        if self._carried_pallet_root and path.startswith(self._carried_pallet_root):
            return ""
        return path

    # ── USD setup ──────────────────────────────────────────────────────────

    def _nav_enabled(self) -> bool:
        """Return True if either navEnabled (legacy) or navGoTo is set."""
        for attr_name in ("navEnabled", "navGoTo"):
            if self._get_bool(attr_name):
                return True
        return False

    def _get_bool(self, attr_name: str) -> bool:
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return False
        attr = prim.GetAttribute(attr_name)
        return bool(attr.Get()) if attr and attr.IsValid() else False

    def _set_bool(self, attr_name: str, value: bool) -> None:
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            attr.Set(value)

    def _get_str(self, attr_name: str) -> str:
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return ""
        attr = prim.GetAttribute(attr_name)
        return str(attr.Get()) if attr and attr.IsValid() and attr.Get() is not None else ""

    def _set_str(self, attr_name: str, value: str) -> None:
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            attr.Set(value)

    def _print(self, msg: str) -> None:
        """Print with the forklift prim name as prefix."""
        print(f"{self._name}  {msg}")

    # ── Graph navigation ───────────────────────────────────────────────────

    def _abort_drop_nav(self) -> None:
        """Abort any in-progress drop navigation so a fresh route can be calculated.

        Clears nav state, navGoTo, navDropTransform, navClosestNode, and navFinalRoute.
        Does NOT touch navGoToDrop or the drop phase (those are protected by the caller).
        """
        self._drop_nav_triggered   = False
        self._aligning             = False
        self._nav_path             = []
        self._nav_idx                  = 0
        self._nav_reverse              = False
        self._last_nav_idx_checked     = -1
        self._turning_at_node          = False
        self._final_align_locked_yaw   = None
        self._flip_phase               = None
        self._flip_prev_node_pos     = None
        self._post_flip_align        = False
        self._crab_steer             = 0.0
        self._crab_last_sign         = 0
        self._lateral_phase          = 0
        prim = self._stage.GetPrimAtPath(self.root)
        if prim and prim.IsValid():
            a = prim.GetAttribute("navGoTo")
            if a and a.IsValid():
                a.Set(False)
            for rel_name in ("navDropTransform", "navClosestNode"):
                r = prim.GetRelationship(rel_name)
                if r:
                    r.SetTargets([])
            a = prim.GetAttribute("navFinalRoute")
            if a and a.IsValid():
                a.Set("")
        self.stop()

    def _trigger_drop_nav(self) -> None:
        """Find the closest graph node to navDropTransform and start navigation there.

        Assumes navDropTransform is already set.  Sets navGoTo=True and
        _drop_nav_triggered=True; also kicks off _start_navigation() so the
        route is published immediately.
        """
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return
        drop_rel = prim.GetRelationship("navDropTransform")
        targets  = drop_rel.GetTargets() if drop_rel else []
        if not targets:
            return

        drop_pos  = self._world_pos(str(targets[0]))
        _, positions = self._build_nav_graph()
        if not positions:
            carb.log_warn(f"Drop nav: waypoint graph empty - going direct to predrop")
            self._drop_nav_triggered = True
            self._drop_phase         = "idle"   # _update_drop computes predrop from navDropTransform
            self._set_bool("navGoToDrop", True)
            return

        drop_path = str(targets[0])

        # Compute predrop now so we can exclude nodes past it on the approach axis.
        # If the forklift would have to cross the predrop X or Y to reach the node,
        # that node is rejected and we use the next-closest one instead.
        predrop_pos, _ = self._compute_predrop(drop_path)
        my_pos         = self._world_pos(self.body_path)
        dx_to_pre      = predrop_pos[0] - my_pos[0]
        dy_to_pre      = predrop_pos[1] - my_pos[1]

        def _before_predrop(npos: tuple[float, float]) -> bool:
            if abs(dx_to_pre) >= abs(dy_to_pre):   # X is the primary approach axis
                if dx_to_pre > 0:
                    return npos[0] <= predrop_pos[0]
                else:
                    return npos[0] >= predrop_pos[0]
            else:                                   # Y is the primary approach axis
                if dy_to_pre > 0:
                    return npos[1] <= predrop_pos[1]
                else:
                    return npos[1] >= predrop_pos[1]

        valid = {
            k: v for k, v in positions.items()
            if _before_predrop(v)
            and math.hypot(v[0] - drop_pos[0], v[1] - drop_pos[1]) >= 2.0
        }
        if not valid:
            self._print("Drop nav: no nodes satisfy constraints - relaxing to before-predrop only")
            valid = {k: v for k, v in positions.items() if _before_predrop(v)} or positions

        closest = self._closest_node(predrop_pos, valid)
        prim.GetRelationship("navClosestNode").SetTargets([Sdf.Path(closest)])
        self._set_bool("navGoTo", True)
        self._drop_nav_triggered = True
        self._print(
            f"Drop nav: heading to {closest.split('/')[-1]}"
            f" -> predrop ({predrop_pos[0]:.2f}, {predrop_pos[1]:.2f})"
        )
        self._start_navigation()

        if not self._nav_path:
            # A* found no route - drive directly to predrop
            carb.log_warn(
                f"Drop nav: no route to {closest.split('/')[-1]} - going direct to predrop"
            )
            self._set_bool("navGoTo", False)
            prim.GetRelationship("navClosestNode").SetTargets([])
            self._drop_phase = "idle"
            self._set_bool("navGoToDrop", True)

    def _poll_drop_area(self) -> None:
        """Called every frame while navWithPallet=True.

        Watches navAreaToDrop and ensures a route to the drop node is always
        computed as soon as an area is available.  Reroutes on area changes.
        Changes are silently ignored while actively dropping (navGoToDrop=True
        or drop phase active).
        """
        if self._get_bool("navGoToDrop") or self._drop_phase != "idle":
            return  # mid-drop - never interrupt

        new_area = self._get_str("navAreaToDrop")

        if new_area != self._prev_area_to_drop:
            # Area changed (including "" -> value, or value -> different value).
            self._print(
                f"navAreaToDrop changed: '{self._prev_area_to_drop}' -> '{new_area}'"
                f" - {'clearing drop nav' if not new_area else 'rerouting to new drop area'}"
            )
            self._prev_area_to_drop = new_area
            self._abort_drop_nav()
            if not new_area:
                return
            self._refresh_drop_transform()
            self._trigger_drop_nav()
            return

        # Same area - trigger only if not yet routed
        if new_area and not self._drop_nav_triggered:
            self._refresh_drop_transform()
            self._trigger_drop_nav()

    def _check_target_changes(self) -> None:
        """Detect runtime changes to navPalletToPickId and reroute to the new pallet.

        Reroute is blocked when navWithPallet=True (pallet already on forks).
        Drop-area changes are handled separately by _poll_drop_area every frame.
        """
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return

        # ── Pick target changed ───────────────────────────────────────────────
        new_pallet_id = self._get_str("navPalletToPickId")
        if new_pallet_id != self._prev_pallet_to_pick_id:
            if self._get_bool("navWithPallet"):
                # Pallet already on forks - ignore silently; the ID change will be
                # visible after the drop completes and navWithPallet goes False.
                pass
            else:
                old_id = self._prev_pallet_to_pick_id
                self._prev_pallet_to_pick_id = new_pallet_id
                self._print(
                    f"navPalletToPickId changed: '{old_id}' -> '{new_pallet_id}'"
                    f" - {'clearing pick nav' if not new_pallet_id else 'rerouting to new pallet'}"
                )
                # Abort current pick nav state
                self._pick_phase             = "idle"
                self._aligning               = False
                self._nav_path               = []
                self._nav_idx                = 0
                self._nav_reverse              = False
                self._last_nav_idx_checked     = -1
                self._turning_at_node          = False
                self._final_align_locked_yaw   = None
                self._flip_phase               = None
                self._flip_prev_node_pos     = None
                self._post_flip_align        = False
                self._crab_steer             = 0.0
                self._crab_last_sign         = 0
                self._lateral_phase          = 0
                for bool_name in ("navGoTo", "navGoToPick"):
                    a = prim.GetAttribute(bool_name)
                    if a and a.IsValid():
                        a.Set(False)
                for rel_name in ("navPalletToPick", "navClosestNode"):
                    r = prim.GetRelationship(rel_name)
                    if r:
                        r.SetTargets([])
                a = prim.GetAttribute("navFinalRoute")
                if a and a.IsValid():
                    a.Set("")
                self.stop()

                if new_pallet_id:
                    self._refresh_pallet_ref()
                    pallet_path = self._get_pallet_prim_path()
                    if pallet_path:
                        self._auto_start_pick_nav(pallet_path)
                        self._start_navigation()

        # Drop area changes are handled continuously by _poll_drop_area (called
        # every frame when navWithPallet=True).  No action needed here.

    def _cancel_home_nav(self) -> None:
        """Abort an in-progress home navigation so a pick/drop order can take over."""
        self._going_home = False
        self._nav_path   = []
        self._aligning   = False
        self._final_align_locked_yaw = None
        self._going_home             = False
        self._home_path              = ""
        self._going_home_final       = False
        self._flip_phase             = None
        self._flip_prev_node_pos     = None
        self._post_flip_align        = False
        self._set_bool("navGoTo", False)
        self._set_bool("navEnabled", False)
        self._set_bool("navGoHome", False)
        prim = self._stage.GetPrimAtPath(self.root)
        if prim and prim.IsValid():
            rel = prim.GetRelationship("navClosestNode")
            if rel:
                rel.SetTargets([])
        self._print(f"Home nav cancelled - pick order received")

    def _check_home_nav(self) -> None:
        """Trigger home navigation: route to the closest graph node to navHome, then reverse in."""
        if self._going_home:
            return
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return
        home_rel = prim.GetRelationship("navHome")
        targets  = home_rel.GetTargets() if home_rel else []
        if not targets:
            return
        home_path = str(targets[0])
        home_pos  = self._world_pos(home_path)
        my_pos    = self._world_pos(self.body_path)
        if math.hypot(home_pos[0] - my_pos[0], home_pos[1] - my_pos[1]) <= self.final_pos_tol:
            return

        # Find the closest graph node to navHome (the approach node, e.g. Node_9).
        # Navigation routes to this node; then the forklift reverses the last leg to navHome.
        # Restrict candidates to nodes that have a direct edge to navHome so the final
        # leg is always a valid graph connection.
        adjacency, positions = self._build_nav_graph()
        positions_no_home = {k: v for k, v in positions.items() if k != home_path}

        # Nodes directly connected to navHome in the graph
        connected_to_home = {
            n for n, neighbors in adjacency.items()
            if any(nb == home_path for nb, _ in neighbors)
        }
        candidates = {k: v for k, v in positions_no_home.items() if k in connected_to_home}
        if not candidates:
            # navHome has no explicit edges - fall back to closest by position
            candidates = positions_no_home or positions

        # Prefer an approach node that is Y-axis-aligned with home (same aisle).
        # This lets the forklift drive straight in X and reverse cleanly into the slot.
        HOME_Y_TOL = 0.2
        home_y = home_pos[1]
        y_aligned = {
            k: v for k, v in candidates.items()
            if abs(v[1] - home_y) <= HOME_Y_TOL
        }
        if y_aligned:
            approach_node = self._closest_node(home_pos, y_aligned)
            nav_target    = approach_node
        else:
            # No Y-aligned node available — go straight to the home node and
            # do final_alignment there directly (no separate reverse-drive phase).
            approach_node = None
            nav_target    = home_path

        prim.GetRelationship("navClosestNode").SetTargets([Sdf.Path(nav_target)])
        self._home_path        = home_path
        self._going_home       = True
        self._going_home_final = False
        self._set_bool("navGoTo", True)
        self._set_bool("navGoHome", True)
        if approach_node:
            self._print(
                f"Returning home -> {home_path.split('/')[-1]}"
                f"  via Y-aligned {approach_node.split('/')[-1]}"
            )
        else:
            self._print(
                f"Returning home -> {home_path.split('/')[-1]}"
                f"  (no Y-aligned node, direct approach)"
            )

    def _start_navigation(self) -> None:
        """Build the waypoint graph, run A*, and store the path to follow."""
        # Raise forks to travel height before moving between waypoints.
        self.set_lift_position(self.pick_lift_height)
        prim = self._stage.GetPrimAtPath(self.root)
        rel  = prim.GetRelationship("navClosestNode") if prim else None
        targets = rel.GetTargets() if rel else []
        if not targets:
            return

        goal = str(targets[0])
        adjacency, positions = self._build_nav_graph()

        if not positions:
            carb.log_warn(f"Waypoint graph is empty")
            return

        my_pos = self._world_pos(self.body_path)
        start  = self._closest_node(my_pos, positions)

        if start == goal:
            self._nav_path           = [goal]
            self._nav_idx            = 0
            self._aligning           = True
            self._align_dir          = 1
            self._lat_overshoot_side   = 0
            self._initial_lateral_sign = 0
            self._flip_prev_node_pos = None   # no prev node when already at goal
            return

        # Always exclude nodes that have another forklift parked on them.
        # Uses world-position distance against all cached collision prims classified
        # as Forklift so this works even when forklifts are sleeping rigid bodies.
        _self_root_prefix = self.root + "/"
        forklift_blocked: set[str] = set()
        for spath in self._static_coll_prims:
            if self._classify_object(spath) != "Forklift":
                continue
            fkl_pos = self._world_pos(spath)
            for npath, (nx, ny) in positions.items():
                if npath in (start, goal):
                    continue
                if math.hypot(nx - fkl_pos[0], ny - fkl_pos[1]) < 1.5:
                    forklift_blocked.add(npath)
        if forklift_blocked:
            names_fkl = ", ".join(p.split("/")[-1] for p in forklift_blocked)
            carb.log_warn(f"[NAV] Nodes blocked by forklift (skipped): {names_fkl}")

        if self._get_bool("navAvoidObjects") or forklift_blocked:
            # Identify nodes (excluding start/goal) that have a collider within 1 m,
            # plus any forklift-blocked nodes found above.
            static_blocked: set[str] = set()
            if self._get_bool("navAvoidObjects"):
                static_blocked = {
                    n for n in positions
                    if n != start and n != goal and not self._node_is_clear(n)
                }
            blocked_nodes = static_blocked | forklift_blocked
            if blocked_nodes:
                names_blocked = ", ".join(p.split("/")[-1] for p in blocked_nodes)
                carb.log_warn(f"[NAV] Nodes with nearby colliders (skipped): {names_blocked}")
                filtered_adj: dict[str, list[tuple[str, float]]] = {
                    n: [(nb, cost) for nb, cost in neighbors if nb not in blocked_nodes]
                    for n, neighbors in adjacency.items()
                    if n not in blocked_nodes
                }
                path = self._find_path(start, goal, filtered_adj, positions)
            else:
                path = self._find_path(start, goal, adjacency, positions)

            if not path:
                carb.log_warn("[NAV] No obstacle-free path - marking navUnreachableTarget, using raw path")
                self._set_bool("navUnreachableTarget", True)
                path = self._find_path(start, goal, adjacency, positions)
            else:
                self._set_bool("navUnreachableTarget", False)
        else:
            path = self._find_path(start, goal, adjacency, positions)

        if not path:
            carb.log_warn(
                f"No path from "
                f"{start.split('/')[-1]} to {goal.split('/')[-1]}"
            )
            return

        self._nav_path             = path
        self._nav_idx              = 0
        self._nav_reverse          = False
        self._last_nav_idx_checked = -1
        names = " -> ".join(p.split("/")[-1] for p in path)
        self._print(f"Path: {names}")

        # Publish node numbers as a readable string on the forklift prim
        node_nums = ", ".join(p.split("_")[-1] for p in path)
        prim2 = self._stage.GetPrimAtPath(self.root)
        if prim2 and prim2.IsValid():
            attr = prim2.GetAttribute("navFinalRoute")
            if attr and attr.IsValid():
                attr.Set(node_nums)

    def _replan_from_current(self) -> None:
        """Re-run A* from the forklift's current position (dynamic obstacle appeared)."""
        self._nav_path             = []
        self._nav_idx              = 0
        self._nav_reverse              = False
        self._last_nav_idx_checked     = -1
        self._turning_at_node          = False
        self._node_monitor_timer       = 0.0   # scan immediately on the new path
        self.stop()
        self._start_navigation()

        # Apply reverse check for the first segment (same logic as _try_reroute_around_pallet).
        if self._nav_path:
            my_pos  = self._world_pos(self.body_path)
            my_yaw  = self._world_yaw(self.body_path)
            fp      = self._world_pos(self._nav_path[0])
            dx_f    = fp[0] - my_pos[0]
            dy_f    = fp[1] - my_pos[1]
            a_first = self._wrap(math.atan2(dy_f, dx_f) - my_yaw)
            self._nav_reverse = abs(a_first) > math.radians(self.nav_reverse_threshold_deg)
            if self._nav_reverse:
                self._print(
                    f"Replan: reversing to first node "
                    f"({math.degrees(abs(a_first)):.1f}° > {self.nav_reverse_threshold_deg}°)"
                )

    def _follow_path(self) -> None:
        """Drive through the computed path node by node.

        At each intermediate node, if the next segment requires a turn larger than
        `preturn_threshold_deg`, the forklift pivots in place before moving on.
        """
        if self._nav_idx >= len(self._nav_path):
            self._nav_path = []
            return

        current_target = self._nav_path[self._nav_idx]
        is_final       = self._nav_idx == len(self._nav_path) - 1

        # Dynamic clearance check - once per new node index
        if self._nav_idx != self._last_nav_idx_checked:
            self._last_nav_idx_checked = self._nav_idx
            node_has_forklift = self._node_has_forklift(current_target)
            if node_has_forklift:
                carb.log_warn(
                    f"[NAV] Node {current_target.split('/')[-1]} occupied by forklift - replanning"
                )
                self._replan_from_current()
                return
            if self._get_bool("navAvoidObjects") and not self._node_is_clear(current_target):
                carb.log_warn(
                    f"[NAV] Node {current_target.split('/')[-1]} blocked dynamically - replanning"
                )
                self._replan_from_current()
                return

        # Periodic ahead-scan: check ALL remaining nodes at node_monitor_interval seconds.
        # Detects obstacles on upcoming nodes before the forklift reaches them.
        if self._node_monitor_timer <= 0:
            self._node_monitor_timer = self.node_monitor_interval
            remaining = self._nav_path[self._nav_idx + 1:]
            for ahead_node in remaining:
                if self._node_has_forklift(ahead_node):
                    carb.log_warn(
                        f"[NAV] Upcoming node {ahead_node.split('/')[-1]} occupied by forklift - replanning early"
                    )
                    self._replan_from_current()
                    return
                if self._get_bool("navAvoidObjects") and not self._node_is_clear(ahead_node):
                    carb.log_warn(
                        f"[NAV] Upcoming node {ahead_node.split('/')[-1]} blocked - replanning early"
                    )
                    self._replan_from_current()
                    return

        target_pos = self._world_pos(current_target)
        my_pos     = self._world_pos(self.body_path)
        my_yaw     = self._world_yaw(self.body_path)

        dx   = target_pos[0] - my_pos[0]
        dy   = target_pos[1] - my_pos[1]
        dist = math.hypot(dx, dy)

        # ── Pre-turn: align heading to next segment before advancing ──
        if self._turning_at_node:
            heading_err = self._wrap(self._preturn_target_yaw - my_yaw)
            if abs(heading_err) <= math.radians(self.preturn_threshold_deg):
                self._turning_at_node = False
                self._align_dir = 1
                # fall through to normal cruise toward current_target
            else:
                turn = max(
                    -self.max_turn,
                    min(self.max_turn, heading_err * self.turn_gain * self._align_dir),
                )
                approach_x = math.cos(self._preturn_target_yaw)
                approach_y = math.sin(self._preturn_target_yaw)
                pdx = self._preturn_pivot[0] - my_pos[0]
                pdy = self._preturn_pivot[1] - my_pos[1]
                along = pdx * approach_x + pdy * approach_y
                if along > self.align_hysteresis:
                    self._align_dir = 1
                elif along < -self.align_hysteresis:
                    self._align_dir = -1

                # Radius clamp: if the forklift drifted more than max_pivot_radius from
                # the pivot node, force the drive direction back toward the node.
                dist_from_pivot = math.hypot(pdx, pdy)
                if dist_from_pivot > self.max_pivot_radius and abs(along) > 0.1:
                    self._align_dir = 1 if along > 0 else -1

                self.drive(self.align_speed * self._align_dir, turn)
                return

        if is_final:
            if dist <= self.arrival_distance:
                self.stop()
                self._aligning             = True
                self._align_dir            = 1
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = 0
                self._lateral_phase        = 0
                self._crab_steer           = 0.0
                self._crab_last_sign       = 0
                self._retreat_for_hdg      = False
                self._final_align_locked_yaw = None
                # Capture the previous path node now (before flip moves the forklift).
                if self._nav_idx > 0:
                    self._flip_prev_node_pos = self._world_pos(
                        self._nav_path[self._nav_idx - 1]
                    )
                    self._print(
                        f"Starting final alignment at "
                        f"{current_target.split('/')[-1]}  "
                        f"prev_node={self._nav_path[self._nav_idx - 1].split('/')[-1]}"
                    )
                else:
                    self._flip_prev_node_pos = None
                    self._print(
                        f"Starting final alignment at "
                        f"{current_target.split('/')[-1]}  (no prev node)"
                    )
                return
        else:
            # ── Look ahead at the next segment to decide when to pre-turn ──
            next_idx    = self._nav_idx + 1
            next_target = self._nav_path[next_idx]
            nx, ny, _   = self._world_pos(next_target)

            seg_angle   = math.atan2(ny - target_pos[1], nx - target_pos[0])
            turn_needed = abs(self._wrap(seg_angle - my_yaw))

            # Decompose into lateral (perpendicular to next segment) and total distance
            lateral          = abs(dx * math.sin(seg_angle) - dy * math.cos(seg_angle))
            early_trigger    = lateral <= self.preturn_lateral_threshold and dist <= self.preturn_lookahead_dist
            fallback_trigger = dist <= self.node_reach_distance

            was_reversing = self._nav_reverse

            # When arriving via reverse, ignore the early lateral trigger and only
            # commit when the forklift is truly close to the node (fallback_trigger).
            # This ensures the pivot happens AT the node, not 3 m before it.
            transition = (early_trigger and not was_reversing) or fallback_trigger

            if transition:
                prev_name = current_target.split("/")[-1]
                next_name = next_target.split("/")[-1]
                self._nav_idx = next_idx

                # Penultimate transition (next node is the final one) uses the tight
                # threshold; all earlier intermediate hops use the loose threshold so
                # the forklift doesn't pivot at every waypoint.
                is_penultimate = (next_idx == len(self._nav_path) - 1)
                threshold_deg  = self.preturn_threshold_deg if is_penultimate else self.preturn_threshold_deg_loose

                if was_reversing:
                    if turn_needed > math.radians(self.nav_reverse_threshold_deg):
                        # Next segment still > 90° - keep reversing to the next node
                        self._nav_reverse = True
                        self._print(
                            f"Continue reverse at {prev_name} "
                            f"({math.degrees(turn_needed):.1f}°) -> {next_name}"
                        )
                    else:
                        # Turn ≤ 90° - switch to forward; the steering during forward
                        # drive corrects the remaining heading error
                        self._nav_reverse = False
                        self._print(
                            f"Reverse->forward at {prev_name} "
                            f"({math.degrees(turn_needed):.1f}°) -> {next_name}"
                        )
                elif turn_needed > math.radians(self.nav_reverse_threshold_deg):
                    # Large turn from forward travel - back up to next node
                    self._nav_reverse = True
                    self._print(
                        f"Reverse segment at {prev_name} "
                        f"({math.degrees(turn_needed):.1f}°) -> {next_name}"
                    )
                elif turn_needed > math.radians(threshold_deg):
                    # Medium turn from forward travel - pivot at this node
                    self._nav_reverse        = False
                    self._turning_at_node    = True
                    self._preturn_target_yaw = seg_angle
                    self._preturn_pivot      = (target_pos[0], target_pos[1])
                    self._align_dir          = 1
                    trigger_label = "early" if early_trigger else "at node"
                    self._print(
                        f"Pre-turn [{trigger_label}] at {prev_name} "
                        f"({math.degrees(turn_needed):.1f}°) -> {next_name}  "
                        f"lateral={lateral:.2f}m  dist={dist:.2f}m"
                        f"  {'penultimate' if is_penultimate else 'loose'}"
                    )
                else:
                    self._nav_reverse = False
                    self._print(f"Reached {prev_name} -> {next_name}")
                return

        angle_err   = self._wrap(math.atan2(dy, dx) - my_yaw)
        heading_deg = abs(math.degrees(angle_err))

        if self._nav_reverse:
            # While reversing, check whether forward travel is now viable.
            # This happens when the forklift has steered enough during backing that the
            # forward heading error to the target drops below nav_reverse_threshold_deg.
            if abs(angle_err) <= math.radians(self.nav_reverse_threshold_deg):
                self._nav_reverse = False
                self._print(
                    f"[NAV] reverse->forward at {current_target.split('/')[-1]}"
                    f"  fwd_err={math.degrees(angle_err):.1f}°"
                )
                # Fall through to forward drive below
            else:
                back_err = self._wrap(angle_err + math.pi)
                turn     = max(-self.max_turn, min(self.max_turn, -back_err * self.turn_gain))
                speed    = abs(self.move_speed)
                self._print(
                    f"reversing -> {current_target.split('/')[-1]}"
                    f"  dist={dist:.3f}  back_err={math.degrees(back_err):.1f}°"
                )
                self.drive(speed, turn)
                return

        # Forward drive (also reached when reverse was just cleared)
        turn  = max(-self.max_turn, min(self.max_turn, angle_err * self.turn_gain))
        speed = (self.approach_speed if dist < self.slow_distance else self.move_speed) if is_final else self.move_speed
        self._print(
            f"travelling to {current_target.split('/')[-1]}"
            f"  x={dx:.3f}  y={dy:.3f}"
            f"  dist={dist:.3f}  heading={heading_deg:.1f}°"
        )
        self.drive(speed, turn)

    # ── Graph building & pathfinding ───────────────────────────────────────

    def _build_nav_graph(
        self,
    ) -> tuple[dict[str, list[tuple[str, float]]], dict[str, tuple[float, float]]]:
        """Parse nodes and edges from the USD stage.

        Returns:
            adjacency: node_path -> [(neighbor_path, cost), ...]
            positions: node_path -> (x, y)
        """
        nodes_prim = self._stage.GetPrimAtPath(self.NODES_ROOT)
        edges_prim = self._stage.GetPrimAtPath(self.EDGES_ROOT)

        adjacency: dict[str, list[tuple[str, float]]] = {}
        positions: dict[str, tuple[float, float]]     = {}
        node_lookup: dict[str, str] = {}

        if nodes_prim and nodes_prim.IsValid():
            for child in nodes_prim.GetChildren():
                name = child.GetName()
                if not name.startswith("Node_"):
                    continue
                num_str = name[5:]
                path    = str(child.GetPath())
                pos     = self._world_pos(path)
                node_lookup[num_str] = path
                positions[path]  = (pos[0], pos[1])
                adjacency[path]  = []

        if edges_prim and edges_prim.IsValid():
            for child in edges_prim.GetChildren():
                name = child.GetName()
                if not name.startswith("Edge_"):
                    continue
                parts = name[5:].split("_")
                if len(parts) != 2:
                    continue
                a, b = parts[0], parts[1]
                if a not in node_lookup or b not in node_lookup:
                    continue
                path_a = node_lookup[a]
                path_b = node_lookup[b]

                w_attr = child.GetAttribute("weight")
                if w_attr and w_attr.Get() is not None:
                    cost = float(w_attr.Get())
                else:
                    ax, ay = positions[path_a]
                    bx, by = positions[path_b]
                    cost = math.hypot(bx - ax, by - ay)

                adjacency[path_a].append((path_b, cost))
                adjacency[path_b].append((path_a, cost))

        return adjacency, positions

    def _closest_node(
        self, pos: tuple[float, float, float], positions: dict[str, tuple[float, float]]
    ) -> str:
        """Return the path of the node nearest to *pos*."""
        best_path = ""
        best_dist = float("inf")
        for node_path, (nx, ny) in positions.items():
            d = math.hypot(nx - pos[0], ny - pos[1])
            if d < best_dist:
                best_dist = d
                best_path = node_path
        return best_path

    def _closest_node_toward(
        self,
        pos: tuple[float, float, float],
        positions: dict[str, tuple[float, float]],
        goal_path: str,
    ) -> str:
        """Return the closest graph node that lies in the forward half-space toward goal.

        Nodes behind the forklift (dot product negative) are skipped. Falls back to
        the globally closest node if nothing is in the forward half-space.
        """
        goal_pos  = self._world_pos(goal_path)
        dx_home   = goal_pos[0] - pos[0]
        dy_home   = goal_pos[1] - pos[1]
        d_home    = math.hypot(dx_home, dy_home)

        best_fwd      = ""
        best_fwd_dist = float("inf")
        fallback      = ""
        fallback_dist = float("inf")

        for node_path, (nx, ny) in positions.items():
            dx = nx - pos[0]
            dy = ny - pos[1]
            d  = math.hypot(dx, dy)

            if d < fallback_dist:
                fallback_dist = d
                fallback      = node_path

            if d_home > 0:
                dot = (dx * dx_home + dy * dy_home) / (d_home * max(d, 1e-4))
                if dot > 0 and d < best_fwd_dist:
                    best_fwd_dist = d
                    best_fwd      = node_path

        return best_fwd if best_fwd else fallback

    def _node_has_forklift(self, node_path: str, radius: float = 1.5) -> bool:
        """Return True if another forklift's body is within *radius* m of this node.

        Checked against the static collision prim cache (world-position based),
        so it works regardless of whether the forklift is moving or stationary.
        """
        node_pos = self._world_pos(node_path)
        for spath in self._static_coll_prims:
            if self._classify_object(spath) != "Forklift":
                continue
            fp = self._world_pos(spath)
            if math.hypot(fp[0] - node_pos[0], fp[1] - node_pos[1]) < radius:
                return True
        return False

    def _node_is_clear(self, node_path: str, radius: float = 1.0) -> bool:
        """Return True if no collidable prim (above floor) lies within *radius* m of node.

        Uses an overlap_box centred on the node position at the given radius.
        Ignores the forklift itself and anything at floor level (Z < 0.02 m).
        """
        from omni.physx import get_physx_scene_query_interface
        pos = self._world_pos(node_path)
        pq  = get_physx_scene_query_interface()
        blocked: list[str] = []

        # Pre-compute the pallet scope prefix once so _report can filter cheaply.
        _pallet_scope = (
            self._pallet_path.rsplit("/", 1)[0] + "/"
            if self._pallet_path else ""
        )

        def _report(hit) -> bool:
            path = self._resolve_hit_path(hit)
            if not path:
                return True
            # Ignore anything inside the target pallet's scope (pallet mesh,
            # stacked product boxes, sub-prims) and the currently carried pallet.
            if _pallet_scope and path.startswith(_pallet_scope):
                return True
            if self._is_carried_pallet(path):
                return True
            blocked.append(path)
            return True

        half = radius
        pq.overlap_box(
            carb.Float3(half, half, half),
            carb.Float3(pos[0], pos[1], pos[2] + half),
            carb.Float4(0.0, 0.0, 0.0, 1.0),
            _report,
            False,
        )
        if blocked:
            carb.log_warn(
                f"[NAV] Node {node_path.split('/')[-1]} blocked by: "
                + ", ".join(p.split("/")[-1] for p in blocked[:3])
            )
        return len(blocked) == 0

    def _find_path(
        self,
        start: str,
        goal: str,
        adjacency: dict[str, list[tuple[str, float]]],
        positions: dict[str, tuple[float, float]],
    ) -> list[str]:
        """A* shortest path from *start* to *goal*.

        Returns an ordered list of node paths (including start and goal),
        or an empty list if no path exists.
        """
        gx, gy = positions[goal]

        def h(p: str) -> float:
            px, py = positions[p]
            return math.hypot(gx - px, gy - py)

        heap: list[tuple[float, float, str, list[str]]] = [(h(start), 0.0, start, [start])]
        visited: set[str] = set()

        while heap:
            _, g, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path

            for neighbor, cost in adjacency.get(current, []):
                if neighbor not in visited:
                    new_g = g + cost
                    heapq.heappush(heap, (new_g + h(neighbor), new_g, neighbor, path + [neighbor]))

        return []

    # ── Alignment ──────────────────────────────────────────────────────────

    def _flip_heading(self, target_yaw: float) -> bool:
        """K-turn to escape a ~180° heading error then fully align at the previous node.

        Phase 'back'   : reverse + max swivel until heading error drops to ≤90°.
        Phase 'go_prev': forward alignment at self._flip_prev_node_pos with target_yaw.
        Returns True when go_prev alignment is complete - caller then switches to the
        backward final alignment toward the home node.
        """
        my_pos  = self._world_pos(self.body_path)
        my_yaw  = self._world_yaw(self.body_path)
        hdg_err = self._wrap(my_yaw - target_yaw)

        if self._flip_phase is None:
            self._flip_phase     = "back"
            self._flip_start_yaw = my_yaw
            self._print(f"Flip: start  heading_err={math.degrees(hdg_err):.1f}°")

        if self._flip_phase == "back":
            # Reverse at full swivel; direction chosen to reduce heading error.
            swivel = self.max_turn * (1 if hdg_err > 0 else -1)
            self.drive(-self.align_speed, swivel)
            self._print(f"flip back  heading_err={math.degrees(hdg_err):.1f}°")
            # Stop reversing once the heading error has dropped to ≤90°.
            if abs(hdg_err) <= math.pi / 2:
                if self._flip_prev_node_pos is not None:
                    # Reset alignment state for the forward approach to the prev node.
                    self._align_dir            = 1
                    self._lat_overshoot_side   = 0
                    self._initial_lateral_sign = 0
                    self._lateral_phase        = 0
                    self._retreat_for_hdg      = False
                    self._flip_phase = "go_prev"
                    self._print(
                        f"Flip: switching to go_prev  "
                        f"heading_err={math.degrees(hdg_err):.1f}°"
                    )
                else:
                    self._flip_phase = None
                    self._print(f"Flip: complete (no prev node)")
                    return True

        elif self._flip_phase == "go_prev":
            dx   = self._flip_prev_node_pos[0] - my_pos[0]
            dy   = self._flip_prev_node_pos[1] - my_pos[1]
            dist = math.hypot(dx, dy)
            self._print(
                f"flip go_prev"
                f"  dist={dist:.3f}  heading_err={math.degrees(hdg_err):.1f}°"
            )
            # Full position+heading alignment at the previous node (forward approach).
            if self._align_to_pose(
                self._flip_prev_node_pos,
                target_yaw,
                pos_tol=self.final_pos_tol,
                angle_tol=self.final_angle_tol,
                log_prefix="flip go_prev",
                no_overshoot=True,
            ):
                self._flip_phase = None
                self._print(f"Flip: complete (aligned at prev node)")
                return True

        return False

    def final_alignment(self) -> None:
        """Precisely align position and heading to navClosestNode.

        For home navigation (going_home=True):
          • Locked heading = bearing from navClosestNode toward navHome.
          • On success: transitions to _going_home_final for the backward drive to navHome.
        For all other nodes:
          • Locked heading = whichever of target_yaw / target_yaw+180° requires less correction.
          • On success: clears nav state and disables navGoTo/navEnabled.
        """
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return

        rel     = prim.GetRelationship("navClosestNode")
        targets = rel.GetTargets() if rel else []
        if not targets:
            self._aligning = False
            return

        target_path = str(targets[0])
        target_pos  = self._world_pos(target_path)
        node_name   = target_path.split("/")[-1]

        # Lock the heading once at alignment start.
        if self._final_align_locked_yaw is None:
            if self._going_home and self._home_path and target_path != self._home_path:
                # Approach node is separate from home: forks must face AWAY from navHome
                # so the forklift can reverse into the slot.
                # locked_yaw = bearing(navHome -> approach_node), i.e. +π from home direction.
                home_pos = self._world_pos(self._home_path)
                dx = home_pos[0] - target_pos[0]
                dy = home_pos[1] - target_pos[1]
                self._final_align_locked_yaw = math.atan2(-dy, -dx)
            else:
                # Derive heading from the last segment (prev_node -> this node) rather than
                # the node's stored orientation, so nodes need no manual rotation in the stage.
                if self._flip_prev_node_pos is not None:
                    dx = target_pos[0] - self._flip_prev_node_pos[0]
                    dy = target_pos[1] - self._flip_prev_node_pos[1]
                    target_yaw = math.atan2(dy, dx)
                else:
                    # No previous node (already at goal when nav started) - keep current heading.
                    target_yaw = self._world_yaw(self.body_path)
                my_yaw = self._world_yaw(self.body_path)
                err0 = abs(self._wrap(my_yaw - target_yaw))
                err1 = abs(self._wrap(my_yaw - (target_yaw + math.pi)))
                self._final_align_locked_yaw = target_yaw + math.pi if err1 < err0 else target_yaw

        # If heading error is ~180°, run K-turn flip then approach backward.
        my_yaw      = self._world_yaw(self.body_path)
        hdg_err_deg = abs(math.degrees(self._wrap(my_yaw - self._final_align_locked_yaw)))
        if hdg_err_deg > self.flip_hdg_threshold or self._flip_phase is not None:
            if not self._flip_heading(self._final_align_locked_yaw):
                return  # still flipping

            # Flip + go_prev complete - approach navClosestNode backward.
            self._align_dir            = -1
            self._lat_overshoot_side   = 0
            self._initial_lateral_sign = 0
            self._lateral_phase        = 0
            self._retreat_for_hdg      = False
            self._post_flip_align      = True

        if self._align_to_pose(
            target_pos, self._final_align_locked_yaw,
            self.final_pos_tol, self.final_angle_tol,
            log_prefix=f"aligning {node_name}",
            no_overshoot=True,
            reverse_approach=self._post_flip_align,
        ):
            self._aligning           = False
            self._nav_path           = []
            self._flip_phase         = None
            self._flip_prev_node_pos = None
            self._post_flip_align    = False

            if self._going_home and target_path != self._home_path:
                # Aligned at the approach node — reverse the last leg to navHome.
                self._align_dir            = -1
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = 0
                self._lateral_phase        = 0
                self._retreat_for_hdg      = False
                self._going_home_final     = True
                # Keep _final_align_locked_yaw and navGoTo alive for _home_final_drive().
                self._print(
                    f"Aligned at {node_name}"
                    f" -> reversing to {self._home_path.split('/')[-1]}"
                )
            elif self._going_home and target_path == self._home_path:
                # Direct home approach (no Y-aligned node existed) - already arrived.
                self._final_align_locked_yaw = None
                self._going_home             = False
                self._going_home_final       = False
                self._home_path              = ""
                for attr_name in ("navEnabled", "navGoTo", "navGoHome"):
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.IsValid():
                        attr.Set(False)
                rel.SetTargets([])
                self._set_str("navFinalRoute", "")
                self._print(f"Home: arrived directly at {node_name}")
            else:
                self._final_align_locked_yaw = None
                self._going_home = False
                for attr_name in ("navEnabled", "navGoTo", "navGoHome"):
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.IsValid():
                        attr.Set(False)
                rel.SetTargets([])
                self._print(f"Aligned at {node_name}")

    def _home_final_drive(self) -> None:
        """Reverse from navClosestNode to navHome (the last leg of home navigation)."""
        home_pos = self._world_pos(self._home_path)
        home_name = self._home_path.split("/")[-1]
        if self._align_to_pose(
            home_pos, self._final_align_locked_yaw,
            self.final_pos_tol, self.final_angle_tol,
            log_prefix=f"home_final -> {home_name}",
            no_overshoot=True,
            reverse_approach=True,
        ):
            self.stop()
            self._going_home             = False
            self._going_home_final       = False
            self._home_path              = ""
            self._final_align_locked_yaw = None
            self._align_dir              = 1
            prim = self._stage.GetPrimAtPath(self.root)
            if prim and prim.IsValid():
                for attr_name in ("navEnabled", "navGoTo", "navGoHome"):
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.IsValid():
                        attr.Set(False)
                rel = prim.GetRelationship("navClosestNode")
                if rel:
                    rel.SetTargets([])
            self._set_str("navFinalRoute", "")
            self._print(f"Home: arrived at {home_name}")

    # ── Alignment ─────────────────────────────────────────────────────────

    def _align_to_pose(
        self,
        target_pos: tuple[float, float, float],
        target_yaw: float,
        pos_tol: float,
        angle_tol: float,
        log_prefix: str = "align_pose",
        no_overshoot: bool = False,
        reverse_approach: bool = False,
    ) -> bool:
        """Drive to an explicit (pos, yaw) using approach-axis hysteresis.

        Uses and updates self._align_dir to determine forward/backward direction.
        Steering is inverted when reversing (rear-wheel steering convention).

        Args:
            target_pos: World-space XYZ target position.
            target_yaw: Target heading in radians (forklift -X axis convention).
            pos_tol: Position tolerance in metres.
            angle_tol: Heading tolerance in degrees.
            log_prefix: Label used in the info log line.
            no_overshoot: When True, apply the Y=0 barrier and crab-walk logic.
            reverse_approach: When True (implies no_overshoot), the forklift
                approaches from along < 0 by reversing toward along = 0.
                "Overshot" = along ≥ 0; retreat drives forward to along = −1.
                Use after a flip manoeuvre when _align_dir is already −1.

        Returns:
            True when both position and heading are within tolerances.
        """
        my_pos = self._world_pos(self.body_path)
        my_yaw = self._world_yaw(self.body_path)

        dx   = target_pos[0] - my_pos[0]
        dy   = target_pos[1] - my_pos[1]
        dist = math.hypot(dx, dy)

        heading_err = self._wrap(target_yaw - my_yaw)
        heading_deg = abs(math.degrees(heading_err))

        if dist <= pos_tol and heading_deg <= angle_tol:
            self.stop()
            return True

        approach_x = math.cos(target_yaw)
        approach_y = math.sin(target_yaw)

        # Along-axis projection -> controls forward/backward direction
        along = dx * approach_x + dy * approach_y

        # Cross-track error: signed lateral offset from the approach axis.
        # Positive = forklift is to the left of the axis when looking along approach direction.
        lateral = dx * (-math.sin(target_yaw)) + dy * math.cos(target_yaw)

        # ── no_overshoot mode ────────────────────────────────────────────────
        if no_overshoot and reverse_approach:
            # Reverse approach: forklift comes from along < 0 (target behind forks)
            # and backs toward along = 0.  Mirror of the forward no_overshoot logic:
            #   along ≥ 0  -> at/past target.
            #   along ≤ −1 -> retreated too far, re-approach.
            #   Crab-walk phases drive FORWARD (away from target) then BACK.
            _log = (
                f"{self._name}  {log_prefix}"
                f"  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={heading_deg:.1f}°"
            )

            # Retreat: crossed along=0 with bad heading -> drive forward to along=−1,
            # with heading correction (forward, normal steer) so crab-walk triggers next pass.
            if self._retreat_for_hdg:
                _traveled = math.hypot(
                    my_pos[0] - self._crab_retreat_start_pos[0],
                    my_pos[1] - self._crab_retreat_start_pos[1],
                )
                # Update rack clearance cap while retreating (forks forward = +local_x dir)
                if self._proximity_blocked:
                    _btype = self._get_str("navObjectDetected")
                    if _btype in ("Rack", "Wall") and self._prox_block_local_x > 0:
                        _remaining = abs(self._prox_block_local_x) - self._prox_half_body_x - 0.25
                        self._crab_retreat_max_dist = min(
                            self._crab_retreat_max_dist,
                            _traveled + max(0.0, _remaining),
                        )
                if along <= -1.0 or _traveled >= self._crab_retreat_max_dist:
                    if _traveled >= self._crab_retreat_max_dist < float("inf"):
                        print(
                            f"{self._name}  {log_prefix}  crab retreat capped by Rack"
                            f"  traveled={_traveled:.2f}m  cap={self._crab_retreat_max_dist:.2f}m"
                        )
                    self._retreat_for_hdg       = False
                    self._crab_retreat_max_dist = float("inf")
                else:
                    print(
                        f"{self._name}  {log_prefix}  crab retreat (hdg fix)"
                        f"  along={along:.3f}  heading={heading_deg:.1f}°"
                    )
                    raw_turn = max(-self.max_turn,
                                   min(self.max_turn, heading_err * self.turn_gain))
                    self.drive(self.align_speed, raw_turn)  # forward + heading correction
                    return False

            if self._lateral_phase != 0:
                if self._lateral_phase == 1:
                    if along < -self.pick_align_back_dist:
                        self._lateral_phase = 2
                        self._crab_phase2_start_heading = heading_deg
                    print(
                        f"{self._name}  {log_prefix}  crab P1 (arc)"
                        f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                        f"  steer={self._crab_steer:.1f}°"
                    )
                    self.drive(self.align_speed,
                               self._lateral_steer_sign * self._crab_steer)
                    return False
                if self._lateral_phase == 2:
                    hdg_increased = heading_deg > self._crab_phase2_start_heading + self.crab_walk_hdg_increase_tol
                    if heading_deg <= angle_tol or along < -self.pick_align_back_dist * 3 or hdg_increased:
                        self._lateral_phase = 3
                    print(
                        f"{self._name}  {log_prefix}  crab P2 (counter-arc)"
                        f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                        f"  Δhdg={heading_deg - self._crab_phase2_start_heading:+.1f}°"
                    )
                    self.drive(self.align_speed,
                               -self._lateral_steer_sign * self._crab_steer)
                    return False
                if self._lateral_phase == 3:
                    if along >= 0.0:
                        self._lateral_phase = 0
                        self.stop()
                        print(
                            f"{self._name}  {log_prefix}  crab P3 done"
                            f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                        )
                        return False
                    print(
                        f"{self._name}  {log_prefix}  crab P3 (return)"
                        f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                    )
                    self.drive(-self.align_speed, 0.0)  # backward toward target
                    return False

            if along >= 0.0:
                if dist <= pos_tol and heading_deg <= angle_tol:
                    self.stop()
                    return True
                if heading_deg <= self.crab_walk_heading_tol and abs(lateral) > pos_tol:
                    new_sign = 1 if lateral > 0 else -1
                    if self._crab_steer == 0.0:
                        self._crab_steer = self.pick_align_steer
                    elif new_sign != self._crab_last_sign:
                        self._crab_steer = max(
                            self.pick_align_steer * 0.15,
                            self._crab_steer * self.crab_steer_decay,
                        )
                    self._crab_last_sign     = new_sign
                    self._lateral_phase      = 1
                    self._lateral_steer_sign = new_sign
                    print(
                        f"{self._name}  {log_prefix}  crab-walk start"
                        f"  lateral={lateral:.3f}m  heading={heading_deg:.1f}°"
                        f"  steer={self._crab_steer:.1f}°"
                    )
                    self.drive(self.align_speed,
                               self._lateral_steer_sign * self._crab_steer)
                    return False
                # Heading not aligned at along=0 - retreat forward 1 m.
                self._retreat_for_hdg        = True
                self._lateral_phase          = 0
                self._crab_retreat_start_pos = (my_pos[0], my_pos[1])
                self._crab_retreat_max_dist  = float("inf")
                print(_log)
                self.drive(self.align_speed, 0.0)
                return False

            # along < 0: reverse toward target; invert swivel for rear-wheel steering.
            raw_turn = max(-self.max_turn,
                           min(self.max_turn,
                               heading_err * self.turn_gain
                               + lateral * self.lateral_turn_gain))
            print(_log)
            self.drive(-self.align_speed, -raw_turn)
            return False

        # ── no_overshoot forward approach (pick pre-pick / final-node) ───────
        # Never drive past along=0 (the target Y line).
        #
        # While along > 0: drive forward with heading + lateral steering (normal).
        # When along ≤ 0 (at or past Y=0):
        #   • Done?          -> return True.
        #   • Heading good, lateral bad -> crab-walk:
        #       Phase 1 - reverse + pos swivel: small arc toward approach axis.
        #       Phase 2 - reverse + neg swivel: restore heading.
        #       Phase 3 - forward straight:     return to Y=0, then re-check.
        #   • Heading bad  -> reverse with heading correction (no forward).
        if no_overshoot:
            _log = (
                f"{self._name}  {log_prefix}"
                f"  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={heading_deg:.1f}°"
            )

            # Retreat: heading was bad at Y=0 - back up to along=1.0 with heading
            # correction (inverted steer for rear-wheel-drive reversing) so that
            # the next forward pass arrives at along=0 already close to target_yaw
            # and the crab-walk trigger fires instead of looping into another retreat.
            if self._retreat_for_hdg:
                _traveled = math.hypot(
                    my_pos[0] - self._crab_retreat_start_pos[0],
                    my_pos[1] - self._crab_retreat_start_pos[1],
                )
                # Update rack clearance cap while retreating (body forward = -local_x dir)
                if self._proximity_blocked:
                    _btype = self._get_str("navObjectDetected")
                    if _btype in ("Rack", "Wall") and self._prox_block_local_x < 0:
                        _remaining = abs(self._prox_block_local_x) - self._prox_half_body_x - 0.25
                        self._crab_retreat_max_dist = min(
                            self._crab_retreat_max_dist,
                            _traveled + max(0.0, _remaining),
                        )
                if along >= 1.0 or _traveled >= self._crab_retreat_max_dist:
                    if _traveled >= self._crab_retreat_max_dist < float("inf"):
                        print(
                            f"{self._name}  {log_prefix}  crab retreat capped by Rack"
                            f"  traveled={_traveled:.2f}m  cap={self._crab_retreat_max_dist:.2f}m"
                        )
                    self._retreat_for_hdg       = False
                    self._crab_retreat_max_dist = float("inf")
                    # Fall through to normal approach below.
                else:
                    print(
                        f"{self._name}  {log_prefix}  crab retreat (hdg fix)"
                        f"  along={along:.3f}  heading={heading_deg:.1f}°"
                    )
                    raw_turn = max(-self.max_turn,
                                   min(self.max_turn, heading_err * self.turn_gain))
                    self.drive(-self.align_speed, -raw_turn)   # backward + inverted steer
                    return False

            # Mid-crab-walk: run phases regardless of current along value.
            if self._lateral_phase != 0:
                if self._lateral_phase == 1:
                    if along > self.pick_align_back_dist:
                        self._lateral_phase = 2
                        self._crab_phase2_start_heading = heading_deg
                    print(
                        f"{self._name}  {log_prefix}  crab P1 (arc)"
                        f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                        f"  steer={self._crab_steer:.1f}°"
                    )
                    self.drive(-self.align_speed,
                               self._lateral_steer_sign * self._crab_steer)
                    return False

                if self._lateral_phase == 2:
                    hdg_increased = heading_deg > self._crab_phase2_start_heading + self.crab_walk_hdg_increase_tol
                    if heading_deg <= angle_tol or along > self.pick_align_back_dist * 3 or hdg_increased:
                        self._lateral_phase = 3
                    print(
                        f"{self._name}  {log_prefix}  crab P2 (counter-arc)"
                        f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                        f"  Δhdg={heading_deg - self._crab_phase2_start_heading:+.1f}°"
                    )
                    self.drive(-self.align_speed,
                               -self._lateral_steer_sign * self._crab_steer)
                    return False

                if self._lateral_phase == 3:
                    if along <= 0.0:
                        self._lateral_phase = 0
                        self.stop()
                        print(
                            f"{self._name}  {log_prefix}  crab P3 done"
                            f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                        )
                        return False
                    print(
                        f"{self._name}  {log_prefix}  crab P3 (return)"
                        f"  along={along:.3f}  lateral={lateral:.3f}  heading={heading_deg:.1f}°"
                    )
                    self.drive(self.align_speed, 0.0)
                    return False

            # Not mid-crab-walk, not retreating.
            if along <= 0.0:
                if dist <= pos_tol and heading_deg <= angle_tol:
                    self.stop()
                    return True

                if heading_deg <= self.crab_walk_heading_tol and abs(lateral) > pos_tol:
                    new_sign = 1 if lateral > 0 else -1
                    if self._crab_steer == 0.0:
                        # First crab walk of this alignment session.
                        self._crab_steer = self.pick_align_steer
                    elif new_sign != self._crab_last_sign:
                        # Direction reversed - decay steer angle to converge.
                        self._crab_steer = max(
                            self.pick_align_steer * 0.15,
                            self._crab_steer * self.crab_steer_decay,
                        )
                    self._crab_last_sign     = new_sign
                    self._lateral_phase      = 1
                    self._lateral_steer_sign = new_sign
                    print(
                        f"{self._name}  {log_prefix}  crab-walk start"
                        f"  lateral={lateral:.3f}m  heading={heading_deg:.1f}°"
                        f"  steer={self._crab_steer:.1f}°"
                    )
                    self.drive(-self.align_speed,
                               self._lateral_steer_sign * self._crab_steer)
                    return False

                # Heading not aligned at Y=0 - retreat 1 m to give approach room.
                self._retreat_for_hdg        = True
                self._lateral_phase          = 0
                self._crab_retreat_start_pos = (my_pos[0], my_pos[1])
                self._crab_retreat_max_dist  = float("inf")
                print(_log)
                self.drive(-self.align_speed, 0.0)
                return False

            # along > 0: approach target forward; never use _align_dir here.
            turn = max(-self.max_turn,
                       min(self.max_turn,
                           heading_err * self.turn_gain
                           + lateral * self.lateral_turn_gain))
            print(_log)
            self.drive(self.align_speed, turn)
            return False
        # ── end no_overshoot ─────────────────────────────────────────────────

        # Lateral overshoot guard - signed, axis-aware.
        # Trigger only AFTER the forklift has crossed the lateral zero line from its starting
        # side, so large initial offsets (still approaching) never cause a false trigger.
        # Recover by driving straight (no steer) in the direction that reduces the overshoot.
        # Threshold selection: wide band while heading is still large, tight once near-aligned.
        if heading_deg > self.align_lateral_angle_thresh:
            lat_limit   = self.align_lateral_limit
            lat_recover = self.align_lateral_recover_dist
        else:
            lat_limit   = self.align_lateral_tight_limit
            lat_recover = self.align_lateral_tight_recover

        if self._initial_lateral_sign == 0:
            self._initial_lateral_sign = 1 if lateral >= 0 else -1

        current_lat_sign = 1 if lateral >= 0 else -1
        crossed = current_lat_sign != self._initial_lateral_sign

        # Final-approach override: when lateral is small and heading is nearly aligned,
        # exit any active recovery and let normal alignment handle the last few centimetres.
        if abs(lateral) < 0.2 and heading_deg < 10.0:
            self._lat_overshoot_side   = 0
            self._initial_lateral_sign = current_lat_sign   # re-arm so trigger won't fire

        if self._lat_overshoot_side == 1:
            # Recovering from +overshoot: drive backward past zero to -lat_recover.
            # Exit early if:
            #   • target reached (lateral ≤ -lat_recover)
            #   • backward motion is making lateral WORSE (diverging toward more positive)
            #   • forklift crossed zero AND heading is already within tight range
            diverging  = lateral > self._lat_last + 0.05
            past_zero  = lateral < 0
            early_exit = past_zero and heading_deg < self.align_lateral_angle_thresh

            if lateral <= -lat_recover or diverging or early_exit:
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = current_lat_sign   # re-arm from new side
            else:
                # Once past zero, add heading steering so the reverse also corrects angle
                steer = (
                    max(-self.max_turn, min(self.max_turn, heading_err * self.turn_gain * (-1)))
                    if past_zero else 0.0
                )
                self._print(f"{log_prefix}  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={heading_deg:.1f}°")
                self._lat_last = lateral
                self.drive(-self.align_speed, steer)
                return False

        elif self._lat_overshoot_side == -1:
            # Recovering from -overshoot: drive forward past zero to +lat_recover.
            diverging  = lateral < self._lat_last - 0.05
            past_zero  = lateral > 0
            early_exit = past_zero and heading_deg < self.align_lateral_angle_thresh

            if lateral >= lat_recover or diverging or early_exit:
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = current_lat_sign
            else:
                steer = (
                    max(-self.max_turn, min(self.max_turn, heading_err * self.turn_gain))
                    if past_zero else 0.0
                )
                self._print(f"{log_prefix}  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={heading_deg:.1f}°")
                self._lat_last = lateral
                self.drive(self.align_speed, steer)
                return False

        # Trigger: only fire after crossing the initial zero line
        if self._lat_overshoot_side == 0 and crossed:
            if lateral > lat_limit:
                self._lat_overshoot_side = 1
                self._lat_last = lateral
                self._print(f"{log_prefix}  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={heading_deg:.1f}°")
                self.drive(-self.align_speed, 0.0)
                return False
            elif lateral < -lat_limit:
                self._lat_overshoot_side = -1
                self._lat_last = lateral
                self._print(f"{log_prefix}  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={heading_deg:.1f}°")
                self.drive(self.align_speed, 0.0)
                return False

        # After a flip, keep _align_dir = -1 for the full backward approach run.
        if not self._post_flip_align:
            if along > self.align_hysteresis:
                self._align_dir = 1
            elif along < -self.align_hysteresis:
                self._align_dir = -1

        # Heading correction keeps the forklift pointing at target_yaw;
        # lateral correction steers the forklift back onto the approach axis line.
        # Both are inverted on reverse via align_dir (rear-wheel steering convention).
        turn = max(
            -self.max_turn,
            min(
                self.max_turn,
                (heading_err * self.turn_gain + lateral * self.lateral_turn_gain)
                * self._align_dir,
            ),
        )

        self._print(f"{log_prefix}  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={heading_deg:.1f}°")

        self.drive(self.align_speed * self._align_dir, turn)
        return False

    # ── Pick system ────────────────────────────────────────────────────────

    def _find_prim_by_pallet_id(self, pallet_id: str) -> str:
        """Traverse the stage and return the path of the prim whose `palletId` attribute matches.

        Skips prims under the forklift's own hierarchy to avoid false matches.
        """
        for prim in self._stage.Traverse():
            if str(prim.GetPath()).startswith(self.root):
                continue
            attr = prim.GetAttribute("palletId")
            if attr and attr.IsValid() and str(attr.Get()) == pallet_id:
                return str(prim.GetPath())
        return ""

    def _refresh_pallet_ref(self) -> None:
        """Resolve navPalletToPickId -> navPalletToPick relationship if not already set."""
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim:
            return

        rel     = prim.GetRelationship("navPalletToPick")
        targets = rel.GetTargets() if rel else []
        if targets:
            return  # already resolved

        id_attr   = prim.GetAttribute("navPalletToPickId")
        pallet_id = str(id_attr.Get()) if id_attr and id_attr.Get() is not None else ""
        if not pallet_id:
            return  # no pallet assigned - nothing to resolve

        pallet_path = self._find_prim_by_pallet_id(pallet_id)
        if pallet_path:
            rel.SetTargets([pallet_path])
            self._print(f"navPalletToPick -> {pallet_path}")
        else:
            carb.log_warn(f"Pallet '{pallet_id}' not found in stage")

    def _collect_pallet_positions(self) -> list[tuple[float, float]]:
        """Return XY world positions of all pallet/box prims currently in the stage."""
        positions: list[tuple[float, float]] = []
        for stage_prim in self._stage.Traverse():
            if self._classify_object(str(stage_prim.GetPath())) in ("Pallet", "Box"):
                pos = self._world_pos(str(stage_prim.GetPath()))
                positions.append((pos[0], pos[1]))
        return positions

    def _is_drop_pos_occupied(
        self,
        pos: tuple[float, float, float],
        pallet_positions: list[tuple[float, float]],
        tolerance_xy: float = 1.0,
    ) -> bool:
        """Return True if any pallet is within tolerance_xy in both X and Y of pos."""
        px, py = pos[0], pos[1]
        return any(
            abs(bx - px) <= tolerance_xy and abs(by - py) <= tolerance_xy
            for bx, by in pallet_positions
        )

    def _refresh_drop_transform(self) -> None:
        """Resolve navAreaToDrop -> navDropTransform relationship.

        Traverses the stage for prims with a 'navRacksPosId' OR 'navBufferPosId' attribute
        whose value matches navAreaToDrop.  Skips positions already occupied by a pallet
        (any pallet within 1 m in X and Y).

        Selection strategy:
          - Buffer positions: prefer the farthest free position within 15 m; if none
            exist within 15 m fall back to the farthest free position at any distance.
          - Rack positions: pick the nearest free position.
        """
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return

        rel     = prim.GetRelationship("navDropTransform")
        targets = rel.GetTargets() if rel else []
        if targets:
            return  # already resolved

        id_attr = prim.GetAttribute("navAreaToDrop")
        area_id = str(id_attr.Get()) if id_attr and id_attr.Get() is not None else ""
        if not area_id:
            return  # no area assigned - nothing to resolve

        my_pos          = self._world_pos(self.body_path)
        pallet_pos      = self._collect_pallet_positions()

        # Separate free candidates by type: (path, dist)
        buffer_free: list[tuple[str, float]] = []
        rack_free:   list[tuple[str, float]] = []

        for stage_prim in self._stage.Traverse():
            matched_attr = ""
            for attr_name in ("navRacksPosId", "navBufferPosId"):
                a = stage_prim.GetAttribute(attr_name)
                if a and a.IsValid() and str(a.Get()) == area_id:
                    matched_attr = attr_name
                    break
            if not matched_attr:
                continue

            prim_path = str(stage_prim.GetPath())
            prim_pos  = self._world_pos(prim_path)

            if self._is_drop_pos_occupied(prim_pos, pallet_pos):
                self._print(f"  drop candidate {prim_path.split('/')[-1]} occupied — skipped")
                continue

            d = math.hypot(prim_pos[0] - my_pos[0], prim_pos[1] - my_pos[1])
            if matched_attr == "navBufferPosId":
                buffer_free.append((prim_path, d))
            else:
                rack_free.append((prim_path, d))

        # Buffer: farthest within 15 m, else farthest overall
        best_path = ""
        best_dist = float("inf")
        best_attr = ""
        if buffer_free:
            within_15 = [(p, d) for p, d in buffer_free if d <= 15.0]
            if within_15:
                best_path, best_dist = max(within_15, key=lambda x: x[1])
            else:
                best_path, best_dist = max(buffer_free, key=lambda x: x[1])
            best_attr = "navBufferPosId"

        # Rack: nearest free (only used when no buffer match)
        if rack_free and not best_path:
            best_path, best_dist = min(rack_free, key=lambda x: x[1])
            best_attr = "navRacksPosId"

        if best_path:
            rel.SetTargets([Sdf.Path(best_path)])
            self._print(
                f"navDropTransform -> {best_path.split('/')[-1]}"
                f"  ({best_attr}={area_id}  dist={best_dist:.2f}m)"
            )
        else:
            carb.log_warn(f"{self._name}  drop area '{area_id}': no free position found")

    def _auto_start_pick_nav(self, pallet_path: str) -> None:
        """Set navGoTo toward the closest approach node to the pallet.

        For dock pallets (path contains /Pallets/Docking) routes toward the matching
        dock entrance prim and records dock state so the retreating phase measures
        distance from the dock instead of from the pick start point.
        For normal pallets the previous approach-node logic applies.
        """
        _, positions = self._build_nav_graph()
        if not positions:
            carb.log_warn(f"{self._name}  Pick nav: waypoint graph empty - going direct to prepick")
            # No graph: skip node navigation and let _update_pick drive directly to prepick
            self._pick_phase = "idle"
            self._set_bool("navGoToPick", True)
            return

        my_pos = self._world_pos(self.body_path)

        if "/Pallets/Docking" in pallet_path:
            dock_path = self._find_dock_for_pallet(pallet_path)
            if dock_path:
                self._picking_from_dock = True
                self._dock_path = dock_path
                # Route toward the dock entrance - the pallet is deep inside the dock
                dock_pos = self._world_pos(dock_path)
                closest  = self._closest_node(dock_pos, positions)
                root_prim = self._stage.GetPrimAtPath(self.root)
                if root_prim and root_prim.IsValid():
                    rel = root_prim.GetRelationship("navClosestNode")
                    if rel:
                        rel.SetTargets([Sdf.Path(closest)])
                predock_pos, predock_yaw = self._compute_predock_approach(dock_path)
                self._prepick_pos  = predock_pos
                self._prepick_yaw  = predock_yaw
                self._pick_phase   = "nav_to_dock"
                self._set_bool("navGoTo", True)
                self._print(
                    f"Dock pick nav -> {closest.split('/')[-1]}"
                    f"  dock={dock_path.split('/')[-1]}"
                    f"  predock ({predock_pos[0]:.2f}, {predock_pos[1]:.2f})"
                )
                return

        # Normal pick navigation
        self._picking_from_dock = False
        self._dock_path = ""
        # Initial pre-pick estimate (forklift pos) - used only to filter candidate nodes.
        prepick_pos, _ = self._compute_prepick(pallet_path)

        pallet_pos   = self._world_pos(pallet_path)
        dist_to_pick = math.hypot(prepick_pos[0] - my_pos[0], prepick_pos[1] - my_pos[1])
        # Restrict to nodes not further from the forklift than the pre-pick position
        # and at least 2 m from the pallet so the approach alignment has room to work.
        approach_nodes = {
            k: v for k, v in positions.items()
            if (math.hypot(v[0] - my_pos[0], v[1] - my_pos[1]) <= dist_to_pick
                and math.hypot(v[0] - pallet_pos[0], v[1] - pallet_pos[1]) >= 2.0)
        }
        closest     = self._closest_node(prepick_pos, approach_nodes or positions)
        closest_xy  = positions.get(closest, (my_pos[0], my_pos[1]))

        # Recompute pre-pick using the chosen node so the +Y/−Y side is pinned to the node.
        prepick_pos, _ = self._compute_prepick(pallet_path, approach_pos=closest_xy)

        root_prim = self._stage.GetPrimAtPath(self.root)
        if root_prim and root_prim.IsValid():
            rel = root_prim.GetRelationship("navClosestNode")
            if rel:
                rel.SetTargets([Sdf.Path(closest)])

        self._pick_phase = "idle"
        self._set_bool("navGoTo", True)
        self._print(
            f"Pick nav -> {closest.split('/')[-1]}"
            f"  pre-pick ({prepick_pos[0]:.2f}, {prepick_pos[1]:.2f})"
        )
        # No "no route" check here - _nav_path has not been planned yet at this point.
        # _update_pick's idle phase calls _start_navigation() and handles the fallback.

    def _find_dock_for_pallet(self, pallet_path: str) -> str:
        """Return the USD path of the Dock_* prim whose Y axis aligns with the pallet.

        Iterates every child of /World/Dockings, projects the pallet position onto
        each dock's local Y axis, and picks the dock with the smallest lateral offset
        below dock_align_lateral_tol.

        Args:
            pallet_path: USD path to the pallet prim.

        Returns:
            Matched dock prim path, or "" if none found.
        """
        pallet_pos    = self._world_pos(pallet_path)
        dockings_prim = self._stage.GetPrimAtPath("/World/Dockings")
        if not dockings_prim or not dockings_prim.IsValid():
            carb.log_warn(f"{self._name}  /World/Dockings not found in stage")
            return ""

        best_path    = ""
        best_lateral = float("inf")

        for child in dockings_prim.GetChildren():
            if not child.GetName().startswith("Dock_"):
                continue
            xform_mat = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            dock_pos  = xform_mat.ExtractTranslation()
            dock_y_w  = xform_mat.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))

            # Work in XY plane only
            dy_len = math.hypot(float(dock_y_w[0]), float(dock_y_w[1]))
            if dy_len < 1e-6:
                continue
            yn_x = float(dock_y_w[0]) / dy_len
            yn_y = float(dock_y_w[1]) / dy_len

            # Vector from dock entrance to pallet in XY
            to_x = pallet_pos[0] - float(dock_pos[0])
            to_y = pallet_pos[1] - float(dock_pos[1])

            # Lateral distance = perpendicular component (2-D cross product magnitude)
            lateral = abs(to_x * (-yn_y) + to_y * yn_x)
            if lateral < self.dock_align_lateral_tol and lateral < best_lateral:
                best_lateral = lateral
                best_path    = str(child.GetPath())

        if not best_path:
            carb.log_warn(f"{self._name}  No dock found for pallet {pallet_path}")
        return best_path

    def _compute_predock_approach(
        self, dock_path: str
    ) -> tuple[tuple[float, float, float], float]:
        """Compute the pre-dock position and approach yaw for a dock entrance prim.

        The pre-dock position is dock_predock_dist metres from the dock entrance along
        the dock's local Y axis, on the side closer to the forklift (outside the dock).
        The returned approach yaw points the forklift's forks (-X local) into the dock.

        Args:
            dock_path: USD path to the Dock_* prim.

        Returns:
            (predock_xyz, approach_yaw_radians)
        """
        dock_prim = self._stage.GetPrimAtPath(dock_path)
        if not dock_prim or not dock_prim.IsValid():
            return (0.0, 0.0, 0.0), 0.0

        xform_mat = UsdGeom.Xformable(dock_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        dock_pos  = xform_mat.ExtractTranslation()
        dock_y_w  = xform_mat.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))

        dy_len = math.hypot(float(dock_y_w[0]), float(dock_y_w[1]))
        if dy_len < 1e-6:
            return (float(dock_pos[0]), float(dock_pos[1]), float(dock_pos[2])), 0.0

        yn_x = float(dock_y_w[0]) / dy_len
        yn_y = float(dock_y_w[1]) / dy_len
        d    = self.dock_predock_dist

        # Two candidates: dock ± d * Y_axis
        pos_plus  = (float(dock_pos[0]) + d * yn_x, float(dock_pos[1]) + d * yn_y, float(dock_pos[2]))
        pos_minus = (float(dock_pos[0]) - d * yn_x, float(dock_pos[1]) - d * yn_y, float(dock_pos[2]))

        # Pick the side closer to the forklift (the outside / approach side)
        my_pos = self._world_pos(self.body_path)
        if math.hypot(pos_plus[0] - my_pos[0], pos_plus[1] - my_pos[1]) <= \
           math.hypot(pos_minus[0] - my_pos[0], pos_minus[1] - my_pos[1]):
            predock_pos  = pos_plus
            into_x, into_y = -yn_x, -yn_y   # toward dock (inward)
        else:
            predock_pos  = pos_minus
            into_x, into_y = yn_x, yn_y

        # Forks face the -X local axis, so approach yaw = direction the forks must point
        approach_yaw = math.atan2(into_y, into_x)
        return predock_pos, approach_yaw

    def _get_pallet_prim_path(self) -> str:
        """Return the prim path stored in navPalletToPick, or empty string."""
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim:
            return ""
        rel     = prim.GetRelationship("navPalletToPick")
        targets = rel.GetTargets() if rel else []
        return str(targets[0]) if targets else ""

    def _compute_prepick(
        self,
        pallet_path: str,
        approach_pos: tuple[float, float] | None = None,
    ) -> tuple[tuple[float, float, float], float]:
        """Compute the pre-pick position and approach yaw for a pallet.

        The pre-pick lies on the pallet's local Y axis (the fork-pocket axis) at
        ``pick_prepick_dist`` from the pallet centre.  +Y or −Y is chosen so that
        the pre-pick is on the same side as ``approach_pos`` - i.e. between the
        approach point and the pallet.

        Args:
            pallet_path:  USD path to the pallet prim.
            approach_pos: XY world position used to choose +Y vs −Y side.
                          If None the forklift's current body position is used.

        Returns:
            (prepick_xyz, approach_yaw_radians) where approach_yaw points from
            the pre-pick position toward the pallet centre (forks direction).
        """
        pallet_prim = self._stage.GetPrimAtPath(pallet_path)
        if not pallet_prim or not pallet_prim.IsValid():
            return (0.0, 0.0, 0.0), 0.0

        xform      = UsdGeom.Xformable(pallet_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pallet_pos = xform.ExtractTranslation()
        pallet_y   = xform.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))

        # Normalise the pallet Y axis in the XY plane
        y_len = math.hypot(float(pallet_y[0]), float(pallet_y[1]))
        if y_len < 1e-6:
            return (float(pallet_pos[0]), float(pallet_pos[1]), float(pallet_pos[2])), 0.0

        yn_x = float(pallet_y[0]) / y_len
        yn_y = float(pallet_y[1]) / y_len
        pz   = float(pallet_pos[2])

        d         = self.pick_prepick_dist
        pos_plus  = (float(pallet_pos[0]) + d * yn_x, float(pallet_pos[1]) + d * yn_y, pz)
        pos_minus = (float(pallet_pos[0]) - d * yn_x, float(pallet_pos[1]) - d * yn_y, pz)

        # Use the supplied approach position (e.g. last node) or fall back to forklift pos
        ref_x, ref_y = approach_pos if approach_pos is not None else self._world_pos(self.body_path)[:2]
        dist_plus    = math.hypot(pos_plus[0]  - ref_x, pos_plus[1]  - ref_y)
        dist_minus   = math.hypot(pos_minus[0] - ref_x, pos_minus[1] - ref_y)

        if dist_plus <= dist_minus:
            prepick_pos             = pos_plus
            toward_x, toward_y     = -yn_x, -yn_y   # from +Y side -> pallet = −Y
        else:
            prepick_pos             = pos_minus
            toward_x, toward_y     = yn_x, yn_y      # from −Y side -> pallet = +Y

        # Forklift forks face local -X; align -X with toward_pallet
        approach_yaw = math.atan2(toward_y, toward_x)

        return prepick_pos, approach_yaw

    def _final_align_for_pick(self) -> None:
        """Pivot in place at the final waypoint node to face the pre-pick position.

        Sequence:
          1. (idle)      Resolve pallet, compute pre-pick anchored to the node's side.
          2. (node_align) Rotate at the node (max_pivot_radius clamp, same as _turning_at_node).
          3.             Once heading error ≤ preturn_threshold_deg -> hand off to prepick_drive.
        """
        # Resolve the final node first - needed to pin the pre-pick Y side.
        prim    = self._stage.GetPrimAtPath(self.root)
        rel     = prim.GetRelationship("navClosestNode") if prim else None
        targets = rel.GetTargets() if rel else []
        if not targets:
            return

        node_path = str(targets[0])
        node_pos  = self._world_pos(node_path)
        node_name = node_path.split("/")[-1]

        # ── First call: resolve pallet and compute pre-pick ────────────────
        # "nav_to_dock" is the dock-pick equivalent of "idle" - predock pos is already
        # in _prepick_pos but the dock-specific retreat target hasn't been set yet.
        if self._pick_phase in ("idle", "nav_to_dock"):
            self._refresh_pallet_ref()
            pallet_path = self._get_pallet_prim_path()
            if not pallet_path:
                return
            self._dock_at_predock = False
            if self._picking_from_dock and self._dock_path:
                prepick_pos, prepick_yaw = self._compute_predock_approach(self._dock_path)
                dock_pos = self._world_pos(self._dock_path)
                out_dx = prepick_pos[0] - dock_pos[0]
                out_dy = prepick_pos[1] - dock_pos[1]
                out_len = math.hypot(out_dx, out_dy) or 1.0
                self._dock_retreat_target_pos = (
                    prepick_pos[0] + self.dock_retreat_extra * out_dx / out_len,
                    prepick_pos[1] + self.dock_retreat_extra * out_dy / out_len,
                )
                label = "predock"
            else:
                # Anchor +Y/−Y choice to the node so pre-pick lies between node and pallet.
                prepick_pos, prepick_yaw = self._compute_prepick(
                    pallet_path, approach_pos=(node_pos[0], node_pos[1])
                )
                label = "prepick"
            self._prepick_pos             = prepick_pos
            self._prepick_yaw             = prepick_yaw
            self._pallet_path             = pallet_path
            self._align_dir               = 1
            self._align_returning_to_node = False
            self._node_align_label        = label

            # If the pre-position is very close to the node, skip the node-pivot
            # and drive straight to the pre-position - prepick_align handles heading.
            dist_node_to_pre = math.hypot(
                prepick_pos[0] - node_pos[0],
                prepick_pos[1] - node_pos[1],
            )
            if dist_node_to_pre < 1.0:
                self._aligning   = False
                self._nav_path   = []
                self._pick_phase = "prepick_drive"
                self._set_bool("navEnabled", False)
                self._set_bool("navGoTo", False)
                self._set_bool("navGoToPick", True)
                rel.SetTargets([])
                self._print(
                    f"Pick: {label} {dist_node_to_pre:.2f}m from {node_name} "
                    f"(<1m) - skip node-align, driving direct to {label}"
                )
                return

            self._pick_phase = "node_align"
            self._print(
                f"Pick node-align: pivoting at {node_name} toward {label} "
                f"({prepick_pos[0]:.2f}, {prepick_pos[1]:.2f})"
            )

        # ── Pivot at the node toward the pre-pick (identical to _turning_at_node) ──
        # Target heading: direction from node -> prepick/predock
        dx         = self._prepick_pos[0] - node_pos[0]
        dy         = self._prepick_pos[1] - node_pos[1]
        target_yaw = math.atan2(dy, dx)

        my_pos      = self._world_pos(self.body_path)
        my_yaw      = self._world_yaw(self.body_path)
        heading_err = self._wrap(target_yaw - my_yaw)

        if abs(heading_err) <= math.radians(self.preturn_threshold_deg):
            # Heading aligned -> transition to prepick_drive
            self._aligning   = False
            self._nav_path   = []
            self._pick_phase = "prepick_drive"
            self._set_bool("navEnabled", False)
            self._set_bool("navGoTo", False)
            self._set_bool("navGoToPick", True)
            rel.SetTargets([])
            self._print(f"Pick node-align done at {node_name} -> {self._node_align_label}_drive")
            return

        # Steering: heading error with align_dir sign (rear-wheel convention)
        turn = max(-self.max_turn,
                   min(self.max_turn, heading_err * self.turn_gain * self._align_dir))

        # Along-axis projection to choose forward / backward
        pivot_dx   = node_pos[0] - my_pos[0]
        pivot_dy   = node_pos[1] - my_pos[1]
        approach_x = math.cos(target_yaw)
        approach_y = math.sin(target_yaw)
        along      = pivot_dx * approach_x + pivot_dy * approach_y

        if along > self.align_hysteresis:
            self._align_dir = 1
        elif along < -self.align_hysteresis:
            self._align_dir = -1

        # Radius clamp: if drift > max_pivot_radius, drive back toward the node
        dist_from_node = math.hypot(pivot_dx, pivot_dy)
        if dist_from_node > self.max_pivot_radius and abs(along) > 0.1:
            self._align_dir = 1 if along > 0 else -1

        self._print(
            f"aligning {node_name} -> {self._node_align_label}  "
            f"hdg_err={math.degrees(heading_err):.1f}°  "
            f"dist={dist_from_node:.3f}  along={along:.3f}"
        )
        self.drive(self.align_speed * self._align_dir, turn)

    def _update_pick(self) -> None:
        """State machine driving the full pick sequence.

        Phases:
          idle         - resolve pallet/prepick, find closest node, start graph nav
          nav_to_node  - follow waypoint graph to the closest node near the pallet
          node_align   - align heading toward the pre-pick position (_final_align_for_pick)
          prepick_drive - cruise to pre-pick position (loose tolerance)
          prepick_align - tight alignment at pre-pick (pos + heading)
          picking      - drive forward until forks are inside the pallet
          lifting      - raise forks and wait
          retreating   - reverse back to pre-pick position
        """
        phase = self._pick_phase

        if phase == "idle":
            self._refresh_pallet_ref()
            pallet_path = self._get_pallet_prim_path()
            if not pallet_path:
                return

            prepick_pos, prepick_yaw = self._compute_prepick(pallet_path)
            self._prepick_pos = prepick_pos
            self._prepick_yaw = prepick_yaw
            self._pallet_path = pallet_path

            # Find the closest waypoint node to the pre-pick position and route there.
            # Only consider nodes that are NOT further from the forklift than the pallet
            # itself - this prevents routing through a node that is past the pallet.
            _, positions = self._build_nav_graph()

            # ── No graph or no route: go directly to prepick ─────────────────
            def _go_direct_prepick(reason: str) -> None:
                carb.log_warn(f"{self._name}  Pick ({reason}): going direct to prepick")
                self._aligning             = False
                self._align_dir            = 1
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = 0
                self._lateral_phase        = 0
                self._retreat_for_hdg      = False
                self._pick_phase           = "prepick_drive"
                self._print(
                    f"Pick (no graph): direct to prepick "
                    f"({prepick_pos[0]:.2f}, {prepick_pos[1]:.2f})"
                )

            if not positions:
                _go_direct_prepick("empty graph")
                return

            my_pos       = self._world_pos(self.body_path)
            pallet_pos   = self._world_pos(self._pallet_path) if self._pallet_path else my_pos
            dist_to_pick = math.hypot(
                prepick_pos[0] - my_pos[0],
                prepick_pos[1] - my_pos[1],
            )
            approach_nodes = {
                k: v for k, v in positions.items()
                if (math.hypot(v[0] - my_pos[0], v[1] - my_pos[1]) <= dist_to_pick
                    and math.hypot(v[0] - pallet_pos[0], v[1] - pallet_pos[1]) >= 2.0)
            }
            closest      = self._closest_node(prepick_pos, approach_nodes or positions)
            closest_xy   = positions.get(closest, (my_pos[0], my_pos[1]))

            # Recompute pre-pick using the chosen node as the approach reference so the
            # pre-pick is guaranteed to be on the pallet Y axis on the node's side.
            prepick_pos, prepick_yaw = self._compute_prepick(
                pallet_path, approach_pos=closest_xy
            )
            self._prepick_pos = prepick_pos
            self._prepick_yaw = prepick_yaw

            prim = self._stage.GetPrimAtPath(self.root)
            if prim and prim.IsValid():
                rel = prim.GetRelationship("navClosestNode")
                if rel:
                    rel.SetTargets([Sdf.Path(closest)])

            self._aligning             = False
            self._align_dir            = 1
            self._lat_overshoot_side   = 0
            self._initial_lateral_sign = 0
            self._lateral_phase        = 0
            self._retreat_for_hdg      = False
            self._start_navigation()

            if not self._nav_path:
                _go_direct_prepick("no route")
                return

            self._pick_phase = "nav_to_node"
            self._print(
                f"Pick: routing to "
                f"{closest.split('/')[-1]}  pre-pick "
                f"({prepick_pos[0]:.2f}, {prepick_pos[1]:.2f})"
            )

        elif phase == "nav_to_node":
            if self._aligning:
                # Arrived at the closest node - reset alignment state, start node heading alignment.
                self._align_dir            = 1
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = 0
                self._lateral_phase        = 0
                self._retreat_for_hdg      = False
                self._pick_phase = "node_align"
                self._final_align_for_pick()
            elif self._nav_path:
                self._follow_path()
            else:
                # Graph built but no path (already at closest node) - go straight to align.
                self._aligning   = True
                self._align_dir  = 1
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = 0
                self._pick_phase = "node_align"
                self._final_align_for_pick()

        elif phase == "node_align":
            self._final_align_for_pick()

        elif phase == "prepick_drive":
            if self._proximity_blocked and self._direct_moving_toward_block(
                (self._prepick_pos[0], self._prepick_pos[1])
            ):
                self.stop()
                return

            my_pos = self._world_pos(self.body_path)
            dx     = self._prepick_pos[0] - my_pos[0]
            dy     = self._prepick_pos[1] - my_pos[1]
            dist   = math.hypot(dx, dy)

            if dist <= self.arrival_distance:
                self.stop()
                self._align_dir      = 1
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = 0
                self._lateral_phase        = 0
                self._lateral_steer_sign   = 1
                self._pick_phase     = "prepick_align"
                self._print(f"Pick: reached pre-pick, aligning")
                return

            my_yaw    = self._world_yaw(self.body_path)
            angle_err = self._wrap(math.atan2(dy, dx) - my_yaw)
            turn      = max(-self.max_turn, min(self.max_turn, angle_err * self.turn_gain))
            speed     = self.approach_speed if dist < self.slow_distance else self.move_speed
            self.drive(speed, turn)

        elif phase == "prepick_align":
            my_pos = self._world_pos(self.body_path)
            dx     = self._prepick_pos[0] - my_pos[0]
            dy     = self._prepick_pos[1] - my_pos[1]
            dist   = math.hypot(dx, dy)
            if dist > self.arrival_distance and self._lateral_phase == 0 and not self._retreat_for_hdg:
                my_yaw    = self._world_yaw(self.body_path)
                angle_err = self._wrap(math.atan2(dy, dx) - my_yaw)
                turn      = max(-self.max_turn, min(self.max_turn, angle_err * self.turn_gain))
                speed     = self.approach_speed if dist < self.slow_distance else self.move_speed
                hdg_deg   = math.degrees(self._wrap(my_yaw - self._prepick_yaw))
                self.drive(speed, turn)
                self._print(
                    f"driving to pre-pick"
                    f"  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={hdg_deg:.1f}°"
                )
                return
            if self._align_to_pose(
                self._prepick_pos, self._prepick_yaw,
                self.pick_align_pos_tol, self.pick_align_angle_tol,
                log_prefix="predock align" if (self._picking_from_dock and not self._dock_at_predock) else "prepick align",
                no_overshoot=True,
            ):
                if self._picking_from_dock and not self._dock_at_predock:
                    # First alignment was predock (outside dock).
                    # Now drive to the true prepick (3 m from pallet, inside dock).
                    self._dock_at_predock = True
                    prepick_pos, prepick_yaw = self._compute_prepick(self._pallet_path)
                    self._prepick_pos = prepick_pos
                    self._prepick_yaw = prepick_yaw
                    self._align_dir            = 1
                    self._lat_overshoot_side   = 0
                    self._initial_lateral_sign = 0
                    self._lateral_phase        = 0
                    self._lateral_steer_sign   = 1
                    self._pick_phase = "prepick_drive"
                    self._print(
                        f"Predock aligned -> prepick"
                        f"  ({prepick_pos[0]:.2f}, {prepick_pos[1]:.2f})"
                    )
                    return

                # Read pallet Z and raise forks to approach height, then dwell
                pallet_z    = self._world_pos(self._pallet_path)[2]
                lift_target = pallet_z + self.pick_lift_approach_offset
                lower_limit = self._lift_lower_limit()
                clamped     = lift_target < lower_limit
                lift_target = max(lower_limit, lift_target)
                self._lift_target_pos = lift_target
                self.set_lift_position(lift_target)
                self._lift_timer = 0.0
                self._align_dir  = 1
                self._pick_phase = "prepick_stop"
                self._print(
                    f"Pick: aligned,"
                    f" pallet Z={pallet_z:.3f}  forks->{lift_target:.3f}"
                    + ("  (clamped to lower limit)" if clamped else "")
                )

        elif phase == "prepick_stop":
            self.stop()
            self._lift_timer += self.delta_time
            current_lift = self._lift_current_pos()
            current_vel  = self._lift_current_vel()
            pos_str      = f"{current_lift:.4f}" if current_lift is not None else "?"
            vel_str      = f"{current_vel:.4f}" if current_vel is not None else "?"
            # Ready when settled (velocity ≤ 0.005 m/s) OR at target position.
            vel_settled  = current_vel is not None and abs(current_vel) <= 0.005
            pos_reached  = (
                current_lift is not None
                and abs(current_lift - self._lift_target_pos) <= self.pick_lift_pos_tol
            )
            lift_ready   = vel_settled or pos_reached
            self._print(
                f"prepick_stop"
                f"  pos={pos_str}  vel={vel_str}  target={self._lift_target_pos:.4f}"
                f"  ready={lift_ready}  t={self._lift_timer:.2f}s"
            )
            if lift_ready and self._lift_timer >= self.pick_prepick_stop_wait:
                my_pos = self._world_pos(self.body_path)
                self._picking_start = (my_pos[0], my_pos[1])
                self._pick_phase = "picking"
                self._print(
                    f"Pick: forks at height"
                    f" (pos={pos_str}  vel={vel_str}), inserting"
                )

        elif phase == "picking":
            my_pos     = self._world_pos(self.body_path)
            pallet_pos = self._world_pos(self._pallet_path)
            # Measure distance driven from prepick - robust when pallet moves with forks.
            # Fallback: body-to-pallet in case forklift started picking from a close position.
            driven = math.hypot(
                my_pos[0] - self._picking_start[0],
                my_pos[1] - self._picking_start[1],
            )
            body_dist = math.hypot(pallet_pos[0] - my_pos[0], pallet_pos[1] - my_pos[1])

            if driven >= self.pick_insert_drive_dist or body_dist <= self.pick_insert_offset:
                self.stop()
                self._lift_target_pos = self.pick_lift_height
                self.set_lift_position(self.pick_lift_height)
                self._lift_timer = 0.0
                self._pick_phase = "lifting"
                self._print(
                    f"Pick: forks inserted"
                    f"  driven={driven:.3f}  body_dist={body_dist:.3f}"
                    f", raising to {self.pick_lift_height:.3f}"
                )
                return

            heading_deg = abs(math.degrees(self._wrap(self._world_yaw(self.body_path) - self._prepick_yaw)))
            self._print(
                f"picking"
                f"  driven={driven:.3f}/{self.pick_insert_drive_dist:.2f}"
                f"  body_dist={body_dist:.3f}"
                f"  heading={heading_deg:.1f}°"
            )
            self.drive(self.align_speed, 0.0)

        elif phase == "lifting":
            self.stop()
            self._lift_timer += self.delta_time
            current_lift = self._lift_current_pos()
            current_vel  = self._lift_current_vel()
            vel_settled  = current_vel is not None and abs(current_vel) <= 0.005
            pos_reached  = (
                current_lift is not None
                and abs(current_lift - self._lift_target_pos) <= self.pick_lift_pos_tol
            )
            lift_ready   = vel_settled or pos_reached
            if lift_ready and self._lift_timer >= self.pick_lift_wait:
                self._set_bool("navWithPallet", True)
                # Capture pallet root path BEFORE the relationship is cleared at pick-complete
                self._carried_pallet_root = self._get_pallet_prim_path()
                self.set_lift_position(self.pick_lift_height)   # travel height
                my_pos = self._world_pos(self.body_path)
                self._pick_retreat_start = (my_pos[0], my_pos[1])
                self._pick_phase = "retreating"
                pos_str = f"{current_lift:.3f}" if current_lift is not None else "?"
                vel_str = f"{current_vel:.4f}" if current_vel is not None else "?"
                self._print(
                    f"Pick: lift in position"
                    f" (pos={pos_str}  vel={vel_str}), retreating {self.drop_retreat_dist}m"
                )

        elif phase == "retreating":
            my_pos = self._world_pos(self.body_path)

            if self._picking_from_dock and self._dock_path:
                # Dock pick: reverse straight back through predock then dock_retreat_extra m more.
                # Done when body reaches _dock_retreat_target_pos (predock + 3 m outside).
                rtx, rty = self._dock_retreat_target_pos
                dist_to_target = math.hypot(my_pos[0] - rtx, my_pos[1] - rty)
                self._print(
                    f"pick dock retreat"
                    f"  dist_to_target={dist_to_target:.3f}  done<={self.arrival_distance:.2f}m"
                )
                done = dist_to_target <= self.arrival_distance
            else:
                # Normal pick: retreat drop_retreat_dist from the lift start point
                travelled = math.hypot(
                    my_pos[0] - self._pick_retreat_start[0],
                    my_pos[1] - self._pick_retreat_start[1],
                )
                dx = my_pos[0] - self._pick_retreat_start[0]
                dy = my_pos[1] - self._pick_retreat_start[1]
                self._print(
                    f"pick retreating"
                    f"  dx={dx:.3f}  dy={dy:.3f}"
                    f"  travelled={travelled:.3f}  target={self.drop_retreat_dist:.2f}m"
                )
                done = travelled >= self.drop_retreat_dist - self.drop_retreat_tol

            if done:
                self.stop()
                self._pick_phase = "idle"
                self._set_bool("navGoToPick", False)
                # Clear pallet reference so drop nav uses final_alignment() not _final_align_for_pick()
                root_prim = self._stage.GetPrimAtPath(self.root)
                if root_prim and root_prim.IsValid():
                    rel = root_prim.GetRelationship("navPalletToPick")
                    if rel:
                        rel.SetTargets([])
                self._print(f"Pick: complete")
                return

            # Reverse straight back - forks remain facing the pallet
            self.drive(-self.align_speed, 0.0)

    # ── Drop system ────────────────────────────────────────────────────────

    def _has_drop_transform(self) -> bool:
        """Return True if navDropTransform relationship has a target."""
        prim = self._stage.GetPrimAtPath(self.root)
        if not prim or not prim.IsValid():
            return False
        rel = prim.GetRelationship("navDropTransform")
        return bool(rel and rel.GetTargets())

    def _compute_predrop(
        self, drop_path: str
    ) -> tuple[tuple[float, float, float], float]:
        """Compute the pre-drop position and approach yaw for the drop transform.

        The approach axis is read from the prim's `dropAlignment` attribute ("X", "Y", or "Z").
        If the attribute is absent the default is "X" (pallet slots on the X faces).

        Args:
            drop_path: USD path to the navDropTransform prim.

        Returns:
            (predrop_xyz, approach_yaw_radians)
        """
        drop_prim = self._stage.GetPrimAtPath(drop_path)
        if not drop_prim or not drop_prim.IsValid():
            return (0.0, 0.0, 0.0), 0.0

        # Determine approach axis from attribute; default X
        axis_attr = drop_prim.GetAttribute("dropAlignment")
        if axis_attr and axis_attr.IsValid() and axis_attr.Get() is not None:
            axis_str = str(axis_attr.Get()).strip().upper()
        else:
            axis_str = "X"
        _axis_map = {"X": Gf.Vec3d(1.0, 0.0, 0.0), "Y": Gf.Vec3d(0.0, 1.0, 0.0), "Z": Gf.Vec3d(0.0, 0.0, 1.0)}
        local_axis = _axis_map.get(axis_str, Gf.Vec3d(1.0, 0.0, 0.0))

        xform    = UsdGeom.Xformable(drop_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        drop_pos = xform.ExtractTranslation()
        drop_y   = xform.TransformDir(local_axis)

        y_len = math.hypot(float(drop_y[0]), float(drop_y[1]))
        if y_len < 1e-6:
            return (float(drop_pos[0]), float(drop_pos[1]), float(drop_pos[2])), 0.0

        yn_x = float(drop_y[0]) / y_len
        yn_y = float(drop_y[1]) / y_len
        dz   = float(drop_pos[2])

        d = self.drop_prepick_dist
        pos_plus  = (float(drop_pos[0]) + d * yn_x, float(drop_pos[1]) + d * yn_y, dz)
        pos_minus = (float(drop_pos[0]) - d * yn_x, float(drop_pos[1]) - d * yn_y, dz)

        my_pos     = self._world_pos(self.body_path)
        dist_plus  = math.hypot(pos_plus[0]  - my_pos[0], pos_plus[1]  - my_pos[1])
        dist_minus = math.hypot(pos_minus[0] - my_pos[0], pos_minus[1] - my_pos[1])

        if dist_plus <= dist_minus:
            predrop_pos    = pos_plus
            toward_x, toward_y = -yn_x, -yn_y
        else:
            predrop_pos    = pos_minus
            toward_x, toward_y = yn_x, yn_y

        approach_yaw = math.atan2(toward_y, toward_x)
        return predrop_pos, approach_yaw

    def _final_align_for_drop(self) -> None:
        """At the final waypoint node, transition directly to the predrop sequence.

        No alignment is performed at the graph node - the forklift drives to the
        predrop position and aligns there (handled by predrop_align in _update_drop).
        Called every frame while _aligning=True and navWithPallet=True.
        """
        prim_root = self._stage.GetPrimAtPath(self.root)
        if not prim_root or not prim_root.IsValid():
            return
        drop_rel = prim_root.GetRelationship("navDropTransform")
        targets  = drop_rel.GetTargets() if drop_rel else []
        if not targets:
            return
        drop_path = str(targets[0])

        # Capture drop area info before relationships are cleared at drop completion.
        # Check navRacksPosId first (Racks/Docks), then navBufferPosId (Buffer zones).
        drop_prim_ref = self._stage.GetPrimAtPath(drop_path)
        if drop_prim_ref and drop_prim_ref.IsValid():
            for _attr_name in ("navRacksPosId", "navBufferPosId"):
                _a = drop_prim_ref.GetAttribute(_attr_name)
                if _a and _a.IsValid() and _a.Get():
                    self._drop_nav_area       = str(_a.Get())
                    self._drop_area_attr_type = _attr_name
                    break

        predrop_pos, predrop_yaw = self._compute_predrop(drop_path)
        self._predrop_pos          = predrop_pos
        self._predrop_yaw          = predrop_yaw
        self._drop_target_path     = drop_path
        drop_world                 = self._world_pos(drop_path)
        self._drop_z               = float(drop_world[2])
        self._align_dir            = 1
        self._lat_overshoot_side   = 0
        self._initial_lateral_sign = 0

        # Transition immediately - alignment happens AT predrop via predrop_align phase
        self._aligning   = False
        self._nav_path   = []
        self._drop_phase = "predrop_drive"
        self._set_bool("navEnabled", False)
        self._set_bool("navGoTo", False)
        self._set_bool("navGoToDrop", True)
        prim = self._stage.GetPrimAtPath(self.root)
        if prim:
            rel = prim.GetRelationship("navClosestNode")
            if rel:
                rel.SetTargets([])
        self._print(
            f"Drop: graph node reached -> predrop "
            f"({predrop_pos[0]:.2f}, {predrop_pos[1]:.2f})"
            f"  yaw={math.degrees(predrop_yaw):.1f}°"
        )

    def _update_drop(self) -> None:
        """State machine driving the full drop sequence.

        Phases:
          idle          - resolve drop transform, compute predrop, transition to predrop_drive
          predrop_drive - cruise to pre-drop position
          predrop_pivot - in-place pivot when arriving with large heading error (> predrop_pivot_hdg_threshold)
          predrop_align - tight alignment at pre-drop (no_overshoot=True)
          predrop_stop  - stop and raise forks to drop_Z + approach offset, wait for settled
          dropping      - drive forward until forks reach the drop position
          lowering      - lower forks to drop_Z + lower offset, wait for settled
          retreating    - reverse to pre-drop, clear navWithPallet, complete
        """
        phase = self._drop_phase

        if phase == "idle":
            prim_root = self._stage.GetPrimAtPath(self.root)
            if not prim_root or not prim_root.IsValid():
                return
            drop_rel = prim_root.GetRelationship("navDropTransform")
            targets  = drop_rel.GetTargets() if drop_rel else []
            if not targets:
                return
            drop_path = str(targets[0])
            predrop_pos, predrop_yaw = self._compute_predrop(drop_path)
            self._predrop_pos      = predrop_pos
            self._predrop_yaw      = predrop_yaw
            self._drop_target_path = drop_path
            drop_world             = self._world_pos(drop_path)
            self._drop_z           = float(drop_world[2])
            self._drop_phase       = "predrop_drive"
            self._print(
                f"Drop: driving to pre-drop "
                f"({predrop_pos[0]:.2f}, {predrop_pos[1]:.2f})  yaw={math.degrees(predrop_yaw):.1f}°"
            )

        elif phase == "predrop_drive":
            if self._proximity_blocked and self._direct_moving_toward_block(
                (self._predrop_pos[0], self._predrop_pos[1])
            ):
                self.stop()
                return

            my_pos = self._world_pos(self.body_path)
            dx     = self._predrop_pos[0] - my_pos[0]
            dy     = self._predrop_pos[1] - my_pos[1]
            dist   = math.hypot(dx, dy)

            if dist <= self.arrival_distance:
                self.stop()
                self._align_dir            = 1
                self._lat_overshoot_side   = 0
                self._initial_lateral_sign = 0
                self._lateral_phase        = 0
                self._retreat_for_hdg      = False
                my_yaw    = self._world_yaw(self.body_path)
                hdg_err   = abs(math.degrees(self._wrap(my_yaw - self._predrop_yaw)))
                if hdg_err > self.predrop_pivot_hdg_threshold:
                    self._drop_phase = "predrop_pivot"
                    self._print(
                        f"Drop: reached pre-drop, heading error"
                        f" {hdg_err:.1f}° > {self.predrop_pivot_hdg_threshold}° - pivoting first"
                    )
                else:
                    self._drop_phase = "predrop_align"
                    self._print(f"Drop: reached pre-drop, aligning")
                return

            my_yaw    = self._world_yaw(self.body_path)
            angle_err = self._wrap(math.atan2(dy, dx) - my_yaw)
            turn      = max(-self.max_turn, min(self.max_turn, angle_err * self.turn_gain))
            speed     = self.approach_speed if dist < self.slow_distance else self.move_speed
            self.drive(speed, turn)

        elif phase == "predrop_pivot":
            my_pos    = self._world_pos(self.body_path)
            my_yaw    = self._world_yaw(self.body_path)
            hdg_err   = self._wrap(my_yaw - self._predrop_yaw)
            hdg_err_d = math.degrees(hdg_err)
            dx = self._predrop_pos[0] - my_pos[0]
            dy = self._predrop_pos[1] - my_pos[1]
            dist = math.hypot(dx, dy)
            self._print(
                f"pre-drop pivot"
                f"  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={hdg_err_d:.1f}°"
            )
            if abs(hdg_err) <= math.radians(self.predrop_pivot_angle_tol):
                self.stop()
                self._drop_phase = "predrop_align"
                self._print(f"Drop: pivot done, aligning")
                return
            # Turn toward predrop_yaw: negative heading_err means we need to go CCW (left)
            direction = -1 if hdg_err > 0 else 1
            self.drive(self.pivot_speed, self.max_turn * direction)

        elif phase == "predrop_align":
            my_pos = self._world_pos(self.body_path)
            dx     = self._predrop_pos[0] - my_pos[0]
            dy     = self._predrop_pos[1] - my_pos[1]
            dist   = math.hypot(dx, dy)
            # Coarse-drive back to pre-drop when pivot/retreat moved us away.
            # Skip if crab-walk or retreat is already active - don't interrupt mid-manoeuvre.
            if dist > self.arrival_distance and self._lateral_phase == 0 and not self._retreat_for_hdg:
                my_yaw    = self._world_yaw(self.body_path)
                angle_err = self._wrap(math.atan2(dy, dx) - my_yaw)
                turn      = max(-self.max_turn, min(self.max_turn, angle_err * self.turn_gain))
                speed     = self.approach_speed if dist < self.slow_distance else self.move_speed
                hdg_deg   = math.degrees(self._wrap(my_yaw - self._predrop_yaw))
                self.drive(speed, turn)
                self._print(
                    f"driving to pre-drop"
                    f"  x={dx:.3f}  y={dy:.3f}  dist={dist:.3f}  heading={hdg_deg:.1f}°"
                )
                return
            if self._align_to_pose(
                self._predrop_pos, self._predrop_yaw,
                self.drop_align_pos_tol, self.drop_align_angle_tol,
                log_prefix="pre-drop align",
                no_overshoot=True,
            ):
                drop_world            = self._world_pos(self._drop_target_path)
                self._drop_z          = float(drop_world[2])
                lift_target           = self._drop_z + self.drop_lift_approach_offset
                lift_target           = max(self._lift_lower_limit(), lift_target)
                self._lift_target_pos = lift_target
                self.set_lift_position(lift_target)
                self._lift_timer = 0.0
                self._drop_phase = "predrop_stop"
                self._print(
                    f"Drop: aligned, forks->{lift_target:.3f}"
                )

        elif phase == "predrop_stop":
            self.stop()
            self._lift_timer += self.delta_time
            current_lift = self._lift_current_pos()
            current_vel  = self._lift_current_vel()
            vel_settled  = current_vel is not None and abs(current_vel) <= 0.005
            pos_reached  = (
                current_lift is not None
                and abs(current_lift - self._lift_target_pos) <= self.pick_lift_pos_tol
            )
            lift_ready = vel_settled or pos_reached
            pos_str = f"{current_lift:.4f}" if current_lift is not None else "?"
            vel_str = f"{current_vel:.4f}" if current_vel is not None else "?"
            self._print(
                f"pre-drop stop"
                f"  pos={pos_str}  vel={vel_str}  target={self._lift_target_pos:.4f}"
                f"  ready={lift_ready}  t={self._lift_timer:.2f}s"
            )
            if lift_ready and self._lift_timer >= self.drop_prepick_stop_wait:
                self._drop_phase = "dropping"
                self._print(
                    f"Drop: forks at height"
                    f" (pos={pos_str}  vel={vel_str}), inserting"
                )

        elif phase == "dropping":
            drop_pos = self._world_pos(self._drop_target_path)
            my_pos   = self._world_pos(self.body_path)
            dist     = math.hypot(drop_pos[0] - my_pos[0], drop_pos[1] - my_pos[1])

            if dist <= self.drop_insert_offset:
                self.stop()
                drop_world            = self._world_pos(self._drop_target_path)
                self._drop_z          = float(drop_world[2])
                lift_target           = self._drop_z + self.drop_lower_offset - self._drop_lower_offset_extra
                lift_target           = max(self._lift_lower_limit(), lift_target)
                self._lift_target_pos = lift_target
                self.set_lift_position(lift_target)
                self._lift_timer = 0.0
                self._drop_phase = "lowering"
                self._print(
                    f"Drop: at position, lowering forks to {lift_target:.3f}"
                    + (f"  (retry extra={self._drop_lower_offset_extra:.3f}m)" if self._drop_lower_offset_extra else "")
                )
                return

            heading_deg = abs(math.degrees(
                self._wrap(self._world_yaw(self.body_path) - self._predrop_yaw)
            ))
            self._print(
                f"dropping"
                f"  x={my_pos[0]:.3f}  y={my_pos[1]:.3f}"
                f"  dist={dist:.3f}  heading={heading_deg:.1f}°"
            )
            self.drive(self.align_speed, 0.0)

        elif phase == "lowering":
            self.stop()
            self._lift_timer += self.delta_time
            current_lift = self._lift_current_pos()
            current_vel  = self._lift_current_vel()
            vel_settled  = current_vel is not None and abs(current_vel) <= 0.005
            pos_reached  = (
                current_lift is not None
                and abs(current_lift - self._lift_target_pos) <= self.pick_lift_pos_tol
            )
            lift_ready = vel_settled or pos_reached
            if lift_ready and self._lift_timer >= self.drop_lift_wait:
                # Don't raise forks yet - keep them low until retreat is complete
                # so the pallet can't re-catch on the way back.
                my_pos = self._world_pos(self.body_path)
                self._drop_retreat_start = (my_pos[0], my_pos[1])
                # Record pallet distance now (before retreat) so the post-drop
                # check can verify the pallet moved away relative to this baseline.
                if self._pallet_path:
                    pp = self._world_pos(self._pallet_path)
                    self._drop_pallet_dist_before = math.hypot(
                        pp[0] - my_pos[0], pp[1] - my_pos[1]
                    )
                else:
                    self._drop_pallet_dist_before = 0.0
                self._drop_phase = "retreating"
                pos_str = f"{current_lift:.3f}" if current_lift is not None else "?"
                self._print(
                    f"Drop: forks lowered (pos={pos_str}),"
                    f" pallet dist before={self._drop_pallet_dist_before:.3f}m,"
                    f" retreating {self.drop_retreat_dist}m"
                )

        elif phase == "retreating":
            my_pos    = self._world_pos(self.body_path)
            travelled = math.hypot(
                my_pos[0] - self._drop_retreat_start[0],
                my_pos[1] - self._drop_retreat_start[1],
            )

            if travelled >= self.drop_retreat_dist - self.drop_retreat_tol:
                self.stop()
                # Raise forks to travel height now that we've fully cleared the pallet.
                self.set_lift_position(self.pick_lift_height)

                # Confirm drop: pallet must be at least 2× its pre-drop distance away.
                # Using a relative check is more reliable than an absolute threshold
                # because it accounts for how close the pallet was when placed.
                pallet_dist_after = float("inf")
                if self._pallet_path:
                    pallet_pos        = self._world_pos(self._pallet_path)
                    pallet_dist_after = math.hypot(
                        pallet_pos[0] - my_pos[0], pallet_pos[1] - my_pos[1]
                    )

                # Baseline: use the recorded before-distance, minimum 0.2 m
                dist_before   = max(self._drop_pallet_dist_before, 0.2)
                drop_confirmed = pallet_dist_after >= 2.0 * dist_before

                if not drop_confirmed:
                    # Pallet still on forks - increase lower offset and re-approach.
                    self._drop_lower_offset_extra += self.drop_lower_retry_increment
                    self._drop_phase = "predrop_drive"
                    self._print(
                        f"Drop: pallet still on forks"
                        f" (before={dist_before:.2f}m  after={pallet_dist_after:.2f}m"
                        f"  needed≥{2.0 * dist_before:.2f}m)"
                        f" - retrying with extra lower={self._drop_lower_offset_extra:.3f}m"
                    )
                    return

                # Pallet confirmed dropped.
                self._reparent_pallet_after_drop()
                # Flush the proximity sensor's stale prim reference to the pallet
                # that was just moved to a new USD path by _reparent_pallet_after_drop.
                self._reinit_proximity_sensor()
                self._set_bool("navWithPallet", False)
                self._carried_pallet_root = ""
                self._drop_lower_offset_extra = 0.0
                self._drop_phase = "idle"
                self._set_bool("navGoToDrop", False)
                self._drop_nav_triggered = False
                root_prim = self._stage.GetPrimAtPath(self.root)
                if root_prim and root_prim.IsValid():
                    rel = root_prim.GetRelationship("navDropTransform")
                    if rel:
                        rel.SetTargets([])
                    # Clear the pallet pick ID so it doesn't re-trigger pick navigation.
                    attr = root_prim.GetAttribute("navPalletToPickId")
                    if attr and attr.IsValid():
                        attr.Set("")
                    # Clear drop area and route display
                    for a_name in ("navAreaToDrop", "navFinalRoute"):
                        a = root_prim.GetAttribute(a_name)
                        if a and a.IsValid():
                            a.Set("")
                self._print(f"Drop: complete  (pallet dist={pallet_dist_after:.2f}m)")

                # If no next pallet is queued, head home and publish the home route
                next_pallet_id = self._get_str("navPalletToPickId")
                if not next_pallet_id:
                    self._check_home_nav()
                    # _check_home_nav sets navClosestNode; compute path now so navFinalRoute is visible
                    if self._going_home:
                        self._aligning             = False
                        self._nav_path             = []
                        self._nav_idx              = 0
                        self._nav_reverse          = False
                        self._last_nav_idx_checked = -1
                        self._turning_at_node      = False
                        self._start_navigation()
                return

            self.drive(-self.align_speed, 0.0)

    def _reparent_pallet_after_drop(self) -> None:
        """Move the pallet's Docking scope to Racks, Buffer, or Docking after a successful drop.

        Only runs when _picking_from_dock is True (pallet originated from
        /World/Pallets/Docking).  The destination is determined by the matched attribute:
          - navBufferPosId matched                -> /World/Pallets/Buffer/<scope_name>
          - navRacksPosId matched, "Rack" in val  -> /World/Pallets/Racks/<scope_name>
          - navRacksPosId matched, "Dock" in val  -> /World/Pallets/Docking/<scope_name>
          - fallback                         -> /World/Pallets/Buffer/<scope_name>

        Uses omni.kit.commands.MovePrim so the operation is undo-able and the
        stage layer is updated correctly.
        """
        if not self._picking_from_dock or not self._pallet_path:
            return

        pallet_prim = self._stage.GetPrimAtPath(self._pallet_path)
        if not pallet_prim or not pallet_prim.IsValid():
            return

        # The pallet prim is the physics body (e.g. blockpallet_b02); the scope
        # one level up is the logical container (e.g. pallet_003) we want to move.
        parent_prim = pallet_prim.GetParent()
        if not parent_prim or not parent_prim.IsValid():
            parent_prim = pallet_prim

        parent_path = str(parent_prim.GetPath())
        if "/Pallets/Docking/" not in parent_path:
            return

        scope_name = parent_prim.GetName()

        if self._drop_area_attr_type == "navBufferPosId":
            dest_root = "/World/Pallets/Buffer"
        elif "Rack" in self._drop_nav_area:
            dest_root = "/World/Pallets/Racks"
        elif "Dock" in self._drop_nav_area:
            dest_root = "/World/Pallets/Docking"
        else:
            dest_root = "/World/Pallets/Buffer"

        dest_path = f"{dest_root}/{scope_name}"

        omni.kit.commands.execute("MovePrim", path_from=parent_path, path_to=dest_path)
        self._print(f"Pallet scope moved: {parent_path} -> {dest_path}")

        self._picking_from_dock   = False
        self._dock_path           = ""
        self._drop_nav_area       = ""
        self._drop_area_attr_type = ""

    # ── Drive helpers ──────────────────────────────────────────────────────

    def drive(self, speed: float, angle: float) -> None:
        """Set drive velocity and swivel angle.

        Args:
            speed: Angular velocity (deg/s). Negative = forward, positive = backward.
            angle: Swivel angle (degrees). Positive = left, negative = right.
        """
        self.set_drive_velocity(speed)
        self.set_swivel_angle(angle)

    def set_drive_velocity(self, velocity: float) -> None:
        prim = self._stage.GetPrimAtPath(self.drive_joint)
        if not prim or not prim.IsValid():
            return
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.GetTargetVelocityAttr().Set(float(velocity))

    def set_swivel_angle(self, angle_deg: float) -> None:
        prim = self._stage.GetPrimAtPath(self.swivel_joint)
        if not prim or not prim.IsValid():
            return
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.GetTargetPositionAttr().Set(float(angle_deg))

    def pivot_turn(self, degrees: float = 90.0, direction: int = 1) -> bool:
        """Turn approximately in place by `degrees`, then stop.

        Uses maximum steer angle and a very slow forward speed to minimise the
        turning radius (rear-wheel steering means a true zero-radius spin is not
        possible, hence "almost in place").

        Call this every frame from an `on_update` handler.  It is self-contained:
        the first call captures the start heading; subsequent calls check progress;
        it returns True and stops once the target angle has been swept.

        Args:
            degrees:   Magnitude of the rotation in degrees (always positive).
            direction: +1 = left (counter-clockwise), -1 = right (clockwise).

        Returns:
            True when the turn is complete; False while still turning.
        """
        if self._pivot_start_yaw is None:
            self._pivot_start_yaw = self._world_yaw(self.body_path)

        current_yaw = self._world_yaw(self.body_path)
        # Signed angular change in the chosen direction (+ve = making progress)
        turned_deg  = math.degrees(self._wrap(current_yaw - self._pivot_start_yaw)) * direction

        if turned_deg >= degrees:
            self.stop()
            self._pivot_start_yaw = None
            return True

        self.drive(self.pivot_speed, self.max_turn * direction)
        return False

    def _lift_lower_limit(self) -> float:
        """Return the lift joint's physics lower limit, falling back to pick_lift_min."""
        prim = self._stage.GetPrimAtPath(self.lift_joint)
        if prim and prim.IsValid():
            attr = prim.GetAttribute("physics:lowerLimit")
            if attr and attr.IsValid():
                val = attr.Get()
                if val is not None:
                    return float(val)
        return self.pick_lift_min

    def _lift_current_pos(self) -> float | None:
        """Read actual lift position from physics simulation state, or None if unavailable.

        Tries the direct USD state attribute first (written by PhysX each step),
        then falls back to PhysxJointStateAPI.
        """
        prim = self._stage.GetPrimAtPath(self.lift_joint)
        if not prim or not prim.IsValid():
            return None
        # Direct attribute - present in the USDA and updated by PhysX every step.
        attr = prim.GetAttribute("state:linear:physics:position")
        if attr and attr.IsValid():
            val = attr.Get()
            if val is not None:
                return float(val)
        # Fallback: PhysxJointStateAPI (may not be applied at runtime).
        try:
            state = PhysxSchema.PhysxJointStateAPI.Get(prim, "linear")
            if state:
                a = state.GetPositionAttr()
                if a and a.IsValid():
                    val = a.Get()
                    if val is not None:
                        return float(val)
        except Exception:
            pass
        return None

    def _lift_current_vel(self) -> float | None:
        """Read actual lift velocity from physics simulation state, or None if unavailable."""
        prim = self._stage.GetPrimAtPath(self.lift_joint)
        if not prim or not prim.IsValid():
            return None
        attr = prim.GetAttribute("state:linear:physics:velocity")
        if attr and attr.IsValid():
            val = attr.Get()
            if val is not None:
                return float(val)
        return None

    def set_lift_position(self, position: float) -> None:
        prim = self._stage.GetPrimAtPath(self.lift_joint)
        if not prim or not prim.IsValid():
            return
        drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        if drive:
            drive.GetTargetPositionAttr().Set(float(position))

    def stop(self) -> None:
        self.drive(0.0, 0.0)

    # ── Math / transform helpers ───────────────────────────────────────────

    def _world_pos(self, path: str) -> tuple[float, float, float]:
        prim = self._stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return (0.0, 0.0, 0.0)
        t = (
            UsdGeom.Xformable(prim)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .ExtractTranslation()
        )
        return (float(t[0]), float(t[1]), float(t[2]))

    def _world_yaw(self, path: str) -> float:
        """Yaw of the prim's local -X axis in world XY (shared convention for forklift and nodes)."""
        prim = self._stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return 0.0
        fwd = (
            UsdGeom.Xformable(prim)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .TransformDir(self._forward_local)
        )
        return math.atan2(float(fwd[1]), float(fwd[0]))

    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
