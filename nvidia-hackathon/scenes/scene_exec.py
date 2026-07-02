"""
FleetMind — warehouse scene + BUS-DRIVEN autonomy, loaded INTO the running Isaac Sim
streaming app via Kit's `--exec` hook (runheadless.sh owns the render loop).

This is the production source of truth for the deployed scene. It:

  build()          -> new stage, warehouse, physics + ground, 2 rigged forklifts,
                      6 pallets + payloads, and 3 staging-zone floor markers.
  start_autonomy() -> play the timeline + register a physics-step callback that drives
                      each forklift along the waypoint route the fleet bus hands it,
                      performs pick / drop (raise/lower fork + carry the pallet prim),
                      and streams pose / phase / carried-pallet telemetry back onto the bus.
  start_bridge()   -> run the FastAPI control bridge (SIM_BACKEND=isaac) in a daemon thread
                      IN THIS SAME PROCESS, so its IsaacNavBackend shares the bus singleton
                      with the controller. The agent + UI talk to :8080 exactly as they do
                      against the mock.

The forklift is a REAR-wheel drive + REAR-wheel steer vehicle. The controller empirically
calibrates the drive sign and steer sign at runtime (once per forklift), then closes a
heading loop toward each waypoint. Joint conventions (verified): back_wheel_drive negative
= forward, back_wheel_swivel positive = left, lift_joint higher = up.
"""

import math
import os
import sys
import threading

import numpy as np
import omni.usd
import omni.timeline
import omni.physx
from pxr import (
    Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics, PhysicsSchemaTools,
)
from isaacsim.core.utils.stage import add_reference_to_stage

try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:  # pragma: no cover
    from isaacsim.core.utils.nucleus import get_assets_root_path

try:
    from isaacsim.core.prims import SingleArticulation
except Exception:  # pragma: no cover
    from isaacsim.core.prims.single_articulation import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

try:
    import carb
    _log = carb.log_warn
except Exception:  # pragma: no cover
    _log = print

# Make the repo's `src/` importable inside Kit's Python (repo root = parent of scenes/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.sim.fleet_bus import bus  # noqa: E402

# ===========================================================================
# CONFIG  — must stay in sync with src/sim/isaac_nav_bridge.py
# ===========================================================================
FORKLIFTS = {                       # name -> (x, y, yaw_degrees)
    "AMR_1": (-6.0, -3.0,   0.0),
    "AMR_2": ( 6.0,  3.0, 180.0),
}
PALLETS = [                         # (x, y) pallet-grid positions -> Pallet_00..05
    (-3.0, -3.0), (-3.0, 0.0), (-3.0, 3.0),
    ( 3.0, -3.0), ( 3.0, 0.0), ( 3.0, 3.0),
]
ZONES = {                           # staging bays (match isaac_nav_bridge.ZONES)
    "stage_1": (-6.0, 7.0),
    "stage_2": ( 0.0, 7.0),
    "stage_3": ( 6.0, 7.0),
}
PALLET_TOP_Z = 0.14
PAYLOAD_DROP = 0.06

# Cinematic overview camera — frames the whole floor (forklifts ±6, pallets ±3,
# staging at y=7). Sits INSIDE the warehouse footprint (eye within the ±6 x-range so
# it never clips the outer walls) at ~7 m — above the 3 m forklifts, below the roof —
# looking down-and-forward across the operating area.
CAMERA_PATH   = "/World/FleetCam"
CAMERA_EYE    = (4.0, -8.5, 7.0)
CAMERA_TARGET = (0.0, 2.0, 0.4)

FORKLIFT_SUBPATH = "/Isaac/Samples/Rigging/Forklift/forklift_b_rigged_cm.usd"
SPAWN_Z = 0.03
GROUND_STATIC_FRICTION  = 0.9
GROUND_DYNAMIC_FRICTION = 0.8

# Controller gains / limits
DRIVE_SPEED   = 2.5                 # back_wheel_drive target (rad/s)
# Calibration drives the truck a SHORT way to learn its drive/steer signs. It MUST be
# gentle: at full DRIVE_SPEED with a big steer probe a 3 m forklift swings ~3 m and can
# ram a wall/rack before it ever gets a mission (its "forward" axis is ~90° off its yaw,
# so "drive forward" at spawn heads sideways). Low speed + a short steer arc keeps it in
# open floor while still giving a clean displacement to read the signs from.
CALIB_SPEED   = 0.9                 # rad/s during calibration (gentle)
MAX_STEER     = math.radians(40.0)
STEER_PROBE   = math.radians(25.0)
K_STEER       = 1.2
WAYPOINT_DIST = 0.6                 # m, advance to next waypoint within this
ARRIVE_DIST   = 0.45                # m, leg goal reached
PICK_ARRIVE   = 0.30                # m, TIGHT final arrival for pick/drop precision
PICK_SNAP     = 1.10                # m, max dist to target for the fork to (dis)engage
LIFT_RAISE    = 0.15
LIFT_RATE     = 0.01                # m/step fork travel
WARMUP_STEPS  = 5
SETTLE_STEPS  = 90
CALIB_DRIVE_STEPS = 40
CALIB_STEER_STEPS = 30
ACT_STEPS     = 30                  # steps to hold during pick/drop lift
# Stall / anti-wedge watchdog. CRITICAL: `moved` is metres per PHYSICS STEP, and the
# truck tops out around 0.01 m/step, so the old 0.05 threshold treated EVERY step
# (even at full speed) as stalled -> the watchdog fired constantly and skipped
# waypoints, corrupting missions (false picks, driving to the wrong place). 0.002
# m/step (~0.12 m/s) flags only a genuine wedge.
STALL_SPEED   = 0.002
STALL_STEPS   = 120                 # ~2 s of no real motion -> attempt a recovery
RECOVER_STEPS = 45                  # reverse-nudge duration to unwedge from a stall
MAX_RECOVER   = 3                   # recovery attempts on one waypoint before skipping
LOG_EVERY     = 120
# ===========================================================================

_RT = {}   # runtime state (keeps physics-step subscription & controllers alive)
_BUS = bus()

# WH_Palette_0N (bus id) <-> /World/Pallets/Pallet_0(N-1) (USD prim)
_PALLET_PATH = {f"WH_Palette_{i + 1:02d}": f"/World/Pallets/Pallet_{i:02d}"
                for i in range(len(PALLETS))}


def _set_pose(stage, prim_path, xyz, yaw_deg=0.0, scale=1.0):
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xform.AddRotateZOp().Set(float(yaw_deg))
    if scale != 1.0:
        xform.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))


def _move_prim_xy(stage, prim_path, x, y, z):
    """Cheap translate-only pose update for a carried pallet (kinematic follow)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(x), float(y), float(z)))
            return
    xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))


def _bbox_size_x(stage, prim_path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        return 0.0
    return float(rng.GetMax()[0] - rng.GetMin()[0])


def _yaw_from_quat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _configure_drives(stage, base_path):
    """Override the rigged asset's shipped drive params (it ships a -200 rad/s wheel
    target from the rigging tutorial, which flings the forklift at spawn)."""
    drive_j = stage.GetPrimAtPath(f"{base_path}/back_wheel_joints/back_wheel_drive")
    steer_j = stage.GetPrimAtPath(f"{base_path}/back_wheel_joints/back_wheel_swivel")
    lift_j  = stage.GetPrimAtPath(f"{base_path}/lift_joint/lift_joint")

    if drive_j and drive_j.IsValid():
        d = UsdPhysics.DriveAPI.Get(drive_j, "angular") or \
            UsdPhysics.DriveAPI.Apply(drive_j, "angular")
        d.CreateStiffnessAttr(0.0)
        d.CreateDampingAttr(1.0e5)
        d.CreateTargetVelocityAttr(0.0)
        d.CreateMaxForceAttr(1.0e7)

    if steer_j and steer_j.IsValid():
        s = UsdPhysics.DriveAPI.Get(steer_j, "angular") or \
            UsdPhysics.DriveAPI.Apply(steer_j, "angular")
        s.CreateStiffnessAttr(1.0e6)
        s.CreateDampingAttr(1.0e5)
        s.CreateTargetPositionAttr(0.0)
        s.CreateMaxForceAttr(1.0e7)

    if lift_j and lift_j.IsValid():
        l = UsdPhysics.DriveAPI.Get(lift_j, "linear") or \
            UsdPhysics.DriveAPI.Apply(lift_j, "linear")
        l.CreateStiffnessAttr(1.0e6)
        l.CreateDampingAttr(1.0e5)
        l.CreateTargetPositionAttr(0.0)
        l.CreateMaxForceAttr(1.0e7)


def _zone_marker(stage, zid, x, y):
    """A thin coloured floor disc so staging bays are visible in the stream."""
    path = f"/World/Zones/{zid}"
    disc = UsdGeom.Cylinder.Define(stage, path)
    disc.CreateRadiusAttr(0.9)
    disc.CreateHeightAttr(0.02)
    disc.CreateAxisAttr("Z")
    disc.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.6, 0.2)])
    _set_pose(stage, path, (x, y, 0.01))


def _look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera local-to-world transform for a camera at `eye` looking at `target`.
    USD cameras look down their local -Z with +Y up."""
    eye = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    fwd = (target - eye).GetNormalized()
    z_axis = -fwd
    x_axis = Gf.Cross(Gf.Vec3d(*up), z_axis).GetNormalized()
    y_axis = Gf.Cross(z_axis, x_axis)
    return Gf.Matrix4d(
        x_axis[0], x_axis[1], x_axis[2], 0.0,
        y_axis[0], y_axis[1], y_axis[2], 0.0,
        z_axis[0], z_axis[1], z_axis[2], 0.0,
        eye[0],    eye[1],    eye[2],    1.0,
    )


def _setup_camera(stage):
    """A cinematic 3/4 overview camera framing the whole warehouse floor."""
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.CreateFocalLengthAttr(18.0)                     # wide-ish so the floor fits
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    cam.CreateFocusDistanceAttr(22.0)
    xform = UsdGeom.Xformable(cam)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(_look_at_matrix(CAMERA_EYE, CAMERA_TARGET))
    _log(f"[FleetMind] FleetCam created at {CAMERA_EYE} -> {CAMERA_TARGET}.")


def _apply_view():
    """Point the streamed viewport at FleetCam and strip the editor chrome so the
    stream shows ONLY the 3D simulation. All best-effort (wrapped per step)."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        if vp is not None:
            vp.camera_path = CAMERA_PATH
            _log("[FleetMind] Viewport camera set to FleetCam.")
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] set active camera skipped: {e}")

    # Hide editor panels (Stage, Property, Console, Content, toolbars, layers…).
    try:
        import omni.ui as ui
        keep = {"Viewport", "Viewport 1", "Viewport 2", "DockSpace"}
        for w in ui.Workspace.get_windows():
            title = getattr(w, "title", "") or ""
            if title not in keep:
                try:
                    w.visible = False
                except Exception:
                    pass
        _log("[FleetMind] Editor panels hidden (viewport-only).")
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] hide panels skipped: {e}")

    # Hide the top menu bar for a fully clean frame.
    try:
        from omni.kit.mainwindow import get_main_window
        mw = get_main_window()
        for getter in ("get_main_menu_bar_widget", "get_menu_bar_widget"):
            fn = getattr(mw, getter, None)
            if fn:
                try:
                    fn().visible = False
                except Exception:
                    pass
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] hide menubar skipped: {e}")


def _schedule_apply_view(delay_frames=90):
    """Run _apply_view after the streaming app's UI is fully up, then unsubscribe."""
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        st = {"n": 0, "sub": None}

        def _cb(_e):
            st["n"] += 1
            if st["n"] >= delay_frames:
                _apply_view()
                if st["sub"] is not None:
                    st["sub"].unsubscribe()
                    st["sub"] = None

        st["sub"] = app.get_update_event_stream().create_subscription_to_pop(
            _cb, name="fleetmind-apply-view")
        _log("[FleetMind] View setup scheduled.")
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] schedule view skipped ({e}); applying immediately.")
        _apply_view()


def build():
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac assets root path")

    warehouse = assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    forklift  = assets_root + FORKLIFT_SUBPATH
    pallet    = assets_root + "/Isaac/Props/Pallet/pallet.usd"
    payloads  = [
        assets_root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
        assets_root + "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
        assets_root + "/Isaac/Props/KLT_Bin/small_KLT.usd",
    ]

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 200.0,
        Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.35),
    )
    mat_path = "/World/PhysicsMaterials/GroundMaterial"
    UsdShade.Material.Define(stage, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    pmat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pmat.CreateStaticFrictionAttr(GROUND_STATIC_FRICTION)
    pmat.CreateDynamicFrictionAttr(GROUND_DYNAMIC_FRICTION)
    pmat.CreateRestitutionAttr(0.0)
    ground_prim = stage.GetPrimAtPath("/World/GroundPlane")
    binding = UsdShade.MaterialBindingAPI.Apply(ground_prim)
    binding.Bind(
        UsdShade.Material(mat_prim),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    add_reference_to_stage(usd_path=warehouse, prim_path="/World/Warehouse")

    add_reference_to_stage(usd_path=forklift, prim_path="/World/_probe")
    size_x = _bbox_size_x(stage, "/World/_probe")
    fork_scale = 0.01 if size_x > 50.0 else 1.0
    stage.RemovePrim("/World/_probe")
    _log(f"[FleetMind] Forklift probe size_x={size_x:.2f} -> scale={fork_scale}")

    UsdGeom.Xform.Define(stage, "/World/AMRs")
    forklift_paths = {}
    for name, (x, y, yaw) in FORKLIFTS.items():
        p = f"/World/AMRs/{name}"
        add_reference_to_stage(usd_path=forklift, prim_path=p)
        _set_pose(stage, p, (x, y, SPAWN_Z), yaw, scale=fork_scale)
        _configure_drives(stage, p)
        forklift_paths[name] = p
        _BUS.register_forklift(name, x, y, math.radians(yaw))
        _log(f"[FleetMind] Placed {name} at ({x}, {y}, yaw={yaw})")

    UsdGeom.Xform.Define(stage, "/World/Pallets")
    UsdGeom.Xform.Define(stage, "/World/Payloads")
    for i, (x, y) in enumerate(PALLETS):
        pp = f"/World/Pallets/Pallet_{i:02d}"
        add_reference_to_stage(usd_path=pallet, prim_path=pp)
        _set_pose(stage, pp, (x, y, 0.0), 0.0)
        yp = f"/World/Payloads/Payload_{i:02d}"
        add_reference_to_stage(usd_path=payloads[i % len(payloads)], prim_path=yp)
        _set_pose(stage, yp, (x, y, PALLET_TOP_Z + PAYLOAD_DROP), 0.0)
        _BUS.set_pallet(f"WH_Palette_{i + 1:02d}", x=x, y=y, carried_by=None, delivered=False)

    UsdGeom.Xform.Define(stage, "/World/Zones")
    for zid, (x, y) in ZONES.items():
        _zone_marker(stage, zid, x, y)
        _BUS.set_zone(zid, x=x, y=y, blocked=False)

    _setup_camera(stage)

    _log(f"[FleetMind] Scene built: {len(FORKLIFTS)} forklifts, {len(PALLETS)} pallets, "
         f"{len(ZONES)} staging zones.")
    return forklift_paths


def _apply_cmd(art, idx, wheel_vel, steer_ang, lift_pos):
    art.apply_action(ArticulationAction(
        joint_velocities=np.array([wheel_vel]),
        joint_indices=np.array([idx["drive"]]),
    ))
    art.apply_action(ArticulationAction(
        joint_positions=np.array([steer_ang, lift_pos]),
        joint_indices=np.array([idx["steer"], idx["lift"]]),
    ))


def _init_controllers():
    ctrls = {}
    for name, path in _RT["paths"].items():
        try:
            art = SingleArticulation(prim_path=path, name=name)
            art.initialize()
            names = list(art.dof_names)
            idx = {
                "drive": names.index("back_wheel_drive"),
                "steer": names.index("back_wheel_swivel"),
                "lift":  names.index("lift_joint"),
            }
            _apply_cmd(art, idx, 0.0, 0.0, 0.0)
            pos0, quat0 = art.get_world_pose()
            ctrls[name] = {
                "art": art, "idx": idx,
                "phase": "settle", "k": 0,
                # Forward = DRIVE_SPEED*drive_dir. drive_dir is FIXED to the documented
                # convention (back_wheel_drive negative = forward) rather than detected by
                # projecting the calibration move onto the yaw axis: this rig's forward
                # axis is ~90° off its reported yaw, so that projection is ~0 and its SIGN
                # is noise — a wrong guess flipped head_off 180° and drove the truck into a
                # wall. head_off (measured directly below) still captures the true travel
                # direction, so navigation is correct regardless of axis convention.
                "drive_dir": -1.0, "steer_sign": 1.0, "head_off": 0.0,
                "p_ref": None, "yaw_ref": None,
                "last_xy": None, "stall": 0,
                "recover": 0, "recover_steer": 1.0, "recover_tries": 0,
                "lift_down": 0.0, "lift_up": LIFT_RAISE, "lift": 0.0,
                # mission execution
                "seq": -1, "legs": [], "leg_i": 0, "wp_i": 0,
                "carrying": None, "carry_path": None,
            }
            _log(f"[FleetMind] {name} articulation ready | dofs={len(names)} "
                 f"| start=({float(pos0[0]):.2f},{float(pos0[1]):.2f})")
        except Exception as e:
            _log(f"[FleetMind] ERROR init {name}: {e}")
    _RT["ctrls"] = ctrls


def _pull_command(name, c):
    """Adopt a newer mission from the bus (resets leg/waypoint cursors)."""
    cmd = _BUS.get_command(name)
    if cmd.seq != c["seq"] and cmd.legs:
        c["seq"] = cmd.seq
        c["legs"] = cmd.legs
        c["leg_i"] = 0
        c["wp_i"] = 0
        c["stall"] = 0
        c["recover"] = 0
        c["recover_tries"] = 0
        c["phase"] = "navigate"


def _phase_label(c):
    if c["phase"] in ("settle", "calib_drive", "calib_steer"):
        return "idle"
    if c["phase"] == "navigate":
        return "carrying" if c["carrying"] else "navigating"
    if c["phase"] == "act":
        leg = c["legs"][c["leg_i"]]
        return "lifting" if leg.action in ("pick", "drop") else "navigating"
    if c["phase"] == "return":
        return "returning"
    return "idle"


def _publish(name, c, x, y, yaw, speed, target, route):
    _BUS.update_telemetry(
        name, x=x, y=y, yaw=yaw, speed=speed,
        phase=_phase_label(c), carrying=c["carrying"], lift_height=c["lift"],
        target=target, route=route, goal_kind=None,
        object_detected="None", object_distance=0.0,
        path_blocked=(c["stall"] > 0 or c.get("recover", 0) > 0),
    )


def _step_one(name, c):
    art, idx = c["art"], c["idx"]
    pos, quat = art.get_world_pose()
    x, y = float(pos[0]), float(pos[1])
    yaw = _yaw_from_quat(quat)
    c["k"] += 1
    speed = 0.0
    if c["last_xy"] is not None:
        speed = math.hypot(x - c["last_xy"][0], y - c["last_xy"][1]) * 60.0

    # ---- one-time calibration ------------------------------------------- #
    if c["phase"] == "settle":
        _apply_cmd(art, idx, 0.0, 0.0, c["lift_down"])
        if c["k"] >= SETTLE_STEPS:
            c["phase"], c["k"] = "calib_drive", 0
            c["p_ref"], c["yaw_ref"] = (x, y), yaw
        _publish(name, c, x, y, yaw, 0.0, None, [])
        c["last_xy"] = (x, y)
        return x, y, yaw

    if c["phase"] == "calib_drive":
        # Drive the FORWARD command (fixed sign) a short way and measure the actual
        # world-space travel direction relative to yaw. head=yaw+head_off then tracks
        # the truck's true motion for ANY axis convention — no projection sign-guessing.
        _apply_cmd(art, idx, CALIB_SPEED * c["drive_dir"], 0.0, c["lift_down"])
        if c["k"] >= CALIB_DRIVE_STEPS:
            dx, dy = x - c["p_ref"][0], y - c["p_ref"][1]
            if math.hypot(dx, dy) > 0.05:
                c["head_off"] = _wrap(math.atan2(dy, dx) - c["yaw_ref"])  # travel_dir - yaw
            c["phase"], c["k"], c["yaw_ref"] = "calib_steer", 0, yaw
        _publish(name, c, x, y, yaw, speed, None, [])
        c["last_xy"] = (x, y)
        return x, y, yaw

    if c["phase"] == "calib_steer":
        _apply_cmd(art, idx, CALIB_SPEED * c["drive_dir"], STEER_PROBE, c["lift_down"])
        if c["k"] >= CALIB_STEER_STEPS:
            dyaw = _wrap(yaw - c["yaw_ref"])
            c["steer_sign"] = 1.0 if dyaw >= 0.0 else -1.0
            c["phase"], c["k"] = "idle", 0
        _publish(name, c, x, y, yaw, speed, None, [])
        c["last_xy"] = (x, y)
        return x, y, yaw

    # ---- idle: hold + carry, wait for a mission ------------------------- #
    if c["phase"] == "idle":
        _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
        _pull_command(name, c)
        _publish(name, c, x, y, yaw, 0.0, None, [])
        c["last_xy"] = (x, y)
        return x, y, yaw

    # A newer mission always preempts.
    _pull_command(name, c)
    leg = c["legs"][c["leg_i"]]

    # ---- navigate the current leg's waypoints --------------------------- #
    if c["phase"] == "navigate":
        wps = leg.waypoints
        if c["wp_i"] >= len(wps):
            c["phase"], c["k"] = "act", 0
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
        elif c["recover"] > 0:
            # Unwedge: reverse along the opposite of travel with counter-steer, then
            # retry the SAME waypoint. This never advances the mission while stuck, so
            # a wall/corner jam can't corrupt the leg (no false arrival / false pick).
            c["recover"] -= 1
            _apply_cmd(art, idx, -DRIVE_SPEED * c["drive_dir"],
                       c["steer_sign"] * MAX_STEER * c["recover_steer"], c["lift"])
            _carry_follow(c, x, y)
            route = [leg.target] if leg.target else []
            _publish(name, c, x, y, yaw, speed, leg.target, route)
            c["last_xy"] = (x, y)
            return x, y, yaw
        else:
            tx, ty = wps[c["wp_i"]]
            dist = math.hypot(tx - x, ty - y)
            is_last = (c["wp_i"] == len(wps) - 1)
            if is_last:
                reach = PICK_ARRIVE if leg.action in ("pick", "drop") else ARRIVE_DIST
            else:
                reach = WAYPOINT_DIST
            # stall watchdog (moved is per physics step; STALL_SPEED is tiny on purpose)
            if c["last_xy"] is not None:
                moved = math.hypot(x - c["last_xy"][0], y - c["last_xy"][1])
                c["stall"] = c["stall"] + 1 if moved < STALL_SPEED else 0
            if dist < reach:
                c["wp_i"] += 1
                c["stall"] = 0
                c["recover_tries"] = 0
            elif c["stall"] >= STALL_STEPS:
                # Genuinely wedged at this waypoint. Prefer a bounded reverse-recovery
                # nudge over blindly skipping (a skip can burn the whole leg and let the
                # fork engage off-target). Only after MAX_RECOVER failed nudges do we
                # skip, so a truly unreachable waypoint still can't deadlock the mission.
                if c["recover_tries"] < MAX_RECOVER:
                    c["recover"] = RECOVER_STEPS
                    c["recover_steer"] = -c["recover_steer"]   # alternate each attempt
                    c["recover_tries"] += 1
                else:
                    c["wp_i"] += 1
                    c["recover_tries"] = 0
                c["stall"] = 0
            else:
                desired = math.atan2(ty - y, tx - x)
                head = _wrap(yaw + c["head_off"])   # empirical travel direction
                herr = _wrap(desired - head)
                steer = c["steer_sign"] * K_STEER * herr
                steer = max(-MAX_STEER, min(MAX_STEER, steer))
                # Constant travel speed — as in the proven single-mission controller.
                # A rear-wheel-steer truck needs forward momentum to change heading, and
                # slowing on the run-in let static friction lock it up mid-leg. Precision
                # comes from the tight PICK_ARRIVE gate + the dense straight run-in
                # waypoints, not from creeping.
                _apply_cmd(art, idx, DRIVE_SPEED * c["drive_dir"], steer, c["lift"])
        route = [leg.target] if leg.target else []
        _publish(name, c, x, y, yaw, speed, leg.target, route)
        # keep a carried pallet glued to the forks
        _carry_follow(c, x, y)
        c["last_xy"] = (x, y)
        return x, y, yaw

    # ---- act: perform the leg's terminal action ------------------------- #
    if c["phase"] == "act":
        # How close are we to where this leg was supposed to end? A stall-watchdog
        # skip can burn through waypoints without the truck actually arriving, so we
        # only let the fork engage/disengage when genuinely on top of the target —
        # otherwise the pallet would teleport onto a stuck truck / be dropped mid-aisle.
        gx, gy = leg.waypoints[-1] if leg.waypoints else (x, y)
        at_target = math.hypot(gx - x, gy - y) <= PICK_SNAP
        if leg.action == "pick":
            c["lift"] = min(c["lift_up"], c["lift"] + LIFT_RATE)
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
            if at_target and c["lift"] >= c["lift_up"] - 1e-3 and c["carrying"] is None:
                c["carrying"] = leg.target
                c["carry_path"] = leg.pallet_path or _PALLET_PATH.get(leg.target)
                _BUS.set_pallet(leg.target, carried_by=name)
            if c["k"] >= ACT_STEPS:
                _advance_leg(c)
        elif leg.action == "drop":
            c["lift"] = max(c["lift_down"], c["lift"] - LIFT_RATE)
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
            if at_target and c["lift"] <= c["lift_down"] + 1e-3 and c["carrying"] is not None:
                dx, dy = leg.drop_xy if leg.drop_xy else (x, y)
                if c["carry_path"]:
                    _move_prim_xy(_RT["stage"], c["carry_path"], dx, dy, 0.0)
                _BUS.set_pallet(c["carrying"], x=dx, y=dy, carried_by=None, delivered=True)
                c["carrying"], c["carry_path"] = None, None
            if c["k"] >= ACT_STEPS:
                _advance_leg(c)
        else:  # goto / home — nothing to actuate
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
            _advance_leg(c)
        _publish(name, c, x, y, yaw, 0.0, leg.target, [leg.target] if leg.target else [])
        c["last_xy"] = (x, y)
        return x, y, yaw

    _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
    _publish(name, c, x, y, yaw, 0.0, None, [])
    c["last_xy"] = (x, y)
    return x, y, yaw


def _advance_leg(c):
    c["leg_i"] += 1
    c["wp_i"] = 0
    c["k"] = 0
    c["stall"] = 0
    c["recover"] = 0
    c["recover_tries"] = 0
    if c["leg_i"] >= len(c["legs"]):
        c["phase"] = "idle"
        c["legs"] = []
        c["leg_i"] = 0
    else:
        c["phase"] = "navigate"


def _carry_follow(c, x, y):
    if c["carrying"] and c["carry_path"]:
        _move_prim_xy(_RT["stage"], c["carry_path"], x, y, PALLET_TOP_Z + c["lift"])


def _on_step(dt):
    _RT["warm"] += 1
    if _RT["warm"] < WARMUP_STEPS:
        return
    if "ctrls" not in _RT:
        _init_controllers()
        return
    _RT["n"] += 1
    do_log = (_RT["n"] % LOG_EVERY == 0)
    for name, c in _RT["ctrls"].items():
        try:
            x, y, yaw = _step_one(name, c)
            if do_log:
                _log(f"[FleetMind] {name} [{_phase_label(c)}] "
                     f"pos=({x:+.2f},{y:+.2f}) yaw={math.degrees(yaw):+.0f} "
                     f"carry={c['carrying']} leg={c['leg_i']}/{len(c['legs'])} "
                     f"wp={c['wp_i']}")
        except Exception as e:
            if do_log:
                _log(f"[FleetMind] {name} step error: {e}")


def start_autonomy(forklift_paths):
    _RT["paths"] = forklift_paths
    _RT["stage"] = omni.usd.get_context().get_stage()
    _RT["warm"] = 0
    _RT["n"] = 0
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(False)
    timeline.play()
    physx = omni.physx.get_physx_interface()
    _RT["sub"] = physx.subscribe_physics_step_events(_on_step)
    _log("[FleetMind] Autonomy started: timeline playing, controller subscribed.")


def start_bridge():
    """Run the control bridge (SIM_BACKEND=isaac) in a daemon thread, in-process, so its
    IsaacNavBackend shares the fleet-bus singleton with this controller."""
    os.environ["SIM_BACKEND"] = "isaac"

    def _serve():
        try:
            import uvicorn
            from src.ros2.bridge_server import app
            _log("[FleetMind] Bridge starting on :8080 (SIM_BACKEND=isaac)")
            uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
        except Exception as e:
            _log(f"[FleetMind] Bridge NOT started ({e}). "
                 f"pip install fastapi uvicorn into Isaac's python to enable it, "
                 f"or run the bridge as a separate SIM_BACKEND=isaac process.")

    threading.Thread(target=_serve, name="fleetmind-bridge", daemon=True).start()


paths = build()
start_autonomy(paths)
start_bridge()
_schedule_apply_view()
