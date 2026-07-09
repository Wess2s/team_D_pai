"""
Isaac Sim backend — the DGX-side counterpart of the offline WarehouseSim.

It exposes the *same* Python interface (snapshot / go_to / pick / drop / go_home / mission /
block_zone) as the mock, so the HTTP bridge, agent and UI are byte-for-byte identical whether
we run the mock or the real scene. Selected with `SIM_BACKEND=isaac`.

Architecture (the real deployed scene has NO waypoint graph, NO staging zones and NO nav*
behaviour script — it is a self-contained 2-forklift AMR scene, see `scenes/scene_exec.py`):

    bridge  ── writes missions (waypoint legs + pick/drop) ──►  fleet_bus  ──►  scene_exec
    bridge  ◄── reads pose / phase / carried pallet telemetry ──  fleet_bus  ◄──  scene_exec

Both the bridge (this module, running inside the Kit process) and the physics-step controller
in `scene_exec.py` share the in-process `fleet_bus` singleton. This module owns the *planning*
surface: it defines the warehouse layout the deployed scene lacks (a routing roadmap + staging
zones), turns each command into an A* waypoint route, and pushes it to the bus. The controller
just follows waypoints and lifts/lowers the fork.

The bridge never imports omni here — all stage manipulation lives in the controller — so this
module imports cleanly off-DGX for linting and the offline smoke test.
"""
from __future__ import annotations

import math

from .fleet_bus import Leg, bus
from ..agent.planning.roadmap import Roadmap


# --------------------------------------------------------------------------- #
# Warehouse layout — mirrors the deployed scene_exec.py constants, plus the
# staging zones + roadmap the raw scene doesn't define.
# --------------------------------------------------------------------------- #
# Forklift spawn poses (name -> (x, y, yaw_deg)); live poses come from telemetry.
FORKLIFTS = {
    "AMR_1": (-6.0, -3.0, 0.0),
    "AMR_2": (6.0, 3.0, 180.0),
}
# Three pallets on the rack grid (reduced from six to keep the aisles clear for
# two-forklift navigation). Exposed to the agent/UI under mock-compatible ids
# (WH_Palette_01..03) mapped to the scene's USD prims (Pallet_00..02) so nothing
# above the bridge changes. Spread left/right so each forklift has non-crossing work.
_PALLET_GRID = [
    (-3.0, -3.0), (-3.0, 3.0), (1.0, 3.0),
]
PALLETS = {f"WH_Palette_{i + 1:02d}": {"xy": xy, "path": f"/World/Pallets/Pallet_{i:02d}"}
           for i, xy in enumerate(_PALLET_GRID)}
# Staging bays the scene lacks — placed along the far (+y) aisle, clear of the racks.
ZONES = {
    "stage_1": (-6.0, 7.0),
    "stage_2": (0.0, 7.0),
    "stage_3": (6.0, 7.0),
}
# Charging docks — each forklift's home is its charger (mirrors scene_exec.CHARGERS), so
# the ops-map can draw the same blue pads the 3D scene shows.
CHARGERS = {
    "charge_1": (-6.0, -3.0),
    "charge_2": (6.0, 3.0),
}

# Densify every route into short, evenly-spaced waypoints so the truck tracks the
# planned line tightly (instead of arcing between far-apart grid nodes) and gets a
# straight, precise final run-in onto the pallet/zone. Because a pallet's own cell is
# blocked in the roadmap, A* arrives from the adjacent node facing the truck, so this
# final leg is already square-on — no forced aisle detour needed. STEP matches the
# controller's WAYPOINT_DIST so it advances smoothly point-to-point.
ROUTE_STEP = 0.6                         # m between densified waypoints
# Standoff for pick/drop: stop slightly OUTSIDE fork reach so normal-speed navigation
# does not drive the tines into the pallet. The scene controller then performs the last
# few centimetres as a slow insertion creep, which avoids pushing the pallet sideways.
APPROACH_LEN = 1.75
# Pickups must enter through the pallet's wider fork slots, which run north/south in the
# map. Route to a point several metres above/below the pallet first, then drive a straight
# vertical runway into the standoff point so the truck is square before the fork inserts.
PICK_RUNUP_LEN = 3.0


class IsaacNavBackend:
    """Adapter over the live Isaac AMR scene, matching WarehouseSim's interface."""

    def __init__(self) -> None:
        self.bus = bus()
        self.hazards: dict[str, dict] = {}   # zone id -> {kind,x,y,t,radius} incident

        # Register forklifts + pallets + zones on the shared bus.
        for name, (x, y, yaw) in FORKLIFTS.items():
            self.bus.register_forklift(name, x, y, math.radians(yaw))
        for pid, meta in PALLETS.items():
            self.bus.set_pallet(pid, x=meta["xy"][0], y=meta["xy"][1],
                                carried_by=None, delivered=False)
        for zid, (zx, zy) in ZONES.items():
            self.bus.set_zone(zid, x=zx, y=zy, blocked=False)

        # Build a routing roadmap over the floor (the deployed scene has no graph).
        # A fixed 20×20 uniform mesh spans the whole floor: fine, evenly-spaced waypoints
        # give the trucks a smoother, more legible trajectory than the old coarse 1 m grid
        # and populate the ops-map with a dense 20×20 marker field. Resting-pallet cells are
        # punched out so A* naturally arcs around loaded rack faces; the bridge still appends
        # the exact goal + a standoff run-in, so nodes needn't line up on pallets exactly.
        self._rm = Roadmap.from_snapshot(self._layout_snapshot(), grid=(20, 20), margin=2.0)
        self.bus.graph = {
            "nodes": {n: list(xy) for n, xy in self._rm.nodes.items()},
            "edges": sorted({tuple(sorted((a, b)))
                             for a, nbrs in self._rm.edges.items() for b in nbrs}),
        }

    # ---- layout / routing helpers --------------------------------------- #
    def _layout_snapshot(self) -> dict:
        """A minimal snapshot (no graph) so Roadmap synthesises a grid around everything."""
        return {
            "forklifts": {n: {"x": x, "y": y} for n, (x, y, _) in FORKLIFTS.items()},
            "pallets": {pid: {"x": m["xy"][0], "y": m["xy"][1], "delivered": False,
                              "carried_by": None} for pid, m in PALLETS.items()},
            "zones": {z: {"x": x, "y": y} for z, (x, y) in ZONES.items()},
        }

    def _fk_pos(self, name: str) -> tuple[float, float]:
        t = self.bus.get_telemetry(name)
        return (t.x, t.y)

    # Radius (m) around another forklift whose roadmap cells are treated as blocked while
    # routing, so a dispatched truck arcs clear of a parked one instead of driving through
    # it. ~1.8 m ≈ the ~2.5 m truck body's half-width plus a safety margin.
    AVOID_RADIUS = 1.8

    def _blocked_by_others(self, name: str) -> set[str]:
        """Roadmap nodes to avoid because another (non-`name`) forklift occupies them.

        This is the geometric half of deconfliction: CBS reasons over the node graph, and
        here we mirror that by punching out each parked truck's footprint so the routed
        world-space path (and its floor overlay) bends around it rather than clipping it.
        """
        blocked: set[str] = set()
        for other in FORKLIFTS:
            if other == name:
                continue
            ox, oy = self._fk_pos(other)
            blocked |= self._rm.nodes_within(ox, oy, self.AVOID_RADIUS)
        return blocked

    def _route_xy(self, start_xy: tuple[float, float],
                  goal_xy: tuple[float, float],
                  avoid: set[str] | None = None) -> list[tuple[float, float]]:
        """Turn-minimising A* over the roadmap, returned as world-space waypoints ending
        exactly on the goal. We plan with `astar_straight` (long straight runs, few
        corners — better for a forklift and clearer as a floor overlay) and then collapse
        collinear nodes so each straight aisle is a single segment before densifying.
        `avoid` blocks nodes under other forklifts so the route steers around them."""
        a = self._rm.nearest(*start_xy)
        b = self._rm.nearest(*goal_xy)
        path = self._rm.astar_straight(a, b, avoid=avoid)
        wpts = [self._rm.nodes[n] for n in path]
        wpts.append(goal_xy)
        wpts = self._rm.collapse_collinear([tuple(p) for p in wpts])
        return wpts

    @staticmethod
    def _densify(wpts: list[tuple[float, float]],
                 step: float = ROUTE_STEP) -> list[tuple[float, float]]:
        """Interpolate so no two consecutive waypoints are more than `step` apart."""
        if len(wpts) < 2:
            return list(wpts)
        out = [wpts[0]]
        for (ax, ay), (bx, by) in zip(wpts, wpts[1:]):
            seg = math.hypot(bx - ax, by - ay)
            n = max(1, int(math.ceil(seg / step)))
            for i in range(1, n + 1):
                f = i / n
                out.append((ax + (bx - ax) * f, ay + (by - ay) * f))
        return out

    def _route_fine(self, start_xy: tuple[float, float],
                    goal_xy: tuple[float, float],
                    avoid: set[str] | None = None) -> list[tuple[float, float]]:
        """A* to the target, densified into short steps for tight tracking and a
        straight, precise final run-in onto the pallet/zone."""
        return self._densify(self._route_xy(start_xy, goal_xy, avoid=avoid))

    def _route_approach(self, start_xy: tuple[float, float],
                        goal_xy: tuple[float, float],
                        approach: float = APPROACH_LEN,
                        avoid: set[str] | None = None) -> list[tuple[float, float]]:
        """Route toward a pick/drop target but STOP ~`approach` m short of it, keeping
        the inward-facing heading. A forklift is a ~2.5 m body: driving its CENTRE onto
        the pallet cell rams the pallet/rack and wedges the truck ~1.3 m out (observed).
        Instead we trim the final run-in so the truck halts just in front, forks reaching
        under the load, then the controller engages the fork there. Heading is preserved
        because we truncate ALONG the planned inward path (never aim the body at the
        obstacle). `avoid` steers the approach clear of other forklifts."""
        full = self._densify(self._route_xy(start_xy, goal_xy, avoid=avoid))
        if len(full) < 2:
            return full
        tx, ty = goal_xy
        out = list(full)
        # Drop trailing points that sit inside the standoff radius of the target.
        while len(out) > 1 and math.hypot(out[-1][0] - tx, out[-1][1] - ty) < approach:
            out.pop()
        # Land the final waypoint exactly `approach` from the target, along the last
        # segment's inward direction, so the truck ends facing the load.
        lx, ly = out[-1]
        d = math.hypot(tx - lx, ty - ly)
        if d > approach:
            f = (d - approach) / d
            out.append((lx + (tx - lx) * f, ly + (ty - ly) * f))
        return out

    @staticmethod
    def _path_len(wpts: list[tuple[float, float]]) -> float:
        return sum(math.hypot(bx - ax, by - ay)
                   for (ax, ay), (bx, by) in zip(wpts, wpts[1:]))

    def _pick_runup_candidates(self, goal_xy: tuple[float, float],
                               approach: float = APPROACH_LEN,
                               runup: float = PICK_RUNUP_LEN
                               ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Upper/lower map-side pick approaches as (runway_start, final_standoff)."""
        gx, gy = goal_xy
        return [
            ((gx, gy - approach - runup), (gx, gy - approach)),
            ((gx, gy + approach + runup), (gx, gy + approach)),
        ]

    def _route_pick(self, start_xy: tuple[float, float],
                    goal_xy: tuple[float, float],
                    avoid: set[str] | None = None) -> list[tuple[float, float]]:
        """Route to an upper/lower run-up, then drive straight into the pallet standoff.

        This makes pickup orthogonal to the pallet and gives the truck a few metres of
        aligned travel before the controller's slow fork-insertion creep begins."""
        options = []
        for runway_start, approach_xy in self._pick_runup_candidates(goal_xy):
            route_to_runway = self._route_xy(start_xy, runway_start, avoid=avoid)
            runway = self._densify([runway_start, approach_xy])
            route = route_to_runway + runway[1:]
            options.append((self._path_len(route), route))
        return min(options, key=lambda item: item[0])[1]

    # ---- commands (same signatures as WarehouseSim) --------------------- #
    def go_to(self, name: str, node: str) -> dict:
        if name not in FORKLIFTS:
            return {"ok": False, "error": f"unknown forklift {name}"}
        if node not in self._rm.nodes:
            return {"ok": False, "error": f"unknown node {node}"}
        goal = self._rm.nodes[node]
        wpts = self._route_xy(self._fk_pos(name), goal, avoid=self._blocked_by_others(name))
        self.bus.send_mission(name, [Leg(action="goto", target=node, waypoints=wpts)])
        return {"ok": True, "route": self._route_ids(wpts)}

    def pick(self, name: str, pallet_id: str) -> dict:
        if name not in FORKLIFTS:
            return {"ok": False, "error": f"unknown forklift {name}"}
        meta = PALLETS.get(pallet_id)
        if not meta:
            return {"ok": False, "error": f"unknown pallet {pallet_id}"}
        wpts = self._route_pick(self._fk_pos(name), meta["xy"],
                                avoid=self._blocked_by_others(name))
        leg = Leg(action="pick", target=pallet_id, waypoints=wpts, pallet_path=meta["path"])
        self.bus.send_mission(name, [leg])
        return {"ok": True, "route": self._route_ids(wpts)}

    def drop(self, name: str, zone_id: str) -> dict:
        if name not in FORKLIFTS:
            return {"ok": False, "error": f"unknown forklift {name}"}
        z = ZONES.get(zone_id)
        if not z:
            return {"ok": False, "error": f"unknown zone {zone_id}"}
        wpts = self._route_approach(self._fk_pos(name), z,
                                    avoid=self._blocked_by_others(name))
        leg = Leg(action="drop", target=zone_id, waypoints=wpts, drop_xy=z)
        self.bus.send_mission(name, [leg])
        return {"ok": True, "route": self._route_ids(wpts)}

    def go_home(self, name: str) -> dict:
        if name not in FORKLIFTS:
            return {"ok": False, "error": f"unknown forklift {name}"}
        hx, hy, _ = FORKLIFTS[name]
        wpts = self._route_xy(self._fk_pos(name), (hx, hy),
                              avoid=self._blocked_by_others(name))
        self.bus.send_mission(name, [Leg(action="home", target="home", waypoints=wpts)])
        return {"ok": True, "route": self._route_ids(wpts)}

    def mission(self, name: str, steps: list) -> dict:
        """Plan a full multi-leg mission (e.g. [["pick", p], ["drop", z]]) as one command.

        Each leg's route starts where the previous leg ends, so pick→drop chains cleanly
        without the bridge polling for completion — the controller walks the legs in order.
        """
        if name not in FORKLIFTS:
            return {"ok": False, "error": f"unknown forklift {name}"}
        legs: list[Leg] = []
        cur = self._fk_pos(name)
        # Steer this truck's whole mission clear of where the other forklifts sit.
        avoid = self._blocked_by_others(name)
        for step in steps:
            kind, target = (list(step) + [None])[:2]
            if kind == "pick":
                meta = PALLETS.get(target)
                if not meta:
                    return {"ok": False, "error": f"unknown pallet {target}"}
                wpts = self._route_pick(cur, meta["xy"], avoid=avoid)
                legs.append(Leg(action="pick", target=target, waypoints=wpts,
                                pallet_path=meta["path"]))
                cur = wpts[-1] if wpts else cur
            elif kind == "drop":
                z = ZONES.get(target)
                if not z:
                    return {"ok": False, "error": f"unknown zone {target}"}
                wpts = self._route_approach(cur, z, avoid=avoid)
                legs.append(Leg(action="drop", target=target, waypoints=wpts, drop_xy=z))
                cur = z
            elif kind == "goto":
                if target not in self._rm.nodes:
                    return {"ok": False, "error": f"unknown node {target}"}
                goal = self._rm.nodes[target]
                wpts = self._route_xy(cur, goal, avoid=avoid)
                legs.append(Leg(action="goto", target=target, waypoints=wpts))
                cur = goal
            elif kind == "home":
                hx, hy, _ = FORKLIFTS[name]
                wpts = self._route_xy(cur, (hx, hy), avoid=avoid)
                legs.append(Leg(action="home", target="home", waypoints=wpts))
                cur = (hx, hy)
            else:
                return {"ok": False, "error": f"unknown step {kind}"}
        if not legs:
            return {"ok": True, "done": True}
        self.bus.send_mission(name, legs)
        return {"ok": True, "route": self._route_ids(legs[0].waypoints)}

    def block_zone(self, zone_id: str, kind: str = "spill") -> dict:
        if zone_id not in ZONES:
            return {"ok": False, "error": f"unknown zone {zone_id}"}
        self.bus.set_zone(zone_id, blocked=True)
        zx, zy = ZONES[zone_id]
        self.hazards[zone_id] = {"zone": zone_id, "kind": kind, "x": zx, "y": zy,
                                 "t": round(self.bus.elapsed(), 2), "radius": 1.6}
        # Fleet reacts: any forklift with undelivered work bound for the now-blocked bay is
        # re-planned so ONLY the leg(s) targeting that bay move to the nearest clear bay —
        # its other pickups/drops (and any pallet already bound for a still-open bay) are
        # preserved. This is the visible "spill -> reroute" moment; trucks with no work for
        # the blocked bay are left untouched.
        rerouted = []
        for name in FORKLIFTS:
            steps = self._reroute_steps(name, zone_id)
            if steps is None:
                continue
            self.mission(name, steps)
            rerouted.append(name)
        return {"ok": True, "blocked": zone_id, "kind": kind, "rerouted": rerouted}

    def _reroute_steps(self, name: str, blocked: str) -> list[list[str]] | None:
        """Rebuild a forklift's REMAINING pick/drop tasks, diverting only the drops bound
        for the blocked bay. Returns a steps list for mission(), or None if this truck has
        no undelivered work for the blocked bay (so it should be left undisturbed).

        Tasks are recovered from the live command legs (paired pick->drop) minus any pallet
        already delivered. A pallet currently on the forks whose destination is still open is
        left for the truck to finish via its current mission (the controller defers a fresh
        mission while carrying) — we only re-issue it when its OWN bay is the blocked one, in
        which case the redirect leads with a drop so the controller adopts it immediately.
        """
        cmd = self.bus.get_command(name)
        t = self.bus.get_telemetry(name)
        carrying = t.carrying

        # Pair pick->drop across the full mission to recover (pallet, destination) tasks.
        tasks: list[tuple[str, str]] = []
        pending_pick: str | None = None
        for leg in cmd.legs:
            if leg.action == "pick":
                pending_pick = leg.target
            elif leg.action == "drop":
                pallet = pending_pick or carrying
                if pallet:
                    tasks.append((pallet, leg.target))
                pending_pick = None

        remaining = [(p, z) for (p, z) in tasks
                     if not self.bus.pallets.get(p, {}).get("delivered")]
        heading_here = (t.target == blocked and t.goal_kind in ("drop", "goto"))
        if not any(z == blocked for (_p, z) in remaining) and not heading_here:
            return None

        alt = self._nearest_free_zone(t.x, t.y, exclude=blocked)
        if not alt:
            return None

        steps: list[list[str]] = []
        for pallet, zone in remaining:
            dest = alt if zone == blocked else zone
            if pallet == carrying:
                # Already on the forks. If its bay is still open the truck finishes it via
                # the current mission (skip here); if its bay is blocked, lead with the drop
                # so the controller redirects the load at once instead of entering the bay.
                if zone == blocked:
                    steps.append(["drop", dest])
            else:
                steps += [["pick", pallet], ["drop", dest]]
        return steps or None

    def _nearest_free_zone(self, x: float, y: float,
                           exclude: str | None = None) -> str | None:
        """Closest un-blocked staging bay to (x, y), or None if all are blocked."""
        best, best_d = None, float("inf")
        for zid, (zx, zy) in ZONES.items():
            if zid == exclude or self.bus.zones.get(zid, {}).get("blocked"):
                continue
            d = math.hypot(zx - x, zy - y)
            if d < best_d:
                best, best_d = zid, d
        return best

    def reset(self) -> dict:
        """Reset the scene to its start state — pallets back on the racks, forklifts home
        on their chargers, full battery, no hazards — WITHOUT restarting Isaac, so the live
        WebRTC stream keeps running and the operator can run demo after demo. Only the
        shared bus is reset here; the scene controller teleports the USD prims back on its
        next physics step (it watches `bus.reset_epoch`)."""
        for name, (x, y, yaw) in FORKLIFTS.items():
            self.bus.clear_command(name)
            self.bus.update_telemetry(
                name, x=x, y=y, yaw=math.radians(yaw), phase="idle", speed=0.0,
                lift_height=0.0, carrying=None, route=[], target=None, goal_kind=None,
                object_detected="None", object_distance=0.0, path_blocked=False,
                battery=100.0)
        for pid, meta in PALLETS.items():
            self.bus.set_pallet(pid, x=meta["xy"][0], y=meta["xy"][1],
                                carried_by=None, delivered=False)
        for zid, (zx, zy) in ZONES.items():
            self.bus.set_zone(zid, x=zx, y=zy, blocked=False)
        self.hazards.clear()
        self.bus.request_reset()
        return {"ok": True, "reset": True}

    # ---- snapshot ------------------------------------------------------- #
    def snapshot(self) -> dict:
        forklifts = {}
        for name in FORKLIFTS:
            t = self.bus.get_telemetry(name)
            forklifts[name] = {
                "x": round(t.x, 3), "y": round(t.y, 3), "yaw": round(t.yaw, 4),
                "phase": t.phase, "speed": round(t.speed, 3),
                "lift_height": round(t.lift_height, 3), "carrying": t.carrying,
                "target": t.target, "goal_kind": t.goal_kind, "route": list(t.route),
                "object_detected": t.object_detected,
                "object_distance": round(t.object_distance, 3),
                "path_blocked": bool(t.path_blocked),
                "battery": round(getattr(t, "battery", 100.0), 1),
            }
        pallets = {pid: {"x": round(p["x"], 3), "y": round(p["y"], 3),
                         "carried_by": p.get("carried_by"), "delivered": bool(p.get("delivered"))}
                   for pid, p in self.bus.pallets.items()}
        zones = {zid: {"x": z["x"], "y": z["y"], "blocked": bool(z.get("blocked")),
                       "hazard": self.hazards.get(zid, {}).get("kind")}
                 for zid, z in self.bus.zones.items()}
        return {
            "t": round(self.bus.elapsed(), 2),
            "forklifts": forklifts,
            "pallets": pallets,
            "zones": zones,
            "chargers": {cid: {"x": x, "y": y} for cid, (x, y) in CHARGERS.items()},
            "hazards": {zid: dict(h) for zid, h in self.hazards.items()},
            "graph": self.bus.graph,
        }

    # ---- misc ----------------------------------------------------------- #
    def _route_ids(self, wpts: list[tuple[float, float]]) -> list[str]:
        return [self._rm.nearest(x, y) for x, y in wpts]
