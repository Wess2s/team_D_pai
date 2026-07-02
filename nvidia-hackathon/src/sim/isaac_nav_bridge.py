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
    (-3.0, -3.0), (-3.0, 3.0), (1.0, 5.0),
]
PALLETS = {f"WH_Palette_{i + 1:02d}": {"xy": xy, "path": f"/World/Pallets/Pallet_{i:02d}"}
           for i, xy in enumerate(_PALLET_GRID)}
# Staging bays the scene lacks — placed along the far (+y) aisle, clear of the racks.
ZONES = {
    "stage_1": (-6.0, 7.0),
    "stage_2": (0.0, 7.0),
    "stage_3": (6.0, 7.0),
}
# Charger docks (mirror scenes/scene_exec.py). Exposed in /state so the UI can
# draw them; each doubles as a forklift home/charge point.
CHARGERS = {
    "charge_1": (-6.0, -3.0),
    "charge_2": (6.0, 3.0),
}
# Battery model computed bridge-side (independent of the scene controller): drains
# with distance driven between /state polls, recharges while parked on a charger.
BATTERY_DRAIN_PER_M = 0.6         # % per metre driven
BATTERY_CHARGE_PER_POLL = 0.4     # % per /state poll while docked idle
CHARGER_RADIUS = 1.8              # m

# Densify every route into short, evenly-spaced waypoints so the truck tracks the
# planned line tightly (instead of arcing between far-apart grid nodes) and gets a
# straight, precise final run-in onto the pallet/zone. Because a pallet's own cell is
# blocked in the roadmap, A* arrives from the adjacent node facing the truck, so this
# final leg is already square-on — no forced aisle detour needed. STEP matches the
# controller's WAYPOINT_DIST so it advances smoothly point-to-point.
ROUTE_STEP = 0.6                         # m between densified waypoints
# Standoff for pick/drop: the forklift halts this far in front of the pallet/zone so its
# body doesn't ram the load; the fork reaches under from here. ~1.4 m ≈ half a truck +
# the forks, tuned to clear the pallet/rack collision that wedged the truck ~1.3 m out.
APPROACH_LEN = 1.4


class IsaacNavBackend:
    """Adapter over the live Isaac AMR scene, matching WarehouseSim's interface."""

    def __init__(self) -> None:
        self.bus = bus()
        self._battery = {name: 100.0 for name in FORKLIFTS}
        self._bat_last: dict[str, tuple[float, float]] = {}

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
        wpts = self._route_approach(self._fk_pos(name), meta["xy"],
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
                wpts = self._route_approach(cur, meta["xy"], avoid=avoid)
                legs.append(Leg(action="pick", target=target, waypoints=wpts,
                                pallet_path=meta["path"]))
                cur = meta["xy"]
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

    def block_zone(self, zone_id: str) -> dict:
        if zone_id not in ZONES:
            return {"ok": False, "error": f"unknown zone {zone_id}"}
        self.bus.set_zone(zone_id, blocked=True)
        return {"ok": True, "blocked": zone_id}

    # ---- snapshot ------------------------------------------------------- #
    def _update_battery(self, name: str, t) -> float:
        """Drain with distance driven since the last poll; recharge while parked on a
        charger. Computed here so it survives regardless of the scene controller."""
        b = self._battery.get(name, 100.0)
        last = self._bat_last.get(name)
        self._bat_last[name] = (t.x, t.y)
        if last is not None:
            moved = math.hypot(t.x - last[0], t.y - last[1])
            if moved > 1e-3:
                b -= moved * BATTERY_DRAIN_PER_M
            elif t.phase in ("idle", "returning"):
                near = min((math.hypot(t.x - cx, t.y - cy) for (cx, cy) in CHARGERS.values()), default=1e9)
                if near < CHARGER_RADIUS:
                    b += BATTERY_CHARGE_PER_POLL
        b = max(0.0, min(100.0, b))
        self._battery[name] = b
        return round(b, 1)

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
                "battery": self._update_battery(name, t),
            }
        pallets = {pid: {"x": round(p["x"], 3), "y": round(p["y"], 3),
                         "carried_by": p.get("carried_by"), "delivered": bool(p.get("delivered"))}
                   for pid, p in self.bus.pallets.items()}
        zones = {zid: {"x": z["x"], "y": z["y"], "blocked": bool(z.get("blocked"))}
                 for zid, z in self.bus.zones.items()}
        chargers = {cid: {"x": x, "y": y} for cid, (x, y) in CHARGERS.items()}
        return {
            "t": round(self.bus.elapsed(), 2),
            "forklifts": forklifts,
            "pallets": pallets,
            "zones": zones,
            "chargers": chargers,
            "graph": self.bus.graph,
        }

    # ---- misc ----------------------------------------------------------- #
    def _route_ids(self, wpts: list[tuple[float, float]]) -> list[str]:
        return [self._rm.nearest(x, y) for x, y in wpts]
