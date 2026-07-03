"""
Kinematic warehouse simulation — the OFFLINE stand-in for Isaac Sim.

This module runs a lightweight, thread-safe, physics-free model of the warehouse:
forklifts drive along a waypoint graph (A*), pick pallets, carry them, and drop them at
staging zones. It exposes exactly the same telemetry surface as the real Isaac forklift
behaviour script (the `nav*` attributes), so the agent, HTTP bridge and UI are developed
and demoed against this today and re-pointed at the real DGX scene tomorrow with **no
changes above the bridge**.

What it mirrors from the real forklift behaviour script:
    control : go_to / pick / drop / home            (bridge POST endpoints)
    telemetry (per forklift):
        pose            x, y, yaw            (odometry)
        phase           idle|navigating|picking|lifting|carrying|dropping|returning
        lift_height     metres               (fork height)
        carrying        pallet id or None    (navWithPallet)
        route           [node, ...]          (navFinalRoute)
        target          node / pallet / zone
        object_detected None|Pallet|Rack|Forklift  (navObjectDetected)
        object_distance metres               (navObjectDistance)
        path_blocked    bool                 (navPathBlocked)
        speed           m/s

Coordinates are in metres, a top-down warehouse frame. Pure standard-library + the
`heapq`/`math` modules — no numpy needed here so it stays trivially importable anywhere.
"""
from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Waypoint graph
# --------------------------------------------------------------------------- #
@dataclass
class Graph:
    """A 2-D waypoint graph: node id -> (x, y), plus an adjacency set."""

    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, nid: str, x: float, y: float) -> None:
        self.nodes[nid] = (x, y)
        self.edges.setdefault(nid, set())

    def add_edge(self, a: str, b: str) -> None:
        self.edges.setdefault(a, set()).add(b)
        self.edges.setdefault(b, set()).add(a)

    def dist(self, a: str, b: str) -> float:
        ax, ay = self.nodes[a]
        bx, by = self.nodes[b]
        return math.hypot(ax - bx, ay - by)

    def nearest(self, x: float, y: float, blocked: set[str] | None = None) -> str:
        blocked = blocked or set()
        return min(
            (n for n in self.nodes if n not in blocked),
            key=lambda n: math.hypot(self.nodes[n][0] - x, self.nodes[n][1] - y),
        )

    def astar(self, start: str, goal: str, blocked: set[str] | None = None) -> list[str]:
        """A* shortest path over the graph. Returns [] if unreachable."""
        blocked = blocked or set()
        if start == goal:
            return [start]
        open_heap: list[tuple[float, str]] = [(0.0, start)]
        came: dict[str, str] = {}
        g = {start: 0.0}
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return list(reversed(path))
            for nxt in self.edges.get(cur, ()):
                if nxt in blocked and nxt != goal:
                    continue
                ng = g[cur] + self.dist(cur, nxt)
                if ng < g.get(nxt, float("inf")):
                    came[nxt] = cur
                    g[nxt] = ng
                    f = ng + self.dist(nxt, goal)
                    heapq.heappush(open_heap, (f, nxt))
        return []


# --------------------------------------------------------------------------- #
# World entities
# --------------------------------------------------------------------------- #
@dataclass
class Pallet:
    id: str
    x: float
    y: float
    node: str = ""          # drivable pick-face node this pallet occupies
    carried_by: str | None = None
    delivered: bool = False


@dataclass
class Zone:
    id: str
    x: float
    y: float
    blocked: bool = False


@dataclass
class Forklift:
    name: str
    x: float
    y: float
    yaw: float = 0.0
    phase: str = "idle"          # idle|navigating|picking|lifting|carrying|dropping|returning
    lift_height: float = 0.0
    carrying: str | None = None
    route: list[str] = field(default_factory=list)   # remaining node ids
    target: str | None = None    # node / pallet / zone id being served
    goal_kind: str | None = None # goto|pick|drop|home
    speed: float = 0.0
    object_detected: str = "None"
    object_distance: float = 0.0
    path_blocked: bool = False
    queue: list[tuple[str, str | None]] = field(default_factory=list)  # pending steps
    _timer: float = 0.0          # seconds remaining in a timed action (pick/drop)


# --------------------------------------------------------------------------- #
# Warehouse simulation
# --------------------------------------------------------------------------- #
class WarehouseSim:
    """
    Thread-safe kinematic warehouse. Call start() to run its own stepping thread,
    or step(dt) manually. All mutating commands are queued-safe via a lock.
    """

    MAX_SPEED = 1.6          # m/s cruise
    APPROACH_SPEED = 0.6     # m/s near a target
    TURN_RATE = math.radians(140)   # rad/s max yaw slew
    PICK_TIME = 2.5          # s to insert + raise forks
    DROP_TIME = 2.0          # s to lower + release
    LIFT_CARRY = 0.5         # m fork height while carrying
    ARRIVE_EPS = 0.25        # m waypoint arrival tolerance
    DETECT_RANGE = 4.0       # m proximity-sensor range

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.graph = Graph()
        self.forklifts: dict[str, Forklift] = {}
        self.pallets: dict[str, Pallet] = {}
        self.zones: dict[str, Zone] = {}
        self.home: dict[str, str] = {}   # forklift -> home node id
        self.hazards: dict[str, dict] = {}   # zone id -> {kind,x,y,t,radius} incident
        self.t = 0.0
        build_demo_warehouse(self)

    # ---- lifecycle ------------------------------------------------------- #
    def start(self, hz: float = 30.0, rtf: float | None = None) -> None:
        """
        Run the stepping thread. `rtf` is a real-time factor (1.0 = wall-clock). Higher
        values fast-forward the sim (useful for tests); falls back to the SIM_RTF env var.
        """
        import os
        if rtf is None:
            rtf = float(os.getenv("SIM_RTF", "1.0"))
        with self._lock:
            if self._running:
                return
            self._running = True
        dt = 1.0 / hz

        def _loop() -> None:
            last = time.time()
            while self._running:
                now = time.time()
                self.step(min((now - last) * rtf, 0.2))
                last = now
                time.sleep(dt)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False

    # ---- geometry helpers ------------------------------------------------ #
    def _rack_nodes(self) -> set[str]:
        return {n for n in self.graph.nodes if n.startswith("rack")}

    def _pallet_nodes(self, exclude: str | None = None) -> set[str]:
        """Pick-face nodes occupied by undelivered, un-carried pallets (obstacles).

        `exclude` is the id of a pallet we're deliberately driving to (its node stays
        drivable so the forklift can reach it).
        """
        blocked = set()
        for pid, p in self.pallets.items():
            if pid == exclude or p.carried_by or p.delivered or not p.node:
                continue
            blocked.add(p.node)
        return blocked

    def _pos_of(self, target: str) -> tuple[float, float] | None:
        if target in self.graph.nodes:
            return self.graph.nodes[target]
        if target in self.pallets:
            p = self.pallets[target]
            return (p.x, p.y)
        if target in self.zones:
            z = self.zones[target]
            return (z.x, z.y)
        return None

    def _route_to(self, fk: Forklift, dest_xy: tuple[float, float],
                  pick_target: str | None = None) -> list[str]:
        # Racks are hard obstacles; other pallets are soft obstacles to route around.
        blocked = self._rack_nodes() | self._pallet_nodes(exclude=pick_target)
        start = self.graph.nearest(fk.x, fk.y, blocked)
        goal = self.graph.nearest(dest_xy[0], dest_xy[1], blocked)
        return self.graph.astar(start, goal, blocked)

    # ---- commands (thread-safe) ----------------------------------------- #
    def _begin(self, fk: Forklift, kind: str, target: str | None) -> dict:
        """Start a single action immediately (internal; assumes lock held)."""
        if kind == "home":
            target = self.home.get(fk.name, self.graph.nearest(fk.x, fk.y))
        # A staging bay under a hazard is closed: never route a drop/goto into it —
        # divert to the nearest clear bay so a truck can't drive into the spill.
        if (kind in ("drop", "goto") and target in self.zones
                and self.zones[target].blocked):
            alt = self._nearest_free_zone(fk.x, fk.y, exclude=target)
            if alt:
                target = alt
        dest = self._pos_of(target) if target else None
        if dest is None:
            return {"ok": False, "error": f"unknown target {target}"}
        if kind == "pick":
            p = self.pallets.get(target)
            if p is None or p.carried_by or p.delivered:
                return {"ok": False, "error": f"{target} unavailable"}
        if kind == "drop":
            if not fk.carrying:
                return {"ok": False, "error": f"{fk.name} carries nothing"}
        pick_target = target if kind == "pick" else None
        fk.route = self._route_to(fk, dest, pick_target=pick_target)[1:]
        fk.target = target
        fk.goal_kind = kind
        fk.phase = {"pick": "navigating", "drop": "carrying",
                    "goto": "navigating", "home": "returning"}[kind]
        fk.path_blocked = not fk.route
        return {"ok": True, "route": fk.route}

    def go_to(self, name: str, node: str) -> dict:
        with self._lock:
            fk = self.forklifts[name]
            fk.queue.clear()
            return self._begin(fk, "goto", node)

    def pick(self, name: str, pallet_id: str) -> dict:
        with self._lock:
            fk = self.forklifts[name]
            fk.queue.clear()
            return self._begin(fk, "pick", pallet_id)

    def drop(self, name: str, zone_id: str) -> dict:
        with self._lock:
            fk = self.forklifts[name]
            fk.queue.clear()
            return self._begin(fk, "drop", zone_id)

    def go_home(self, name: str) -> dict:
        with self._lock:
            fk = self.forklifts[name]
            fk.queue.clear()
            return self._begin(fk, "home", None)

    def mission(self, name: str, steps: list) -> dict:
        """
        Queue a multi-step mission, e.g. move a pallet to a zone:
            mission("forklift1", [["pick", "WH_Palette_01"], ["drop", "stage_1"]])
        Each step is [kind, target] with kind in goto|pick|drop|home.
        """
        with self._lock:
            fk = self.forklifts[name]
            fk.queue = [tuple(s) if len(s) == 2 else (s[0], None) for s in steps]
            return self._start_next(fk)

    def _start_next(self, fk: Forklift) -> dict:
        """Pop and begin the next queued step, if any (assumes lock held)."""
        if not fk.queue:
            return {"ok": True, "done": True}
        kind, target = fk.queue.pop(0)
        return self._begin(fk, kind, target)

    def block_zone(self, zone_id: str, kind: str = "spill") -> dict:
        with self._lock:
            if zone_id not in self.zones:
                return {"ok": False, "error": f"unknown zone {zone_id}"}
            z = self.zones[zone_id]
            z.blocked = True
            self.hazards[zone_id] = {"zone": zone_id, "kind": kind,
                                     "x": z.x, "y": z.y, "t": round(self.t, 2),
                                     "radius": 1.6}
            # Fleet reacts: any forklift heading to the now-blocked bay is diverted to
            # the nearest clear staging bay — whether it is already carrying or still en
            # route to pick up (its queued drop is rewritten). This is the visible
            # "spill → reroute" moment; trucks not heading there are untouched.
            rerouted = []
            for fk in self.forklifts.values():
                diverted = False
                # 1) rewrite any queued drop/goto step aimed at the blocked bay
                for i, (k, t) in enumerate(fk.queue):
                    if k in ("drop", "goto") and t == zone_id:
                        alt = self._nearest_free_zone(fk.x, fk.y, exclude=zone_id)
                        if alt:
                            fk.queue[i] = (k, alt)
                            diverted = True
                # 2) if actively heading into the blocked bay right now, divert this leg
                if fk.target == zone_id and fk.goal_kind in ("drop", "goto"):
                    alt = self._nearest_free_zone(fk.x, fk.y, exclude=zone_id)
                    if alt:
                        self._begin(fk, fk.goal_kind, alt)
                        diverted = True
                if diverted:
                    rerouted.append(fk.name)
            return {"ok": True, "blocked": zone_id, "kind": kind,
                    "rerouted": rerouted}

    def _nearest_free_zone(self, x: float, y: float,
                           exclude: str | None = None) -> str | None:
        """Closest un-blocked staging bay to (x, y), or None if all are blocked."""
        best, best_d = None, float("inf")
        for zid, z in self.zones.items():
            if zid == exclude or z.blocked:
                continue
            d = math.hypot(z.x - x, z.y - y)
            if d < best_d:
                best, best_d = zid, d
        return best

    # ---- simulation step ------------------------------------------------- #
    def step(self, dt: float) -> None:
        with self._lock:
            self.t += dt
            for fk in self.forklifts.values():
                self._update_proximity(fk)
                self._step_forklift(fk, dt)

    def _update_proximity(self, fk: Forklift) -> None:
        """Nearest other forklift within sensor range -> object_detected/distance."""
        best_d, best_kind = self.DETECT_RANGE, "None"
        for other in self.forklifts.values():
            if other.name == fk.name:
                continue
            d = math.hypot(other.x - fk.x, other.y - fk.y)
            if d < best_d:
                best_d, best_kind = d, "Forklift"
        fk.object_detected = best_kind
        fk.object_distance = round(best_d, 2) if best_kind != "None" else 0.0

    def _step_forklift(self, fk: Forklift, dt: float) -> None:
        # Timed actions (pick/drop) take priority.
        if fk._timer > 0.0:
            fk._timer = max(0.0, fk._timer - dt)
            if fk.phase == "picking":
                fk.lift_height = self.LIFT_CARRY * (1 - fk._timer / self.PICK_TIME)
            elif fk.phase == "dropping":
                fk.lift_height = self.LIFT_CARRY * (fk._timer / self.DROP_TIME)
            if fk._timer == 0.0:
                self._finish_action(fk)
            return

        if not fk.route:
            if fk.phase in ("navigating", "carrying", "returning"):
                self._arrive(fk)
            else:
                fk.speed = 0.0
            return

        # Drive toward the next waypoint.
        nx, ny = self.graph.nodes[fk.route[0]]
        dx, dy = nx - fk.x, ny - fk.y
        d = math.hypot(dx, dy)
        if d < self.ARRIVE_EPS:
            fk.route.pop(0)
            return

        target_yaw = math.atan2(dy, dx)
        fk.yaw = _slew(fk.yaw, target_yaw, self.TURN_RATE * dt)

        # Slow down for the final waypoint; keep a carried pallet steady.
        cruise = self.MAX_SPEED if len(fk.route) > 1 else self.APPROACH_SPEED
        # Yield if another forklift is close ahead (simple mutual avoidance).
        if fk.object_detected == "Forklift" and fk.object_distance < 1.8:
            cruise *= 0.25
            fk.path_blocked = True
        else:
            fk.path_blocked = False

        # Only advance along heading once roughly facing the waypoint.
        facing = abs(_wrap(target_yaw - fk.yaw)) < math.radians(35)
        fk.speed = cruise if facing else 0.0
        step = fk.speed * dt
        if step > 0:
            fk.x += math.cos(fk.yaw) * step
            fk.y += math.sin(fk.yaw) * step

        # Drag a carried pallet along with the forks.
        if fk.carrying and fk.carrying in self.pallets:
            p = self.pallets[fk.carrying]
            p.x = fk.x + math.cos(fk.yaw) * 0.9
            p.y = fk.y + math.sin(fk.yaw) * 0.9

    def _arrive(self, fk: Forklift) -> None:
        """Reached the end of a route — begin the terminal action for the goal."""
        fk.speed = 0.0
        if fk.goal_kind == "pick":
            fk.phase = "picking"
            fk._timer = self.PICK_TIME
        elif fk.goal_kind == "drop":
            fk.phase = "dropping"
            fk._timer = self.DROP_TIME
        else:  # goto / home
            fk.goal_kind = None
            fk.target = None
            fk.phase = "idle"
            self._start_next(fk)

    def _finish_action(self, fk: Forklift) -> None:
        if fk.phase == "picking" and fk.target in self.pallets:
            p = self.pallets[fk.target]
            p.carried_by = fk.name
            fk.carrying = p.id
            fk.lift_height = self.LIFT_CARRY
            fk.phase = "carrying"
            fk.goal_kind = None
            fk.target = None
            self._start_next(fk)
        elif fk.phase == "dropping":
            if fk.carrying and fk.carrying in self.pallets:
                p = self.pallets[fk.carrying]
                p.carried_by = None
                p.delivered = True
                if fk.target in self.zones:
                    z = self.zones[fk.target]
                    p.x, p.y = z.x, z.y
            fk.carrying = None
            fk.lift_height = 0.0
            fk.phase = "idle"
            fk.goal_kind = None
            fk.target = None
            self._start_next(fk)

    # ---- snapshot -------------------------------------------------------- #
    def snapshot(self) -> dict:
        """Full world state — the payload the bridge serves at /state."""
        with self._lock:
            return {
                "t": round(self.t, 2),
                "forklifts": {
                    fk.name: {
                        "x": round(fk.x, 3),
                        "y": round(fk.y, 3),
                        "yaw": round(fk.yaw, 4),
                        "phase": fk.phase,
                        "speed": round(fk.speed, 3),
                        "lift_height": round(fk.lift_height, 3),
                        "carrying": fk.carrying,
                        "target": fk.target,
                        "goal_kind": fk.goal_kind,
                        "route": list(fk.route),
                        "object_detected": fk.object_detected,
                        "object_distance": fk.object_distance,
                        "path_blocked": fk.path_blocked,
                    }
                    for fk in self.forklifts.values()
                },
                "pallets": {
                    p.id: {
                        "x": round(p.x, 3),
                        "y": round(p.y, 3),
                        "carried_by": p.carried_by,
                        "delivered": p.delivered,
                    }
                    for p in self.pallets.values()
                },
                "zones": {
                    z.id: {"x": z.x, "y": z.y, "blocked": z.blocked,
                           "hazard": self.hazards.get(z.id, {}).get("kind")}
                    for z in self.zones.values()
                },
                "hazards": {zid: dict(h) for zid, h in self.hazards.items()},
                "chargers": {
                    f"charge_{i + 1}": {"x": self.graph.nodes[node][0],
                                        "y": self.graph.nodes[node][1]}
                    for i, node in enumerate(self.home.values())
                    if node in self.graph.nodes
                },
                "graph": {
                    "nodes": {n: list(xy) for n, xy in self.graph.nodes.items()},
                    "edges": [[a, b] for a, nbrs in self.graph.edges.items()
                              for b in nbrs if a < b],
                },
            }


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def _wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _slew(current: float, target: float, max_delta: float) -> float:
    """Rotate `current` toward `target` by at most `max_delta`."""
    diff = _wrap(target - current)
    if abs(diff) <= max_delta:
        return _wrap(target)
    return _wrap(current + math.copysign(max_delta, diff))


# --------------------------------------------------------------------------- #
# Demo warehouse layout
# --------------------------------------------------------------------------- #
def build_demo_warehouse(sim: WarehouseSim) -> None:
    """
    Build a compact demo warehouse:
      - a drivable aisle grid (columns x rows) of waypoints
      - two rack blocks (obstacles) the A* routes around
      - pallets on the rack faces to pick
      - staging/drop zones on the right
      - three forklifts parked at a home/charging row on the left
    Frame: x rightwards 0..28 m, y upwards 0..16 m.
    """
    g = sim.graph
    cols = list(range(0, 8))        # 8 columns
    rows = list(range(0, 5))        # 5 rows
    sx, sy = 4.0, 3.5               # metres between waypoints

    def nid(c: int, r: int) -> str:
        return f"n{c}_{r}"

    # Grid nodes + 4-neighbour edges.
    for c in cols:
        for r in rows:
            g.add_node(nid(c, r), 1.0 + c * sx, 1.5 + r * sy)
    for c in cols:
        for r in rows:
            if c + 1 in cols:
                g.add_edge(nid(c, r), nid(c + 1, r))
            if r + 1 in rows:
                g.add_edge(nid(c, r), nid(c, r + 1))

    # Rack blocks: mark a couple of interior nodes as racks (obstacles). Pallets sit on
    # the drivable node just south of each rack (the pick face).
    rack_cells = [(2, 2), (2, 3), (5, 2), (5, 3)]
    for (c, r) in rack_cells:
        old = nid(c, r)
        x, y = g.nodes[old]
        # rename to a rack node so pathfinding treats it as blocked
        g.nodes.pop(old)
        rnode = f"rack{c}_{r}"
        g.nodes[rnode] = (x, y)
        g.edges[rnode] = g.edges.pop(old)
        for nbrs in g.edges.values():
            if old in nbrs:
                nbrs.discard(old)
                nbrs.add(rnode)

    # Pallets on pick faces (the drivable node just below each rack column).
    pick_faces = {
        "WH_Palette_01": nid(2, 1),
        "WH_Palette_02": nid(2, 0),
        "WH_Palette_03": nid(5, 1),
        "WH_Palette_04": nid(5, 0),
    }
    for pid, node in pick_faces.items():
        x, y = g.nodes[node]
        sim.pallets[pid] = Pallet(pid, x + 1.2, y, node=node)   # offset to the rack face

    # Drop / staging zones on the right edge.
    for i, r in enumerate((4, 3, 2)):
        node = nid(7, r)
        x, y = g.nodes[node]
        zid = f"stage_{i + 1}"
        sim.zones[zid] = Zone(zid, x + 1.0, y)

    # Home / charging row for three forklifts on the far left.
    homes = {"forklift1": nid(0, 4), "forklift2": nid(0, 3), "forklift3": nid(0, 2)}
    for name, node in homes.items():
        x, y = g.nodes[node]
        sim.forklifts[name] = Forklift(name, x, y, yaw=0.0)
        sim.home[name] = node
