"""
Roadmap — a waypoint graph the multi-agent planner (CBS) and router reason over.

Backend-agnostic: it is built straight from a `/state` snapshot, so it works identically
against the offline mock and the real Isaac scene. If the snapshot already carries a
waypoint graph (the mock does), we adopt it verbatim. If it doesn't (the real Isaac
`AMR` scene has no `/World/WaypointGraph`), we synthesise a uniform 4-connected grid over
the bounding box of everything on the floor, punching out the cells occupied by pallets so
routes naturally avoid them.

Only the standard library is used so this imports anywhere (agent side, off-DGX).
"""
from __future__ import annotations

import heapq
import math
import itertools

# Node id -> (x, y). Edges are an undirected adjacency map.
Coord = tuple[float, float]


class Roadmap:
    def __init__(self, nodes: dict[str, Coord], edges: dict[str, set[str]]) -> None:
        self.nodes = nodes
        self.edges = edges

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def from_snapshot(cls, snap: dict, cell: float = 2.0, margin: float = 2.0,
                      grid: tuple[int, int] | None = None) -> "Roadmap":
        """Adopt the snapshot's graph if present, else synthesise a grid.

        Nodes whose id starts with ``rack`` are warehouse obstacles (the mock exposes them
        so the UI can draw rack slabs); they are dropped here so planned paths never route
        through a rack. When ``grid=(nx, ny)`` is given (and the snapshot has no graph of
        its own), a fixed ``nx × ny`` uniform mesh is laid over the floor instead of a
        cell-sized grid — this is what the live Isaac scene uses for its 20×20 waypoint map.
        """
        g = snap.get("graph") or {}
        gnodes = g.get("nodes") or {}
        if gnodes:
            nodes = {n: (float(xy[0]), float(xy[1])) for n, xy in gnodes.items()
                     if not n.startswith("rack")}
            edges: dict[str, set[str]] = {n: set() for n in nodes}
            for a, b in g.get("edges", []):
                if a in edges and b in edges:
                    edges[a].add(b)
                    edges[b].add(a)
            return cls(nodes, edges)
        if grid is not None:
            return cls._uniform_grid(snap, grid[0], grid[1], margin)
        return cls._synthesise_grid(snap, cell, margin)

    @staticmethod
    def _floor_bounds(snap: dict, margin: float) -> tuple[float, float, float, float]:
        pts: list[Coord] = []
        for fk in snap.get("forklifts", {}).values():
            pts.append((fk["x"], fk["y"]))
        for p in snap.get("pallets", {}).values():
            pts.append((p["x"], p["y"]))
        for z in snap.get("zones", {}).values():
            pts.append((z["x"], z["y"]))
        if not pts:
            pts = [(0.0, 0.0)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)

    @classmethod
    def _uniform_grid(cls, snap: dict, nx: int, ny: int, margin: float) -> "Roadmap":
        """A fixed ``nx × ny`` 4-connected mesh spanning the floor bounding box.

        Cells nearest a resting pallet are punched out so routes naturally arc around
        loaded rack faces; the fine spacing (vs. the old 1 m grid) gives the trucks a much
        smoother, more legible trajectory and denser on-map waypoint markers.
        """
        nx, ny = max(2, nx), max(2, ny)
        min_x, max_x, min_y, max_y = cls._floor_bounds(snap, margin)
        step_x = (max_x - min_x) / (nx - 1)
        step_y = (max_y - min_y) / (ny - 1)

        def cell_of(x: float, y: float) -> tuple[int, int]:
            c = min(nx - 1, max(0, round((x - min_x) / step_x))) if step_x else 0
            r = min(ny - 1, max(0, round((y - min_y) / step_y))) if step_y else 0
            return (c, r)

        blocked_cells: set[tuple[int, int]] = set()
        for p in snap.get("pallets", {}).values():
            if p.get("delivered") or p.get("carried_by"):
                continue
            blocked_cells.add(cell_of(p["x"], p["y"]))

        nodes: dict[str, Coord] = {}
        for c in range(nx):
            for r in range(ny):
                if (c, r) in blocked_cells:
                    continue
                nodes[f"g{c}_{r}"] = (min_x + c * step_x, min_y + r * step_y)

        edges: dict[str, set[str]] = {n: set() for n in nodes}
        for c in range(nx):
            for r in range(ny):
                a = f"g{c}_{r}"
                if a not in nodes:
                    continue
                for dc, dr in ((1, 0), (0, 1)):
                    b = f"g{c + dc}_{r + dr}"
                    if b in nodes:
                        edges[a].add(b)
                        edges[b].add(a)
        return cls(nodes, edges)

    @classmethod
    def _synthesise_grid(cls, snap: dict, cell: float, margin: float) -> "Roadmap":
        pts: list[Coord] = []
        for fk in snap.get("forklifts", {}).values():
            pts.append((fk["x"], fk["y"]))
        for p in snap.get("pallets", {}).values():
            pts.append((p["x"], p["y"]))
        for z in snap.get("zones", {}).values():
            pts.append((z["x"], z["y"]))
        if not pts:
            pts = [(0.0, 0.0)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin

        # Cells blocked by pallets (so the grid routes around loaded rack faces).
        blocked_cells: set[tuple[int, int]] = set()
        for p in snap.get("pallets", {}).values():
            if p.get("delivered") or p.get("carried_by"):
                continue
            blocked_cells.add((round((p["x"] - min_x) / cell), round((p["y"] - min_y) / cell)))

        nodes: dict[str, Coord] = {}
        ncols = max(1, int((max_x - min_x) / cell) + 1)
        nrows = max(1, int((max_y - min_y) / cell) + 1)
        for c in range(ncols):
            for r in range(nrows):
                if (c, r) in blocked_cells:
                    continue
                nodes[f"g{c}_{r}"] = (min_x + c * cell, min_y + r * cell)

        edges: dict[str, set[str]] = {n: set() for n in nodes}
        for c in range(ncols):
            for r in range(nrows):
                a = f"g{c}_{r}"
                if a not in nodes:
                    continue
                for dc, dr in ((1, 0), (0, 1)):
                    b = f"g{c + dc}_{r + dr}"
                    if b in nodes:
                        edges[a].add(b)
                        edges[b].add(a)
        return cls(nodes, edges)

    # ---- queries --------------------------------------------------------- #
    def dist(self, a: str, b: str) -> float:
        ax, ay = self.nodes[a]
        bx, by = self.nodes[b]
        return math.hypot(ax - bx, ay - by)

    def nearest(self, x: float, y: float) -> str:
        return min(self.nodes, key=lambda n: math.hypot(self.nodes[n][0] - x, self.nodes[n][1] - y))

    def nodes_within(self, x: float, y: float, radius: float) -> set[str]:
        """All node ids whose cell centre lies within `radius` of (x, y) — used to mark the
        floor footprint of a parked forklift as blocked so others route around it."""
        r2 = radius * radius
        return {n for n, (nx, ny) in self.nodes.items()
                if (nx - x) ** 2 + (ny - y) ** 2 <= r2}

    def neighbours(self, n: str) -> set[str]:
        return self.edges.get(n, set())

    def astar(self, start: str, goal: str,
              avoid: set[str] | None = None) -> list[str]:
        """Plain shortest path (no time dimension). [] if unreachable.

        `avoid` is a set of node ids to treat as blocked (e.g. cells occupied by another
        forklift), except the goal itself is always reachable so a target adjacent to an
        obstacle can still be delivered to."""
        if start == goal:
            return [start]
        av = avoid or set()
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
                if nxt in av and nxt != goal:
                    continue
                ng = g[cur] + self.dist(cur, nxt)
                if ng < g.get(nxt, float("inf")):
                    came[nxt] = cur
                    g[nxt] = ng
                    heapq.heappush(open_heap, (ng + self.dist(nxt, goal), nxt))
        return []

    def _dir(self, a: str, b: str) -> tuple[int, int]:
        """Quantised step direction a->b on the axis-aligned grid (one of ±x/±y)."""
        ax, ay = self.nodes[a]
        bx, by = self.nodes[b]
        dx, dy = bx - ax, by - ay
        n = math.hypot(dx, dy) or 1.0
        return (round(dx / n), round(dy / n))

    def astar_straight(self, start: str, goal: str,
                       turn_penalty: float = 4.0,
                       avoid: set[str] | None = None) -> list[str]:
        """Shortest path that MINIMISES turns — long straight runs, few corners.

        Same 4-connected grid as `astar`, but every change of heading costs an extra
        `turn_penalty`, so the search prefers to keep driving in a straight line and only
        turns when it must. The result is clean L-shaped / Manhattan routes (great for a
        forklift, which reverses/steers awkwardly) instead of the diagonal-looking
        staircase a plain distance A* produces. Falls back to `astar` if unreachable.

        `avoid` is a set of node ids to steer around (e.g. cells occupied by another
        forklift) so a dispatched truck's route bends around a parked one instead of
        driving through it; the goal node is never blocked so targets stay reachable.
        """
        if start == goal:
            return [start]
        av = avoid or set()
        tie = itertools.count()
        # state = (node, incoming_direction); direction None at the start.
        start_state = (start, None)
        open_heap: list[tuple] = [(self.dist(start, goal), next(tie), start, None)]
        g: dict[tuple, float] = {start_state: 0.0}
        came: dict[tuple, tuple] = {}
        while open_heap:
            _, _, cur, pdir = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                st = (cur, pdir)
                while st in came:
                    st = came[st]
                    path.append(st[0])
                return list(reversed(path))
            for nxt in self.edges.get(cur, ()):
                if nxt in av and nxt != goal:
                    continue
                ndir = self._dir(cur, nxt)
                turn = turn_penalty if (pdir is not None and ndir != pdir) else 0.0
                ng = g[(cur, pdir)] + self.dist(cur, nxt) + turn
                st = (nxt, ndir)
                if ng < g.get(st, float("inf")):
                    came[st] = (cur, pdir)
                    g[st] = ng
                    heapq.heappush(open_heap,
                                   (ng + self.dist(nxt, goal), next(tie), nxt, ndir))
        return self.astar(start, goal, avoid=av)

    @staticmethod
    def collapse_collinear(pts: list[Coord], eps: float = 1e-3) -> list[Coord]:
        """Drop interior points that lie on the straight line between their neighbours,
        so a long straight run becomes a single segment (fewer, cleaner waypoints)."""
        if len(pts) < 3:
            return list(pts)
        out = [pts[0]]
        for i in range(1, len(pts) - 1):
            ax, ay = out[-1]
            bx, by = pts[i]
            cx, cy = pts[i + 1]
            # cross product of (b-a) x (c-a); ~0 => collinear, so b is redundant.
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(cross) > eps:
                out.append(pts[i])
        out.append(pts[-1])
        return out

    def path_length(self, path: list[str]) -> float:
        return sum(self.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
