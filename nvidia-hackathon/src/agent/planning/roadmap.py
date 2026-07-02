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

# Node id -> (x, y). Edges are an undirected adjacency map.
Coord = tuple[float, float]


class Roadmap:
    def __init__(self, nodes: dict[str, Coord], edges: dict[str, set[str]]) -> None:
        self.nodes = nodes
        self.edges = edges

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def from_snapshot(cls, snap: dict, cell: float = 2.0, margin: float = 2.0) -> "Roadmap":
        """Adopt the snapshot's graph if present, else synthesise a grid.

        Nodes whose id starts with ``rack`` are warehouse obstacles (the mock exposes them
        so the UI can draw rack slabs); they are dropped here so planned paths never route
        through a rack.
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
        return cls._synthesise_grid(snap, cell, margin)

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

    def neighbours(self, n: str) -> set[str]:
        return self.edges.get(n, set())

    def astar(self, start: str, goal: str) -> list[str]:
        """Plain shortest path (no time dimension). [] if unreachable."""
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
                ng = g[cur] + self.dist(cur, nxt)
                if ng < g.get(nxt, float("inf")):
                    came[nxt] = cur
                    g[nxt] = ng
                    heapq.heappush(open_heap, (ng + self.dist(nxt, goal), nxt))
        return []

    def path_length(self, path: list[str]) -> float:
        return sum(self.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
