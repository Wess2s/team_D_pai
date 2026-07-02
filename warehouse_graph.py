from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

from logistics_models import Node


@dataclass
class WarehouseGraph:
    nodes: list[Node]
    node_index: dict[str, int]
    distance_matrix: list[list[float]]

    def get_distance(self, from_node_id: str, to_node_id: str) -> float:
        i = self.node_index[from_node_id]
        j = self.node_index[to_node_id]
        return self.distance_matrix[i][j]


def build_warehouse_graph(nodes: list[Node], blocked_pairs: set[tuple[str, str]] | None = None) -> WarehouseGraph:
    if not nodes:
        raise ValueError("At least one node is required to build WarehouseGraph")

    blocked_pairs = blocked_pairs or set()
    node_index = {node.id: idx for idx, node in enumerate(nodes)}

    matrix: list[list[float]] = []
    for node_i in nodes:
        row: list[float] = []
        for node_j in nodes:
            if node_i.id == node_j.id:
                row.append(0.0)
                continue

            if (node_i.id, node_j.id) in blocked_pairs:
                row.append(float("inf"))
                continue

            distance = math.dist((node_i.x, node_i.y), (node_j.x, node_j.y))
            row.append(round(distance, 3))
        matrix.append(row)

    return WarehouseGraph(nodes=nodes, node_index=node_index, distance_matrix=matrix)


# --------------------------------------------------------------------------- #
# Real Isaac Sim nav-graph support.
#
# The live FleetMind bridge exposes a real routing graph in GET /state as
# ``graph = {"nodes": {id: [x, y], ...}, "edges": [[a, b], ...]}`` (undirected
# grid adjacency). ``build_graph_from_state`` uses that real connectivity to
# compute *routing* distances between the logistics entities (forklift starts,
# pallets, zones, charging) instead of naive straight-line distances — each
# entity is attached to its nearest grid node and entity-to-entity cost is the
# shortest path over the real edges. The result is a small entity-only
# ``WarehouseGraph`` whose complete distance matrix already reflects real aisle
# routing, so the downstream cuOpt solver / CBS need no changes.
# --------------------------------------------------------------------------- #


def _grid_adjacency(
    grid_nodes: dict[str, list[float]],
    edges: list[list[str]],
    blocked_nodes: set[str] | None = None,
    blocked_edges: set[tuple[str, str]] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Build a weighted (euclidean per-edge) adjacency list from the nav graph."""
    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()
    adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in grid_nodes}
    for edge in edges:
        if len(edge) < 2:
            continue
        a, b = str(edge[0]), str(edge[1])
        if a not in grid_nodes or b not in grid_nodes:
            continue
        if a in blocked_nodes or b in blocked_nodes:
            continue
        if (a, b) in blocked_edges or (b, a) in blocked_edges:
            continue
        w = math.dist(grid_nodes[a], grid_nodes[b])
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def _dijkstra(adj: dict[str, list[tuple[str, float]]], source: str) -> dict[str, float]:
    """Single-source shortest paths over the weighted adjacency list."""
    dist: dict[str, float] = {source: 0.0}
    pq: list[tuple[float, str]] = [(0.0, source)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist.get(node, float("inf")):
            continue
        for neighbor, w in adj.get(node, []):
            nd = d + w
            if nd < dist.get(neighbor, float("inf")):
                dist[neighbor] = nd
                heapq.heappush(pq, (nd, neighbor))
    return dist


def _nearest_grid_node(grid_nodes: dict[str, list[float]], x: float, y: float) -> tuple[str, float]:
    best_id = ""
    best_d = float("inf")
    for nid, (nx, ny) in grid_nodes.items():
        d = math.dist((x, y), (nx, ny))
        if d < best_d:
            best_d = d
            best_id = nid
    return best_id, best_d


def nearest_grid_node(state: dict[str, Any], x: float, y: float) -> str:
    """Public helper: id of the nav-graph node closest to ``(x, y)``."""
    grid_nodes = {
        str(nid): [float(xy[0]), float(xy[1])]
        for nid, xy in ((state.get("graph") or {}).get("nodes") or {}).items()
    }
    if not grid_nodes:
        raise ValueError("state.graph has no nodes")
    return _nearest_grid_node(grid_nodes, x, y)[0]


def build_grid_graph_from_state(
    state: dict[str, Any],
    blocked_nodes: set[str] | None = None,
    blocked_edges: set[tuple[str, str]] | None = None,
) -> WarehouseGraph:
    """Build a **grid-level** ``WarehouseGraph`` over the real nav nodes/edges.

    Unlike :func:`build_graph_from_state` (entity-only, complete matrix), this
    returns the full grid: ``node_index`` over every nav-graph node and a
    ``distance_matrix`` where only real edges are finite (euclidean length) and
    all non-adjacent pairs are ``inf``. This is exactly the sparse connectivity
    CBS needs — its A* expands the true grid neighbours, so vertex/edge
    conflicts are resolved on the real aisles, not on abstract entity links.
    """
    graph = state.get("graph") or {}
    raw_nodes = graph.get("nodes") or {}
    raw_edges = graph.get("edges") or []
    grid_nodes: dict[str, list[float]] = {
        str(nid): [float(xy[0]), float(xy[1])] for nid, xy in raw_nodes.items()
    }
    if not grid_nodes or not raw_edges:
        raise ValueError("state.graph has no usable nodes/edges for grid graph")

    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()

    node_list = list(grid_nodes.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_list)}
    n = len(node_list)
    inf = float("inf")
    matrix: list[list[float]] = [[0.0 if i == j else inf for j in range(n)] for i in range(n)]

    for edge in raw_edges:
        if len(edge) < 2:
            continue
        a, b = str(edge[0]), str(edge[1])
        if a not in node_index or b not in node_index:
            continue
        if a in blocked_nodes or b in blocked_nodes:
            continue
        if (a, b) in blocked_edges or (b, a) in blocked_edges:
            continue
        w = round(math.dist(grid_nodes[a], grid_nodes[b]), 3)
        i, j = node_index[a], node_index[b]
        matrix[i][j] = w
        matrix[j][i] = w

    return WarehouseGraph(
        nodes=[Node(id=nid, node_type="waypoint", x=grid_nodes[nid][0], y=grid_nodes[nid][1]) for nid in node_list],
        node_index=node_index,
        distance_matrix=matrix,
    )


def build_graph_from_state(
    state: dict[str, Any],
    entity_nodes: list[Node],
    blocked_nodes: set[str] | None = None,
    blocked_edges: set[tuple[str, str]] | None = None,
) -> WarehouseGraph:
    """Build an entity-only ``WarehouseGraph`` whose distances follow the real
    Isaac nav graph.

    ``state`` is a live ``/state`` snapshot (must contain ``graph.nodes`` and
    ``graph.edges``). ``entity_nodes`` are the logistics nodes (pallet/zone/
    depot/charging) produced by the scene extractor. Each entity is attached to
    its nearest grid node; the matrix cell (i, j) is
    ``attach_i + shortest_path(near_i, near_j) + attach_j``.

    Raises ``ValueError`` if the state has no usable nav graph — callers may then
    fall back to :func:`build_warehouse_graph` (euclidean).
    """
    graph = state.get("graph") or {}
    raw_nodes = graph.get("nodes") or {}
    raw_edges = graph.get("edges") or []
    grid_nodes: dict[str, list[float]] = {
        str(nid): [float(xy[0]), float(xy[1])] for nid, xy in raw_nodes.items()
    }
    if not grid_nodes or not raw_edges:
        raise ValueError("state.graph has no usable nodes/edges for nav-graph routing")

    adj = _grid_adjacency(grid_nodes, raw_edges, blocked_nodes, blocked_edges)

    # Attach each entity to its nearest grid node.
    attach_node: dict[str, str] = {}
    attach_cost: dict[str, float] = {}
    for node in entity_nodes:
        near_id, near_d = _nearest_grid_node(grid_nodes, node.x, node.y)
        attach_node[node.id] = near_id
        attach_cost[node.id] = near_d

    # Shortest paths from each distinct attach node.
    dijkstra_cache: dict[str, dict[str, float]] = {}
    for anchor in set(attach_node.values()):
        dijkstra_cache[anchor] = _dijkstra(adj, anchor)

    node_index = {node.id: idx for idx, node in enumerate(entity_nodes)}
    matrix: list[list[float]] = []
    for node_i in entity_nodes:
        row: list[float] = []
        anchor_i = attach_node[node_i.id]
        dist_from_i = dijkstra_cache[anchor_i]
        for node_j in entity_nodes:
            if node_i.id == node_j.id:
                row.append(0.0)
                continue
            anchor_j = attach_node[node_j.id]
            grid_cost = dist_from_i.get(anchor_j, float("inf"))
            if math.isinf(grid_cost):
                row.append(float("inf"))
            else:
                total = attach_cost[node_i.id] + grid_cost + attach_cost[node_j.id]
                row.append(round(total, 3))
        matrix.append(row)

    return WarehouseGraph(nodes=entity_nodes, node_index=node_index, distance_matrix=matrix)

