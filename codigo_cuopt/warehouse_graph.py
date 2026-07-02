from __future__ import annotations

import math
from dataclasses import dataclass

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
