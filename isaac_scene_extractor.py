from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from logistics_models import Node


@dataclass
class IsaacEntity:
    entity_id: str
    kind: str
    x: float
    y: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IsaacSceneSnapshot:
    entities: list[IsaacEntity]


def extract_from_isaac_stage(stage: Any) -> IsaacSceneSnapshot:
    """Extract relevant entities from an Isaac Sim stage.

    This is intentionally lightweight for hackathon speed. It only extracts
    entities relevant for logistics optimization, not full scene geometry.
    """
    entities: list[IsaacEntity] = []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lower_path = path.lower()

        if "storage" in lower_path:
            kind = "storage"
        elif "dock" in lower_path:
            kind = "dock"
        elif "charge" in lower_path:
            kind = "charging"
        elif "waypoint" in lower_path:
            kind = "waypoint"
        elif "forklift" in lower_path:
            kind = "vehicle"
        elif "pallet" in lower_path:
            kind = "pallet"
        else:
            continue

        attr_x = prim.GetAttribute("x")
        attr_y = prim.GetAttribute("y")
        x = float(attr_x.Get()) if attr_x and attr_x.IsValid() and attr_x.Get() is not None else 0.0
        y = float(attr_y.Get()) if attr_y and attr_y.IsValid() and attr_y.Get() is not None else 0.0

        entities.append(IsaacEntity(entity_id=path, kind=kind, x=x, y=y))

    return IsaacSceneSnapshot(entities=entities)


def snapshot_to_graph_nodes(snapshot: IsaacSceneSnapshot) -> list[Node]:
    nodes: list[Node] = []
    for entity in snapshot.entities:
        if entity.kind not in {"storage", "dock", "charging", "waypoint", "depot"}:
            continue
        node_type = "depot" if entity.kind == "vehicle" else entity.kind
        nodes.append(
            Node(
                id=entity.entity_id,
                node_type=node_type,  # type: ignore[arg-type]
                x=entity.x,
                y=entity.y,
                metadata=entity.metadata,
            )
        )
    return nodes


def build_mock_warehouse_snapshot() -> IsaacSceneSnapshot:
    """Build a small warehouse snapshot to test the E2E flow without Isaac Sim runtime."""
    entities = [
        IsaacEntity("Depot_Main", "depot", 0.0, 0.0),
        IsaacEntity("Storage_A", "storage", 3.0, 8.0),
        IsaacEntity("Storage_B", "storage", 8.0, 9.0),
        IsaacEntity("Storage_C", "storage", 12.0, 8.0),
        IsaacEntity("Dock_01", "dock", 18.0, 2.0),
        IsaacEntity("Dock_02", "dock", 18.0, 6.0),
        IsaacEntity("Charging_01", "charging", 1.0, 1.0),
        IsaacEntity("Waypoint_01", "waypoint", 10.0, 4.0),
    ]
    return IsaacSceneSnapshot(entities=entities)
