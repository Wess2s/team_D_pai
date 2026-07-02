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


def extract_from_state_snapshot(state: dict[str, Any]) -> IsaacSceneSnapshot:
    """Extract entities from a live FleetMind ``GET /state`` snapshot.

    This is the *real-environment* counterpart to :func:`extract_from_isaac_stage`.
    The deployed scene is not reachable via direct USD traversal from off-DGX, but
    the HTTP bridge exposes the same ground truth (forklift poses, pallet poses +
    carry/delivery flags, staging-zone poses + blocked flag, and the nav graph).
    Every entity keeps its real scene id so missions dispatch back with no name
    translation.

    Kinds produced: ``forklift``, ``pallet``, ``dock`` (staging zone),
    ``waypoint`` (nav-graph node). Charging nodes are not present in this scene
    and must be supplied via config (see env_config.VehicleDefaults.charging_xy).
    """
    entities: list[IsaacEntity] = []

    for fid, info in (state.get("forklifts") or {}).items():
        entities.append(
            IsaacEntity(
                entity_id=str(fid),
                kind="forklift",
                x=float(info.get("x", 0.0)),
                y=float(info.get("y", 0.0)),
                metadata={
                    "phase": info.get("phase", "idle"),
                    "carrying": info.get("carrying"),
                    "yaw": info.get("yaw", 0.0),
                    "path_blocked": info.get("path_blocked", False),
                    "object_detected": info.get("object_detected"),
                },
            )
        )

    for pid, info in (state.get("pallets") or {}).items():
        entities.append(
            IsaacEntity(
                entity_id=str(pid),
                kind="pallet",
                x=float(info.get("x", 0.0)),
                y=float(info.get("y", 0.0)),
                metadata={
                    "carried_by": info.get("carried_by"),
                    "delivered": info.get("delivered", False),
                },
            )
        )

    for zid, info in (state.get("zones") or {}).items():
        entities.append(
            IsaacEntity(
                entity_id=str(zid),
                kind="dock",
                x=float(info.get("x", 0.0)),
                y=float(info.get("y", 0.0)),
                metadata={"blocked": info.get("blocked", False)},
            )
        )

    for nid, xy in ((state.get("graph") or {}).get("nodes") or {}).items():
        entities.append(
            IsaacEntity(entity_id=str(nid), kind="waypoint", x=float(xy[0]), y=float(xy[1]))
        )

    return IsaacSceneSnapshot(entities=entities)


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
