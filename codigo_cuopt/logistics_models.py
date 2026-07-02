from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


NodeType = Literal["storage", "dock", "charging", "waypoint", "depot"]
ObjectiveType = Literal["min_distance", "min_time", "min_makespan"]
SolverStatus = Literal["success", "error"]


@dataclass
class Node:
    id: str
    node_type: NodeType
    x: float
    y: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Vehicle:
    id: str
    start_node_id: str
    capacity: float
    charging_node_id: str
    speed_mps: float = 1.2
    battery_level_pct: float = 100.0
    min_battery_reserve_pct: float = 15.0
    energy_per_meter_pct: float = 0.3
    max_shift_time_s: float = 8 * 3600
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    pallet_id: str
    pickup_node_id: str
    delivery_node_id: str
    demand: float = 1.0
    priority: int = 1
    service_time_s: float = 10.0
    pickup_time_window: tuple[float, float] = (0.0, 24 * 3600)
    delivery_time_window: tuple[float, float] = (0.0, 24 * 3600)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CuOptConstraints:
    vehicle_max_jobs: int | None = None
    allow_split_deliveries: bool = False
    node_occupancy_buffer_s: float = 3.0
    min_turnaround_s: float = 2.0
    blocked_edges: list[tuple[str, str]] = field(default_factory=list)
    blocked_nodes: list[str] = field(default_factory=list)
    max_route_time_s: float | None = None


@dataclass
class CuOptInput:
    nodes: list[Node]
    vehicles: list[Vehicle]
    jobs: list[Job]
    cost_matrix: list[list[float]]
    objective: ObjectiveType = "min_distance"
    constraints: CuOptConstraints = field(default_factory=CuOptConstraints)
    node_index: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteStop:
    node_id: str
    stop_type: Literal["start", "pickup", "delivery", "charging"]
    job_id: str = ""
    pallet_id: str = ""
    eta_s: float = 0.0
    departure_s: float = 0.0
    travel_cost: float = 0.0
    wait_s: float = 0.0


@dataclass
class VehicleRoute:
    vehicle_id: str
    stops: list[RouteStop] = field(default_factory=list)
    assigned_job_ids: list[str] = field(default_factory=list)
    route_cost: float = 0.0
    total_time_s: float = 0.0
    total_wait_s: float = 0.0
    consumed_battery_pct: float = 0.0


@dataclass
class CuOptOutput:
    status: SolverStatus
    vehicle_routes: list[VehicleRoute]
    total_cost: float
    errors: list[str] = field(default_factory=list)
    solver_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NavigationCommand:
    command_type: Literal["navigate", "pickup", "dropoff", "charge"]
    vehicle_id: str
    node_id: str
    x: float
    y: float
    pallet_id: str = ""
    job_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionPlan:
    mission_id: str
    objective: ObjectiveType
    vehicle_commands: dict[str, list[NavigationCommand]]
    total_cost: float
    status: SolverStatus
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionEvent:
    timestamp_s: float
    event_type: Literal["delay", "aisle_block", "human_crossing", "low_battery", "replan"]
    vehicle_id: str
    details: str
    severity: Literal["info", "warning", "critical"] = "info"


@dataclass
class ExecutionReport:
    status: Literal["ok", "degraded", "failed"]
    mission_id: str
    on_time_ratio: float
    events: list[ExecutionEvent] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
