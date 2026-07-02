"""
env_config.py
-------------
External configuration for adapting the generic cuOpt/CBS pipeline to a *real*
Isaac Sim environment (the FleetMind warehouse served over HTTP on :8080).

Everything an integrator might need to change per-deployment lives in
``fleet_config.json`` — base URL, entity naming conventions, prim templates,
nav-graph usage, vehicle defaults (battery/speed, which the live telemetry does
NOT expose), job strategy, cuOpt endpoint and CBS/replan parameters.

Loading is defensive: unknown keys are ignored, missing keys fall back to the
dataclass defaults below, and a missing file yields an all-default config so the
pipeline still runs. Nothing here is hardcoded into the algorithms.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_config.json")


@dataclass
class NamingConfig:
    forklift_prefix: str = "AMR_"
    pallet_prefix: str = "WH_Palette_"
    zone_prefix: str = "stage_"
    pallet_prim_template: str = "/World/Pallets/Pallet_{index:02d}"
    forklift_prim_template: str = "/World/AMRs/{forklift_id}"
    zone_prim_template: str = "/World/Zones/{zone_id}"


@dataclass
class GraphConfig:
    use_nav_graph: bool = True
    attach_strategy: str = "nearest"
    fallback_to_euclidean: bool = True


@dataclass
class VehicleDefaults:
    speed_mps: float = 1.2
    capacity: float = 1.0
    battery_level_pct: float = 100.0
    energy_per_meter_pct: float = 0.0
    min_battery_reserve_pct: float = 15.0
    charging_xy: tuple[float, float] = (0.0, 0.0)


@dataclass
class JobsConfig:
    strategy: str = "round_robin"


@dataclass
class CuOptConfig:
    endpoint_env: str = "CUOPT_URL"
    api_key_env: str = "CUOPT_API_KEY"
    objective: str = "min_distance"
    timeout_s: float = 30.0


@dataclass
class CbsConfig:
    enabled: bool = True
    max_expansions: int = 500


@dataclass
class ReplanConfig:
    enabled: bool = True
    triggers: list[str] = field(
        default_factory=lambda: ["path_blocked", "object_detected", "conflict", "pallet_moved", "low_battery"]
    )
    low_battery_pct: float = 20.0


@dataclass
class FleetConfig:
    base_url: str = "http://localhost:8080"
    http_timeout_s: float = 10.0
    naming: NamingConfig = field(default_factory=NamingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    vehicles: VehicleDefaults = field(default_factory=VehicleDefaults)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    cuopt: CuOptConfig = field(default_factory=CuOptConfig)
    cbs: CbsConfig = field(default_factory=CbsConfig)
    replan: ReplanConfig = field(default_factory=ReplanConfig)


def _merge(dc: Any, data: dict[str, Any]) -> None:
    """Shallow-merge a dict onto a dataclass instance, ignoring unknown keys."""
    for key, value in (data or {}).items():
        if hasattr(dc, key):
            setattr(dc, key, value)


def load_fleet_config(path: str | None = None) -> FleetConfig:
    """Load ``fleet_config.json`` (or defaults) into a typed ``FleetConfig``."""
    cfg = FleetConfig()
    resolved = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(resolved):
        return cfg

    with open(resolved, encoding="utf-8") as fh:
        data = json.load(fh)

    cfg.base_url = str(data.get("base_url", cfg.base_url))
    cfg.http_timeout_s = float(data.get("http_timeout_s", cfg.http_timeout_s))

    _merge(cfg.naming, data.get("naming", {}))
    _merge(cfg.graph, data.get("graph", {}))
    _merge(cfg.vehicles, data.get("vehicles", {}))
    _merge(cfg.jobs, data.get("jobs", {}))
    _merge(cfg.cuopt, data.get("cuopt", {}))
    _merge(cfg.cbs, data.get("cbs", {}))
    _merge(cfg.replan, data.get("replan", {}))

    # Normalise charging_xy to a tuple.
    cfg.vehicles.charging_xy = tuple(cfg.vehicles.charging_xy)  # type: ignore[assignment]
    return cfg
