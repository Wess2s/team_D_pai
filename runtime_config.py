from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any


@dataclass
class CuOptServiceConfig:
    endpoint: str = ""
    api_key_env: str = "CUOPT_API_KEY"
    timeout_s: float = 30.0
    enabled: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class PlannerConfig:
    objective: str = "min_makespan"
    enable_cbs: bool = True
    enable_replan: bool = True
    max_replan_attempts: int = 3
    cbs_max_expansions: int = 500


@dataclass
class RuntimeConfig:
    cuopt: CuOptServiceConfig = field(default_factory=CuOptServiceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RuntimeConfig":
        cfg = RuntimeConfig()

        cuopt_data = data.get("cuopt", {})
        cfg.cuopt.endpoint = str(cuopt_data.get("endpoint", cfg.cuopt.endpoint))
        cfg.cuopt.api_key_env = str(cuopt_data.get("api_key_env", cfg.cuopt.api_key_env))
        cfg.cuopt.timeout_s = float(cuopt_data.get("timeout_s", cfg.cuopt.timeout_s))
        cfg.cuopt.enabled = bool(cuopt_data.get("enabled", cfg.cuopt.enabled))

        log_data = data.get("logging", {})
        cfg.logging.level = str(log_data.get("level", cfg.logging.level)).upper()

        planner_data = data.get("planner", {})
        cfg.planner.objective = str(planner_data.get("objective", cfg.planner.objective))
        cfg.planner.enable_cbs = bool(planner_data.get("enable_cbs", cfg.planner.enable_cbs))
        cfg.planner.enable_replan = bool(planner_data.get("enable_replan", cfg.planner.enable_replan))
        cfg.planner.max_replan_attempts = int(
            planner_data.get("max_replan_attempts", cfg.planner.max_replan_attempts)
        )
        cfg.planner.cbs_max_expansions = int(
            planner_data.get("cbs_max_expansions", cfg.planner.cbs_max_expansions)
        )
        return cfg


def load_runtime_config(config_path: str = "runtime_config.json") -> RuntimeConfig:
    path = Path(config_path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return RuntimeConfig.from_dict(payload)

    env_payload = {
        "cuopt": {
            "endpoint": os.environ.get("CUOPT_ENDPOINT", ""),
            "api_key_env": os.environ.get("CUOPT_API_KEY_ENV", "CUOPT_API_KEY"),
            "enabled": os.environ.get("CUOPT_ENABLED", "false").lower() == "true",
        },
        "logging": {
            "level": os.environ.get("PHYSICAL_AI_LOG_LEVEL", "INFO"),
        },
    }
    return RuntimeConfig.from_dict(env_payload)
