from __future__ import annotations

from typing import Any


def execute_mission_dry_run(mission_plan: dict[str, Any]) -> dict[str, Any]:
    executed = []
    for vehicle_id, commands in mission_plan.get("vehicle_commands", {}).items():
        for cmd in commands:
            executed.append(
                {
                    "vehicle_id": vehicle_id,
                    "command_type": cmd.get("command_type", "navigate"),
                    "node_id": cmd.get("node_id", ""),
                    "job_id": cmd.get("job_id", ""),
                    "status": "simulated_ok",
                }
            )
    return {
        "status": "ok",
        "executed_commands": executed,
        "count": len(executed),
    }
