from __future__ import annotations

from typing import Any


def execute_mission_on_isaac(
    mission_plan: dict[str, Any],
    base_url: str = "http://localhost:8080",
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Dispatch a solved mission plan to the running FleetMind Isaac Sim.

    Thin wrapper over ``isaac_dispatch.dispatch_mission`` so the planner loop can
    drive the real sim through the same execution seam as the dry run.
    """
    from isaac_dispatch import dispatch_mission

    return dispatch_mission(mission_plan, base_url, dry_run=dry_run)


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
