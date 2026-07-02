from __future__ import annotations

import logging
import os
from typing import Any

from cbs_planner import CBSConfig, plan_checkpoint_cbs
from cuopt_adapter import CuOptSolver, HttpCuOptAdapter, MockCuOptAdapter
from mission_translator import translate_solution_to_mission
from replanner import build_replan_request
from route_executor import execute_mission_dry_run
from runtime_config import RuntimeConfig


def _build_solver(config: RuntimeConfig) -> CuOptSolver:
    """Select the real NVIDIA cuOpt service when configured, falling back to
    the local heuristic mock otherwise. Both implement the same
    `CuOptSolver` protocol, so the rest of the pipeline is unaffected."""
    if config.cuopt.enabled and config.cuopt.endpoint:
        api_key = os.environ.get(config.cuopt.api_key_env, "")
        return HttpCuOptAdapter(
            endpoint=config.cuopt.endpoint,
            api_key=api_key,
            poll_timeout_s=config.cuopt.timeout_s,
        )
    return MockCuOptAdapter()


def _checkpoints_from_solution(solution_dict: dict[str, Any]) -> dict[str, list[str]]:
    checkpoints: dict[str, list[str]] = {}
    for route in solution_dict.get("vehicle_routes", []):
        vehicle_id = str(route.get("vehicle_id", ""))
        if not vehicle_id:
            continue
        nodes = [stop.get("node_id", "") for stop in route.get("stops", []) if stop.get("node_id")]
        if nodes:
            checkpoints[vehicle_id] = nodes
    return checkpoints


def run_planner_loop(
    *,
    config: RuntimeConfig,
    cuopt_input: Any,
    nodes: list[Any],
    node_index: dict[str, int],
    cost_matrix: list[list[float]],
    human_occupancy: dict[str, list[tuple[int, int]]],
    mission_id: str,
) -> dict[str, Any]:
    logger = logging.getLogger("physical_ai.planner")
    logger.setLevel(config.logging.level)

    solver = _build_solver(config)
    solution = solver.solve(cuopt_input)
    mission = translate_solution_to_mission(
        mission_id=mission_id,
        objective=cuopt_input.objective,
        nodes=nodes,
        solution=solution,
    )

    cbs_output = {"status": "disabled", "agent_paths": {}, "conflicts_resolved": 0, "unresolved_conflict": None}
    if config.planner.enable_cbs:
        cbs_output = plan_checkpoint_cbs(
            checkpoints=_checkpoints_from_solution(solution.to_dict()),
            node_index=node_index,
            cost_matrix=cost_matrix,
            blocked_nodes=cuopt_input.constraints.blocked_nodes,
            blocked_edges=cuopt_input.constraints.blocked_edges,
            human_occupancy=human_occupancy,
            config=CBSConfig(max_expansions=config.planner.cbs_max_expansions),
        )
        logger.info("CBS status=%s conflicts=%s", cbs_output.get("status"), cbs_output.get("conflicts_resolved"))

    execution = execute_mission_dry_run(mission.to_dict())

    replan_payload = None
    if config.planner.enable_replan and cbs_output.get("status") == "error":
        replan_payload = build_replan_request(
            cuopt_input=cuopt_input.to_dict(),
            critical_events=[
                {
                    "event_type": "aisle_block",
                    "details": str(cbs_output.get("unresolved_conflict", {})),
                }
            ],
        )

    return {
        "cuopt_output": solution.to_dict(),
        "mission_plan": mission.to_dict(),
        "cbs_output": cbs_output,
        "execution_dry_run": execution,
        "replan_request": replan_payload,
    }
