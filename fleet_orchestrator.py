"""
fleet_orchestrator.py
---------------------
End-to-end entry point that connects the cuOpt logistics solver to the running
FleetMind Isaac Sim.

Pipeline:
    GET /state              (live_state_adapter.fetch_state)
      -> Node/Vehicle/Job   (live_state_adapter.build_nodes_vehicles_jobs)
      -> WarehouseGraph     (warehouse_graph.build_warehouse_graph)
      -> CuOptInput         (cuopt_adapter.build_cuopt_input)
      -> CuOptOutput        (MockCuOptAdapter / HttpCuOptAdapter .solve)
      -> MissionPlan        (mission_translator.translate_solution_to_mission)
      -> POST /mission       (isaac_dispatch.dispatch_mission)

Usage:
    python fleet_orchestrator.py --dry-run
    python fleet_orchestrator.py --objective min_makespan
    python fleet_orchestrator.py --jobs jobs.json
    python fleet_orchestrator.py --base-url http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a plain script from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cbs_integration import cbs_dispatch_order, deconflict_mission  # noqa: E402
from cuopt_adapter import HttpCuOptAdapter, MockCuOptAdapter, build_cuopt_input  # noqa: E402
from env_config import FleetConfig, load_fleet_config  # noqa: E402
from isaac_dispatch import dispatch_mission  # noqa: E402
from live_state_adapter import build_nodes_vehicles_jobs, fetch_state  # noqa: E402
from mission_translator import translate_solution_to_mission  # noqa: E402
from warehouse_graph import build_graph_from_state, build_warehouse_graph  # noqa: E402


def _build_solver(cfg: FleetConfig):
    """Use the real NVIDIA cuOpt service when configured, else the local mock."""
    endpoint = os.environ.get(cfg.cuopt.endpoint_env, "").strip()
    if endpoint:
        return HttpCuOptAdapter(
            endpoint=endpoint,
            api_key=os.environ.get(cfg.cuopt.api_key_env, ""),
        )
    return MockCuOptAdapter()


def _build_graph(state: dict, nodes, cfg: FleetConfig):
    """Prefer the real Isaac nav graph; fall back to euclidean when unavailable."""
    if cfg.graph.use_nav_graph:
        try:
            return build_graph_from_state(state, nodes), "nav_graph"
        except ValueError:
            if not cfg.graph.fallback_to_euclidean:
                raise
    return build_warehouse_graph(nodes), "euclidean"


def _order_dispatch(mission_dict: dict, order: list[str]) -> dict:
    """Reorder vehicle_commands by a CBS-implied dispatch order (stable)."""
    vc = mission_dict.get("vehicle_commands") or {}
    ordered = {v: vc[v] for v in order if v in vc}
    for v, cmds in vc.items():  # keep any not covered by CBS order
        ordered.setdefault(v, cmds)
    mission_dict["vehicle_commands"] = ordered
    return mission_dict


def run(
    base_url: str | None = None,
    objective: str | None = None,
    job_spec: list[dict[str, str]] | None = None,
    *,
    config_path: str | None = None,
    include_busy: bool = False,
    dry_run: bool = False,
    execute: str = "mission",
) -> dict:
    cfg = load_fleet_config(config_path)
    base_url = base_url or cfg.base_url
    objective = objective or cfg.cuopt.objective

    state = fetch_state(base_url, timeout_s=cfg.http_timeout_s)
    nodes, vehicles, jobs = build_nodes_vehicles_jobs(
        state,
        job_spec,
        include_busy=include_busy,
        vehicle_speed_mps=cfg.vehicles.speed_mps,
        vehicle_capacity=cfg.vehicles.capacity,
        allow_delivered_fallback=cfg.jobs.allow_delivered_fallback,
    )

    graph, graph_kind = _build_graph(state, nodes, cfg)
    cuopt_input = build_cuopt_input(
        graph=graph, vehicles=vehicles, jobs=jobs, objective=objective
    )

    solution = _build_solver(cfg).solve(cuopt_input)
    mission = translate_solution_to_mission(
        mission_id="mission_live_isaac",
        objective=objective,
        nodes=nodes,
        solution=solution,
    )
    mission_dict = mission.to_dict()

    # Conflict-Based Search over the real nav grid (deconfliction + ordering).
    cbs_summary: dict = {"status": "skipped"}
    if cfg.cbs.enabled:
        cbs_result = deconflict_mission(
            mission_dict, state,
            max_expansions=cfg.cbs.max_expansions,
            clearance_m=cfg.cbs.clearance_m,
            inflate_obstacles=cfg.cbs.inflate_obstacles,
        )
        mission_dict = _order_dispatch(mission_dict, cbs_dispatch_order(cbs_result))
        cbs_summary = {
            "status": cbs_result.get("status"),
            "conflicts_resolved": cbs_result.get("conflicts_resolved", 0),
            "unresolved_conflict": cbs_result.get("unresolved_conflict"),
            "dispatch_order": list((mission_dict.get("vehicle_commands") or {}).keys()),
        }

    # Execution: faithful CBS stepping (opt-in) or semantic /mission dispatch.
    if execute == "cbs" and not dry_run and cfg.cbs.enabled:
        from cbs_executor import execute_cbs_paths

        dispatch = execute_cbs_paths(cbs_result, state, base_url)
    else:
        dispatch = dispatch_mission(mission_dict, base_url, dry_run=dry_run, timeout_s=cfg.http_timeout_s)

    return {
        "solver_status": solution.status,
        "solver_errors": solution.errors,
        "graph_kind": graph_kind,
        "cbs": cbs_summary,
        "jobs": [{"pallet": j.pallet_id, "zone": j.delivery_node_id} for j in jobs],
        "vehicles": [v.id for v in vehicles],
        "mission_plan": mission_dict,
        "dispatch": dispatch,
    }


def _load_job_spec(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("--jobs file must contain a JSON array of {pallet, zone} objects")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Connect cuOpt planning to the live FleetMind Isaac Sim.")
    parser.add_argument("--config", help="Path to fleet_config.json (defaults to repo copy)")
    parser.add_argument("--base-url", default=None, help="Override FleetMind bridge base URL")
    parser.add_argument(
        "--objective",
        default=None,
        choices=["min_distance", "min_time", "min_makespan"],
        help="cuOpt optimization objective (default from config)",
    )
    parser.add_argument("--jobs", help="Path to a JSON array of {pallet, zone} job overrides")
    parser.add_argument("--include-busy", action="store_true", help="Also plan for non-idle forklifts")
    parser.add_argument(
        "--execute",
        default="mission",
        choices=["mission", "cbs"],
        help="Execution mode: 'mission' (semantic /mission dispatch) or 'cbs' (step CBS grid paths via /goto)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute + print steps without POSTing")
    args = parser.parse_args(argv)

    job_spec = _load_job_spec(args.jobs) if args.jobs else None

    try:
        result = run(
            base_url=args.base_url,
            objective=args.objective,
            job_spec=job_spec,
            config_path=args.config,
            include_busy=args.include_busy,
            dry_run=args.dry_run,
            execute=args.execute,
        )
    except Exception as exc:  # surface a clean message to the CLI
        print(f"[fleet_orchestrator] ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result["dispatch"]["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
