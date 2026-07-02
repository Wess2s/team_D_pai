"""
cbs_integration.py
------------------
Wires the generic Conflict-Based Search planner (``cbs_planner.plan_checkpoint_cbs``)
onto the *real* Isaac nav graph.

A solved cuOpt ``MissionPlan`` gives, per forklift, an ordered list of physical
stops (start -> pickup -> delivery -> ...). This module snaps each stop to its
nearest real nav-graph node to obtain per-agent **grid checkpoints**, then runs
CBS over the real grid (via ``build_grid_graph_from_state``) so vertex/edge
conflicts are deconflicted on the actual aisles — with blocked staging zones and
optional human-occupancy windows honoured.

Output is the CBS result (``agent_paths`` = timed grid-node sequences) plus a
per-agent ``actions`` map (which grid checkpoint is a pick/drop and its target),
which the optional ``cbs_executor`` can step through with ``POST /goto``.
"""
from __future__ import annotations

from typing import Any

from cbs_planner import CBSConfig, plan_checkpoint_cbs
from warehouse_graph import build_grid_graph_from_state, nearest_grid_node


def mission_to_grid_checkpoints(
    mission_plan: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, str]]]]:
    """Snap each vehicle's mission stops to real nav-graph nodes.

    Returns ``(checkpoints, actions)`` where
    ``checkpoints[vehicle] = [grid_node, ...]`` (start + each pick/drop stop) and
    ``actions[vehicle]`` is aligned 1:1 with checkpoints, each entry being
    ``{"kind": "start"|"pick"|"drop", "target": <pallet_or_zone>, "node": grid}``.
    """
    forklifts = state.get("forklifts") or {}
    checkpoints: dict[str, list[str]] = {}
    actions: dict[str, list[dict[str, str]]] = {}

    for vehicle_id, commands in (mission_plan.get("vehicle_commands") or {}).items():
        info = forklifts.get(vehicle_id, {})
        start_node = nearest_grid_node(state, float(info.get("x", 0.0)), float(info.get("y", 0.0)))
        nodes = [start_node]
        acts = [{"kind": "start", "target": vehicle_id, "node": start_node}]

        for cmd in commands:
            ctype = cmd.get("command_type")
            if ctype not in {"pickup", "dropoff"}:
                continue
            grid = nearest_grid_node(state, float(cmd.get("x", 0.0)), float(cmd.get("y", 0.0)))
            if ctype == "pickup":
                acts.append({"kind": "pick", "target": cmd.get("pallet_id") or cmd.get("node_id", ""), "node": grid})
            else:
                acts.append({"kind": "drop", "target": cmd.get("node_id", ""), "node": grid})
            nodes.append(grid)

        # A checkpoint list of length 1 (start only) has nothing to deconflict.
        if len(nodes) > 1:
            checkpoints[vehicle_id] = nodes
            actions[vehicle_id] = acts

    return checkpoints, actions


def blocked_grid_nodes_from_state(state: dict[str, Any]) -> list[str]:
    """Grid nodes to treat as blocked: the nav node nearest each blocked zone."""
    blocked: list[str] = []
    for _zid, info in (state.get("zones") or {}).items():
        if info.get("blocked"):
            blocked.append(nearest_grid_node(state, float(info["x"]), float(info["y"])))
    return blocked


def deconflict_mission(
    mission_plan: dict[str, Any],
    state: dict[str, Any],
    *,
    human_occupancy: dict[str, list[tuple[int, int]]] | None = None,
    max_expansions: int = 500,
) -> dict[str, Any]:
    """Run CBS over the real grid for a solved mission.

    Returns ``{status, conflicts_resolved, unresolved_conflict, agent_paths,
    actions, checkpoints}``. On an empty/degenerate plan it returns a trivial
    success so callers can always dispatch.
    """
    checkpoints, actions = mission_to_grid_checkpoints(mission_plan, state)
    if not checkpoints:
        return {
            "status": "success",
            "conflicts_resolved": 0,
            "unresolved_conflict": None,
            "agent_paths": {},
            "actions": actions,
            "checkpoints": checkpoints,
        }

    blocked_nodes = blocked_grid_nodes_from_state(state)
    grid = build_grid_graph_from_state(state, blocked_nodes=set(blocked_nodes))

    result = plan_checkpoint_cbs(
        checkpoints=checkpoints,
        node_index=grid.node_index,
        cost_matrix=grid.distance_matrix,
        blocked_nodes=blocked_nodes,
        human_occupancy=human_occupancy or {},
        config=CBSConfig(max_expansions=max_expansions),
    )
    result["actions"] = actions
    result["checkpoints"] = checkpoints
    return result


def cbs_dispatch_order(cbs_result: dict[str, Any]) -> list[str]:
    """Order vehicles by CBS path completion time (shortest finisher first).

    Used to stagger semantic ``/mission`` dispatch so lower-contention agents go
    first; falls back to input order when no timed paths are available.
    """
    agent_paths = cbs_result.get("agent_paths") or {}
    if not agent_paths:
        return list((cbs_result.get("checkpoints") or {}).keys())

    def finish_time(vid: str) -> int:
        path = agent_paths.get(vid) or []
        return path[-1][1] if path else 0

    return sorted(agent_paths.keys(), key=finish_time)
