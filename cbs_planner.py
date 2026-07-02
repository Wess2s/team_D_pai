from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
from typing import Any


@dataclass(frozen=True)
class CBSConstraint:
    agent_id: str
    time_step: int
    node_id: str = ""
    edge: tuple[str, str] | None = None


@dataclass
class _SearchNode:
    total_cost: int
    tie_break: int
    constraints: list[CBSConstraint]
    paths: dict[str, list[tuple[str, int]]]


@dataclass
class CBSConfig:
    max_neighbors: int = 4
    max_time_steps: int = 250
    goal_hold_steps: int = 2
    max_expansions: int = 300
    use_bypass: bool = True


def _build_sparse_adjacency(
    node_ids: list[str],
    node_index: dict[str, int],
    cost_matrix: list[list[float]],
    max_neighbors: int,
    blocked_nodes: set[str],
    blocked_edges: set[tuple[str, str]],
) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for node_id in node_ids:
        if node_id in blocked_nodes:
            adjacency[node_id] = []
            continue

        i = node_index[node_id]
        candidates: list[tuple[float, str]] = []
        for other_id in node_ids:
            if other_id == node_id or other_id in blocked_nodes:
                continue
            if (node_id, other_id) in blocked_edges:
                continue
            j = node_index[other_id]
            c = cost_matrix[i][j]
            if math.isinf(c):
                continue
            candidates.append((c, other_id))
        candidates.sort(key=lambda x: x[0])
        adjacency[node_id] = [other_id for _, other_id in candidates[:max_neighbors]]
    return adjacency


def _violates_constraint(
    agent_id: str,
    constraints: list[CBSConstraint],
    to_node: str,
    from_node: str,
    time_step: int,
) -> bool:
    for constraint in constraints:
        if constraint.agent_id != agent_id:
            continue
        if constraint.time_step != time_step:
            continue
        if constraint.node_id and constraint.node_id == to_node:
            return True
        if constraint.edge is not None and constraint.edge == (from_node, to_node):
            return True
    return False


def _violates_human_occupancy(to_node: str, time_step: int, human_occupancy: dict[str, list[tuple[int, int]]]) -> bool:
    windows = human_occupancy.get(to_node, [])
    for start_t, end_t in windows:
        if start_t <= time_step <= end_t:
            return True
    return False


def _low_level_a_star_waypoints(
    agent_id: str,
    waypoints: list[str],
    adjacency: dict[str, list[str]],
    node_index: dict[str, int],
    cost_matrix: list[list[float]],
    constraints: list[CBSConstraint],
    human_occupancy: dict[str, list[tuple[int, int]]],
    max_time_steps: int,
    goal_hold_steps: int,
) -> list[tuple[str, int]] | None:
    """Full-horizon A* that visits every waypoint of an agent's mission in
    order within a single search, instead of resolving one checkpoint stage
    at a time. This removes the artificial synchronization barrier where an
    agent that already reached its next stop still has to wait for every
    other agent to finish their current stage before anyone advances.

    Each waypoint (job pickup/delivery/charge/etc.) must be occupied for
    `goal_hold_steps` consecutive steps before the agent is considered to
    have serviced it and can move on to the next one, mirroring real
    load/unload dwell time.
    """
    if not waypoints:
        return None

    start_node = waypoints[0]
    final_wp_idx = len(waypoints) - 1

    if final_wp_idx == 0:
        return [(start_node, 0)]

    def heuristic(node: str, wp_idx: int) -> float:
        target_idx = min(wp_idx, final_wp_idx)
        return cost_matrix[node_index[node]][node_index[waypoints[target_idx]]]

    # State: (node, wp_idx, hold_count, time_step). wp_idx = index of the
    # waypoint the agent is currently trying to reach and hold at.
    start_state = (start_node, 1, 0, 0)
    open_heap: list[tuple[float, int, int, tuple[str, int, int, int]]] = []
    tie_break = itertools.count()
    heapq.heappush(open_heap, (0.0, 0, next(tie_break), start_state))

    parent: dict[tuple[str, int, int, int], tuple[str, int, int, int] | None] = {start_state: None}
    g_score: dict[tuple[str, int, int, int], int] = {start_state: 0}

    while open_heap:
        _, g, _, state = heapq.heappop(open_heap)
        node, wp_idx, hold_count, t = state

        if g > g_score.get(state, 10**9):
            continue

        if wp_idx > final_wp_idx:
            path: list[tuple[str, int]] = []
            cur: tuple[str, int, int, int] | None = state
            while cur is not None:
                path.append((cur[0], cur[3]))
                cur = parent[cur]
            path.reverse()
            return path

        if t >= max_time_steps:
            continue

        target_node = waypoints[wp_idx]
        successors = [node] + adjacency.get(node, [])
        for next_node in successors:
            next_time = t + 1
            if _violates_human_occupancy(next_node, next_time, human_occupancy):
                continue
            if _violates_constraint(agent_id, constraints, next_node, node, next_time):
                continue

            if next_node == target_node:
                reached_hold = hold_count + 1 if next_node == node else 1
            else:
                reached_hold = 0

            if reached_hold >= goal_hold_steps:
                next_wp_idx = wp_idx + 1
                next_hold = 0
            else:
                next_wp_idx = wp_idx
                next_hold = reached_hold

            next_state = (next_node, next_wp_idx, next_hold, next_time)
            tentative_g = g + 1
            if tentative_g >= g_score.get(next_state, 10**9):
                continue

            g_score[next_state] = tentative_g
            parent[next_state] = state
            h = heuristic(next_node, next_wp_idx)
            heapq.heappush(open_heap, (tentative_g + h, tentative_g, next(tie_break), next_state))

    return None


def _node_at_time(path: list[tuple[str, int]], time_step: int) -> str:
    best_node = path[0][0]
    for node_id, t in path:
        if t > time_step:
            break
        best_node = node_id
    return best_node


def _detect_first_conflict(
    paths: dict[str, list[tuple[str, int]]],
    multi_capacity_nodes: set[str] | None = None,
) -> dict[str, Any] | None:
    if not paths:
        return None
    multi_capacity_nodes = multi_capacity_nodes or set()
    max_t = max(path[-1][1] for path in paths.values())
    agent_ids = sorted(paths.keys())

    for t in range(0, max_t + 1):
        # Vertex conflicts (allow shared depot at initial time).
        if t > 0:
            occupancy: dict[str, str] = {}
            for agent_id in agent_ids:
                node_t = _node_at_time(paths[agent_id], t)
                if node_t in multi_capacity_nodes:
                    continue
                if node_t in occupancy:
                    return {
                        "type": "vertex",
                        "time": t,
                        "node": node_t,
                        "agents": [occupancy[node_t], agent_id],
                    }
                occupancy[node_t] = agent_id

        if t == 0:
            continue

        # Edge swap conflicts.
        for i, a1 in enumerate(agent_ids):
            for a2 in agent_ids[i + 1 :]:
                a1_prev = _node_at_time(paths[a1], t - 1)
                a1_curr = _node_at_time(paths[a1], t)
                a2_prev = _node_at_time(paths[a2], t - 1)
                a2_curr = _node_at_time(paths[a2], t)
                if a1_prev == a2_curr and a2_prev == a1_curr and a1_prev != a1_curr:
                    return {
                        "type": "edge",
                        "time": t,
                        "edge_a1": (a1_prev, a1_curr),
                        "edge_a2": (a2_prev, a2_curr),
                        "agents": [a1, a2],
                    }
    return None


def _sum_path_cost(paths: dict[str, list[tuple[str, int]]]) -> int:
    return sum(max(len(path) - 1, 0) for path in paths.values())


def _conflicts_match(a: dict[str, Any] | None, b: dict[str, Any]) -> bool:
    if a is None:
        return False
    return a.get("time") == b.get("time") and set(a.get("agents", [])) == set(b.get("agents", []))


def plan_checkpoint_cbs(
    checkpoints: dict[str, list[str]],
    node_index: dict[str, int],
    cost_matrix: list[list[float]],
    blocked_nodes: list[str] | None = None,
    blocked_edges: list[tuple[str, str]] | None = None,
    human_occupancy: dict[str, list[tuple[int, int]]] | None = None,
    config: CBSConfig | None = None,
) -> dict[str, Any]:
    """Full-horizon Conflict-Based Search over each agent's complete
    checkpoint sequence (all pickups/deliveries/charging stops), not one
    synchronized stage at a time.

    High-level search: every constraint-tree node holds one complete path
    per agent; the first vertex/edge conflict is resolved by branching into
    two children, each adding a constraint to one of the conflicting agents
    and replanning just that agent's full path. A standard CBS "bypass"
    optimization is applied first: if replanning under the new constraint
    produces a same-cost-or-cheaper path that no longer reproduces the same
    conflict, it is adopted in place without branching, which keeps the
    search tree small.
    """
    cfg = config or CBSConfig()
    blocked_nodes_set = set(blocked_nodes or [])
    blocked_edges_set = set(blocked_edges or [])
    blocked_edges_set |= {(b, a) for (a, b) in blocked_edges_set}
    human_occupancy = human_occupancy or {}

    if not checkpoints:
        return {
            "status": "success",
            "agent_paths": {},
            "conflicts_resolved": 0,
            "unresolved_conflict": None,
        }

    node_ids = list(node_index.keys())
    multi_capacity_nodes = {
        node_id for node_id in node_ids if "charging" in node_id.lower() or "depot" in node_id.lower()
    }
    checkpoint_counts: dict[str, int] = {}
    for stops in checkpoints.values():
        for node_id in stops:
            checkpoint_counts[node_id] = checkpoint_counts.get(node_id, 0) + 1
    for node_id, count in checkpoint_counts.items():
        if count > 1:
            multi_capacity_nodes.add(node_id)

    adjacency = _build_sparse_adjacency(
        node_ids=node_ids,
        node_index=node_index,
        cost_matrix=cost_matrix,
        max_neighbors=cfg.max_neighbors,
        blocked_nodes=blocked_nodes_set,
        blocked_edges=blocked_edges_set,
    )

    agent_ids = list(checkpoints.keys())

    def plan_agent(agent_id: str, constraints: list[CBSConstraint]) -> list[tuple[str, int]] | None:
        return _low_level_a_star_waypoints(
            agent_id=agent_id,
            waypoints=checkpoints[agent_id],
            adjacency=adjacency,
            node_index=node_index,
            cost_matrix=cost_matrix,
            constraints=constraints,
            human_occupancy=human_occupancy,
            max_time_steps=cfg.max_time_steps,
            goal_hold_steps=cfg.goal_hold_steps,
        )

    root_paths: dict[str, list[tuple[str, int]]] = {}
    for agent_id in agent_ids:
        path = plan_agent(agent_id, [])
        if path is None:
            return {
                "status": "error",
                "reason": f"No feasible full-horizon path for {agent_id}",
                "agent_paths": {},
                "conflicts_resolved": 0,
                "unresolved_conflict": None,
            }
        root_paths[agent_id] = path

    seq = itertools.count()
    root = _SearchNode(total_cost=_sum_path_cost(root_paths), tie_break=0, constraints=[], paths=root_paths)
    open_nodes: list[tuple[int, int, int, _SearchNode]] = [(root.total_cost, root.tie_break, next(seq), root)]

    conflicts_resolved = 0
    expansions = 0
    best_effort_paths = root_paths
    unresolved_conflict: dict[str, Any] | None = None

    while open_nodes and expansions < cfg.max_expansions:
        _, _, _, current = heapq.heappop(open_nodes)
        expansions += 1
        best_effort_paths = current.paths

        conflict = _detect_first_conflict(current.paths, multi_capacity_nodes=multi_capacity_nodes)
        if conflict is None:
            return {
                "status": "success",
                "agent_paths": current.paths,
                "conflicts_resolved": conflicts_resolved,
                "unresolved_conflict": None,
            }

        conflicts_resolved += 1
        unresolved_conflict = conflict
        a1, a2 = conflict["agents"]

        bypassed = False
        children: list[_SearchNode] = []

        for constrained_agent in (a1, a2):
            new_constraints = list(current.constraints)
            if conflict["type"] == "vertex":
                new_constraints.append(
                    CBSConstraint(agent_id=constrained_agent, time_step=conflict["time"], node_id=conflict["node"])
                )
            else:
                edge = conflict["edge_a1"] if constrained_agent == a1 else conflict["edge_a2"]
                new_constraints.append(
                    CBSConstraint(agent_id=constrained_agent, time_step=conflict["time"], edge=edge)
                )

            replanned = plan_agent(constrained_agent, new_constraints)
            if replanned is None:
                continue

            old_cost = max(len(current.paths[constrained_agent]) - 1, 0)
            new_cost = max(len(replanned) - 1, 0)

            if cfg.use_bypass and not bypassed and new_cost <= old_cost:
                bypass_paths = dict(current.paths)
                bypass_paths[constrained_agent] = replanned
                bypass_conflict = _detect_first_conflict(bypass_paths, multi_capacity_nodes=multi_capacity_nodes)
                if not _conflicts_match(bypass_conflict, conflict):
                    # Same-cost reroute that no longer reproduces this exact
                    # conflict: keep it without permanently constraining the
                    # agent or growing the search tree.
                    current.paths = bypass_paths
                    current.total_cost = _sum_path_cost(bypass_paths)
                    heapq.heappush(open_nodes, (current.total_cost, current.tie_break, next(seq), current))
                    bypassed = True
                    continue

            new_paths = dict(current.paths)
            new_paths[constrained_agent] = replanned
            children.append(
                _SearchNode(
                    total_cost=_sum_path_cost(new_paths),
                    tie_break=expansions,
                    constraints=new_constraints,
                    paths=new_paths,
                )
            )

        if bypassed:
            continue

        for child in children:
            heapq.heappush(open_nodes, (child.total_cost, child.tie_break, next(seq), child))

    return {
        "status": "error",
        "reason": "CBS did not converge within expansion limit",
        "agent_paths": best_effort_paths,
        "conflicts_resolved": conflicts_resolved,
        "unresolved_conflict": unresolved_conflict,
    }
