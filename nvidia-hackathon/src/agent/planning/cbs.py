"""
Conflict-Based Search (CBS) — proactive multi-agent path deconfliction.

This is the "CBS" in Geo-CBS: given each forklift a start node and a goal node on the
shared roadmap, it plans time-parameterised paths that are guaranteed not to collide —
no two forklifts occupy the same node at the same tick (vertex conflict) and no two swap
across an edge (edge conflict).

Two levels, the classic Sharon et al. formulation:
  * low level  — space-time A* for a single agent honouring a set of constraints
                 (agent must NOT be at node v at time t).
  * high level — best-first search over a constraint tree: find the first conflict between
                 the current paths, branch by forbidding it for one agent or the other,
                 and repeat until a conflict-free set is found.

Pure standard library. Node-expansion caps keep it real-time-safe for a small fleet; if the
budget is exhausted it returns the best (possibly still-staggered) set it has, which the
dispatcher then releases with time offsets — motion stays safe either way.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

from .roadmap import Roadmap

# A constraint forbids `agent` from occupying `node` at integer time `t`.
Constraint = tuple[str, str, int]
# A conflict: two agents meeting at a node (vertex) or swapping along an edge.


@dataclass(order=True)
class _CTNode:
    """A node in the high-level constraint tree (ordered by total cost for the heap)."""
    cost: float
    seq: int
    constraints: dict[str, set[Constraint]] = field(compare=False, default_factory=dict)
    paths: dict[str, list[str]] = field(compare=False, default_factory=dict)


def _space_time_astar(
    rm: Roadmap,
    start: str,
    goal: str,
    constraints: set[Constraint],
    agent: str,
    max_t: int,
    max_expansions: int = 20000,
) -> list[str] | None:
    """Shortest timed path start->goal avoiding (agent, node, t) constraints.

    Returns a list of node ids indexed by time tick (waiting = repeated node), or None.
    """
    # Constraints relevant to this agent: {(node, t)} forbidden.
    forbidden = {(n, t) for (a, n, t) in constraints if a == agent}
    # Latest time this agent is constrained — must plan at least this far to "settle".
    horizon = max([t for (_n, t) in forbidden], default=0)

    start_state = (start, 0)
    open_heap: list[tuple[float, int, tuple[str, int]]] = []
    tie = itertools.count()
    heapq.heappush(open_heap, (rm.dist(start, goal), next(tie), start_state))
    g = {start_state: 0.0}
    came: dict[tuple[str, int], tuple[str, int]] = {}
    expansions = 0

    while open_heap:
        expansions += 1
        if expansions > max_expansions:
            return None
        _, _, (node, t) = heapq.heappop(open_heap)

        if node == goal and t >= horizon:
            # Reconstruct.
            path = [node]
            state = (node, t)
            while state in came:
                state = came[state]
                path.append(state[0])
            return list(reversed(path))

        if t >= max_t:
            continue

        # Candidate moves: wait, or step to a neighbour.
        for nxt in list(rm.neighbours(node)) + [node]:
            nt = t + 1
            if (nxt, nt) in forbidden:
                continue
            # Edge (swap) conflicts are handled at the high level via vertex constraints
            # on the relevant ticks; here we only enforce vertex occupancy.
            step_cost = rm.dist(node, nxt) if nxt != node else rm.dist(node, goal) * 0.001 + 0.1
            state = (nxt, nt)
            ng = g[(node, t)] + step_cost
            if ng < g.get(state, float("inf")):
                g[state] = ng
                came[state] = (node, t)
                f = ng + rm.dist(nxt, goal)
                heapq.heappush(open_heap, (f, next(tie), state))
    return None


def _first_conflict(paths: dict[str, list[str]]) -> tuple | None:
    """Return the first (vertex or edge) conflict between any two agent paths, or None."""
    agents = list(paths)
    horizon = max((len(p) for p in paths.values()), default=0)

    def at(p: list[str], t: int) -> str:
        return p[t] if t < len(p) else p[-1] if p else ""

    for t in range(horizon):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai, aj = agents[i], agents[j]
                pi, pj = paths[ai], paths[aj]
                # Vertex conflict: same node at the same tick.
                if at(pi, t) and at(pi, t) == at(pj, t):
                    return ("vertex", ai, aj, at(pi, t), t)
                # Edge (swap) conflict: they exchange nodes between t and t+1.
                if t + 1 < horizon:
                    if at(pi, t) == at(pj, t + 1) and at(pj, t) == at(pi, t + 1) \
                            and at(pi, t) != at(pi, t + 1):
                        return ("edge", ai, aj, (at(pi, t), at(pi, t + 1)), t)
    return None


@dataclass
class CBSResult:
    paths: dict[str, list[str]]          # agent -> timed node path
    conflicts_found: int                 # conflicts encountered while solving
    resolved: bool                       # True if a fully conflict-free set was found
    expansions: int                      # high-level tree nodes expanded


def solve(
    rm: Roadmap,
    agents: dict[str, tuple[str, str]],   # agent -> (start_node, goal_node)
    max_high_level: int = 200,
) -> CBSResult:
    """Plan conflict-free paths for all agents. Falls back gracefully if the budget runs out."""
    # Drop agents with no goal or unreachable start/goal up front.
    active = {a: (s, gg) for a, (s, gg) in agents.items()
              if s in rm.nodes and gg in rm.nodes}
    if not active:
        return CBSResult({}, 0, True, 0)

    max_t = 4 * (len(rm.nodes) ** 0.5 + len(active)) + 30
    max_t = int(max_t)

    def plan_all(constraints: dict[str, set[Constraint]]) -> dict[str, list[str]] | None:
        out: dict[str, list[str]] = {}
        for a, (s, gg) in active.items():
            p = _space_time_astar(rm, s, gg, constraints.get(a, set()), a, max_t)
            if p is None:
                return None
            out[a] = p
        return out

    seq = itertools.count()
    root_constraints: dict[str, set[Constraint]] = {a: set() for a in active}
    root_paths = plan_all(root_constraints)
    if root_paths is None:
        # Cannot even plan individually — hand back straight A* paths, unresolved.
        return CBSResult({a: rm.astar(s, gg) for a, (s, gg) in active.items()}, 0, False, 0)

    root = _CTNode(_total_cost(rm, root_paths), next(seq), root_constraints, root_paths)
    open_tree: list[_CTNode] = [root]
    conflicts_found = 0
    expansions = 0

    while open_tree and expansions < max_high_level:
        expansions += 1
        node = heapq.heappop(open_tree)
        conflict = _first_conflict(node.paths)
        if conflict is None:
            return CBSResult(node.paths, conflicts_found, True, expansions)
        conflicts_found += 1

        kind, a1, a2, where, t = conflict
        # Branch: add a constraint to each agent in turn and replan just that agent.
        for agent, forbid in _branch_constraints(kind, a1, a2, where, t):
            child_constraints = {a: set(cs) for a, cs in node.constraints.items()}
            child_constraints.setdefault(agent, set()).update(forbid)
            new_path = _space_time_astar(
                rm, active[agent][0], active[agent][1],
                child_constraints[agent], agent, max_t,
            )
            if new_path is None:
                continue
            child_paths = dict(node.paths)
            child_paths[agent] = new_path
            child = _CTNode(_total_cost(rm, child_paths), next(seq),
                            child_constraints, child_paths)
            heapq.heappush(open_tree, child)

    # Budget exhausted — return the best (root/last) set, flagged unresolved so the
    # dispatcher staggers releases to stay safe.
    best = node.paths if "node" in dir() else root_paths
    return CBSResult(best, conflicts_found, False, expansions)


def _branch_constraints(kind, a1, a2, where, t):
    """Yield (agent, {constraints}) pairs to explore for a conflict."""
    if kind == "vertex":
        node = where
        yield a1, {(a1, node, t)}
        yield a2, {(a2, node, t)}
    else:  # edge swap (u->v for a1, v->u for a2) between t and t+1
        u, v = where
        # Forbid a1 from reaching v at t+1, or a2 from reaching u at t+1.
        yield a1, {(a1, v, t + 1)}
        yield a2, {(a2, u, t + 1)}


def _total_cost(rm: Roadmap, paths: dict[str, list[str]]) -> float:
    return sum(rm.path_length(p) + 0.01 * len(p) for p in paths.values())
