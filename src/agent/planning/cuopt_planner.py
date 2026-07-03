"""
cuOpt planner — fleet task assignment + routing for pallet moves.

Given the live fleet (forklifts + positions) and a set of pallet→zone move tasks, this
decides *which forklift does which pallets, in what order* — a capacitated pickup-and-
delivery VRP (each forklift carries one pallet at a time). It prefers NVIDIA cuOpt when a
self-hosted cuOpt server is reachable (`CUOPT_URL`), and otherwise falls back to a local
solver (greedy cheapest-insertion + 2-opt) that returns the identical result shape, so the
rest of the stack never knows the difference.

Distances come from the shared Roadmap (real aisle travel cost), not straight-line, so the
plan respects the warehouse layout.

Result shape:
    Plan(routes={forklift: [(pallet_id, zone_id), ...]}, total_cost=float, solver="cuopt"|"local")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .roadmap import Roadmap

try:
    import requests
except Exception:  # requests optional off-DGX
    requests = None


# Battery model (mirrors scenes/scene_exec.py). A truck spends BATTERY_DRAIN_PER_M % of
# charge per metre, so its remaining travel range in metres is battery% * RANGE_PER_PCT.
# cuOpt uses that as each vehicle's max route cost, and penalises low charge so the fuller
# truck is preferred — i.e. it optimises travel cost AND preserves fleet battery. The
# drain (0.5 m per 1%, == 2.0%/m in the sim) makes charge fall fast enough that the busiest
# truck in a multi-pallet dispatch ends ~20% — below LOW_BATTERY — so on the NEXT dispatch
# cuOpt holds it back and sends it to charge while the fuller truck takes the work. A lone
# truck clearing EVERY pallet (~57 m) now exceeds its 50 m full range, so cuOpt splits the
# job across both trucks instead of solo-routing one. Busiest split leg (~40 m) < 50 m so no
# truck strands mid-tour.
BATTERY_RANGE_PER_PCT = 0.5     # metres of range per 1% charge (== 1 / 2.0%/m)
BATTERY_PREF_WEIGHT   = 0.4     # fixed-cost penalty per 1% of missing charge
LOW_BATTERY           = 40.0    # below this a truck is held back and sent to recharge


@dataclass
class Plan:
    routes: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    total_cost: float = 0.0
    solver: str = "local"

    def assignments(self) -> list[tuple[str, str, str]]:
        """Flat [(forklift, pallet, zone), ...] in execution order."""
        out = []
        for fk, tasks in self.routes.items():
            for pallet, zone in tasks:
                out.append((fk, pallet, zone))
        return out


def _nearest_zone(rm: Roadmap, pallet_xy, zones: dict) -> str | None:
    best, best_d = None, float("inf")
    pn = rm.nearest(*pallet_xy)
    for zid, z in zones.items():
        if z.get("blocked"):
            continue
        zn = rm.nearest(z["x"], z["y"])
        d = rm.path_length(rm.astar(pn, zn)) or rm.dist(pn, zn)
        if d < best_d:
            best, best_d = zid, d
    return best


def _travel(rm: Roadmap, a_xy, b_xy) -> float:
    an, bn = rm.nearest(*a_xy), rm.nearest(*b_xy)
    p = rm.astar(an, bn)
    return rm.path_length(p) if p else rm.dist(an, bn)


def plan_moves(
    snap: dict,
    rm: Roadmap,
    pallets: list[str] | None = None,
    zones: list[str] | None = None,
) -> Plan:
    """Assign & order pallet→zone moves across the free fleet."""
    forklifts = {
        n: fk for n, fk in snap.get("forklifts", {}).items()
        if fk.get("phase") in ("idle",) and not fk.get("carrying")
    }
    if not forklifts:  # nobody free — let caller handle it
        return Plan({}, 0.0, "none")

    # Battery-aware fleet: hold back trucks too low to safely take work so they recharge.
    # If that would strand the whole fleet, fall back to using them all (best effort).
    charged = {n: fk for n, fk in forklifts.items()
               if fk.get("battery", 100.0) >= LOW_BATTERY}
    if charged:
        forklifts = charged

    all_pallets = snap.get("pallets", {})
    if pallets is None:
        pallets = [pid for pid, p in all_pallets.items()
                   if not p.get("carried_by") and not p.get("delivered")]
    all_zones = snap.get("zones", {})
    zone_ids = zones or [z for z, zd in all_zones.items() if not zd.get("blocked")]
    if not pallets or not zone_ids:
        return Plan({}, 0.0, "none")

    # Each pallet → a staging zone. Spread across the available bays (round-robin over
    # bays ordered by id) so staging fills evenly, which is both realistic and legible.
    tasks: list[tuple[str, str]] = []
    zsub = {z: all_zones[z] for z in zone_ids if z in all_zones}
    ordered_zones = sorted(zsub) or zone_ids
    for i, pid in enumerate(sorted(pallets)):
        if pid not in all_pallets:
            continue
        z = ordered_zones[i % len(ordered_zones)] if ordered_zones else zone_ids[0]
        tasks.append((pid, z))

    if requests is not None and os.getenv("CUOPT_URL"):
        try:
            return _solve_cuopt(snap, rm, forklifts, tasks, all_pallets, all_zones)
        except Exception as exc:  # never crash dispatch; log why we fell back
            import sys
            print(f"[cuopt_planner] cuOpt call failed, using local fallback: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return _solve_local(snap, rm, forklifts, tasks, all_pallets, all_zones)


# --------------------------------------------------------------------------- #
# Local fallback solver — greedy cheapest-insertion + per-vehicle 2-opt.
# --------------------------------------------------------------------------- #
def _solve_local(snap, rm, forklifts, tasks, all_pallets, all_zones) -> Plan:
    routes: dict[str, list[tuple[str, str]]] = {n: [] for n in forklifts}
    # Track each forklift's "cursor" position and accumulated tour cost. We assign to
    # minimise the *makespan* (busiest forklift), so work spreads across the fleet
    # instead of piling onto whoever happens to be marginally closest.
    pos = {n: (fk["x"], fk["y"]) for n, fk in forklifts.items()}
    load = {n: 0.0 for n in forklifts}

    remaining = list(tasks)
    while remaining:
        best = None  # (resulting_makespan, added_cost, forklift, task_index)
        for ti, (pid, zid) in enumerate(remaining):
            p = all_pallets[pid]
            z = all_zones[zid]
            for n in forklifts:
                to_pick = _travel(rm, pos[n], (p["x"], p["y"]))
                pick_to_drop = _travel(rm, (p["x"], p["y"]), (z["x"], z["y"]))
                added = to_pick + pick_to_drop
                resulting = load[n] + added
                key = (resulting, added)
                if best is None or key < best[0]:
                    best = (key, n, ti)
        (_key, fk, ti) = best
        pid, zid = remaining.pop(ti)
        z = all_zones[zid]
        p = all_pallets[pid]
        routes[fk].append((pid, zid))
        load[fk] += _travel(rm, pos[fk], (p["x"], p["y"])) + \
            _travel(rm, (p["x"], p["y"]), (z["x"], z["y"]))
        pos[fk] = (z["x"], z["y"])   # forklift now sits at the drop zone

    # Order each forklift's tasks by nearest-neighbour then a 2-opt pass on pick points.
    for n in routes:
        routes[n] = _two_opt(rm, snap["forklifts"][n], routes[n], all_pallets, all_zones)

    total = _plan_cost(rm, snap, routes, all_pallets, all_zones)
    routes = {n: r for n, r in routes.items() if r}
    return Plan(routes, total, "local")


def _two_opt(rm, fk, tasks, all_pallets, all_zones):
    """Local 2-opt improvement on a single forklift's task order (by pickup point)."""
    if len(tasks) < 3:
        return tasks
    def cost(order):
        cur = (fk["x"], fk["y"])
        c = 0.0
        for pid, zid in order:
            p, z = all_pallets[pid], all_zones[zid]
            c += _travel(rm, cur, (p["x"], p["y"])) + _travel(rm, (p["x"], p["y"]), (z["x"], z["y"]))
            cur = (z["x"], z["y"])
        return c
    best = tasks[:]
    best_c = cost(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cc = cost(cand)
                if cc < best_c - 1e-6:
                    best, best_c, improved = cand, cc, True
    return best


# --------------------------------------------------------------------------- #
# cuOpt self-hosted solver.
# --------------------------------------------------------------------------- #
def _solve_cuopt(snap, rm, forklifts, tasks, all_pallets, all_zones) -> Plan:
    """Format a pickup-delivery VRP for the cuOpt self-hosted server and parse the routes."""
    url = os.getenv("CUOPT_URL", "").rstrip("/")

    fk_names = list(forklifts)
    # Location list: [vehicle starts..., then pickup/delivery pair per task].
    locations: list[tuple[float, float]] = [
        (forklifts[n]["x"], forklifts[n]["y"]) for n in fk_names
    ]
    pickup_idx, delivery_idx, demands = [], [], []
    for pid, zid in tasks:
        p, z = all_pallets[pid], all_zones[zid]
        locations.append((p["x"], p["y"]));  pickup_idx.append(len(locations) - 1)
        locations.append((z["x"], z["y"]));  delivery_idx.append(len(locations) - 1)

    # Roadmap-aware cost matrix.
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = _travel(rm, locations[i], locations[j])

    payload = {
        "cost_matrix_data": {"data": {"0": matrix}},
        "task_data": {
            "task_locations": pickup_idx + delivery_idx,
            "demand": [[1] * len(pickup_idx) + [-1] * len(delivery_idx)],
            # cuOpt expects ORDER indices (position within task_locations), not location
            # indices: task_locations = [pickups..., deliveries...], so pickup k is order k
            # and its delivery is order len(pickups)+k.
            "pickup_and_delivery_pairs": [
                [k, len(pickup_idx) + k] for k in range(len(pickup_idx))
            ],
        },
        "fleet_data": {
            "vehicle_locations": [[i, i] for i in range(len(fk_names))],
            "capacities": [[1] * len(fk_names)],
            # Use the whole fleet so the multi-robot coordination (and CBS deconfliction)
            # is exercised, rather than letting cuOpt minimise to a single vehicle.
            "min_vehicles": min(len(fk_names), len(tasks)),
            # A forklift finishes at the drop zone — it does NOT drive back to its start.
            # Without this, cuOpt minimises the round-trip-to-depot and can pick a truck
            # whose home is near the drop over the one actually closest to the pickup.
            "drop_return_trips": [True] * len(fk_names),
            # Battery awareness: cap each truck's route by its remaining range so it is
            # never sent further than its charge allows, and add a fixed penalty that
            # grows as charge drops so cuOpt prefers the fuller truck (maximising the
            # fleet's battery while still minimising travel).
            "vehicle_max_costs": [
                round(forklifts[n].get("battery", 100.0) * BATTERY_RANGE_PER_PCT, 2)
                for n in fk_names
            ],
            "vehicle_fixed_costs": [
                round((100.0 - forklifts[n].get("battery", 100.0)) * BATTERY_PREF_WEIGHT, 2)
                for n in fk_names
            ],
        },
        "solver_config": {"time_limit": 2.0},
    }
    data = _cuopt_solve(url, payload)
    routes = _parse_cuopt(data, fk_names, tasks, pickup_idx)
    if not routes:
        # cuOpt assigned no vehicle (e.g. reported the VRP infeasible). Never silently
        # drop a mission when there are free forklifts and pending tasks — raise so the
        # caller logs it and falls back to the local solver.
        raise RuntimeError("cuOpt returned no vehicle assignments (infeasible)")
    total = _plan_cost(rm, snap, routes, all_pallets, all_zones)
    return Plan(routes, total, "cuopt")


def _cuopt_solve(url: str, payload: dict) -> dict:
    """Submit a VRP to the self-hosted cuOpt REST server and return the solution JSON.

    cuOpt >= 25.x is async: POST /cuopt/request returns a reqId, then the solution is
    polled from GET /cuopt/solution/{reqId}. Older builds returned the solution inline,
    so we handle both. CLIENT-VERSION: custom skips the client/server version check.
    """
    import time

    headers = {"CLIENT-VERSION": "custom"}
    resp = requests.post(f"{url}/cuopt/request", json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Inline solution (older builds) — already has the solver response.
    if "response" in data:
        return data

    req_id = data.get("reqId") or data.get("id")
    if not req_id:
        raise RuntimeError(f"cuOpt returned no reqId: {data}")

    # Poll for the completed solution.
    deadline = time.time() + 20.0
    while time.time() < deadline:
        sol = requests.get(f"{url}/cuopt/solution/{req_id}", headers=headers, timeout=15)
        sol.raise_for_status()
        body = sol.json()
        if "response" in body:
            return body
        time.sleep(0.25)
    raise TimeoutError(f"cuOpt solution {req_id} did not complete in time")


def _parse_cuopt(data, fk_names, tasks, pickup_idx) -> dict[str, list[tuple[str, str]]]:
    """Map cuOpt's per-vehicle location sequence back to (pallet, zone) task order."""
    loc_to_task = {loc: i for i, loc in enumerate(pickup_idx)}
    resp = data.get("response", data).get("solver_response", data.get("response", {}))
    vehicle_data = (resp or {}).get("vehicle_data", {})
    routes: dict[str, list[tuple[str, str]]] = {n: [] for n in fk_names}
    for vkey, vinfo in vehicle_data.items():
        try:
            vi = int(vkey)
        except (TypeError, ValueError):
            vi = fk_names.index(vkey) if vkey in fk_names else 0
        name = fk_names[vi] if vi < len(fk_names) else fk_names[0]
        for loc in vinfo.get("route", []):
            if loc in loc_to_task:
                routes[name].append(tasks[loc_to_task[loc]])
    return {n: r for n, r in routes.items() if r}


def _plan_cost(rm, snap, routes, all_pallets, all_zones) -> float:
    total = 0.0
    for n, tasks in routes.items():
        fk = snap["forklifts"][n]
        cur = (fk["x"], fk["y"])
        for pid, zid in tasks:
            p, z = all_pallets[pid], all_zones[zid]
            total += _travel(rm, cur, (p["x"], p["y"]))
            total += _travel(rm, (p["x"], p["y"]), (z["x"], z["y"]))
            cur = (z["x"], z["y"])
    return round(total, 2)
