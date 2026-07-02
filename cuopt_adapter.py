from __future__ import annotations

import json
import time
from dataclasses import asdict
from math import isinf
import heapq
from typing import Protocol
from urllib import request

from logistics_models import CuOptInput, CuOptOutput, Job, RouteStop, Vehicle, VehicleRoute
from warehouse_graph import WarehouseGraph


class CuOptSolver(Protocol):
    def solve(self, cuopt_input: CuOptInput) -> CuOptOutput:
        ...


def build_cuopt_input(graph: WarehouseGraph, vehicles: list[Vehicle], jobs: list[Job], objective: str = "min_distance") -> CuOptInput:
    return CuOptInput(
        nodes=graph.nodes,
        vehicles=vehicles,
        jobs=jobs,
        cost_matrix=graph.distance_matrix,
        objective=objective,  # type: ignore[arg-type]
        node_index=graph.node_index,
    )


class MockCuOptAdapter:
    """Dependency-free VRPTW heuristic that mimics the construction +
    improvement pattern of a real solver like NVIDIA cuOpt, while keeping the
    same in/out contract so it is a drop-in for `HttpCuOptAdapter`.

    Pipeline:
      1. Regret-2 insertion construction, processed in priority tiers so
         urgent jobs are still locked in first, but ties within a tier are
         broken by "how much worse off we'd be if we didn't act now"
         (regret) instead of arbitrary id order or pure nearest-vehicle.
      2. Local search improvement: relocate / swap / or-opt moves over the
         (job -> vehicle) assignment and per-vehicle visiting order, each
         move re-validated for time windows, battery and blocked nodes
         through the same simulator used during construction, keeping every
         accepted move feasible by construction.

    This is intentionally capped (`LOCAL_SEARCH_JOB_LIMIT`, rebuild budget)
    so it stays fast for hackathon-scale fleets; production-scale problems
    should go through `HttpCuOptAdapter` to the real GPU solver instead.
    """

    LOCAL_SEARCH_JOB_LIMIT = 60
    REGRET_CONSTRUCTION_JOB_LIMIT = 30
    LOCAL_SEARCH_MAX_ROUNDS = 25
    MAX_REBUILDS = 6000

    def solve(self, cuopt_input: CuOptInput) -> CuOptOutput:
        if not cuopt_input.vehicles:
            return CuOptOutput(
                status="error",
                vehicle_routes=[],
                total_cost=0.0,
                errors=["No vehicles available"],
                solver_name="mock-cuopt",
            )

        vehicle_lookup = {vehicle.id: vehicle for vehicle in cuopt_input.vehicles}
        blocked_nodes = set(cuopt_input.constraints.blocked_nodes)
        blocked_edges = set(cuopt_input.constraints.blocked_edges)
        blocked_edges |= {(b, a) for (a, b) in blocked_edges}
        node_ids = [node.id for node in cuopt_input.nodes]
        shortest_cost_cache: dict[tuple[str, str], float] = {}

        def direct_cost(from_node_id: str, to_node_id: str) -> float:
            i = cuopt_input.node_index[from_node_id]
            j = cuopt_input.node_index[to_node_id]
            return cuopt_input.cost_matrix[i][j]

        def cost_between(from_node_id: str, to_node_id: str) -> float:
            key = (from_node_id, to_node_id)
            if key in shortest_cost_cache:
                return shortest_cost_cache[key]
            if from_node_id == to_node_id:
                return 0.0
            if from_node_id in blocked_nodes or to_node_id in blocked_nodes:
                shortest_cost_cache[key] = float("inf")
                return float("inf")

            pq: list[tuple[float, str]] = [(0.0, from_node_id)]
            dist = {node_id: float("inf") for node_id in node_ids}
            dist[from_node_id] = 0.0

            while pq:
                current_cost, node_id = heapq.heappop(pq)
                if node_id == to_node_id:
                    shortest_cost_cache[key] = current_cost
                    return current_cost
                if current_cost > dist[node_id]:
                    continue
                for neighbor in node_ids:
                    if neighbor == node_id or neighbor in blocked_nodes:
                        continue
                    if (node_id, neighbor) in blocked_edges:
                        continue
                    edge_cost = direct_cost(node_id, neighbor)
                    if isinf(edge_cost):
                        continue
                    next_cost = current_cost + edge_cost
                    if next_cost < dist[neighbor]:
                        dist[neighbor] = next_cost
                        heapq.heappush(pq, (next_cost, neighbor))

            shortest_cost_cache[key] = float("inf")
            return float("inf")

        def travel_time_s(vehicle: Vehicle, from_node_id: str, to_node_id: str) -> float:
            distance = cost_between(from_node_id, to_node_id)
            if isinf(distance):
                return float("inf")
            speed = max(vehicle.speed_mps, 0.1)
            return distance / speed

        def build_solution(assignment: dict[str, str], order: list[Job]) -> CuOptOutput:
            """Deterministically simulate one full multi-vehicle schedule for
            a fixed job->vehicle assignment and a fixed global service order.
            No backtracking happens inside this function: every job in
            `order` is committed to its assigned vehicle in sequence, so the
            result is always internally consistent."""

            routes = {
                vehicle.id: VehicleRoute(
                    vehicle_id=vehicle.id,
                    stops=[RouteStop(node_id=vehicle.start_node_id, stop_type="start", eta_s=0.0, departure_s=0.0)],
                )
                for vehicle in cuopt_input.vehicles
            }
            vehicle_positions = {vehicle.id: vehicle.start_node_id for vehicle in cuopt_input.vehicles}
            vehicle_times = {vehicle.id: 0.0 for vehicle in cuopt_input.vehicles}
            vehicle_battery = {vehicle.id: vehicle.battery_level_pct for vehicle in cuopt_input.vehicles}
            node_reservations: dict[str, list[tuple[float, float]]] = {}

            def reserve_node(node_id: str, start_s: float, duration_s: float) -> tuple[float, float, float]:
                reservations = node_reservations.setdefault(node_id, [])
                begin = start_s
                end = begin + duration_s
                buffer_s = max(cuopt_input.constraints.node_occupancy_buffer_s, 0.0)
                waiting = 0.0

                changed = True
                while changed:
                    changed = False
                    for reserved_start, reserved_end in sorted(reservations):
                        if end + buffer_s <= reserved_start or begin >= reserved_end + buffer_s:
                            continue
                        new_begin = reserved_end + buffer_s
                        waiting += new_begin - begin
                        begin = new_begin
                        end = begin + duration_s
                        changed = True
                reservations.append((begin, end))
                return begin, end, waiting

            total_cost = 0.0

            for job in order:
                vehicle_id = assignment.get(job.id)
                vehicle = vehicle_lookup.get(vehicle_id or "")
                if vehicle is None:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"No vehicle assignment for job {job.id}"],
                        solver_name="mock-cuopt",
                    )

                route = routes[vehicle.id]
                if (
                    cuopt_input.constraints.vehicle_max_jobs is not None
                    and len(route.assigned_job_ids) >= cuopt_input.constraints.vehicle_max_jobs
                ):
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Vehicle {vehicle.id} exceeded max assigned jobs"],
                        solver_name="mock-cuopt",
                    )

                current_node = vehicle_positions[vehicle.id]
                current_time = vehicle_times[vehicle.id]
                if job.pickup_node_id in blocked_nodes or job.delivery_node_id in blocked_nodes:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Job {job.id} touches a blocked node"],
                        solver_name="mock-cuopt",
                    )

                t_to_pick = travel_time_s(vehicle, current_node, job.pickup_node_id)
                t_to_del = travel_time_s(vehicle, job.pickup_node_id, job.delivery_node_id)
                if isinf(t_to_pick) or isinf(t_to_del):
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"No feasible path for job {job.id}"],
                        solver_name="mock-cuopt",
                    )

                arrival_pick = current_time + t_to_pick
                tw_pick_start, tw_pick_end = job.pickup_time_window
                tw_del_start, tw_del_end = job.delivery_time_window
                if arrival_pick > tw_pick_end:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Job {job.id} missed pickup time window"],
                        solver_name="mock-cuopt",
                    )
                pickup_start = max(arrival_pick, tw_pick_start)
                pickup_service = max(job.service_time_s, cuopt_input.constraints.min_turnaround_s)
                pickup_start_res, pickup_end_res, wait_pick = reserve_node(job.pickup_node_id, pickup_start, pickup_service)
                if pickup_start_res > tw_pick_end:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Job {job.id} pickup window exceeded after node contention"],
                        solver_name="mock-cuopt",
                    )

                arrival_del = pickup_end_res + t_to_del
                if arrival_del > tw_del_end:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Job {job.id} missed delivery time window"],
                        solver_name="mock-cuopt",
                    )
                delivery_start = max(arrival_del, tw_del_start)
                delivery_service = max(job.service_time_s, cuopt_input.constraints.min_turnaround_s)
                delivery_start_res, delivery_end_res, wait_del = reserve_node(job.delivery_node_id, delivery_start, delivery_service)
                if delivery_start_res > tw_del_end:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Job {job.id} delivery window exceeded after node contention"],
                        solver_name="mock-cuopt",
                    )

                distance_m = cost_between(current_node, job.pickup_node_id) + cost_between(job.pickup_node_id, job.delivery_node_id)
                battery_needed = distance_m * vehicle.energy_per_meter_pct
                projected_battery = vehicle_battery[vehicle.id] - battery_needed
                if projected_battery < vehicle.min_battery_reserve_pct:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Job {job.id} would break battery reserve for {vehicle.id}"],
                        solver_name="mock-cuopt",
                    )

                incremental_cost = distance_m
                if cuopt_input.objective in ("min_time", "min_makespan"):
                    incremental_cost = delivery_end_res - current_time

                route.stops.append(
                    RouteStop(
                        node_id=job.pickup_node_id,
                        stop_type="pickup",
                        job_id=job.id,
                        pallet_id=job.pallet_id,
                        eta_s=round(pickup_start_res, 3),
                        departure_s=round(pickup_end_res, 3),
                        travel_cost=round(t_to_pick, 3),
                        wait_s=round(wait_pick, 3),
                    )
                )
                route.stops.append(
                    RouteStop(
                        node_id=job.delivery_node_id,
                        stop_type="delivery",
                        job_id=job.id,
                        pallet_id=job.pallet_id,
                        eta_s=round(delivery_start_res, 3),
                        departure_s=round(delivery_end_res, 3),
                        travel_cost=round(t_to_del, 3),
                        wait_s=round(wait_del, 3),
                    )
                )
                route.assigned_job_ids.append(job.id)
                route.route_cost += incremental_cost
                route.total_time_s = delivery_end_res
                route.total_wait_s += wait_pick + wait_del
                route.consumed_battery_pct += battery_needed
                total_cost += incremental_cost
                vehicle_positions[vehicle.id] = job.delivery_node_id
                vehicle_times[vehicle.id] = delivery_end_res
                vehicle_battery[vehicle.id] -= battery_needed

                max_route = cuopt_input.constraints.max_route_time_s
                if max_route is not None and route.total_time_s > max_route:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Vehicle {vehicle.id} exceeded max route time"],
                        solver_name="mock-cuopt",
                    )

            for vehicle_id, route in routes.items():
                vehicle = vehicle_lookup[vehicle_id]
                charging_node = vehicle.charging_node_id
                if vehicle_positions[vehicle_id] == charging_node:
                    continue
                t_to_charge = travel_time_s(vehicle, vehicle_positions[vehicle_id], charging_node)
                if isinf(t_to_charge):
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=list(routes.values()),
                        total_cost=total_cost,
                        errors=[f"Vehicle {vehicle_id} cannot reach charging node"],
                        solver_name="mock-cuopt",
                    )
                charge_start, charge_end, wait_charge = reserve_node(
                    charging_node,
                    vehicle_times[vehicle_id] + t_to_charge,
                    max(cuopt_input.constraints.min_turnaround_s, 5.0),
                )
                back_cost = cost_between(vehicle_positions[vehicle_id], charging_node)
                route.stops.append(
                    RouteStop(
                        node_id=charging_node,
                        stop_type="charging",
                        eta_s=round(charge_start, 3),
                        departure_s=round(charge_end, 3),
                        travel_cost=round(t_to_charge, 3),
                        wait_s=round(wait_charge, 3),
                    )
                )
                if cuopt_input.objective == "min_distance":
                    route.route_cost += back_cost
                    total_cost += back_cost
                else:
                    route.route_cost += t_to_charge + wait_charge
                    total_cost += t_to_charge + wait_charge
                route.total_time_s = charge_end
                route.total_wait_s += wait_charge
                vehicle_times[vehicle_id] = charge_end

            return CuOptOutput(
                status="success",
                vehicle_routes=list(routes.values()),
                total_cost=round(total_cost, 3),
                solver_name="mock-cuopt",
            )

        rebuild_budget = [self.MAX_REBUILDS]

        def budgeted_build(assignment: dict[str, str], order: list[Job]) -> CuOptOutput | None:
            if rebuild_budget[0] <= 0:
                return None
            rebuild_budget[0] -= 1
            return build_solution(assignment, order)

        assignment, order, constructed = self._construct(
            vehicles=cuopt_input.vehicles,
            jobs=cuopt_input.jobs,
            max_jobs=cuopt_input.constraints.vehicle_max_jobs,
            build_solution=budgeted_build,
        )
        if constructed.status != "success":
            return constructed

        if len(order) <= self.LOCAL_SEARCH_JOB_LIMIT:
            assignment, order, constructed = self._local_search(
                vehicles=cuopt_input.vehicles,
                assignment=assignment,
                order=order,
                base_output=constructed,
                build_solution=budgeted_build,
            )

        return constructed

    def _construct(
        self,
        vehicles: list[Vehicle],
        jobs: list[Job],
        max_jobs: int | None,
        build_solution,
    ) -> tuple[dict[str, str], list[Job], CuOptOutput]:
        """Regret-2 insertion within descending-priority tiers.

        Within a tier, at each step every still-unassigned job is scored by
        the gap between its best and second-best vehicle option (regret).
        The job that would suffer most from being delayed is committed
        first, which produces materially better fleet balance than a fixed
        priority/id ordering once two or more vehicles are in play.
        Falls back to plain nearest-vehicle greedy above
        `REGRET_CONSTRUCTION_JOB_LIMIT` to keep runtime bounded.
        """
        assignment: dict[str, str] = {}
        order: list[Job] = []
        current_output = CuOptOutput(status="success", vehicle_routes=[], total_cost=0.0, solver_name="mock-cuopt")

        use_regret = len(jobs) <= self.REGRET_CONSTRUCTION_JOB_LIMIT

        tiers: dict[int, list[Job]] = {}
        for job in jobs:
            tiers.setdefault(job.priority, []).append(job)

        for priority in sorted(tiers.keys(), reverse=True):
            remaining = sorted(tiers[priority], key=lambda j: j.id)
            while remaining:
                assigned_counts: dict[str, int] = {}
                for vehicle_id in assignment.values():
                    assigned_counts[vehicle_id] = assigned_counts.get(vehicle_id, 0) + 1

                # Plain greedy (large job counts): only score the next job in
                # fixed priority/id order against every vehicle, no regret
                # comparison across jobs.
                candidate_jobs = remaining[:1] if not use_regret else remaining

                best_choice = None  # (sort_key, job, vehicle_id, output)
                for job in candidate_jobs:
                    candidates: list[tuple[float, str, CuOptOutput]] = []
                    for vehicle in vehicles:
                        if max_jobs is not None and assigned_counts.get(vehicle.id, 0) >= max_jobs:
                            continue
                        trial_output = build_solution({**assignment, job.id: vehicle.id}, order + [job])
                        if trial_output is None:
                            # Rebuild budget exhausted: fall back to first feasible option.
                            continue
                        if trial_output.status == "success":
                            candidates.append((trial_output.total_cost, vehicle.id, trial_output))

                    if not candidates:
                        current_output.errors = [f"Could not assign job {job.id}"]
                        current_output.status = "error"
                        return assignment, order, current_output

                    candidates.sort(key=lambda c: c[0])
                    best_cost, best_vehicle, best_output = candidates[0]
                    # A job with only one feasible vehicle has unbounded
                    # regret: there is no fallback if we defer it and lose
                    # that vehicle to another job, so it must be locked in
                    # ahead of any job that still has options.
                    second_cost = candidates[1][0] if len(candidates) > 1 else float("inf")
                    regret = second_cost - best_cost
                    sort_key = (regret, -best_cost) if use_regret else (0.0, -best_cost)
                    if best_choice is None or sort_key > best_choice[0]:
                        best_choice = (sort_key, job, best_vehicle, best_output)

                assert best_choice is not None
                _, chosen_job, chosen_vehicle, chosen_output = best_choice
                assignment[chosen_job.id] = chosen_vehicle
                order.append(chosen_job)
                current_output = chosen_output
                remaining.remove(chosen_job)

        return assignment, order, current_output

    def _local_search(
        self,
        vehicles: list[Vehicle],
        assignment: dict[str, str],
        order: list[Job],
        base_output: CuOptOutput,
        build_solution,
    ) -> tuple[dict[str, str], list[Job], CuOptOutput]:
        """Relocate / swap / or-opt local search over the assignment and
        service order. Every candidate move is re-validated through
        `build_solution`, so infeasible moves (time windows, battery,
        blocked nodes, max route time) are simply rejected."""
        EPS = 1e-6
        best_assignment = dict(assignment)
        best_order = list(order)
        best_output = base_output
        vehicle_ids = [v.id for v in vehicles]

        for _ in range(self.LOCAL_SEARCH_MAX_ROUNDS):
            improved = False

            # Relocate: move a single job to a different vehicle.
            for job in best_order:
                current_vehicle = best_assignment[job.id]
                for vehicle_id in vehicle_ids:
                    if vehicle_id == current_vehicle:
                        continue
                    trial_assignment = dict(best_assignment)
                    trial_assignment[job.id] = vehicle_id
                    trial_output = build_solution(trial_assignment, best_order)
                    if trial_output is None:
                        break
                    if trial_output.status == "success" and trial_output.total_cost < best_output.total_cost - EPS:
                        best_assignment = trial_assignment
                        best_output = trial_output
                        improved = True

            # Or-opt: swap service order of two jobs on the same vehicle.
            for i in range(len(best_order)):
                for j in range(i + 1, len(best_order)):
                    job_i, job_j = best_order[i], best_order[j]
                    if best_assignment[job_i.id] != best_assignment[job_j.id]:
                        continue
                    trial_order = list(best_order)
                    trial_order[i], trial_order[j] = trial_order[j], trial_order[i]
                    trial_output = build_solution(best_assignment, trial_order)
                    if trial_output is None:
                        break
                    if trial_output.status == "success" and trial_output.total_cost < best_output.total_cost - EPS:
                        best_order = trial_order
                        best_output = trial_output
                        improved = True

            # Swap: exchange vehicle assignment of two jobs on different vehicles.
            for i in range(len(best_order)):
                for j in range(i + 1, len(best_order)):
                    job_i, job_j = best_order[i], best_order[j]
                    if best_assignment[job_i.id] == best_assignment[job_j.id]:
                        continue
                    trial_assignment = dict(best_assignment)
                    trial_assignment[job_i.id], trial_assignment[job_j.id] = (
                        trial_assignment[job_j.id],
                        trial_assignment[job_i.id],
                    )
                    trial_output = build_solution(trial_assignment, best_order)
                    if trial_output is None:
                        break
                    if trial_output.status == "success" and trial_output.total_cost < best_output.total_cost - EPS:
                        best_assignment = trial_assignment
                        best_output = trial_output
                        improved = True

            if not improved:
                break

        return best_assignment, best_order, best_output


class HttpCuOptAdapter:
    """Adapter for the real NVIDIA cuOpt microservice (self-hosted server or
    the managed NIM endpoint at https://optimize.api.nvidia.com).

    Maps the internal `CuOptInput`/`CuOptOutput` contract to cuOpt's native
    request schema (`cost_matrix_data` / `task_data` / `fleet_data` /
    `solver_config`) and back, and follows the documented async pattern:
    POST /cuopt/request -> {"reqId": ...} -> poll GET /cuopt/requests/{id}
    until a terminal status is returned.

    Swap this in for `MockCuOptAdapter` once `CUOPT_ENDPOINT` /
    `CUOPT_API_KEY` point at a running cuOpt instance; the rest of the
    pipeline (planner, CBS, mission translator) is unaffected because both
    adapters speak the same `CuOptOutput` contract.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        poll_interval_s: float = 0.5,
        poll_timeout_s: float = 30.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _to_cuopt_payload(self, cuopt_input: CuOptInput) -> dict:
        """Build the native cuOpt request body.

        cuOpt models pickup-and-delivery pairs, time windows and vehicle
        breaks as flat parallel arrays keyed by task/vehicle index, so this
        flattens our node/job/vehicle dataclasses into that shape.
        """
        node_order = [node.id for node in cuopt_input.nodes]

        pickup_indices = [node_order.index(job.pickup_node_id) for job in cuopt_input.jobs]
        delivery_indices = [node_order.index(job.delivery_node_id) for job in cuopt_input.jobs]

        task_data = {
            "task_locations": pickup_indices + delivery_indices,
            "pickup_and_delivery_pairs": [[i, i + len(cuopt_input.jobs)] for i in range(len(cuopt_input.jobs))],
            "task_time_windows": (
                [list(job.pickup_time_window) for job in cuopt_input.jobs]
                + [list(job.delivery_time_window) for job in cuopt_input.jobs]
            ),
            "service_times": [job.service_time_s for job in cuopt_input.jobs] * 2,
            "demand": [[job.demand for job in cuopt_input.jobs] + [-job.demand for job in cuopt_input.jobs]],
            "task_priorities": [job.priority for job in cuopt_input.jobs] * 2,
            "order_ids": [job.id for job in cuopt_input.jobs] + [f"{job.id}::delivery" for job in cuopt_input.jobs],
        }

        fleet_data = {
            "vehicle_locations": [
                [node_order.index(v.start_node_id), node_order.index(v.charging_node_id)] for v in cuopt_input.vehicles
            ],
            "capacities": [[v.capacity for v in cuopt_input.vehicles]],
            "vehicle_ids": [v.id for v in cuopt_input.vehicles],
            "vehicle_max_times": [v.max_shift_time_s for v in cuopt_input.vehicles],
            "vehicle_break_time_windows": None,
        }
        if cuopt_input.constraints.vehicle_max_jobs is not None:
            fleet_data["vehicle_max_num_tasks"] = [
                cuopt_input.constraints.vehicle_max_jobs for _ in cuopt_input.vehicles
            ]

        objective_map = {
            "min_distance": "COST",
            "min_time": "TRAVEL_TIME",
            "min_makespan": "MAKESPAN",
        }
        solver_config = {
            "time_limit": self.poll_timeout_s,
            "objective": objective_map.get(cuopt_input.objective, "COST"),
        }
        if cuopt_input.constraints.blocked_edges or cuopt_input.constraints.blocked_nodes:
            solver_config["infeasible_cost"] = 1e9

        return {
            "cost_matrix_data": {"data": {"1": cuopt_input.cost_matrix}},
            "task_data": task_data,
            "fleet_data": fleet_data,
            "solver_config": solver_config,
        }

    def _from_cuopt_response(self, body: dict, cuopt_input: CuOptInput) -> CuOptOutput:
        response = body.get("response", body)
        solver_response = response.get("solver_response", response)

        status = str(solver_response.get("status", "")).lower()
        if status not in ("success", "optimal", "feasible"):
            return CuOptOutput(
                status="error",
                vehicle_routes=[],
                total_cost=0.0,
                errors=[str(solver_response.get("status", "cuOpt solve failed"))],
                solver_name="cuopt-nim",
            )

        node_order = [node.id for node in cuopt_input.nodes]
        job_lookup = {job.id: job for job in cuopt_input.jobs}
        vehicle_routes: list[VehicleRoute] = []
        raw_routes = solver_response.get("vehicle_data", {})

        for vehicle_key, vehicle_route_data in raw_routes.items():
            stops: list[RouteStop] = []
            assigned_job_ids: list[str] = []
            route_indices = vehicle_route_data.get("route", [])
            types = vehicle_route_data.get("type", [])
            arrival_times = vehicle_route_data.get("arrival_stamp", [])
            task_ids = vehicle_route_data.get("task_id", [])

            for idx, node_idx in enumerate(route_indices):
                node_id = node_order[node_idx] if 0 <= node_idx < len(node_order) else str(node_idx)
                stop_type_raw = types[idx] if idx < len(types) else "Enroute"
                eta = float(arrival_times[idx]) if idx < len(arrival_times) else 0.0
                task_id = str(task_ids[idx]) if idx < len(task_ids) else ""

                stop_type = "start"
                job_id = ""
                if stop_type_raw in ("Pickup", "Delivery"):
                    job_id = task_id.replace("::delivery", "")
                    stop_type = "pickup" if stop_type_raw == "Pickup" else "delivery"
                    if job_id and job_id not in assigned_job_ids and job_id in job_lookup:
                        assigned_job_ids.append(job_id)

                stops.append(
                    RouteStop(
                        node_id=node_id,
                        stop_type=stop_type,  # type: ignore[arg-type]
                        job_id=job_id,
                        pallet_id=job_lookup[job_id].pallet_id if job_id in job_lookup else "",
                        eta_s=eta,
                        departure_s=eta,
                    )
                )

            vehicle_routes.append(
                VehicleRoute(
                    vehicle_id=str(vehicle_key),
                    stops=stops,
                    assigned_job_ids=assigned_job_ids,
                    route_cost=float(vehicle_route_data.get("route_total_cost", 0.0)),
                )
            )

        return CuOptOutput(
            status="success",
            vehicle_routes=vehicle_routes,
            total_cost=float(solver_response.get("solution_cost", 0.0)),
            solver_name="cuopt-nim",
        )

    def solve(self, cuopt_input: CuOptInput) -> CuOptOutput:
        payload = self._to_cuopt_payload(cuopt_input)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(f"{self.endpoint}/cuopt/request", data=body, headers=self._headers(), method="POST")

        try:
            with request.urlopen(req, timeout=self.poll_timeout_s) as resp:
                submit_body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return CuOptOutput(
                status="error",
                vehicle_routes=[],
                total_cost=0.0,
                errors=[f"cuOpt submit failed: {exc}"],
                solver_name="cuopt-nim",
            )

        # Some cuOpt deployments answer synchronously; others return a
        # request id that must be polled.
        request_id = submit_body.get("reqId") or submit_body.get("request_id")
        if not request_id:
            try:
                return self._from_cuopt_response(submit_body, cuopt_input)
            except Exception as exc:
                return CuOptOutput(
                    status="error",
                    vehicle_routes=[],
                    total_cost=0.0,
                    errors=[f"Invalid cuOpt response format: {exc}"],
                    solver_name="cuopt-nim",
                )

        deadline = time.monotonic() + self.poll_timeout_s
        poll_req_url = f"{self.endpoint}/cuopt/requests/{request_id}"
        while time.monotonic() < deadline:
            try:
                poll_req = request.Request(poll_req_url, headers=self._headers(), method="GET")
                with request.urlopen(poll_req, timeout=self.poll_timeout_s) as resp:
                    poll_body = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                return CuOptOutput(
                    status="error",
                    vehicle_routes=[],
                    total_cost=0.0,
                    errors=[f"cuOpt poll failed: {exc}"],
                    solver_name="cuopt-nim",
                )

            poll_status = str(poll_body.get("status", "")).lower()
            if poll_status in ("", "success", "complete", "completed", "error", "failed"):
                try:
                    return self._from_cuopt_response(poll_body, cuopt_input)
                except Exception as exc:
                    return CuOptOutput(
                        status="error",
                        vehicle_routes=[],
                        total_cost=0.0,
                        errors=[f"Invalid cuOpt response format: {exc}"],
                        solver_name="cuopt-nim",
                    )
            time.sleep(self.poll_interval_s)

        return CuOptOutput(
            status="error",
            vehicle_routes=[],
            total_cost=0.0,
            errors=[f"cuOpt request {request_id} timed out after {self.poll_timeout_s}s"],
            solver_name="cuopt-nim",
        )


def cuopt_input_to_debug_json(cuopt_input: CuOptInput) -> str:
    return json.dumps(cuopt_input.to_dict(), indent=2)


def cuopt_output_to_debug_json(cuopt_output: CuOptOutput) -> str:
    return json.dumps(asdict(cuopt_output), indent=2)
