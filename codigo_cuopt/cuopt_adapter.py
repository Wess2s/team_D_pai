from __future__ import annotations

import json
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
    """Simple deterministic solver that keeps an interface compatible with real cuOpt integration."""

    def solve(self, cuopt_input: CuOptInput) -> CuOptOutput:
        if not cuopt_input.vehicles:
            return CuOptOutput(
                status="error",
                vehicle_routes=[],
                total_cost=0.0,
                errors=["No vehicles available"],
                solver_name="mock-cuopt",
            )

        jobs = sorted(cuopt_input.jobs, key=lambda j: (-j.priority, j.id))
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
        vehicle_lookup = {vehicle.id: vehicle for vehicle in cuopt_input.vehicles}
        blocked_nodes = set(cuopt_input.constraints.blocked_nodes)
        blocked_edges = set(cuopt_input.constraints.blocked_edges)
        blocked_edges |= {(b, a) for (a, b) in blocked_edges}
        node_ids = [node.id for node in cuopt_input.nodes]
        shortest_cost_cache: dict[tuple[str, str], float] = {}
        node_reservations: dict[str, list[tuple[float, float]]] = {}

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
                    if neighbor == node_id:
                        continue
                    if neighbor in blocked_nodes:
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

        def estimate_job_assignment(vehicle: Vehicle, job: Job) -> tuple[float, dict] | None:
            current_node = vehicle_positions[vehicle.id]
            current_time = vehicle_times[vehicle.id]
            if job.pickup_node_id in blocked_nodes or job.delivery_node_id in blocked_nodes:
                return None

            t_to_pick = travel_time_s(vehicle, current_node, job.pickup_node_id)
            t_to_del = travel_time_s(vehicle, job.pickup_node_id, job.delivery_node_id)
            if isinf(t_to_pick) or isinf(t_to_del):
                return None

            arrival_pick = current_time + t_to_pick
            tw_pick_start, tw_pick_end = job.pickup_time_window
            tw_del_start, tw_del_end = job.delivery_time_window
            if arrival_pick > tw_pick_end:
                return None
            pickup_start = max(arrival_pick, tw_pick_start)

            pickup_service = max(job.service_time_s, cuopt_input.constraints.min_turnaround_s)
            pickup_start_res, pickup_end_res, wait_pick = reserve_node(job.pickup_node_id, pickup_start, pickup_service)
            if pickup_start_res > tw_pick_end:
                # Rollback reservation
                node_reservations[job.pickup_node_id].pop()
                return None

            arrival_del = pickup_end_res + t_to_del
            if arrival_del > tw_del_end:
                node_reservations[job.pickup_node_id].pop()
                return None
            delivery_start = max(arrival_del, tw_del_start)
            delivery_service = max(job.service_time_s, cuopt_input.constraints.min_turnaround_s)
            delivery_start_res, delivery_end_res, wait_del = reserve_node(job.delivery_node_id, delivery_start, delivery_service)
            if delivery_start_res > tw_del_end:
                node_reservations[job.pickup_node_id].pop()
                node_reservations[job.delivery_node_id].pop()
                return None

            distance_m = cost_between(current_node, job.pickup_node_id) + cost_between(job.pickup_node_id, job.delivery_node_id)
            battery_needed = distance_m * vehicle.energy_per_meter_pct
            projected_battery = vehicle_battery[vehicle.id] - battery_needed
            if projected_battery < vehicle.min_battery_reserve_pct:
                node_reservations[job.pickup_node_id].pop()
                node_reservations[job.delivery_node_id].pop()
                return None

            incremental_cost = distance_m
            if cuopt_input.objective in ("min_time", "min_makespan"):
                incremental_cost = delivery_end_res - current_time

            return incremental_cost, {
                "pickup_start": pickup_start_res,
                "pickup_end": pickup_end_res,
                "delivery_start": delivery_start_res,
                "delivery_end": delivery_end_res,
                "wait_pick": wait_pick,
                "wait_del": wait_del,
                "distance_m": distance_m,
                "battery_needed": battery_needed,
                "travel_to_pick": t_to_pick,
                "travel_to_del": t_to_del,
            }

        def rollback_estimate(job: Job) -> None:
            if node_reservations.get(job.pickup_node_id):
                node_reservations[job.pickup_node_id].pop()
            if node_reservations.get(job.delivery_node_id):
                node_reservations[job.delivery_node_id].pop()

        total_cost = 0.0

        for job in jobs:
            best_vehicle_id = None
            best_cost = float("inf")
            best_data = None
            for vehicle in cuopt_input.vehicles:
                route = routes[vehicle.id]
                if (
                    cuopt_input.constraints.vehicle_max_jobs is not None
                    and len(route.assigned_job_ids) >= cuopt_input.constraints.vehicle_max_jobs
                ):
                    continue
                estimate = estimate_job_assignment(vehicle, job)
                if estimate is None:
                    continue
                incremental_cost, assignment_data = estimate
                if incremental_cost < best_cost:
                    best_cost = incremental_cost
                    best_vehicle_id = vehicle.id
                    best_data = assignment_data
                rollback_estimate(job)

            if best_vehicle_id is None or best_data is None:
                return CuOptOutput(
                    status="error",
                    vehicle_routes=list(routes.values()),
                    total_cost=total_cost,
                    errors=[f"Could not assign job {job.id}"],
                    solver_name="mock-cuopt",
                )

            # Reserve with selected vehicle after candidate loop.
            vehicle = vehicle_lookup[best_vehicle_id]
            confirmed = estimate_job_assignment(vehicle, job)
            if confirmed is None:
                return CuOptOutput(
                    status="error",
                    vehicle_routes=list(routes.values()),
                    total_cost=total_cost,
                    errors=[f"Lost reservation while assigning job {job.id}"],
                    solver_name="mock-cuopt",
                )
            _, data = confirmed

            route = routes[best_vehicle_id]
            route.stops.append(
                RouteStop(
                    node_id=job.pickup_node_id,
                    stop_type="pickup",
                    job_id=job.id,
                    pallet_id=job.pallet_id,
                    eta_s=round(data["pickup_start"], 3),
                    departure_s=round(data["pickup_end"], 3),
                    travel_cost=round(data["travel_to_pick"], 3),
                    wait_s=round(data["wait_pick"], 3),
                )
            )
            route.stops.append(
                RouteStop(
                    node_id=job.delivery_node_id,
                    stop_type="delivery",
                    job_id=job.id,
                    pallet_id=job.pallet_id,
                    eta_s=round(data["delivery_start"], 3),
                    departure_s=round(data["delivery_end"], 3),
                    travel_cost=round(data["travel_to_del"], 3),
                    wait_s=round(data["wait_del"], 3),
                )
            )
            route.assigned_job_ids.append(job.id)
            route.route_cost += best_cost
            route.total_time_s = data["delivery_end"]
            route.total_wait_s += data["wait_pick"] + data["wait_del"]
            route.consumed_battery_pct += data["battery_needed"]
            total_cost += best_cost
            vehicle_positions[best_vehicle_id] = job.delivery_node_id
            vehicle_times[best_vehicle_id] = data["delivery_end"]
            vehicle_battery[best_vehicle_id] -= data["battery_needed"]

            max_route = cuopt_input.constraints.max_route_time_s
            if max_route is not None and route.total_time_s > max_route:
                return CuOptOutput(
                    status="error",
                    vehicle_routes=list(routes.values()),
                    total_cost=total_cost,
                    errors=[f"Vehicle {best_vehicle_id} exceeded max route time"],
                    solver_name="mock-cuopt",
                )

        for vehicle_id, route in routes.items():
            charging_node = vehicle_lookup[vehicle_id].charging_node_id
            if vehicle_positions[vehicle_id] != charging_node:
                t_to_charge = travel_time_s(vehicle_lookup[vehicle_id], vehicle_positions[vehicle_id], charging_node)
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


class HttpCuOptAdapter:
    """Adapter for future replacement with real cuOpt service endpoint."""

    def __init__(self, endpoint: str, api_key: str = "") -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def solve(self, cuopt_input: CuOptInput) -> CuOptOutput:
        payload = json.dumps(cuopt_input.to_dict()).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(self.endpoint, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return CuOptOutput(
                status="error",
                vehicle_routes=[],
                total_cost=0.0,
                errors=[f"cuOpt HTTP call failed: {exc}"],
                solver_name="http-cuopt",
            )

        try:
            routes = []
            for raw_route in body.get("vehicle_routes", []):
                stops = [RouteStop(**raw_stop) for raw_stop in raw_route.get("stops", [])]
                routes.append(
                    VehicleRoute(
                        vehicle_id=raw_route.get("vehicle_id", ""),
                        stops=stops,
                        assigned_job_ids=raw_route.get("assigned_job_ids", []),
                        route_cost=float(raw_route.get("route_cost", 0.0)),
                    )
                )
            return CuOptOutput(
                status=body.get("status", "error"),
                vehicle_routes=routes,
                total_cost=float(body.get("total_cost", 0.0)),
                errors=body.get("errors", []),
                solver_name="http-cuopt",
            )
        except Exception as exc:
            return CuOptOutput(
                status="error",
                vehicle_routes=[],
                total_cost=0.0,
                errors=[f"Invalid cuOpt response format: {exc}"],
                solver_name="http-cuopt",
            )


def cuopt_input_to_debug_json(cuopt_input: CuOptInput) -> str:
    return json.dumps(cuopt_input.to_dict(), indent=2)


def cuopt_output_to_debug_json(cuopt_output: CuOptOutput) -> str:
    return json.dumps(asdict(cuopt_output), indent=2)
