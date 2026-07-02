from __future__ import annotations

from typing import Any


def allocate_jobs_weighted(
    vehicles: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    node_index: dict[str, int],
    cost_matrix: list[list[float]],
) -> dict[str, list[str]]:
    allocation: dict[str, list[str]] = {str(v.get("id", "")): [] for v in vehicles if v.get("id")}

    def distance(a: str, b: str) -> float:
        return float(cost_matrix[node_index[a]][node_index[b]])

    vehicle_pos = {str(v.get("id", "")): str(v.get("start_node_id", "")) for v in vehicles if v.get("id")}
    vehicle_battery = {str(v.get("id", "")): float(v.get("battery_level_pct", 100.0)) for v in vehicles if v.get("id")}

    ordered_jobs = sorted(jobs, key=lambda j: (-int(j.get("priority", 1)), str(j.get("id", ""))))
    for job in ordered_jobs:
        jid = str(job.get("id", ""))
        pickup = str(job.get("pickup_node_id", ""))
        delivery = str(job.get("delivery_node_id", ""))
        if not jid or not pickup or not delivery:
            continue

        best_vehicle = ""
        best_score = float("inf")
        for vehicle in vehicles:
            vid = str(vehicle.get("id", ""))
            if not vid:
                continue
            if not vehicle_pos.get(vid):
                continue
            trip_distance = distance(vehicle_pos[vid], pickup) + distance(pickup, delivery)
            # Penalize low battery and high assigned load.
            load_penalty = len(allocation.get(vid, [])) * 2.0
            battery_penalty = 0.0 if vehicle_battery[vid] > 30.0 else 15.0
            score = trip_distance + load_penalty + battery_penalty
            if score < best_score:
                best_score = score
                best_vehicle = vid

        if not best_vehicle:
            continue

        allocation[best_vehicle].append(jid)
        vehicle_pos[best_vehicle] = delivery
        vehicle_battery[best_vehicle] = max(0.0, vehicle_battery[best_vehicle] - best_score * 0.2)

    return allocation
