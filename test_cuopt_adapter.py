from __future__ import annotations

import unittest

from cuopt_adapter import MockCuOptAdapter, build_cuopt_input
from logistics_models import CuOptConstraints, Job, Vehicle
from warehouse_graph import build_warehouse_graph
from logistics_models import Node


def _starvation_scenario():
    """Two vehicles, two jobs, one job per vehicle (vehicle_max_jobs=1).

    job_1 is cheaper for V2 than for V1, but not exclusive to it.
    job_2 has a pickup/delivery window so tight only V2 (already parked on
    top of it) can make it; V1 is far too slow. A solver with no lookahead
    that services jobs strictly in id order will hand V2 to job_1 first and
    then find job_2 unassignable, even though a feasible assignment exists.
    """
    nodes = [
        Node(id="V1_Start", node_type="depot", x=0.0, y=0.0),
        Node(id="V2_Start", node_type="depot", x=10.0, y=0.0),
        Node(id="Job1_Pos", node_type="storage", x=9.0, y=0.0),
        Node(id="Job2_Pos", node_type="dock", x=10.0, y=0.0),
    ]
    graph = build_warehouse_graph(nodes)
    vehicles = [
        Vehicle(id="V1", start_node_id="V1_Start", capacity=5.0, charging_node_id="V1_Start", speed_mps=1.2),
        Vehicle(id="V2", start_node_id="V2_Start", capacity=5.0, charging_node_id="V2_Start", speed_mps=1.2),
    ]
    jobs = [
        Job(id="job_1", pallet_id="p1", pickup_node_id="Job1_Pos", delivery_node_id="Job1_Pos", priority=1, service_time_s=0.0),
        Job(
            id="job_2",
            pallet_id="p2",
            pickup_node_id="Job2_Pos",
            delivery_node_id="Job2_Pos",
            priority=1,
            service_time_s=0.0,
            pickup_time_window=(0, 0.5),
            delivery_time_window=(0, 0.5),
        ),
    ]
    cuopt_input = build_cuopt_input(graph=graph, vehicles=vehicles, jobs=jobs, objective="min_distance")
    cuopt_input.constraints = CuOptConstraints(vehicle_max_jobs=1, min_turnaround_s=0.0, node_occupancy_buffer_s=0.0)
    return cuopt_input


class TestMockCuOptAdapter(unittest.TestCase):
    def test_regret_insertion_avoids_greedy_starvation(self) -> None:
        cuopt_input = _starvation_scenario()
        result = MockCuOptAdapter().solve(cuopt_input)

        self.assertEqual(result.status, "success")
        assignment = {job_id: route.vehicle_id for route in result.vehicle_routes for job_id in route.assigned_job_ids}
        self.assertEqual(assignment.get("job_2"), "V2")
        self.assertEqual(set(assignment.keys()), {"job_1", "job_2"})

    def test_plain_greedy_fallback_actually_starves_without_regret(self) -> None:
        """Sanity check that the scenario above is a real trap, not a
        trivially-solvable one: with regret disabled the same solver fails,
        which is exactly the failure mode regret insertion exists to fix."""
        cuopt_input = _starvation_scenario()
        solver = MockCuOptAdapter()
        solver.REGRET_CONSTRUCTION_JOB_LIMIT = 0
        solver.LOCAL_SEARCH_JOB_LIMIT = 0
        result = solver.solve(cuopt_input)
        self.assertEqual(result.status, "error")

    def test_no_vehicles_reports_error(self) -> None:
        nodes = [Node(id="A", node_type="depot", x=0.0, y=0.0)]
        graph = build_warehouse_graph(nodes)
        cuopt_input = build_cuopt_input(graph=graph, vehicles=[], jobs=[])
        result = MockCuOptAdapter().solve(cuopt_input)
        self.assertEqual(result.status, "error")


if __name__ == "__main__":
    unittest.main()
