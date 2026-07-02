from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal

from cbs_planner import CBSConfig, plan_checkpoint_cbs
from cuopt_adapter import MockCuOptAdapter, build_cuopt_input
from factory_simulator import simulate_factory_execution
from isaac_scene_extractor import build_mock_warehouse_snapshot, snapshot_to_graph_nodes
from logistics_models import CuOptConstraints, Job, Vehicle
from mission_translator import translate_solution_to_mission
from planner import run_planner_loop
from runtime_config import load_runtime_config
from warehouse_graph import build_warehouse_graph


ScenarioName = Literal["single_forklift", "dual_forklift", "urgent_jobs", "factory_realistic"]


def _build_demo_vehicles(scenario: ScenarioName = "dual_forklift") -> list[Vehicle]:
	if scenario == "single_forklift":
		return [
			Vehicle(
				id="forklift_1",
				start_node_id="Depot_Main",
				capacity=2.0,
				charging_node_id="Charging_01",
				speed_mps=1.1,
			)
		]

	if scenario == "factory_realistic":
		return [
			Vehicle(
				id="forklift_1",
				start_node_id="Depot_Main",
				capacity=2.0,
				charging_node_id="Charging_01",
				speed_mps=1.0,
				battery_level_pct=78.0,
				energy_per_meter_pct=0.45,
			),
			Vehicle(
				id="forklift_2",
				start_node_id="Depot_Main",
				capacity=2.0,
				charging_node_id="Charging_01",
				speed_mps=1.3,
				battery_level_pct=62.0,
				energy_per_meter_pct=0.4,
			),
		]

	return [
		Vehicle(
			id="forklift_1",
			start_node_id="Depot_Main",
			capacity=2.0,
			charging_node_id="Charging_01",
		),
		Vehicle(
			id="forklift_2",
			start_node_id="Depot_Main",
			capacity=2.0,
			charging_node_id="Charging_01",
		),
	]


def _build_demo_jobs(scenario: ScenarioName = "dual_forklift") -> list[Job]:
	jobs = [
		Job(
			id="job_001",
			pallet_id="Pallet_03",
			pickup_node_id="Storage_A",
			delivery_node_id="Dock_01",
			priority=3,
			pickup_time_window=(0, 300),
			delivery_time_window=(0, 600),
		),
		Job(
			id="job_002",
			pallet_id="Pallet_04",
			pickup_node_id="Storage_B",
			delivery_node_id="Dock_02",
			priority=2,
			pickup_time_window=(0, 420),
			delivery_time_window=(0, 720),
		),
		Job(
			id="job_003",
			pallet_id="Pallet_05",
			pickup_node_id="Storage_C",
			delivery_node_id="Dock_01",
			priority=1,
			pickup_time_window=(0, 540),
			delivery_time_window=(0, 900),
		),
	]

	if scenario == "urgent_jobs":
		jobs[0].priority = 10
		jobs[0].service_time_s = 20.0
		jobs[0].pickup_time_window = (0, 180)
		jobs[1].priority = 8
		jobs[2].priority = 5

	if scenario == "factory_realistic":
		jobs.append(
			Job(
				id="job_004",
				pallet_id="Pallet_09",
				pickup_node_id="Storage_A",
				delivery_node_id="Dock_02",
				priority=7,
				service_time_s=18.0,
				pickup_time_window=(60, 420),
				delivery_time_window=(120, 900),
			)
		)

	return jobs


def _build_constraints(scenario: ScenarioName) -> CuOptConstraints:
	constraints = CuOptConstraints(
		vehicle_max_jobs=3,
		node_occupancy_buffer_s=4.0,
		min_turnaround_s=3.0,
		max_route_time_s=3600,
	)

	if scenario == "factory_realistic":
		constraints.blocked_edges = [
			("Storage_B", "Dock_02"),
		]
		constraints.node_occupancy_buffer_s = 6.0
		constraints.max_route_time_s = 2400

	return constraints


def _build_human_occupancy(scenario: ScenarioName) -> dict[str, list[tuple[int, int]]]:
	if scenario == "factory_realistic":
		return {
			"Dock_01": [(35, 70), (120, 145)],
			"Dock_02": [(65, 90)],
			"Waypoint_01": [(80, 110)],
		}
	return {
		"Dock_01": [(45, 55)],
	}


def _build_checkpoints_from_solution(solution_dict: dict) -> dict[str, list[str]]:
	checkpoints: dict[str, list[str]] = {}
	for route in solution_dict.get("vehicle_routes", []):
		vehicle_id = route.get("vehicle_id", "")
		if not vehicle_id:
			continue
		stops = route.get("stops", [])
		ordered_nodes = [stop.get("node_id", "") for stop in stops if stop.get("node_id")]
		if ordered_nodes:
			checkpoints[vehicle_id] = ordered_nodes
	return checkpoints


def run_demo_pipeline(scenario: ScenarioName = "dual_forklift") -> dict:
	snapshot = build_mock_warehouse_snapshot()
	nodes = snapshot_to_graph_nodes(snapshot)
	graph = build_warehouse_graph(nodes)

	vehicles = _build_demo_vehicles(scenario=scenario)
	jobs = _build_demo_jobs(scenario=scenario)
	constraints = _build_constraints(scenario)

	objective = "min_distance"
	if scenario == "factory_realistic":
		objective = "min_makespan"

	cuopt_input = build_cuopt_input(graph=graph, vehicles=vehicles, jobs=jobs, objective=objective)
	cuopt_input.constraints = constraints

	solver = MockCuOptAdapter()
	solution = solver.solve(cuopt_input)
	mission = translate_solution_to_mission(
		mission_id=f"mission_hackathon_{scenario}",
		objective=cuopt_input.objective,
		nodes=nodes,
		solution=solution,
	)

	human_occupancy = _build_human_occupancy(scenario)
	checkpoints = _build_checkpoints_from_solution(solution.to_dict())
	cbs_result = plan_checkpoint_cbs(
		checkpoints=checkpoints,
		node_index=graph.node_index,
		cost_matrix=graph.distance_matrix,
		blocked_nodes=cuopt_input.constraints.blocked_nodes,
		blocked_edges=cuopt_input.constraints.blocked_edges,
		human_occupancy=human_occupancy,
		config=CBSConfig(
			max_neighbors=4,
			max_time_steps=300,
			goal_hold_steps=2,
			max_expansions=500,
		),
	)

	execution = simulate_factory_execution(mission)

	runtime_cfg = load_runtime_config()
	planner_loop = run_planner_loop(
		config=runtime_cfg,
		cuopt_input=cuopt_input,
		nodes=nodes,
		node_index=graph.node_index,
		cost_matrix=graph.distance_matrix,
		human_occupancy=human_occupancy,
		mission_id=f"mission_hackathon_{scenario}",
	)

	return {
		"scenario": scenario,
		"isaac_snapshot": asdict(snapshot),
		"warehouse_graph": {
			"nodes": [asdict(n) for n in nodes],
			"node_index": graph.node_index,
			"distance_matrix": graph.distance_matrix,
		},
		"cuopt_input": cuopt_input.to_dict(),
		"cuopt_output": solution.to_dict(),
		"human_occupancy": human_occupancy,
		"cbs_output": cbs_result,
		"mission_plan": mission.to_dict(),
		"execution_report": execution.to_dict(),
		"planner_loop": planner_loop,
	}


if __name__ == "__main__":
	result = run_demo_pipeline()
	print(json.dumps(result, indent=2))
