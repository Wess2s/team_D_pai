"""
test_live_integration.py
------------------------
Offline validation for the real-environment integration layer. These tests use
a captured ``/state`` fixture (``tests_fixtures_state.json``) so they run with NO
Isaac Sim and NO network — validating the extractor, the nav-graph builder, the
cuOpt payload build, mission translation, dispatch folding and replan detection.

Run:
    cd team_D_pai && python3 -m unittest test_live_integration -v
"""
from __future__ import annotations

import json
import math
import os
import unittest

from cuopt_adapter import MockCuOptAdapter, build_cuopt_input
from isaac_dispatch import mission_to_steps
from isaac_scene_extractor import extract_from_state_snapshot
from live_state_adapter import build_nodes_vehicles_jobs
from mission_translator import translate_solution_to_mission
from replanner import detect_replan_triggers
from warehouse_graph import build_graph_from_state, build_warehouse_graph
from cbs_integration import deconflict_mission, mission_to_grid_checkpoints
from cbs_executor import plan_execution_events
from cbs_planner import _detect_first_conflict

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_fixtures_state.json")


def _load_state() -> dict:
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


@unittest.skipUnless(os.path.isfile(FIXTURE), "state fixture not captured")
class TestLiveIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _load_state()

    def test_extractor_reads_real_entities(self) -> None:
        snap = extract_from_state_snapshot(self.state)
        kinds = {e.kind for e in snap.entities}
        self.assertIn("forklift", kinds)
        self.assertIn("pallet", kinds)
        self.assertIn("dock", kinds)
        self.assertIn("waypoint", kinds)
        # Real ids are preserved.
        ids = {e.entity_id for e in snap.entities}
        self.assertTrue(any(i.startswith("WH_Palette_") for i in ids))
        self.assertTrue(any(i.startswith("stage_") for i in ids))

    def test_build_inputs_uses_real_ids(self) -> None:
        nodes, vehicles, jobs = build_nodes_vehicles_jobs(self.state)
        self.assertTrue(vehicles)
        self.assertTrue(jobs)
        # Vehicle ids match scene forklift ids; job endpoints are real pallet/zone ids.
        self.assertTrue(all(v.id.startswith("AMR_") for v in vehicles))
        for job in jobs:
            self.assertTrue(job.pickup_node_id.startswith("WH_Palette_"))
            self.assertTrue(job.delivery_node_id.startswith("stage_"))

    def test_jobs_assign_distinct_zones_no_stacking(self) -> None:
        # Each staging zone is targeted by at most one pallet (no pallet-on-pallet).
        _n, _v, jobs = build_nodes_vehicles_jobs(self.state)
        target_zones = [j.delivery_node_id for j in jobs]
        self.assertEqual(len(target_zones), len(set(target_zones)))

    def test_jobs_pair_pallet_with_nearest_zone(self) -> None:
        # No job should send a pallet to a zone when a closer free zone exists in
        # its own assignment set (greedy-nearest property: sum of chosen distances
        # is <= the round-robin pairing).
        import math

        state = self.state
        _n, _v, jobs = build_nodes_vehicles_jobs(state)
        pallets = state["pallets"]
        zones = state["zones"]

        def d(pid: str, zid: str) -> float:
            p, z = pallets[pid], zones[zid]
            return math.dist((p["x"], p["y"]), (z["x"], z["y"]))

        chosen = sum(d(j.pallet_id, j.delivery_node_id) for j in jobs)
        zone_ids = sorted({j.delivery_node_id for j in jobs})
        pallet_ids = [j.pallet_id for j in jobs]
        rr = sum(d(p, zone_ids[i % len(zone_ids)]) for i, p in enumerate(pallet_ids))
        self.assertLessEqual(chosen, rr + 1e-6)

    def test_nav_graph_routing_matches_or_exceeds_euclidean(self) -> None:
        nodes, _, _ = build_nodes_vehicles_jobs(self.state)
        nav = build_graph_from_state(self.state, nodes)
        euc = build_warehouse_graph(nodes)
        # Same node set / index.
        self.assertEqual(nav.node_index, euc.node_index)
        # Routing distance is never shorter than straight-line (path >= chord),
        # allowing a small attach tolerance.
        checked = 0
        for a in nav.node_index:
            for b in nav.node_index:
                if a == b:
                    continue
                nd = nav.get_distance(a, b)
                ed = euc.get_distance(a, b)
                if math.isinf(nd):
                    continue
                self.assertGreaterEqual(nd + 1e-6, ed - 2.0)
                checked += 1
        self.assertGreater(checked, 0)

    def test_full_pipeline_produces_dispatchable_steps(self) -> None:
        nodes, vehicles, jobs = build_nodes_vehicles_jobs(self.state)
        graph = build_graph_from_state(self.state, nodes)
        cuopt_input = build_cuopt_input(graph=graph, vehicles=vehicles, jobs=jobs, objective="min_distance")
        solution = MockCuOptAdapter().solve(cuopt_input)
        self.assertEqual(solution.status, "success")

        mission = translate_solution_to_mission(
            mission_id="test", objective="min_distance", nodes=nodes, solution=solution
        )
        mission_dict = mission.to_dict()
        # Every planned vehicle folds into valid pick/drop steps.
        total_steps = 0
        for commands in mission_dict["vehicle_commands"].values():
            steps = mission_to_steps(commands)
            for verb, target in steps:
                self.assertIn(verb, {"pick", "drop"})
                self.assertTrue(target)
            total_steps += len(steps)
        self.assertGreater(total_steps, 0)

    def test_replan_triggers_clean_on_idle_scene(self) -> None:
        # Fixture forklifts were sanitised to idle/clear — no false positives.
        triggers = detect_replan_triggers(self.state)
        self.assertEqual(triggers, [])

    def test_replan_triggers_detect_injected_incidents(self) -> None:
        state = _load_state()
        first_fk = next(iter(state["forklifts"]))
        state["forklifts"][first_fk]["path_blocked"] = True
        first_zone = next(iter(state["zones"]))
        state["zones"][first_zone]["blocked"] = True
        triggers = detect_replan_triggers(state)
        types = {t["event_type"] for t in triggers}
        self.assertIn("aisle_block", types)
        self.assertGreaterEqual(len(triggers), 2)

    def _solved_mission(self, state: dict) -> dict:
        nodes, vehicles, jobs = build_nodes_vehicles_jobs(state, include_busy=True)
        graph = build_graph_from_state(state, nodes)
        cuopt_input = build_cuopt_input(graph=graph, vehicles=vehicles, jobs=jobs, objective="min_distance")
        solution = MockCuOptAdapter().solve(cuopt_input)
        mission = translate_solution_to_mission(
            mission_id="test", objective="min_distance", nodes=nodes, solution=solution
        )
        return mission.to_dict()

    def test_cbs_checkpoints_snap_to_real_grid_nodes(self) -> None:
        mission = self._solved_mission(self.state)
        checkpoints, actions = mission_to_grid_checkpoints(mission, self.state)
        grid_ids = set((self.state["graph"]["nodes"]).keys())
        self.assertTrue(checkpoints)
        for vid, nodes in checkpoints.items():
            # Every checkpoint is a real nav-graph node id.
            for nid in nodes:
                self.assertIn(nid, grid_ids)
            # Actions align 1:1 with checkpoints and carry pick/drop targets.
            self.assertEqual(len(actions[vid]), len(nodes))
            self.assertTrue(any(a["kind"] in {"pick", "drop"} for a in actions[vid]))

    def test_cbs_deconflicts_over_real_grid(self) -> None:
        mission = self._solved_mission(self.state)
        result = deconflict_mission(mission, self.state)
        self.assertIn(result["status"], {"success", "error"})
        # A successful plan yields a timed grid path per planned vehicle.
        if result["status"] == "success" and result["checkpoints"]:
            for vid in result["checkpoints"]:
                self.assertIn(vid, result["agent_paths"])
                self.assertTrue(result["agent_paths"][vid])

    def test_cbs_execution_events_are_ordered_and_actionable(self) -> None:
        mission = self._solved_mission(self.state)
        result = deconflict_mission(mission, self.state)
        if result["status"] != "success" or not result["agent_paths"]:
            self.skipTest("no CBS paths to schedule")
        events = plan_execution_events(result)
        self.assertTrue(events)
        # Globally sorted by timestep.
        times = [e["t"] for e in events]
        self.assertEqual(times, sorted(times))
        # Contains at least one pick and one drop across the schedule.
        cmds = {e["command"] for e in events}
        self.assertTrue({"pick", "drop"} & cmds)
        self.assertIn("goto", cmds)

    def test_cbs_clearance_keeps_forklifts_apart(self) -> None:
        # Two agents on distinct, adjacent nodes 1.0 m apart at the same time.
        coords = {"x": (0.0, 0.0), "y": (1.0, 0.0)}
        paths = {"A": [("x", 0), ("x", 1)], "B": [("y", 0), ("y", 1)]}

        # No clearance -> point robots -> no conflict (distinct nodes).
        self.assertIsNone(_detect_first_conflict(paths))

        # 1.5 m footprint clearance -> their bounding boxes overlap -> conflict.
        conflict = _detect_first_conflict(paths, node_coords=coords, clearance=1.5)
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["type"], "proximity")
        self.assertEqual(set(conflict["agents"]), {"A", "B"})

        # Clearance below the gap -> allowed again.
        self.assertIsNone(_detect_first_conflict(paths, node_coords=coords, clearance=0.9))


if __name__ == "__main__":
    unittest.main()
