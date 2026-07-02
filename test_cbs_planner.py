from __future__ import annotations

import unittest

from cbs_planner import CBSConfig, plan_checkpoint_cbs


class TestCBSPlanner(unittest.TestCase):
    def test_crossing_routes_conflict_resolved(self) -> None:
        # Diamond graph with potential edge conflict if both swap in the center.
        node_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        cost_matrix = [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
        checkpoints = {
            "forklift_1": ["A", "C"],
            "forklift_2": ["C", "A"],
        }

        result = plan_checkpoint_cbs(
            checkpoints=checkpoints,
            node_index=node_index,
            cost_matrix=cost_matrix,
            blocked_nodes=[],
            blocked_edges=[],
            human_occupancy={},
            config=CBSConfig(max_neighbors=3, max_expansions=300, max_time_steps=60),
        )

        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["unresolved_conflict"])

    def test_short_mission_is_not_held_hostage_by_a_longer_one(self) -> None:
        # Full-horizon CBS should let an agent with a short, disjoint
        # mission finish on its own schedule instead of being resynced to
        # a much longer mission at every checkpoint stage.
        n = 11
        node_ids = [f"L{i}" for i in range(n)]
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        cost_matrix = [[float(abs(i - j)) for j in range(n)] for i in range(n)]

        checkpoints = {
            "fast": ["L0", "L1"],
            "slow": ["L10", "L7", "L4"],
        }

        result = plan_checkpoint_cbs(
            checkpoints=checkpoints,
            node_index=node_index,
            cost_matrix=cost_matrix,
            blocked_nodes=[],
            blocked_edges=[],
            human_occupancy={},
            config=CBSConfig(max_neighbors=2, max_time_steps=60, goal_hold_steps=1, max_expansions=200),
        )

        self.assertEqual(result["status"], "success")
        fast_steps = result["agent_paths"]["fast"][-1][1]
        slow_steps = result["agent_paths"]["slow"][-1][1]
        self.assertLessEqual(fast_steps, 2)
        self.assertGreater(slow_steps, fast_steps)

    def test_unreachable_reports_error(self) -> None:
        node_index = {"A": 0, "B": 1}
        cost_matrix = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        checkpoints = {
            "forklift_1": ["A", "B"],
        }

        result = plan_checkpoint_cbs(
            checkpoints=checkpoints,
            node_index=node_index,
            cost_matrix=cost_matrix,
            blocked_nodes=["B"],
            blocked_edges=[],
            human_occupancy={},
            config=CBSConfig(max_expansions=30, max_time_steps=20),
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("reason", result)


if __name__ == "__main__":
    unittest.main()
