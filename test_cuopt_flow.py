from __future__ import annotations

import json
import unittest

from cuopt import run_demo_pipeline


class TestCuOptFlow(unittest.TestCase):
    def test_mock_end_to_end_pipeline(self) -> None:
        result = run_demo_pipeline(scenario="dual_forklift")

        self.assertIn("cuopt_input", result)
        self.assertIn("cuopt_output", result)
        self.assertIn("mission_plan", result)

        cuopt_input = result["cuopt_input"]
        cuopt_output = result["cuopt_output"]
        mission_plan = result["mission_plan"]

        self.assertGreaterEqual(len(cuopt_input["nodes"]), 5)
        self.assertGreaterEqual(len(cuopt_input["vehicles"]), 1)
        self.assertEqual(len(cuopt_input["jobs"]), 3)

        self.assertEqual(cuopt_output["status"], "success")
        self.assertGreater(cuopt_output["total_cost"], 0.0)

        all_assigned_jobs = []
        for route in cuopt_output["vehicle_routes"]:
            all_assigned_jobs.extend(route["assigned_job_ids"])
        self.assertCountEqual(all_assigned_jobs, ["job_001", "job_002", "job_003"])

        self.assertEqual(mission_plan["status"], "success")
        self.assertIn("forklift_1", mission_plan["vehicle_commands"])

    def test_single_forklift_scenario(self) -> None:
        result = run_demo_pipeline(scenario="single_forklift")
        vehicles = result["cuopt_input"]["vehicles"]
        self.assertEqual(len(vehicles), 1)
        self.assertEqual(result["cuopt_output"]["status"], "success")

    def test_urgent_jobs_scenario(self) -> None:
        result = run_demo_pipeline(scenario="urgent_jobs")
        priorities = [job["priority"] for job in result["cuopt_input"]["jobs"]]
        self.assertIn(10, priorities)
        self.assertEqual(result["cuopt_output"]["status"], "success")

    def test_factory_realistic_scenario(self) -> None:
        result = run_demo_pipeline(scenario="factory_realistic")
        self.assertEqual(result["cuopt_input"]["objective"], "min_makespan")
        self.assertIn("execution_report", result)
        self.assertIn("events", result["execution_report"])
        self.assertGreater(len(result["cuopt_input"]["jobs"]), 3)
        self.assertIn(result["execution_report"]["status"], ["ok", "degraded", "failed"])

    def test_pipeline_json_serializable(self) -> None:
        result = run_demo_pipeline()
        encoded = json.dumps(result)
        self.assertTrue(encoded.startswith("{"))


if __name__ == "__main__":
    unittest.main()
