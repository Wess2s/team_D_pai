from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuopt import run_demo_pipeline
from demo_visualizer import render_ascii_map, render_route_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick cuOpt warehouse demo runner")
    parser.add_argument(
        "--scenario",
        choices=["single_forklift", "dual_forklift", "urgent_jobs", "factory_realistic", "all"],
        default="all",
        help="Scenario to execute",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output folder for JSON result files",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Disable ASCII map rendering",
    )
    return parser.parse_args()


def _run_and_print(scenario: str, json_out: str, show_map: bool) -> None:
    result = run_demo_pipeline(scenario=scenario)  # type: ignore[arg-type]
    print("\n" + "=" * 80)
    print(render_route_summary(result))
    if show_map:
        print("\n" + render_ascii_map(result))

    if json_out:
        out_dir = Path(json_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"demo_result_{scenario}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_file}")


def main() -> None:
    args = _parse_args()
    scenarios = [args.scenario]
    if args.scenario == "all":
        scenarios = ["single_forklift", "dual_forklift", "urgent_jobs", "factory_realistic"]

    for scenario in scenarios:
        _run_and_print(scenario=scenario, json_out=args.json_out, show_map=not args.no_map)


if __name__ == "__main__":
    main()
