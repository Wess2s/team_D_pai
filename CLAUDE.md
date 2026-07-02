# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when operating in this repository.

## Project overview

**team_D_pai — Geo-CBS Fleet Orchestrator.** Natural-language control of an NVIDIA Isaac Sim forklift simulation. Two pipelines converge on the same USD-based sim:

1. **LLM task pipeline** (root files) — operator text → `llm_task_parser.py` (NVIDIA NIM LLM) → structured `RobotTask` objects → `sim_bridge.py` → TCP socket (`localhost:8765`) → Isaac Sim receiver (`serveractivation.py`).
2. **cuOpt logistics pipeline** (`codigo_cuopt/`) — warehouse snapshot → graph + distance matrix → cuOpt multi-vehicle routing solver → mission plan (navigate/pickup/dropoff/charge commands) → optional execution simulation.

## Top-level files (root)

| File | Role |
| --- | --- |
| `llm_task_parser.py` | LLM task parser — converts natural-language commands to `RobotTask` dataclass lists via NVIDIA NIM (`meta/llama-3.1-8b-instruct`). Defines the `ActionType` enum and system prompt with valid stage identifiers. |
| `sim_bridge.py` | TCP dispatch layer — translates each `RobotTask` into USD attribute write scripts and sends them over a socket to Isaac Sim's receiver. |
| `serveractivation.py` | One-time Isaac Sim Script Editor snippet — starts a TCP server inside Isaac Sim that `exec()`s incoming Python on the main thread. |

## codigo_cuopt/ pipeline

Data flow: **scene snapshot → warehouse graph → cuOpt input → solver → mission plan → execution report**

| File | Responsibility |
| --- | --- |
| `logistics_models.py` | Core data contracts — `Node`, `Vehicle`, `Job`, `CuOptInput/Output`, `MissionPlan`, `NavigationCommand`, `ExecutionReport` (all typed dataclasses). The single source of truth for the inter-module API. |
| `isaac_scene_extractor.py` | Extracts relevant entities (`storage`, `dock`, `charging`, `waypoint`) from an Isaac Sim USD stage; provides `build_mock_warehouse_snapshot()` for offline testing. |
| `warehouse_graph.py` | Builds a `WarehouseGraph` (node list, index, distance matrix) from nodes with optional blocked edges. |
| `cuopt_adapter.py` | `build_cuopt_input()` contract + two solver implementations: `MockCuOptAdapter` (deterministic greedy planner with battery/time-window/constraint checks) and `HttpCuOptAdapter` (HTTP POST placeholder for a real cuOpt service). |
| `mission_translator.py` | Converts `CuOptOutput` routes into `NavigationCommand` lists (`navigate`, `pickup`, `dropoff`, `charge`). |
| `factory_simulator.py` | Simulates execution-time incidents (human crossing, aisle block, low-battery) and produces an `ExecutionReport` KPI. |
| `cuopt.py` | Orchestrator — wires the full mock pipeline end-to-end; `run_demo_pipeline(scenario)` is the entry point. |
| `demo_runner.py` | CLI wrapper over `cuopt.py` with scenario selection, ASCII map rendering, and JSON output. Run: `python demo_runner.py --scenario all`. |
| `demo_visualizer.py` | Route summary + ASCII warehouse map renderer for terminal demos. |

## Common development commands

```bash
# Run the root LLM pipeline (parser only — needs NVIDIA_API_KEY)
python llm_task_parser.py

# Run the root dispatch pipeline (needs Isaac Sim receiver on port 8765)
python sim_bridge.py

# Run the cuOpt mock E2E demo
python codigo_cuopt/cuopt.py

# Run all cuOpt scenarios with ASCII maps
python codigo_cuopt/demo_runner.py --scenario all

# Save scenario JSON outputs
python codigo_cuopt/demo_runner.py --scenario all --json-out demo_outputs

# Run unittest suite (in codigo_cuopt/)
cd codigo_cuopt && python -m unittest -v
```

## Key configuration constants

**sim_bridge.py**: `ISAAC_SIM_PORT=8765`, `FORKLIFT_PATH="/World/forklift"`, `TIMEOUT_S=10`

**llm_task_parser.py**: Model is `meta/llama-3.1-8b-instruct` (deterministic via `temperature=0`). Valid pallets, drop areas, and robots are hardcoded in `_SYSTEM_PROMPT`.

**cuOpt contracts** (logistics_models.py):
- Node types: `storage`, `dock`, `charging`, `waypoint`, `depot`
- Objectives: `min_distance`, `min_time`, `min_makespan`
- Vehicle fields include battery capacity, energy/meter, time windows, max shift
- Jobs have pickup/delivery time windows and priority

## Architecture notes

- The **RobotTask schema** in the LLM pipeline (root) is a simple action dispatcher — each task maps directly to USD attribute writes on the `/World/forklift` prim. There is no multi-robot coordination.
- The **cuOpt pipeline** (`codigo_cuopt/`) handles multi-vehicle routing with battery constraints, time windows, node occupancy reservations, and blocked edges. It produces per-vehicle command lists that are *intended* to be fed into the same Isaac Sim bridge or a separate fleet controller.
- `MockCuOptAdapter` is greedy (priority-sorted job assignment) with incremental cost estimation and rollback — not an exact solver. Replace with `HttpCuOptAdapter` when a real cuOpt endpoint is available; keep the internal contracts (`CuOptInput`/`CuOptOutput`) unchanged.
- USD files (`forklift.usd`, `omniverse3_0.usd`) are binary crates — open in Isaac Sim / USD toolbox.
