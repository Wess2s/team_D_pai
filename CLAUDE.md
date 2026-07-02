# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when operating in this repository.

## Project overview

**team_D_pai — Geo-CBS Fleet Orchestrator.** Natural-language + optimization-driven
control of an NVIDIA Isaac Sim forklift fleet. All planning modules live at the **repo
root** (this is the canonical, complete version — the former `codigo_cuopt/` subset has
been removed).

The **real** simulation is the FleetMind warehouse (`nvidia-hackathon/scenes/scene_exec.py`),
which runs headless in Isaac Sim and exposes an **HTTP bridge on `:8080`**
(`GET /state`, `POST /mission | /goto | /pick | /drop | /home | /block_zone`). The scene
has 2 forklifts (`AMR_1`, `AMR_2`), 6 pallets (`WH_Palette_01..06`), 3 staging zones
(`stage_1..3`) and a real nav graph (~249 nodes / 454 edges) surfaced in `/state`.

Primary pipeline (drives the real sim):

```
GET /state ─► live_state_adapter ─► Node/Vehicle/Job (real ids)
           ─► warehouse_graph.build_graph_from_state (real nav-graph routing)
           ─► cuopt_adapter (Mock or Http) ─► CuOptOutput
           ─► mission_translator ─► MissionPlan(NavigationCommand[])
           ─► cbs_integration.deconflict_mission (CBS over the real grid)
           ─► isaac_dispatch (POST /mission)  OR  cbs_executor (step /goto)
```

`fleet_orchestrator.py` is the end-to-end entry point.

## Integration / live-sim modules (root)

| File | Role |
| --- | --- |
| `fleet_orchestrator.py` | **Entry point.** `fetch → build inputs → graph → cuOpt → translate → CBS → dispatch`. CLI flags: `--dry-run`, `--include-busy`, `--objective`, `--jobs`, `--config`, `--execute {mission,cbs}`. |
| `live_state_adapter.py` | `fetch_state()` (`GET /state`) + `build_nodes_vehicles_jobs()` — turns the live snapshot into cuOpt contracts, reusing real ids (filters busy forklifts / carried / delivered pallets / blocked zones). |
| `isaac_dispatch.py` | `mission_to_steps()` folds `NavigationCommand`s into `[["pick",p],["drop",z]]`; `dispatch_mission()` POSTs them to `/mission`. |
| `cbs_integration.py` | `mission_to_grid_checkpoints()` snaps mission stops to real nav-graph nodes; `deconflict_mission()` runs CBS over the real grid; `cbs_dispatch_order()` orders vehicles by CBS finish time. |
| `cbs_executor.py` | Opt-in faithful executor: `plan_execution_events()` (pure) + `execute_cbs_paths()` steps forklifts through CBS grid paths via `POST /goto`, issuing `/pick`/`/drop` at checkpoints. |
| `env_config.py` / `fleet_config.json` | External config (base URL, naming conventions, nav-graph usage, vehicle defaults, cuOpt endpoint envs, CBS/replan params). Nothing hardcoded into algorithms. |

## cuOpt / CBS planning modules (root)

Data flow: **scene snapshot → warehouse graph → cuOpt input → solver → mission plan → CBS → execution**

| File | Responsibility |
| --- | --- |
| `logistics_models.py` | Core typed contracts — `Node`, `Vehicle`, `Job`, `CuOptInput/Output`, `MissionPlan`, `NavigationCommand`, `ExecutionReport`. Single source of truth. |
| `isaac_scene_extractor.py` | `extract_from_state_snapshot(state)` reads the live `/state` (real path); `extract_from_isaac_stage(stage)` reads a USD stage; `build_mock_warehouse_snapshot()` for offline tests. |
| `warehouse_graph.py` | `build_warehouse_graph()` (euclidean), `build_graph_from_state()` (entity-only, real nav-graph routing via Dijkstra), `build_grid_graph_from_state()` (full grid for CBS), `nearest_grid_node()`. |
| `cuopt_adapter.py` | `build_cuopt_input()` + `MockCuOptAdapter` (regret-2 insertion + local-search VRPTW heuristic) and `HttpCuOptAdapter` (real NVIDIA cuOpt schema, async submit/poll). |
| `mission_translator.py` | Converts `CuOptOutput` routes into `NavigationCommand` lists (`navigate`, `pickup`, `dropoff`, `charge`). |
| `cbs_planner.py` | Full-horizon multi-agent Conflict-Based Search (`plan_checkpoint_cbs`): vertex/edge conflicts, bypass optimization, blocked nodes/edges, human-occupancy windows. |
| `planner.py` | `run_planner_loop()` — solver select → CBS → mission translate → `route_executor` → replan payload. |
| `task_allocator.py` | Weighted pre-allocation policy (priority + distance + battery). |
| `state_manager.py` | Runtime `WorldState` model for agents/jobs/events. |
| `route_executor.py` | `execute_mission_dry_run()` (simulate) and `execute_mission_on_isaac()` (dispatch to the live sim). |
| `replanner.py` | `detect_replan_triggers(state)` (path_blocked / object / conflict / blocked zone / pallet moved / low battery) + `build_replan_request()`. |
| `runtime_config.py` | JSON/env config for `planner.py` (cuOpt service toggle, CBS/replan). |
| `factory_simulator.py` | Offline incident simulation → `ExecutionReport` KPI. |
| `cuopt.py` / `demo_runner.py` / `demo_visualizer.py` | Offline mock E2E demo, CLI runner and ASCII visualiser. |

## Legacy / superseded

| File | Status |
| --- | --- |
| `llm_task_parser.py` | LLM parser (NVIDIA NIM) → `RobotTask`. Uses old vocab (`blockpallet_*`, `Buffer_*`). Not wired to the live HTTP sim. |
| `sim_bridge.py`, `serveractivation.py` | **Superseded.** TCP `:8765` + `/World/forklift` USD-attribute bridge — does NOT match the deployed FleetMind scene (HTTP `:8080`, `/World/AMRs/AMR_1`). Kept for reference. |

## Common development commands

```bash
# Live orchestrator (needs the FleetMind sim on :8080)
python3 fleet_orchestrator.py --dry-run                 # preview steps, no POST
python3 fleet_orchestrator.py                           # dispatch idle forklifts
python3 fleet_orchestrator.py --include-busy            # include non-idle forklifts
python3 fleet_orchestrator.py --objective min_makespan  # min_distance | min_time | min_makespan
python3 fleet_orchestrator.py --jobs jobs.json          # [{ "pallet","zone" }] overrides
python3 fleet_orchestrator.py --execute cbs             # step CBS grid paths via /goto

# Offline mock E2E demo (no Isaac)
python3 cuopt.py
python3 demo_runner.py --scenario all

# Tests (all offline; test_live_integration uses a captured /state fixture)
python3 -m unittest test_cuopt_flow test_cuopt_adapter test_cbs_planner test_live_integration -v
```

## Key configuration

**fleet_config.json** (loaded by `env_config.load_fleet_config`): `base_url` (default
`http://localhost:8080`), `naming` (prim/id conventions), `graph.use_nav_graph`,
`vehicles` (speed/capacity/battery defaults — the live telemetry has **no** battery),
`cuopt.endpoint_env`/`api_key_env`, `cbs`, `replan`.

**Real cuOpt:** set `CUOPT_URL` (+ optional `CUOPT_API_KEY`) → `HttpCuOptAdapter`;
otherwise `MockCuOptAdapter`.

**cuOpt contracts** (logistics_models.py): Node types `storage`/`dock`/`charging`/`waypoint`/`depot`;
objectives `min_distance`/`min_time`/`min_makespan`.

## Architecture notes

- The orchestrator sends only **semantic** targets to `/mission`; the sim's bridge does the
  A* routing internally. For dispatch, entity ids are reused as cuOpt ids (`AMR_1`,
  `WH_Palette_01`, `stage_1`) so no name translation is needed.
- **CBS** runs over the **real** nav grid (`build_grid_graph_from_state`), so vertex/edge
  conflicts are resolved on actual aisles. In the default `--execute mission` mode CBS output
  orders/staggers dispatch (advisory). `--execute cbs` steps grid nodes via `/goto` to honour
  the deconflicted timing (best-effort: sim motion is continuous, so CBS timesteps act as an
  ordering barrier, not exact seconds).
- **No battery telemetry** in the real scene — battery values come from config and are
  advisory; `detect_replan_triggers` never false-positives on it.
- USD files (`forklift.usd`, `omniverse3_0.usd`) are binary crates — open in Isaac Sim / USD toolbox.
