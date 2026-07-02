# team_D_pai

## Isaac Sim -> cuOpt Integration (Hackathon MVP)

This repository now contains a minimum but extensible pipeline for logistics planning:

Natural language/UI command
-> Planner
-> Logical logistics jobs
-> Warehouse graph abstraction from Isaac Sim
-> cuOpt input
-> cuOpt route solution
-> Mission plan
-> Isaac Sim navigation commands / visualization

## Key Principle

cuOpt should not receive the full Isaac Sim scene.

Isaac Sim remains the source of physical truth and visualization, but cuOpt receives only an optimizable abstraction:

- nodes/waypoints/zones
- distance or cost matrix
- vehicles and capacities
- pickup/delivery jobs
- constraints and objective

## Modules

- `isaac_scene_extractor.py`
  - extracts relevant scene entities from Isaac Sim stage into a lightweight snapshot
  - includes `build_mock_warehouse_snapshot()` for local testing without Isaac runtime

- `warehouse_graph.py`
  - builds the logical graph and distance matrix

- `cuopt_adapter.py`
  - `build_cuopt_input()` creates the solver input contract
  - `MockCuOptAdapter` is a dependency-free VRPTW heuristic: regret-2
    insertion construction (processed in descending priority tiers) followed
    by a relocate/swap/or-opt local search pass. Every candidate move is
    re-validated for time windows, battery reserve, blocked nodes/edges and
    max route time before being accepted, so the improvement pass can never
    produce an infeasible route. See `test_cuopt_adapter.py` for a
    constructed scenario where plain greedy insertion starves a job and
    regret insertion fixes it.
  - `HttpCuOptAdapter` maps the internal contract to the real NVIDIA cuOpt
    microservice schema (`cost_matrix_data` / `task_data` / `fleet_data` /
    `solver_config`) and follows its async submit-then-poll pattern
    (`POST /cuopt/request` -> `GET /cuopt/requests/{id}`). Point
    `runtime_config.json`'s `cuopt.endpoint` / `cuopt.enabled` at a running
    cuOpt instance (self-hosted or the managed NIM endpoint) and
    `planner.py` will use it instead of the mock automatically.

- `mission_translator.py`
  - translates logical route stops into executable mission commands
  - emits `navigate`, `pickup`, `dropoff`, `charge`

- `cbs_planner.py`
  - full-horizon multi-agent Conflict-Based Search (CBS): each agent gets
    one continuous A* path across *all* of its checkpoints (pickups,
    deliveries, charging) searched in a single pass, instead of resolving
    one synchronized stage at a time. This removes the artificial barrier
    where an agent with a short mission had to wait for every other agent
    to finish its current stage before advancing to its next stop, and
    fixes a time-axis discontinuity the old per-stage design had across
    stage boundaries.
  - resolves vertex and edge conflicts by branching the constraint tree,
    with the standard CBS "bypass" optimization: a same-cost reroute that
    clears the conflict is adopted without permanently constraining the
    agent or growing the search tree
  - supports blocked aisles and human occupancy time windows

- `factory_simulator.py`
  - simulates execution-time incidents in factory operations
  - generates KPI report: on-time ratio, event severity, recommended actions

- `planner.py`
  - high-level orchestration loop for optimization + CBS + execution dry-run + replan payload

- `task_allocator.py`
  - weighted pre-allocation policy (priority + distance + battery safety)

- `state_manager.py`
  - runtime state model for agents/jobs/events

- `route_executor.py`
  - execution abstraction with dry-run mode

- `replanner.py`
  - derives replan requests from critical events and updates constraints

- `runtime_config.py`
  - JSON/env driven runtime configuration for planner and cuOpt service toggles

- `logistics_models.py`
  - typed dataclasses for all contracts and JSON serialization

- `cuopt.py`
  - orchestrates full mock E2E flow and prints traceable JSON

- `test_cuopt_flow.py`
  - minimum end-to-end test

## cuOpt Input Contract

`CuOptInput` includes:

- `nodes`: id, node_type, physical coordinates
- `node_index`: mapping `node_id -> matrix_index`
- `cost_matrix`: travel cost or distance matrix
- `vehicles`: start node, capacity, charging node
- `jobs`: pickup, delivery, priority, demand, service time
- `jobs` time windows: pickup and delivery allowed intervals
- `constraints`: basic constraints placeholder
- `constraints` advanced: blocked edges/nodes, occupancy buffer, max route time
- `objective`: `min_distance`, `min_time`, or `min_makespan`

## cuOpt Output Contract

`CuOptOutput` includes:

- `status`: `success` or `error`
- `vehicle_routes`: per-vehicle route with ordered stops
- `assigned_job_ids`: job assignment traceability
- `total_cost`: total objective cost
- `errors`: diagnostics

## Mission Translation

`mission_translator.py` converts logical route stops into commands that a planner/controller/UI can execute:

- `navigate` to physical waypoint/zone
- `pickup` pallet at storage
- `dropoff` pallet at dock/target
- `charge` at charging node

These commands can be consumed by:

- a forklift controller
- existing Isaac Sim bridge scripts
- a mission visualization UI

## Factory-Grade Enhancements Included

- collision-aware temporal scheduling via node reservation windows
- CBS conflict resolution for multi-forklift path synchronization
  - vertex conflict: two robots in same node at same time
  - edge conflict: head-on swap in opposite directions
- blocked aisle/edge constraints
- pickup and delivery time windows
- battery-aware assignment checks
- route time guardrails (`max_route_time_s`)
- dynamic human occupancy windows in critical nodes (dock/intersections)
- execution simulation with realistic incidents:
  - human crossing near dock
  - temporary aisle obstruction
  - low-battery late-charge warnings
- operational KPI output (`execution_report`)

## CBS Output Contract

`cbs_output` includes:

- `status`: `success` or `error`
- `agent_paths`: per-vehicle time-indexed node sequence
- `conflicts_resolved`: number of CBS conflicts branched and resolved
- `unresolved_conflict`: remaining conflict object when planner cannot converge

## Planner Loop Output Contract

`planner_loop` includes:

- `cuopt_output`: optimization result in the same internal schema
- `mission_plan`: executable command list with timing metadata
- `cbs_output`: collision-conflict resolution output
- `execution_dry_run`: deterministic execution simulation output
- `replan_request`: generated payload for adaptive re-optimization when needed

## Run Demo

```bash
python cuopt.py
```

This prints a full JSON payload containing:

- extracted/mock scene snapshot
- warehouse graph and distance matrix
- generated cuOpt input
- mock cuOpt solution
- final mission plan ready for Isaac/UI integration

## Quick Visual Demo (Recommended)

Run all scenarios with a simple ASCII map and route summary:

```bash
python demo_runner.py --scenario all
```

Run one scenario:

```bash
python demo_runner.py --scenario single_forklift
python demo_runner.py --scenario dual_forklift
python demo_runner.py --scenario urgent_jobs
python demo_runner.py --scenario factory_realistic
```

Save JSON outputs for each scenario:

```bash
python demo_runner.py --scenario all --json-out demo_outputs
```

Disable map rendering if you only want textual summary:

```bash
python demo_runner.py --scenario all --no-map
```

## Run Tests

```bash
python -m unittest -v
```

Run CBS-focused tests only:

```bash
python -m unittest -v test_cbs_planner.py
```

## Next Step for Real cuOpt

`HttpCuOptAdapter` already speaks the real NVIDIA cuOpt request/response
schema. To switch on it:

```bash
export CUOPT_ENDPOINT="https://<your-cuopt-host>"
export CUOPT_API_KEY="<key>"
export CUOPT_ENABLED=true
python cuopt.py
```

or set `cuopt.endpoint` / `cuopt.enabled` / `cuopt.api_key_env` in
`runtime_config.json`. `planner.py` selects between `HttpCuOptAdapter` and
`MockCuOptAdapter` automatically based on that config; the rest of the
pipeline (CBS, mission translator, execution) is unaffected either way.

## NVIDIA NIM Integration

`llm_task_parser.py` calls an NVIDIA NIM-hosted LLM (`https://integrate.api.nvidia.com/v1`)
to turn natural-language operator commands into structured `RobotTask`
objects, which `sim_bridge.py` then dispatches into Isaac Sim over the
`serveractivation.py` TCP bridge. Set `NVIDIA_API_KEY` from
<https://build.nvidia.com> before using it — no key ships with this repo.

## Solver and Planner Quality Notes

- `cuopt_adapter.py`'s `MockCuOptAdapter` and `cbs_planner.py`'s
  `plan_checkpoint_cbs` are both capped (job/agent count, expansion/rebuild
  budgets) so they stay fast at hackathon scale. For production-scale
  fleets, point `HttpCuOptAdapter` at a real GPU-accelerated cuOpt instance.
- Run `python -m unittest -v` to see the regression tests, including two
  that specifically demonstrate the improvements: a starved-job scenario
  that plain greedy insertion fails and regret insertion solves
  (`test_cuopt_adapter.py`), and a short-mission-vs-long-mission scenario
  showing CBS no longer resyncs an agent's short path to a much longer one
  (`test_cbs_planner.py`).
