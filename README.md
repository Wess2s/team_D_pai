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
  - `MockCuOptAdapter` provides deterministic mock solving without external dependencies
  - `HttpCuOptAdapter` is a stable placeholder for real cuOpt endpoint integration

- `mission_translator.py`
  - translates logical route stops into executable mission commands
  - emits `navigate`, `pickup`, `dropoff`, `charge`

- `factory_simulator.py`
  - simulates execution-time incidents in factory operations
  - generates KPI report: on-time ratio, event severity, recommended actions

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
- blocked aisle/edge constraints
- pickup and delivery time windows
- battery-aware assignment checks
- route time guardrails (`max_route_time_s`)
- execution simulation with realistic incidents:
  - human crossing near dock
  - temporary aisle obstruction
  - low-battery late-charge warnings
- operational KPI output (`execution_report`)

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

## Next Step for Real cuOpt

Replace `MockCuOptAdapter` with `HttpCuOptAdapter` and map the payload to the exact cuOpt API schema. Keep the same internal contracts so planner, translator, and execution layers remain unchanged.
