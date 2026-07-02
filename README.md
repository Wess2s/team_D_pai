# team_D_pai — Geo-CBS Fleet Orchestrator

Natural-language control of an NVIDIA **Isaac Sim** forklift simulation. An operator
types a plain-English command (e.g. *"move blockpallet_a09 one metre to the left"*),
an LLM converts it into structured robot tasks, and those tasks are pushed into a
running Isaac Sim instance where they drive a forklift robot via USD attributes.

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

## Repository layout

| File | Description |
| --- | --- |
| [llm_task_parser.py](llm_task_parser.py) | **Step 1.** Converts a natural-language command into a list of structured `RobotTask` objects using an NVIDIA NIM-hosted LLM (`meta/llama-3.1-8b-instruct`). Defines the task schema and the system prompt that enumerates valid robots, pallets, and drop areas. |
| [sim_bridge.py](sim_bridge.py) | **Step 2.** Translates each `RobotTask` into a Python snippet that sets USD attributes on the forklift prim, then sends it to a running Isaac Sim over a TCP socket (`localhost:8765`). |
| [serveractivation.py](serveractivation.py) | **Receiver.** The one-time snippet you paste into Isaac Sim's *Script Editor*. It starts a TCP server on port `8765` inside Isaac Sim's Python process and `exec()`s incoming scripts on the main thread. |
| [forklift.usd](forklift.usd) | USD (binary crate) asset for the forklift robot. |
| [omniverse3_0.usd](omniverse3_0.usd) | USD (binary crate) scene/stage. |
| README.md | This document. |

## How it works

1. **Parse** — `parse_command(text)` in [llm_task_parser.py](llm_task_parser.py) sends the
   operator's command to the LLM and returns a list of `RobotTask` dataclasses.
2. **Dispatch** — `dispatch_task(task)` in [sim_bridge.py](sim_bridge.py) builds a USD-manipulation
   script for the task's action and sends it over TCP to Isaac Sim.
3. **Execute** — the receiver from [serveractivation.py](serveractivation.py), running inside
   Isaac Sim, executes the script on the main thread, setting navigation attributes on the
   `/World/forklift` prim.
4. **Drive** — the forklift behaviour (running inside the sim) reads those attributes each
   frame and moves the robot.

### RobotTask schema

Defined in [llm_task_parser.py](llm_task_parser.py):

| Field | Type | Meaning |
| --- | --- | --- |
| `action` | `pick` \| `drop` \| `move` \| `go_home` \| `stop` | What the forklift should do |
| `robot_id` | str | Forklift prim name (e.g. `forklift_b_sensor`) |
| `pallet_id` | str | Pallet to pick (e.g. `blockpallet_a09`) |
| `drop_area_id` | str | Named drop destination (e.g. `Buffer_A`, `Rack_01`) |
| `offset_x` | float | Metres along world X (+ right/east, − left/west) |
| `offset_y` | float | Metres along world Y (+ forward/north, − back/south) |
| `notes` | str | LLM explanation or ambiguity note |

### Known stage identifiers

- **Robots:** `forklift_b_sensor`
- **Pallets:** `blockpallet_b02`, `blockpallet_a06`, `blockpallet_c01`, `blockpallet_a09`
- **Drop areas:** `Buffer_A`, `Buffer_B`, `Rack_01`, `Rack_02`, `Dock_01`

## Prerequisites

- NVIDIA Isaac Sim (tested with the `nvcr.io/nvidia/isaac-sim:6.0.1` container).
- Python 3.10+ with the OpenAI client:
  ```bash
  pip install openai
  ```
- An NVIDIA API key for NVIDIA NIM (build.nvidia.com), exported as an environment variable:
  ```bash
  export NVIDIA_API_KEY='nvapi-...'
  ```

> **Security:** do not commit API keys to source. Read the key from `NVIDIA_API_KEY`
> only, and rotate any key that has ever been hardcoded or shared.

## Usage

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

```
[SimBridge] Listening on 8765
[SimBridge] Server started
```

### 2. Send a command from your host

```python
from llm_task_parser import parse_command
from sim_bridge import dispatch_task

tasks = parse_command("move blockpallet_a09 one metre to the left")
for t in tasks:
    print(dispatch_task(t))
```

### 3. Run the built-in tests

```bash
# Parser only (needs NVIDIA_API_KEY)
python llm_task_parser.py

# End-to-end dispatch (needs the receiver running in Isaac Sim)
python sim_bridge.py
```

## Configuration

`sim_bridge.py` exposes these constants at the top of the file:

| Constant | Default | Purpose |
| --- | --- | --- |
| `ISAAC_SIM_HOST` | `localhost` | Host running Isaac Sim (change for a remote sim) |
| `ISAAC_SIM_PORT` | `8765` | TCP port of the in-sim receiver |
| `FORKLIFT_PATH` | `/World/forklift` | USD path of the forklift prim |
| `TIMEOUT_S` | `10` | Socket timeout in seconds |

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
