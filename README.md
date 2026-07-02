# team_D_pai — Geo-CBS Fleet Orchestrator

Natural-language control of an NVIDIA **Isaac Sim** forklift simulation. An operator
types a plain-English command (e.g. *"move blockpallet_a09 one metre to the left"*),
an LLM converts it into structured robot tasks, and those tasks are pushed into a
running Isaac Sim instance where they drive a forklift robot via USD attributes.

```
Operator text ──▶ llm_task_parser.py ──▶ RobotTask(s) ──▶ sim_bridge.py ──▶ TCP:8765 ──▶ Isaac Sim
                     (NVIDIA NIM LLM)                        (client)                    (serveractivation.py receiver)
```

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

### 1. Start the receiver inside Isaac Sim

In Isaac Sim: **Window → Script Editor**, paste the contents of
[serveractivation.py](serveractivation.py), and click **Run**. You should see:

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

## Notes

- The USD files are stored in binary crate format; open them in Isaac Sim / USD tooling.
- The receiver executes arbitrary Python inside Isaac Sim. Only run it on a trusted,
  isolated network — the TCP server binds to `0.0.0.0:8765` with no authentication.
