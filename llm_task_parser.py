cat > llm_task_parser.py << 'EOF'

"""
llm_task_parser.py
------------------
Step 1 of the Geo-CBS Fleet Orchestrator UI.

Converts a natural-language operator command into a list of structured
RobotTask objects that backup.py / ForkliftBehaviourBase can execute via
its USD attributes (navPalletToPickId, navAreaToDrop, navGoToPick, etc.).

Requires:
    pip install openai
    export NVIDIA_API_KEY=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>

Quick test:
    python llm_task_parser.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Literal

from openai import OpenAI

# ── Task schema ────────────────────────────────────────────────────────────

ActionType = Literal["pick", "drop", "move", "go_home", "stop"]


@dataclass
class RobotTask:
    """One structured task for one forklift robot.

    Mapping to backup.py USD attributes:
        action="pick"      -> navPalletToPickId = pallet_id, navGoToPick = True
        action="drop"      -> navAreaToDrop     = drop_area_id, navGoToDrop = True
        action="move"      -> pick then drop/offset (UI layer resolves offset -> drop pos)
        action="go_home"   -> navGoHome = True
        action="stop"      -> all nav bools = False
    """
    action:       ActionType  # what to do
    robot_id:     str         # forklift prim name, e.g. "forklift_b_sensor"
    pallet_id:    str = ""    # navPalletToPickId, e.g. "blockpallet_b02"
    drop_area_id: str = ""    # navAreaToDrop, e.g. "Buffer_A" or "Rack_01"
    offset_x:     float = 0.0 # world-X metres to move (positive = right/east)
    offset_y:     float = 0.0 # world-Y metres to move (positive = forward/north)
    notes:        str = ""    # LLM explanation or ambiguity note


# ── System prompt ──────────────────────────────────────────────────────────
#
# Kept compact and deterministic.  Enumerate real stage prim names so the LLM
# never invents values that won't resolve in USD.

_SYSTEM_PROMPT = """
You are a warehouse robot task planner for an Isaac Sim forklift simulation.
Convert the operator's natural-language command into a JSON array of tasks.

AVAILABLE ROBOTS:
  ["forklift_b_sensor"]

AVAILABLE PALLETS  (use exact id strings):
  ["blockpallet_b02", "blockpallet_a06", "blockpallet_c01", "blockpallet_a09"]

AVAILABLE DROP AREAS  (use exact id strings):
  ["Buffer_A", "Buffer_B", "Rack_01", "Rack_02", "Dock_01"]

TASK SCHEMA (every element of the array must match this):
{
  "action":       "pick" | "drop" | "move" | "go_home" | "stop",
  "robot_id":     string,   // which forklift
  "pallet_id":    string,   // required for pick and move
  "drop_area_id": string,   // required for drop; omit when using offset
  "offset_x":     number,   // metres along world X  (+= right/east, -= left/west)
  "offset_y":     number,   // metres along world Y  (+= forward/north, -= back/south)
  "notes":        string    // brief explanation or uncertainty note
}

RULES:
1. "move <pallet> N metres <direction>" -> action="move", set pallet_id and the
   matching offset_x / offset_y. Do NOT set drop_area_id when using an offset.
2. "put / drop <pallet> in/at <area>"  -> action="move", set pallet_id and drop_area_id.
3. "pick up <pallet> and drop it in <area>" -> action="move", set BOTH pallet_id AND
   drop_area_id. Never split this into two tasks.
4. "pick up <pallet>" (no destination)  -> action="pick", set pallet_id only.
5. "go home" / "return"                -> action="go_home".
6. "stop" / "halt"                     -> action="stop".
7. If the pallet is not named, default to "blockpallet_b02" and note the ambiguity.
8. If the drop area is not named and no offset is given, default to "Buffer_A" and note it.
9. Return ONLY the JSON array. No extra text, no markdown fences.
""".strip()


# ── NIM client ─────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY", "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")
    if not api_key:
        raise EnvironmentError(
            "NVIDIA_API_KEY is not set.\n"
            "Get a free key at https://build.nvidia.com (click 'Get API Key') and run:\n"
            "  export NVIDIA_API_KEY='nvapi-...'"
        )
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )


# ── Parser ─────────────────────────────────────────────────────────────────

def parse_command(
    command: str,
    model: str = "meta/llama-3.1-8b-instruct",
) -> list[RobotTask]:
    """Parse a natural-language command into structured RobotTask objects.

    Args:
        command: e.g. "move pallet 1 one metre to the right"
        model:   NIM model name. llama-3.1-8b is fast; swap for
                 nvidia/llama-3.1-nemotron-70b-instruct for higher accuracy.

    Returns:
        List of RobotTask objects ready to dispatch to the forklift.

    Raises:
        ValueError:       LLM returned malformed or unexpected JSON.
        EnvironmentError: NVIDIA_API_KEY is not set.
        openai.APIError:  Network / auth failure.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": command},
        ],
        temperature=0.0,   # deterministic output
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Parse and normalise: model may return {"tasks": [...]} or directly [...]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw output:\n{raw}") from exc

    if isinstance(parsed, dict):
        # Unwrap common wrapper keys
        for key in ("tasks", "result", "commands", "actions"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            # Single task returned as a dict — wrap it
            parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array from LLM, got {type(parsed).__name__}:\n{raw}")

    tasks: list[RobotTask] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        tasks.append(RobotTask(
            action       = item.get("action", "stop"),
            robot_id     = item.get("robot_id", "forklift_b_sensor"),
            pallet_id    = item.get("pallet_id", ""),
            drop_area_id = item.get("drop_area_id", ""),
            offset_x     = float(item.get("offset_x", 0.0)),
            offset_y     = float(item.get("offset_y", 0.0)),
            notes        = item.get("notes", ""),
        ))

    return tasks


# ── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_COMMANDS = [
        "move pallet 1 one metre to the right",
        "pick up blockpallet_a06 and drop it in Rack_01",
        "move blockpallet_a09 two metres forward and one metre to the left",
        "send the forklift home",
        "stop everything",
    ]

    for cmd in TEST_COMMANDS:
        print(f"\nCommand : {cmd!r}")
        try:
            tasks = parse_command(cmd)
            for t in tasks:
                print(json.dumps(asdict(t), indent=2))
        except EnvironmentError as exc:
            print(f"  CONFIG ERROR: {exc}")
            break
        except Exception as exc:
            print(f"  ERROR: {exc}")
EOF