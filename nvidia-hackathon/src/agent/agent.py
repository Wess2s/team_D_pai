"""
The agent loop — NIM + tool calling, with an offline fallback.

A minimal, dependency-light ReAct-style loop you can run against a local NIM. When the NIM
is unreachable (e.g. the DGX is off outside the event), it transparently falls back to a
rule-based intent parser (offline_intent.py) so the whole agent → tools → sim → telemetry
loop stays fully demoable offline. When NIM is up, the LLM drives.

Run:  python -m src.agent.agent "move pallet 1 to stage 1"
"""
from __future__ import annotations

import json
import sys

from . import offline_intent
from .tools import TOOL_SCHEMAS, call_tool
from . import tools as _tools

try:
    from .nim_client import chat
except Exception:  # openai import issues shouldn't kill the offline path
    chat = None

SYSTEM_PROMPT = """You are FleetMind, a warehouse fleet coordinator controlling a fleet of
autonomous forklifts in a simulated warehouse. You move pallets from rack pick-faces to
staging zones.

Translate operator commands into tool calls:
- Report state with get_fleet_status.
- To relocate a single pallet, call move_pallet(pallet, zone) — it picks the nearest free
  forklift automatically and chains pick+drop. Pass robot only if the operator names one.
- For "clear the racks" / "move everything" / "optimise the fleet", call
  optimize_and_dispatch — it uses NVIDIA cuOpt to assign pallets to forklifts (roadmap-aware
  VRP) and Conflict-Based Search to plan collision-free paths, then dispatches the whole job.
- Preview a plan without acting via plan_routes.
- Return idle trucks with send_home. Check risk with detect_conflict. Mark incidents with
  block_zone.

After acting, summarise what you did in one or two plain-English sentences for the operator."""


def _solver_label(solver: str | None) -> str:
    """Operator-facing name for the optimiser that produced a plan."""
    if solver == "cuopt":
        return "NVIDIA cuOpt"
    if solver == "local":
        return "cuOpt local fallback"
    return "cuOpt"


def _narrate_offline(message: str, results: list[tuple[str, dict, str]]) -> str:
    """Build a short operator-facing summary from executed offline steps."""
    if not results:
        return "No action taken."
    if len(results) == 1 and results[0][0] == "get_fleet_status":
        snap = json.loads(results[0][2])
        return "Fleet status:\n" + snap.get("summary", "(no state)")
    parts = []
    for name, kwargs, out in results:
        if name == "move_pallet":
            data = json.loads(out)
            if not data.get("ok"):
                parts.append(data.get("error", "move failed"))
            else:
                cbsd = data.get("cbs", {})
                asg = "; ".join(f"{fk}: {', '.join(t)}"
                                for fk, t in (_tools.LAST_PLAN.get("assignments") or {}).items())
                parts.append(
                    f"{_solver_label(data.get('solver'))} routed {asg or kwargs.get('pallet')}, "
                    f"cost {data.get('total_cost')}; CBS checked "
                    f"{cbsd.get('conflicts_found', 0)} conflict(s), "
                    f"{'all resolved' if cbsd.get('resolved') else 'staggered releases'}")
        elif name == "optimize_and_dispatch":
            data = json.loads(out)
            if not data.get("ok"):
                parts.append(data.get("error", "nothing to dispatch"))
            else:
                cbsd = data.get("cbs", {})
                n = len(data.get("dispatched", []))
                parts.append(
                    f"{_solver_label(data.get('solver'))} assigned {n} forklift(s), "
                    f"tour cost {data.get('total_cost')}; CBS checked "
                    f"{cbsd.get('conflicts_found', 0)} conflict(s), "
                    f"{'all resolved' if cbsd.get('resolved') else 'staggered releases'}")
        elif name == "plan_routes":
            data = json.loads(out)
            if not data.get("ok"):
                parts.append(data.get("error", "no plan"))
            else:
                asg = "; ".join(f"{fk}: {', '.join(t)}"
                                for fk, t in data.get("assignments", {}).items())
                parts.append(f"{_solver_label(data.get('solver'))} plan (cost "
                             f"{data.get('total_cost')}) — {asg}")
        elif name == "pick_pallet":
            parts.append(f"picking {kwargs.get('pallet')}")
        elif name == "drop_pallet":
            parts.append(f"{kwargs.get('robot')} dropping at {kwargs.get('zone')}")
        elif name == "send_home":
            parts.append(f"{kwargs.get('robot')} returning home")
        elif name == "block_zone":
            parts.append(f"blocked {kwargs.get('zone')}")
        elif name == "detect_conflict":
            conf = json.loads(results[0][2]).get("conflicts", [])
            parts.append("no conflicts detected" if not conf else "; ".join(conf))
    return "Dispatched: " + "; ".join(parts) + "."


def run_offline(user_message: str) -> str:
    """Rule-based path — used when NIM is unreachable."""
    steps = offline_intent.parse(user_message)
    results = []
    for name, kwargs in steps:
        out = call_tool(name, json.dumps(kwargs))
        print(f"  → {name}({kwargs}) = {out}")
        results.append((name, kwargs, out))
    return _narrate_offline(user_message, results)


def _nim_available() -> bool:
    if chat is None:
        return False
    try:
        chat([{"role": "user", "content": "ok"}])
        return True
    except Exception:
        return False


def run(user_message: str, max_iters: int = 8) -> str:
    """Run the agent. Uses NIM tool-calling if reachable, else the offline parser."""
    if not _nim_available():
        return run_offline(user_message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_iters):
        resp = chat(messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            result = call_tool(tc.function.name, tc.function.arguments)
            print(f"  → {tc.function.name}({tc.function.arguments}) = {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Reached max iterations without a final answer."


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "What is the status of the fleet?"
    print(f"\nOperator: {prompt}\n")
    print(f"\nAgent: {run(prompt)}\n")
