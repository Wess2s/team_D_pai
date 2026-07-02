"""
isaac_dispatch.py
-----------------
Executes a solved cuOpt ``MissionPlan`` on the running FleetMind Isaac Sim by
translating each vehicle's ``NavigationCommand`` list into the sim bridge's
``POST /mission`` contract:

    steps = [["pick", "WH_Palette_01"], ["drop", "stage_1"], ...]

The sim's HTTP bridge (``:8080``) already handles A* routing / waypoint
following internally, so only the *semantic* pick/drop targets are sent — the
``navigate`` and ``charge`` commands are folded away.

This is the real-execution counterpart to ``route_executor.execute_mission_dry_run``.
"""
from __future__ import annotations

import json
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "http://localhost:8080"


def mission_to_steps(commands: list[dict[str, Any]]) -> list[list[str]]:
    """Fold a vehicle's NavigationCommand dicts into ``/mission`` steps.

    ``pickup`` -> ``["pick", <pallet_id>]`` and ``dropoff`` -> ``["drop", <zone>]``.
    ``navigate`` / ``charge`` commands are dropped (the bridge routes internally).
    """
    steps: list[list[str]] = []
    for cmd in commands:
        ctype = cmd.get("command_type")
        if ctype == "pickup":
            target = cmd.get("pallet_id") or cmd.get("node_id")
            if target:
                steps.append(["pick", target])
        elif ctype == "dropoff":
            target = cmd.get("node_id")
            if target:
                steps.append(["drop", target])
    return steps


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (trusted local sim)
        body = resp.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def dispatch_mission(
    mission_plan: dict[str, Any],
    base_url: str = DEFAULT_BASE_URL,
    *,
    dry_run: bool = False,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Send every vehicle's folded mission to the sim via ``POST /mission``.

    When ``dry_run`` is True the steps are computed and returned but no HTTP
    request is made.
    """
    mission_url = base_url.rstrip("/") + "/mission"
    dispatched: list[dict[str, Any]] = []
    errors: list[str] = []

    for vehicle_id, commands in (mission_plan.get("vehicle_commands") or {}).items():
        steps = mission_to_steps(commands)
        if not steps:
            continue

        record: dict[str, Any] = {"robot": vehicle_id, "steps": steps}
        if dry_run:
            record["status"] = "dry_run"
        else:
            try:
                record["response"] = _post_json(
                    mission_url, {"robot": vehicle_id, "steps": steps}, timeout_s
                )
                record["status"] = "dispatched"
            except (error.URLError, TimeoutError) as exc:
                record["status"] = "error"
                record["error"] = str(exc)
                errors.append(f"{vehicle_id}: {exc}")
        dispatched.append(record)

    return {
        "status": "ok" if not errors else "partial",
        "dry_run": dry_run,
        "dispatched": dispatched,
        "errors": errors,
    }


def fetch_state(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 5.0) -> dict[str, Any]:
    """Convenience re-export so callers can poll telemetry without a second import."""
    url = base_url.rstrip("/") + "/state"
    with request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 (trusted local sim)
        return json.loads(resp.read().decode())
