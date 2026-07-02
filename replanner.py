from __future__ import annotations

from copy import deepcopy
from typing import Any


def detect_replan_triggers(
    state: dict[str, Any],
    *,
    expected_pallets: dict[str, tuple[float, float]] | None = None,
    low_battery_pct: float = 20.0,
    position_tolerance_m: float = 0.75,
) -> list[dict[str, Any]]:
    """Inspect a live ``/state`` snapshot and return concrete replan triggers.

    Each trigger is ``{"event_type", "vehicle_id", "details", "severity"}`` —
    shaped to feed straight into :func:`build_replan_request` (via the mission's
    critical-events list). Detected conditions, grounded in the real telemetry:

    * ``path_blocked`` / ``object_detected`` — per-forklift obstruction flags.
    * ``conflict`` — the bridge's live conflict overlay (``state["conflicts"]``).
    * ``aisle_block`` — a staging zone reporting ``blocked: true``.
    * ``pallet_moved`` — a pallet no longer at its expected pose (optional).
    * ``low_battery`` — only if the scene actually reports battery (it currently
      does not; kept as a forward-compatible hook, never false-positives).
    """
    triggers: list[dict[str, Any]] = []

    for fid, info in (state.get("forklifts") or {}).items():
        if info.get("path_blocked"):
            triggers.append(
                {"event_type": "aisle_block", "vehicle_id": str(fid),
                 "details": f"path_blocked near {info.get('target', 'route')}", "severity": "critical"}
            )
        obj = info.get("object_detected")
        if obj and str(obj).lower() not in {"none", ""}:
            triggers.append(
                {"event_type": "human_crossing", "vehicle_id": str(fid),
                 "details": f"object_detected {obj}", "severity": "warning"}
            )
        battery = info.get("battery_pct", info.get("battery"))
        if battery is not None and float(battery) <= low_battery_pct:
            triggers.append(
                {"event_type": "low_battery", "vehicle_id": str(fid),
                 "details": f"battery {battery}%", "severity": "warning"}
            )

    for conflict in state.get("conflicts") or []:
        triggers.append(
            {"event_type": "aisle_block", "vehicle_id": str(conflict.get("vehicle_id", "")),
             "details": f"conflict {conflict}", "severity": "critical"}
        )

    for zid, info in (state.get("zones") or {}).items():
        if info.get("blocked"):
            triggers.append(
                {"event_type": "aisle_block", "vehicle_id": "",
                 "details": f"zone blocked {zid}", "severity": "critical"}
            )

    if expected_pallets:
        for pid, (ex, ey) in expected_pallets.items():
            info = (state.get("pallets") or {}).get(pid)
            if info is None or info.get("carried_by") or info.get("delivered"):
                continue
            dx = float(info.get("x", ex)) - ex
            dy = float(info.get("y", ey)) - ey
            if (dx * dx + dy * dy) ** 0.5 > position_tolerance_m:
                triggers.append(
                    {"event_type": "replan", "vehicle_id": "",
                     "details": f"pallet_moved {pid}", "severity": "warning"}
                )

    return triggers


def build_replan_request(
    cuopt_input: dict[str, Any],
    critical_events: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = deepcopy(cuopt_input)
    constraints = updated.setdefault("constraints", {})
    blocked_nodes = set(constraints.get("blocked_nodes", []))

    for event in critical_events:
        event_type = str(event.get("event_type", ""))
        details = str(event.get("details", ""))
        if event_type in {"aisle_block", "human_crossing"}:
            # Parse last token as best-effort node id from event details.
            words = details.replace(".", "").split()
            if words:
                candidate = words[-1]
                if "_" in candidate:
                    blocked_nodes.add(candidate)

    constraints["blocked_nodes"] = sorted(blocked_nodes)
    return updated
