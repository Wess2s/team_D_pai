"""
cbs_executor.py
---------------
Optional faithful executor for CBS-deconflicted missions.

The default orchestrator path dispatches semantic ``/mission`` steps and lets the
sim route internally (fast, but the sim re-routes freely so CBS waits are only
advisory). This module instead *steps* each forklift through the exact
CBS-planned grid-node sequence with ``POST /goto``, and issues ``/pick`` /
``/drop`` at the action checkpoints — so the deconflicted timing is actually
honoured on the floor.

``plan_execution_events`` is pure and unit-tested; ``execute_cbs_paths`` performs
the bounded HTTP loop with arrival polling.
"""
from __future__ import annotations

import time
from typing import Any
from urllib import request

DEFAULT_BASE_URL = "http://localhost:8080"


def _compress_path(path: list[list[Any]]) -> list[tuple[int, str]]:
    """Collapse a CBS ``[(node, t), ...]`` path to timesteps where node changes."""
    out: list[tuple[int, str]] = []
    last: str | None = None
    for node, t in path:
        if node != last:
            out.append((int(t), str(node)))
            last = str(node)
    return out


def plan_execution_events(cbs_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a CBS result into a globally ordered list of execution events.

    Each event: ``{"t": int, "vehicle": str, "command": "goto"|"pick"|"drop",
    "target": str}``. Action checkpoints (pick/drop) become ``/pick`` / ``/drop``
    commands; every other grid transition becomes a ``/goto``. Events are sorted
    by ``(t, vehicle)`` to give a deterministic, wait-honouring schedule.
    """
    agent_paths = cbs_result.get("agent_paths") or {}
    actions = cbs_result.get("actions") or {}
    events: list[dict[str, Any]] = []

    for vid, path in agent_paths.items():
        moves = _compress_path(path)
        # Pending pick/drop checkpoints for this vehicle, in order.
        pending = [a for a in actions.get(vid, []) if a["kind"] in {"pick", "drop"}]
        ptr = 0
        for t, node in moves:
            if ptr < len(pending) and node == pending[ptr]["node"]:
                events.append({"t": t, "vehicle": vid, "command": pending[ptr]["kind"], "target": pending[ptr]["target"]})
                ptr += 1
            else:
                events.append({"t": t, "vehicle": vid, "command": "goto", "target": node})
        # Any unmatched trailing actions (node collapsed away) still get issued.
        while ptr < len(pending):
            last_t = moves[-1][0] if moves else 0
            events.append({"t": last_t + 1, "vehicle": vid, "command": pending[ptr]["kind"], "target": pending[ptr]["target"]})
            ptr += 1

    events.sort(key=lambda e: (e["t"], e["vehicle"]))
    return events


def _post(base_url: str, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    import json

    req = request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (trusted local sim)
        return json.loads(resp.read().decode())


def _get_state(base_url: str, timeout_s: float) -> dict[str, Any]:
    import json

    with request.urlopen(base_url.rstrip("/") + "/state", timeout=timeout_s) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def execute_cbs_paths(
    cbs_result: dict[str, Any],
    state: dict[str, Any],
    base_url: str = DEFAULT_BASE_URL,
    *,
    arrival_tol_m: float = 0.8,
    step_timeout_s: float = 30.0,
    poll_interval_s: float = 0.5,
    max_wall_s: float = 600.0,
) -> dict[str, Any]:
    """Step forklifts through the CBS grid paths, honouring per-timestep ordering.

    Best-effort: the sim performs continuous motion between grid nodes, so CBS
    timesteps act as an ordering/synchronisation barrier rather than exact
    seconds. Bounded by ``max_wall_s``. Returns an execution log.

    NOTE: this issues real motion commands to the live sim — run it as the sole
    connected controller.
    """
    grid_nodes = {
        str(nid): (float(xy[0]), float(xy[1]))
        for nid, xy in ((state.get("graph") or {}).get("nodes") or {}).items()
    }
    events = plan_execution_events(cbs_result)
    log: list[dict[str, Any]] = []
    started = time.monotonic()

    # Group events by timestep so a whole barrier is issued, then awaited.
    from itertools import groupby

    for t, group in groupby(events, key=lambda e: e["t"]):
        if time.monotonic() - started > max_wall_s:
            log.append({"t": t, "status": "aborted_max_wall"})
            break

        awaiting: list[tuple[str, str]] = []  # (vehicle, target_node)
        for ev in group:
            vid, cmd, target = ev["vehicle"], ev["command"], ev["target"]
            if cmd == "goto":
                resp = _post(base_url, "/goto", {"robot": vid, "node": target}, step_timeout_s)
                awaiting.append((vid, target))
            elif cmd == "pick":
                resp = _post(base_url, "/pick", {"robot": vid, "pallet": target}, step_timeout_s)
            else:  # drop
                resp = _post(base_url, "/drop", {"robot": vid, "zone": target}, step_timeout_s)
            log.append({"t": t, "vehicle": vid, "command": cmd, "target": target, "ok": resp.get("ok", True)})

        # Await arrival of the goto commands issued at this barrier.
        deadline = time.monotonic() + step_timeout_s
        while awaiting and time.monotonic() < deadline:
            snap = _get_state(base_url, step_timeout_s)
            still: list[tuple[str, str]] = []
            for vid, node in awaiting:
                fk = (snap.get("forklifts") or {}).get(vid, {})
                nx, ny = grid_nodes.get(node, (None, None))
                if nx is None:
                    continue
                dist = ((float(fk.get("x", 1e9)) - nx) ** 2 + (float(fk.get("y", 1e9)) - ny) ** 2) ** 0.5
                if dist > arrival_tol_m:
                    still.append((vid, node))
            awaiting = still
            if awaiting:
                time.sleep(poll_interval_s)

    return {"status": "ok", "events": len(events), "log": log}
