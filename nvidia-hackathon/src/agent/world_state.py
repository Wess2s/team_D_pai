"""
World-state helpers — reasoning utilities over a warehouse snapshot.

The single source of truth for live state is the sim bridge (`/state`), which returns a
snapshot of forklifts, pallets, zones and the waypoint graph. These helpers let the agent
tools reason over that snapshot (pick the nearest free forklift, summarise the fleet in
plain English) without duplicating world state. When the bridge is unreachable, an empty
snapshot is used so the agent can still respond gracefully.
"""
from __future__ import annotations

import math

# Phases in which a forklift is free to take a new mission.
FREE_PHASES = {"idle"}


def empty_snapshot() -> dict:
    return {"forklifts": {}, "pallets": {}, "zones": {}, "graph": {"nodes": {}, "edges": []}}


def nearest_free_forklift(snap: dict, target_xy: tuple[float, float]) -> str | None:
    """Name of the closest idle, empty forklift to a point; None if all busy."""
    free = [
        (name, fk) for name, fk in snap.get("forklifts", {}).items()
        if fk.get("phase") in FREE_PHASES and not fk.get("carrying")
    ]
    if not free:
        return None
    return min(
        free,
        key=lambda nf: math.hypot(nf[1]["x"] - target_xy[0], nf[1]["y"] - target_xy[1]),
    )[0]


def pallet_xy(snap: dict, pallet_id: str) -> tuple[float, float] | None:
    p = snap.get("pallets", {}).get(pallet_id)
    return (p["x"], p["y"]) if p else None


def available_pallets(snap: dict) -> list[str]:
    return [
        pid for pid, p in snap.get("pallets", {}).items()
        if not p.get("carried_by") and not p.get("delivered")
    ]


def summarise(snap: dict) -> str:
    """One-line-per-forklift plain-English fleet summary for the operator/LLM."""
    lines: list[str] = []
    for name, fk in snap.get("forklifts", {}).items():
        bits = [f"{name}: {fk.get('phase', '?')}"]
        if fk.get("carrying"):
            bits.append(f"carrying {fk['carrying']}")
        if fk.get("target"):
            bits.append(f"→ {fk['target']}")
        bits.append(f"@({fk.get('x', 0):.1f},{fk.get('y', 0):.1f})")
        lines.append(", ".join(bits))
    pend = available_pallets(snap)
    if pend:
        lines.append(f"pallets awaiting pickup: {', '.join(pend)}")
    delivered = [pid for pid, p in snap.get("pallets", {}).items() if p.get("delivered")]
    if delivered:
        lines.append(f"delivered: {', '.join(delivered)}")
    return "\n".join(lines) if lines else "(no fleet state available)"
