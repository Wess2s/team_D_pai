"""
live_state_adapter.py
---------------------
Bridge between the running FleetMind Isaac Sim (HTTP API on :8080) and the
cuOpt logistics solver in this repo.

It fetches the live world snapshot from ``GET /state`` and converts it into the
solver's typed contracts (``Node`` / ``Vehicle`` / ``Job``) using the *real*
scene entity ids so the resulting mission can be dispatched straight back to the
sim without any name translation:

    forklift AMR_1/AMR_2  -> Vehicle(id="AMR_1", start_node_id="AMR_1__start")
    pallet   WH_Palette_0X -> Node(id="WH_Palette_0X", node_type="storage")
    zone     stage_Y       -> Node(id="stage_Y",       node_type="dock")

Jobs (which pallet goes to which zone) come from a static demo spec by default
(all available pallets distributed round-robin across the staging zones), or
from an explicit ``[{"pallet": ..., "zone": ...}, ...]`` override.
"""
from __future__ import annotations

import json
from typing import Any
from urllib import request

from logistics_models import Job, Node, Vehicle

DEFAULT_BASE_URL = "http://localhost:8080"

# A single synthetic charging node keeps the solver's battery bookkeeping happy
# even though the demo warehouse has no physical charger.
CHARGING_NODE_ID = "charge_depot"


def fetch_state(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 5.0) -> dict[str, Any]:
    """Fetch the live world snapshot from the FleetMind bridge ``GET /state``."""
    url = base_url.rstrip("/") + "/state"
    with request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 (trusted local sim)
        return json.loads(resp.read().decode())


def _available_forklifts(state: dict[str, Any], include_busy: bool) -> dict[str, dict[str, Any]]:
    forklifts: dict[str, dict[str, Any]] = {}
    for fid, info in (state.get("forklifts") or {}).items():
        phase = str(info.get("phase", "idle"))
        carrying = info.get("carrying")
        if not include_busy and (phase != "idle" or carrying):
            continue
        forklifts[fid] = info
    return forklifts


def _available_pallets(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pallets: dict[str, dict[str, Any]] = {}
    for pid, info in (state.get("pallets") or {}).items():
        if info.get("delivered"):
            continue
        if info.get("carried_by"):
            continue
        pallets[pid] = info
    return pallets


def _open_zones(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        zid: info
        for zid, info in (state.get("zones") or {}).items()
        if not info.get("blocked")
    }


def _not_carried_pallets(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """All pallets not currently on a fork (delivered ones included)."""
    return {
        pid: info
        for pid, info in (state.get("pallets") or {}).items()
        if not info.get("carried_by")
    }


def _assign_distinct_zones(
    pallets: dict[str, dict[str, Any]],
    zones: dict[str, dict[str, Any]],
    *,
    avoid_current: bool = False,
) -> list[dict[str, str]]:
    """Bijective pallet->zone assignment minimising travel (greedy nearest).

    Each zone is used at most once (no stacking) and each pallet is paired with
    its closest still-free zone (no cross-room trips). When ``avoid_current`` is
    set, a pallet is not re-assigned to the zone it already sits on (used for the
    delivered-pallet recycle so the move is actually visible). If more pallets
    than zones exist, only ``len(zones)`` jobs are produced — the rest stay put
    rather than being stacked.
    """
    import math

    def dist(pid: str, zid: str) -> float:
        p, z = pallets[pid], zones[zid]
        return math.dist((float(p["x"]), float(p["y"])), (float(z["x"]), float(z["y"])))

    current_zone: dict[str, str] = {}
    if avoid_current and zones:
        for pid in pallets:
            current_zone[pid] = min(zones, key=lambda zid, _p=pid: dist(_p, zid))

    candidates = sorted(
        (dist(pid, zid), pid, zid) for pid in pallets for zid in zones
    )
    used_p: set[str] = set()
    used_z: set[str] = set()
    pairs: list[dict[str, str]] = []
    for _d, pid, zid in candidates:
        if pid in used_p or zid in used_z:
            continue
        if avoid_current and current_zone.get(pid) == zid:
            continue
        pairs.append({"pallet": pid, "zone": zid})
        used_p.add(pid)
        used_z.add(zid)
    return sorted(pairs, key=lambda j: j["pallet"])


def build_jobs_spec(
    state: dict[str, Any],
    job_spec: list[dict[str, str]] | None = None,
    *,
    allow_delivered_fallback: bool = True,
) -> list[dict[str, str]]:
    """Resolve the pickup/delivery job list.

    If ``job_spec`` is given it is used verbatim (each item ``{"pallet","zone"}``).
    Otherwise each available (not-yet-delivered) pallet is paired with its nearest
    *distinct* open staging zone — so no two pallets target the same zone (no
    stacking) and no pallet is dragged across the room. When everything is already
    delivered and ``allow_delivered_fallback`` is set, delivered pallets are
    re-cycled to their nearest *other* zone so the scene keeps working.
    """
    if job_spec:
        return job_spec

    zones = _open_zones(state)
    if not zones:
        return []

    available = _available_pallets(state)
    if available:
        return _assign_distinct_zones(available, zones)

    if not allow_delivered_fallback:
        return []

    return _assign_distinct_zones(_not_carried_pallets(state), zones, avoid_current=True)


def build_nodes_vehicles_jobs(
    state: dict[str, Any],
    job_spec: list[dict[str, str]] | None = None,
    *,
    include_busy: bool = False,
    vehicle_speed_mps: float = 1.2,
    vehicle_capacity: float = 1.0,
    allow_delivered_fallback: bool = True,
) -> tuple[list[Node], list[Vehicle], list[Job]]:
    """Convert a live ``/state`` snapshot into cuOpt solver inputs.

    Returns ``(nodes, vehicles, jobs)`` ready for ``build_warehouse_graph`` /
    ``build_cuopt_input``. Raises ``ValueError`` when the scene has no usable
    forklift, pallet or zone.
    """
    forklifts = _available_forklifts(state, include_busy=include_busy)
    if not forklifts:
        raise ValueError("No available (idle) forklifts found in /state")

    resolved_jobs = build_jobs_spec(state, job_spec, allow_delivered_fallback=allow_delivered_fallback)
    if not resolved_jobs:
        raise ValueError("No jobs could be built (no available pallets or open zones)")

    pallets = state.get("pallets") or {}
    zones = state.get("zones") or {}

    nodes: list[Node] = []
    seen_node_ids: set[str] = set()

    def add_node(node: Node) -> None:
        if node.id in seen_node_ids:
            return
        seen_node_ids.add(node.id)
        nodes.append(node)

    # Charging node (synthetic, unused by the demo but required by the contract).
    add_node(Node(id=CHARGING_NODE_ID, node_type="charging", x=0.0, y=0.0))

    # Pallet pickup nodes + zone delivery nodes, only for entities referenced by jobs.
    jobs: list[Job] = []
    for idx, spec in enumerate(resolved_jobs):
        pallet_id = spec["pallet"]
        zone_id = spec["zone"]

        pallet = pallets.get(pallet_id)
        zone = zones.get(zone_id)
        if pallet is None or zone is None:
            raise ValueError(f"Job references unknown pallet/zone: {spec}")

        add_node(Node(id=pallet_id, node_type="storage", x=float(pallet["x"]), y=float(pallet["y"])))
        add_node(Node(id=zone_id, node_type="dock", x=float(zone["x"]), y=float(zone["y"])))

        jobs.append(
            Job(
                id=f"job_{idx + 1:03d}",
                pallet_id=pallet_id,
                pickup_node_id=pallet_id,
                delivery_node_id=zone_id,
            )
        )

    # Per-forklift start (depot) nodes + vehicles.
    vehicles: list[Vehicle] = []
    for fid, info in forklifts.items():
        start_node_id = f"{fid}__start"
        add_node(
            Node(
                id=start_node_id,
                node_type="depot",
                x=float(info.get("x", 0.0)),
                y=float(info.get("y", 0.0)),
            )
        )
        vehicles.append(
            Vehicle(
                id=fid,
                start_node_id=start_node_id,
                capacity=vehicle_capacity,
                charging_node_id=CHARGING_NODE_ID,
                speed_mps=vehicle_speed_mps,
            )
        )

    return nodes, vehicles, jobs
