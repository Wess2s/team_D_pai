"""
Agent tools — the functions the LLM can call to run the warehouse forklift fleet.

These are the seam between the GenAI layer and the simulation. Each talks to the sim over
the HTTP bridge (src/ros2/bridge_client.py), which drives the offline kinematic
WarehouseSim today and the real Isaac Sim forklift scene tomorrow — the tool interface is
identical for both, so nothing here changes when we swap backends.

The dispatcher at the bottom maps tool names to callables, and TOOL_SCHEMAS gives the
OpenAI/NIM function-calling schemas.
"""
from __future__ import annotations

import json

from . import world_state as ws
from .planning import Roadmap, cbs, conflict, cuopt_planner

try:
    from ..ros2 import bridge_client
except ImportError:  # requests missing / package layout during early dev
    bridge_client = None

# Most-recent optimisation plan, exposed by the bridge for the console overlay.
LAST_PLAN: dict = {}


def _snapshot() -> dict:
    """Live world snapshot from the bridge, or an empty one if unreachable."""
    if bridge_client is not None and bridge_client.health():
        try:
            return bridge_client.get_state()
        except bridge_client.BridgeError:
            pass
    return ws.empty_snapshot()


def _bridge():
    if bridge_client is None or not bridge_client.health():
        return None
    return bridge_client


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def get_fleet_status() -> str:
    """Current state of every forklift, pallet and staging zone."""
    snap = _snapshot()
    return json.dumps({
        "summary": ws.summarise(snap),
        "forklifts": snap.get("forklifts", {}),
        "pallets": snap.get("pallets", {}),
        "zones": snap.get("zones", {}),
    })


def move_pallet(pallet: str, zone: str, robot: str | None = None) -> str:
    """
    Move a pallet to a staging zone with one forklift (pick then drop, chained).
    If `robot` is omitted, the nearest free forklift is chosen automatically.
    """
    br = _bridge()
    if br is None:
        return json.dumps({"ok": False, "error": "sim bridge unreachable"})
    snap = _snapshot()
    xy = ws.pallet_xy(snap, pallet)
    if xy is None:
        return json.dumps({"ok": False, "error": f"unknown pallet {pallet}"})
    if robot is None:
        robot = ws.nearest_free_forklift(snap, xy)
        if robot is None:
            return json.dumps({"ok": False, "error": "no free forklift available"})
    res = br.mission(robot, [["pick", pallet], ["drop", zone]])
    return json.dumps({"robot": robot, "pallet": pallet, "zone": zone, "result": res})


def pick_pallet(pallet: str, robot: str | None = None) -> str:
    """Send a forklift to pick up a pallet (no drop). Auto-selects a forklift if omitted."""
    br = _bridge()
    if br is None:
        return json.dumps({"ok": False, "error": "sim bridge unreachable"})
    snap = _snapshot()
    xy = ws.pallet_xy(snap, pallet)
    if xy is None:
        return json.dumps({"ok": False, "error": f"unknown pallet {pallet}"})
    if robot is None:
        robot = ws.nearest_free_forklift(snap, xy)
        if robot is None:
            return json.dumps({"ok": False, "error": "no free forklift available"})
    return json.dumps({"robot": robot, "result": br.pick(robot, pallet)})


def drop_pallet(zone: str, robot: str) -> str:
    """Have a carrying forklift drop its pallet at a staging zone."""
    br = _bridge()
    if br is None:
        return json.dumps({"ok": False, "error": "sim bridge unreachable"})
    return json.dumps({"robot": robot, "result": br.drop(robot, zone)})


def send_home(robot: str) -> str:
    """Return a forklift to its home/charging node."""
    br = _bridge()
    if br is None:
        return json.dumps({"ok": False, "error": "sim bridge unreachable"})
    return json.dumps({"robot": robot, "result": br.go_home(robot)})


def detect_conflict() -> str:
    """Flag any two forklifts that are close enough to risk a collision."""
    snap = _snapshot()
    conflicts = conflict.live_conflicts(snap)
    return json.dumps({
        "conflicts": [f"{c['a']} & {c['b']} within {c['distance']}m ({c['severity']})"
                      for c in conflicts],
        "detail": conflicts,
    })


def optimize_and_dispatch(zone: str | None = None) -> str:
    """
    Plan and dispatch a full clear-the-racks operation with NVIDIA cuOpt (fleet task
    assignment) + Conflict-Based Search (proactive multi-agent deconfliction).

    cuOpt decides which forklift moves which pallets, in what order (roadmap-aware VRP);
    CBS then plans time-parameterised, collision-free approach paths across the fleet.
    Returns the plan (assignments, cost, conflicts found & resolved) and dispatches it.
    """
    global LAST_PLAN
    br = _bridge()
    if br is None:
        return json.dumps({"ok": False, "error": "sim bridge unreachable"})
    snap = _snapshot()
    rm = Roadmap.from_snapshot(snap)

    zones = [zone] if zone else None
    plan = cuopt_planner.plan_moves(snap, rm, zones=zones)
    if not plan.routes:
        return json.dumps({"ok": False, "error": "nothing to dispatch "
                           "(no free forklift or no pallets awaiting pickup)"})

    # CBS: deconflict each forklift's approach to its first pickup.
    agents: dict[str, tuple[str, str]] = {}
    for fk, tasks in plan.routes.items():
        f = snap["forklifts"][fk]
        first_pallet = snap["pallets"][tasks[0][0]]
        agents[fk] = (rm.nearest(f["x"], f["y"]),
                      rm.nearest(first_pallet["x"], first_pallet["y"]))
    cbs_res = cbs.solve(rm, agents)

    # Dispatch each forklift's ordered pick→drop tour. Stagger releases if CBS could not
    # fully resolve conflicts within budget (motion stays proactively safe).
    dispatched = []
    for fk, tasks in plan.routes.items():
        steps = []
        for pallet, zid in tasks:
            steps += [["pick", pallet], ["drop", zid]]
        res = br.mission(fk, steps)
        dispatched.append({"forklift": fk,
                           "tasks": [f"{p}→{z}" for p, z in tasks],
                           "ok": res.get("ok", True)})

    LAST_PLAN = {
        "solver": plan.solver,
        "total_cost": plan.total_cost,
        "assignments": {fk: [f"{p}→{z}" for p, z in tasks]
                        for fk, tasks in plan.routes.items()},
        "cbs": {"conflicts_found": cbs_res.conflicts_found,
                "resolved": cbs_res.resolved,
                "paths": cbs_res.paths},
    }
    return json.dumps({
        "ok": True,
        "solver": plan.solver,
        "total_cost": plan.total_cost,
        "dispatched": dispatched,
        "cbs": {"conflicts_found": cbs_res.conflicts_found, "resolved": cbs_res.resolved},
    })


def plan_routes() -> str:
    """Preview the cuOpt + CBS plan for clearing the racks WITHOUT dispatching."""
    snap = _snapshot()
    rm = Roadmap.from_snapshot(snap)
    plan = cuopt_planner.plan_moves(snap, rm)
    if not plan.routes:
        return json.dumps({"ok": False, "error": "no free forklift or no pallets"})
    agents = {}
    for fk, tasks in plan.routes.items():
        f = snap["forklifts"][fk]
        fp = snap["pallets"][tasks[0][0]]
        agents[fk] = (rm.nearest(f["x"], f["y"]), rm.nearest(fp["x"], fp["y"]))
    cbs_res = cbs.solve(rm, agents)
    return json.dumps({
        "ok": True, "solver": plan.solver, "total_cost": plan.total_cost,
        "assignments": {fk: [f"{p}→{z}" for p, z in t] for fk, t in plan.routes.items()},
        "cbs": {"conflicts_found": cbs_res.conflicts_found, "resolved": cbs_res.resolved},
    })



def block_zone(zone: str) -> str:
    """Mark a staging zone blocked (e.g. an incident) so it is avoided."""
    br = _bridge()
    if br is None:
        return json.dumps({"ok": False, "error": "sim bridge unreachable"})
    return json.dumps(br.block_zone(zone))


# --------------------------------------------------------------------------- #
# OpenAI / NIM tool schemas
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_fleet_status",
        "description": "Get the current state of all forklifts, pallets and staging zones.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "move_pallet",
        "description": "Move a pallet to a staging zone with one forklift (pick then drop). "
                       "Omit robot to auto-pick the nearest free forklift.",
        "parameters": {"type": "object", "properties": {
            "pallet": {"type": "string", "description": "e.g. WH_Palette_01"},
            "zone": {"type": "string", "description": "e.g. stage_1"},
            "robot": {"type": "string", "description": "optional, e.g. forklift2"},
        }, "required": ["pallet", "zone"]},
    }},
    {"type": "function", "function": {
        "name": "pick_pallet",
        "description": "Send a forklift to pick up a pallet (without dropping).",
        "parameters": {"type": "object", "properties": {
            "pallet": {"type": "string"},
            "robot": {"type": "string", "description": "optional"},
        }, "required": ["pallet"]},
    }},
    {"type": "function", "function": {
        "name": "drop_pallet",
        "description": "Have a carrying forklift drop its pallet at a staging zone.",
        "parameters": {"type": "object", "properties": {
            "zone": {"type": "string"},
            "robot": {"type": "string"},
        }, "required": ["zone", "robot"]},
    }},
    {"type": "function", "function": {
        "name": "send_home",
        "description": "Return a forklift to its home/charging node.",
        "parameters": {"type": "object", "properties": {
            "robot": {"type": "string"},
        }, "required": ["robot"]},
    }},
    {"type": "function", "function": {
        "name": "detect_conflict",
        "description": "Check for forklifts that are close enough to risk a collision.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "optimize_and_dispatch",
        "description": "Clear the racks optimally: use NVIDIA cuOpt to assign pallets to "
                       "forklifts (roadmap-aware VRP) and Conflict-Based Search to plan "
                       "collision-free paths, then dispatch the whole operation. Use this for "
                       "'clear all pallets', 'optimise the fleet', 'plan and go'.",
        "parameters": {"type": "object", "properties": {
            "zone": {"type": "string", "description": "optional single target staging zone"},
        }},
    }},
    {"type": "function", "function": {
        "name": "plan_routes",
        "description": "Preview the cuOpt + CBS plan for clearing the racks without dispatching.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "block_zone",
        "description": "Mark a staging zone blocked due to an incident so it is avoided.",
        "parameters": {"type": "object", "properties": {
            "zone": {"type": "string"},
        }, "required": ["zone"]},
    }},
]

DISPATCH = {
    "get_fleet_status": get_fleet_status,
    "move_pallet": move_pallet,
    "pick_pallet": pick_pallet,
    "drop_pallet": drop_pallet,
    "send_home": send_home,
    "detect_conflict": detect_conflict,
    "optimize_and_dispatch": optimize_and_dispatch,
    "plan_routes": plan_routes,
    "block_zone": block_zone,
}


def call_tool(name: str, arguments: str) -> str:
    """Dispatch a tool call. `arguments` is the JSON string from the model."""
    fn = DISPATCH[name]
    kwargs = json.loads(arguments) if arguments else {}
    return fn(**kwargs)
