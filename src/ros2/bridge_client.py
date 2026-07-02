"""
HTTP bridge client — the agent (DGX) side.

Lets the agent talk to the warehouse sim (or the real Isaac scene) running on a different
machine over HTTP, instead of needing cross-machine ROS 2 discovery. Your tools call this;
the sim machine runs `bridge_server.py`.

Contract (matches src/ros2/bridge_server.py):
    GET  /health                 -> {"ok": true, "backend": "mock"|"isaac"}
    GET  /state                  -> full world snapshot (forklifts/pallets/zones/graph)
    POST /goto   {robot, node}   -> drive a forklift to a waypoint/zone/pallet
    POST /pick   {robot, pallet} -> navigate to + pick a pallet
    POST /drop   {robot, zone}   -> carry to + drop at a staging zone
    POST /home   {robot}         -> return a forklift to its home node
    POST /block_zone {zone}      -> mark a staging zone blocked (incident)

If the bridge is unreachable, callers should fall back to local stub state so the agent
keeps working during development.
"""
from __future__ import annotations

import os

import requests

BRIDGE_URL = os.getenv("BRIDGE_URL", "http://localhost:8080")
TIMEOUT = float(os.getenv("BRIDGE_TIMEOUT", "5"))


class BridgeError(RuntimeError):
    """Raised when the sim bridge cannot be reached or returns an error."""


def health() -> bool:
    """Return True if the sim bridge is reachable."""
    try:
        r = requests.get(f"{BRIDGE_URL}/health", timeout=TIMEOUT)
        return r.ok and r.json().get("ok", False)
    except requests.RequestException:
        return False


def get_state() -> dict:
    """Fetch the current world snapshot from the sim machine."""
    try:
        r = requests.get(f"{BRIDGE_URL}/state", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise BridgeError(f"get_state failed: {exc}") from exc


def _post(path: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{BRIDGE_URL}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        raise BridgeError(f"{path} failed: {exc}") from exc


def goto(robot: str, node: str) -> dict:
    """Drive a forklift to a named waypoint/zone/pallet."""
    return _post("/goto", {"robot": robot, "node": node})


def pick(robot: str, pallet: str) -> dict:
    """Navigate a forklift to a pallet and pick it up."""
    return _post("/pick", {"robot": robot, "pallet": pallet})


def drop(robot: str, zone: str) -> dict:
    """Carry the held pallet to a staging zone and drop it."""
    return _post("/drop", {"robot": robot, "zone": zone})


def go_home(robot: str) -> dict:
    """Return a forklift to its home/charging node."""
    return _post("/home", {"robot": robot})


def mission(robot: str, steps: list) -> dict:
    """Queue a chained mission, e.g. [["pick","WH_Palette_01"],["drop","stage_1"]]."""
    return _post("/mission", {"robot": robot, "steps": steps})


def block_zone(zone: str) -> dict:
    """Tell the sim a zone is blocked (e.g. an incident)."""
    return _post("/block_zone", {"zone": zone})


if __name__ == "__main__":
    print("bridge reachable:", health())
    if health():
        snap = get_state()
        print("forklifts:", list(snap.get("forklifts", {})))
        print("pallets:", list(snap.get("pallets", {})))
