"""
HTTP bridge server — the SIMULATION machine side (forklift warehouse).

Exposes a small REST API the agent (on the DGX) calls, and drives either:
  * an in-process **kinematic WarehouseSim** (default — runs offline, no Isaac needed), or
  * the real Isaac Sim forklift behaviour script via its `nav*` USD attributes
    (`SIM_BACKEND=isaac`, wired in src/sim/isaac_nav_bridge.py — used on the DGX).

The HTTP contract is identical for both backends, so the agent + UI never change when we
swap the mock for the real scene.

Run (offline mock — today):
    pip install fastapi uvicorn
    python -m uvicorn src.ros2.bridge_server:app --host 0.0.0.0 --port 8080

Run (on the DGX next to Isaac Sim — tomorrow):
    SIM_BACKEND=isaac python -m uvicorn src.ros2.bridge_server:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /health                 -> {ok, backend}
    GET  /state                  -> full world snapshot (forklifts/pallets/zones/graph)
    POST /goto   {robot, node}   -> drive a forklift to a waypoint/zone/pallet
    POST /pick   {robot, pallet} -> navigate to + pick a pallet
    POST /drop   {robot, zone}   -> carry to + drop at a staging zone
    POST /home   {robot}         -> return a forklift to its home node
    POST /block_zone {zone}      -> mark a staging zone blocked (incident)
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response as _Response

BACKEND = os.getenv("SIM_BACKEND", "mock").lower()   # mock | isaac

app = FastAPI(title="FleetMind Sim Bridge")

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui", "web"))


# --------------------------------------------------------------------------- #
# Backend selection — identical HTTP contract either way.
# --------------------------------------------------------------------------- #
if BACKEND == "isaac":
    # Real Isaac Sim forklift behaviour script, addressed over its nav* attributes.
    from ..sim.isaac_nav_bridge import IsaacNavBackend

    _sim = IsaacNavBackend()
else:
    from ..sim.warehouse_sim import WarehouseSim

    _sim = WarehouseSim()
    _sim.start()   # background stepping thread (30 Hz)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class GoTo(BaseModel):
    robot: str
    node: str


class Pick(BaseModel):
    robot: str
    pallet: str


class Drop(BaseModel):
    robot: str
    zone: str


class Home(BaseModel):
    robot: str


class Mission(BaseModel):
    robot: str
    steps: list[list[str]]   # [["pick","WH_Palette_01"], ["drop","stage_1"]]


class ZoneReq(BaseModel):
    zone: str
    kind: str = "spill"


class ChatReq(BaseModel):
    message: str


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"ok": True, "backend": BACKEND}


@app.get("/state")
def state() -> dict:
    snap = _sim.snapshot()
    try:  # live conflict search overlay (never break the state feed)
        from ..agent.planning import conflict
        snap["conflicts"] = conflict.live_conflicts(snap)
    except Exception:
        snap["conflicts"] = []
    return snap


@app.get("/plan")
def plan() -> dict:
    """Most-recent cuOpt + CBS optimisation plan (for the console overlay)."""
    from ..agent import tools
    return tools.LAST_PLAN or {}



@app.post("/goto")
def goto(req: GoTo) -> dict:
    return _sim.go_to(req.robot, req.node)


@app.post("/pick")
def pick(req: Pick) -> dict:
    return _sim.pick(req.robot, req.pallet)


@app.post("/drop")
def drop(req: Drop) -> dict:
    return _sim.drop(req.robot, req.zone)


@app.post("/home")
def home(req: Home) -> dict:
    return _sim.go_home(req.robot)


@app.post("/mission")
def mission(req: Mission) -> dict:
    return _sim.mission(req.robot, req.steps)


@app.post("/block_zone")
def block_zone(req: ZoneReq) -> dict:
    return _sim.block_zone(req.zone, req.kind)


@app.post("/reset")
def reset() -> dict:
    """Reset the scene to its start state for a fresh demo (pallets restocked, forklifts
    home, hazards cleared) without restarting the backend. Also clears the cached plan so
    the console overlay starts blank."""
    from ..agent import tools
    tools.LAST_PLAN = None
    return _sim.reset()


@app.post("/chat")
def chat(req: ChatReq) -> dict:
    """Operator natural-language command → GenAI agent → tool calls on this sim."""
    from ..agent.agent import run as agent_run
    try:
        reply = agent_run(req.message)
    except Exception as exc:  # keep the console alive if a backend hiccups
        reply = f"(agent error: {exc})"
    return {"reply": reply}


# --------------------------------------------------------------------------- #
# Static operator console (served at / — the mission-control UI).
# Mounted last so the API routes above always take precedence.
#
# No-cache: the console HTML/JS/CSS is redeployed between demos; without this the
# browser serves stale assets (missing new buttons, old stream code). Force the
# browser to revalidate every load so a plain refresh always picks up new UI.
# --------------------------------------------------------------------------- #
class _NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp: _Response = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


if os.path.isdir(WEB_DIR):
    app.mount("/", _NoCacheStatic(directory=WEB_DIR, html=True), name="console")

