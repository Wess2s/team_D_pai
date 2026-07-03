"""
Fleet bus — the in-process command/telemetry channel between the FastAPI bridge backend
(`IsaacNavBackend`) and the Isaac Sim scene controller (`scene_exec.py`).

Both live inside the same Isaac Kit process, so a thread-safe module-level singleton is all
we need. The bridge writes a *mission* (an ordered list of legs, each a waypoint route plus a
terminal action) per forklift; the physics-step controller consumes it, drives the real
articulation leg by leg, and writes back *telemetry* (pose, phase, carried pallet, lift
height, speed). The bridge's `snapshot()` reads that telemetry to build the exact same
`/state` payload shape the mock produces, so the agent + UI are byte-for-byte identical
across backends.

Pure standard library — imports fine off-DGX for linting/tests.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Leg:
    """One step of a forklift mission: drive a waypoint route, then act."""
    action: str = "goto"                            # goto | pick | drop | home
    target: str = ""                                # pallet id / zone id / node id
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    pallet_path: str = ""                            # USD path of the pallet to pick
    drop_xy: tuple[float, float] | None = None      # where a carried pallet is placed


@dataclass
class Command:
    """An ordered mission for one forklift (written by the bridge)."""
    seq: int = 0                                    # bumps on each new command
    legs: list[Leg] = field(default_factory=list)


@dataclass
class Telemetry:
    """Live per-forklift state (written by the scene controller)."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    phase: str = "idle"          # idle|navigating|picking|lifting|carrying|dropping|returning
    carrying: str | None = None  # pallet id on the forks
    lift_height: float = 0.0
    speed: float = 0.0
    route: list[str] = field(default_factory=list)
    target: str | None = None
    goal_kind: str | None = None
    object_detected: str = "None"
    object_distance: float = 0.0
    path_blocked: bool = False
    battery: float = 100.0       # state of charge %, drains with travel, trickles at home


class FleetBus:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cmd: dict[str, Command] = {}
        self._tel: dict[str, Telemetry] = {}
        self.pallets: dict[str, dict] = {}   # pallet id -> {x,y,carried_by,delivered}
        self.zones: dict[str, dict] = {}     # zone id   -> {x,y,blocked}
        self.graph: dict = {"nodes": {}, "edges": []}
        self.t0 = time.time()
        self.reset_epoch = 0                 # bumps when a between-demo reset is requested

    # ---- registration ---------------------------------------------------- #
    def register_forklift(self, name: str, x: float, y: float, yaw: float = 0.0) -> None:
        with self._lock:
            self._tel.setdefault(name, Telemetry(x=x, y=y, yaw=yaw))
            self._cmd.setdefault(name, Command())

    def names(self) -> list[str]:
        with self._lock:
            return list(self._tel)

    # ---- command side (bridge writes, controller reads) ------------------ #
    def send_mission(self, name: str, legs: list[Leg]) -> None:
        with self._lock:
            prev = self._cmd.get(name, Command())
            self._cmd[name] = Command(seq=prev.seq + 1, legs=list(legs))

    def get_command(self, name: str) -> Command:
        with self._lock:
            return self._cmd.get(name, Command())

    def clear_command(self, name: str) -> None:
        with self._lock:
            prev = self._cmd.get(name, Command())
            self._cmd[name] = Command(seq=prev.seq, legs=[])

    # ---- telemetry side (controller writes, bridge reads) ---------------- #
    def update_telemetry(self, name: str, **fields) -> None:
        with self._lock:
            tel = self._tel.setdefault(name, Telemetry())
            for k, v in fields.items():
                setattr(tel, k, v)

    def get_telemetry(self, name: str) -> Telemetry:
        with self._lock:
            return self._tel.get(name, Telemetry())

    def set_pallet(self, pid: str, **fields) -> None:
        with self._lock:
            self.pallets.setdefault(pid, {"x": 0.0, "y": 0.0,
                                          "carried_by": None, "delivered": False})
            self.pallets[pid].update(fields)

    def set_zone(self, zid: str, **fields) -> None:
        with self._lock:
            self.zones.setdefault(zid, {"x": 0.0, "y": 0.0, "blocked": False})
            self.zones[zid].update(fields)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def request_reset(self) -> None:
        """Signal a between-demo reset. The scene controller watches `reset_epoch` and,
        when it changes, teleports every pallet/forklift prim back to its spawn pose — so
        the scene resets in place without restarting Isaac (keeping the live stream up)."""
        with self._lock:
            self.reset_epoch += 1


# Module-level singleton shared across the Kit process.
_BUS: FleetBus | None = None


def bus() -> FleetBus:
    global _BUS
    if _BUS is None:
        _BUS = FleetBus()
    return _BUS
