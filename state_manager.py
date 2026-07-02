from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AgentState = Literal["idle", "moving", "picking", "dropping", "charging", "blocked", "error"]
JobState = Literal["pending", "assigned", "in_progress", "completed", "failed"]


@dataclass
class AgentRuntimeState:
    agent_id: str
    current_node: str
    state: AgentState = "idle"
    battery_pct: float = 100.0
    assigned_jobs: list[str] = field(default_factory=list)


@dataclass
class JobRuntimeState:
    job_id: str
    state: JobState = "pending"
    assigned_agent_id: str = ""


@dataclass
class WorldState:
    mission_id: str
    agents: dict[str, AgentRuntimeState] = field(default_factory=dict)
    jobs: dict[str, JobRuntimeState] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def mark_event(self, event_type: str, details: str, severity: str = "info") -> None:
        self.events.append({"type": event_type, "details": details, "severity": severity})


def build_initial_world_state(mission_id: str, vehicles: list[dict], jobs: list[dict]) -> WorldState:
    state = WorldState(mission_id=mission_id)
    for vehicle in vehicles:
        aid = str(vehicle.get("id", ""))
        if not aid:
            continue
        state.agents[aid] = AgentRuntimeState(
            agent_id=aid,
            current_node=str(vehicle.get("start_node_id", "")),
            battery_pct=float(vehicle.get("battery_level_pct", 100.0)),
        )

    for job in jobs:
        jid = str(job.get("id", ""))
        if not jid:
            continue
        state.jobs[jid] = JobRuntimeState(job_id=jid)

    return state
