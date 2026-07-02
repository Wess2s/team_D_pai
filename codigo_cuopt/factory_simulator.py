from __future__ import annotations

from logistics_models import ExecutionEvent, ExecutionReport, MissionPlan


def simulate_factory_execution(mission: MissionPlan) -> ExecutionReport:
    events: list[ExecutionEvent] = []
    delayed_commands = 0
    total_commands = 0

    for vehicle_id, commands in mission.vehicle_commands.items():
        for idx, command in enumerate(commands):
            total_commands += 1
            eta_s = float(command.metadata.get("eta_s", 0.0))

            # Simulate human crossing around busy docks.
            if command.node_id.startswith("Dock") and idx % 3 == 0:
                events.append(
                    ExecutionEvent(
                        timestamp_s=eta_s,
                        event_type="human_crossing",
                        vehicle_id=vehicle_id,
                        details=f"Pedestrian crossing detected near {command.node_id}",
                        severity="warning",
                    )
                )
                delayed_commands += 1

            # Simulate occasional aisle block on storage lanes.
            if command.node_id.startswith("Storage") and idx % 4 == 1:
                events.append(
                    ExecutionEvent(
                        timestamp_s=eta_s,
                        event_type="aisle_block",
                        vehicle_id=vehicle_id,
                        details=f"Temporary aisle obstruction at {command.node_id}",
                        severity="warning",
                    )
                )
                delayed_commands += 1

            # Simulate low battery warning on charge operations arriving late.
            if command.command_type == "charge" and eta_s > 120:
                events.append(
                    ExecutionEvent(
                        timestamp_s=eta_s,
                        event_type="low_battery",
                        vehicle_id=vehicle_id,
                        details="Vehicle reached charging later than expected",
                        severity="critical",
                    )
                )

    on_time_ratio = 1.0
    if total_commands:
        on_time_ratio = max(0.0, 1.0 - (delayed_commands / total_commands))

    status = "ok"
    if any(event.severity == "critical" for event in events):
        status = "degraded"
    if on_time_ratio < 0.75:
        status = "failed"

    actions = [
        "Enable dynamic re-planning every 30 seconds when aisle_block is detected.",
        "Add geofenced slow zones around dock intersections for human safety.",
        "Trigger pre-charge policy when projected battery at mission end < 20%.",
    ]

    if status == "failed":
        actions.append("Escalate to supervisor and freeze non-critical tasks until congestion clears.")

    return ExecutionReport(
        status=status,
        mission_id=mission.mission_id,
        on_time_ratio=round(on_time_ratio, 3),
        events=events,
        recommended_actions=actions,
    )
