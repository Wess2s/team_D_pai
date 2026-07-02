from __future__ import annotations

from logistics_models import CuOptOutput, MissionPlan, NavigationCommand, Node


def translate_solution_to_mission(
    mission_id: str,
    objective: str,
    nodes: list[Node],
    solution: CuOptOutput,
) -> MissionPlan:
    node_lookup = {node.id: node for node in nodes}
    vehicle_commands: dict[str, list[NavigationCommand]] = {}

    for vehicle_route in solution.vehicle_routes:
        commands: list[NavigationCommand] = []
        for stop in vehicle_route.stops:
            node = node_lookup.get(stop.node_id)
            if node is None:
                continue

            if stop.stop_type == "start":
                continue

            commands.append(
                NavigationCommand(
                    command_type="navigate",
                    vehicle_id=vehicle_route.vehicle_id,
                    node_id=stop.node_id,
                    x=node.x,
                    y=node.y,
                    job_id=stop.job_id,
                    pallet_id=stop.pallet_id,
                    metadata={
                        "eta_s": stop.eta_s,
                        "departure_s": stop.departure_s,
                        "wait_s": stop.wait_s,
                    },
                )
            )

            if stop.stop_type == "pickup":
                commands.append(
                    NavigationCommand(
                        command_type="pickup",
                        vehicle_id=vehicle_route.vehicle_id,
                        node_id=stop.node_id,
                        x=node.x,
                        y=node.y,
                        pallet_id=stop.pallet_id,
                        job_id=stop.job_id,
                        metadata={
                            "eta_s": stop.eta_s,
                            "departure_s": stop.departure_s,
                        },
                    )
                )
            elif stop.stop_type == "delivery":
                commands.append(
                    NavigationCommand(
                        command_type="dropoff",
                        vehicle_id=vehicle_route.vehicle_id,
                        node_id=stop.node_id,
                        x=node.x,
                        y=node.y,
                        pallet_id=stop.pallet_id,
                        job_id=stop.job_id,
                        metadata={
                            "eta_s": stop.eta_s,
                            "departure_s": stop.departure_s,
                        },
                    )
                )
            elif stop.stop_type == "charging":
                commands.append(
                    NavigationCommand(
                        command_type="charge",
                        vehicle_id=vehicle_route.vehicle_id,
                        node_id=stop.node_id,
                        x=node.x,
                        y=node.y,
                        metadata={
                            "eta_s": stop.eta_s,
                            "departure_s": stop.departure_s,
                        },
                    )
                )

        vehicle_commands[vehicle_route.vehicle_id] = commands

    return MissionPlan(
        mission_id=mission_id,
        objective=objective,  # type: ignore[arg-type]
        vehicle_commands=vehicle_commands,
        total_cost=solution.total_cost,
        status=solution.status,
        errors=solution.errors,
    )
