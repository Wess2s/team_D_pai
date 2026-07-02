from __future__ import annotations

from collections import defaultdict


def render_route_summary(result: dict) -> str:
    cuopt_output = result.get("cuopt_output", {})
    execution_report = result.get("execution_report", {})
    cbs_output = result.get("cbs_output", {})
    lines = [
        f"Scenario: {result.get('scenario', 'n/a')}",
        f"Solver: {cuopt_output.get('solver_name', 'n/a')}",
        f"Status: {cuopt_output.get('status', 'n/a')}",
        f"Total cost: {cuopt_output.get('total_cost', 0.0)}",
        "",
        "Vehicle routes:",
    ]

    for route in cuopt_output.get("vehicle_routes", []):
        vehicle_id = route.get("vehicle_id", "unknown")
        stops = route.get("stops", [])
        path = " -> ".join(stop.get("node_id", "?") for stop in stops)
        lines.append(f"- {vehicle_id}: {path}")
        lines.append(
            "  "
            f"jobs={route.get('assigned_job_ids', [])} "
            f"cost={round(route.get('route_cost', 0.0), 3)} "
            f"time_s={round(route.get('total_time_s', 0.0), 2)} "
            f"wait_s={round(route.get('total_wait_s', 0.0), 2)} "
            f"battery_pct={round(route.get('consumed_battery_pct', 0.0), 2)}"
        )

    if execution_report:
        lines.append("")
        lines.append(
            "Execution report: "
            f"status={execution_report.get('status', 'n/a')} "
            f"on_time_ratio={execution_report.get('on_time_ratio', 'n/a')}"
        )
        events = execution_report.get("events", [])[:3]
        if events:
            lines.append("Top incidents:")
            for event in events:
                lines.append(
                    f"- t={event.get('timestamp_s', 0)} "
                    f"{event.get('vehicle_id', 'n/a')} "
                    f"{event.get('event_type', 'event')} "
                    f"[{event.get('severity', 'info')}]"
                )

    if cbs_output:
        lines.append("")
        lines.append(
            "CBS: "
            f"status={cbs_output.get('status', 'n/a')} "
            f"conflicts_resolved={cbs_output.get('conflicts_resolved', 0)}"
        )
        if cbs_output.get("unresolved_conflict"):
            lines.append(f"- unresolved_conflict={cbs_output.get('unresolved_conflict')}")

    return "\n".join(lines)


def render_ascii_map(result: dict, width: int = 60, height: int = 18) -> str:
    nodes = result.get("warehouse_graph", {}).get("nodes", [])
    routes = result.get("cuopt_output", {}).get("vehicle_routes", [])

    if not nodes:
        return "[ascii-map] No nodes available"

    xs = [float(n.get("x", 0.0)) for n in nodes]
    ys = [float(n.get("y", 0.0)) for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    node_pos: dict[str, tuple[int, int]] = {}
    for node in nodes:
        x = float(node.get("x", 0.0))
        y = float(node.get("y", 0.0))
        cx = int((x - min_x) / x_span * (width - 1))
        cy = int((y - min_y) / y_span * (height - 1))
        cy = (height - 1) - cy

        node_type = node.get("node_type", "waypoint")
        symbol = {
            "depot": "D",
            "storage": "S",
            "dock": "K",
            "charging": "C",
            "waypoint": "W",
        }.get(node_type, "N")
        grid[cy][cx] = symbol
        node_pos[node.get("id", "")] = (cx, cy)

    # Overlay route traces with lightweight vehicle-specific markers.
    marker_cycle = ["1", "2", "3", "4", "5"]
    for idx, route in enumerate(routes):
        marker = marker_cycle[idx % len(marker_cycle)]
        for stop in route.get("stops", []):
            node_id = stop.get("node_id", "")
            if node_id in node_pos:
                x, y = node_pos[node_id]
                if grid[y][x] == " ":
                    grid[y][x] = marker

    lines = ["+" + "-" * width + "+"]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")

    labels = defaultdict(list)
    for node in nodes:
        labels[node.get("node_type", "other")].append(node.get("id", ""))

    lines.append("Legend: D=Depot S=Storage K=Dock C=Charging W=Waypoint")
    lines.append("Nodes:")
    for node_type, ids in labels.items():
        lines.append(f"- {node_type}: {', '.join(ids)}")

    return "\n".join(lines)
