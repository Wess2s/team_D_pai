from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_replan_request(
    cuopt_input: dict[str, Any],
    critical_events: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = deepcopy(cuopt_input)
    constraints = updated.setdefault("constraints", {})
    blocked_nodes = set(constraints.get("blocked_nodes", []))

    for event in critical_events:
        event_type = str(event.get("event_type", ""))
        details = str(event.get("details", ""))
        if event_type in {"aisle_block", "human_crossing"}:
            # Parse last token as best-effort node id from event details.
            words = details.replace(".", "").split()
            if words:
                candidate = words[-1]
                if "_" in candidate:
                    blocked_nodes.add(candidate)

    constraints["blocked_nodes"] = sorted(blocked_nodes)
    return updated
