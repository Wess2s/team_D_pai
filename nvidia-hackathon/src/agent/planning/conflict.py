"""
Conflict detection — live spatial risk + planned-route deconfliction checks.

`live_conflicts` scans the current snapshot for forklifts that are close and moving toward
each other (imminent collision risk) — used for the operator "detect conflicts" answer and
the console's live conflict indicator.

`route_conflicts` compares the CBS-planned timed paths for overlaps (a sanity check that
the plan is actually conflict-free before dispatch).
"""
from __future__ import annotations

import math

WARN_DIST = 2.5      # m — flag forklifts closer than this…
CLOSING_DOT = -0.1   # …and closing (velocity dot product negative)


def live_conflicts(snap: dict, warn_dist: float = WARN_DIST) -> list[dict]:
    """Pairs of forklifts at collision risk right now."""
    fks = list(snap.get("forklifts", {}).items())
    out: list[dict] = []
    for i in range(len(fks)):
        for j in range(i + 1, len(fks)):
            na, a = fks[i]
            nb, b = fks[j]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            d = math.hypot(dx, dy)
            if d >= warn_dist:
                continue
            sa, sb = a.get("speed", 0.0), b.get("speed", 0.0)
            if sa < 0.01 and sb < 0.01:
                continue  # both parked — no risk
            # Closing test: are their heading velocities reducing the gap?
            va = (math.cos(a.get("yaw", 0.0)) * sa, math.sin(a.get("yaw", 0.0)) * sa)
            vb = (math.cos(b.get("yaw", 0.0)) * sb, math.sin(b.get("yaw", 0.0)) * sb)
            rel = (va[0] - vb[0], va[1] - vb[1])
            closing = (rel[0] * dx + rel[1] * dy) / (d or 1.0)
            if closing > CLOSING_DOT:  # moving apart or parallel
                continue
            out.append({
                "a": na, "b": nb, "distance": round(d, 2),
                "severity": "high" if d < warn_dist * 0.6 else "medium",
            })
    return out


def route_conflicts(paths: dict[str, list[str]]) -> list[dict]:
    """Vertex/edge overlaps between timed paths (should be empty after CBS)."""
    agents = list(paths)
    horizon = max((len(p) for p in paths.values()), default=0)

    def at(p, t):
        return p[t] if t < len(p) else (p[-1] if p else "")

    out = []
    for t in range(horizon):
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai, aj = agents[i], agents[j]
                if at(paths[ai], t) and at(paths[ai], t) == at(paths[aj], t):
                    out.append({"type": "vertex", "a": ai, "b": aj,
                                "node": at(paths[ai], t), "t": t})
    return out
