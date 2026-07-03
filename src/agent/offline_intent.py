"""
Offline intent parser — a rule-based fallback for when the NIM/LLM is unreachable.

The DGX (and its NIM) is offline outside the event, so the live-LLM path can't run. This
module maps natural-language operator commands straight to tool calls with regexes, so the
whole agent → tools → sim → telemetry loop is fully demoable offline. When NIM is back, the
agent uses it and this stays as a graceful fallback.

Returns a list of (tool_name, kwargs) steps to execute, plus a short narration template.
"""
from __future__ import annotations

import re

from . import tools

PALLET_RE = re.compile(r"(?:wh[_ ]?)?pal[a-z]*[ _]?(\d+)", re.I)
ZONE_RE = re.compile(r"(?:stag(?:e|ing)|zone|bay|area)[_ ]?(\d+)", re.I)
FORKLIFT_RE = re.compile(r"(?:forklift|truck|amr|fork)[_ ]?(\d+)", re.I)


def _hazard_kind(message: str) -> str:
    """Classify the incident named in an operator message (default spill)."""
    m = message.lower()
    if "fire" in m or "smoke" in m or "flame" in m:
        return "fire"
    if "spill" in m or "leak" in m or "wet" in m:
        return "spill"
    if "block" in m or "close" in m:
        return "blocked"
    return "hazard"


def _pallet(token_num: str) -> str:
    return f"WH_Palette_{int(token_num):02d}"


def _zone(token_num: str) -> str:
    return f"stage_{int(token_num)}"


def _forklift(token_num: str) -> str:
    return f"forklift{int(token_num)}"


def _all_forklifts() -> list[str]:
    """Live forklift names from the bridge, or the default fleet if unreachable."""
    snap = tools._snapshot()
    return list(snap.get("forklifts", {})) or ["forklift1", "forklift2", "forklift3"]


def parse(message: str) -> list[tuple[str, dict]]:
    """Turn an operator message into an ordered list of (tool, kwargs) steps."""
    m = message.lower().strip()
    pallets = [_pallet(n) for n in PALLET_RE.findall(m)]
    zones = [_zone(n) for n in ZONE_RE.findall(m)]
    forklifts = [_forklift(n) for n in FORKLIFT_RE.findall(m)]

    # --- status / where ------------------------------------------------- #
    if any(w in m for w in ("status", "where", "what's happening", "whats happening",
                            "report", "situation", "how are")):
        return [("get_fleet_status", {})]

    # --- conflict check ------------------------------------------------- #
    if "conflict" in m or "collision" in m or "too close" in m:
        return [("detect_conflict", {})]

    # --- plan preview (no dispatch) ------------------------------------- #
    if ("plan" in m or "preview" in m or "what would" in m) and not pallets \
            and any(w in m for w in ("clear", "rack", "all", "optimi", "route", "fleet")):
        return [("plan_routes", {})]

    # --- incident / block ----------------------------------------------- #
    if any(w in m for w in ("block", "incident", "spill", "hazard", "fire", "leak",
                            "close ", "emergency", "evacuate")) and zones:
        kind = _hazard_kind(m)
        return [("block_zone", {"zone": z, "kind": kind}) for z in zones]

    # --- send home / return --------------------------------------------- #
    if any(w in m for w in ("home", "charge", "charging", "park", "return", "dock")):
        targets = forklifts or _all_forklifts()
        return [("send_home", {"robot": r}) for r in targets]

    # --- clear the racks / optimise (cuOpt task assignment + CBS deconfliction) - #
    if any(w in m for w in ("clear", "all pallet", "everything", "empty the rack",
                            "restock", "unload", "optimi", "dispatch")) and not pallets:
        kwargs = {"zone": zones[0]} if zones else {}
        return [("optimize_and_dispatch", kwargs)]

    # --- explicit move pallet -> zone ----------------------------------- #
    if pallets and zones:
        steps = []
        for i, pid in enumerate(pallets):
            z = zones[i] if i < len(zones) else zones[-1]
            kwargs = {"pallet": pid, "zone": z}
            if i < len(forklifts):
                kwargs["robot"] = forklifts[i]
            steps.append(("move_pallet", kwargs))
        return steps

    # --- pick only ------------------------------------------------------ #
    if pallets and ("pick" in m or "grab" in m or "collect" in m or "fetch" in m):
        steps = []
        for i, pid in enumerate(pallets):
            kwargs = {"pallet": pid}
            if i < len(forklifts):
                kwargs["robot"] = forklifts[i]
            steps.append(("pick_pallet", kwargs))
        return steps

    # --- drop only ------------------------------------------------------ #
    if zones and forklifts and ("drop" in m or "release" in m or "put down" in m):
        return [("drop_pallet", {"zone": zones[0], "robot": forklifts[0]})]

    # --- fallback: just report state ------------------------------------ #
    return [("get_fleet_status", {})]
