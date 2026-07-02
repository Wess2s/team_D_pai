#!/usr/bin/env python3
"""
FleetMind — Warehouse scene builder (environment-as-code).

Builds a Simple Warehouse populated with:
  * 2 forklifts acting as AMRs  (/World/AMRs/AMR_1, /World/AMRs/AMR_2)
  * A grid of pallets           (/World/Pallets/Pallet_XX)
  * Pickable objects on pallets (/World/Payloads/...) as rigid bodies

Runs headless and serves a WebRTC livestream so the team can view it.
Edit the CONFIG block below to change the layout — this file is the shared,
version-controlled source of truth for the simulation environment.
"""

import argparse
import math

# ---------------------------------------------------------------------------
# CLI / livestream config (parse BEFORE creating SimulationApp)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--publicIp", default="127.0.0.1",
                    help="Public/Tailscale IP advertised to the WebRTC client")
parser.add_argument("--no-stream", action="store_true",
                    help="Build the scene without starting the livestream")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

# ---------------------------------------------------------------------------
# Enable WebRTC livestream (after SimulationApp exists)
# ---------------------------------------------------------------------------
import carb  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

if not args.no_stream:
    settings = carb.settings.get_settings()
    settings.set("/exts/omni.kit.livestream.app/primaryStream/publicIp", args.publicIp)
    settings.set("/exts/omni.kit.livestream.app/primaryStream/signalPort", 49100)
    settings.set("/exts/omni.kit.livestream.app/primaryStream/streamPort", 47998)
    enable_extension("omni.kit.livestream.webrtc")
    carb.log_warn(f"[FleetMind] WebRTC livestream on {args.publicIp} (signal 49100 / stream 47998)")

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:  # pragma: no cover - namespace fallback
    from isaacsim.core.utils.nucleus import get_assets_root_path

# ===========================================================================
# CONFIG  — edit this block to change the environment
# ===========================================================================
# Forklift AMRs: name -> (x, y, yaw_degrees)
FORKLIFTS = {
    "AMR_1": (-6.0, -3.0,   0.0),
    "AMR_2": ( 6.0,  3.0, 180.0),
}

# Pallet grid positions (x, y). Objects are placed on top of each pallet.
PALLETS = [
    (-3.0, -3.0), (-3.0, 0.0), (-3.0, 3.0),
    ( 3.0, -3.0), ( 3.0, 0.0), ( 3.0, 3.0),
]

PALLET_TOP_Z = 0.14   # approx height of the pallet deck (metres)
PAYLOAD_DROP = 0.06   # spawn objects slightly above the deck so physics settles
# ===========================================================================

assets_root = get_assets_root_path()
if assets_root is None:
    raise RuntimeError("Could not resolve Isaac assets root path")

WAREHOUSE_USD = assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
FORKLIFT_USD  = assets_root + "/Isaac/Props/Forklift/forklift.usd"
PALLET_USD    = assets_root + "/Isaac/Props/Pallet/pallet.usd"

# Verified pickable assets (rigid bodies with collision baked in)
PAYLOAD_USDS = [
    assets_root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    assets_root + "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
    assets_root + "/Isaac/Props/KLT_Bin/small_KLT.usd",
]


def set_pose(stage, prim_path, xyz, yaw_deg=0.0):
    """Set a world translate + Z rotation on a prim (API-stable via USD ops)."""
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xform.AddRotateZOp().Set(float(yaw_deg))


def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # --- Environment -------------------------------------------------------
    add_reference_to_stage(usd_path=WAREHOUSE_USD, prim_path="/World/Warehouse")

    # --- Forklift AMRs -----------------------------------------------------
    UsdGeom.Xform.Define(stage, "/World/AMRs")
    for name, (x, y, yaw) in FORKLIFTS.items():
        prim_path = f"/World/AMRs/{name}"
        add_reference_to_stage(usd_path=FORKLIFT_USD, prim_path=prim_path)
        set_pose(stage, prim_path, (x, y, 0.0), yaw)
        carb.log_warn(f"[FleetMind] Placed {name} at ({x}, {y}, yaw={yaw})")

    # --- Pallets + payloads ------------------------------------------------
    UsdGeom.Xform.Define(stage, "/World/Pallets")
    UsdGeom.Xform.Define(stage, "/World/Payloads")
    for i, (x, y) in enumerate(PALLETS):
        pallet_path = f"/World/Pallets/Pallet_{i:02d}"
        add_reference_to_stage(usd_path=PALLET_USD, prim_path=pallet_path)
        set_pose(stage, pallet_path, (x, y, 0.0), 0.0)

        # place one pickable object on top of this pallet
        payload_usd = PAYLOAD_USDS[i % len(PAYLOAD_USDS)]
        payload_path = f"/World/Payloads/Payload_{i:02d}"
        add_reference_to_stage(usd_path=payload_usd, prim_path=payload_path)
        set_pose(stage, payload_path, (x, y, PALLET_TOP_Z + PAYLOAD_DROP), 0.0)

    carb.log_warn(
        f"[FleetMind] Scene built: {len(FORKLIFTS)} AMRs, "
        f"{len(PALLETS)} pallets, {len(PALLETS)} payloads"
    )

    # --- Simulate ----------------------------------------------------------
    world.reset()
    carb.log_warn("[FleetMind] World ready — entering sim loop. Connect the WebRTC client.")
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
