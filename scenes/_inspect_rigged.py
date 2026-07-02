"""Inspect the rigged forklift_b: articulation root, joints (+types/axes/drives)."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage
try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:
    from isaacsim.core.utils.nucleus import get_assets_root_path

root = get_assets_root_path()
usd = root + "/Isaac/Samples/Rigging/Forklift/forklift_b_rigged_cm.usd"
print("MARK_ASSET:", usd)
add_reference_to_stage(usd_path=usd, prim_path="/F")
stage = omni.usd.get_context().get_stage()

artroots, joints, rigids = [], [], []
for p in stage.Traverse():
    t = str(p.GetTypeName())
    path = str(p.GetPath())
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        artroots.append(path)
    if "Joint" in t and t != "PhysicsJoint":
        drive_axes = []
        for api in p.GetAppliedSchemas():
            if "DriveAPI" in api:
                drive_axes.append(api.split(":")[-1] if ":" in api else api)
        axis = ""
        a = p.GetAttribute("physics:axis")
        if a and a.HasAuthoredValue():
            axis = a.Get()
        joints.append((path, t, axis, ",".join(drive_axes)))
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        rigids.append(path)

print("MARK_ARTROOTS:", len(artroots))
for a in artroots: print("  ART:", a)
print("MARK_JOINTS:", len(joints))
for path, t, axis, drives in joints:
    print(f"  JOINT type={t:22s} axis={str(axis):4s} drives=[{drives}]  {path}")
print("MARK_RIGIDS:", len(rigids))
for r in rigids: print("  RB:", r)

print("MARK_HIER")
base = stage.GetPrimAtPath("/F")
for p in Usd.PrimRange(base):
    d = str(p.GetPath()).count("/")
    if d <= 4:
        print("  " * (d-1) + p.GetName() + f" [{p.GetTypeName()}]")
print("MARK_DONE")
app.close()
