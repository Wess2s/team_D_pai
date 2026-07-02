"""Inspect the Isaac forklift USD: articulation, joints, rigid bodies, meshes."""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from isaacsim.core.utils.stage import add_reference_to_stage
try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:
    from isaacsim.core.utils.nucleus import get_assets_root_path

root = get_assets_root_path()
usd = root + "/Isaac/Props/Forklift/forklift.usd"
print("ASSET:", usd)
add_reference_to_stage(usd_path=usd, prim_path="/F")
stage = omni.usd.get_context().get_stage()

joints, artroots, rigids, drives = [], [], [], []
for p in stage.Traverse():
    t = p.GetTypeName()
    path = str(p.GetPath())
    if "Joint" in str(t):
        joints.append((path, str(t)))
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        artroots.append(path)
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        rigids.append(path)
    if p.HasAPI(UsdPhysics.DriveAPI) or any("drive" in str(s).lower() for s in p.GetAppliedSchemas()):
        drives.append(path)

print("\n=== ARTICULATION ROOTS ===")
for a in artroots: print(" ", a)
print("  (count:", len(artroots), ")")
print("\n=== JOINTS ===")
for j, t in joints: print(f"  {t:28s} {j}")
print("  (count:", len(joints), ")")
print("\n=== RIGID BODIES ===")
for r in rigids[:40]: print(" ", r)
print("  (count:", len(rigids), ")")
print("\n=== PRIMS WITH DRIVE ===")
for d in drives: print(" ", d)

print("\n=== TOP-LEVEL HIERARCHY (depth<=3) ===")
base = stage.GetPrimAtPath("/F")
for p in Usd.PrimRange(base):
    d = str(p.GetPath()).count("/")
    if d <= 4:
        print("  " * (d-1) + p.GetName() + f"  [{p.GetTypeName()}]")

app.close()
