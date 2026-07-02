import omni.client
from isaacsim.storage.native import get_assets_root_path

root = get_assets_root_path()
print("ROOT", root)
for sub in [
    "/Isaac/Environments/Simple_Warehouse/Props/",
    "/Isaac/Props/Pallet/",
]:
    res, ents = omni.client.list(root + sub)
    print("==", sub, res)
    for e in ents:
        n = e.relative_path
        if "alet" in n.lower() or "allet" in n.lower():
            print("   ", n)
