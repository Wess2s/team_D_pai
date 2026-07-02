cat > sim_bridge.py << 'EOF'
"""
sim_bridge.py
-------------
Step 2 of the Geo-CBS Fleet Orchestrator UI.

Sends a RobotTask (from llm_task_parser.py) into a running Isaac Sim instance
by sending Python code over a TCP socket to a custom receiver running inside the sim.

How it works:
  1. You paste a one-time receiver snippet into Isaac Sim's Script Editor and run it.
     (Window -> Script Editor, then paste + click Run)
  2. That snippet starts a TCP server on port 8765 INSIDE Isaac Sim's Python process.
  3. sim_bridge.py connects to port 8765 and sends a JSON payload {"script": "..."}.
  4. The receiver exec()s the script, which sets USD attributes on the forklift prim.
  5. backup.py reads those attributes every frame and drives the robot.

Prerequisites (do once per Isaac Sim session):
  Open Isaac Sim -> Window -> Script Editor
  Paste the receiver snippet from _run_script()'s docstring and click Run.

Usage:
  from llm_task_parser import parse_command
  from sim_bridge import dispatch_task
  tasks = parse_command("move pallet 2 one metre to the right")
  for t in tasks:
      dispatch_task(t)
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict

import socket
import json

# ── Config ─────────────────────────────────────────────────────────────────

ISAAC_SIM_HOST  = "localhost"   # Isaac Sim machine (change to remote IP if needed)
ISAAC_SIM_PORT  = 8765          # Custom TCP receiver started via Isaac Sim Script Editor
FORKLIFT_PATH   = "/World/forklift"   # USD path of the forklift prim
TIMEOUT_S       = 10


# ── Dispatch ────────────────────────────────────────────────────────────────

def dispatch_task(task) -> dict:
    """Translate one RobotTask into Isaac Sim USD attribute writes.

    Args:
        task: RobotTask dataclass instance from llm_task_parser.py

    Returns:
        dict with keys "ok" (bool) and "message" (str).
    """
    action = task.action

    if action == "pick":
        script = _script_pick(task.pallet_id)

    elif action == "move":
        if task.drop_area_id:
            # Named destination: pick + set drop area
            script = _script_move_to_area(task.pallet_id, task.drop_area_id)
        else:
            # Offset destination: pick + compute world target from current pallet pos + offset
            script = _script_move_by_offset(task.pallet_id, task.offset_x, task.offset_y)

    elif action == "drop":
        script = _script_drop(task.drop_area_id)

    elif action == "go_home":
        script = _script_go_home()

    elif action == "stop":
        script = _script_stop()

    else:
        return {"ok": False, "message": f"Unknown action: {action}"}

    return _run_script(script)


# ── Script builders ─────────────────────────────────────────────────────────

def _forklift_prim_setup() -> str:
    """Common preamble: get stage and forklift prim."""
    return textwrap.dedent(f"""
        import omni.usd
        from pxr import Sdf
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath("{FORKLIFT_PATH}")
        if not prim or not prim.IsValid():
            raise RuntimeError("Forklift prim not found at {FORKLIFT_PATH}")
        def set_bool(name, val):
            a = prim.GetAttribute(name)
            if a and a.IsValid(): a.Set(val)
        def set_str(name, val):
            a = prim.GetAttribute(name)
            if a and a.IsValid(): a.Set(val)
        def clear_rel(name):
            r = prim.GetRelationship(name)
            if r: r.SetTargets([])
    """).strip()


def _script_pick(pallet_id: str) -> str:
    return _forklift_prim_setup() + textwrap.dedent(f"""

        # Stop any current activity first
        set_bool("navGoTo",     False)
        set_bool("navGoToPick", False)
        set_bool("navGoToDrop", False)
        set_bool("navGoHome",   False)
        clear_rel("navPalletToPick")

        # Set new pick target
        set_str("navPalletToPickId", "{pallet_id}")
        set_bool("navGoToPick", True)
        print("[bridge] pick {pallet_id} -> navGoToPick=True")
    """)


def _script_move_to_area(pallet_id: str, drop_area_id: str) -> str:
    return _forklift_prim_setup() + textwrap.dedent(f"""

        set_bool("navGoTo",     False)
        set_bool("navGoToPick", False)
        set_bool("navGoToDrop", False)
        set_bool("navGoHome",   False)
        clear_rel("navPalletToPick")
        clear_rel("navDropTransform")

        set_str("navPalletToPickId", "{pallet_id}")
        set_str("navAreaToDrop",     "{drop_area_id}")
        set_bool("navGoToPick", True)
        print("[bridge] move {pallet_id} -> {drop_area_id}")
    """)


def _script_move_by_offset(pallet_id: str, offset_x: float, offset_y: float) -> str:
    """Pick the pallet, then create/update a temporary drop-transform prim at
    (pallet_world_pos + offset) and set it as navDropTransform."""
    return _forklift_prim_setup() + textwrap.dedent(f"""

        import math
        from pxr import UsdGeom, Gf, Usd

        # Find the pallet prim by palletId attribute
        pallet_prim = None
        for p in stage.Traverse():
            a = p.GetAttribute("palletId")
            if a and a.IsValid() and str(a.Get()) == "{pallet_id}":
                pallet_prim = p
                break
        if pallet_prim is None:
            # Fall back: search by prim name containing the id
            for p in stage.Traverse():
                if "{pallet_id}".lower() in str(p.GetPath()).lower():
                    pallet_prim = p
                    break
        if pallet_prim is None:
            raise RuntimeError("Pallet '{pallet_id}' not found in stage")

        xform   = UsdGeom.Xformable(pallet_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos     = xform.ExtractTranslation()
        target  = Gf.Vec3d(float(pos[0]) + {offset_x},
                           float(pos[1]) + {offset_y},
                           float(pos[2]))

        # Create or update a temporary drop-transform prim
        drop_path = "/World/TempDropTransform"
        drop_prim = stage.GetPrimAtPath(drop_path)
        if not drop_prim or not drop_prim.IsValid():
            drop_prim = stage.DefinePrim(drop_path, "Xform")
        xf = UsdGeom.Xformable(drop_prim)
        ops = {{op.GetOpName() for op in xf.GetOrderedXformOps()}}
        if "xformOp:translate" not in ops:
            xf.AddTranslateOp().Set(target)
        else:
            drop_prim.GetAttribute("xformOp:translate").Set(target)

        set_bool("navGoTo",     False)
        set_bool("navGoToPick", False)
        set_bool("navGoToDrop", False)
        set_bool("navGoHome",   False)
        clear_rel("navPalletToPick")
        clear_rel("navDropTransform")

        set_str("navPalletToPickId", "{pallet_id}")

        rel = prim.GetRelationship("navDropTransform")
        if rel: rel.SetTargets([Sdf.Path(drop_path)])

        set_bool("navGoToPick", True)
        print(f"[bridge] move {pallet_id} by offset ({offset_x}, {offset_y}) -> {{target}}")
    """)


def _script_drop(drop_area_id: str) -> str:
    return _forklift_prim_setup() + textwrap.dedent(f"""

        set_str("navAreaToDrop", "{drop_area_id}")
        set_bool("navGoToDrop", True)
        print("[bridge] drop -> {drop_area_id}")
    """)


def _script_go_home() -> str:
    return _forklift_prim_setup() + textwrap.dedent("""

        set_bool("navGoTo",     False)
        set_bool("navGoToPick", False)
        set_bool("navGoToDrop", False)
        set_bool("navGoHome",   True)
        print("[bridge] go_home -> navGoHome=True")
    """)


def _script_stop() -> str:
    return _forklift_prim_setup() + textwrap.dedent("""

        for attr in ("navGoTo", "navGoToPick", "navGoToDrop", "navGoHome"):
            set_bool(attr, False)
        print("[bridge] stop -> all nav bools cleared")
    """)


# ── TCP socket sender ────────────────────────────────────────────────────────

def _run_script(script: str) -> dict:
    """Send a Python script to the custom TCP receiver running inside Isaac Sim.

    Before using sim_bridge.py you must paste and run this one-time setup
    snippet in Isaac Sim's Script Editor (Window -> Script Editor):

        import threading, socket, json
        import omni.usd
        def _handle(conn):
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk: break
                data += chunk
            try:
                ns = {"stage": omni.usd.get_context().get_stage()}
                exec(json.loads(data.decode()).get("script", ""), ns)
                conn.sendall(b'{"status":"ok"}')
            except Exception as e:
                conn.sendall(json.dumps({"status":"error","message":str(e)}).encode())
            conn.close()
        def _serve():
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 8765)); s.listen(5)
            print("[SimBridge] Listening on 8765")
            while True:
                conn,_ = s.accept()
                threading.Thread(target=_handle, args=(conn,), daemon=True).start()
        threading.Thread(target=_serve, daemon=True).start()
    """
    payload = json.dumps({"script": script}).encode()
    try:
        with socket.create_connection((ISAAC_SIM_HOST, ISAAC_SIM_PORT), timeout=TIMEOUT_S) as sock:
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)   # signal end-of-message
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        result = json.loads(response.decode())
        if result.get("status") == "ok":
            return {"ok": True, "message": "Script executed successfully"}
        else:
            return {"ok": False, "message": result.get("message", "Unknown error from sim")}
    except ConnectionRefusedError:
        return {
            "ok": False,
            "message": (
                f"Connection refused on {ISAAC_SIM_HOST}:{ISAAC_SIM_PORT}.\n"
                "Make sure you ran the receiver snippet in Isaac Sim's Script Editor first."
            ),
        }
    except Exception as exc:
        return {"ok": False, "message": f"TCP error: {exc}"}


# ── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from llm_task_parser import parse_command

    TEST_COMMANDS = [
        "move blockpallet_a09 one metre to the left",
    ]

    for cmd in TEST_COMMANDS:
        print(f"\nCommand : {cmd!r}")
        tasks = parse_command(cmd)
        for task in tasks:
            print(f"  Task   : {task.action}  pallet={task.pallet_id}  area={task.drop_area_id}  offset=({task.offset_x},{task.offset_y})")
            result = dispatch_task(task)
            status = "OK" if result["ok"] else "FAIL"
            print(f"  Result : [{status}] {result['message'][:120]}")
EOF
