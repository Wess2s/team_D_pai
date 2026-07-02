import threading
import socket
import json
import queue
import omni.usd
import omni.kit.app

_q = queue.Queue()

def _execute_pending(event):
    while not _q.empty():
        code, evt, holder = _q.get_nowait()
        try:
            ns = {"stage": omni.usd.get_context().get_stage()}
            exec(code, ns)
            holder["result"] = {"status": "ok"}
        except Exception as e:
            holder["result"] = {"status": "error", "message": str(e)}
        evt.set()

_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_push(
    _execute_pending, name="sim_bridge_update"
)

def _handle(conn):
    data = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    code = json.loads(data.decode()).get("script", "")
    holder = {}
    evt = threading.Event()
    _q.put((code, evt, holder))
    evt.wait(timeout=30)
    result = holder.get("result", {"status": "error", "message": "main-thread timeout"})
    conn.sendall(json.dumps(result).encode())
    conn.close()

def _serve():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 8765))
    s.listen(5)
    print("[SimBridge] Listening on 8765")
    while True:
        conn, _ = s.accept()
        threading.Thread(target=_handle, args=(conn,), daemon=True).start()

threading.Thread(target=_serve, daemon=True).start()
print("[SimBridge] Server started")