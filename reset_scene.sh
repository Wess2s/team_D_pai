#!/usr/bin/env bash
#
# reset_scene.sh — reset the FleetMind Isaac Sim scene to its initial state.
#
# Restarts the isaac-sim Docker container (the only reliable reset — the HTTP
# bridge exposes no /reset endpoint), then waits for the :8080 bridge to come
# back healthy and prints the fresh scene state.
#
# Usage:
#   ./reset_scene.sh                 # restart, wait, show state
#   ./reset_scene.sh --no-wait       # just issue the restart and exit
#   CONTAINER=isaac-sim BASE_URL=http://localhost:8080 ./reset_scene.sh
#
set -euo pipefail

CONTAINER="${CONTAINER:-isaac-sim}"
BASE_URL="${BASE_URL:-http://localhost:8080}"
TIMEOUT_S="${TIMEOUT_S:-120}"     # max seconds to wait for the bridge
POLL_S="${POLL_S:-5}"

wait_for_bridge=true
[[ "${1:-}" == "--no-wait" ]] && wait_for_bridge=false

echo "[reset] restarting container '${CONTAINER}'..."
if ! docker restart "${CONTAINER}" >/dev/null; then
    echo "[reset] ERROR: could not restart container '${CONTAINER}'." >&2
    echo "[reset] Is it running? Try: docker ps -a --filter name=${CONTAINER}" >&2
    exit 1
fi
echo "[reset] restart issued."

if ! $wait_for_bridge; then
    echo "[reset] --no-wait set; not polling the bridge."
    exit 0
fi

echo "[reset] waiting for bridge at ${BASE_URL}/health (up to ${TIMEOUT_S}s)..."
elapsed=0
ready=false
while (( elapsed < TIMEOUT_S )); do
    if curl -s --max-time 2 "${BASE_URL}/health" 2>/dev/null | grep -q '"ok":true'; then
        ready=true
        break
    fi
    sleep "${POLL_S}"
    elapsed=$(( elapsed + POLL_S ))
done

if ! $ready; then
    echo "[reset] WARNING: bridge not healthy after ${TIMEOUT_S}s. Check: docker logs --tail 50 ${CONTAINER}" >&2
    exit 2
fi
echo "[reset] bridge READY after ~${elapsed}s."

# Print the fresh scene state (best-effort; needs python3 + repo modules).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="${BASE_URL}" python3 - "${SCRIPT_DIR}" <<'PY' || true
import os, sys
sys.path.insert(0, sys.argv[1])
from isaac_dispatch import fetch_state
s = fetch_state(os.environ.get("BASE_URL", "http://localhost:8080"))
print("[reset] scene state:")
for k, v in (s.get("forklifts") or {}).items():
    print(f"  {k}: phase={v['phase']} pos=({v['x']:.1f},{v['y']:.1f}) carrying={v['carrying']}")
pallets = s.get("pallets") or {}
delivered = sum(1 for p in pallets.values() if p.get("delivered"))
print(f"  pallets: {len(pallets)} (delivered {delivered}) | zones: {len(s.get('zones') or {})}")
PY

echo "[reset] done. Run:  python3 fleet_orchestrator.py --include-busy"
