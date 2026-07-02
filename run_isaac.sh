#!/usr/bin/env bash
# run_isaac.sh — stop any existing isaac-sim container and recreate it
# mounting THIS repo's nvidia-hackathon/ folder as /workspace.
# Always edit in team_D_pai/nvidia-hackathon/; never touch /home/deloitte/nvidia-hackathon.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${REPO_DIR}/nvidia-hackathon"
CONTAINER="isaac-sim"
IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1"
BASE_URL="${BASE_URL:-http://localhost:8080}"
TIMEOUT_S="${TIMEOUT_S:-180}"

echo "[run_isaac] workspace : ${WORKSPACE}"
echo "[run_isaac] container : ${CONTAINER}"

# ── 1. compile-check before touching the container ─────────────────────────
echo "[run_isaac] compile-checking Python…"
python3 -m py_compile \
    "${WORKSPACE}/scenes/scene_exec.py" \
    "${WORKSPACE}/src/sim/fleet_bus.py" \
    "${WORKSPACE}/src/sim/isaac_nav_bridge.py" \
    "${WORKSPACE}/src/sim/warehouse_sim.py" \
    "${WORKSPACE}/src/ros2/bridge_server.py" \
    && echo "[run_isaac] Python OK"
node --check "${WORKSPACE}/src/ui/web/app.js" && echo "[run_isaac] JS OK"

# ── 2. stop + remove old container (if any) ────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "[run_isaac] removing existing ${CONTAINER}…"
    docker rm -f "${CONTAINER}" >/dev/null
fi

# ── 3. run fresh container mounting THIS repo ───────────────────────────────
echo "[run_isaac] starting container from ${WORKSPACE}…"
docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    --network=host \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ISAACSIM_HOST=100.104.13.18 \
    -v /home/deloitte/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
    -v /home/deloitte/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
    -v /home/deloitte/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
    -v /home/deloitte/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
    -v /home/deloitte/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
    -v /home/deloitte/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
    -v /home/deloitte/.cache/ov/hub:/var/cache/hub:rw \
    -v "${WORKSPACE}:/workspace:ro" \
    -u 1234:1234 \
    --entrypoint bash "${IMAGE}" \
    -lc "./runheadless.sh --exec /workspace/scenes/scene_exec.py" \
    >/dev/null

echo "[run_isaac] container started — mount: ${WORKSPACE} → /workspace"
echo "[run_isaac] waiting for bridge at ${BASE_URL}/health (up to ${TIMEOUT_S}s)…"

elapsed=0
while (( elapsed < TIMEOUT_S )); do
    if curl -s --max-time 2 "${BASE_URL}/health" 2>/dev/null | grep -q '"ok":true'; then
        echo "[run_isaac] BRIDGE READY (~${elapsed}s)"
        break
    fi
    sleep 5
    elapsed=$(( elapsed + 5 ))
done

if (( elapsed >= TIMEOUT_S )); then
    echo "[run_isaac] WARNING: bridge not healthy after ${TIMEOUT_S}s" >&2
    echo "[run_isaac]   check: docker logs --tail 60 ${CONTAINER}" >&2
    exit 2
fi

# ── 4. confirm mount and basic state ───────────────────────────────────────
echo "[run_isaac] mount in use: $(docker inspect ${CONTAINER} --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}')"
echo "[run_isaac] /state summary:"
python3 - <<'PY'
import json
from urllib import request
s = json.loads(request.urlopen("http://localhost:8080/state", timeout=5).read())
for k, v in s.get("forklifts", {}).items():
    print(f"  {k}: phase={v['phase']} battery={v.get('battery', '?')}%")
print(f"  chargers: {list((s.get('chargers') or {}).keys())}")
print(f"  pallets: {len(s.get('pallets', {}))} | zones: {len(s.get('zones', {}))}")
PY
