#!/usr/bin/env bash
# FleetMind — launch the warehouse scene inside Isaac Sim's proven streaming app.
# The scene is loaded via Kit's --exec hook into runheadless.sh (WebRTC + NVENC).
# Usage:  ./run_scene.sh            (uses the DGX Tailscale IP by default)
#         PUBLIC_IP=1.2.3.4 ./run_scene.sh
set -euo pipefail

PUBLIC_IP="127.0.0.1" #"${PUBLIC_IP:-0.0.0.1}"
IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1"
PROJ="$HOME/team_D_pai"
# cuOpt self-hosted REST server (same host, --network=host) so the in-process planner
# always uses NVIDIA cuOpt — never the local fallback. Override with CUOPT_URL=... .
# -e "ISAACSIM_HOST=$PUBLIC_IP" \
CUOPT_URL="${CUOPT_URL:-http://localhost:5000}"

echo "[FleetMind] Removing any existing isaac-sim container..."
docker rm -f isaac-sim >/dev/null 2>&1 || true

echo "[FleetMind] Launching streaming scene (publicIp=$PUBLIC_IP, cuopt=$CUOPT_URL)..."
nohup docker run --name isaac-sim --gpus all --network=host \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -e "NVIDIA_DRIVER_CAPABILITIES=all" \
  -e "CUOPT_URL=$CUOPT_URL" \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -v ~/.cache/ov/hub:/var/cache/hub:rw \
  -v "$PROJ":/workspace:ro \
  -u 1234:1234 \
  --entrypoint bash "$IMAGE" \
  -lc "./runheadless.sh --exec /workspace/scenes/scene_exec.py" \
  > ~/isaac_scene.log 2>&1 &

echo "[FleetMind] Started (PID $!)."
echo "[FleetMind] Connect the Isaac Sim WebRTC client to $PUBLIC_IP (signal 49100 / stream 47998)."
echo "[FleetMind] Tailing log (Ctrl-C stops tailing; sim keeps running)."
sleep 3
tail -f ~/isaac_scene.log
