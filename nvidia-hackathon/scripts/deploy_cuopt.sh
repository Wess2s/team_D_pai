#!/usr/bin/env bash
# Deploy a self-hosted cuOpt REST server on the DGX (detached, always-on).
# The FleetMind planner uses this whenever CUOPT_URL is set, so it never falls
# back to the local solver. run_scene.sh points isaac-sim at http://localhost:5000.
# Usage: bash scripts/deploy_cuopt.sh
set -euo pipefail

# Public examples image (no NGC subscription needed). Update the tag if a newer one exists.
IMAGE="${CUOPT_IMAGE:-nvidia/cuopt:25.12.0a-cuda12.9-py3.13}"
PORT="${CUOPT_PORT:-5000}"
NAME="${CUOPT_NAME:-cuopt}"

echo "[cuOpt] Pulling $IMAGE ..."
docker pull "$IMAGE"

echo "[cuOpt] Removing any existing '$NAME' container..."
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[cuOpt] Launching detached cuOpt server on :$PORT ..."
# Use the image's built-in entrypoint/CMD (python3.13 -m cuopt_server.cuopt_service).
# Do NOT override the command: cuopt_server is installed only for python3.13, while the
# image's /usr/bin/python3 is 3.10. Configure via the server's env vars instead.
docker run -d --restart unless-stopped --name "$NAME" --gpus all --network=host \
    -e "CUOPT_SERVER_PORT=$PORT" \
    -e "CUOPT_SERVER_IP=0.0.0.0" \
    "$IMAGE"

echo "[cuOpt] Waiting for health on http://localhost:$PORT/cuopt/health ..."
for i in $(seq 1 60); do
    if curl -fsS "http://localhost:$PORT/cuopt/health" >/dev/null 2>&1; then
        echo "[cuOpt] Healthy -> $(curl -fsS http://localhost:$PORT/cuopt/health)"
        exit 0
    fi
    sleep 5
done

echo "[cuOpt] WARNING: health check did not pass in time. Recent logs:" >&2
docker logs --tail 30 "$NAME" >&2 || true
exit 1
