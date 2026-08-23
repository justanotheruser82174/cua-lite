#!/bin/bash
# Entrypoint for the cua-lite/mobilegym container: serve the prebuilt vite UI on the
# container-internal port 4173, then run the FastAPI RPC server in the foreground
# (PID1). `docker run --init` (tini) reaps the http.server + chromium children.
#
# Only the RPC server (8000) is published to the host; vite (4173) stays internal
# (the in-container MobileGymEnv navigates to http://localhost:4173).
set -euo pipefail

VITE_PORT="${MOBILEGYM_VITE_PORT:-4173}"
RPC_PORT="${MOBILEGYM_RPC_PORT:-8000}"

echo "[mobilegym container] serving vite dist on :${VITE_PORT} (internal)" >&2
python -m http.server "${VITE_PORT}" --directory /opt/mobilegym/dist >/tmp/vite.log 2>&1 &

echo "[mobilegym container] launching RPC server on :${RPC_PORT}" >&2
exec uvicorn server:app --app-dir /usr/local/bin --host 0.0.0.0 --port "${RPC_PORT}"
