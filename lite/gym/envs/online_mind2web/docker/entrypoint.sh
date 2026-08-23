#!/bin/bash
# Entrypoint for the cua-lite/online_mind2web shared-backend container.
#
# One long-lived container owns a Playwright Chromium browser. Each trajectory
# gets a fresh browser context/page via the FastAPI RPC server.
set -euo pipefail

RPC_PORT="${ONLINE_MIND2WEB_RPC_PORT:-8000}"

echo "[online_mind2web container] launching RPC server on :${RPC_PORT}" >&2
exec uvicorn server:app --app-dir /usr/local/bin --host 0.0.0.0 --port "${RPC_PORT}"
