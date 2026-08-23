#!/bin/bash
# Entrypoint for the cua-lite/webharbor.webvoyager shared-backend container.
#
# One container owns both halves of the benchmark:
#   1. WebHarbor's 15 self-hosted WebVoyager mirror sites.
#   2. The FastAPI RPC server that owns Selenium Chrome sessions.
#
# `docker run --init` should be used by the host services so Chromium and
# WebHarbor child processes are reaped by tini.
set -euo pipefail

RPC_PORT="${WEBHARBOR_WEBVOYAGER_RPC_PORT:-8000}"
START_WEBSYN="${WEBHARBOR_WEBVOYAGER_START_WEBSYN:-1}"
WEBSYN_CONTROL_PORT=8101

wait_for_websyn() {
  local deadline
  deadline=$((SECONDS + ${WEBSYN_START_TIMEOUT_S:-120}))
  while (( SECONDS < deadline )); do
    if python3 - "$WEBSYN_CONTROL_PORT" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

port = int(sys.argv[1])
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
    data = json.loads(r.read())
if not data.get("ok"):
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

if [[ "$START_WEBSYN" != "0" && "$START_WEBSYN" != "false" && "$START_WEBSYN" != "False" ]]; then
  echo "[webharbor.webvoyager container] launching WebHarbor mirrors on :40000-40014" >&2
  /opt/websyn_start.sh &
  WEBSYN_PID="$!"
  if ! wait_for_websyn; then
    echo "[webharbor.webvoyager container] WebHarbor did not become healthy; recent site/control logs:" >&2
    ls -1 /tmp/websyn_*.log 2>/dev/null | tail -5 | xargs -r -I{} sh -c 'echo "--- {}"; tail -40 "{}"' >&2
    kill "$WEBSYN_PID" >/dev/null 2>&1 || true
    exit 1
  fi
  echo "[webharbor.webvoyager container] WebHarbor mirrors healthy on control :${WEBSYN_CONTROL_PORT}" >&2
fi

echo "[webharbor.webvoyager container] launching RPC server on :${RPC_PORT}" >&2
exec uvicorn server:app --app-dir /usr/local/bin --host 0.0.0.0 --port "${RPC_PORT}"
