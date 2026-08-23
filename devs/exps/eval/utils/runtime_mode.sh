#!/usr/bin/env bash
# Shared eval runtime-mode guard.
#
# Source from devs/exps/eval/*/run.sh with EVAL_ENV_ID set. Official eval
# campaigns default to server mode and fail fast when the env-server is absent
# or does not serve the requested env. Use EVAL_ALLOW_DIRECT=1 for an explicit
# direct-mode dev run; this clears env-server vars so gym.make cannot route
# remotely by accident.

if [ -z "${EVAL_ENV_ID:-}" ]; then
  echo "[run.sh] ERROR: EVAL_ENV_ID must be set before sourcing runtime_mode.sh" >&2
  exit 1
fi

if [ -z "${SESSION_ID:-}" ]; then
  _EVAL_SAFE_ENV_ID="$(printf '%s' "${EVAL_ENV_ID}" | tr -c 'A-Za-z0-9_.-' '-')"
  _EVAL_SAFE_RUN_ID="$(printf '%s' "${EVAL_RUN_ID:-run}" | tr -c 'A-Za-z0-9_.-' '-')"
  export SESSION_ID="eval-${_EVAL_SAFE_ENV_ID}-${_EVAL_SAFE_RUN_ID}-$$"
  unset _EVAL_SAFE_ENV_ID _EVAL_SAFE_RUN_ID
fi

if [ "${EVAL_ALLOW_DIRECT:-0}" = "1" ]; then
  unset CUA_LITE_ENV_SERVER_URL
  unset CUA_LITE_ENV_SERVER_TOKEN
  export CUA_LITE_EVAL_RUNTIME_MODE=direct
  echo "[run.sh] runtime_mode=direct (EVAL_ALLOW_DIRECT=1)"
else
  if [ -z "${CUA_LITE_ENV_SERVER_URL:-}" ] || [ -z "${CUA_LITE_ENV_SERVER_TOKEN:-}" ]; then
    echo "[run.sh] ERROR: env-server required for official eval." >&2
    echo "         Set CUA_LITE_ENV_SERVER_URL and CUA_LITE_ENV_SERVER_TOKEN," >&2
    echo "         or set EVAL_ALLOW_DIRECT=1 for an explicit direct-mode dev run." >&2
    exit 1
  fi

  _EVAL_ENV_URL="${CUA_LITE_ENV_SERVER_URL%/}/envs/${EVAL_ENV_ID}"
  _EVAL_AUTH="Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}"
  _EVAL_RESP=$(curl -sf --max-time "${EVAL_ENV_SERVER_PROBE_TIMEOUT:-30}" \
    -H "${_EVAL_AUTH}" "${_EVAL_ENV_URL}") || {
      echo "[run.sh] ERROR: env-server probe failed for ${_EVAL_ENV_URL}" >&2
      echo "         Check URL/token, server readiness, and that the server serves ${EVAL_ENV_ID}." >&2
      exit 1
    }
  if ! printf '%s' "${_EVAL_RESP}" | grep -Eq '"available"[[:space:]]*:[[:space:]]*true'; then
    _EVAL_ERR=$(printf '%s' "${_EVAL_RESP}" \
      | grep -oE '"error"[[:space:]]*:[[:space:]]*"[^"]*"' \
      | cut -d\" -f4 | head -c 240)
    echo "[run.sh] ERROR: env-server reports ${EVAL_ENV_ID} unavailable: ${_EVAL_ERR:-<no error field>}" >&2
    exit 1
  fi
  export CUA_LITE_EVAL_RUNTIME_MODE=server
  echo "[run.sh] runtime_mode=server env_id=${EVAL_ENV_ID} session_id=${SESSION_ID} env_server_url_present=true"
  unset _EVAL_ENV_URL _EVAL_AUTH _EVAL_RESP _EVAL_ERR
fi
