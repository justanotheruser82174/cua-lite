#!/usr/bin/env bash
# Parallel smoke-validation matrix for env-server lifecycle families.
#
# Runs one real gpt-5.5 single-task rollout per env in parallel. Required:
# OPENAI_API_KEY. In MODE=server, also set TOKEN for the env-server bearer;
# PORT defaults to 30220. MODE=direct skips the remote env-server.
#
# The example below covers SINGLETON, DEDICATED, and PURE env families. Add
# env-specific config paths before including envs that lack GPT defaults.
#
# Usage:
#   OPENAI_API_KEY=<key> PORT=30220 TOKEN=<env-server-token> MODE=server \
#     bash scripts/experiments/smoke_matrix.sh \
#       mobilegym online_mind2web webharbor.webvoyager screenspot_pro lite.osworld
set -u
PORT=${PORT:-30220}; MODE=${MODE:-server}
if [ "$MODE" = "server" ]; then
  : "${TOKEN:?Set TOKEN=<env-server-token> for MODE=server}"
else
  TOKEN=${TOKEN:-}
fi
URL="http://127.0.0.1:${PORT}"
ROOT=".logs/smoke/matrix_${MODE}"
mkdir -p "$ROOT"
# API-model smoke needs OpenAI credentials. Source credentials before calling
# this script; do not execute shell text from user rc files here.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this smoke script}"

UV_PY=(uv run python)
PIDS=()
PID_ENVS=()

# Resolve one task-id for an env. Prefer the env-server catalog (server mode); fall back to a
# direct registry probe (direct mode) so this works without a running server.
resolve_tid() {
  local e="$1"
  if [ "$MODE" = "server" ]; then
    curl -s -H "Authorization: Bearer $TOKEN" "$URL/envs/$e/tasks" 2>/dev/null \
      | SMOKE_ENV="$e" python3 -c "import itertools,json,os,sys; d=json.load(sys.stdin); sp=d.get('splits') or {}; meta=d.get('metadata') or {}; ids=list(itertools.chain.from_iterable(sp.values())); e=os.environ['SMOKE_ENV']; print(next((tid for tid in ids if not (e == 'lite.cuagym' and (meta.get(tid, {}).get('others') or {}).get('exclude_reason'))), ''))" 2>/dev/null
  else
    env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN SMOKE_ENV="$e" \
      "${UV_PY[@]}" -c "import itertools,os; import lite.gym as gym; e=os.environ['SMOKE_ENV']; sp=gym.registry.task_ids(e); ids=list(itertools.chain.from_iterable(sp.values())) if isinstance(sp, dict) else list(sp); print(next((tid for tid in ids if not (e == 'lite.cuagym' and (gym.registry.task_metadata(e, tid).others or {}).get('exclude_reason'))), ''))" 2>/dev/null
  fi
}

# First GPT config that exists for an env: default family, collect recipe,
# env default, then env SoM.
resolve_cfg() {
  local e="$1"
  local family="$e"
  [[ "$e" == lite.cuaworld.* ]] && family="lite.cuaworld"
  ls scripts/configs/gpt/default/$family.yaml \
     scripts/configs/gpt/recipes/collect/$e.yaml \
     scripts/configs/gpt/default/$e/default.yaml \
     scripts/configs/gpt/default/$e/som.yaml 2>/dev/null | head -1
}

for E in "$@"; do
  TID=$(resolve_tid "$E")
  CFG=$(resolve_cfg "$E")
  if [ -z "$TID" ] || [ -z "$CFG" ]; then
    echo "SKIP   $E (task_id='$TID' cfg='$CFG')"
    continue
  fi
  if [ "$MODE" = "server" ]; then
    RUN_ENV=(env CUA_LITE_ENV_SERVER_URL="$URL" CUA_LITE_ENV_SERVER_TOKEN="$TOKEN")
  else
    RUN_ENV=(env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN)
  fi
  "${RUN_ENV[@]}" nohup "${UV_PY[@]}" scripts/rollout.py \
      --model-id gpt-5.5 --env-id "$E" --task-id "$TID" \
      --config-path "$CFG" --log-root "$ROOT/$E" > "$ROOT/${E}.out" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  PID_ENVS+=("$E")
  echo "LAUNCH $E mode=$MODE task=$TID pid=$pid cfg=$CFG"
done

status=0
for i in "${!PIDS[@]}"; do
  env_id="${PID_ENVS[$i]}"
  pid="${PIDS[$i]}"
  if wait "$pid"; then
    echo "PASS   $env_id (log: $ROOT/${env_id}.out)"
  else
    rc=$?
    echo "FAIL   $env_id rc=$rc (log: $ROOT/${env_id}.out)"
    status=1
  fi
done
echo "--- smoke matrix complete; logs in $ROOT/<env>.out ---"
exit "$status"
