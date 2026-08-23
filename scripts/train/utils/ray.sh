#!/usr/bin/env bash
#
# Ray utilities: port detection and head-node startup.
#
# Usage:
#   source scripts/train/utils/ray.sh
#   start_ray        # uses NUM_GPUS from caller, auto-detects free ports

# Find a free TCP port starting from a given base port.
find_free_port() {
  local port="${1:?Usage: find_free_port BASE_PORT}"
  if command -v ss >/dev/null 2>&1; then
    while ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .; do
      ((port++))
    done
    echo "$port"
    return
  fi
  python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
while True:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind(("", port))
        except OSError:
            port += 1
            continue
    print(port)
    break
PY
}

# Start a Ray head node with auto-detected free ports.
# Expects NUM_GPUS to be set by the caller (total GPUs visible to Ray).
# Exports RAY_GCS_PORT, RAY_PORT, RAY_MIN_WORKER_PORT, and
# RAY_AGENT_LISTEN_PORT for downstream use.
start_ray() {
  export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
  export no_proxy="127.0.0.1,${MASTER_ADDR}"

  # Start from 6390, clear of the conventional redis range (6379-6389).
  RAY_GCS_PORT=$(find_free_port "${RAY_GCS_PORT:-6390}")
  RAY_PORT=$(find_free_port "${RAY_PORT:-8265}")
  RAY_MIN_WORKER_PORT=$(find_free_port "${RAY_MIN_WORKER_PORT:-11000}")
  # Dashboard agent listen port (hardcoded to 52365 by default).
  RAY_AGENT_LISTEN_PORT=$(find_free_port "${RAY_AGENT_LISTEN_PORT:-52365}")
  export RAY_GCS_PORT RAY_PORT RAY_MIN_WORKER_PORT RAY_AGENT_LISTEN_PORT

  echo "Ray ports:" \
       "gcs=${RAY_GCS_PORT}" \
       "dashboard=${RAY_PORT}" \
       "agent_listen=${RAY_AGENT_LISTEN_PORT}" \
       "min_worker=${RAY_MIN_WORKER_PORT}"

  # Survive long actor-quiet windows in slime async rollout. Under heavy
  # emulator/container load, rollout workers can block inside backend calls long
  # enough for Ray's client idle timeout to close the channel.
  export RAY_grpc_client_idle_timeout_ms=${RAY_grpc_client_idle_timeout_ms:-1800000}
  export RAY_grpc_client_keepalive_time_ms=${RAY_grpc_client_keepalive_time_ms:-30000}
  export RAY_grpc_client_keepalive_timeout_ms=${RAY_grpc_client_keepalive_timeout_ms:-120000}
  : "${RAY_core_worker_rpc_server_reconnect_timeout_max_s:=1200}"
  export RAY_core_worker_rpc_server_reconnect_timeout_max_s
  # Widen actor health-check budgets too; large rollout-batch serialization can
  # briefly starve a GIL-bound actor even after the idle-timeout bump above.
  export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-120}
  export RAY_health_check_period_ms=${RAY_health_check_period_ms:-10000}
  export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-120000}
  export RAY_health_check_initial_delay_ms=${RAY_health_check_initial_delay_ms:-600000}
  # Server-side gRPC keepalive (emit pings while idle so peers don't drop us).
  export RAY_grpc_keepalive_time_ms=${RAY_grpc_keepalive_time_ms:-30000}
  export RAY_grpc_keepalive_timeout_ms=${RAY_grpc_keepalive_timeout_ms:-120000}

  # Plasma object store cap. Ray's default uses host RAM and ignores cgroup
  # limits; inside constrained containers that can over-allocate before
  # training starts.
  #
  # Detection priority:
  #   1. ``OBJECT_STORE_MEM`` env var override (operator pins explicit value)
  #   2. cgroup v2 memory.max          (container w/ limit)
  #   3. cgroup v1 memory.limit_in_bytes (container w/ limit, old kernel)
  #   4. /proc/meminfo MemTotal         (bare metal / unbounded cgroup)
  #
  # cgroup v1's "unlimited" sentinel is 9223372036854771712 (~8 EiB); treat
  # that and v2's literal "max" as "no cgroup limit set" → fall through to
  # /proc/meminfo so bare-metal / unbounded callers still get a sane cap.
  if [ -z "${OBJECT_STORE_MEM:-}" ]; then
    # Mirror find_free_port's local discipline — these are
    # function-internal helpers that mustn't leak into the caller's shell.
    local _CG_V2 _CG_V1 _MEM_BUDGET _MEM_BUDGET_SRC _v
    _CG_V2=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)
    _CG_V1=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)
    _MEM_BUDGET=0
    _MEM_BUDGET_SRC="unknown"
    for _v in "$_CG_V2" "$_CG_V1"; do
      if [ -n "$_v" ] && [ "$_v" != "max" ] \
         && [ "$_v" -gt 0 ] 2>/dev/null \
         && [ "$_v" -lt 9223372036854771712 ] 2>/dev/null; then
        _MEM_BUDGET=$_v
        _MEM_BUDGET_SRC="cgroup"
        break
      fi
    done
    if [ "$_MEM_BUDGET" -eq 0 ]; then
      _MEM_BUDGET=$(awk '/^MemTotal:/{print $2 * 1024}' /proc/meminfo)
      _MEM_BUDGET_SRC="host RAM (no cgroup limit)"
    fi
    # 15% of budget — leaves 85% for Megatron weights, sglang stage,
    # Python heap, pinned memory. Tune via OBJECT_STORE_MEM for 8B+ models
    # (recommend 10–12%) or for SFT with dp_size=1 large shards (bump up).
    OBJECT_STORE_MEM=$(( _MEM_BUDGET / 100 * 15 ))
    echo "Plasma capped at $((OBJECT_STORE_MEM / 1024 / 1024 / 1024)) GiB" \
         "(15% of $((_MEM_BUDGET / 1024 / 1024 / 1024)) GiB ${_MEM_BUDGET_SRC})"
  else
    echo "Plasma capped at $((OBJECT_STORE_MEM / 1024 / 1024 / 1024)) GiB" \
         "(operator override via OBJECT_STORE_MEM)"
  fi

  ray start --head --port=${RAY_GCS_PORT} \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus ${NUM_GPUS} \
    --object-store-memory=${OBJECT_STORE_MEM} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${RAY_PORT} \
    --dashboard-agent-listen-port=${RAY_AGENT_LISTEN_PORT} \
    --min-worker-port=${RAY_MIN_WORKER_PORT}
}
