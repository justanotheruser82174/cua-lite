#!/usr/bin/env bash

#
# Launch a frozen teacher sglang /generate server on a dynamic free port.
#
# Sourced, then called as a function. The only required arg is ``--gpus``.
# TEACHER_PATH / TEACHER_MODEL_ID / TEACHER_MEM_FRACTION / TEACHER_LOG can
# override the model path, download repo, memory fraction, and log path:
#
#   source "${CUA_LITE_ROOT}/scripts/train/utils/serve_teacher.sh"
#   serve_teacher --gpus "0"
#   export DAGGER_TEACHER_URL="$TEACHER_URL"
#
# Args (all optional except --gpus):
#   --gpus  CSV     CUDA_VISIBLE_DEVICES for the teacher, e.g. "0" or "0,1" (REQUIRED)
#   --model PATH    teacher checkpoint dir (default $TEACHER_PATH; downloaded if missing)
#   --model-id ID   HF repo id for the download-if-missing (default $TEACHER_MODEL_ID)
#   --tp    N       tensor-parallel size (default = number of --gpus)
#   --mem   F       --mem-fraction-static (default $TEACHER_MEM_FRACTION or 0.45)
#   --polls N       health polls (5s each) per attempt before giving up (default 90 → 450s)
#   --tries N       fresh-port attempts before failing (default 3)
#   --extra "..."   extra whitespace-split sglang flags
#
# Sets TEACHER_URL, TEACHER_PORT, TEACHER_PID on success; TEACHER_URL is exported.

serve_teacher() {
  local model="${TEACHER_PATH:-/root/models/Qwen/Qwen3-VL-4B-Instruct}"
  local model_id="${TEACHER_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
  local gpus=""
  local tp=""
  local mem="${TEACHER_MEM_FRACTION:-0.45}"
  local polls=90
  local tries=3
  local extra=""
  local extra_args=()

  while [ $# -gt 0 ]; do
    case "$1" in
      --gpus)     gpus="$2";     shift 2;;
      --model)    model="$2";    shift 2;;
      --model-id) model_id="$2"; shift 2;;
      --tp)       tp="$2";       shift 2;;
      --mem)      mem="$2";      shift 2;;
      --polls)    polls="$2";    shift 2;;
      --tries)    tries="$2";    shift 2;;
      --extra)    extra="$2"; read -r -a extra_args <<< "$extra"; shift 2;;
      *) echo "[serve_teacher] unknown arg: $1" >&2; return 2;;
    esac
  done
  [ -n "$gpus" ] || { echo "[serve_teacher] --gpus required" >&2; return 2; }
  [ -n "$tp" ] || tp=$(awk -F, '{print NF}' <<< "$gpus")          # default tp = #gpus
  if [ ! -e "$model" ]; then
    echo "[serve_teacher] $model missing → hf download $model_id"
    if ! hf download "$model_id" --local-dir "$model"; then
      echo "[serve_teacher] download failed" >&2
      return 1
    fi
  fi
  local log="${TEACHER_LOG:-/tmp/teacher.log}"

  # One launch attempt on a fresh free port; 0 = healthy, 1 = died / timed out.
  _serve_teacher_once() {
    TEACHER_PORT=$(python - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)
    echo "[serve_teacher] launch model=$model gpus=$gpus tp=$tp port=$TEACHER_PORT mem=$mem"
	    CUDA_VISIBLE_DEVICES="$gpus" nohup python -m sglang.launch_server \
	      --model-path "$model" --host 0.0.0.0 --port "$TEACHER_PORT" \
	      --tp-size "$tp" --mem-fraction-static "$mem" --disable-cuda-graph "${extra_args[@]}" \
	      > "$log" 2>&1 &
    TEACHER_PID=$!
    local i
    for i in $(seq 1 "$polls"); do
      curl -sf -m3 "http://127.0.0.1:$TEACHER_PORT/health" >/dev/null 2>&1 && return 0
      kill -0 "$TEACHER_PID" 2>/dev/null || return 1
      sleep 5
    done
    return 1
  }

  local attempt
  for attempt in $(seq 1 "$tries"); do
   if _serve_teacher_once; then
      export TEACHER_URL="http://127.0.0.1:$TEACHER_PORT/generate"
      echo "[serve_teacher] UP → $TEACHER_URL (pid $TEACHER_PID)"
      unset -f _serve_teacher_once
      return 0
   fi
    echo "[serve_teacher] attempt $attempt/$tries failed; retrying on a fresh port"
    kill -9 "${TEACHER_PID:-0}" 2>/dev/null || true
    sleep 2
  done
  echo "[serve_teacher] FAILED after $tries attempts — see $log" >&2
  unset -f _serve_teacher_once
  return 1
}
