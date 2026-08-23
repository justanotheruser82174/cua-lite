#!/usr/bin/env bash
# Launch the Slime training container.
#
# Model/env separation is the default: this container is pure
# model training + rollout orchestration. All env runtimes (docker,
# KVM, android emulators, browsers, redis) live on env nodes behind
# `serve_env.py`. See [docs/slime.md](/docs/slime.md) +
# [docs/envs.md](/docs/envs.md) for the full topology.
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train/slime/launch.sh         # specific GPUs
#   SESSION_ID=train-osworld-qwen3_vl_2b bash scripts/train/slime/launch.sh # explicit name
#   MEMORY_LIMIT=1.2t bash scripts/train/slime/launch.sh                    # bigger cap
#   MEMORY_LIMIT=none bash scripts/train/slime/launch.sh                    # no cgroup cap
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CUA_LITE_HOST_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

# =============================================================================================== #
#                                    Image And Memory Defaults                                    #
# =============================================================================================== #

IMAGE="slimerl/slime:v0.3.0"

# Default container cgroup memory cap. ray.sh's plasma sizer reads
# /sys/fs/cgroup/memory.max from inside the container and budgets accordingly,
# so a cap here gives a precise reading there too. Tune MEMORY_LIMIT for the
# host and workload; use MEMORY_LIMIT=none only on a dedicated machine.
#
# Overrides:
#   MEMORY_LIMIT=1.2t / =400g / ...  → docker --memory=<value>
#   MEMORY_LIMIT=none                → no --memory flag; risks host OOM kill
MEMORY_LIMIT="${MEMORY_LIMIT:-1t}"

# =============================================================================================== #
#                                       Container Identity                                        #
# =============================================================================================== #

# Explicit: SESSION_ID=foo -> lite.slime-foo.
# Auto: no SESSION_ID -> smallest free numeric id -> lite.slime-1, ...
# SESSION_ID is propagated into the container so the env server can tag docker /
# emulator resources by it and scope cleanup precisely.
if [ -z "${SESSION_ID:-}" ]; then
  i=1
  while docker container inspect "lite.slime-${i}" &>/dev/null; do ((i++)); done
  SESSION_ID="${i}"
fi
CONTAINER_NAME="lite.slime-${SESSION_ID}"
echo "Container: ${CONTAINER_NAME}  (SESSION_ID=${SESSION_ID})"

# =============================================================================================== #
#                                      Docker Runtime Args                                        #
# =============================================================================================== #

args=(--name "$CONTAINER_NAME" --init --ipc=host
      --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864)

# Apply MEMORY_LIMIT unless explicitly disabled.
if [ "$MEMORY_LIMIT" != "none" ] && [ -n "$MEMORY_LIMIT" ]; then
  args+=(--memory="$MEMORY_LIMIT")
  echo "Memory cap: --memory=$MEMORY_LIMIT (override with MEMORY_LIMIT=... or =none)"
else
  echo "Memory cap: DISABLED (MEMORY_LIMIT=none — container can use up to host RAM)"
fi

# Network: use Docker's default bridge. Reach env-server via its external IP,
# e.g. ``CUA_LITE_ENV_SERVER_URL=http://<env-host>:30100``; container-local
# ``localhost`` points at the training container, not the env-server host.

# Env var passthrough (training-only — API agent keys live on env nodes).
args+=(
  -e CUA_LITE_ROOT=/workspaces/cua-lite
  -e CUA_LITE_DATASETS_ROOT=/workspaces/cua-lite/.data/huggingface
  -e CUA_LITE_ENV_SERVER_URL                # env server URL (local or remote; see docs/envs.md)
  -e CUA_LITE_ENV_SERVER_TOKEN
  -e HF_TOKEN                               # gated-model download auth
  -e HF_XET_HIGH_PERFORMANCE                # optional host-tuned HF Xet parallelism
  -e WANDB_API_KEY
  -e SESSION_ID="$SESSION_ID"               # tag for env server cleanup scoping
)

# =============================================================================================== #
#                                      Host Mounts And Cache                                      #
# =============================================================================================== #

args+=(-v "${CUA_LITE_HOST_ROOT}":/workspaces/cua-lite)

# Optional shared HF hub cache (read-only) so models resolve from a host-side
# cache instead of re-downloading per container. Point HF_SHARED_HUB_CACHE at the
# host HUB dir that holds `models--<org>--<name>/snapshots/<hash>/`; slime/init.sh
# links cached snapshots into `/root/models/<org>/<name>/`, which is the layout
# megatron and sglang read. The mount is read-only to protect the shared cache.
if [ -n "${HF_SHARED_HUB_CACHE:-}" ] && [ -d "${HF_SHARED_HUB_CACHE}" ]; then
  args+=(-v "${HF_SHARED_HUB_CACHE}":/root/.cache/huggingface/hub:ro)
  echo "Shared HF hub cache: ${HF_SHARED_HUB_CACHE}"
  echo "  mounted at /root/.cache/huggingface/hub (ro)"
  echo "  symlinked into /root/models by slime/init.sh"
fi

# =============================================================================================== #
#                                      Launch And Init                                           #
# =============================================================================================== #

# -- Launch container ----------------------------------------------------------

# --gpus "device=4,5,6,7" makes physical GPUs 4-7 appear as 0-3 inside the
# container — no need to set CUDA_VISIBLE_DEVICES inside.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "${CUA_LITE_SLIME_ALL_VISIBLE_GPUS:-0}" != "1" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is required for slime launch." >&2
    echo "       Set it to the exact host GPUs to reserve, for example CUDA_VISIBLE_DEVICES=4,5." >&2
    echo "       On a dedicated host only, set CUA_LITE_SLIME_ALL_VISIBLE_GPUS=1 to use all GPUs." >&2
    exit 1
  fi
  docker run -d --gpus all "${args[@]}" "$IMAGE" sleep infinity
else
  docker run -d --gpus "\"device=${CUDA_VISIBLE_DEVICES}\"" "${args[@]}" "$IMAGE" sleep infinity
fi

# -- Initialize container ------------------------------------------------------

# Per-container init (HF login + editable install re-point). Always run —
# the container is unusable for training without these steps.
echo "Running slime/init.sh inside ${CONTAINER_NAME}..."
docker exec "${CONTAINER_NAME}" bash /workspaces/cua-lite/scripts/train/slime/init.sh
