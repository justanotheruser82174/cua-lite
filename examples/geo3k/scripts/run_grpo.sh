#!/bin/bash
#
# Geo3K GRPO training on Slime.
#
# Purpose:
#   Thin wrapper over scripts/train/run_grpo.sh for the Geo3K geometry-QA env.
#   It selects the model-family-specific Geo3K config, registers the out-of-tree
#   env on Ray actors, and forces direct in-process env mode.
#
# Required:
#   Run inside the Slime container after exporting a Geo3K prompt parquet.
#   Set NUM_TRAIN_GPUS and NUM_ROLLOUT_GPUS for the container you launched.
#
# Common overrides:
#   Inputs:
#     MODEL_ID, HF_CKPT, GEO3K_SOURCE, PROMPT_DATA, CONFIG_PATH
#   Capacity:
#     NUM_TRAIN_GPUS, NUM_ROLLOUT_GPUS, ENV_CONCURRENCY
#   Training:
#     ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, NUM_STEPS_PER_ROLLOUT, TP_SIZE
#   Output/logging:
#     DUMP, DUMP_DIR, WANDB_API_KEY
#
# Example:
#   MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
#     NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 \
#     PROMPT_DATA=/root/datasets/cua-lite/geo3k/train.parquet \
#     ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=1 \
#     ENV_CONCURRENCY=256 \
#     bash examples/geo3k/scripts/run_grpo.sh

# =============================================================================================== #
#                                            Bootstrap                                            #
# =============================================================================================== #

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CUA_LITE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
CANONICAL="${CUA_LITE_ROOT}/scripts/train/run_grpo.sh"

if [ ! -f "$CANONICAL" ]; then
    echo "Error: $CANONICAL not found (are you inside the Slime container?)" >&2
    exit 1
fi

# =============================================================================================== #
#                                         Geo3K Defaults                                          #
# =============================================================================================== #

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
export ENV_ID="${ENV_ID:-geo3k}"
source "${CUA_LITE_ROOT}/scripts/train/utils/models.sh"

export CONFIG_PATH="${CONFIG_PATH:-${CUA_LITE_ROOT}/examples/geo3k/configs/${MODEL_FAMILY}/geo3k.yaml}"
export PROMPT_DATA="${PROMPT_DATA:-/root/datasets/cua-lite/${ENV_ID}/train.parquet}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export ENV_CONCURRENCY="${ENV_CONCURRENCY:-256}"
export ROLLOUT_MODULE="${ROLLOUT_MODULE:-lite.train.rollout.grpo}"
export CUA_LITE_REGISTRATION_MODULES="${CUA_LITE_REGISTRATION_MODULES:+${CUA_LITE_REGISTRATION_MODULES},}examples.geo3k.registration"
export WANDB_GROUP_SUFFIX="${WANDB_GROUP_SUFFIX:-_geo3k}"

if [ -n "${CUA_LITE_ENV_SERVER_URL:-}${CUA_LITE_ENV_SERVER_TOKEN:-}" ]; then
    echo "[run_grpo_geo3k] using direct mode; ignoring inherited env-server settings"
fi
unset CUA_LITE_ENV_SERVER_URL CUA_LITE_ENV_SERVER_TOKEN

# =============================================================================================== #
#                                          Geo3K Checks                                           #
# =============================================================================================== #

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: CONFIG_PATH not found: $CONFIG_PATH" >&2
    exit 1
fi

if [ ! -f "$PROMPT_DATA" ]; then
    cat >&2 <<EOF
Error: PROMPT_DATA not found: $PROMPT_DATA

Create it with:
  bash examples/geo3k/scripts/install.sh
  mkdir -p "$(dirname "$PROMPT_DATA")"
  export GEO3K_SOURCE="${GEO3K_SOURCE:-${CUA_LITE_ROOT}/examples/geo3k/.cache/geo3k_imgurl/train.parquet}"
  python -m examples.geo3k.export_tasks \\
    --env-id "$ENV_ID" --split train \\
    -o "$PROMPT_DATA"

Or override it:
  PROMPT_DATA=/path/to/train.parquet NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 \\
    bash examples/geo3k/scripts/run_grpo.sh
EOF
    exit 1
fi

if [ -z "${GEO3K_SOURCE:-}" ]; then
    cat >&2 <<EOF
Error: GEO3K_SOURCE is required for Geo3K GRPO.

Set it to the source parquet used to export PROMPT_DATA, for example:
  export GEO3K_SOURCE=${CUA_LITE_ROOT}/examples/geo3k/.cache/geo3k_imgurl/train.parquet
EOF
    exit 1
fi

if [ ! -f "$GEO3K_SOURCE" ]; then
    echo "Error: GEO3K_SOURCE not found: $GEO3K_SOURCE" >&2
    exit 1
fi
export CUA_LITE_RAY_ENV_VARS="${CUA_LITE_RAY_ENV_VARS:+${CUA_LITE_RAY_ENV_VARS},}GEO3K_SOURCE"

if [ -z "${NUM_TRAIN_GPUS:-}" ] || [ -z "${NUM_ROLLOUT_GPUS:-}" ]; then
    cat >&2 <<EOF
Error: Geo3K GRPO requires explicit GPU counts.

Set both for the Slime container you launched, for example:
  NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 bash examples/geo3k/scripts/run_grpo.sh
EOF
    exit 1
fi

# =============================================================================================== #
#                                    Delegate To GRPO Launcher                                    #
# =============================================================================================== #

echo "[run_grpo_geo3k] MODEL_ID=${MODEL_ID}"
echo "[run_grpo_geo3k] MODEL_FAMILY=${MODEL_FAMILY}"
echo "[run_grpo_geo3k] ROLLOUT_MODULE=${ROLLOUT_MODULE}"
echo "[run_grpo_geo3k] CONFIG_PATH=${CONFIG_PATH}"
echo "[run_grpo_geo3k] registration modules: ${CUA_LITE_REGISTRATION_MODULES}"
echo "[run_grpo_geo3k] GEO3K_SOURCE=${GEO3K_SOURCE}"

exec bash "$CANONICAL" "$@"
