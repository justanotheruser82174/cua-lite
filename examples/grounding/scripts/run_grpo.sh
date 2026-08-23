#!/bin/bash
#
# Grounding GRPO training on Slime.
#
# Purpose:
#   Thin wrapper over scripts/train/run_grpo.sh for the multi-turn grounding
#   example. It selects the model-family-specific grounding config and routes
#   direct in-process rollouts through examples.grounding.rollout_grpo.
#
# Required:
#   Run inside the Slime container after installing OSWorld-G data and exporting
#   prompt parquet rows. Set NUM_TRAIN_GPUS and NUM_ROLLOUT_GPUS for the
#   launched GPUs.
#
# Common overrides:
#   Inputs:
#     MODEL_ID, HF_CKPT, PROMPT_DATA, EVAL_PROMPT_DATA, CONFIG_PATH
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
#     PROMPT_DATA=/root/datasets/cua-lite/osworld_g/train.parquet \
#     EVAL_PROMPT_DATA=/root/datasets/cua-lite/osworld_g/eval.parquet \
#     ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=1 \
#     ENV_CONCURRENCY=256 \
#     bash examples/grounding/scripts/run_grpo.sh

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
#                                       Grounding Defaults                                        #
# =============================================================================================== #

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
export ENV_ID="${ENV_ID:-osworld_g}"
source "${CUA_LITE_ROOT}/scripts/train/utils/models.sh"

export CONFIG_PATH="${CONFIG_PATH:-${CUA_LITE_ROOT}/examples/grounding/configs/${MODEL_FAMILY}/${ENV_ID}.regionfocus.yaml}"
export PROMPT_DATA="${PROMPT_DATA:-/root/datasets/cua-lite/${ENV_ID}/train.parquet}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export ENV_CONCURRENCY="${ENV_CONCURRENCY:-256}"
export ROLLOUT_MODULE="${ROLLOUT_MODULE:-examples.grounding.rollout_grpo}"
export WANDB_GROUP_SUFFIX="${WANDB_GROUP_SUFFIX:-_regionfocus}"

if [ -n "${CUA_LITE_ENV_SERVER_URL:-}${CUA_LITE_ENV_SERVER_TOKEN:-}" ]; then
    echo "[run_grpo_grounding] using direct mode; ignoring inherited env-server settings"
fi
unset CUA_LITE_ENV_SERVER_URL CUA_LITE_ENV_SERVER_TOKEN

# =============================================================================================== #
#                                        Grounding Checks                                         #
# =============================================================================================== #

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: CONFIG_PATH not found: $CONFIG_PATH" >&2
    exit 1
fi

if [ ! -f "$PROMPT_DATA" ]; then
    cat >&2 <<EOF
Error: PROMPT_DATA not found: $PROMPT_DATA

Create it with:
  python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py
  mkdir -p "$(dirname "$PROMPT_DATA")"
  python -m lite.train.export.export_tasks \\
    --env-id "$ENV_ID" --split eval \\
    --filter "lambda m: not m.others.get('exclude_reason')" \\
    -o "$(dirname "$PROMPT_DATA")/all.parquet"
  python -m lite.data.split \\
    -i "$(dirname "$PROMPT_DATA")/all.parquet" --eval-size 64 \\
    --train-output "$PROMPT_DATA" \\
    --eval-output "$(dirname "$PROMPT_DATA")/eval.parquet"

Or override it:
  PROMPT_DATA=/path/to/train.parquet NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 \\
    bash examples/grounding/scripts/run_grpo.sh
EOF
    exit 1
fi

if [ -z "${NUM_TRAIN_GPUS:-}" ] || [ -z "${NUM_ROLLOUT_GPUS:-}" ]; then
    cat >&2 <<EOF
Error: Grounding GRPO requires explicit GPU counts.

Set both for the Slime container you launched, for example:
  NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 bash examples/grounding/scripts/run_grpo.sh
EOF
    exit 1
fi

# =============================================================================================== #
#                                    Delegate To GRPO Launcher                                    #
# =============================================================================================== #

echo "[run_grpo_grounding] MODEL_ID=${MODEL_ID}"
echo "[run_grpo_grounding] MODEL_FAMILY=${MODEL_FAMILY}"
echo "[run_grpo_grounding] ROLLOUT_MODULE=${ROLLOUT_MODULE}"
echo "[run_grpo_grounding] CONFIG_PATH=${CONFIG_PATH}"

exec bash "$CANONICAL" "$@"
