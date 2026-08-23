#!/bin/bash
#
# CUA-Lite GSPO training on Slime.
#
# Purpose:
#   Thin wrapper over run_grpo.sh that switches the advantage estimator to GSPO,
#   tightens the clip range, uses one rollout step by default, and disables
#   radix packing by default.
#
# Required:
#   Run inside the Slime container. Same live-training inputs as run_grpo.sh:
#   ENV_ID and PROMPT_DATA. For server-backed envs, also start the env-server
#   and set CUA_LITE_ENV_SERVER_URL and CUA_LITE_ENV_SERVER_TOKEN.
#
# Common overrides:
#   Inputs:
#     MODEL_ID, HF_CKPT, ENV_ID, PROMPT_DATA, EVAL_PROMPT_DATA, CONFIG_PATH
#   Capacity:
#     ASYNC, NUM_TRAIN_GPUS, NUM_ROLLOUT_GPUS, GPUS_PER_ENGINE, ENV_CONCURRENCY
#   GSPO:
#     ALGO, ADV_ESTIMATOR, EPS_CLIP, EPS_CLIP_HIGH, EPS_CLIP_C,
#     NUM_STEPS_PER_ROLLOUT, CUA_LITE_DISABLE_RADIX
#   Output/logging:
#     SAVE, SAVE_DIR, SAVE_HF_DIR, DUMP, DUMP_DIR, WANDB_API_KEY
#   Advanced:
#     ROLLOUT_MODULE, MEM_FRACTION, GRADS_FP32, OPTIM_CPU_OFFLOAD
#   Debug:
#     DEBUG, CUA_LITE_SHELL_TRACE, CUA_LITE_ROLLOUT_PROC_WORKERS,
#     ROLLOUT_STALL_TIMEOUT_S
#   The launch summary is printed by run_grpo.sh after these defaults are exported.
#
# Policy:
#   Faithful GSPO uses NUM_STEPS_PER_ROLLOUT=1. The wrapper leaves it overridable
#   so callers can intentionally trade strict on-policy updates for the GRPO-style
#   multi-step schedule.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
#     MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
#     HF_CKPT=/workspaces/cua-lite/.ckpts/<sft-checkpoint>/hf/<iter> \
#     ENV_ID=mobilegym \
#     PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
#     EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
#     CUA_LITE_ENV_SERVER_URL=http://<env-server-host>:<port> \
#     CUA_LITE_ENV_SERVER_TOKEN=<token> \
#     ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 NUM_STEPS_PER_ROLLOUT=1 \
#     LR=1e-6 NUM_ROLLOUT=40 SAVE=0 \
#     bash /workspaces/cua-lite/scripts/train/run_gspo.sh

# =============================================================================================== #
#                                            Bootstrap                                            #
# =============================================================================================== #

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# =============================================================================================== #
#                                          GSPO Defaults                                          #
# =============================================================================================== #

# GSPO overrides — each still respects a caller-supplied non-empty value via :-.
export ALGO=${ALGO:-gspo}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-gspo}
export EPS_CLIP=${EPS_CLIP:-1e-4}
export EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-2e-4}
export NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-1}

# Make the "one ratio per TURN" contract real. slime computes GSPO's length-normalized
# sequence ratio per training Sample, so a Sample must equal a single turn. Radix packing
# (lite/train/rollout/core/segmenter.py) packs multiple prefix-extensional turns into one Sample →
# the ratio would be averaged over the whole packed segment (≈ the trajectory for text
# multi-turn), NOT the turn — a different, silently-mistuned gradient. GSPO's ratio is a
# nonlinear (geometric-mean) function of the token grouping, so packing is NOT gradient-
# neutral here (unlike GRPO, which is per-token → packing-invariant and keeps it ON). So
# disable packing for GSPO. Overridable, but changing it changes the GSPO objective.
export CUA_LITE_DISABLE_RADIX=${CUA_LITE_DISABLE_RADIX:-1}

# =============================================================================================== #
#                                    Delegate To GRPO Launcher                                    #
# =============================================================================================== #

exec bash "${SCRIPT_DIR}/run_grpo.sh" "$@"
