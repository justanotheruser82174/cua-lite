#!/bin/bash

#
# CUA-Lite REINFORCE / filtered-BC training on Slime.
#
# Purpose:
#   Run online filtered behavior cloning: roll out the current policy, keep
#   successful trajectories, and train with SFT loss on those filtered samples.
#
# Required:
#   Run inside the Slime container. Set ENV_ID and PROMPT_DATA. For
#   server-backed envs, export CUA_LITE_ENV_SERVER_URL and
#   CUA_LITE_ENV_SERVER_TOKEN before launching.
#
# Common overrides:
#   Inputs:
#     MODEL_ID, HF_CKPT, ENV_ID, PROMPT_DATA, EVAL_PROMPT_DATA, CONFIG_PATH
#   Capacity:
#     ASYNC, NUM_TRAIN_GPUS, NUM_ROLLOUT_GPUS, GPUS_PER_ENGINE, ENV_CONCURRENCY
#   Training:
#     ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, NUM_STEPS_PER_ROLLOUT, LR,
#     TP_SIZE, MBS, MAX_TOKENS_PER_GPU
#   Output/logging:
#     SAVE, SAVE_DIR, SAVE_HF_DIR, DUMP, DUMP_DIR, WANDB_API_KEY
#   Advanced:
#     ROLLOUT_MODULE, MEM_FRACTION, GRADS_FP32, OPTIM_CPU_OFFLOAD
#   Debug:
#     DEBUG, CUA_LITE_SHELL_TRACE, CUA_LITE_ROLLOUT_PROC_WORKERS,
#     ROLLOUT_STALL_TIMEOUT_S
#   REINFORCE has no PPO clip; set NUM_STEPS_PER_ROLLOUT=1 for a strictly
#   on-policy gradient.
#
# Related launchers:
#   run_grpo.sh is the clipped-policy-gradient path.
#   run_dagger.sh shares the online SFT loss path but uses teacher targets.
#   run_sft.sh is the offline SFT path.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_TRAIN_GPUS=4 \
#     ENV_ID=lite.osworld \
#     PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/train.parquet \
#     CUA_LITE_ENV_SERVER_URL=http://<env-server-host>:<port> \
#     CUA_LITE_ENV_SERVER_TOKEN=<token> \
#     bash /workspaces/cua-lite/scripts/train/run_reinforce.sh

# =============================================================================================== #
#                                            Bootstrap                                            #
# =============================================================================================== #

# Abort on error. Shell tracing is a separate opt-in because xtrace expands
# runtime-env JSON and can expose env-server/W&B credentials in logs.
set -e
if [ "${CUA_LITE_SHELL_TRACE:-0}" = "1" ]; then
   echo "Warning: CUA_LITE_SHELL_TRACE=1 may print credentials in shell logs." >&2
   set -x
elif [ "${DEBUG:-0}" = "1" ]; then
   echo "DEBUG=1 is kept for downstream code; use CUA_LITE_SHELL_TRACE=1 for shell xtrace." >&2
fi
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CUA_LITE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
UTILS_DIR="${CUA_LITE_ROOT}/scripts/train/utils"

source "${UTILS_DIR}/cleanup.sh"

# =============================================================================================== #
#                                 Resolve Inputs And Run Identity                                 #
# =============================================================================================== #

# -- Operator inputs -----------------------------------------------------------

MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-VL-4B-Instruct"}
ENV_ID=${ENV_ID:?"ENV_ID is required"}  # used for wandb group name and eval dataset label
PROMPT_DATA=${PROMPT_DATA:?"PROMPT_DATA is required"}
EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA:-""}
SESSION_ID=${SESSION_ID:-"manual-$(date +%Y%m%d_%H%M%S)-$$"}

# -- GPU topology --------------------------------------------------------------

NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-4}
NUM_ROLLOUT_GPUS=${NUM_ROLLOUT_GPUS:-${NUM_TRAIN_GPUS}}
ASYNC=${ASYNC:-0}
if [ "${ASYNC}" = "1" ]; then
   SYNC_MODE=async
   NUM_GPUS=$(( NUM_TRAIN_GPUS + NUM_ROLLOUT_GPUS ))
else
   SYNC_MODE=sync
   if [ "${NUM_ROLLOUT_GPUS}" != "${NUM_TRAIN_GPUS}" ]; then
      echo "Error: sync/colocate mode requires NUM_ROLLOUT_GPUS=NUM_TRAIN_GPUS." >&2
      echo "       Set ASYNC=1 for separate rollout GPUs." >&2
      exit 1
   fi
   NUM_GPUS=${NUM_TRAIN_GPUS}
fi
if [ "${USE_TIS:-0}" != "0" ]; then
   echo "Error: USE_TIS has been retired from this launcher; do not pass it." >&2
   exit 1
fi
if [ -n "${CUA_LITE_ENV_SERVER_URL:-}" ] && [ -z "${CUA_LITE_ENV_SERVER_TOKEN:-}" ]; then
   echo "Error: CUA_LITE_ENV_SERVER_TOKEN is required when CUA_LITE_ENV_SERVER_URL is set." >&2
   exit 1
fi

# -- Model family and run identity --------------------------------------------

# Sets MODEL_ARGS_FILE / MODEL_FAMILY and shared launch-shape helpers.
source "${UTILS_DIR}/models.sh"

_require_positive_int() {
   local name=$1 value=$2
   if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: ${name}=${value} must be a positive integer." >&2
      exit 1
   fi
}

DEFAULT_CONFIG_PATH="${CUA_LITE_ROOT}/scripts/configs/${MODEL_FAMILY}/compact/${ENV_ID}.yaml"
CONFIG_PATH=${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}
if [ ! -f "${CONFIG_PATH}" ]; then
   echo "Error: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
   echo "       Set CONFIG_PATH explicitly, or choose an ENV_ID with a compact training config." >&2
   exit 1
fi

# =============================================================================================== #
#                                   Preflight And Local Assets                                    #
# =============================================================================================== #

# -- Env-server preflight ------------------------------------------------------

# Probe ENV_ID availability and clean prior-session leftovers before starting
# Ray. Full fail-fast rationale lives in scripts/train/utils/preflight.sh.
source "${UTILS_DIR}/preflight.sh"

# -- Hardware probe ------------------------------------------------------------

# Controls NCCL_NVLS_ENABLE in the Ray worker runtime env.
source "${UTILS_DIR}/nvlink.sh"

# -- Model checkpoint ----------------------------------------------------------

# HF_CKPT override skips download (e.g. feed an SFT'd ckpt back into RL).
HF_CKPT=${HF_CKPT:-"/root/models/${MODEL_ID}"}
mkdir -p /root/models
if [ "${HF_CKPT}" = "/root/models/${MODEL_ID}" ] && [ ! -d "/root/models/${MODEL_ID}" ]; then
   hf download "${MODEL_ID}" --local-dir "/root/models/${MODEL_ID}"
fi

# =============================================================================================== #
#                                 Persistence And Debug Artifacts                                 #
# =============================================================================================== #

# -- Checkpoints ---------------------------------------------------------------

# ALGO labels the checkpoint dirs + wandb group. One canonical run path
# (model-first, model_slug) → checkpoints + wandb share it.
ALGO=${ALGO:-reinforce}
RUN_REL="${MODEL_SLUG}/${ENV_ID}/${ALGO}/${SYNC_MODE}"
SAVE_DIR=${SAVE_DIR:-"/root/checkpoints/${RUN_REL}/megatron"}
SAVE_HF_DIR=${SAVE_HF_DIR:-"/root/checkpoints/${RUN_REL}/hf/iter_{rollout_id}"}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
# Build CKPT_ARGS from SAVE_DIR/SAVE_HF_DIR/HF_CKPT + SAVE/NO_SAVE_OPTIM/RESUME (shared helper).
source "${UTILS_DIR}/ckpt_args.sh"

# -- Rollout dumps -------------------------------------------------------------

DUMP_DIR=${DUMP_DIR:-"/tmp/cua-lite/${ALGO}/${MODEL_SLUG}/${ENV_ID}/$(date +%Y%m%d_%H%M%S)"}
if [ "${DUMP:-0}" = "1" ]; then
   DUMP_ARGS=(--dump-details "${DUMP_DIR}")
   echo "Dumping rollout details to ${DUMP_DIR}"
else
   DUMP_ARGS=()
   echo "Not dumping rollout details (set DUMP=1 to enable)"
fi

# =============================================================================================== #
#                                     Rollout And Evaluation                                      #
# =============================================================================================== #

# -- Rollout batch shape -------------------------------------------------------

# REINFORCE: n_samples_per_prompt=1 (no group normalization needed).
# ROLLOUT_BATCH_SIZE should be large to compensate for filtering.
# Effective training batch ≈ ROLLOUT_BATCH_SIZE * nonzero_return_rate.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
_require_positive_int "ROLLOUT_BATCH_SIZE" "${ROLLOUT_BATCH_SIZE}"
_require_positive_int "N_SAMPLES_PER_PROMPT" "${N_SAMPLES_PER_PROMPT:-1}"

# Short agentic turns are usually ~100 tokens; this caps runaway loops that
# OOM logits. Raise for long-CoT agents.
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512}

# 1.0 follows Slime convention. Lower for precise multi-turn agents where
# per-step error compounds.
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}

# slime v0.3.0 derives GBS from --num-steps-per-rollout; do NOT pass
# --global-batch-size. cua-lite pads filtered trajectories back to the full
# rollout count so num_groups is an exact multiple of GBS. Default 8 for minibatch
# granularity; steps 2..k are off-policy & UNcorrected (no clip) — set k=1 for strict
# on-policy. GBS = rbs*nsp // 8 GROUPS per optimizer step.
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-8}
_require_positive_int "NUM_STEPS_PER_ROLLOUT" "${NUM_STEPS_PER_ROLLOUT}"
# Preflight (mirrors run_grpo.sh): k must be a divisor of rbs*nsp within
# [1, rbs*nsp] — otherwise slime floor-drops trailing groups silently
# (non-divisor) or hits a bare ZeroDivisionError (k too large).
_N_TRAJ=$((ROLLOUT_BATCH_SIZE * ${N_SAMPLES_PER_PROMPT:-1}))
if [ "${NUM_STEPS_PER_ROLLOUT}" -lt 1 ] || [ "${NUM_STEPS_PER_ROLLOUT}" -gt "${_N_TRAJ}" ] \
   || [ $((_N_TRAJ % NUM_STEPS_PER_ROLLOUT)) -ne 0 ]; then
   echo "Error: NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT} must be a divisor of" \
        "ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT=${_N_TRAJ} (and within [1, ${_N_TRAJ}])." >&2
   exit 1
fi

# -- Rollout entrypoints -------------------------------------------------------

ROLLOUT_MODULE=${ROLLOUT_MODULE:-lite.train.rollout.reinforce}
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key problem
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-1000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-1}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --custom-generate-function-path "${ROLLOUT_MODULE}.generate"
   --custom-convert-samples-to-train-data-path "${ROLLOUT_MODULE}.convert_samples_to_train_data"
   --rollout-function-path "${ROLLOUT_MODULE}.generate_rollout"
   --custom-config-path "${CONFIG_PATH}"
   --multimodal-lazy-expand-fn-path lite.train.utils.multimodal_expand.expand
)

# -- Evaluation ----------------------------------------------------------------

# RL defaults SKIP_EVAL_BEFORE_TRAIN=0 (don't skip) — baseline is a meaningful
# reference for the reward trajectory. SFT defaults to 1 (skip).
if [ -n "$EVAL_PROMPT_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL:-5}"
      --eval-prompt-data "${ENV_ID}_eval" "${EVAL_PROMPT_DATA}"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
      --eval-temperature 0
      --eval-top-p 1
   )
   if [ "${SKIP_EVAL_BEFORE_TRAIN:-0}" != "0" ]; then
      EVAL_ARGS+=(--skip-eval-before-train)
   fi
else
   EVAL_ARGS=()
fi

# =============================================================================================== #
#                                       Training Arguments                                        #
# =============================================================================================== #

# -- REINFORCE objective -------------------------------------------------------

# REINFORCE uses SFT loss on filtered successful trajectories.
REINFORCE_ARGS=(
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
)

# -- Optimizer -----------------------------------------------------------------

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# OPTIM_CPU_OFFLOAD=1 moves the fp32 Adam state to host RAM. Use it when
# distributed optimizer sharding is insufficient on constrained GPUs.
if [ "${OPTIM_CPU_OFFLOAD:-0}" = "1" ]; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload --use-precision-aware-optimizer)
fi

# DIST_OPTIM=1 shards fp32 Adam state + master params across DP ranks by default.
if [ "${DIST_OPTIM:-1}" = "1" ]; then
   OPTIMIZER_ARGS+=(--use-distributed-optimizer)
fi

# -- Rollout inference ---------------------------------------------------------

GPUS_PER_ENGINE=${GPUS_PER_ENGINE:-1}
NUM_ENGINES=$(resolve_num_engines "${GPUS_PER_ENGINE}" "${NUM_ROLLOUT_GPUS}")
ENV_CONCURRENCY=${ENV_CONCURRENCY:-32}
# Per-engine server concurrency is ceil(ENV_CONCURRENCY / NUM_ENGINES).
SERVER_CONCURRENCY=$(( (ENV_CONCURRENCY + NUM_ENGINES - 1) / NUM_ENGINES ))
SGLANG_CUDA_GRAPH_BS=(
   1 2 4 8
   16 24 32 40 48 56 64 72
   80 88 96 104 112 120 128
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${MEM_FRACTION:-0.6}"
   --sglang-cuda-graph-bs "${SGLANG_CUDA_GRAPH_BS[@]}"
   --sglang-server-concurrency "${SERVER_CONCURRENCY}"
   --router-policy "${ROUTER_POLICY:-round_robin}"
)

# -- W&B -----------------------------------------------------------------------

if [ -n "${WANDB_API_KEY:-}" ]; then
   WANDB_PROJECT=${WANDB_PROJECT_OVERRIDE:-cua-lite-dev}
   WANDB_GROUP=${WANDB_GROUP_OVERRIDE:-${RUN_REL}${WANDB_GROUP_SUFFIX:-}}
   WANDB_ARGS=(
      --use-wandb
      # dev bucket; WANDB_PROJECT_OVERRIDE=cua-lite for curated/public runs
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP}"
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

# -- Slime resources -----------------------------------------------------------

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "${NUM_TRAIN_GPUS}"
   --rollout-num-gpus "${NUM_ROLLOUT_GPUS}"
)

if [ "${ASYNC}" != "1" ]; then
   MISC_ARGS+=(--colocate)
fi

# =============================================================================================== #
#                                        Megatron Backend                                         #
# =============================================================================================== #

# TP: default 1 (pure DP + ZeRO-1); raise TP_SIZE explicitly on OOM. resolve_tp
# defaults + validates divisibility (models.sh).
TP_SIZE=$(resolve_tp "${TP_SIZE:-}" "${NUM_TRAIN_GPUS}")
DP_SIZE=$(( NUM_TRAIN_GPUS / TP_SIZE ))
printf 'Resolved parallelism: TP=%s PP=1 CP=1 EP=1 DP=%s\n' \
   "${TP_SIZE}" "${DP_SIZE}"
printf '  train_gpus=%s model_family=%s\n' \
   "${NUM_TRAIN_GPUS}" "${MODEL_FAMILY}"

BACKEND_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size "${TP_SIZE}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

if [ "${TP_SIZE}" -gt 1 ]; then
   BACKEND_ARGS+=(--sequence-parallel)
fi

if [ "${RECOMPUTE:-1}" = "1" ]; then
   BACKEND_ARGS+=(
      --recompute-granularity full
      --recompute-method "${RECOMPUTE_METHOD:-uniform}"
      --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
   )
fi

if [ "${GRADS_FP32:-1}" = "1" ]; then
   BACKEND_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi

# Batching / attention format (see run_grpo.sh for the full rationale).
# DEFAULT = BSHD + fixed micro-batch (MBS) for ALL families (predictable per-GPU
# memory, the honest OOM knob for image-heavy VLM). Opt into THD/dynamic packing
# ONLY by setting MAX_TOKENS_PER_GPU — barred for Qwen3.5 (GDN can't pack: THD
# corrupts the recurrent state boundaries / megatron-core NotImplementedError).
if [ -n "${MAX_TOKENS_PER_GPU:-}" ]; then
   if [ "${MODEL_FAMILY}" = "qwen3_5" ]; then
      echo "ERROR: Qwen3.5 (GDN) cannot use THD packing — unset MAX_TOKENS_PER_GPU and use MBS (BSHD)." >&2
      exit 1
   fi
   BACKEND_ARGS+=(
      --use-dynamic-batch-size
      --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   )
else
   MBS=${MBS:-1}
   BACKEND_ARGS+=(--qkv-format bshd --micro-batch-size "${MBS}")
fi

# =============================================================================================== #
#                               Launch: Ray Runtime And Job Submit                                #
# =============================================================================================== #

# -- Model arguments -----------------------------------------------------------

SLIME_DIR="${CUA_LITE_ROOT}/slime"
source "${CUA_LITE_ROOT}/scripts/train/models/${MODEL_ARGS_FILE}.sh"

# -- Ray startup ---------------------------------------------------------------

source "${UTILS_DIR}/ray.sh"
source "${UTILS_DIR}/runtime_env.sh"
start_ray

# -- Worker runtime env --------------------------------------------------------

# CUA_LITE_ENV_SERVER_{URL,TOKEN}: LiteEnvClient reads these to reach the
# env server during rollout (see docs/slime.md + docs/envs.md).
# CUA_LITE_REGISTRATION_MODULES registers out-of-tree direct-mode envs in Ray
# workers; the driver/preflight shell is not enough because workers do not
# inherit arbitrary launcher env.
# Segmenter/rollout knobs must be forwarded explicitly: Ray workers do not
# inherit the driver shell.
RUNTIME_ENV_JSON="$(build_runtime_env_json \
  "PYTHONPATH=/root/Megatron-LM/:${CUA_LITE_ROOT}:${SLIME_DIR}" \
  "CUDA_DEVICE_MAX_CONNECTIONS=1" \
  "NCCL_NVLS_ENABLE=${HAS_NVLINK}" \
  "MEGATRON_CONFIG_LOCK_DIR=${MEGATRON_CONFIG_LOCK_DIR:-/tmp}" \
  "SESSION_ID=${SESSION_ID}" \
  "CUA_LITE_ENV_SERVER_URL=${CUA_LITE_ENV_SERVER_URL}" \
  "CUA_LITE_ENV_SERVER_TOKEN=${CUA_LITE_ENV_SERVER_TOKEN}" \
  "CUA_LITE_REGISTRATION_MODULES=${CUA_LITE_REGISTRATION_MODULES:-}" \
  "CUA_LITE_DISABLE_RADIX=${CUA_LITE_DISABLE_RADIX:-0}" \
  "CUA_LITE_MULTIMODAL_LAZY_EXPAND=${CUA_LITE_MULTIMODAL_LAZY_EXPAND:-0}" \
  "CUA_LITE_MULTIMODAL_FP32=${CUA_LITE_MULTIMODAL_FP32:-0}" \
  "CUA_LITE_ROLLOUT_PROC_WORKERS=${CUA_LITE_ROLLOUT_PROC_WORKERS:-8}" \
  "ROLLOUT_STALL_TIMEOUT_S=${ROLLOUT_STALL_TIMEOUT_S:-1800}" \
)"

# -- Submit --------------------------------------------------------------------

# ASYNC=1 uses train_async.py (separate rollout/train GPUs); else train.py (colocate).
TRAIN_ENTRYPOINT="${SLIME_DIR}/train.py"
if [ "${ASYNC}" = "1" ]; then
   TRAIN_ENTRYPOINT="${SLIME_DIR}/train_async.py"
fi

printf '\n'
printf 'Launching CUA-Lite REINFORCE\n'
printf '  model=%s env=%s mode=%s\n' "${MODEL_ID}" "${ENV_ID}" "${SYNC_MODE}"
printf '  rollout=batch:%s samples:%s steps:%s module:%s\n' \
   "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT:-1}" \
   "${NUM_STEPS_PER_ROLLOUT}" "${ROLLOUT_MODULE}"
printf '  config=%s\n' "${CONFIG_PATH}"
printf '  checkpoints=%s\n' "${SAVE_DIR}"
printf '  ray_dashboard=http://127.0.0.1:%s\n' "${RAY_PORT}"
printf '\n'

ray job submit --address="http://127.0.0.1:${RAY_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${TRAIN_ENTRYPOINT}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${REINFORCE_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DUMP_ARGS[@]}"
