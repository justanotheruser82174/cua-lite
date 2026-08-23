#!/bin/bash

#
# CUA-Lite offline SFT training on Slime.
#
# Purpose:
#   Train from model-ready SFT parquet produced by export_sft. Unlike the
#   online scripts, this path does not roll out in an environment and does not
#   support eval-before-train.
#
# Required:
#   Run inside the Slime container. Set PROMPT_DATA to an SFT parquet with
#   serialized LiteRLStep records and processed_images metadata.
#
# Common overrides:
#   Inputs:
#     MODEL_ID, HF_CKPT, PROMPT_DATA
#   Capacity:
#     NUM_TRAIN_GPUS, TP_SIZE, GLOBAL_BATCH_SIZE, ROLLOUT_BATCH_SIZE
#   Training:
#     NUM_EPOCH, LR, MIN_LR, MBS, MAX_TOKENS_PER_GPU
#   Output/logging:
#     SAVE, SAVE_DIR, SAVE_HF_DIR, RESUME, NO_SAVE_OPTIM, DUMP, DUMP_DIR,
#     WANDB_API_KEY, WANDB_PROJECT_OVERRIDE, WANDB_GROUP_OVERRIDE,
#     WANDB_GROUP_SUFFIX
#   Advanced:
#     PP_SIZE, CP_SIZE, GRADS_FP32, DIST_OPTIM, OPTIM_CPU_OFFLOAD, RECOMPUTE,
#     RECOMPUTE_METHOD, RECOMPUTE_NUM_LAYERS
#   Debug:
#     DEBUG, CUA_LITE_SHELL_TRACE, CUA_LITE_ROLLOUT_PROC_WORKERS
#   EVAL_PROMPT_DATA is intentionally rejected.
#
# Related launchers:
#   run_grpo.sh, run_reinforce.sh, and run_dagger.sh are the online paths.
#   run_gspo.sh is a thin GSPO wrapper over run_grpo.sh.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_TRAIN_GPUS=4 \
#     MODEL_ID=Qwen/Qwen3-VL-4B-Instruct \
#     PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_vl/scalecua/train.parquet \
#     bash /workspaces/cua-lite/scripts/train/run_sft.sh

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
NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-4}
PROMPT_DATA=${PROMPT_DATA:?"PROMPT_DATA is required"}
EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA:-""}
SESSION_ID=${SESSION_ID:-"manual-$(date +%Y%m%d_%H%M%S)-$$"}

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

# =============================================================================================== #
#                                   Preflight And Local Assets                                    #
# =============================================================================================== #

# -- Hardware probe ------------------------------------------------------------

# Controls NCCL_NVLS_ENABLE in the Ray worker runtime env.
source "${UTILS_DIR}/nvlink.sh"

# -- GPU health preflight ------------------------------------------------------

# Fail before Megatron init if no visible GPU is usable. Respects
# CUDA_VISIBLE_DEVICES, so excluding a bad GPU passes.
if ! python -c "import torch
n = torch.cuda.device_count()
if n < 1:
    raise SystemExit(1)
for i in range(n):
    torch.zeros(1, device='cuda:%d' % i)
    torch.cuda.synchronize(i)" 2>/dev/null; then
   echo "PREFLIGHT FAIL: no visible GPU is usable." >&2
   echo "Reset a wedged GPU, or exclude it via CUDA_VISIBLE_DEVICES." >&2
   exit 1
fi

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

# One canonical run path (model-first, model_slug) → checkpoints + wandb share it.
# SFT trains on a model-ready parquet (not an env): label it by PROMPT_DATA's parent dir.
DATA_SLUG="$(basename "$(dirname "${PROMPT_DATA}")")"
RUN_REL="${MODEL_SLUG}/sft/${DATA_SLUG}"
SAVE_DIR=${SAVE_DIR:-"/root/checkpoints/${RUN_REL}/megatron"}
SAVE_HF_DIR=${SAVE_HF_DIR:-"/root/checkpoints/${RUN_REL}/hf/iter_{rollout_id}"}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
# Build CKPT_ARGS from SAVE_DIR/SAVE_HF_DIR/HF_CKPT + SAVE/NO_SAVE_OPTIM/RESUME (shared helper).
source "${UTILS_DIR}/ckpt_args.sh"

# -- Detail dumps --------------------------------------------------------------

DUMP_DIR=${DUMP_DIR:-"/tmp/cua-lite/sft/${MODEL_SLUG}/$(date +%Y%m%d_%H%M%S)"}
if [ "${DUMP:-0}" = "1" ]; then
   DUMP_ARGS=(--dump-details "${DUMP_DIR}")
   echo "Dumping details to ${DUMP_DIR}"
else
   DUMP_ARGS=()
   echo "Not dumping details (set DUMP=1 to enable)"
fi

# =============================================================================================== #
#                                   SFT Data And Loss Arguments                                   #
# =============================================================================================== #

# -- Batch shape ---------------------------------------------------------------

# Offline SFT has one meaningful batch knob: GLOBAL_BATCH_SIZE.
# GLOBAL_BATCH_SIZE = trajectories (GROUPS) per optimizer step — the ONLY batch knob that
# affects SFT training (total steps = dataset*num_epoch/GBS; slime counts GROUPS, each
# group's avg_turns segments come along). To use GBS=2: just set GLOBAL_BATCH_SIZE=2.
#
# ROLLOUT_BATCH_SIZE is NOT a training knob for SFT — it's only slime's data-FETCH
# granularity (slime reuses the RL loop "fetch rollout_batch_size, take num_steps steps,
# repeat"; in RL that boundary is the on-policy weight-sync, but SFT has no rollout engine).
# So it defaults to GLOBAL_BATCH_SIZE -> num_steps_per_rollout=1 -> "fetch one batch, do one
# step, repeat" (a single batch size, as you'd expect). Raise it (a multiple of GBS) ONLY to
# amortize data-loading I/O across several steps — it does NOT change the training math.
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}
_require_positive_int "GLOBAL_BATCH_SIZE" "${GLOBAL_BATCH_SIZE}"
_require_positive_int "ROLLOUT_BATCH_SIZE" "${ROLLOUT_BATCH_SIZE}"
# Preflight: GBS must divide the SFT fetch batch within [1, ROLLOUT_BATCH_SIZE]
# else slime floor-drops trailing GROUPS silently or hits a bare ZeroDivisionError.
_N_TRAJ=${ROLLOUT_BATCH_SIZE}
if [ "${GLOBAL_BATCH_SIZE}" -lt 1 ] || [ "${GLOBAL_BATCH_SIZE}" -gt "${_N_TRAJ}" ] \
   || [ $((_N_TRAJ % GLOBAL_BATCH_SIZE)) -ne 0 ]; then
   echo "Error: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be a divisor of" \
        "ROLLOUT_BATCH_SIZE=${_N_TRAJ} (and within [1, ${_N_TRAJ}])." >&2
   exit 1
fi

# -- SFT data path -------------------------------------------------------------

# No --apply-chat-template / --multimodal-keys / --tool-key:
# lite.train.rollout.sft handles all data processing directly via processor.
# One row per trajectory.
#   --input-key steps: slime passes the `steps` column (list of serialized
#       LiteRLStep structs, pre-tokenized by export_sft) as sample.prompt;
#       lite.train.rollout.sft deserializes them and fans out one slime.Sample per segment.
#       Each step's image_indices order is positional: the Nth index supplies the
#       Nth processor-owned image slot in the stored prompt.
#   --metadata-key processed_images: PNG bytes (one entry per physical
#       trajectory image, deduped from the per-step view that v1 stored).
SFT_ARGS=(
   --rollout-function-path lite.train.rollout.sft.generate_rollout
   --prompt-data "${PROMPT_DATA}"
   --input-key steps
   --metadata-key processed_images
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH:-2}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   # GLOBAL_BATCH_SIZE counts GROUPS (trajectories); slime groups each trajectory's
   # segments by their shared index. SFT sets this DIRECTLY (offline — see Batch above).
   # (--use-dynamic-global-batch-size was removed in v0.3.0.)
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   # Disable RL components
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
   --multimodal-lazy-expand-fn-path lite.train.utils.multimodal_expand.expand
)

# -- Eval guard ----------------------------------------------------------------

# slime's eval falls through to eval_function_path = rollout_function_path =
# lite.train.rollout.sft.generate_rollout, which asserts `not evaluation`
# (SFT reads a fixed parquet — there is nothing to roll out). With v0.3.0's
# eval-before-train in train_async.py this would crash at rollout 0, so fail
# fast here with a clear message instead.
if [ -n "$EVAL_PROMPT_DATA" ]; then
   echo "Error: EVAL_PROMPT_DATA is not supported by run_sft.sh —" \
        "lite.train.rollout.sft.generate_rollout has no evaluation mode." >&2
   exit 1
fi
EVAL_ARGS=()

# =============================================================================================== #
#                                       Training Arguments                                        #
# =============================================================================================== #

# -- Optimizer -----------------------------------------------------------------

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-2e-6}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR:-1e-6}"
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
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

# =============================================================================================== #
#                                        Megatron Backend                                         #
# =============================================================================================== #

# CUA-Lite screenshots are large (1920x1080) → many vision tokens.
# TP: default 1 (pure DP + ZeRO-1); raise TP_SIZE explicitly on OOM. PP/CP
# remain explicit advanced knobs for operators who need pipeline/context split.
TP_SIZE=$(resolve_tp "${TP_SIZE:-}" "${NUM_TRAIN_GPUS}")
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
_require_positive_int "PP_SIZE" "${PP_SIZE}"
_require_positive_int "CP_SIZE" "${CP_SIZE}"
MODEL_PARALLEL_SIZE=$(( TP_SIZE * PP_SIZE * CP_SIZE ))
if [ "$(( NUM_TRAIN_GPUS % MODEL_PARALLEL_SIZE ))" -ne 0 ]; then
   echo "Error: TP_SIZE*PP_SIZE*CP_SIZE=${MODEL_PARALLEL_SIZE} must divide" \
        "NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS}." >&2
   exit 1
fi
DP_SIZE=$(( NUM_TRAIN_GPUS / MODEL_PARALLEL_SIZE ))
printf 'Resolved parallelism: TP=%s PP=%s CP=%s EP=1 DP=%s\n' \
   "${TP_SIZE}" "${PP_SIZE}" "${CP_SIZE}" "${DP_SIZE}"
printf '  train_gpus=%s model_family=%s\n' \
   "${NUM_TRAIN_GPUS}" "${MODEL_FAMILY}"

BACKEND_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size "${TP_SIZE}"
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
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

# fp32 grad accumulation is on by default. Set GRADS_FP32=0 for bf16 grads to
# save optimizer memory on constrained GPUs.
if [ "${GRADS_FP32:-1}" = "1" ]; then
   BACKEND_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi

# Batching / attention format. DEFAULT = BSHD + fixed micro-batch (MBS) for ALL
# families: predictable per-GPU memory and the honest OOM knob for image-heavy
# VLM — max-tokens can't go below one sample and one multi-image sample is already
# ~2k tokens, so tuning it to relieve OOM is a fragile dead-end. Opt into
# THD/dynamic packing ONLY by explicitly setting MAX_TOKENS_PER_GPU. NOT allowed
# for Qwen3.5: its GatedDeltaNet (GDN) kernel accumulates per-sequence recurrent
# state and THD packs sequences end-to-end → corrupted state boundaries / wrong
# gradients (megatron-core hard-raises NotImplementedError on packed GDN).
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

NUM_GPUS=${NUM_TRAIN_GPUS}
source "${UTILS_DIR}/ray.sh"
source "${UTILS_DIR}/runtime_env.sh"
start_ray

# -- Worker runtime env --------------------------------------------------------

# Segmenter knobs must be forwarded explicitly: Ray workers do not inherit the
# driver shell, and SFT uses the same multimodal segmenter as rollout/GRPO.
RUNTIME_ENV_JSON="$(build_runtime_env_json \
  "PYTHONPATH=/root/Megatron-LM/:${CUA_LITE_ROOT}:${SLIME_DIR}" \
  "CUDA_DEVICE_MAX_CONNECTIONS=1" \
  "NCCL_NVLS_ENABLE=${HAS_NVLINK}" \
  "MEGATRON_CONFIG_LOCK_DIR=${MEGATRON_CONFIG_LOCK_DIR:-/tmp}" \
  "SESSION_ID=${SESSION_ID}" \
  "CUA_LITE_DISABLE_RADIX=${CUA_LITE_DISABLE_RADIX:-0}" \
  "CUA_LITE_MULTIMODAL_LAZY_EXPAND=${CUA_LITE_MULTIMODAL_LAZY_EXPAND:-0}" \
  "CUA_LITE_MULTIMODAL_FP32=${CUA_LITE_MULTIMODAL_FP32:-0}" \
  "CUA_LITE_ROLLOUT_PROC_WORKERS=${CUA_LITE_ROLLOUT_PROC_WORKERS:-8}" \
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
)"

# -- Submit --------------------------------------------------------------------

printf '\n'
printf 'Launching CUA-Lite SFT\n'
printf '  model=%s data=%s\n' "${MODEL_ID}" "${PROMPT_DATA}"
printf '  batch=global:%s fetch:%s epochs:%s\n' \
   "${GLOBAL_BATCH_SIZE}" "${ROLLOUT_BATCH_SIZE}" "${NUM_EPOCH:-2}"
printf '  checkpoints=%s\n' "${SAVE_DIR}"
printf '  ray_dashboard=http://127.0.0.1:%s\n' "${RAY_PORT}"
printf '\n'

ray job submit --address="http://127.0.0.1:${RAY_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_DIR}/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_TRAIN_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" \
   "${DUMP_ARGS[@]}"
