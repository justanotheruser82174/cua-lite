#!/usr/bin/env bash

#
# Build Megatron checkpoint args into the caller-visible CKPT_ARGS array.
#
# Shared by run_grpo.sh / run_reinforce.sh / run_dagger.sh / run_sft.sh. The
# SAVE_DIR layout differs by algorithm; SAVE / NO_SAVE_OPTIM / RESUME handling
# stays identical.
#
#   SAVE_DIR=... SAVE_HF_DIR=... SAVE_INTERVAL=...
#   source "${CUA_LITE_ROOT}/scripts/train/utils/ckpt_args.sh"
#   ... python train.py ... "${CKPT_ARGS[@]}"
#
# Inputs: HF_CKPT; SAVE_DIR for saving/resume; SAVE_HF_DIR when SAVE=1;
# SAVE_INTERVAL; SAVE, NO_SAVE_OPTIM, RESUME.
#

CKPT_ARGS=(--hf-checkpoint "${HF_CKPT}")
if [ "${SAVE:-1}" = "1" ]; then
   CKPT_ARGS+=(
      --save "${SAVE_DIR}"
      --save-interval "${SAVE_INTERVAL:-5}"
      --save-hf "${SAVE_HF_DIR}"
   )
   if [ "${NO_SAVE_OPTIM:-1}" = "1" ]; then
      CKPT_ARGS+=(--no-save-optim)
      echo "Saving checkpoints to ${SAVE_DIR} (+ HF to ${SAVE_HF_DIR}) every" \
           "${SAVE_INTERVAL:-5} steps (weights only, no optimizer — not resumable)"
   else
      echo "Saving checkpoints to ${SAVE_DIR} (+ HF to ${SAVE_HF_DIR}) every" \
           "${SAVE_INTERVAL:-5} steps"
   fi
else
   echo "Not saving checkpoints (SAVE=0; default SAVE=1)"
fi
if [ "${RESUME:-0}" = "1" ] && [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
   CKPT_ARGS+=(--load "${SAVE_DIR}")
   echo "Resuming from checkpoint: ${SAVE_DIR}"
elif [ "${RESUME:-0}" = "1" ]; then
   echo "Warning: RESUME=1 but no checkpoint found at ${SAVE_DIR}"
   echo "Starting from --hf-checkpoint"
fi
