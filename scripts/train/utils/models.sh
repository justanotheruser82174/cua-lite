# Shared model contract and launch-shape helpers for train launch scripts.
#
# Sourced by run_grpo.sh / run_reinforce.sh / run_dagger.sh / run_sft.sh
# after MODEL_ID is set:
#
#   source "${CUA_LITE_ROOT}/scripts/train/utils/models.sh"
#
# Provides:
#   MODEL_ARGS_FILE      — model-args template name under scripts/train/models/
#   MODEL_FAMILY         — config-family directory: qwen3_vl or qwen3_5
#   MODEL_SLUG           — model id with "/" replaced by "_" for output paths
#   resolve_tp REQ N       — echoes the TP to use: REQ if given (must divide N),
#                            else the default TP=1
#   resolve_num_engines G N — echoes floor(N/G) after guarding that G and N are
#                            positive and at least one rollout engine can start
#
# Fails fast (exit 1, which exits the sourcing script) on unknown MODEL_ID.

# Map MODEL_ID → model-args file under scripts/train/models/.
# MAI-UI checkpoints share Qwen3-VL backbone templates.
declare -A MODEL_ARGS_MAP=(
  [Qwen/Qwen3-VL-2B-Instruct]=Qwen3-VL-2B
  [Qwen/Qwen3-VL-4B-Instruct]=Qwen3-VL-4B
  [Qwen/Qwen3-VL-8B-Instruct]=Qwen3-VL-8B
  [Qwen/Qwen3-VL-2B-Thinking]=Qwen3-VL-2B
  [Qwen/Qwen3-VL-4B-Thinking]=Qwen3-VL-4B
  [Qwen/Qwen3-VL-8B-Thinking]=Qwen3-VL-8B
  [Qwen/Qwen2.5-VL-3B-Instruct]=Qwen2.5-VL-3B
  [Tongyi-MAI/MAI-UI-2B]=Qwen3-VL-2B
  [Tongyi-MAI/MAI-UI-8B]=Qwen3-VL-8B
  [Qwen/Qwen3.5-2B]=Qwen3.5-2B
  [Qwen/Qwen3.5-2B-Base]=Qwen3.5-2B
  [Qwen/Qwen3.5-4B]=Qwen3.5-4B
  [Qwen/Qwen3.5-4B-Base]=Qwen3.5-4B
  [Qwen/Qwen3.5-9B]=Qwen3.5-9B
  [Qwen/Qwen3.5-9B-Base]=Qwen3.5-9B
  [Qwen/Qwen3.5-27B]=Qwen3.5-27B
)
MODEL_ARGS_FILE="${MODEL_ARGS_MAP[$MODEL_ID]:-}"
if [ -z "$MODEL_ARGS_FILE" ]; then
   echo "Error: MODEL_ID must be one of: ${!MODEL_ARGS_MAP[*]}" >&2
   exit 1
fi

# Run slug for output paths + wandb: model_id "/" -> "_" (matches
# lite.infer.rollout's model_slug), so a run's wandb name == its rollout log
# path == its checkpoint sub-path. Launchers build RUN_REL="${MODEL_SLUG}/<recipe>"
# (model-first) and feed it to SAVE_DIR + --wandb-group.
MODEL_SLUG="${MODEL_ID//\//_}"

# Model family → config directory. Backend and packing defaults are owned by
# the launchers and model-args files, not this helper.
case "$MODEL_ARGS_FILE" in
  Qwen3.5-*)               MODEL_FAMILY=qwen3_5 ;;
  Qwen3-VL-*|Qwen2.5-VL-*) MODEL_FAMILY=qwen3_vl ;;
  *) echo "Error: cannot derive MODEL_FAMILY from ${MODEL_ARGS_FILE}" >&2; exit 1 ;;
esac

# TP defaults live next to the model family mapping. TP=1 keeps pure DP fast;
# set TP_SIZE only as a memory valve.

# resolve_tp REQ N  →  echoes the TP to use (REQ = requested TP or "", N = NUM_TRAIN_GPUS).
#
# Requested TP_SIZE must divide NUM_TRAIN_GPUS.
#
_models_require_positive_int() {
   local name=$1 value=$2
   if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "Error: ${name}=${value} must be a positive integer." >&2
      exit 1
   fi
}


# Fails fast if REQ does not divide N (DP = N/TP must be an integer).
resolve_tp() {
   local n=$2 tp=${1:-1}
   _models_require_positive_int "TP_SIZE" "$tp"
   _models_require_positive_int "NUM_TRAIN_GPUS" "$n"
   if [ "$(( n % tp ))" -ne 0 ]; then
      echo "Error: TP_SIZE=$tp does not divide NUM_TRAIN_GPUS=$n (DP=N/TP must be an integer)." >&2
      exit 1
   fi
   echo "${tp}"
}


resolve_num_engines() {
   local gpus_per_engine=${1:-1} num_rollout_gpus=$2
   _models_require_positive_int "GPUS_PER_ENGINE" "$gpus_per_engine"
   _models_require_positive_int "NUM_ROLLOUT_GPUS" "$num_rollout_gpus"
   if [ "$num_rollout_gpus" -lt "$gpus_per_engine" ]; then
      echo "Error: GPUS_PER_ENGINE=$gpus_per_engine exceeds" \
           "NUM_ROLLOUT_GPUS=$num_rollout_gpus." >&2
      echo "       That would start zero rollout engines." >&2
      exit 1
   fi
   echo "$(( num_rollout_gpus / gpus_per_engine ))"
}
