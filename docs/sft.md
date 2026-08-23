# SFT Training

Data preprocessing runs on the host (not inside the Slime container) by default. For host uv setup, see [README.md#installation](/README.md#installation).

Training runs inside the Slime container. For container setup, see [docs/slime.md](/docs/slime.md).

---

## SFT from a HuggingFace dataset

Download a published dataset from the Hub (canonical `cua-lite/<Name>` layout),
export a model-ready SFT parquet, then train. Steps 1–2 run on the **host**;
step 3 runs inside the **Slime container** (the repo is mounted at
`/workspaces/cua-lite`).

This section uses [**`cua-lite/ScaleCUA`**](https://huggingface.co/datasets/cua-lite/ScaleCUA), 
the preprocessed CUA corpus with multiple partition cohorts whose row metadata
uses `dims=[platform, task_type]`, including `grounding.action`,
`grounding.point`, `grounding.bbox`, and `use`.

<!-- > **Dataset name note:** `cua-lite/Lite.ScaleCUA` is a different dataset: it is
> a CUA-Lite rollout mirror from `lite.scalecua`, and should be treated as a
> pure `use` trajectory dataset. The examples below intentionally use
> `cua-lite/ScaleCUA` only. -->

### Qwen2.5-VL-3B — grounding.action only

```bash
# --- host ---

# step 1: pull ScaleCUA from HF into the local dataset cache
#   --allow-patterns pulls only the cohorts you'll train on (saves time + disk).
#   Drop it to pull the full dataset.
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
uv run python -m lite.data.hf.download ScaleCUA \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA" \
  --allow-patterns "(desktop|browser|mobile)/grounding.action/**"

# step 2: export model-ready SFT parquet
#   --config = model recipe (adapter + resolution); --data-paths = the downloaded
#   dataset root. Step 1 already limited the local copy to grounding.action cohorts
#   (grounding.action is single-step, so default.yaml's history window is moot here).
#   --model-id = the model you'll TRAIN (step 3 MODEL_ID): steps are tokenized at
#   export, so the chat-template boundary must match the student. Reads the host
#   HF cache.
#   --head 100 for a smoke test; --sample 7000 for the memorise experiment;
#   drop both for the full dataset.
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen2_5_vl/recipes/sft/default.yaml \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --head 100 \
  -o .data/sft/qwen2_5_vl-action/scalecua/train.parquet

# --- Slime container ---

# step 3: train
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen2_5_vl-action/scalecua/train.parquet \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

### Qwen3-VL-2B — ScaleCUA train split, all cohorts

```bash
# --- host ---

# step 1: pull full ScaleCUA from HF
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
uv run python -m lite.data.hf.download ScaleCUA \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA"

# step 2: export
#   point --data-paths at the dataset root; export_sft keeps the train split by
#   default and picks up every CUA cohort under it
#   (grounding.action, grounding.point, grounding.bbox, use, …). Pass
#   --splits train validation only when you intentionally want both splits.
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_vl/recipes/sft/default.yaml \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  -o .data/sft/qwen3_vl/scalecua/train.parquet

# --- Slime container ---

# step 3: train
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_vl/scalecua/train.parquet \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

---

## SFT from local rollout

Collect trajectories from a teacher model **on the host**, export them to an SFT
parquet, then SFT the student **in the Slime container**, and eval base vs SFT
back on the host. `export_sft` accepts both local rollout directories and
downloaded dataset directories. Use direct export for local distillation runs;
use [docs/examples/rollout_to_hf.md](/docs/examples/rollout_to_hf.md) when you
want to publish or share the data.

Example below: **Android-World** distill — the teacher rolls out on the
androidworld **train** split (4 trajectories per task) and the 2B student trains
on the successful ones (Qwen3-VL-8B teacher → Qwen3-VL-2B student). The teacher
collection uses the rollout `default` config; the student export and eval use
the same `compact` config so the trained and evaluated student surface matches.

For per-environment setup, see [docs/envs.md#installation](/docs/envs.md#installation). 
Running data collection through a managed env-server is recommended — see [docs/envs.md#env-server](/docs/envs.md#env-server).

```bash
# --- host ---

# step 1: collect Qwen3-VL-8B teacher trajectories (4 rollouts/task).
#   --group-shared-seed false → distinct init per sample (more SFT diversity).
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct --env-id androidworld \
  --splits train --group-size 4 --concurrency 32 \
  --filter "lambda m: m.others.get('complexity') <= 2.0" \
  --sampling-kwargs '{"temperature": 1.0, "top_p": 1}' \
  --config-path scripts/configs/qwen3_vl/default/androidworld.yaml \
  --log-root .logs/rollout/Qwen_Qwen3-VL-8B-Instruct/androidworld/train_g4 \
  --group-shared-seed false

# step 2: direct local export from successful rollouts.
#   Use this fast path for local distillation data. Use the compact config here
#   because the student will train/eval with it.
#   --image-root . = repo root for local rollout assets.
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_vl/compact/androidworld.yaml \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths .logs/rollout/Qwen_Qwen3-VL-8B-Instruct/androidworld/train_g4 \
  --image-root . \
  --filter "lambda m: (m.others.get('episode_return') or 0) >= 1.0" \
  -o .data/sft/qwen3_vl/androidworld/train.parquet

# For publish/share flows, stage the rollout first; see docs/examples/rollout_to_hf.md.

# --- Slime container ---

# step 3: SFT the Qwen3-VL-2B student. SAVE_HF_DIR lives under the mounted repo so
#   the host eval below can load the ckpt.
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_TRAIN_GPUS=4 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  SAVE=1 NO_SAVE_OPTIM=1 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_vl/androidworld/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_vl-2b/androidworld/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh

# --- host ---

# step 4: eval base vs SFT on the EVAL split (greedy). For the SFT run, add
#   --model-path pointing at the saved iter (replace iter_<N>).
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct --env-id androidworld \
  --splits eval --concurrency 16 \
  --filter "lambda m: m.others.get('complexity') <= 2.0" \
  --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
  --config-path scripts/configs/qwen3_vl/compact/androidworld.yaml \
  --log-root .logs/rollout/Qwen_Qwen3-VL-2B-Instruct/androidworld/eval_base

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct --env-id androidworld \
  --model-path .ckpts/qwen3_vl-2b/androidworld/sft/iter_<N> \
  --splits eval --concurrency 16 \
  --filter "lambda m: m.others.get('complexity') <= 2.0" \
  --sampling-kwargs '{"temperature": 0, "top_p": 1}' \
  --config-path scripts/configs/qwen3_vl/compact/androidworld.yaml \
  --log-root .logs/rollout/Qwen_Qwen3-VL-2B-Instruct/androidworld/eval_sft
```
