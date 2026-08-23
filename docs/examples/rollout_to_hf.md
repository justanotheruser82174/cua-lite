# Push Local Rollout Data to HuggingFace

Publish a local rollout run as a HuggingFace dataset, then train from the
published copy. Use this walkthrough when you want to share or archive a
collected run; the example below uploads a private dataset repo, so consumers
must have Hub access to that repo. If you only need local distillation from a
fresh run, use the direct `export_sft` path in
[docs/sft.md#sft-from-local-rollout](/docs/sft.md#sft-from-local-rollout)
instead.

Pipeline: **collect → stage with an explicit filter → upload → download →
export_sft → train**.
Everything runs on the **host** (from the repo root) except the SFT step (Slime
container). Host setup: [README.md#installation](/README.md#installation); env setup:
[docs/envs.md#installation](/docs/envs.md#installation) / [docs/envs.md#env-server](/docs/envs.md#env-server);
slime: [docs/slime.md](/docs/slime.md).

```bash
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"   # local dataset cache
export HF_ORG="<your-hf-user-or-org>"
export OPENAI_API_KEY="..."                              # collect-side model key
```

## Producer — publish a rollout dataset

```bash
# --- host ---

# 1. collect (GPT-5.5 teacher on the lite.scalecua `rl` split)
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.scalecua \
  --splits rl --sample 64 --seed 0 --concurrency 48 --max-attempts 2 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml --log-root .logs/rollout/scalecua-rl

# 2. stage successful rows as a cua-lite/<Name> dataset.
uv run python -m lite.data.hf.stage \
  --log-roots .logs/rollout/scalecua-rl \
  --filter "lambda m: (m.others.get('episode_return') or 0) > 0.5" \
  --name Lite.ScaleCUA.Test \
  --description "Lite.ScaleCUA rl-split GPT-5.5 desktop trajectories"

# 3. upload a private dataset repo to the Hub
uv run python -m lite.data.hf.upload Lite.ScaleCUA.Test --org "$HF_ORG" --private
```

Drop `--private` to publish a public dataset.

## Consumer — train from the Hub

A consumer with access to the private dataset can run this flow with just the
dataset — the [docs/sft.md#sft-from-a-huggingface-dataset](/docs/sft.md#sft-from-a-huggingface-dataset) flow:

```bash
# --- host ---

# 4. download into the local dataset cache
uv run python -m lite.data.hf.download Lite.ScaleCUA.Test \
  --org "$HF_ORG" --overwrite \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.Test"

# 5. export_sft a model-ready SFT parquet (--image-root = the dir ABOVE cua-lite/)
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.Test" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  -o .data/sft/qwen3_vl/lite.scalecua/train.parquet
```

Train in the Slime container (repo mounted at `/workspaces/cua-lite`, so the host
`.data/sft/…` is `/workspaces/cua-lite/.data/sft/…` there) — see [docs/slime.md](/docs/slime.md):

```bash
# --- Slime container ---
# 6. SFT Qwen3-VL-2B. Small data → overfit settings (more epochs, higher LR), 2 GPUs.
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=10 GLOBAL_BATCH_SIZE=4 LR=1e-5 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_vl/lite.scalecua/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_vl-2b/lite.scalecua/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

Eval base vs SFT on the held-out `lite.osworld` eval split (replace `iter_<N>` with the saved iter):

```bash
# --- host ---
# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --env-id lite.osworld --splits eval --sample 48 --seed 7 --concurrency 8 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  --log-root .logs/rollout/eval_base &

# SFT checkpoint (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct --model-path .ckpts/qwen3_vl-2b/lite.scalecua/sft/iter_<N> \
  --env-id lite.osworld --splits eval --sample 48 --seed 7 --concurrency 8 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  --log-root .logs/rollout/eval_sft &

wait
```

Score each run from `.logs/rollout/<tag>/summary.json` → `stats.mean_episode_return`.

<details>
<summary><b>Advanced: extend a published run</b></summary>

Grow a published run without re-collecting what's already there: `unstage` it to
a local rollout directory, resume `scripts/rollout.py` there, then stage and
upload the combined result.

```bash
# --- host ---

# 8. download + unstage to a local rollout directory. --splits MUST match the
#    split you resume with.
uv run python -m lite.data.hf.download Lite.ScaleCUA.Test \
  --org "$HF_ORG" --overwrite \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.Test"
uv run python -m lite.data.hf.unstage \
  --dataset "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.Test" \
  --log-root .logs/rollout/scalecua-rl-more --splits rl

# 9. collect MORE in the SAME directory — resume skips existing samples.
uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.scalecua \
  --splits rl --sample 128 --seed 0 --concurrency 48 --max-attempts 2 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/recipes/collect/lite.scalecua.yaml --log-root .logs/rollout/scalecua-rl-more

# 10. stage successful rows from the combined run, and upload it again.
#     Use a fresh staging path so the downloaded source dataset remains
#     available while old unstaged rows still reference its image store.
export COMBINED_STAGING="${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.Test.combined"
uv run python -m lite.data.hf.stage \
  --log-roots .logs/rollout/scalecua-rl-more \
  --filter "lambda m: (m.others.get('episode_return') or 0) > 0.5" \
  --name Lite.ScaleCUA.Test \
  --out "$COMBINED_STAGING"
uv run python -m lite.data.hf.upload Lite.ScaleCUA.Test \
  --org "$HF_ORG" --private --staging "$COMBINED_STAGING"
```

</details>
