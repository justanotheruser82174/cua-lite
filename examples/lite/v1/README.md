# Lite v1 — desktop.use consolidated training walkthrough

<!-- Data prep runs on the host; training runs inside the Slime container (repo mounted
at `/workspaces/cua-lite`). Host setup: [README.md#installation](/README.md#installation).
Container: [docs/slime.md](/docs/slime.md). Per-env setup: [docs/envs.md#installation](/docs/envs.md#installation).

## WebGym

This is an example of training Qwen3.5-4B on WebGym (adaptive reasoning, see [webgym.yaml](/examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml)).

### SFT

Supervised fine-tuning Qwen3.5-4B on WebGym trajectories rolled out by GPT-5.5.

#### Export

Export runs on the host — download WebGym and build a model-ready SFT parquet for Qwen3.5-4B:

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"

# 1. download WebGym into the local dataset cache
uv run python -m lite.data.hf.download WebGym \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/WebGym"

# 2. export a model-ready SFT parquet (--image-root = the dir ABOVE cua-lite/)
uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/WebGym" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  -o .data/sft/qwen3_5-reasoning/webgym/train.parquet
```

#### Train

Train in the Slime container — see [docs/slime.md](/docs/slime.md) for setup. The repo
is mounted at `/workspaces/cua-lite`, so the host `.data/sft/qwen3_5-reasoning/webgym/…` above is
`/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/webgym/…` there.

```bash
# --- Slime container ---
# 3. SFT at TP=2 (8 GPUs → DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU
#    (qwen3_5/GDN can't THD-pack). Saves one ckpt per epoch (NUM_EPOCH=3 → iter_…369).
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/webgym/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/webgym/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

- `MBS=1` is the safe start (4-image samples are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval — base vs SFT

Eval runs on the host against a running env-server. Install WebGym (see its
[README](/lite/gym/envs/webgym/README.md)) and start the env-server with the OpenAI
judge creds (see the [env-server guide](/docs/envs.md#env-server)). Export
`CUA_LITE_ENV_SERVER_URL` and `CUA_LITE_ENV_SERVER_TOKEN` before these commands;
for a direct-mode comparison, prefix the rollout command with
`env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN`. Then, on the
held-out eval split:

```bash
# --- host ---  (replace iter_<N> with the saved iter, e.g. iter_369 for epoch 3)

# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --env-id webgym --splits eval --sample 128 --seed 42 --concurrency 16 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/default/webgym.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/webgym/base &

# SFT checkpoint (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-reasoning-4b/webgym/sft/iter_<N> \
  --env-id webgym --splits eval --sample 128 --seed 42 --concurrency 16 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/webgym/sft &

wait
```

Score each run from `.logs/rollout/<model_slug>/<env_id>/<role>/summary.json` → `stats.mean_episode_return`
(denominator is `num_valid`). Qwen3.5-* are reasoning models — `temperature=0`
above is for a clean deterministic base-vs-SFT comparison; switch to `0.6` (the A3
baseline) to match paper numbers.

### RL

GRPO **continues from the SFT checkpoint** — the same adaptive-reasoning recipe
([webgym.yaml](/examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml)), now
optimized against the WebGym verifier. `run_grpo.sh`'s `HF_CKPT` overrides the
init weights with the SFT'd checkpoint (skipping the base-model download), so RL
starts from the SFT policy instead of the base model.

#### Data

No export step — the two task parquets ship in the repo and are passed in place:

- **train** — [`webgym_popular_2102.parquet`](/lite/gym/envs/webgym/data/webgym_popular_2102.parquet)
  (2,102 popular-website tasks, `split=train`)
- **eval** — [`webgym_webvoyager_val.parquet`](/lite/gym/envs/webgym/data/webgym_webvoyager_val.parquet)
  (69 held-out WebVoyager tasks, `split=eval`)

#### Train

RL runs inside the Slime container against a **running env-server** with WebGym
installed and the OpenAI judge creds exported (same prerequisite as the SFT eval;
see [docs/grpo.md](/docs/grpo.md#env-server-prerequisite) and
[docs/envs.md#env-server](/docs/envs.md#env-server)). The repo is mounted at
`/workspaces/cua-lite`, so the host `.ckpts/qwen3_5-reasoning-4b/webgym/sft/…` from SFT is
`/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/webgym/sft/…` there. Replace `iter_<N>` with the
SFT iter you trained (e.g. `iter_369` for epoch 3).

```bash
# --- Slime container ---
# sync, 8 GPU (colocate), 4B, TP=2 → DP=4. HF_CKPT = the SFT checkpoint (RL-on-SFT).
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=8 TP_SIZE=2 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/webgym/sft/iter_<N> \
  ENV_ID=webgym \
  PROMPT_DATA=/workspaces/cua-lite/lite/gym/envs/webgym/data/webgym_popular_2102.parquet \
  EVAL_PROMPT_DATA=/workspaces/cua-lite/lite/gym/envs/webgym/data/webgym_webvoyager_val.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/webgym/rl/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh

# async, 8 GPU (4 train + 4 rollout), 4B, TP=2. Add USE_TIS=1 for the off-policy correction.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_TRAIN_GPUS=4 NUM_ROLLOUT_GPUS=4 ASYNC=1 TP_SIZE=2 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  HF_CKPT=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/webgym/sft/iter_<N> \
  ENV_ID=webgym \
  PROMPT_DATA=/workspaces/cua-lite/lite/gym/envs/webgym/data/webgym_popular_2102.parquet \
  EVAL_PROMPT_DATA=/workspaces/cua-lite/lite/gym/envs/webgym/data/webgym_webvoyager_val.parquet \
  ENV_CONCURRENCY=32 \
  ROLLOUT_BATCH_SIZE=16 \
  CONFIG_PATH=/workspaces/cua-lite/examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/webgym/rl/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_grpo.sh
```

- `HF_CKPT` points at the **SFT** HF checkpoint dir (not Megatron) — it seeds the
  tokenizer + sglang + initial weights. Drop it to RL from the base model instead.
- `CONFIG_PATH` is the **same** reasoning recipe used for SFT export + eval, so the
  RL rollouts match the SFT agent surface (history, system prompt, nav tools).
- WebGym uses per-difficulty step budgets (no `max_steps` in the config); eval
  reports against the 69 WebVoyager tasks each rollout round.

#### Eval — base vs SFT vs RL

Score all three on the **same** held-out WebVoyager set the RL optimized against —
`--prompt-data` points the rollout at that parquet directly (env_id + task_id come
from each row's `env_key`), so the comparison is apples-to-apples. Needs the
env-server up with WebGym + the OpenAI judge (same prerequisite as above).

```bash
# --- host ---  (replace iter_<N> per run with the saved SFT / RL iters)
EVAL=lite/gym/envs/webgym/data/webgym_webvoyager_val.parquet

# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --env-id webgym --prompt-data $EVAL --seed 42 --concurrency 16 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/default/webgym.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/webgym/base &

# SFT checkpoint (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-reasoning-4b/webgym/sft/iter_<N> \
  --env-id webgym --prompt-data $EVAL --seed 42 --concurrency 16 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/webgym/sft &

# RL checkpoint (GPU 2)
CUDA_VISIBLE_DEVICES=2 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-reasoning-4b/webgym/rl/iter_<N> \
  --env-id webgym --prompt-data $EVAL --seed 42 --concurrency 16 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/reasoning/webgym.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/webgym/rl &

wait
```

Compare `stats.mean_episode_return` across `qwen3_5_base`, `qwen3_5_sft`, and
`qwen3_5_rl` from each `summary.json`. RL should lift the SFT policy further on
the held-out WebVoyager tasks.

## Lite.OSWorld

This is an example of training Qwen3.5-4B on Lite.OSWorld (desktop computer-use on
Ubuntu/GNOME, inline reasoning, see [desktop.use.yaml](/examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml)).

### SFT

Supervised fine-tuning Qwen3.5-4B on Lite.OSWorld trajectories rolled out by GPT-5.5.

#### Export

Export runs on the host — download Lite.OSWorld and build a model-ready SFT parquet for Qwen3.5-4B:

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"

# 1. download Lite.OSWorld into the local dataset cache
uv run python -m lite.data.hf.download Lite.OSWorld \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.OSWorld"

# 2. export a model-ready SFT parquet (--image-root = the dir ABOVE cua-lite/)
uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.OSWorld" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  -o .data/sft/qwen3_5-reasoning/lite.osworld/train.parquet
```

#### Train

Train in the Slime container — the repo is mounted at `/workspaces/cua-lite`, so the
host `.data/sft/…` above is `/workspaces/cua-lite/.data/sft/…` there.

```bash
# --- Slime container ---
# SFT at TP=2 (8 GPUs → DP=4). BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU
# (qwen3_5/GDN can't THD-pack). Saves one ckpt per epoch.
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/lite.osworld/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/lite.osworld/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

- `MBS=1` is the safe start (4-image desktop steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval — base vs SFT

Eval runs on the host against a running env-server. Install Lite.OSWorld (Docker
desktop env — see its [README](/lite/gym/envs/lite/osworld/README.md)) and start the
env-server (see the [env-server guide](/docs/envs.md#env-server)). Lite.OSWorld scores
with programmatic evaluators, so no LLM-judge creds are needed. Then, on the held-out
eval split:

```bash
# --- host ---  (replace iter_<N> with the saved iter, e.g. iter_369 for epoch 3)

# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --env-id lite.osworld --splits eval --sample 128 --seed 42 --concurrency 8 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/default/desktop.use.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/base &

# SFT checkpoint (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-reasoning-4b/lite.osworld/sft/iter_<N> \
  --env-id lite.osworld --splits eval --sample 128 --seed 42 --concurrency 8 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft &

wait
```

Score each run from `.logs/rollout/<model_slug>/<env_id>/<role>/summary.json` → `stats.mean_episode_return`
(denominator is `num_valid`). The desktop env is slow (~2-3 min/task), so
`--concurrency 8` and `--sample 128` keep the held-out eval tractable; adjust the sample
for a looser/tighter estimate.

## Lite.CUAGym

Use the shared reasoning recipe
[`desktop.use.yaml`](/examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml)
to export the published Lite.CUAGym trajectories and train Qwen3.5-4B:

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
uv run python -m lite.data.hf.download Lite.CUAGym \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAGym"
uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAGym" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  -o .data/sft/qwen3_5-reasoning/lite.cuagym/train.parquet

# --- Slime container ---
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/lite.cuagym/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/lite.cuagym/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

`lite.cuagym` exposes a train split only. Use it for training and environment
smoke checks, but do not report training-set rollout return as a held-out
benchmark result.

## Lite.CUAWorld

CUAWorld trajectories span many `lite.cuaworld.<software>` env IDs, but they
share one agent protocol. The export config does not pin an `env_id`;
`export_sft` uses the saved `LiteSample.messages` and `LiteSample.metadata`
(including `valid_actions` and `extra_tool_schemas`) define the trajectory
surface, while `agent_kwargs` select the target adapter/render settings.

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
uv run python -m lite.data.hf.download Lite.CUAWorld \
  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAWorld"
uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAWorld" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  -o .data/sft/qwen3_5-reasoning/lite.cuaworld/train.parquet

# --- Slime container ---
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/lite.cuaworld/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/lite.cuaworld/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

Evaluate a checkpoint per software on the upstream `eval` split. Do not set a
global `max_steps`: each CUAWorld task carries its own upstream budget.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --model-path .ckpts/qwen3_5-reasoning-4b/lite.cuaworld/sft/iter_<N> \
  --env-id lite.cuaworld.pymol --splits eval --sample 64 --seed 42 --concurrency 4 \
  --config-path scripts/configs/qwen3_5/default/lite.cuaworld.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.cuaworld/pymol-sft
``` -->

## Desktop

Train one model on **several desktop.use datasets at once** by concatenating them into a single
SFT parquet — here the desktop family, **Lite.OSWorld + Lite.ScaleCUA + Lite.CUAGym**; add more as
they land. They share the same `qwen3_5` reasoning adapter and identical `agent_kwargs`, so one
config drives the export. Lite.ScaleCUA is far larger than the others, so we cap it to 5000 train
trajectories (as in [README.md](/README.md#sft-any-cua-on-any-datasets)) and concatenate that with the
full smaller sets into one training set.

### SFT

#### Export

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"

# 1. download every dataset into the local dataset cache (skip any you already pulled above)
uv run python -m lite.data.hf.download Lite.OSWorld  --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.OSWorld"
uv run python -m lite.data.hf.download Lite.ScaleCUA --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA"
uv run python -m lite.data.hf.download Lite.CUAGym   --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAGym"
# Lite.CUAWorld excluded from the mix: ~173/1816 trajectories survived $FILTER (mostly tagged
# `incomplete` / `complex_shell`) — a ~19 GB pull for ~10% yield.
# UNVERIFIED, measured once on 2026-07-30 and NOT reproducible from this repo: the pair describes
# the published HF dataset, and nothing checked in here carries per-trajectory counts (the tables
# in /devs/data/lite.cuaworld/AGENTS.md count *tasks*, not trajectories). Re-deriving it needs the
# ~19 GB pull, after which `export_sft` prints the pair directly:
#   uv run python -m lite.train.export.export_sft ... \
#     --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAWorld" --filter "$FILTER"
#   denominator = sum of the "loaded <file>: <N> rows" lines it prints; numerator = that sum minus
#   the "Filter (...): <N> rows excluded" line.
# Treat the pair as a dated hint, not a fact; if you pull the dataset, re-measure and replace it
# here along with the date. Uncomment the next line and add Lite.CUAWorld back to the internalize
# loop + the `rest.parquet` --data-paths below to include it.
# uv run python -m lite.data.hf.download Lite.CUAWorld --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAWorld"

# 2. internalize CoT: desktop.use.yaml reasons in Qwen3.5's NATIVE <think> channel
#    (enable_thinking, no prompted Thought: system prompt), but the teacher stores its Thought
#    as an inline_reasoning content part. Move it into reasoning_content (which the chat
#    template renders as <think>). The transform is non-inplace and drops the source reasoning
#    fields, so there is NO duplicated reasoning; images stay path-referenced
#    to the source dataset (no image copy — the .think parquet is tiny).
for ds in Lite.OSWorld Lite.ScaleCUA Lite.CUAGym; do  # add Lite.CUAWorld to re-enable it
  uv run python examples/lite/v1/internalize_cot.py \
    --in "${CUA_LITE_DATASETS_ROOT}/cua-lite/${ds}" --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/${ds}.think"
done

# 3. export the desktop.use SFT parquet(s). --filter selects the training set from the ANNOTATED
#    datasets: filter.py keeps every trajectory and tags quality gates in
#    metadata.others.exclude_reason (nothing dropped), while metadata.others.episode_return carries reward — so
#    keep the clean, successful rows. Lite.ScaleCUA is far larger than the rest, so cap it to
#    5000 train trajectories exactly like README.md (--sample 5000 --seed 42). export_sft's
#    --sample is GLOBAL over --data-paths, so export Lite.ScaleCUA on its own (to cap only it) and the
#    others concatenated, then merge the two into one training parquet.
OUT=.data/sft/qwen3_5-reasoning/desktop.use
FILTER="lambda m: not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5"

uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.think" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "$FILTER" --sample 5000 --seed 42 \
  -o "$OUT/scalecua_5k.parquet"

uv run python -m lite.train.export.export_sft \
  --config examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml \
  --model-id Qwen/Qwen3.5-4B \
  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.OSWorld.think" \
               "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.CUAGym.think" \
  --image-root "${CUA_LITE_DATASETS_ROOT}" \
  --filter "$FILTER" \
  -o "$OUT/rest.parquet"

# 4. concat the two into one training set
uv run python -m lite.data.merge -i "$OUT/rest.parquet" "$OUT/scalecua_5k.parquet" -o "$OUT/train.parquet"
```

#### Train

```bash
# --- Slime container ---
# Same SFT recipe, pointed at the desktop.use parquet. SFT at TP=2 (8 GPUs → DP=4).
# BSHD + MBS; do NOT pass MAX_TOKENS_PER_GPU (qwen3_5/GDN can't THD-pack). One ckpt per epoch.
TP_SIZE=2 MBS=1 NUM_TRAIN_GPUS=8 \
  MODEL_ID=Qwen/Qwen3.5-4B \
  SAVE=1 NO_SAVE_OPTIM=1 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=32 LR=5e-6 \
  PROMPT_DATA=/workspaces/cua-lite/.data/sft/qwen3_5-reasoning/desktop.use/train.parquet \
  SAVE_HF_DIR=/workspaces/cua-lite/.ckpts/qwen3_5-reasoning-4b/desktop.use/sft/iter_{rollout_id} \
  bash /workspaces/cua-lite/scripts/train/run_sft.sh
```

- `MBS=1` is the safe start (4-image steps are heavy); raise to `MBS=2` only if it fits.
- TP=2 fits the 4-image step at 4B; fall back to `TP_SIZE=4` only if it OOMs.

#### Eval

`lite.cuagym` currently has no eval split. Evaluate one or more concrete
`lite.cuaworld.<software>` environments on their `eval` splits, and use the
[Lite.OSWorld](#liteosworld) held-out split to confirm the desktop.use model also
holds up outside its training distributions.

For example, evaluate on the `lite.osworld`:
```bash
# --- host ---  (replace iter_<N> with the saved iter, e.g. iter_369 for epoch 3)

# base (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B \
  --env-id lite.osworld --splits eval --sample 128 --seed 42 --concurrency 8 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/default/desktop.use.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/base &

# SFT checkpoint (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/rollout.py \
  --model-id Qwen/Qwen3.5-4B --model-path .ckpts/qwen3_5-reasoning-4b/desktop.use/sft/iter_<N> \
  --env-id lite.osworld --splits eval --sample 128 --seed 42 --concurrency 8 \
  --sampling-kwargs '{"temperature":0,"top_p":1}' \
  --config-path examples/lite/v1/configs/qwen3_5/reasoning/desktop.use.yaml \
  --log-root .logs/rollout/Qwen_Qwen3.5-4B/lite.osworld/sft &

wait
```

<!-- 
## Ablations

Keep the evaluation task IDs and seeds fixed while changing one factor:

- base model vs single-environment SFT vs mixed SFT;
- inline reasoning enabled vs disabled;
- native screenshots vs the 1280x720 agent-side resize;
- single-environment SFT vs the Lite.OSWorld + Lite.CUAGym + Lite.CUAWorld mixture;
- SFT checkpoint vs RL continued from that same SFT checkpoint.

Report `stats.mean_episode_return`, `num_valid`, and the exact env/config path
for every run. Do not compare runs that use different task sets, sampling
configurations, or task budgets. -->
