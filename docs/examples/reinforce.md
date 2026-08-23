# REINFORCE / Filtered-BC Training

All commands run inside the Slime container. For container setup, see [docs/slime.md](/docs/slime.md).

## Env-server prerequisite

Slime drives environments through a managed env-server (required — it handles rollout load balancing and environment resource management). Before any command below, the env-server must be running and the container must have `CUA_LITE_ENV_SERVER_URL` + `CUA_LITE_ENV_SERVER_TOKEN` exported. For per-environment setup, see [docs/envs.md#installation](/docs/envs.md#installation); to start the server, see [docs/envs.md#env-server](/docs/envs.md#env-server).

## Variants

Each example below trains **Qwen3-VL-2B**, **sync** (colocate, 2 GPU). Two switches apply everywhere:

- **Async** (1 train + 1 rollout GPU): add `NUM_TRAIN_GPUS=1 NUM_ROLLOUT_GPUS=1 ASYNC=1`.
- **Qwen3.5**: set `MODEL_ID=Qwen/Qwen3.5-2B` and swap the config family to `qwen3_5/` (e.g. `.../scripts/configs/qwen3_5/compact/<env>.yaml`).

---

## ScreenSpot-Pro

Full task set (1581 tasks) with a proper train/eval split (128 random tasks held out for eval).

```bash
# step 1: export all tasks, then split off 128 for eval
python -m lite.train.export.export_tasks --env-id screenspot_pro --split eval \
  -o /root/datasets/cua-lite/screenspot_pro/all.parquet
python -m lite.data.split \
  -i /root/datasets/cua-lite/screenspot_pro/all.parquet --eval-size 128 \
  --train-output /root/datasets/cua-lite/screenspot_pro/train.parquet \
  --eval-output /root/datasets/cua-lite/screenspot_pro/eval.parquet

# step 2: train
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=screenspot_pro \
  PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/screenspot_pro/eval.parquet \
  ENV_CONCURRENCY=32 ROLLOUT_BATCH_SIZE=128 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/screenspot_pro.yaml \
  bash /workspaces/cua-lite/scripts/train/run_reinforce.sh
```

---

## Lite.OSWorld

Domain-filtered train split (676 tasks). Eval: 64 sampled from the 96 matching eval tasks.
The domain filter drops infeasible/auth tasks and hard domains where small models near-zero out.

```bash
# step 1: export train + eval (64 sampled), same domain filter on both
python -m lite.train.export.export_tasks --env-id lite.osworld --split train \
  --filter "lambda m: not m.others.get('exclude_reason') and m.others.get('domain') in ('chrome', 'vs_code', 'os', 'gimp')" \
  -o /root/datasets/cua-lite/lite.osworld/train.parquet
python -m lite.train.export.export_tasks --env-id lite.osworld --split eval --sample 64 \
  --filter "lambda m: not m.others.get('exclude_reason') and m.others.get('domain') in ('chrome', 'vs_code', 'os', 'gimp')" \
  -o /root/datasets/cua-lite/lite.osworld/eval_64.parquet

# step 2: train
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=lite.osworld \
  PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/lite.osworld/eval_64.parquet \
  ENV_CONCURRENCY=32 ROLLOUT_BATCH_SIZE=128 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/lite.osworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_reinforce.sh
```

---

## WebGym

The HuggingFace dataset has native train (292k) / test (1,167 OOD, unseen websites) splits.
Difficulty 1-3 = easy, 4-6 = medium, 7+ = hard. Dev defaults to easy only.

```bash
# step 1: export easy tasks only (difficulty <= 3) — 1024 sampled for train, 64 for eval
python -m lite.train.export.export_tasks --env-id webgym --split train --sample 1024 \
  --filter "lambda m: m.others.get('difficulty', 0) <= 3" \
  -o /root/datasets/cua-lite/webgym/train.parquet
python -m lite.train.export.export_tasks --env-id webgym --split eval --head 64 \
  --filter "lambda m: m.others.get('difficulty', 0) <= 3" \
  -o /root/datasets/cua-lite/webgym/eval.parquet

# step 2: train (browser pool size comes from webgym's configs/default.yaml `instances:`, or WEBGYM_CONFIG)
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=webgym \
  PROMPT_DATA=/root/datasets/cua-lite/webgym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/webgym/eval.parquet \
  ENV_CONCURRENCY=32 ROLLOUT_BATCH_SIZE=128 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/webgym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_reinforce.sh
```

---

## AndroidWorld

Easy + medium tasks only (86 per split, filtering out hard tasks where small models have
near-zero success). Train split (`perturb_*`) gets random params each reset; eval split gets
deterministic params via `seed=42` (baked into eval task registration).

```bash
# step 1: export train (perturb_* IDs, randomized) + eval (original IDs, seeded)
python -m lite.train.export.export_tasks --env-id androidworld --split train \
  --filter "lambda m: m.others.get('complexity') <= 2.0" \
  -o /root/datasets/cua-lite/androidworld/train.parquet
python -m lite.train.export.export_tasks --env-id androidworld --split eval \
  --filter "lambda m: m.others.get('complexity') <= 2.0" \
  -o /root/datasets/cua-lite/androidworld/eval.parquet

# step 2: train
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=androidworld \
  PROMPT_DATA=/root/datasets/cua-lite/androidworld/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/androidworld/eval.parquet \
  ENV_CONCURRENCY=32 ROLLOUT_BATCH_SIZE=128 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/androidworld.yaml \
  bash /workspaces/cua-lite/scripts/train/run_reinforce.sh
```

---

## MobileGym

Browser-simulated mobile env (24 apps, deterministic state-diff judge, Docker, no KVM). Train
split = 160 randomized tasks; eval split = 256 deterministic seeded tasks filtered to **L1+L2**
(easy + medium) so small models have a non-trivial baseline. Difficulty is in
`m.others['difficulty']` as `L1/L2/L3/L4`.

```bash
# step 1: export full 160-task train (randomized) + L1+L2 eval (deterministic seeds)
python -m lite.train.export.export_tasks --env-id mobilegym --split train \
  -o /root/datasets/cua-lite/mobilegym/train.parquet
python -m lite.train.export.export_tasks --env-id mobilegym --split eval \
  --filter "lambda m: m.others.get('difficulty') in ('L1', 'L2')" \
  -o /root/datasets/cua-lite/mobilegym/eval.parquet

# step 2: train
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_GPUS=2 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/train.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval.parquet \
  ENV_CONCURRENCY=32 ROLLOUT_BATCH_SIZE=128 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/compact/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_reinforce.sh
```

Siblings: [docs/grpo.md](/docs/grpo.md), [docs/examples/dagger.md](/docs/examples/dagger.md), [docs/sft.md](/docs/sft.md).
