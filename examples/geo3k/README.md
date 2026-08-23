# Geo3K

Geo3K is a geometry QA env. Each task presents a geometry problem with optional
image context; the agent replies with a natural-language answer and the env
scores it. The default config is single-turn (`max_turns: 1`). The `.mt.yaml`
configs raise `max_turns`, so wrong answers return feedback and the agent can
revise before the attempt limit is reached.

This directory provides data export, local rollout, and Slime GRPO examples for
Qwen3-VL and Qwen3.5 models.

Run the commands below from the repo root after the root installation steps in
[`README.md`](/README.md).

## Data

```bash
uv run bash examples/geo3k/scripts/install.sh

mkdir -p .data
export GEO3K_SOURCE="$PWD/examples/geo3k/.cache/geo3k_imgurl/train.parquet"
uv run python -m examples.geo3k.export_tasks \
  --env-id geo3k --split train --head 128 \
  -o .data/geo3k_tasks_train_head128.parquet
```

## Rollout

Geo3K is local in-process. Unset inherited env-server variables and register
the example module before calling [`scripts/rollout.py`](/scripts/rollout.py).

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 \
  GEO3K_SOURCE="$PWD/examples/geo3k/.cache/geo3k_imgurl/train.parquet" \
  CUA_LITE_REGISTRATION_MODULES=examples.geo3k.registration \
  uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --env-id geo3k \
    --prompt-data .data/geo3k_tasks_train_head128.parquet \
    --head 32 --concurrency 32 \
    --config-path examples/geo3k/configs/qwen3_vl/geo3k.yaml
    # multi-turn: --config-path examples/geo3k/configs/qwen3_vl/geo3k.mt.yaml
```

Add `--model-path <snapshot-or-checkpoint>` to use a local model. Optional
`--sampling-kwargs`, `--engine-kwargs`, `--log-root`, and `--debug` work the
same way as the root rollout examples.

## GRPO

Launch Slime on the host. See [`docs/slime.md`](/docs/slime.md) for Slime setup
details.

```bash
SESSION_ID=geo3k-grpo CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train/slime/launch.sh

docker exec -it lite.slime-geo3k-grpo bash
```

Inside the container:

```bash
cd /workspaces/cua-lite

bash examples/geo3k/scripts/install.sh

mkdir -p .data
export GEO3K_SOURCE=/workspaces/cua-lite/examples/geo3k/.cache/geo3k_imgurl/train.parquet
python -m examples.geo3k.export_tasks \
  --env-id geo3k --split train --head 128 \
  -o .data/geo3k_tasks_train_head128.parquet

MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 \
PROMPT_DATA=/workspaces/cua-lite/.data/geo3k_tasks_train_head128.parquet \
CONFIG_PATH=/workspaces/cua-lite/examples/geo3k/configs/qwen3_vl/geo3k.yaml \
ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=8 \
NUM_STEPS_PER_ROLLOUT=1 ENV_CONCURRENCY=256 \
bash examples/geo3k/scripts/run_grpo.sh
# multi-turn: CONFIG_PATH=/workspaces/cua-lite/examples/geo3k/configs/qwen3_vl/geo3k.mt.yaml
```

Switch model family by changing `MODEL_ID` and `CONFIG_PATH` together (for
example `Qwen/Qwen3.5-2B` with `configs/qwen3_5/geo3k.yaml`). See
[`docs/grpo.md`](/docs/grpo.md) for the full set of training knobs.
