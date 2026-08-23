# Wordle

Wordle is a text-only multi-turn env. Each task hides a four-letter word drawn
from a catalog the observation prints in full; the agent guesses in natural
language and the env marks every letter `correct`, `present`, or `absent`, then
asks for the next guess until the word is solved or the attempt limit is
reached.

This directory provides task export, local rollout, and Slime GRPO examples for
Qwen3-VL and Qwen3.5 models.

Run the commands below from the repo root after the root installation steps in
[`README.md`](/README.md).

## Data

The word catalog ships in [`words.py`](/examples/wordle/words.py); task ids run
from `word_000000` to `word_000255`.

```bash
mkdir -p .data
PYTHONPATH="$PWD" uv run python -m examples.wordle.export_tasks \
  --env-id wordle --split train \
  -o .data/wordle_tasks_train.parquet
```

## Rollout

Wordle is local in-process. Unset inherited env-server variables and register
the example module before calling [`scripts/rollout.py`](/scripts/rollout.py).

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES=0 \
  CUA_LITE_REGISTRATION_MODULES=examples.wordle.registration \
  uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --env-id wordle \
    --prompt-data .data/wordle_tasks_train.parquet \
    --head 32 --concurrency 32 \
    --config-path examples/wordle/configs/qwen3_vl/wordle.yaml
```

Add `--model-path <snapshot-or-checkpoint>` to use a local model. Optional
`--sampling-kwargs`, `--engine-kwargs`, `--log-root`, and `--debug` work the
same way as the root rollout examples.

## GRPO

Launch Slime on the host. See [`docs/slime.md`](/docs/slime.md) for Slime setup
details.

```bash
SESSION_ID=wordle-grpo CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train/slime/launch.sh

docker exec -it lite.slime-wordle-grpo bash
```

Inside the container:

```bash
cd /workspaces/cua-lite

mkdir -p /root/datasets/cua-lite/wordle
python -m examples.wordle.export_tasks \
  --env-id wordle --split train \
  -o /root/datasets/cua-lite/wordle/train.parquet

MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 \
PROMPT_DATA=/root/datasets/cua-lite/wordle/train.parquet \
CONFIG_PATH=/workspaces/cua-lite/examples/wordle/configs/qwen3_vl/wordle.yaml \
ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=8 \
NUM_STEPS_PER_ROLLOUT=1 ENV_CONCURRENCY=256 \
bash examples/wordle/scripts/run_grpo.sh
```

Switch model family by changing `MODEL_ID` and `CONFIG_PATH` together (for
example `Qwen/Qwen3.5-2B` with `configs/qwen3_5/wordle.yaml`). See
[`docs/grpo.md`](/docs/grpo.md) for the full set of training knobs.
