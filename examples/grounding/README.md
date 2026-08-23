# Grounding

Grounding is a multi-turn RL example for OSWorld-G.
[RegionFocus](https://arxiv.org/abs/2505.00684) is a test-time grounding
algorithm; this example uses that interaction pattern as the grounding harness:
predict an initial point, self-judge it, then run seed/crop refinement plus
aggregation when the initial point is likely wrong. The final action is a point
click that OSWorld-G scores.

This directory provides the grounding adapter/agent, local rollout, and Slime
GRPO examples for Qwen3-VL and Qwen3.5 models.

Run the commands below from the repo root after the root installation steps in
[`README.md`](/README.md).

## Rollout

Install OSWorld-G data, then run a small click-only rollout. Refusal tasks are
filtered out because they require `report_infeasible`, not a point click.

```bash
uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py

env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  CUDA_VISIBLE_DEVICES=0 \
  uv run python -m examples.grounding.rollout \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --env-id osworld_g \
    --config-path examples/grounding/configs/qwen3_vl/osworld_g.regionfocus.yaml \
    --filter "lambda m: not m.others.get('exclude_reason')" \
    --head 32 --concurrency 32
```

Add `--model-path <snapshot-or-checkpoint>` to use a local model, or
`--sglang-server-url http://host:port` to reuse an existing server. Rollout logs
carry the usual `turn_NNNN/*` files plus `turn_NNNN/regionfocus/` traces.

## GRPO

Launch Slime on the host. See [`docs/slime.md`](/docs/slime.md) for Slime setup
details.

```bash
SESSION_ID=regionfocus-grpo CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train/slime/launch.sh

docker exec -it lite.slime-regionfocus-grpo bash
```

Inside the container, install OSWorld-G data, export prompt rows, and split
train/eval:

```bash
cd /workspaces/cua-lite
unset CUA_LITE_ENV_SERVER_URL CUA_LITE_ENV_SERVER_TOKEN

python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py

mkdir -p /root/datasets/cua-lite/osworld_g
python -m lite.train.export.export_tasks \
  --env-id osworld_g --split eval \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  -o /root/datasets/cua-lite/osworld_g/all.parquet

python -m lite.data.split \
  -i /root/datasets/cua-lite/osworld_g/all.parquet --eval-size 64 \
  --train-output /root/datasets/cua-lite/osworld_g/train.parquet \
  --eval-output /root/datasets/cua-lite/osworld_g/eval.parquet
```

Run grounding GRPO:

```bash
MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
NUM_TRAIN_GPUS=2 NUM_ROLLOUT_GPUS=2 \
PROMPT_DATA=/root/datasets/cua-lite/osworld_g/train.parquet \
EVAL_PROMPT_DATA=/root/datasets/cua-lite/osworld_g/eval.parquet \
CONFIG_PATH=/workspaces/cua-lite/examples/grounding/configs/qwen3_vl/osworld_g.regionfocus.yaml \
ROLLOUT_BATCH_SIZE=32 N_SAMPLES_PER_PROMPT=8 \
NUM_STEPS_PER_ROLLOUT=1 ENV_CONCURRENCY=256 \
bash examples/grounding/scripts/run_grpo.sh
```

Switch model family by changing `MODEL_ID` and `CONFIG_PATH` together (for
example `Qwen/Qwen3.5-2B` with `configs/qwen3_5/osworld_g.regionfocus.yaml`).
See [`docs/grpo.md`](/docs/grpo.md) for the full set of training knobs.
