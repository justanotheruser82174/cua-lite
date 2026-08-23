# DAgger (Online Teacher-Forcing) Training

All commands run inside the Slime container. For container setup, see [docs/slime.md](/docs/slime.md).

DAgger is **online SFT**: the student rolls out in the env with its **own** actions, then each
visited step's SFT target is relabeled with a **teacher's** generated action (the executed
actions + context history stay the student's own on-policy rollout — only the target is
swapped). Trained with `sft_loss`; advantages disabled. Unlike GRPO/REINFORCE it needs a
**teacher**, which `run_dagger.sh` serves automatically (no separate command).

## Env-server prerequisite

Slime drives environments through a managed env-server (required — it handles rollout load balancing and environment resource management). Before any command below, the env-server must be running and the container must have `CUA_LITE_ENV_SERVER_URL` + `CUA_LITE_ENV_SERVER_TOKEN` exported. For per-environment setup, see [docs/envs.md#installation](/docs/envs.md#installation); to start the server, see [docs/envs.md#env-server](/docs/envs.md#env-server).

DAgger currently has an explicit recipe only for MobileGym. Single-step
grounding envs such as `screenspot_pro` and `osworld_g` are intentionally
rejected by `run_dagger.sh`; use the RegionFocus GRPO/REINFORCE grounding
recipes unless a dedicated DAgger grounding config is added.

## Teacher

`NUM_TEACHER_GPUS` (default 1) carves that many GPUs off the **top** of `CUDA_VISIBLE_DEVICES`
for the frozen teacher (served by [serve_teacher.sh](/scripts/train/utils/serve_teacher.sh) —
handles download + dynamic port + retry); the student keeps the rest. `TEACHER_PATH` picks the
teacher model. `NUM_TEACHER_GPUS=0` ⇒ no local teacher: set the URL yourself, and ask for
self-distillation by name with `teacher_url="self"` — there is no implicit fallback to the rollout
engine. Teacher URL precedence: yaml `dagger.teacher_url` > `DAGGER_TEACHER_URL`; if neither is set
`_teacher_url` raises.

---

## MobileGym

Browser-simulated mobile env. Example: **2B student ← 4B teacher** on a 40-task L1+L2 subset.

```bash
# step 1: export the 40-task L1+L2 subset (data processing runs in the container).
#   train = eval here for a "can DAgger fit?" probe (deterministic eval seeds).
python -m lite.train.export.export_tasks --env-id mobilegym --split eval \
  --filter "lambda m: m.others.get('difficulty') in ('L1', 'L2')" \
  --head 40 -o /root/datasets/cua-lite/mobilegym/eval_l1l2_40.parquet

# step 2: train (sync, 2 GPU) — teacher 4B on GPU0, student 2B on GPU1 (NUM_TEACHER_GPUS=1 off the top).
CUDA_VISIBLE_DEVICES=0,1 NUM_TEACHER_GPUS=1 \
  MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
  TEACHER_PATH=/root/models/Qwen/Qwen3-VL-4B-Instruct \
  ENV_ID=mobilegym \
  PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval_l1l2_40.parquet \
  EVAL_PROMPT_DATA=/root/datasets/cua-lite/mobilegym/eval_l1l2_40.parquet \
  ROLLOUT_BATCH_SIZE=40 N_SAMPLES_PER_PROMPT=1 \
  ENV_CONCURRENCY=40 \
  CONFIG_PATH=/workspaces/cua-lite/scripts/configs/qwen3_vl/recipes/dagger/mobilegym.yaml \
  bash /workspaces/cua-lite/scripts/train/run_dagger.sh
```

Watch the teacher boot: `docker exec <container> tail -f /tmp/dagger_teacher.log`.
Key metrics: `eval/mobilegym_eval` (mean episode return), `rollout/nonzero_return_rate`,
`rollout/dagger_unparseable_rate` (teacher relabel coverage — high = bad signal).

### Key knobs

| Var | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-VL-2B-Instruct` | **student** HF model id |
| `TEACHER_PATH` | `Qwen/Qwen3-VL-4B-Instruct` | teacher model served by `serve_teacher.sh` |
| `NUM_TEACHER_GPUS` | `1` | GPUs carved off the top for the teacher (`0` = no local teacher; set `DAGGER_TEACHER_URL` yourself, `"self"` included) |
| `HF_CKPT` | `/root/models/${MODEL_ID}` | override student init (e.g. feed an SFT'd ckpt back in) |
| `NUM_STEPS_PER_ROLLOUT` | `8` | optimizer steps/rollout; GBS = `rbs*n_samples // k` (groups/step) |
| `dagger.step_filter` (yaml) | `keep_all` | imitate every student-visited step; `success_traj` = anti-DAgger ablation |

DUMP=1 dataflow check + the full design rationale live in the
[run_dagger.sh](/scripts/train/run_dagger.sh) header.

Siblings: [docs/grpo.md](/docs/grpo.md), [docs/examples/reinforce.md](/docs/examples/reinforce.md), [docs/sft.md](/docs/sft.md).
