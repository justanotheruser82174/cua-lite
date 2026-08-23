# androidworld · qwen3_vl_2b @ 2026-04-28T03-16_a671f2b · run_0

- **Recipe**: `devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md`
- **Commit**: `a671f2b` — `train(recipes): bump SFT NUM_EPOCH 2 → 5 to match current practice`
- **Host / GPUs**: `gpublaze` / `4,5`
- **Container**: `lite.slime-train.androidworld.qwen3_vl_2b` (started 2026-04-28 03:18 PDT)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `<pending>`
  - `grpo`: `<pending>`
- **Artifacts**: `.exps/train/androidworld/qwen3_vl_2b/2026-04-28T03-16_a671f2b/run_0/`
- **Started**: `2026-04-28 03:18 PDT`
- **Last updated**: `2026-04-28 03:23 PDT`
- **Notes**: 2B self-distill recipe (SFT NUM_EPOCH=5). Teacher pool: 86 train tasks (complexity ≤ 2.0).

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `in_progress` | — |
| sft             | `not started` | — |
| grpo            | `not started` | — |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3-VL-2B-Instruct` | `androidworld` eval | — | — | — |

## Highlights

- Step 1.1 done: 86 train tasks pinned to `teacher_pool.parquet`.
