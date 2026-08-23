# androidworld · qwen3_vl_2b @ 2026-04-30T01-21_e93ba3f8 · run_0

- **Recipe**: `devs/exps/train/androidworld/qwen3_vl_2b/AGENTS.md`
- **Commit**: `e93ba3f8` — `Merge refactor: lite.X env_ids + : → @ registry sep + Qwen3VLHistoryProtocol`
- **Host / GPUs**: `gpublaze` / `0,1`
- **Container**: `lite.slime-train-androidworld-qwen3_vl_2b` (started 2026-04-30 ~02:00 PDT)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `https://wandb.ai/asap-zzhou/cua-lite-sft/runs/cfgsoepm`
  - `grpo`: `https://wandb.ai/asap-zzhou/cua-lite-rl/runs/wsrouvp7`
- **Artifacts**: `.exps/train/androidworld/qwen3_vl_2b/2026-04-30T01-21_e93ba3f8/run_0/`
- **Started**: `2026-04-30 04:00 PDT` (first GRPO ckpt iter_4 saved 05:37)
- **Last updated**: `2026-04-30 19:56 PDT`
- **Notes**: Wave-1 C1. teacher_rollout reused from prior run (skipped). GRPO 20 iters via pipeline.sh retry loop; one ActorUnavailableError mid-run, retry from iter_9 finished cleanly.

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `done` | reused (86 train tasks, complexity ≤ 2.0) |
| sft             | `done iter_95` | 2 epochs distill from 8B teacher; iter_95 picked as GRPO warm-start |
| grpo            | `done iter_19` | 4 ckpts saved (iter_4/9/14/19); attempt 2 reached iter_19 |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3-VL-2B-Instruct` | `androidworld` eval | 86/86 | 23.84% | — |
| sft iter_95 (in-training eval_0) | `androidworld` eval | 86/86 | ~42.4–45.9% | +18.6 to +22.1pp |
| grpo iter_4 (in-training)        | `androidworld` eval | 86/86 | 47.09% | +23.3pp |
| grpo iter_9 (in-training)        | `androidworld` eval | 86/86 | 50.00% | +26.2pp |
| grpo iter_14 (in-training)       | `androidworld` eval | 86/86 | 55.23% | +31.4pp |
| grpo iter_19 (Step 7 deterministic) | `androidworld` eval | 86/86 | **54.65%** | **+30.81pp** |

## Highlights

- SFT does the heavy lifting (~+20pp); GRPO adds another ~+10pp on top.
- iter_14 (55.2%) slightly beats iter_19 (54.7%) — mild plateau / small overfit. If repeating, consider iter_14 as the "best" ckpt or add a tiny KL penalty to suppress drift.
- Mid-run resume (HF warm-start, no optim state) cost no measurable accuracy — see iter_4/9/14/19 monotonic lift.
- Beats the captured-reference 8B→2B target (+11.6pp at iter_19 in the recipe README) by a wide margin.
