# androidworld · qwen3_5_2b @ 2026-04-30T01-21_e93ba3f8 · run_0

- **Recipe**: `devs/exps/train/androidworld/qwen3_5_2b/AGENTS.md`
- **Commit**: `e93ba3f8` — `Merge refactor: lite.X env_ids + : → @ registry sep + Qwen3VLHistoryProtocol`
- **Host / GPUs**: `gpublaze` / `4,5`
- **Container**: `lite.slime-train-androidworld-qwen3_5_2b` (started 2026-04-30 ~02:00 PDT)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `https://wandb.ai/asap-zzhou/cua-lite-sft/runs/jjaieqpw`
  - `grpo`: `https://wandb.ai/asap-zzhou/cua-lite-rl/runs/c47xu4s6`
- **Artifacts**: `.exps/train/androidworld/qwen3_5_2b/2026-04-30T01-21_e93ba3f8/run_0/`
- **Started**: `2026-04-30 03:00 PDT` (SFT done by 03:18; GRPO Step 6 launched ~04:00)
- **Last updated**: `2026-04-30 19:56 PDT`
- **Notes**: Wave-1 C3. Reuses companion qwen3_vl_2b's teacher_rollout (text-only SFT export, qwen3_5 chat template). Pipeline retry loop fired once; attempt 2 from sft/iter_95 reached iter_19 cleanly.

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `done` | reused from companion qwen3_vl_2b run |
| sft             | `done iter_95` | run_sft_qwen3_5.sh (poisons venv to transformers ≥5); iter_95 picked as GRPO warm-start |
| grpo            | `done iter_19` | 4 ckpts saved (iter_4/9/14/19); attempt 2 reached iter_19 at 14:04 PDT |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3.5-2B`         | `androidworld` eval | 86/86 | 5.23% | — |
| sft iter_95 (in-training eval_0) | `androidworld` eval | 86/86 | ~48.3–53.5% | +43.0 to +48.3pp |
| grpo iter_4 (in-training)        | `androidworld` eval | 86/86 | 45.93% | +40.7pp |
| grpo iter_9 (in-training)        | `androidworld` eval | 86/86 | 63.95% | +58.7pp |
| grpo iter_14 (in-training)       | `androidworld` eval | 86/86 | 65.12% | +59.9pp |
| grpo iter_19 (in-training)       | `androidworld` eval | 86/86 | 61.63% | +56.4pp |
| grpo iter_19 (Step 7 deterministic) | `androidworld` eval | 86/86 | **62.21%** | **+56.98pp** |

## Highlights

- text-only Qwen3.5-2B starts at **5.2%** on androidworld (no vision → can't ground accessibility tree alone), but SFT distill from the 8B vision teacher pushes it to **~50%** in 2 epochs, and GRPO adds another ~+12pp on top. Net **+57pp** end-to-end.
- iter_14 (65.1%) outperforms iter_19 (61.6%) — same plateau pattern as the qwen3_vl companion, slightly more pronounced. iter_14 is the "best" ckpt by in-training eval.
- venv mutation from run_sft_qwen3_5.sh did not contaminate sibling runs (the qwen3_vl container is separate).
- GRPO mixed-group ratio sat at ~70% throughout — way above the G2 acceptance gate of 50%.
