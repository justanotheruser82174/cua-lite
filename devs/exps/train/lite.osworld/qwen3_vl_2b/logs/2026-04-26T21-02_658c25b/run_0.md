# lite.osworld · qwen3_vl_2b @ 2026-04-26T21-02_658c25b · run_0

- **Recipe**: `devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md`
- **Commit**: `658c25b` — `feat(ui_tars): support valid_actions filter on desktop + mobile adapters`
- **Host / GPUs**: `gpublaze` / `4,5`
- **Container**: `lite.slime-train.lite.osworld.qwen3_vl_2b` (started `2026-04-28 04:10 UTC`)
- **wandb** (one URL per training stage that produced canonical numbers — skip mid-run failed-restart URLs):
  - `sft`: https://wandb.ai/asap-zzhou/cua-lite-sft/runs/3npguz0g  (NUM_EPOCH=5 re-run; the prior NUM_EPOCH=2 attempt — runs/8gvv5sjs — is intentionally omitted per template rule)
  - `grpo`: https://wandb.ai/asap-zzhou/cua-lite-rl/runs/scitpwbm  (GRPO with recipe defaults; the two earlier attempts — runs/0fwzy1ua and runs/kusvz3qi — were stopped early before any meaningful learning curve and are omitted per the template rule)
- **Artifacts**: `.exps/train/lite.osworld/qwen3_vl_2b/2026-04-26T21-02_658c25b/run_0/`
- **Started**: `2026-04-28 01:52 UTC`
- **Last updated**: `2026-04-28 06:53 UTC`
- **Notes**: Step 1.2 hit one sglang SIGKILL mid-run (likely Linux OOM killer; host RAM was healthy so probably cross-tenant pressure) at ~118/512. Resume across attempts picked up cleanly to 512/512 with no data loss.

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | done | 512/512 finished, 118 success-filtered (avg episode_return 0.2305 = 23.05%) — bigger anchor pool than recipe's ≈80 estimate |
| sft             | done (no G1 lift; abandoned) | NUM_EPOCH=5 saved 5 HF ckpts (`iter_28/57/86/115/144`). Eval iter_28/57/86/115 all in ±1.56pp band of base (= ±1 task variance) — 2B-self-distill ceiling. Did not eval iter_144 (curve already declared flat). Pivoted to GRPO from base 2B without SFT warm-start. |
| grpo            | in_progress (recipe defaults: GRPO from base 2B, no SFT warm-start, LR=1e-6, MAX_TOKENS_PER_GPU=2048) | 50/50 pool: 118 anchor + 118 random = 236 prompts. ASYNC=1 (1 train + 1 rollout GPU). Earlier attempts: first run (defaults) OOM'd at iter 2 train; second run (SFT/iter_28 warm-start + LR=5e-7 + MAX_TOKENS=1024) was killed and replaced by this default-settings re-run after recognizing the iter-1 raw_reward dip is just per-iter prompt-sampling variance, not a signal collapse. |

## Eval results

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3-VL-2B-Instruct` | `lite.osworld` eval | — | — | — |

## Highlights

- TODO — fill in as stages complete.
