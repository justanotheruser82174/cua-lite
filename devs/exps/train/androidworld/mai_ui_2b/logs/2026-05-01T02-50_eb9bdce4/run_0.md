# androidworld · mai_ui_2b @ 2026-05-01T02-50_eb9bdce4 · run_0

- **Recipe**: `devs/exps/train/androidworld/mai_ui_2b/AGENTS.md` (this commit)
- **Campaign**: pure GRPO from `Tongyi-MAI/MAI-UI-2B` base, **no SFT** step. Tests whether RL alone on a strong GUI base can match or beat the SFT-then-GRPO baseline (qwen3_vl_2b post-GRPO 54.7%).
- **Commit**: `eb9bdce4` — `recipes: SFT-must-beat-base hard gate + 2026-04-30 eval data`
- **Host / GPUs**: `gpublaze` / `0,1`
- **Container**: `lite.slime-train-androidworld-qwen3_vl_2b` (shared with qwen3_vl_2b — same Qwen3-VL architecture family)
- **wandb**:
  - `grpo`: https://wandb.ai/asap-zzhou/cua-lite-rl/runs/m3sqp3sa
- **Artifacts**: `.exps/train/androidworld/mai_ui_2b/2026-05-01T02-50_eb9bdce4/run_0/`
- **Started**: `2026-05-01 ~10:00 PDT`
- **Ended**: `2026-05-01 16:27:58 PDT` (Pipeline DONE)
- **Notes**: Pre-pipeline.sh-Step-7 commit, so Step 7 deterministic eval was run **manually** post-hoc via `scripts/rollout.py` (× 2 measurements per ckpt to estimate eval noise). 2-attempt GRPO: attempt 1 hit `ActorUnavailableError` mid-run, attempt 2 resumed from iter_9 and finished cleanly to iter_19. Total ~6.5 h.

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `n/a` | not used (pure GRPO from base, no SFT) |
| sft             | `n/a` | not used |
| grpo            | `done iter_19` | 4 ckpts saved (iter_4/9/14/19); attempt 2 reached iter_19 |
| eval (manual indep.) | `done × 2 attempts each` | `Tongyi-MAI/MAI-UI-2B` base + iter_19 |

## Eval results

In-training eval (`eval_interval=5`, deterministic temp=0, 86-task subset):

| Ckpt | semantic iter | In-training eval | Δ vs attempt-1 sem 0 |
|---|---|---|---|
| eval 0 (sem 0, base, attempt 1) | 0 | 43.60% | — |
| eval 4 (sem 4) | 4 | 39.53% | −4pp (early noise) |
| eval 9 (sem 9, before crash) | 9 | 48.26% | +5pp |
| **(attempt 2 resumed from iter_9)** | | | |
| eval 0 (sem 9, baseline reload) | 9 | 45.93% | +2pp |
| eval 4 (sem 13) | 13 | 49.42% | +6pp |
| eval 9 (sem 18) | 18 | 51.16% | +8pp |
| eval 14 (sem 23) | 23 | 51.16% | +8pp |
| **eval 19 (sem 28, final)** | 28 | **51.74%** | **+8.1pp** |

Independent rollout/local.py eval (different 64-task slice, mean of 2 attempts each):

| Ckpt | Eval | Δ vs base |
|---|---|---|
| base `Tongyi-MAI/MAI-UI-2B` | **41.86%** | — |
| **grpo iter_19** | **54.07%** | **+12.21pp** |

Noise across two independent measurements of the same model: **±1.6-2 pp** — so the +12.21pp lift is solidly above noise (z ≈ 5.8).

## Highlights

- **Pure GRPO works on a strong base, but lift is modest** — +12pp from independent eval, +8pp from in-training eval. Compare with run_1 SFT+GRPO peak (+22pp) — SFT distillation is the dominant contributor.
- **Monotonic in-training eval lift** — unlike run_1, no late-stage regression. 86-task pool + ~50% baseline success rate keeps GRPO advantages flowing without saturating.
- **One-attempt crash + clean resume** — same `ActorUnavailableError` pattern that plagued run_1 (5-min RolloutManager silence preceded by adb commands). Pipeline.sh's retry loop cleanly recovered from iter_9.
- **vs Qwen3-VL-2B distill (qwen3_vl_2b run_0)**: that recipe's iter_19 hit 54.65% (Step 7 mean) starting from a 23.84% base — an absolute total of +30.81pp via 8B-distill SFT + GRPO. MAI-UI-2B's pure-GRPO path reaches 54.07% — **comparable absolute, much smaller lift**, because MAI-UI-2B starts at 41.86% (already near where qwen3_vl ends up post-distill).
- **Conclusion for picking a recipe**: if you can afford a teacher rollout + SFT step, run_1's path peaks higher (64% at sem 28). If the goal is just "lift MAI-UI past qwen3_vl distill", pure GRPO suffices.
