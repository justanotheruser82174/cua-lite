# androidworld · mai_ui_2b @ 2026-05-01T02-50_eb9bdce4 · run_1_sft_grpo

- **Recipe**: `devs/exps/train/androidworld/mai_ui_2b/AGENTS.md` (this commit)
- **Campaign**: SFT (2 epochs from 8B-teacher trajectories) + GRPO 20 iters. Tests whether SFT distillation on top of an already-GUI-specialist base (MAI-UI-2B, base ~42%) still pays off vs pure GRPO (run_0).
- **Commit**: `eb9bdce4` — `recipes: SFT-must-beat-base hard gate + 2026-04-30 eval data`
- **Host / GPUs**: `gpublaze` / `0,1`
- **Container**: `lite.slime-train-androidworld-qwen3_vl_2b` (shared with qwen3_vl_2b)
- **wandb**:
  - `sft`: https://wandb.ai/asap-zzhou/cua-lite-sft/runs/otr46r9j
  - `grpo`: https://wandb.ai/asap-zzhou/cua-lite-rl/runs/sx7g6qcy (one of multiple — multiple ray-restart attempts created multiple wandb runs; see `pipeline.out` for all)
- **Artifacts**: `.exps/train/androidworld/mai_ui_2b/2026-05-01T02-50_eb9bdce4/run_1_sft_grpo/`
- **Started**: `2026-05-01 ~13:00 PDT` (Step 2/3 SFT first, then Step 6 GRPO at 20:03 PDT)
- **Ended**: `2026-05-02 ~07:42 PDT` (Pipeline DONE on attempt 5 reaching iter_19; Step 7 eval running at end-of-snapshot)
- **Notes**: Reused teacher rollout from `qwen3_vl_2b/2026-04-30T01-21_e93ba3f8/run_0/teacher_rollout` (symlink, no copy). 5 GRPO attempts: attempts 1-4 crashed via `ActorUnavailableError` (see [Stability section](/devs/exps/train/androidworld/mai_ui_2b/AGENTS.md#stability-notes-ray-actorunavailableerror) of recipe README); attempt 5 (with both ray.sh + adb-timeout patches landed mid-run-1) survived 4h21m and reached iter_19. iter_19 eval inside attempt 5 hit `ActorUnavailableError` during emulator reset for the eval phase, so iter_19's deterministic eval comes from Step 7 (running at snapshot time).

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `reused` | symlink to qwen3_vl_2b's 86-task / `complexity ≤ 2.0` / 8B-Qwen3-VL teacher rollout |
| sft             | `done iter_95` | 2 epochs on 194 success-filtered trajectories (`episode_return ≥ 1.0`); iter_47 + iter_95 saved |
| grpo            | `done iter_19` | 4 ckpts (iter_4/9/14/19) — final values are attempt 5's; intermediate-attempt versions preserved in `grpo/hf_archive/` |
| eval (Step 7)   | `running at snapshot time` | `eval_base/Tongyi-MAI_MAI-UI-2B/` + `eval_grpo/iter_19/Tongyi-MAI_MAI-UI-2B/` × 3 attempts each |

## Eval results — full trajectory across all GRPO attempts

In-training eval (`eval_interval=5`, deterministic temp=0, 86-task subset). Each row is one measurement; "semantic iter" is the cumulative GRPO rollout count (constant across attempt restarts; differs from the on-disk `iter_N` filename which is **per-attempt** rollout_id and gets overwritten on resume).

| Ckpt source | semantic iter | eval | Δ vs base 41.86% | Notes |
|---|---|---|---|---|
| base `Tongyi-MAI/MAI-UI-2B` (indep eval, mean of 2) | — | **41.86%** | — | from run_0's manual indep eval |
| post-SFT iter_95 (eval_0 of attempt 1) | 0 | **56.98%** | **+15.12pp** | SFT alone — single biggest jump |
| GRPO eval 4 (attempt 2) | 9 | 59.30% | +17.44pp | |
| GRPO eval 9 (attempt 2) | 14 | 56.98% | +15.12pp | |
| GRPO eval 0 (attempt 3 reload from iter_9) | 14 | 53.49% | +11.63pp | same model — eval noise ~3pp |
| **GRPO eval 4 (attempt 3)** | **18** | **62.79%** | **+20.93pp** | first peak |
| GRPO eval 9 (attempt 3) | 23 | 59.30% | +17.44pp | |
| GRPO eval 0 (attempt 4 reload) | 23 | 60.47% | +18.61pp | |
| **GRPO eval 4 (attempt 4)** | **28** | **65.12%** | **+23.26pp** | **global peak** |
| GRPO eval 0 (attempt 5 reload from iter_9) | 23 | 58.14% | +16.28pp | |
| GRPO eval 4 (attempt 5) | 28 | 62.79% | +20.93pp | sem-28 mean across a4+a5 = **63.96%** |
| GRPO eval 9 (attempt 5) | 32 | 59.30% | +17.44pp | regression begins |
| GRPO eval 14 (attempt 5) | 37 | 56.40% | +14.54pp | back at SFT-only level |
| GRPO eval 19 (attempt 5, final) | 42 | (eval failed mid-run; Step 7 deterministic pending) | — | |

## Highlights

- **SFT does most of the work** — +15.12pp from a 2-epoch distill of 8B-Qwen3-VL teacher trajectories alone (eval_0 of GRPO attempt 1 = 56.98%, vs base 41.86%).
- **GRPO peaks at semantic iter 28** with mean 63.96% (+22.1pp), then **regresses** monotonically to 56.4% by sem 37 — a classic small-pool reward-hack signature on this 86-task setup.
- **For MAI-UI-2B on this env, iter_4 (sem 28) is the best ckpt — not iter_19**. Opposite of qwen3_vl_2b on this env (iter_14/19 best). Because MAI-UI starts at 42% (vs Qwen3-VL-2B's 24%), there's less GRPO headroom and over-training is more pronounced.
- **Mixed-group ratio averages ~43%** (range 12-69%) → about half of GRPO batches have non-zero advantage. Combined with `lr=1e-6` + `KL=0` + small task pool, the policy update signal is dominated by per-rollout sampling variance once past sem 28.
- **5 attempts of GRPO** — attempts 1-4 crashed via `ActorUnavailableError` after 1-2 h each; attempt 5 (post-patch) survived 4h21m and reached iter_19. See [Stability notes](/devs/exps/train/androidworld/mai_ui_2b/AGENTS.md#stability-notes-ray-actorunavailableerror).
- **Teacher rollout reuse worked cleanly** — symlinking qwen3_vl_2b's teacher_rollout means the SFT signal is comparable across the two recipes (qwen3_vl_2b SFT iter_95 hit 42-46% on this env; MAI-UI-2B SFT iter_95 hits 57% — much higher because MAI-UI-2B's base pretrain already encodes most of the GUI vocabulary the SFT signal is teaching).

## Comparison: run_0 (pure GRPO) vs run_1 (SFT+GRPO)

| Stage | run_0 | run_1 |
|---|---|---|
| Base | 41.86% | 41.86% |
| Post-SFT (mid-pipeline) | n/a | 56.98% (+15.1pp) |
| **Final / peak** | iter_19 = 54.07% (indep eval) | iter_4 sem 28 = **63.96%** (in-training mean of 2) |
| Total Δ | +12.2pp | **+22.1pp peak** / +14.5pp tail |
| Trend shape | monotonic-rising over 28 iters | rising to peak at sem 28, then **regressing** |
| Best ckpt to use | iter_19 | **iter_4** (sem 28) |

**Conclusion**: SFT pre-training on top of MAI-UI-2B base **adds +10pp** of peak performance over pure GRPO, but trades monotonic improvement for a peak-then-decline shape. Pick `run_1`'s `iter_4` snapshot (preserved as `hf_archive/iter_4_attempt4_semantic28_peak65pct/`) as the deployable model from this campaign.
