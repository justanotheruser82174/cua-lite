# lite.osworld · qwen3_5_2b @ 2026-04-30T01-21_e93ba3f8 · run_0

- **Recipe**: `devs/exps/train/lite.osworld/qwen3_5_2b/AGENTS.md`
- **Commit**: `e93ba3f8` — `Merge refactor: lite.X env_ids + : → @ registry sep + Qwen3VLHistoryProtocol`
- **Host / GPUs**: `gpublaze` / `0,1`
- **Container**: `lite.slime-train-lite.osworld-qwen3_5_2b` (started 2026-04-30 13:14 PDT — Wave-2, after C1 freed GPU 0,1)
- **wandb** (skip mid-run failed-restart URLs):
  - `sft`: `https://wandb.ai/asap-zzhou/cua-lite-sft/runs/lhnjhz8x`
  - `grpo`: `https://wandb.ai/asap-zzhou/cua-lite-rl/runs/tkm101d8`
- **Artifacts**: `.exps/train/lite.osworld/qwen3_5_2b/2026-04-30T01-21_e93ba3f8/run_0/`
- **Started**: `2026-04-30 13:14 PDT` (Step 2 export; SFT done by 13:53; GRPO Step 6 launched 13:54)
- **Last updated**: `2026-04-30 22:08 PDT`
- **Notes**: Wave-2 C4. Companion `lite.osworld/qwen3_vl_2b` provides teacher_rollout. ENV_CONCURRENCY=16 (vs C2's 32) since both share the lite.osworld env layer and rootless docker port allocation contends. Pipeline stable; 0 retries fired. Stopped at iter_14 to free GPU for an independent deterministic eval pass mirroring C2's.

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `done` | reused from companion qwen3_vl_2b run |
| sft             | `done iter_35` | qwen3_5 chat template; run_sft_qwen3_5.sh poisoned this venv to transformers ≥5 (expected) |
| grpo            | `stopped at iter_14` | 3 ckpts saved (iter_4/9/14, latest 19:12 PDT). Killed before iter_19 to free GPU for `eval_independent`. |
| eval_independent | `done` | base + sft iter_35 + grpo iter_4/9/14, same `random.seed(42)` 64-task subset, T=0. Artifacts under `eval_independent/<name>/Qwen_Qwen3.5-2B/summary.json`. |

## Eval results

Independent deterministic re-eval (T=0, same 64-task subset across all rows):

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3.5-2B`             | `lite.osworld` 64-task | 64/64 | **4.69%**  | — |
| sft iter_35                          | `lite.osworld` 64-task | 64/64 | **7.81%**  | **+3.13pp** ✅ |
| grpo iter_4                          | `lite.osworld` 64-task | 64/64 | **7.81%**  | +3.13pp |
| grpo iter_9                          | `lite.osworld` 64-task | 64/64 | **7.81%**  | +3.13pp |
| grpo iter_14                         | `lite.osworld` 64-task | 64/64 | **7.81%**  | +3.13pp |

(In-training eval values from GRPO Step 6 stdout — 7.81% at eval_0, 10.94 at eval_4, 6.25 at eval_9, 9.38 at eval_14 — turn out to be noisy reads of essentially the same underlying performance; the deterministic re-eval pins all 4 GRPO ckpts at the same 7.81%.)

## Highlights

- **G1 marginal pass**: SFT iter_35 lifted 4.69 → 7.81 = +3.13pp. Above the 0pp regression line but below the 5pp target the recipe README now sets. Take it as "SFT didn't hurt" rather than "SFT helped meaningfully".
- **G3 fail**: GRPO contributed 0pp on top of SFT — all 4 grpo ckpts (iter_4/9/14, plus the still-on-disk attempt-1 saves) measure identical to SFT. The training-time eval curve fluctuated (7.8 → 10.9 → 6.2 → 9.4) but those swings are within binom SE on a 64-task set; the stable deterministic value is 7.81%. lite.osworld + text-only model gives reward signal too sparse for GRPO to find a gradient.
- **Compare with androidworld / qwen3.5 (C3)**: same model family, same recipe shape, but +57pp end-to-end on androidworld vs the +3pp here. Difference is reward density: androidworld's a11y tree gives a text-only model real signal; lite.osworld doesn't.
- **No mid-run incidents** — ENV_CONCURRENCY=16 + the host-shared port reservation file (commit `7c994423`) kept this run completely retry-free, in stark contrast to C2.
