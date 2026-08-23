# lite.osworld · qwen3_vl_2b @ 2026-04-30T01-21_e93ba3f8 · run_0

- **Recipe**: `devs/exps/train/lite.osworld/qwen3_vl_2b/AGENTS.md`
- **Commit**: `e93ba3f8` — `Merge refactor: lite.X env_ids + : → @ registry sep + Qwen3VLHistoryProtocol`
- **Host / GPUs**: `gpublaze` / `2,3`
- **Container**: `lite.slime-train-lite.osworld-qwen3_vl_2b` (started 2026-04-30 ~02:00 PDT)
- **wandb** (skip mid-run failed-restart URLs — multiple GRPO retries; canonical run = the one whose ckpts persist):
  - `sft`: `https://wandb.ai/asap-zzhou/cua-lite-sft/runs/p0o5jctl`
  - `grpo`: pending — multiple attempts (1uic6nzc / 874ufh5p / fyof6ypx / t18v3664 / w1gabj4a). Pick the run that owns the final iter_19 once the retry loop terminates.
- **Artifacts**: `.exps/train/lite.osworld/qwen3_vl_2b/2026-04-30T01-21_e93ba3f8/run_0/`
- **Started**: `2026-04-30 02:48 PDT` (first GRPO eval_0 at 09:55 PDT after 8B teacher ramp)
- **Last updated**: `2026-04-30 22:08 PDT`
- **Notes**: Wave-1 C2. **`done` (stopped at iter_14, +1 independent eval pass)**. Pipeline ran attempts 1–4 with heavy lite.osworld task-level noise (rootless docker port collisions, gimp eval timeouts, container-API stalls). Real bug found mid-run: `asyncio.wait` could hang forever on a single stuck env container; fix in commit `a985ea6c` (rollout stall timeout). attempt 4 was killed before iter_19 to make room for an independent deterministic eval of base + sft iter_35 + grpo iter_4/9/14.

## Stages run

| Stage | Status | Output |
|---|---|---|
| teacher_rollout | `done` | 256 trajectories pinned to `teacher_pool.parquet` |
| sft             | `done iter_35` | 2 epochs distill; iter_35 picked as GRPO warm-start |
| grpo            | `stopped at iter_14` | 3 ckpts saved (iter_4/9/14, latest mtime 15:06 PDT). attempt 4 killed before iter_19 to free GPU for `eval_independent`. |
| eval_independent | `done` | base + sft iter_35 + grpo iter_4/9/14, all on the same `random.seed(42)` 64-task subset, T=0. Artifacts under `eval_independent/<name>/Qwen_Qwen3-VL-2B-Instruct/summary.json`. |

## Eval results

`grpo_pool_50_50.parquet` = 148 train tasks (74 anchor success + 74 random); eval = 64 random tasks (no anchor bias). The independent eval re-uses the same 64-task subset (same seed) so values are apples-to-apples.

| Ckpt | Eval set | Finished | Mean episode return | Δ vs base |
|---|---|---|---|---|
| base `Qwen/Qwen3-VL-2B-Instruct` | `lite.osworld` 64-task | 64/64 | **10.94%** | — |
| sft iter_35                      | `lite.osworld` 64-task | 64/64 | **6.25%**  | **−4.69pp** ⚠️ |
| grpo iter_4                      | `lite.osworld` 64-task | 64/64 | **7.81%**  | −3.13pp |
| grpo iter_9                      | `lite.osworld` 64-task | 64/64 | **9.38%**  | −1.56pp |
| grpo iter_14                     | `lite.osworld` 64-task | 64/64 | **12.50%** | **+1.56pp** |

(In-training eval values from GRPO Step 6 stdout are also recorded in `logs/step6_grpo.log`; they correspond to the same model weights but were sampled inside the training loop. The table above is the deterministic re-eval that should be cited.)

## Highlights

- **G1 fail**: SFT iter_35 lost 4.7pp vs base. The current pipeline at `NUM_EPOCH=2` undertrains SFT on this env — a captured 5-epoch run on the same env crossed base only at 2 epochs (+43%) and peaked at 4 epochs (+57%). 2 epochs is borderline and on this run landed before crossing.
- **GRPO climbed out of the SFT hole**: 6.25 → 7.81 → 9.38 → 12.50 monotonic; +1.56pp over base at iter_14. Real signal (3 monotonic steps, mixed-group ratio climbed 20%→40% in attempt 3 logs), but the apparent +6pp from in-training eval was mostly GRPO recovering from a regressed warm-start, not net learning above base.
- **Acceptance gate update**: The recipe README now treats SFT-must-beat-base as a hard pipeline gate (G1 in `## Acceptance gates`). Pipelines should refuse to start GRPO when SFT < base + 5pp; the recipe's recovery options include re-running SFT at NUM_EPOCH=5 or skipping SFT and starting GRPO from base.
- **Mid-run rollout fix**: a hung env container kept `asyncio.wait(FIRST_COMPLETED)` blocked for 2h45m before manual kill. Fixed by adding `ROLLOUT_STALL_TIMEOUT_S=1800` guard at all 4 wait sites in `lite/train/rollout/engine.py` (commit `a985ea6c`). Future retries will surface as rc≠0 within 30 min instead of silently stalling.
- iter_4/9/14 ckpt **mtimes shown above reflect attempt 3 saves** (latest 15:06 PDT). attempt 4 may have begun overwriting them before being killed; the eval used the on-disk ckpt at the time it ran (22:00–22:05 PDT).
