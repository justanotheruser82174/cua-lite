# osworld @ 2026-04-26T18-38_8b9d4bd · run_0

- **Commit**: `8b9d4bd46a86ec657bb19127d71aa8a43ac67317` — `eval: long-run runner + matrix wiring for the eval.md campaign` (with uncommitted patches: `lite/gym/envs/osworld/main.py` step_timeout 30s → 180s, `lite/utils/sglang.py` graceful-exit deadlock fix, `lite/utils/rollout.py` `--max-attempts` default 3 → 10).
- **Host / GPUs**: `gpublaze` / `0-3`
- **Notes**: spans 2026-04-26 → 04-27. Raw artifacts: `.exps/eval/osworld/2026-04-26T18-38_8b9d4bd/run_0/<slug>/`. Several mid-campaign mop-up cycles after step-timeout bumps (30s → 90s → 180s) and SGLang graceful-exit deadlock fixes. Eligible task count: **332** of 369 raw (37 excluded — 30 infeasible + 8 google_auth + 0 evaluator_bug; sets defined at `lite/gym/envs/osworld/main.py:480`).

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3.5-9B`               | 330/332 | 0.3172 |
| `Qwen/Qwen3-VL-32B-Instruct`    | 330/332 | 0.3088 |
| `meituan/EvoCUA-8B-20260105`    | 328/332 | 0.3059 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 330/332 | 0.2891 |
| `Qwen/Qwen3.5-4B`               | 329/332 | 0.2850 |
| `Qwen/Qwen3-VL-4B-Instruct`     | 330/332 | 0.2622 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 329/332 | 0.2438 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 330/332 | 0.1598 |
| `OpenGVLab/ScaleCUA-7B`         | 329/332 | 0.1500 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 331/332 | 0.1495 |
| `Qwen/Qwen3.5-2B`               | 330/332 | 0.1236 |
| ⚠️ `Qwen/Qwen3.5-27B`           | _**0/332**_ | _**—**_ |
| ⚠️ `xlangai/OpenCUA-7B`         | _**0/332**_ | _**—**_ |

## Highlights

- Top-3 dense: `Qwen3.5-9B` (0.3172) > `Qwen3-VL-32B` (0.3088) > `EvoCUA-8B` (0.3059). Qwen3.5-9B's long-context attention + mamba mix edges out a 32B VL model on this env.
- `Qwen3-VL` scaling: 32B (0.3088) > 8B (0.2891) > 4B (0.2622) > 2B (0.1598) — clean monotonic.
- `Qwen3.5` scaling: 9B (0.3172) > 4B (0.2850) >> 2B (0.1236) — also monotonic; 9B vs 32B gap (0.32 vs 0.31) is smaller than 9B vs 4B gap (0.32 vs 0.29).
- `UI-TARS-1.5-7B` (0.2438) > `UI-TARS-7B-DPO` (0.1495) — the 1.5 generation is a clear upgrade.
- `ScaleCUA-7B` (0.1500) underperforms most 7-9B baselines.
- `Qwen3.5-27B` not started — wave 3 (tp=4) was halted by user.
- `OpenCUA-7B` not started — locked to `backends=["hf"]`; rollout only auto-starts SGLang.

## Residual unfinished tasks

11 finished models cluster between 328-331/332. The unfinished tasks are concentrated on a small set of OSWorld task IDs whose evaluator is fragile in our setup:

| Task ID prefix | Models still failing | Failure mode |
|---|---|---|
| `0e5303d4` | 10/11 | `FileNotFoundError: cache/<task>/gold_lecture_slides_gold/lecture_slides` — gold-data file permanently missing on our filesystem. Effectively deterministic-broken (1/11 finished historically — likely a single download race that succeeded). |
| `ee9a3c83` | 7/11 | flaky — `step()` evaluator timeout / connect timeout |
| `185f29bd` | 3/11 | flaky — same family |
| `da46d875` | 4/11 | flaky — same family |
| `1de60575` | 2/11 | flaky — same family |
| `7e287123` | 1/11 (UI-TARS-1.5-7B only) | flaky |

Total residual = 31 task-runs across 11 models. Of those, **11 are `0e5303d4`** (one per model, deterministic). After excluding `0e5303d4`, ~20 are flaky-recoverable but recovery rate is low (~10-30% per attempt) — extra `--max-attempts` cycles produced 1-2 net recoveries per model in the 2026-04-27 mop-up wave.

## Mop-up history

This campaign ran a tight failure-restart loop on the same `EVAL_RUN_ID=2026-04-26`:

1. **Initial run** — 30s step_timeout → 9-12 timeouts/model from heavy file-comparison evaluators (`compare_pdfs`, `compare_archive`, `compare_epub`, `compare_table`).
2. **Bump to 90s** + relaunch → recovered ~5-6 tasks/model.
3. **Bump to 180s** + relaunch → recovered ~2 more tasks/model.
4. **SGLang graceful-exit deadlock fix** (`lite/utils/sglang.py`: close httpx client before SIGTERM, SIGKILL fallback after 30s) — eliminated end-of-run hangs that were blocking GPU release between cycles.
5. **`--max-attempts` 3 → 10** (`lite/utils/rollout.py:443`) and final relaunch on Qwen3-VL-32B + UI-TARS-7B-DPO + Qwen3-VL-{2B,4B,8B} + Qwen3.5-{2B,4B,9B} → recovered 1-2 more tasks (gain bottlenecked by deterministic `0e5303d4` failures consuming most of each retry round).
6. **Final state**: 11 models finished at 328-331/332. `_EVALUATOR_BUG_TASK_IDS` was kept empty since no task fails on every agent — `0e5303d4` came closest at 10/11 fail.
