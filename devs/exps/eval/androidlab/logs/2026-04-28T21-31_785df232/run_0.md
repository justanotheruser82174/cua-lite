# androidlab @ 2026-04-28T21-31_785df232 · run_0

- **Commit**: `785df232`. Pipeline-relevant change vs prior `8b9d4bd`: `1bdef611` Qwen3.5 mobile `left_click`→`click` alias fix.
- **Host / GPUs**: `gpublaze` / 0-3.
- **Artifacts**: `.exps/eval/androidlab/2026-04-28T21-31_785df232/run_0/`
- **Started / Last updated**: `2026-04-29 02:46 / 14:36 UTC`
- **Notes**: Subset re-run per user — Qwen3.5 {2,4,9}B re-evaluated at this commit (27B excluded). Other 10 matrix models carried over from `8b9d4bd` (📌, suffixed `(@8b9d4bd)`) — see README "Carry-forward convention". 9B at tp=2 (dp=2 on GPUs 0,3); 2B/4B at tp=1, concurrency=8. See README "Known Qwen3.5 mobile failures" + "Known docker timeouts".

## Results

📌 = result carried forward from earlier commit (no pipeline change affecting that model since).

| Model | Finished | Mean episode return |
|---|---|---|
| 📌 `Tongyi-MAI/MAI-UI-8B` | 138/138 | **0.5217 (@8b9d4bd)** |
| 📌 `Qwen/Qwen3-VL-32B-Instruct` | 138/138 | 0.4275 (@8b9d4bd) |
| 📌 `Qwen/Qwen3-VL-4B-Instruct` | 138/138 | 0.3913 (@8b9d4bd) |
| 📌 `ByteDance-Seed/UI-TARS-7B-DPO` | 138/138 | 0.3841 (@8b9d4bd) |
| 📌 `stepfun-ai/GELab-Zero-4B-preview` | 138/138 | 0.3768 (@8b9d4bd) |
| 📌 `Tongyi-MAI/MAI-UI-2B` | 138/138 | 0.3696 (@8b9d4bd) |
| 📌 `ByteDance-Seed/UI-TARS-1.5-7B` | 138/138 | 0.3551 (@8b9d4bd) |
| `Qwen/Qwen3.5-9B` | 138/138 | 0.3478 |
| `Qwen/Qwen3.5-4B` | 138/138 | 0.3478 |
| 📌 `Qwen/Qwen3-VL-8B-Instruct` | 138/138 | 0.3478 (@8b9d4bd) |
| `Qwen/Qwen3.5-2B` | 138/138 | 0.3333 |
| 📌 `Qwen/Qwen3-VL-2B-Instruct` | 138/138 | 0.2899 (@8b9d4bd) |
| 📌 `MarsXL/UI-Voyager` | 138/138 | 0.2826 (@8b9d4bd) |
| ⚠️ `Qwen/Qwen3.5-27B` | _**0/138**_ | _**—**_ |

## Highlights

- **4B +7.25pp** vs prior 0.2754 (35.8% leakage). 12 fail→pass, 2 pass→fail.
- **2B unchanged** (-0.7pp, within noise). Prior diagnosis correct: 0% leakage → no fix gain.
- **9B essentially flat** (-2.17pp): 5 up / 8 down on apples-to-apples. Prior 10% leakage was too small for fix to dominate variance. Headline 9B win is on `androidworld` (73.6% leakage there) — see that snapshot.
- 4B and 9B tie at 0.3478 — scaling flat on this env's task distribution.
