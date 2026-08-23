# androidworld @ 2026-04-28T21-31_785df232 · run_0

- **Commit**: `785df232`. Pipeline-relevant change vs prior `8b9d4bd`: `1bdef611` Qwen3.5 mobile `left_click`→`click` alias fix.
- **Host / GPUs**: `gpublaze` / 0-3.
- **Artifacts**: `.exps/eval/androidworld/2026-04-28T21-31_785df232/run_0/`
- **Started / Last updated**: `2026-04-29 01:28 / 14:36 UTC`
- **Notes**: Subset re-run per user — Qwen3.5 {2,4,9}B re-evaluated at this commit (27B excluded from subset). Other 10 matrix models carried over from `8b9d4bd` (📌 marker, prefix `(@8b9d4bd)` on MER) since the only pipeline change between them, `1bdef611` Qwen3.5-mobile-alias fix, doesn't affect non-Qwen3.5 models — see README "Carry-forward convention" for the rule. Final run config (now in `run.sh`): `--concurrency 4`, `--env-kwargs '{"step_timeout": 180}'`, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for 4B (4B/androidworld OOM-prone — see README "Known Qwen3.5 mobile failures").

## Results

📌 = result carried forward from earlier commit (no pipeline change affecting that model since).

| Model | Finished | Mean episode return |
|---|---|---|
| 📌 `MarsXL/UI-Voyager` | 116/116 | 0.6810 (@8b9d4bd) |
| `Qwen/Qwen3.5-9B` | 116/116 | **0.5733** |
| 📌 `Qwen/Qwen3-VL-32B-Instruct` | 116/116 | 0.5474 (@8b9d4bd) |
| 📌 `Tongyi-MAI/MAI-UI-8B` | 116/116 | 0.5345 (@8b9d4bd) |
| `Qwen/Qwen3.5-4B` | 116/116 | 0.5216 |
| 📌 `Qwen/Qwen3-VL-8B-Instruct` | 116/116 | 0.4784 (@8b9d4bd) |
| 📌 `stepfun-ai/GELab-Zero-4B-preview` | 116/116 | 0.4440 (@8b9d4bd) |
| 📌 `Tongyi-MAI/MAI-UI-2B` | 116/116 | 0.4138 (@8b9d4bd) |
| 📌 `Qwen/Qwen3-VL-2B-Instruct` | 116/116 | 0.3793 (@8b9d4bd) |
| 📌 `Qwen/Qwen3-VL-4B-Instruct` | 116/116 | 0.3448 (@8b9d4bd) |
| `Qwen/Qwen3.5-2B` | 116/116 | 0.2802 |
| 📌 `ByteDance-Seed/UI-TARS-1.5-7B` | 116/116 | 0.2112 (@8b9d4bd) |
| 📌 `ByteDance-Seed/UI-TARS-7B-DPO` | 116/116 | 0.1250 (@8b9d4bd) |
| ⚠️ `Qwen/Qwen3.5-27B` | _**0/116**_ | _**—**_ |

## Highlights

- 🎯 **9B +40.95pp** vs prior 0.1638 (73.6% leakage). Apples-to-apples 116/116: 52 fail→pass, 4 pass→fail, 60 same. Jumps to #2 in the matrix (UI-Voyager 0.681 > 9B 0.573 > Qwen3-VL-32B 0.547).
- **4B +4.74pp** (15.5% leakage). 14 fail→pass, 8 pass→fail.
- **2B unchanged** (0.2876 → 0.2876 apples-to-apples on 113 overlap; 10 up / 10 down). Prior diagnosis correct: 0% leakage → no fix gain.
- Recovering 2B's full 116/116 needed `step_timeout=180s` override (default 30s was firing on 2B's slow steps under host contention) — see README "Known Qwen3.5 mobile failures".
