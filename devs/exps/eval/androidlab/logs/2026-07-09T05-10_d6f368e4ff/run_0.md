# androidlab @ 2026-07-04T20-37_82b70d153d · run_0

- **Commit**: `82b70d153d` — `update browsergym.miniwob figures`; appended `gpt-5.5` from `d6f368e4ff` — `Correct WebHarbor WebVoyager run_0`
- **Host / GPUs**: `gpublaze` / 6-7.
- **Artifacts**: `.exps/eval/androidlab/2026-07-04T20-37_82b70d153d/run_0/`
- **Started**: `2026-07-05 08:11 UTC`
- **Last updated**: `2026-07-09 07:43 PDT`
- **Notes**: Env-server `http://localhost:30100` with token `lhr`; androidlab exposes 138 eval tasks. Runner uses `--concurrency 8` and `step_timeout=180`. Shared HF cache is used when present; otherwise local `~/.cache/huggingface` is used. Initial Qwen3-VL 2B/4B attempts reached sglang but all env creates failed with HTTP 501 because `cua-lite/androidlab:latest` was missing/stale; the image was rebuilt and those artifact dirs were restarted. `gpt-5.5` was appended from GPT API artifacts at `.exps/eval/androidlab/2026-07-09T05-10_d6f368e4ff/run_0/gpt-5.5/` with `EVAL_CONCURRENCY=4`.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-5.5` | 138/138 | **0.4565** |
| `Qwen/Qwen3-VL-32B-Instruct` | 138/138 | 0.4130 |
| `Qwen/Qwen3.5-27B` | 138/138 | 0.3986 |
| `Qwen/Qwen3-VL-4B-Instruct` | 138/138 | 0.3913 |
| `Qwen/Qwen3-VL-8B-Instruct` | 138/138 | 0.3913 |
| `stepfun-ai/GELab-Zero-4B-preview` | 138/138 | 0.3841 |
| `Tongyi-MAI/MAI-UI-2B` | 138/138 | 0.3696 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 138/138 | 0.3551 |
| `Qwen/Qwen3.5-4B` | 138/138 | 0.3478 |
| `Qwen/Qwen3.5-2B` | 138/138 | 0.3333 |
| `Qwen/Qwen3.5-9B` | 138/138 | 0.3261 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 138/138 | 0.3261 |
| `MarsXL/UI-Voyager` | 138/138 | 0.2754 |
| `Qwen/Qwen3-VL-2B-Instruct` | 138/138 | 0.2319 |

## Highlights

- **Qwen3-VL scales early, then plateaus**: 2B -> 4B gives the large jump (+15.9pp, 0.2319 -> 0.3913), 4B -> 8B is flat, and 8B -> 32B adds only +2.2pp. AndroidLab looks bottlenecked by policy/task robustness after the 4B tier, not raw visual-language capacity.
- **Qwen3.5 is non-monotonic until 27B**: 2B/4B/9B cluster in a narrow 2.2pp band (0.3261-0.3478), while 27B reaches 0.3986 and lands second overall. The 9B drop relative to 4B suggests size alone is not reliable on this env without the larger 27B jump.
- **Top models are tightly packed**: the best four results span only 2.9pp (0.3841-0.4130). `Qwen3-VL-32B` leads, but `Qwen3.5-27B`, `Qwen3-VL-{4,8}B`, and GELab-Zero are all within a small eval-noise-sensitive band.
- **Specialist UI agents are competitive but not dominant**: GELab-Zero (0.3841) nearly matches Qwen3-VL 4B/8B, MAI-UI-2B (0.3696) beats both UI-TARS variants, and MarsXL/UI-Voyager trails the field at 0.2754.
- **Compared with the 2026-04-28 snapshot**, Qwen3-VL-4B is exactly stable at 0.3913, Qwen3-VL-8B improves +4.35pp, Qwen3-VL-32B slips -1.45pp, and MarsXL is essentially flat (-0.72pp). The main story is a reshuffled plateau, not a clean scaling-law curve.
- `gpt-5.5` completed all 138 AndroidLab tasks at 0.4565, above the previous top row in this snapshot.
