# lite.osworld @ 2026-04-26T18-38_8b9d4bd · run_0

- **Commit**: `8b9d4bd46a86ec657bb19127d71aa8a43ac67317` — `eval: long-run runner + matrix wiring for the eval.md campaign`
- **Host / GPUs**: `gpublaze` / `0-3`
- **Notes**: spans 2026-04-26 → 04-27. Raw artifacts: `.exps/eval/lite.osworld/2026-04-26T18-38_8b9d4bd/run_0/<slug>/`. Several Qwen3.5/Qwen3-VL sglang graceful-exit deadlocks resolved via manual kill+resume during the run.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3.5-9B`               | 329/329 | 0.3131 |
| `meituan/EvoCUA-8B-20260105`    | 329/329 | 0.2979 |
| `Qwen/Qwen3-VL-32B-Instruct`    | 329/329 | 0.2888 |
| `Qwen/Qwen3-VL-4B-Instruct`     | 329/329 | 0.2553 |
| `Qwen/Qwen3.5-4B`               | 329/329 | 0.2523 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 329/329 | 0.2492 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 329/329 | 0.2097 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 329/329 | 0.1824 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 329/329 | 0.1763 |
| `OpenGVLab/ScaleCUA-7B`         | 329/329 | 0.1398 |
| `Qwen/Qwen3.5-2B`               | 329/329 | 0.1246 |
| ⚠️ `Qwen/Qwen3.5-27B`           | _**163/329**_ | _**0.4233**_ |
| ⚠️ `xlangai/OpenCUA-7B`         | _**0/329**_   | _**—**_       |

## Highlights

- `Qwen3.5-27B` is partial (163/329) — wave 3 halted by user. The 0.4233 figure is over the early subset only (likely easier-task bias); not directly comparable to the others.
- `OpenCUA-7B` not started — locked to `backends=["hf"]` in `lite/utils/agents.py`; `scripts/rollout.py` only auto-starts SGLang. Will run once an HF backend lands in rollout.
- Among fully-finished runs: `Qwen3.5-9B` (0.3131) > `EvoCUA-8B` (0.2979) > `Qwen3-VL-32B` (0.2888) — long-context (`history_n=100`) Qwen3.5 edges out the 32B Qwen3-VL on this env.
- `Qwen3-VL-{4B, 8B}` ≈ 0.25 — flat scaling between 4B and 8B at this checkpoint.
