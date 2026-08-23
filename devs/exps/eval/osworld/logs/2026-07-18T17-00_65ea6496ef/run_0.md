# osworld @ 2026-07-18T17-00_65ea6496ef · run_0

- **Commit**: `4694f38a40` — `Update MiniWoB eval run_0 snapshot`
- **Host / GPUs**: `gpublaze` / `0,3`
- **Artifacts**: `.exps/eval/osworld/2026-07-18T17-00_65ea6496ef/run_0/`
- **Started**: `2026-07-17 17:46 PDT`
- **Last updated**: `2026-07-18 08:58 PDT`

## Results

Sorted by MER desc. Running progress table; rows remain marked not-started until each model produces a summary.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3.5-27B` | 325/325 | 0.4619 |
| `Qwen/Qwen3.5-9B` | 325/325 | 0.3807 |
| `Qwen/Qwen3-VL-32B-Instruct` | 325/325 | 0.3583 |
| `Qwen/Qwen3-VL-8B-Instruct` | 325/325 | 0.3424 |
| `Qwen/Qwen3.5-4B` | 325/325 | 0.3033 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 325/325 | 0.2943 |
| `Qwen/Qwen3-VL-4B-Instruct` | 325/325 | 0.2907 |
| `Qwen/Qwen3-VL-2B-Instruct` | 325/325 | 0.2048 |
| `OpenGVLab/ScaleCUA-7B` | 325/325 | 0.1503 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 325/325 | 0.1474 |
| `Qwen/Qwen3.5-2B` | 325/325 | 0.1071 |
| ⚠️ `meituan/EvoCUA-8B-20260105` | _**324/325**_ | _**0.4281**_ |

## Highlights

- `Qwen/Qwen3-VL-2B-Instruct` finished cleanly at 325/325 with MER 0.2048.
- `Qwen/Qwen3-VL-4B-Instruct` finished cleanly at 325/325 with MER 0.2907.
- `Qwen/Qwen3-VL-8B-Instruct` finished cleanly at 325/325 with MER 0.3424.
- `Qwen/Qwen3.5-2B` finished cleanly at 325/325 with MER 0.1071.
- `Qwen/Qwen3.5-4B` finished cleanly at 325/325 with MER 0.3033.
- `Qwen/Qwen3.5-9B` finished cleanly at 325/325 with MER 0.3807.
- `Qwen/Qwen3.5-27B` finished cleanly at 325/325 with MER 0.4619 after a resume pass recovered its two initially invalid samples.
- `Qwen/Qwen3-VL-32B-Instruct` finished cleanly at 325/325 with MER 0.3583.
- `ByteDance-Seed/UI-TARS-7B-DPO` finished cleanly at 325/325 with MER 0.1474.
- `ByteDance-Seed/UI-TARS-1.5-7B` finished cleanly at 325/325 with MER 0.2943.
- `OpenGVLab/ScaleCUA-7B` finished cleanly at 325/325 with MER 0.1503.
- ⚠️ `meituan/EvoCUA-8B-20260105` remains at 324/325 with MER 0.4281 after an additional resume pass; the final invalid task did not recover.
- `xlangai/OpenCUA-7B` is listed for matrix visibility, but is not scheduled initially because the eval README marks it HF-backend-only and this `run.sh` only auto-starts SGLang-backed agents.
