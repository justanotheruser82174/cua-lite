# mobileworld @ 2026-07-10T15-40_651af22040 · run_0

- **Commit**: `651af22040` — `add gpt-5.6-sol`
- **Host / GPUs**: `gpublaze` / `0`
- **Artifacts**: `.exps/eval/mobileworld/2026-07-10T15-40_651af22040/run_0/`
- **Started**: `2026-07-10 19:17 PDT`
- **Last updated**: `2026-07-11 21:19 PDT`
- **Notes**: consolidated snapshot from the artifact tree above. Rows marked ` are carry-forward artifacts copied into this commit directory; their MER cells show the original run commit from `run_info.txt`. Current-commit rows are `Qwen/Qwen3.5-27B`, `stepfun-ai/GELab-Zero-4B-preview`, and `MarsXL/UI-Voyager`. Concurrency varied by runner (`1`, `4`, `8`, or `32`); all rows use `mobileworld` eval split with `step_timeout=240`.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-5.5` | 161/161 | 0.5342 (@bdb13593aa) |
| `Tongyi-MAI/MAI-UI-8B` | 161/161 | 0.2422 (@d982edcfb3) |
| `Qwen/Qwen3.5-27B` | 161/161 | 0.1304 |
| `Qwen/Qwen3-VL-32B-Instruct` | 161/161 | 0.1242 (@d982edcfb3) |
| `stepfun-ai/GELab-Zero-4B-preview` | 161/161 | 0.0932 |
| `Qwen/Qwen3.5-9B` | 161/161 | 0.0870 (@d6683dd519) |
| `Tongyi-MAI/MAI-UI-2B` | 161/161 | 0.0807 (@d982edcfb3) |
| `Qwen/Qwen3.5-4B` | 161/161 | 0.0745 (@6b023d0d7a) |
| `Qwen/Qwen3-VL-8B-Instruct` | 161/161 | 0.0683 (@b05b2b8104) |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 161/161 | 0.0621 (@d982edcfb3) |
| `Qwen/Qwen3-VL-4B-Instruct` | 161/161 | 0.0621 (@73bc9a9bf9) |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 161/161 | 0.0559 (@d982edcfb3) |
| `Qwen/Qwen3-VL-2B-Instruct` | 161/161 | 0.0559 (@8abf32a0de) |
| `Qwen/Qwen3.5-2B` | 161/161 | 0.0559 (@032cfe7b67) |
| `MarsXL/UI-Voyager` | 161/161 | 0.0186 |

## Highlights

- All summarized models are complete at `161/161`; there are no partial or not-started rows in this artifact tree.
- `gpt-5.5` is the clear top result at `0.5342`, more than double the best local/open model in this snapshot.
- Among local/open models, `Tongyi-MAI/MAI-UI-8B` leads at `0.2422`, followed by `Qwen/Qwen3.5-27B` at `0.1304` and `Qwen/Qwen3-VL-32B-Instruct` at `0.1242`.
- Current-commit additions are complete: `Qwen/Qwen3.5-27B` (`0.1304`), `stepfun-ai/GELab-Zero-4B-preview` (`0.0932`), and `MarsXL/UI-Voyager` (`0.0186`).
- Smaller Qwen and UI-TARS models cluster around `0.0559` to `0.0745`; `MarsXL/UI-Voyager` trails this set on Mobileworld.
