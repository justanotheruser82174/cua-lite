# lite.osworld @ 2026-07-04T23-38_bc2a8585ec · run_0

- **Commit**: `bc2a8585ec` — `teaser: use the standard lite.gym cursor in the animation`; appended `gpt-5.5` from `8ad4aaa03c` — `Correct WebHarbor WebVoyager run_0`
- **Host / GPUs**: `gpublaze` / `6-7`
- **Artifacts**: `.exps/eval/lite.osworld/2026-07-04T23-38_bc2a8585ec/run_0/`
- **Rerun artifacts**: `.exps/eval/lite.osworld/2026-07-06T12-30_8ffce97029/run_0/`
- **Started**: `2026-07-06 01:41 PDT`
- **Last updated**: `2026-07-09 11:14 PDT`
- **Notes**: env server `http://localhost:30100`, token `lhr`; `HF_HOME=/srv/share/huggingface` first, fallback to user cache if needed. Current launcher filter reports **321** eligible tasks (`369 -> 321`). MAI-UI and UI-Voyager were requested but have no lite.osworld config/launcher in this checkout; OpenCUA remains HF-backend-only per the campaign spec. `gpt-5.5` was appended from GPT API artifacts at `.exps/eval/lite.osworld/2026-07-09T06-01_8ad4aaa03c/run_0/gpt-5.5/`; first pass used `EVAL_CONCURRENCY=4`, then resumed with `EVAL_CONCURRENCY=8`.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-5.5` | 321/321 | 0.4766 |
| `Qwen/Qwen3.5-27B` | 321/321 | 0.3053 |
| `meituan/EvoCUA-8B-20260105` | 321/321 | 0.3022 |
| `Qwen/Qwen3.5-9B` | 321/321 | 0.2555 |
| `Qwen/Qwen3-VL-32B-Instruct` | 321/321 | 0.2212 |
| `Qwen/Qwen3.5-4B` | 321/321 | 0.2112 |
| `Qwen/Qwen3-VL-8B-Instruct` | 321/321 | 0.2150 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 321/321 | 0.1931 |
| `Qwen/Qwen3-VL-4B-Instruct` | 321/321 | 0.1869 |
| `Qwen/Qwen3-VL-2B-Instruct` | 321/321 | 0.1215 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 321/321 | 0.1121 |
| `OpenGVLab/ScaleCUA-7B` | 321/321 | 0.0997 |
| `xlangai/OpenCUA-7B` | 321/321 | 0.0997 |
| `Qwen/Qwen3.5-2B` | 321/321 | 0.0841 |

## Highlights

- `Qwen3-VL-{2B,4B,8B,32B}`, `Qwen3.5-2B`, `UI-TARS-{7B-DPO,1.5-7B}`, and `ScaleCUA-7B` completed after the initial run plus automatic mop-up attempts; all reached 321/321.
- Current completed ranking: `Qwen3-VL-32B` (0.2212) > `Qwen3-VL-8B` (0.2150) > `UI-TARS-1.5-7B` (0.1931) > `Qwen3-VL-4B` (0.1869) > `Qwen3-VL-2B` (0.1215) > `UI-TARS-7B-DPO` (0.1121) > `ScaleCUA-7B` (0.0997) > `Qwen3.5-2B` (0.0841) > `Qwen3.5-27B` (0.0685).
- `Qwen3.5-27B` was fully rerun after the earlier incomplete `319/321` result. The rerun initially reached 319/321 at 0.0690; a resume pass with `--max-attempts 10` reran the two missing VS Code samples and produced a complete 321/321 summary at 0.0685. The earlier 319/321 result is superseded and should not be used.
- `Qwen3.5-27B` remains anomalously low even after a complete rerun; treat this as suspect pending investigation of the `qwen3_5` agent/protocol/model-serving path rather than a normal scaling result.
- `Qwen3.5-4B` reached max attempts and plateaued at 320/321; remaining invalid task: `osworld_vs_code_c6bf789c`.
- `Qwen3.5-9B` reached max attempts and plateaued at 319/321; remaining invalid tasks: `osworld_vs_code_9d425400`, `osworld_vs_code_e2b5e914`.
- `EvoCUA-8B-20260105` reached max attempts and plateaued at 319/321; remaining invalid tasks: `osworld_libreoffice_calc_1334ca3e`, `osworld_vs_code_53ad5833`.
- `xlangai/OpenCUA-7B` not started — HF-backend-only in the campaign spec; `scripts/rollout.py` currently auto-starts SGLang.
- `Tongyi-MAI/MAI-UI-2B` not started — no `scripts/configs/mai_ui/default/lite.osworld.yaml` and no lite.osworld `run.sh` case in this checkout.
- `MarsXL/UI-Voyager` not started — no `scripts/configs/ui_voyager/default/lite.osworld.yaml` and no lite.osworld `run.sh` case in this checkout.
- `gpt-5.5` completed all 321 filtered Lite OSWorld eval tasks at 0.4766 after one resume pass.
