# osworld_g @ 2026-07-06T20-34_0180ceefd8 · run_0

- **Commit**: `0180ceefd8` — `eval: update lite.osworld run_0`; appended `gpt-5.5` from `d30ef912ca` — `Correct WebHarbor WebVoyager run_0`
- **Host / GPUs**: `gpublaze` / 6-7.
- **Artifacts**: `.exps/eval/osworld_g/2026-07-06T20-34_0180ceefd8/run_0/`
- **Started**: `2026-07-06 20:45 PDT`
- **Last updated**: `2026-07-09 03:27 PDT`
- **Notes**: 510 / 564 tasks after dropping 54 `exclude_reason=refusal` tasks. Running grounding-matrix models on env server `http://localhost:30100` with token `lhr`; `HF_HOME=/srv/share/huggingface` unless a model is missing from the shared cache. Fara was appended from the later fixed-config artifact `.exps/eval/osworld_g/2026-07-07T23-25_b05b2b8104/run_0/microsoft_Fara-7B/` (`b05b2b8104`). `gpt-5.5` was appended from GPT API artifacts at `.exps/eval/osworld_g/2026-07-09T03-09_d30ef912ca/run_0/gpt-5.5/` with `EVAL_CONCURRENCY=16`.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-5.5` | 510/510 | 0.7667 |
| `Qwen/Qwen3.5-27B`  | 510/510 | 0.7294 |
| `Qwen/Qwen3-VL-32B-Instruct`       | 510/510 | 0.7216 |
| `Qwen/Qwen3.5-9B`                  | 510/510 | 0.6941 |
| `Tongyi-MAI/MAI-UI-8B`             | 510/510 | 0.6882 |
| `Qwen/Qwen3-VL-4B-Instruct`        | 510/510 | 0.6549 |
| `Qwen/Qwen3-VL-8B-Instruct`        | 510/510 | 0.6431 |
| `Qwen/Qwen3.5-4B`                  | 510/510 | 0.6353 |
| `ByteDance-Seed/UI-TARS-1.5-7B`    | 510/510 | 0.6000 |
| `Tongyi-MAI/MAI-UI-2B`             | 510/510 | 0.5824 |
| `meituan/EvoCUA-8B-20260105`       | 510/510 | 0.5745 |
| `Qwen/Qwen3.5-2B`                  | 510/510 | 0.5569 |
| `Qwen/Qwen3-VL-2B-Instruct`        | 510/510 | 0.5275 |
| `microsoft/Fara-7B`                | 510/510 | 0.5020 |
| `ByteDance-Seed/UI-TARS-7B-DPO`    | 510/510 | 0.4843 |

## Highlights

- Qwen3-VL-4B completed at 0.6549; Qwen3-VL-2B completed at 0.5275.
- Qwen3-VL-8B completed at 0.6431; Qwen3.5-2B completed at 0.5569.
- Qwen3.5-9B completed at 0.6941; Qwen3.5-4B completed at 0.6353.
- UI-TARS-1.5-7B completed at 0.6000; UI-TARS-7B-DPO completed at 0.4843.
- MAI-UI-2B completed at 0.5824.
- EvoCUA-8B completed at 0.5745.
- MAI-UI-8B completed at 0.6882.
- Qwen3-VL-32B completed at 0.7216.
- Qwen3.5-27B rerun with the sglang GDN fix completed at 0.7294 (`Qwen_Qwen3.5-27B__sglang-gdnfix`). The earlier 0.1667 artifact is retained as the pre-fix run.
- Fara-7B completed at 0.5020 from the later `b05b2b8104` artifact, after removing agent-side screenshot stretching and serving via `cua-lite/Fara-7B`.
- `gpt-5.5` completed all 510 filtered OSWorld-G tasks at 0.7667, above the previous top row in this snapshot.
- All listed models completed.

## Breakdown

Two-step regen: `reaggregate_breakdown.py --axes paper_category,box_type,GUI_types .exps/eval/osworld_g/2026-07-06T20-34_0180ceefd8/run_0/` -> `render_breakdown.py .exps/eval/osworld_g/2026-07-06T20-34_0180ceefd8/run_0/`. The wide `GUI_types` axis (33 columns) is omitted here; rerun the renderer if needed.

### Breakdown — by `paper_category`

| Model | `element_recognition` | `fine_grained_manipulation` | `layout_understanding` | `text_matching` | Avg |
|---|---:|---:|---:|---:|---:|
| `gpt-5.5` | 0.7941 | 0.6742 | 0.7866 | 0.8250 | 0.7830 |
| `Qwen_Qwen3.5-27B__sglang-gdnfix` | 0.7745 | 0.6288 | 0.7531 | 0.8042 | 0.7557 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.7484 | 0.6061 | 0.7573 | 0.8375 | 0.7535 |
| `Qwen_Qwen3.5-9B` | 0.7353 | 0.5530 | 0.7490 | 0.8083 | 0.7317 |
| `Tongyi-MAI_MAI-UI-8B` | 0.7190 | 0.5909 | 0.7406 | 0.8083 | 0.7296 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.6830 | 0.5530 | 0.6569 | 0.7875 | 0.6848 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.6438 | 0.5909 | 0.6611 | 0.7958 | 0.6805 |
| `Qwen_Qwen3.5-4B` | 0.6634 | 0.5303 | 0.6653 | 0.7625 | 0.6707 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.6013 | 0.5455 | 0.5900 | 0.7667 | 0.6336 |
| `meituan_EvoCUA-8B-20260105` | 0.5588 | 0.5227 | 0.5858 | 0.7792 | 0.6183 |
| `Tongyi-MAI_MAI-UI-2B` | 0.6111 | 0.4318 | 0.6360 | 0.6917 | 0.6129 |
| `Qwen_Qwen3.5-2B` | 0.6013 | 0.4167 | 0.6192 | 0.6625 | 0.5954 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.5458 | 0.4318 | 0.5858 | 0.6583 | 0.5692 |
| `microsoft_Fara-7B` | 0.5196 | 0.4242 | 0.5188 | 0.6458 | 0.5387 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4869 | 0.4091 | 0.5146 | 0.6125 | 0.5158 |
| `Qwen_Qwen3.5-27B` | 0.1765 | 0.0682 | 0.2301 | 0.2250 | 0.1876 |

### Breakdown — by `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| `gpt-5.5` | 0.7553 | 0.9000 | 0.7667 |
| `Qwen_Qwen3.5-27B__sglang-gdnfix` | 0.7149 | 0.9000 | 0.7294 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.7128 | 0.8250 | 0.7216 |
| `Qwen_Qwen3.5-9B` | 0.6830 | 0.8250 | 0.6941 |
| `Tongyi-MAI_MAI-UI-8B` | 0.6745 | 0.8500 | 0.6882 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.6447 | 0.7750 | 0.6549 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.6277 | 0.8250 | 0.6431 |
| `Qwen_Qwen3.5-4B` | 0.6191 | 0.8250 | 0.6353 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.5851 | 0.7750 | 0.6000 |
| `Tongyi-MAI_MAI-UI-2B` | 0.5617 | 0.8250 | 0.5824 |
| `meituan_EvoCUA-8B-20260105` | 0.5532 | 0.8250 | 0.5745 |
| `Qwen_Qwen3.5-2B` | 0.5447 | 0.7000 | 0.5569 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.5149 | 0.6750 | 0.5275 |
| `microsoft_Fara-7B` | 0.4787 | 0.7750 | 0.5020 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4638 | 0.7250 | 0.4843 |
| `Qwen_Qwen3.5-27B` | 0.1638 | 0.2000 | 0.1667 |
