# osworld_g @ 2026-04-28T21-31_785df232 · run_0

- **Commit**: `785df232`. Pipeline-relevant change vs prior `ff51fcf`: `e32c7948` per-family `grounding.point` harness — env now declares `task_type="grounding.point"` (was `"action"`), routing to per-family `:grounding.point` adapters.
- **Host / GPUs**: `gpublaze` / 0-3.
- **Artifacts**: `.exps/eval/osworld_g/2026-04-28T21-31_785df232/run_0/`
- **Started / Last updated**: `2026-04-28 21:45 / 22:18 UTC` (~33 min wall)
- **Notes**: 510 / 564 (54 `box_type=refusal` filtered). Matrix adds `Tongyi-MAI/MAI-UI-{2,8}B` (run.sh `case` updated).

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3.5-27B`              | 510/510 | **0.7392** |
| `Qwen/Qwen3-VL-32B-Instruct`    | 510/510 | 0.7098 |
| `Tongyi-MAI/MAI-UI-8B`          | 510/510 | 0.6941 |
| `Qwen/Qwen3.5-9B`               | 510/510 | 0.6882 |
| `Qwen/Qwen3-VL-4B-Instruct`     | 510/510 | 0.6431 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 510/510 | 0.6392 |
| `Qwen/Qwen3.5-4B`               | 510/510 | 0.6176 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 510/510 | 0.5922 |
| `Tongyi-MAI/MAI-UI-2B`          | 510/510 | 0.5824 |
| `meituan/EvoCUA-8B-20260105`    | 510/510 | 0.5569 |
| `Qwen/Qwen3.5-2B`               | 510/510 | 0.5529 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 510/510 | 0.5020 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 510/510 | 0.4765 |

## Highlights

- Qwen3.5-27B retakes #1 (0.7392), 3pp ahead of Qwen3-VL-32B (vs near-tie at `ff51fcf`).
- Qwen3.5 family monotonic: 27B > 9B > 4B > 2B. Qwen3-VL flatter, 4B≈8B at ~0.64.
- MAI-UI-8B debuts at #3 (0.694), beats both 9B-class peers; 2B variant beats Qwen3.5-2B and Qwen3-VL-2B.
- UI-TARS-1.5-7B (0.592) >> UI-TARS-7B-DPO (0.477); DPO is matrix floor.
- vs `ff51fcf`: Qwen3.5 family +5–7pp; Qwen3-VL barely moves. Driver is the `grounding.point` per-family routing benefiting Qwen3.5's function-calling pretraining shape.

## Breakdown

Two-step regen: `reaggregate_breakdown.py --axes paper_category,box_type,GUI_types .exps/eval/osworld_g/2026-04-28T21-31_785df232/run_0/` → `render_breakdown.py .exps/eval/osworld_g/2026-04-28T21-31_785df232/run_0/`. The wide `GUI_types` axis (33 columns) is omitted here — re-run the renderer if needed.

### Breakdown — by `paper_category`

| Model | `element_recognition` | `fine_grained_manipulation` | `layout_understanding` | `text_matching` | Avg |
|---|---:|---:|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.7810 | 0.6364 | 0.7531 | 0.8208 | 0.7634 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.7353 | 0.5985 | 0.7531 | 0.8208 | 0.7426 |
| `Tongyi-MAI_MAI-UI-8B` | 0.7320 | 0.5985 | 0.7322 | 0.8042 | 0.7317 |
| `Qwen_Qwen3.5-9B` | 0.7190 | 0.5606 | 0.7322 | 0.8083 | 0.7230 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.6438 | 0.5758 | 0.6653 | 0.7875 | 0.6772 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.6699 | 0.5303 | 0.6485 | 0.7833 | 0.6739 |
| `Qwen_Qwen3.5-4B` | 0.6569 | 0.4848 | 0.6736 | 0.7417 | 0.6587 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.5980 | 0.5379 | 0.5941 | 0.7583 | 0.6303 |
| `Tongyi-MAI_MAI-UI-2B` | 0.6078 | 0.4545 | 0.6276 | 0.6917 | 0.6129 |
| `meituan_EvoCUA-8B-20260105` | 0.5458 | 0.5000 | 0.5690 | 0.7625 | 0.6020 |
| `Qwen_Qwen3.5-2B` | 0.6111 | 0.4015 | 0.6192 | 0.6458 | 0.5921 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.5131 | 0.4167 | 0.5439 | 0.6375 | 0.5398 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4837 | 0.4015 | 0.4979 | 0.6167 | 0.5104 |

### Breakdown — by `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.7255 | 0.9000 | 0.7392 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.7000 | 0.8250 | 0.7098 |
| `Tongyi-MAI_MAI-UI-8B` | 0.6787 | 0.8750 | 0.6941 |
| `Qwen_Qwen3.5-9B` | 0.6766 | 0.8250 | 0.6882 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.6319 | 0.7750 | 0.6431 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.6234 | 0.8250 | 0.6392 |
| `Qwen_Qwen3.5-4B` | 0.6000 | 0.8250 | 0.6176 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.5766 | 0.7750 | 0.5922 |
| `Tongyi-MAI_MAI-UI-2B` | 0.5617 | 0.8250 | 0.5824 |
| `meituan_EvoCUA-8B-20260105` | 0.5362 | 0.8000 | 0.5569 |
| `Qwen_Qwen3.5-2B` | 0.5404 | 0.7000 | 0.5529 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.4957 | 0.5750 | 0.5020 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4553 | 0.7250 | 0.4765 |
