# screenspot_pro @ 2026-04-28T21-31_785df232 · run_0

- **Commit**: `785df232`. Pipeline-relevant change vs prior `ff51fcf`: same as sibling osworld_g — `e32c7948` per-family `grounding.point` harness.
- **Host / GPUs**: `gpublaze` / 0-3.
- **Artifacts**: `.exps/eval/screenspot_pro/2026-04-28T21-31_785df232/run_0/`
- **Started / Last updated**: `2026-04-28 21:53 / 2026-04-29 01:25 UTC` (~3.5 hr wall — 4K-resolution images dominate).
- **Notes**: 1581 tasks, no filter. Matrix adds `Tongyi-MAI/MAI-UI-{2,8}B`.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3.5-27B`              | 1581/1581 | **0.6863** |
| `Tongyi-MAI/MAI-UI-8B`          | 1581/1581 | 0.5996 |
| `Qwen/Qwen3-VL-4B-Instruct`     | 1581/1581 | 0.5769 |
| `Qwen/Qwen3-VL-32B-Instruct`    | 1581/1581 | 0.5693 |
| `Qwen/Qwen3.5-4B`               | 1581/1581 | 0.5579 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 1581/1581 | 0.5452 |
| `Qwen/Qwen3.5-2B`               | 1581/1581 | 0.5446 |
| `Qwen/Qwen3.5-9B`               | 1581/1581 | 0.5433 |
| `Tongyi-MAI/MAI-UI-2B`          | 1581/1581 | 0.5275 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 1581/1581 | 0.4921 |
| `meituan/EvoCUA-8B-20260105`    | 1581/1581 | 0.4314 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 1581/1581 | 0.3922 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 1581/1581 | 0.3251 |

## Highlights

- Qwen3.5-27B dominates (0.686), 9pp ahead of #2; +12.2pp vs `ff51fcf`. Driver is its 0.475 icon score vs everyone else < 0.34.
- MAI-UI-8B at #2 (0.600), beats Qwen3-VL-32B. Strongest non-flagship on both grounding envs.
- Qwen3-VL non-monotonic at 4B: 4B (0.577) ≈ 32B (0.569) > 8B (0.545) — same "4B punches up" pattern as `ff51fcf` but stronger.
- Qwen3.5 non-monotonic in middle: 27B >> 4B ≈ 2B ≈ 9B. 9B/4B reversal vs `ff51fcf` (variance under new routing).
- vs `ff51fcf`: Qwen3.5 family +3 to +14pp; EvoCUA-8B +7.2pp; Qwen3-VL barely moves. Same pattern as osworld_g.
- UI-TARS-7B-DPO is matrix floor (0.325), 17pp behind UI-TARS-1.5-7B.
- Icon/text gap large for everyone; Qwen3.5-27B's icon (0.475) is the standout.

## Breakdown

Two-step regen: `reaggregate_breakdown.py --axes group,ui_type,application .exps/eval/screenspot_pro/2026-04-28T21-31_785df232/run_0/` → `render_breakdown.py .exps/eval/screenspot_pro/2026-04-28T21-31_785df232/run_0/`. The 26-column `application` axis is omitted here. The 12-cell `group × ui_type` cross-product matches the paper's canonical layout — re-render via `--cross "group,ui_type"`.

### Breakdown — by `group`

| Model | `CAD` | `Creative` | `Dev` | `OS` | `Office` | `Scientific` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.6628 | 0.6070 | 0.6890 | 0.6990 | 0.8478 | 0.6575 | 0.6863 |
| `Tongyi-MAI_MAI-UI-8B` | 0.6015 | 0.5367 | 0.5385 | 0.5918 | 0.8043 | 0.5748 | 0.5996 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.4828 | 0.5103 | 0.5585 | 0.6020 | 0.7435 | 0.6142 | 0.5769 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.5287 | 0.5660 | 0.4916 | 0.4592 | 0.7609 | 0.6181 | 0.5693 |
| `Qwen_Qwen3.5-4B` | 0.4674 | 0.5015 | 0.5652 | 0.5612 | 0.7000 | 0.5866 | 0.5579 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.4943 | 0.4927 | 0.5084 | 0.5204 | 0.7087 | 0.5827 | 0.5452 |
| `Qwen_Qwen3.5-2B` | 0.4330 | 0.4985 | 0.4950 | 0.5663 | 0.6957 | 0.6260 | 0.5446 |
| `Qwen_Qwen3.5-9B` | 0.4789 | 0.4164 | 0.6120 | 0.5204 | 0.7000 | 0.5748 | 0.5433 |
| `Tongyi-MAI_MAI-UI-2B` | 0.3831 | 0.4809 | 0.5050 | 0.5459 | 0.7174 | 0.5787 | 0.5275 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.4598 | 0.4428 | 0.4348 | 0.3980 | 0.7043 | 0.5394 | 0.4921 |
| `meituan_EvoCUA-8B-20260105` | 0.2989 | 0.4047 | 0.4281 | 0.3827 | 0.6174 | 0.4764 | 0.4314 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.2989 | 0.3578 | 0.3679 | 0.4133 | 0.5739 | 0.3819 | 0.3922 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1916 | 0.3196 | 0.2876 | 0.2296 | 0.4913 | 0.4370 | 0.3251 |

### Breakdown — by `ui_type`

| Model | `icon` | `text` | Avg |
|---|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.4752 | 0.8168 | 0.6863 |
| `Tongyi-MAI_MAI-UI-8B` | 0.3361 | 0.7625 | 0.5996 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.3013 | 0.7472 | 0.5769 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.2815 | 0.7472 | 0.5693 |
| `Qwen_Qwen3.5-4B` | 0.2980 | 0.7185 | 0.5579 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.2219 | 0.7451 | 0.5452 |
| `Qwen_Qwen3.5-2B` | 0.3311 | 0.6766 | 0.5446 |
| `Qwen_Qwen3.5-9B` | 0.3195 | 0.6817 | 0.5433 |
| `Tongyi-MAI_MAI-UI-2B` | 0.2781 | 0.6817 | 0.5275 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.2119 | 0.6653 | 0.4921 |
| `meituan_EvoCUA-8B-20260105` | 0.1540 | 0.6029 | 0.4314 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.2020 | 0.5097 | 0.3922 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1060 | 0.4606 | 0.3251 |

### Breakdown — by `group` × `ui_type`

| Model | `CAD × icon` | `CAD × text` | `Creative × icon` | `Creative × text` | `Dev × icon` | `Dev × text` | `OS × icon` | `OS × text` | `Office × icon` | `Office × text` | `Scientific × icon` | `Scientific × text` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.3438 | 0.7665 | 0.3846 | 0.7677 | 0.5310 | 0.8377 | 0.5281 | 0.8411 | 0.7170 | 0.8870 | 0.4364 | 0.8264 | 0.6863 |
| `Tongyi-MAI_MAI-UI-8B` | 0.2969 | 0.7005 | 0.2727 | 0.7273 | 0.3103 | 0.7532 | 0.4270 | 0.7290 | 0.5660 | 0.8757 | 0.2909 | 0.7917 | 0.5996 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.2031 | 0.5736 | 0.2238 | 0.7172 | 0.3172 | 0.7857 | 0.3708 | 0.7944 | 0.4151 | 0.8418 | 0.3273 | 0.8333 | 0.5769 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.3281 | 0.5939 | 0.2657 | 0.7828 | 0.2069 | 0.7597 | 0.2247 | 0.6542 | 0.4717 | 0.8475 | 0.3273 | 0.8403 | 0.5693 |
| `Qwen_Qwen3.5-4B` | 0.1719 | 0.5635 | 0.2448 | 0.6869 | 0.3379 | 0.7792 | 0.3708 | 0.7196 | 0.3774 | 0.7966 | 0.2909 | 0.8125 | 0.5579 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.1719 | 0.5990 | 0.1608 | 0.7323 | 0.2000 | 0.7987 | 0.2472 | 0.7477 | 0.3019 | 0.8305 | 0.3000 | 0.7986 | 0.5452 |
| `Qwen_Qwen3.5-2B` | 0.2656 | 0.4873 | 0.2727 | 0.6616 | 0.2552 | 0.7208 | 0.4157 | 0.6916 | 0.4906 | 0.7571 | 0.4000 | 0.7986 | 0.5446 |
| `Qwen_Qwen3.5-9B` | 0.1562 | 0.5838 | 0.2308 | 0.5505 | 0.3862 | 0.8247 | 0.3258 | 0.6822 | 0.5094 | 0.7571 | 0.3455 | 0.7500 | 0.5433 |
| `Tongyi-MAI_MAI-UI-2B` | 0.2031 | 0.4416 | 0.2378 | 0.6566 | 0.2483 | 0.7468 | 0.3258 | 0.7290 | 0.4151 | 0.8079 | 0.3091 | 0.7847 | 0.5275 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.1719 | 0.5533 | 0.2028 | 0.6162 | 0.1724 | 0.6818 | 0.1910 | 0.5701 | 0.4340 | 0.7853 | 0.2091 | 0.7917 | 0.4921 |
| `meituan_EvoCUA-8B-20260105` | 0.1094 | 0.3604 | 0.1189 | 0.6111 | 0.1448 | 0.6948 | 0.1573 | 0.5701 | 0.2264 | 0.7345 | 0.2000 | 0.6875 | 0.4314 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.1875 | 0.3350 | 0.1678 | 0.4949 | 0.1586 | 0.5649 | 0.2809 | 0.5234 | 0.3208 | 0.6497 | 0.1909 | 0.5278 | 0.3922 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.0781 | 0.2284 | 0.0699 | 0.5000 | 0.0552 | 0.5065 | 0.1236 | 0.3178 | 0.1509 | 0.5932 | 0.2000 | 0.6181 | 0.3251 |
