# screenspot_pro @ 2026-04-28T05-32_ff51fcf · run_0

- **Commit**: `ff51fcf` — `agents(grounding): clean grounding.action route — remove _is_navigation_task gate`. Same pipeline diff as the sibling osworld_g snapshot: env declares `task_type="grounding.action"`; `Qwen3VLBaseAdapter._is_navigation_task` removed; class-level `system_prompt` / `user_prompt_template` defaults per subclass; YAML overrides removed. Smoke test (`.logs/grounding_smoke_new/`) confirmed byte-identical prompts + per-task MERs vs the pre-refactor smoke at `.logs/grounding_smoke/`.
- **Host / GPUs**: `gpublaze` / 2-GPU dispatcher on GPU 2,3 (shared with another tenant's small workload).
- **Started**: `2026-04-28 12:34 UTC`
- **Last updated**: `2026-04-28 16:18 UTC` (campaign duration: 224 min, of which screenspot_pro Phase 1 + Phase 2 took ~190 min — screenspot_pro is ~3-4× slower per task than osworld_g due to native 4K images / larger vision token counts).
- **Notes**: 1581 tasks, no exclusion filter (no refusals in this benchmark). Two of the 11 models were partial carry-overs from the prior `2026-04-28T03-16_a671f2b/` dir (renamed in place after the grounding refactor since pipeline state changed): Qwen3-VL-8B was at 957/1581 and Qwen3.5-2B at 1020/1581 when stopped; both resumed and completed under the new commit. Phase 2 (Qwen3-VL-32B + Qwen3.5-27B at tp=2) ran ~60min and ~50min respectively.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3.5-27B`              | 1581/1581 | **0.5642** |
| `Qwen/Qwen3-VL-4B-Instruct`     | 1581/1581 | 0.5408 |
| `Qwen/Qwen3-VL-32B-Instruct`    | 1581/1581 | 0.5395 |
| `Qwen/Qwen3.5-4B`               | 1581/1581 | 0.5300 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 1581/1581 | 0.5085 |
| `Qwen/Qwen3.5-9B`               | 1581/1581 | 0.4883 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 1581/1581 | 0.4794 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 1581/1581 | 0.4168 |
| `Qwen/Qwen3.5-2B`               | 1581/1581 | 0.4016 |
| `meituan/EvoCUA-8B-20260105`    | 1581/1581 | 0.3599 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 1581/1581 | 0.3137 |

## Highlights

- **Qwen3.5-27B is the only model > 0.55** (MER 0.5642), 2pp ahead of the 4B-tier cluster. The other large-class peers (Qwen3-VL-32B at 0.540, Qwen3.5-9B at 0.488, Qwen3-VL-8B at 0.509) all trail it — and notably Qwen3.5-27B beats Qwen3-VL-32B by 2.5pp here despite osworld_g being a tie. Larger-context / longer-reasoning checkpoints help most on the harder Pro distribution.
- **Qwen3-VL-4B punches above its weight**: 0.5408 — between 32B and 8B, and above 8B by 3pp! This is the standout finding of the campaign. Hypothesis: 4B's smaller visual encoder produces a more focused attention map at native 4K, while 8B / 32B over-reason; or 4B's SFT data covered grounding better. Worth a per-task breakdown to confirm which subset 4B trades on.
- **Qwen3.5-4B (0.530) ≈ Qwen3-VL-8B (0.509)** — the same "Qwen3.5 punches up" pattern, though less extreme. 4B-class clearly has better grounding behavior than 8B-9B on this benchmark.
- **Qwen3-VL family at 32B / 8B / 4B / 2B** is **non-monotonic** for screenspot_pro: 4B (0.541) > 32B (0.540) > 8B (0.509) > 2B (0.417). osworld_g had clean monotonic scaling; the screenspot_pro reversal at 4B/8B is real.
- **Qwen3.5 family at 27B / 9B / 4B / 2B** is also non-monotonic in the middle: 27B (0.564) > 4B (0.530) > 9B (0.488) > 2B (0.402). 4B beats 9B by 4pp.
- **EvoCUA-8B underperforms** here (0.360) — second-to-last. Same model on osworld_g placed mid-pack at 0.520 — so its SFT shape doesn't transfer well to Pro's harder, higher-resolution grounding.
- **UI-TARS family**: UI-TARS-1.5-7B (0.479) > UI-TARS-7B-DPO (0.314) — the 1.5 advantage is even bigger here than on osworld_g (16.6pp gap vs 10.6pp gap). DPO is the weakest model on screenspot_pro by a wide margin.
- **Sub-2B-class baseline**: Qwen3-VL-2B (0.417) > Qwen3.5-2B (0.402) — Qwen3-VL still slightly stronger same-size, but the gap is small (1.5pp).

## Breakdown

Two-step regen: `reaggregate_breakdown.py --axes group,ui_type,application .exps/eval/screenspot_pro/2026-04-28T05-32_ff51fcf/run_0/` → `render_breakdown.py .exps/eval/screenspot_pro/2026-04-28T05-32_ff51fcf/run_0/`. The 12-cell `group × ui_type` cross-product matches the paper's canonical layout — re-render via `--cross "group,ui_type"`. The 26-column `application` axis is omitted here.

### Breakdown — by `group`

| Model | `CAD` | `Creative` | `Dev` | `OS` | `Office` | `Scientific` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.5709 | 0.4545 | 0.6388 | 0.5918 | 0.7870 | 0.3937 | 0.5642 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.4215 | 0.4897 | 0.5318 | 0.5510 | 0.7087 | 0.5827 | 0.5408 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.5134 | 0.5572 | 0.4381 | 0.4133 | 0.7174 | 0.5984 | 0.5395 |
| `Qwen_Qwen3.5-4B` | 0.4291 | 0.4927 | 0.5351 | 0.5153 | 0.6565 | 0.5748 | 0.5300 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.4291 | 0.4399 | 0.5017 | 0.4898 | 0.6435 | 0.5827 | 0.5085 |
| `Qwen_Qwen3.5-9B` | 0.3870 | 0.3900 | 0.5151 | 0.4898 | 0.7043 | 0.4961 | 0.4883 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.4253 | 0.4340 | 0.4114 | 0.3827 | 0.7000 | 0.5512 | 0.4794 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.2797 | 0.4194 | 0.3478 | 0.4337 | 0.6043 | 0.4528 | 0.4168 |
| `Qwen_Qwen3.5-2B` | 0.2414 | 0.3695 | 0.3378 | 0.3827 | 0.5826 | 0.5354 | 0.4016 |
| `meituan_EvoCUA-8B-20260105` | 0.2069 | 0.3636 | 0.3278 | 0.3163 | 0.5261 | 0.4331 | 0.3599 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1762 | 0.3109 | 0.2876 | 0.2092 | 0.4739 | 0.4252 | 0.3137 |

### Breakdown — by `ui_type`

| Model | `icon` | `text` | Avg |
|---|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.3328 | 0.7073 | 0.5642 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.2732 | 0.7062 | 0.5408 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.2500 | 0.7185 | 0.5395 |
| `Qwen_Qwen3.5-4B` | 0.2632 | 0.6950 | 0.5300 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.2036 | 0.6970 | 0.5085 |
| `Qwen_Qwen3.5-9B` | 0.2715 | 0.6223 | 0.4883 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.2020 | 0.6510 | 0.4794 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.2119 | 0.5435 | 0.4168 |
| `Qwen_Qwen3.5-2B` | 0.1937 | 0.5302 | 0.4016 |
| `meituan_EvoCUA-8B-20260105` | 0.1192 | 0.5087 | 0.3599 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1060 | 0.4422 | 0.3137 |

### Breakdown — by `group` × `ui_type`

Paper-canonical 12-cell layout (icon vs text within each app group).

| Model | `CAD × icon` | `CAD × text` | `Creative × icon` | `Creative × text` | `Dev × icon` | `Dev × text` | `OS × icon` | `OS × text` | `Office × icon` | `Office × text` | `Scientific × icon` | `Scientific × text` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.2031 | 0.6904 | 0.2517 | 0.6010 | 0.4414 | 0.8247 | 0.3820 | 0.7664 | 0.5660 | 0.8531 | 0.2182 | 0.5278 | 0.5642 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.1562 | 0.5076 | 0.2098 | 0.6919 | 0.2897 | 0.7597 | 0.3146 | 0.7477 | 0.3396 | 0.8192 | 0.3364 | 0.7708 | 0.5408 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.2969 | 0.5838 | 0.2378 | 0.7879 | 0.1517 | 0.7078 | 0.1798 | 0.6075 | 0.4528 | 0.7966 | 0.3273 | 0.8056 | 0.5395 |
| `Qwen_Qwen3.5-4B` | 0.1719 | 0.5127 | 0.2098 | 0.6970 | 0.2897 | 0.7662 | 0.3034 | 0.6916 | 0.3585 | 0.7458 | 0.2727 | 0.8056 | 0.5300 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.1250 | 0.5279 | 0.1329 | 0.6616 | 0.2000 | 0.7857 | 0.2247 | 0.7103 | 0.3019 | 0.7458 | 0.2818 | 0.8125 | 0.5085 |
| `Qwen_Qwen3.5-9B` | 0.1562 | 0.4619 | 0.2238 | 0.5101 | 0.2966 | 0.7208 | 0.2584 | 0.6822 | 0.5283 | 0.7571 | 0.2545 | 0.6806 | 0.4883 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.1719 | 0.5076 | 0.1748 | 0.6212 | 0.1379 | 0.6688 | 0.1910 | 0.5421 | 0.4340 | 0.7797 | 0.2364 | 0.7917 | 0.4794 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.1719 | 0.3147 | 0.1888 | 0.5859 | 0.1241 | 0.5584 | 0.2809 | 0.5607 | 0.3962 | 0.6667 | 0.2364 | 0.6181 | 0.4168 |
| `Qwen_Qwen3.5-2B` | 0.0781 | 0.2944 | 0.1818 | 0.5051 | 0.1310 | 0.5325 | 0.1798 | 0.5514 | 0.3019 | 0.6667 | 0.3182 | 0.7014 | 0.4016 |
| `meituan_EvoCUA-8B-20260105` | 0.0781 | 0.2487 | 0.0839 | 0.5657 | 0.0759 | 0.5649 | 0.1124 | 0.4860 | 0.2264 | 0.6158 | 0.2000 | 0.6111 | 0.3599 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.0625 | 0.2132 | 0.0769 | 0.4798 | 0.0483 | 0.5130 | 0.1124 | 0.2897 | 0.1698 | 0.5650 | 0.2091 | 0.5903 | 0.3137 |
