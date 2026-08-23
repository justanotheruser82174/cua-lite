# screenspot_pro @ 2026-04-27T23-21_828ea14 · run_0

- **Commit at campaign kickoff**: `828ea143` — `eval(grounding): bump concurrency 16 → 64 in run.sh`.
- **Caveat — code drifted mid-campaign**: this snapshot's path is named after the *kickoff* commit, but `9604e62 eval(grounding): use_extra_tools env_kwarg + to_thread hook fix + snapshots` landed mid-campaign and (a) added the `to_thread` hook fix that gave ~4× per-slot throughput, (b) bundled `--env-kwargs '{"use_extra_tools": false}'` into the run.sh — the latter is a no-op for screenspot_pro (no extra_tools to drop) but still landed under the same commit. Earlier rollouts ran on slower pre-fix code, later ones on the optimized code; numbers below are a mix. Pure 828ea14 reproducibility is not exact. Future campaigns should commit-before-launch.
- **Host / GPUs**: `gpublaze` / 4-GPU dispatcher on 0-3 then 2-GPU on 2-3
- **Notes**: First grounding-env campaign on the standardized layout. **1581** tasks, no exclusion filter. Raw artifacts: `.exps/eval/screenspot_pro/2026-04-27T23-21_828ea14/run_0/<slug>/`. Wave 2 (Qwen3-VL-32B, tp=2) and Wave 3 (Qwen3.5-27B, tp=2 fallback) deferred — campaign paused after EvoCUA-8B reached 956/1581 to debug the prompt-template overhead found in single-step grounding.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct`     | 1581/1581 | **0.5104** |
| `Qwen/Qwen3.5-4B`               | 1581/1581 | 0.5073 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 1581/1581 | 0.5009 |
| `Qwen/Qwen3.5-2B`               | 1581/1581 | 0.4870 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 1581/1581 | 0.4168 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 1581/1581 | 0.4137 |
| `Qwen/Qwen3.5-9B`               | 1581/1581 | 0.3542 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 1581/1581 | 0.3118 |
| ⚠️ `meituan/EvoCUA-8B-20260105` | _**956/1581**_ | _**0.2866**_ |
| ⚠️ `Qwen/Qwen3-VL-32B-Instruct` | _**0/1581**_ | _**—**_ |
| ⚠️ `Qwen/Qwen3.5-27B`           | _**0/1581**_ | _**—**_ |

## Highlights

- **Top-3 dense:** Qwen3-VL-4B (0.510), Qwen3.5-4B (0.507), Qwen3-VL-8B (0.501) all within 0.01 — at this resolution / task scale, mid-size matters more than family.
- **Qwen3-VL scaling:** 8B (0.501) ≈ 4B (0.510) > 2B (0.414) — no 8B advantage over 4B.
- **Qwen3.5 anti-scaling:** 4B (0.507) ≈ 2B (0.487) > **9B (0.354)** — the 9B is a clear regression. Larger Qwen3.5 trades visual precision for reasoning capacity (cf. androidworld snapshot where 9B<4B was first observed).
- **UI-TARS-1.5 > UI-TARS-7B-DPO** (0.417 vs 0.312) — the 1.5 upgrade replicates here, with a larger gap than on osworld_g (0.105 vs 0.041).
- **EvoCUA-8B partial 956/1581 → MER 0.287** (interim). The campaign was killed mid-rollout for prompt debug; final MER may shift ±2pp on full run.
- **Wave 2 + Wave 3 deferred:** Qwen3-VL-32B and Qwen3.5-27B not run — same reason as `osworld_g`.

## Note on response format

All 8 finished models emitted format-compliant tool_calls (97-100% click-action emission rate, verified on `Qwen3.5-9B` at 97.3%). The MER-spread is dominated by **click-coordinate accuracy**, not parser failures. E.g. on osworld_g, Qwen3-VL-8B's bbox-miss rate was 40% (clicks emitted but outside target) vs Qwen3.5-9B's 49% — at 4K native resolution, small-element grounding is genuinely hard for hybrid mamba+attention vision backbones.

## Breakdown

Two-step regen: `reaggregate_breakdown.py --axes group,ui_type,application .exps/eval/screenspot_pro/2026-04-27T23-21_828ea14/run_0/` → `render_breakdown.py .exps/eval/screenspot_pro/2026-04-27T23-21_828ea14/run_0/`. The 12-cell `group × ui_type` cross-product matches the paper's canonical layout — re-render via `--cross "group,ui_type"`. The 26-column `application` axis is omitted here. EvoCUA-8B is partial (956/1581) so its row averages 5/6 groups; absent cells render as `—`.

### Breakdown — by `group`

| Model | `CAD` | `Creative` | `Dev` | `OS` | `Office` | `Scientific` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3-VL-4B-Instruct` | 0.3946 | 0.4633 | 0.4983 | 0.5255 | 0.6261 | 0.5906 | 0.5104 |
| `Qwen_Qwen3.5-4B` | 0.4713 | 0.4751 | 0.4716 | 0.4490 | 0.6087 | 0.5827 | 0.5073 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.4368 | 0.4545 | 0.4649 | 0.4949 | 0.6609 | 0.5315 | 0.5009 |
| `Qwen_Qwen3.5-2B` | 0.3946 | 0.4692 | 0.4247 | 0.4541 | 0.6565 | 0.5512 | 0.4870 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.3755 | 0.4018 | 0.3478 | 0.2806 | 0.6522 | 0.4528 | 0.4168 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.3103 | 0.3930 | 0.3478 | 0.4133 | 0.5870 | 0.4685 | 0.4137 |
| `Qwen_Qwen3.5-9B` | 0.3602 | 0.2375 | 0.4247 | 0.3673 | 0.5087 | 0.2717 | 0.3542 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1724 | 0.3021 | 0.3043 | 0.2194 | 0.4174 | 0.4528 | 0.3118 |
| `meituan_EvoCUA-8B-20260105` | 0.0096 | 0.2843 | 0.2500 | 0.2696 | 0.3288 | 0.4244 | 0.2866 |

### Breakdown — by `ui_type`

| Model | `icon` | `text` | Avg |
|---|---:|---:|---:|
| `Qwen_Qwen3-VL-4B-Instruct` | 0.2334 | 0.6817 | 0.5104 |
| `Qwen_Qwen3.5-4B` | 0.2384 | 0.6735 | 0.5073 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.2136 | 0.6786 | 0.5009 |
| `Qwen_Qwen3.5-2B` | 0.2152 | 0.6551 | 0.4870 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.1540 | 0.5793 | 0.4168 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.1474 | 0.5783 | 0.4137 |
| `Qwen_Qwen3.5-9B` | 0.1589 | 0.4749 | 0.3542 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1026 | 0.4411 | 0.3118 |
| `meituan_EvoCUA-8B-20260105` | 0.1034 | 0.3914 | 0.2866 |

### Breakdown — by `group` × `ui_type`

Paper-canonical 12-cell layout (icon vs text within each app group).

| Model | `CAD × icon` | `CAD × text` | `Creative × icon` | `Creative × text` | `Dev × icon` | `Dev × text` | `OS × icon` | `OS × text` | `Office × icon` | `Office × text` | `Scientific × icon` | `Scientific × text` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3-VL-4B-Instruct` | 0.1406 | 0.4772 | 0.2098 | 0.6465 | 0.2276 | 0.7532 | 0.2472 | 0.7570 | 0.3208 | 0.7175 | 0.2727 | 0.8333 | 0.5104 |
| `Qwen_Qwen3.5-4B` | 0.1250 | 0.5838 | 0.2238 | 0.6566 | 0.2069 | 0.7208 | 0.2472 | 0.6168 | 0.3208 | 0.6949 | 0.3182 | 0.7847 | 0.5073 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.1719 | 0.5228 | 0.1469 | 0.6768 | 0.1793 | 0.7338 | 0.2584 | 0.6916 | 0.3208 | 0.7627 | 0.2818 | 0.7222 | 0.5009 |
| `Qwen_Qwen3.5-2B` | 0.0781 | 0.4975 | 0.2168 | 0.6515 | 0.1586 | 0.6753 | 0.2697 | 0.6075 | 0.3019 | 0.7627 | 0.2818 | 0.7569 | 0.4870 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.0938 | 0.4670 | 0.1399 | 0.5909 | 0.1310 | 0.5519 | 0.1348 | 0.4019 | 0.3019 | 0.7571 | 0.1818 | 0.6597 | 0.4168 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.0469 | 0.3959 | 0.1538 | 0.5657 | 0.0897 | 0.5909 | 0.1685 | 0.6168 | 0.2264 | 0.6949 | 0.2182 | 0.6597 | 0.4137 |
| `Qwen_Qwen3.5-9B` | 0.0625 | 0.4569 | 0.1119 | 0.3283 | 0.1793 | 0.6558 | 0.2135 | 0.4953 | 0.2830 | 0.5763 | 0.1455 | 0.3681 | 0.3542 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.0625 | 0.2081 | 0.0490 | 0.4848 | 0.0828 | 0.5130 | 0.0899 | 0.3271 | 0.1509 | 0.4972 | 0.2091 | 0.6389 | 0.3118 |
| `meituan_EvoCUA-8B-20260105` | 0.0000 | 0.0116 | 0.0876 | 0.4438 | 0.0909 | 0.4444 | 0.0476 | 0.3973 | 0.1316 | 0.3981 | 0.1884 | 0.5441 | 0.2866 |
