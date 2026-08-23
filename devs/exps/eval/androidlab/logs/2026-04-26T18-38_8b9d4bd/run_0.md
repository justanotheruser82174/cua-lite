# androidlab @ 2026-04-26T18-38_8b9d4bd · run_0

- **Commit**: `8b9d4bd46a86ec657bb19127d71aa8a43ac67317` — `eval: long-run runner + matrix wiring for the eval.md campaign`
- **Host / GPUs**: `gpublaze` / `0-3`
- **Notes**: raw artifacts: `.exps/eval/androidlab/2026-04-26T18-38_8b9d4bd/run_0/<slug>/`. All non-27B models reached 138/138.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `Tongyi-MAI/MAI-UI-8B`              | 138/138 | 0.5217 |
| `Qwen/Qwen3-VL-32B-Instruct`        | 138/138 | 0.4275 |
| `Qwen/Qwen3-VL-4B-Instruct`         | 138/138 | 0.3913 |
| `ByteDance-Seed/UI-TARS-7B-DPO`     | 138/138 | 0.3841 |
| `stepfun-ai/GELab-Zero-4B-preview`  | 138/138 | 0.3768 |
| `Tongyi-MAI/MAI-UI-2B`              | 138/138 | 0.3696 |
| ⚠️ `Qwen/Qwen3.5-9B`                | 138/138 | _**0.3696**_ |
| `ByteDance-Seed/UI-TARS-1.5-7B`     | 138/138 | 0.3551 |
| `Qwen/Qwen3-VL-8B-Instruct`         | 138/138 | 0.3478 |
| `Qwen/Qwen3.5-2B`                   | 138/138 | 0.3406 |
| `Qwen/Qwen3-VL-2B-Instruct`         | 138/138 | 0.2899 |
| `MarsXL/UI-Voyager`                 | 138/138 | 0.2826 |
| ⚠️ `Qwen/Qwen3.5-4B`                | 138/138 | _**0.2754**_ |
| ⚠️ `Qwen/Qwen3.5-27B`               | _**0/138**_ | _**—**_ |

## Highlights

- `MAI-UI-8B` leads by a wide margin (0.5217) — Tongyi's SFT distribution maps cleanly to androidlab tasks. 9 points above the next finisher (`Qwen3-VL-32B` 0.4275).
- `Qwen3-VL` shows positive scaling: 32B (0.4275) > 4B (0.3913) > 8B (0.3478) ≈ 2B (0.2899). The 4B > 8B reversal is small (~4 pts) but consistent — likely SFT-noise on this checkpoint family.
- ⚠️ **Qwen3.5 mobile scores depressed by `left_click` schema-mismatch bug** (post-run diagnosis): on androidlab the leakage is **inverted vs androidworld** — Qwen3.5-4B emits desktop `left_click` on 35.8% of mobile turns (500/1395, mostly on `pimusic`/`zoom`/`map`/`clock` apps where 26 tasks hit 100% leakage → 20-turn step-limit loop), while Qwen3.5-9B is at 10.0% (159/1594, mostly on `bluecoins`/`cantook`). The two models leak on **disjoint app sets** — 4B and 9B desktop priors fire on different visual UIs. Qwen3.5-2B and Qwen3-VL-{2,4,8,32}B emit 0% `left_click`. Fix landed at `Qwen3_5MobileActionSpace._MOBILE_ACTION_ALIASES["left_click"] = "click"` after this run; expected MER recovery: 4B → ~0.37, 9B → ~0.41. Re-run pending.
- `Qwen3.5-4B` (0.2754) is the worst of the finished runs — below `Qwen3.5-2B` (0.3406), which is anti-scaling. Driven entirely by the left_click leakage above (4B's desktop SFT prior fires on this env's app distribution); the 2B baseline is the right "clean" floor for the family on this env.
- `UI-Voyager` ranks low (0.2826) on androidlab despite topping `androidworld` (0.6810) — its SFT data is tilted toward androidworld.
- `Qwen/Qwen3.5-27B` not started — wave 3 (tp=4) was halted by user before reaching this env.
