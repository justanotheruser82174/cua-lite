# androidworld @ 2026-04-26T18-38_8b9d4bd · run_0

- **Commit**: `8b9d4bd46a86ec657bb19127d71aa8a43ac67317` — `eval: long-run runner + matrix wiring for the eval.md campaign`
- **Host / GPUs**: `gpublaze` / `0-3`
- **Notes**: raw artifacts: `.exps/eval/androidworld/2026-04-26T18-38_8b9d4bd/run_0/<slug>/`. 12 of 13 models reached 116/116. `Qwen/Qwen3.5-2B` recovered to 113/116 after 2026-04-27 mop-up — the original 8-task gap (a single SGLang `httpx.ConnectError` event) was retried with `--max-attempts 10` + `tp_size=2` + `mem_fraction_static=0.65` + `concurrency=4` (Qwen3.5's mamba state cache OOMs at tp=1 / default mem fraction with 8 concurrent requests). 5 tasks recovered; 3 left as residual.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `MarsXL/UI-Voyager`               | 116/116 | 0.6810 |
| `Qwen/Qwen3-VL-32B-Instruct`      | 116/116 | 0.5474 |
| `Tongyi-MAI/MAI-UI-8B`            | 116/116 | 0.5345 |
| `Qwen/Qwen3-VL-8B-Instruct`       | 116/116 | 0.4784 |
| ⚠️ `Qwen/Qwen3.5-4B`              | 116/116 | _**0.4741**_ |
| `stepfun-ai/GELab-Zero-4B-preview`| 116/116 | 0.4440 |
| `Tongyi-MAI/MAI-UI-2B`            | 116/116 | 0.4138 |
| `Qwen/Qwen3-VL-2B-Instruct`       | 116/116 | 0.3793 |
| `Qwen/Qwen3-VL-4B-Instruct`       | 116/116 | 0.3448 |
| ⚠️ `Qwen/Qwen3.5-2B`              | _**113/116**_ | _**0.2876**_ |
| `ByteDance-Seed/UI-TARS-1.5-7B`   | 116/116 | 0.2112 |
| ⚠️ `Qwen/Qwen3.5-9B`              | 116/116 | _**0.1638**_ |
| `ByteDance-Seed/UI-TARS-7B-DPO`   | 116/116 | 0.1250 |
| ⚠️ `Qwen/Qwen3.5-27B`             | _**0/116**_   | _**—**_       |

## Highlights

- `MarsXL/UI-Voyager` leads decisively (0.6810) — its SFT distribution is androidworld-heavy (matches its public eval focus). 13 points above the next finisher (`Qwen3-VL-32B` 0.5474).
- `Qwen3-VL` scaling is **clean and steep**: 32B (0.5474) > 8B (0.4784) >> 4B (0.3448) ≈ 2B (0.3793). The 32B-vs-8B gap is small but consistent; the 8B-vs-4B drop is the largest single step. The 4B-vs-2B inversion is small (~0.03).
- ⚠️ **Qwen3.5 mobile scores depressed by `left_click` schema-mismatch bug** (post-run diagnosis): Qwen3.5-9B emits desktop `left_click` on 73.6% of mobile turns (1404/1907), Qwen3.5-4B on 15.5% (208/1342), Qwen3.5-2B on 0%. The mobile_use schema only declares the canonical 8-action enum; `left_click` falls through to env-side `noop`, collapsing 9B's MER from a 4B-baseline ~0.47 down to 0.16. Qwen3-VL-{2,4,8,32}B emit 0% `left_click` on the same env — leakage is Qwen3.5-family-specific (likely XML/Hermes tool_call format + "mouse" wording in schema description activating desktop SFT prior). Fix landed at `Qwen3_5MobileActionSpace._MOBILE_ACTION_ALIASES["left_click"] = "click"` after this run; expected MER recovery: 9B → ~0.46, 4B → ~0.52. Re-run pending.
- `Qwen3.5-4B` (0.4741) > `Qwen3.5-9B` (0.1638) is **anti-scaling and large** — 31 points. Driven entirely by the left_click leakage above (9B's desktop SFT prior is strongest in the family); the 4B baseline is the right "clean" floor for the family on this env.
- `MAI-UI-8B` (0.5345) > `MAI-UI-2B` (0.4138) — clean MAI scaling, both strong.
- `UI-TARS-1.5-7B` (0.2112) clearly beats `UI-TARS-7B-DPO` (0.1250). Generation 1.5 wins again, in line with the osworld and androidlab patterns.
- `OpenGVLab/ScaleCUA-7B` and `meituan/EvoCUA-8B` were excluded from this env's matrix (they don't ship an androidworld action set).
- `Qwen3.5-27B` not started — wave 3 was halted by user.
- `OpenCUA-7B` not started — locked to `backends=["hf"]`; rollout only auto-starts SGLang.

## Mop-up history

The original Qwen3.5-2B run halted at 108/116 (8 task dirs left from a single SGLang `httpx.ConnectError` event). 2026-04-27 mop-up:

1. **`--max-attempts` 3 → 10** in `lite/utils/rollout.py` so resume cycles can use more rounds.
2. **First relaunch (tp=1)** died with `torch.OutOfMemoryError` mid-rollout. Recovered 1 task before crashing (108 → 109).
3. **Second relaunch (`tp_size=2`, `dp_size=1`)** ran briefly then OOM'd again — sglang per-process default memory fraction + Qwen3.5 mamba state caches at concurrency=8 still pushed past 80 GiB on the rank-0 GPU. Recovered 4 more tasks before crashing (109 → 113).
4. **Third relaunch (`tp_size=2` + `mem_fraction_static=0.65` + `concurrency=4`)** ran the safe config but was stopped before any of the 3 residual tasks landed. 113/116 is the final figure for this campaign.

The 3 residual tasks are `OsmAndTrack`, `SimpleCalendarAddOneEventInTwoWeeks`, `VlcCreateTwoPlaylists`.
