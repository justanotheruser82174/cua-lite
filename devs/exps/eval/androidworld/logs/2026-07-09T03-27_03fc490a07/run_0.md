# androidworld @ 2026-07-07T03-53_4fa96a6e41 · run_0

- **Commit**: `4fa96a6e41` — `docs(eval): update androidworld run_0 qwen3-vl-2b`; appended `gpt-5.5` from `03fc490a07` — `Update WebHarbor WebVoyager run_0`
- **Host / GPUs**: `gpublaze` / 6-7.
- **Artifacts**: `.exps/eval/androidworld/2026-07-07T03-53_4fa96a6e41/run_0/`
- **Started**: `2026-07-07 02:35 PDT`
- **Last updated**: `2026-07-09 05:27 PDT`
- **Notes**: User-requested subset: `Qwen/Qwen3-VL-{2,4,8,32}B-Instruct`, `Qwen/Qwen3.5-{2,4,9,27}B`, `ByteDance-Seed/UI-TARS-7B-DPO`, `ByteDance-Seed/UI-TARS-1.5-7B`, `Tongyi-MAI/MAI-UI-2B`, `MarsXL/UI-Voyager`, `stepfun-ai/GELab-Zero-4B-preview`. Env-server: `http://localhost:30100`, token `lhr`. Run config from `devs/exps/eval/androidworld/run.sh`: `--concurrency 4`, `--env-kwargs '{"step_timeout": 180}'`. `Qwen/Qwen3-VL-{2,4}B-Instruct` completed under the pre-doc-commit artifact dir `2026-07-06T20-17_f0e5ef6e6b` and was copied forward; no androidworld pipeline files changed between `f0e5ef6e6b` and `4fa96a6e41`. Mop-up reruns on `2026-07-07` resumed incomplete rows in place. `Qwen/Qwen3.5-27B` and `Qwen/Qwen3-VL-32B-Instruct` needed direct `scripts/rollout.py` launches with `--engine-kwargs '{"disable_custom_all_reduce": true}'` after sglang custom all-reduce CUDA graph capture failed under `run.sh`. A second rerun pass on `2026-07-07` completed one additional `Qwen/Qwen3-VL-8B-Instruct` sample and retried the remaining env-server-bound samples again. `gpt-5.5` was appended from GPT API artifacts at `.exps/eval/androidworld/2026-07-09T03-27_03fc490a07/run_0/gpt-5.5/` with `EVAL_CONCURRENCY=4`.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `MarsXL/UI-Voyager` | 116/116 | 0.7414 |
| `gpt-5.5` | 116/116 | 0.7414 |
| `Qwen/Qwen3-VL-32B-Instruct` | 116/116 | 0.6379 |
| `Qwen/Qwen3.5-9B` | 116/116 | 0.5388 |
| `Qwen/Qwen3.5-4B` | 116/116 | 0.5259 |
| `Tongyi-MAI/MAI-UI-2B` | 116/116 | 0.4009 |
| `Qwen/Qwen3-VL-4B-Instruct` | 116/116 | 0.3793 |
| `Qwen/Qwen3-VL-2B-Instruct` | 116/116 | 0.2931 |
| `Qwen/Qwen3.5-2B` | 116/116 | 0.2759 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 116/116 | 0.1983 |
| ⚠️ `Qwen/Qwen3.5-27B` | _**115/116**_ | _**0.6348**_ |
| ⚠️ `ByteDance-Seed/UI-TARS-7B-DPO` | _**115/116**_ | _**0.1174**_ |
| `Qwen/Qwen3-VL-8B-Instruct` | 116/116 | 0.5000 |
| `stepfun-ai/GELab-Zero-4B-preview` | 116/116 | 0.4009 |

## Highlights

- **UI-Voyager is still the clear top row** at 0.7414, +10.35pp over the strongest completed Qwen row (`Qwen3-VL-32B-Instruct`, 0.6379). The next tier is tightly packed: `Qwen3-VL-32B-Instruct` at 0.6379 and `Qwen3.5-27B` at 0.6348 valid-only. If the missing `SimpleSmsResend` sample were scored over all 116 tasks, `Qwen3.5-27B` would land in `[0.6293, 0.6379]` depending on reward 0/1.
- **Qwen3-VL scales monotonically** across the tested sizes: 2B→4B is +8.62pp, 4B→8B is +12.07pp, and 8B→32B is +13.79pp. The gains do not saturate in this range; the 32B model is +34.48pp over the 2B model.
- **Qwen3.5 gets most of its gain at 4B, then needs 27B to move again**: 2B→4B is +25.00pp, 4B→9B is only +1.29pp, and 9B→27B is +9.60pp on valid samples. At comparable sizes, Qwen3.5-4B beats Qwen3-VL-4B by +14.66pp, while Qwen3.5-9B beats Qwen3-VL-8B by +3.88pp.
- **Small UI-specialized agents are competitive but not uniformly better.** `MAI-UI-2B` and `GELab-Zero-4B-preview` both end at 0.4009: above `Qwen3-VL-4B-Instruct` by +2.16pp, but below `Qwen3.5-4B` by -12.50pp. `UI-TARS-1.5-7B` is much lower at 0.1983, and `UI-TARS-7B-DPO` remains below it on the valid subset.
- **The remaining incompletes are infrastructure-bound, not scored failures.** Both `Qwen3.5-27B` and `UI-TARS-7B-DPO` are missing only `SimpleSmsResend`; repeated reruns reached `/step` HTTP 500 before a sample summary could be written. For `Qwen3.5-27B`, the captured container log showed the first action was a long press (`adb shell input swipe 771 1877 771 1877 1000`), after which ADB lost `emulator-5554`, gRPC to `127.0.0.1:8554` was refused, and AndroidWorld raised `AdbControllerError`.
- **Step caps mostly behaved as intended, with one explicit override.** `SimpleSmsResend` has complexity 1.2, so the derived cap is 12 steps; the observed failures happened before truncation because the emulator/ADB backend crashed. `VlcCreateTwoPlaylists` has complexity 4.8, so the expected cap is 48; the derived cap did not stop one resumed loop, so the final Qwen3-VL-8B retry passed `--env-kwargs '{"step_timeout": 180, "max_steps": 48}'` explicitly and truncated at 48 frames with reward 0.0.
- `gpt-5.5` completed all 116 AndroidWorld eval tasks at 0.7414, matching the top `MarsXL/UI-Voyager` row in this snapshot.
