# screenspot_pro @ 2026-07-09T04-31_5b887a39c1 · run_0

- **Commit**: `1936546595` — `eval: allow webarena som thinking variants`; appended `gpt-5.5` from `5b887a39c1` — `Correct WebHarbor WebVoyager run_0`
- **Host / GPUs**: `gpublaze` / `0,6`
- **Artifacts**: `.exps/eval/screenspot_pro/2026-07-09T04-31_5b887a39c1/run_0/`
- **Started**: `2026-07-03 23:02 PDT`
- **Last updated**: `2026-07-09 06:01 PDT`
- **Notes**: 1581 tasks, no filter. Usable eval GPUs are `0,6`; GPU `4` is idle but user-marked broken and unused. GPUs 1,2,3,5,7 are occupied by another user's VLLM workers. A stale completed `Qwen/Qwen3-VL-2B-Instruct` SGLang server on GPU 0 was terminated before the `Qwen/Qwen3-VL-32B-Instruct` run. `Qwen/Qwen3-VL-4B-Instruct` hit `/srv` ENOSPC at 1348/1581; generated videos/screenshots in this campaign were deleted while preserving summaries/json, then the model was resumed in the same log root. `MarsXL/UI-Voyager` and `stepfun-ai/GELab-Zero-4B-preview` are not launched for `screenspot_pro`: `devs/exps/eval/AGENTS.md` marks them out of scope for grounding envs and `screenspot_pro/run.sh` has no config case for them. `gpt-5.5` was appended from GPT API artifacts at `.exps/eval/screenspot_pro/2026-07-09T04-31_5b887a39c1/run_0/gpt-5.5/` with `EVAL_CONCURRENCY=16`.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | 1581/1581 | 0.5769 |
| `Qwen/Qwen3.5-4B` | 1581/1581 | 0.5693 |
| `Qwen/Qwen3-VL-32B-Instruct` | 1581/1581 | 0.5686 |
| `Qwen/Qwen3.5-9B` | 1581/1581 | 0.5560 |
| `Qwen/Qwen3-VL-8B-Instruct` | 1581/1581 | 0.5515 |
| `Qwen/Qwen3.5-2B` | 1581/1581 | 0.5364 |
| `Tongyi-MAI/MAI-UI-2B` | 1581/1581 | 0.5262 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 1581/1581 | 0.4883 |
| `gpt-5.5` | 1581/1581 | 0.4687 |
| `meituan/EvoCUA-8B-20260105` | 1581/1581 | 0.4295 |
| `Qwen/Qwen3-VL-2B-Instruct` | 1581/1581 | 0.4194 |
| `microsoft/Fara-7B` | 1581/1581 | 0.4156 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 1581/1581 | 0.3283 |
| `Qwen/Qwen3.5-27B` | 1581/1581 | 0.0702 |

## Highlights

- Top cluster is very compressed: Qwen3-VL-4B leads (0.577), but Qwen3.5-4B and Qwen3-VL-32B are within 0.9pp; top-5 spread is only 2.5pp.
- Qwen3-VL is non-monotonic: 4B ~= 32B > 8B >> 2B. 32B does not buy a clear gain on this grounding set.
- Qwen3.5 is also non-monotonic, with 4B > 9B > 2B and 27B collapsing to matrix floor (0.070). The 27B run is uniformly weak across icon/text and all groups, so this looks like a model/config interaction rather than one hard category.
- MAI-UI-2B is competitive for its size (0.526), only 1.0pp behind Qwen3.5-2B and ahead of both UI-TARS variants, EvoCUA, and Qwen3-VL-2B.
- UI-TARS-1.5-7B substantially improves over UI-TARS-7B-DPO (+16.0pp), but remains 3.8pp behind MAI-UI-2B and 8.9pp behind the leader.
- EvoCUA-8B (0.430) edges Qwen3-VL-2B by 1.0pp but trails UI-TARS-1.5-7B by 5.9pp; it is strongest on Office text and weakest on CAD/icon.
- Icon/text gap is the dominant failure mode. Even the leader has 0.303 icon vs 0.746 text; Qwen3.5-9B has the best icon score (0.343) but gives back enough text accuracy to miss the top cluster.
- Office is the easiest group for nearly every model; CAD and icon-heavy slices are the consistent stress cases.
- `gpt-5.5` completed all 1581 tasks at 0.4687; this lands below `UI-TARS-1.5-7B` and above `EvoCUA-8B` on this grounding set.
- All supported `screenspot_pro` models in scope are complete. `MarsXL/UI-Voyager` and `stepfun-ai/GELab-Zero-4B-preview` remain not started because the eval matrix marks them out of scope for grounding envs.

## Breakdown

### Breakdown — by `group`

| Model | `CAD` | `Creative` | `Dev` | `OS` | `Office` | `Scientific` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3-VL-4B-Instruct` | 0.4904 | 0.5044 | 0.5652 | 0.6020 | 0.7478 | 0.6024 | 0.5768 |
| `Qwen_Qwen3.5-4B` | 0.4866 | 0.5161 | 0.5552 | 0.5816 | 0.7261 | 0.5906 | 0.5693 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.5249 | 0.5689 | 0.4816 | 0.4592 | 0.7565 | 0.6299 | 0.5686 |
| `Qwen_Qwen3.5-9B` | 0.4943 | 0.4164 | 0.6321 | 0.5408 | 0.7217 | 0.5787 | 0.5560 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.5172 | 0.4868 | 0.5251 | 0.5051 | 0.7174 | 0.5906 | 0.5515 |
| `Qwen_Qwen3.5-2B` | 0.4215 | 0.4809 | 0.4950 | 0.5714 | 0.6870 | 0.6142 | 0.5364 |
| `Tongyi-MAI_MAI-UI-2B` | 0.3716 | 0.4633 | 0.5084 | 0.5765 | 0.7304 | 0.5669 | 0.5262 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.4483 | 0.4340 | 0.4348 | 0.3980 | 0.7087 | 0.5354 | 0.4883 |
| `gpt-5.5` | 0.4291 | 0.5220 | 0.3211 | 0.2449 | 0.6565 | 0.6142 | 0.4687 |
| `meituan_EvoCUA-8B-20260105` | 0.2835 | 0.4135 | 0.4281 | 0.3776 | 0.5957 | 0.4921 | 0.4295 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.2912 | 0.3988 | 0.3746 | 0.4439 | 0.6087 | 0.4409 | 0.4194 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1877 | 0.3255 | 0.2943 | 0.2398 | 0.4957 | 0.4331 | 0.3283 |
| `Qwen_Qwen3.5-27B` | 0.0651 | 0.0850 | 0.1070 | 0.0357 | 0.0522 | 0.0551 | 0.0702 |

### Breakdown — by `ui_type`

| Model | `icon` | `text` | Avg |
|---|---:|---:|---:|
| `Qwen_Qwen3-VL-4B-Instruct` | 0.3030 | 0.7462 | 0.5769 |
| `Qwen_Qwen3.5-4B` | 0.3063 | 0.7318 | 0.5693 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.2765 | 0.7492 | 0.5686 |
| `Qwen_Qwen3.5-9B` | 0.3427 | 0.6878 | 0.5560 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.2268 | 0.7523 | 0.5515 |
| `Qwen_Qwen3.5-2B` | 0.3245 | 0.6673 | 0.5364 |
| `Tongyi-MAI_MAI-UI-2B` | 0.2715 | 0.6837 | 0.5262 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.2086 | 0.6612 | 0.4883 |
| `gpt-5.5` | 0.3245 | 0.5578 | 0.4687 |
| `meituan_EvoCUA-8B-20260105` | 0.1540 | 0.5998 | 0.4295 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.1987 | 0.5558 | 0.4194 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.1076 | 0.4647 | 0.3283 |
| `Qwen_Qwen3.5-27B` | 0.0265 | 0.0972 | 0.0702 |

### Breakdown — by `group` × `ui_type`

| Model | `CAD × icon` | `CAD × text` | `Creative × icon` | `Creative × text` | `Dev × icon` | `Dev × text` | `OS × icon` | `OS × text` | `Office × icon` | `Office × text` | `Scientific × icon` | `Scientific × text` | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen_Qwen3-VL-4B-Instruct` | 0.2188 | 0.5787 | 0.2168 | 0.7121 | 0.3103 | 0.8052 | 0.3708 | 0.7944 | 0.4528 | 0.8362 | 0.3273 | 0.8125 | 0.5769 |
| `Qwen_Qwen3.5-4B` | 0.1875 | 0.5838 | 0.2378 | 0.7172 | 0.3103 | 0.7857 | 0.4045 | 0.7290 | 0.4906 | 0.7966 | 0.2909 | 0.8194 | 0.5693 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.2812 | 0.6041 | 0.2657 | 0.7879 | 0.2000 | 0.7468 | 0.2135 | 0.6636 | 0.4717 | 0.8418 | 0.3455 | 0.8472 | 0.5686 |
| `Qwen_Qwen3.5-9B` | 0.1875 | 0.5939 | 0.2448 | 0.5404 | 0.4138 | 0.8377 | 0.3708 | 0.6822 | 0.5660 | 0.7684 | 0.3364 | 0.7639 | 0.5560 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.1719 | 0.6294 | 0.1469 | 0.7323 | 0.2207 | 0.8117 | 0.2247 | 0.7383 | 0.3208 | 0.8362 | 0.3273 | 0.7917 | 0.5515 |
| `Qwen_Qwen3.5-2B` | 0.2344 | 0.4822 | 0.2587 | 0.6414 | 0.2483 | 0.7273 | 0.4382 | 0.6822 | 0.5094 | 0.7401 | 0.3818 | 0.7917 | 0.5364 |
| `Tongyi-MAI_MAI-UI-2B` | 0.1719 | 0.4365 | 0.2028 | 0.6515 | 0.2621 | 0.7403 | 0.3708 | 0.7477 | 0.3962 | 0.8305 | 0.2909 | 0.7778 | 0.5262 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.1719 | 0.5381 | 0.1888 | 0.6111 | 0.1655 | 0.6883 | 0.2022 | 0.5607 | 0.4151 | 0.7966 | 0.2182 | 0.7778 | 0.4883 |
| `meituan_EvoCUA-8B-20260105` | 0.1094 | 0.3401 | 0.1119 | 0.6313 | 0.1448 | 0.6948 | 0.1573 | 0.5607 | 0.2075 | 0.7119 | 0.2182 | 0.7014 | 0.4295 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.1719 | 0.3299 | 0.1818 | 0.5556 | 0.1241 | 0.6104 | 0.2697 | 0.5888 | 0.3396 | 0.6893 | 0.2091 | 0.6181 | 0.4194 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.0625 | 0.2284 | 0.0769 | 0.5051 | 0.0552 | 0.5195 | 0.1236 | 0.3364 | 0.1509 | 0.5989 | 0.2091 | 0.6042 | 0.3283 |
| `Qwen_Qwen3.5-27B` | 0.0000 | 0.0863 | 0.0280 | 0.1263 | 0.0345 | 0.1753 | 0.0000 | 0.0654 | 0.0377 | 0.0565 | 0.0455 | 0.0625 | 0.0702 |
