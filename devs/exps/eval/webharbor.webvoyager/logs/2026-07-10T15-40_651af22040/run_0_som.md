# webharbor.webvoyager @ 2026-07-10T15-40_651af22040 · run_0_som

- **Commit**: `651af22040ed43136ade5a83457ef693cf93eee7` — `add gpt-5.6-sol`
- **Host / GPUs**: `gpublaze` / not recorded in summary
- **Artifacts**: `.exps/eval/webharbor.webvoyager/2026-07-10T15-40_651af22040/run_0_som/`
- **Started**: `2026-07-10 15:40 PDT`
- **Last updated**: `2026-07-12 12:01 PDT`
- **Notes**: SoM-config WebHarbor WebVoyager run. Results are read from the aggregated `run_0_som.json` and cross-checked against each model's `summary.json`. All listed rows are complete over the full 643-task eval split. The aggregate JSON stores `config_id: default`, but this Markdown reports the actual `run_0_som` configuration.
- **Latest event**: generated `run_0_som` summary and figures

## Results

| Config | Model | Finished | Mean episode return |
|---|---|---|---|
| `som` | `Qwen/Qwen3-VL-32B-Instruct` | 643/643 | 0.5117 |
| `som` | `Qwen/Qwen3.5-9B` | 643/643 | 0.4666 |
| `som` | `Qwen/Qwen3-VL-8B-Instruct` | 643/643 | 0.4339 |
| `som` | `Qwen/Qwen3.5-27B` | 643/643 | 0.3950 |
| `som` | `Qwen/Qwen3-VL-4B-Instruct` | 643/643 | 0.3779 |
| `som` | `Qwen/Qwen3.5-4B` | 643/643 | 0.2768 |
| `som` | `Qwen/Qwen3.5-2B` | 643/643 | 0.1073 |
| `som` | `Qwen/Qwen3-VL-2B-Instruct` | 643/643 | 0.0109 |

## Highlights

- Progress: 8/8 config runs complete; 0 partial; 0 empty.
- Current best completed row: `Qwen/Qwen3-VL-32B-Instruct` `som` at 0.5117.
- Best Qwen3.5 row: `Qwen/Qwen3.5-9B` `som` at 0.4666.
- Figures: `figures/run_0_som/bars.png`, `figures/run_0_som/scaling.png`.
