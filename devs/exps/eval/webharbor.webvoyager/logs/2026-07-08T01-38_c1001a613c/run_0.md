# webharbor.webvoyager @ 2026-07-08T01-38_c1001a613c · run_0

- **Commit**: `c1001a613cb7eb953ab6b16b637ce3c13a1053da` — `Fix WebHarbor eval path reuse`
- **Host / GPUs**: `gpublaze` / `6-7`
- **Artifacts**: `.exps/eval/webharbor.webvoyager/2026-07-08T01-38_c1001a613c/run_0/`
- **Started**: `2026-07-08 01:38 PDT`
- **Last updated**: `2026-07-10 01:52 PDT`
- **Notes**: Qwen3-VL and Qwen3.5 are the requested model families with committed `webharbor.webvoyager/{default,som}.yaml` configs. The other requested agents currently have no WebHarbor WebVoyager config/run.sh mapping and are listed below. This run uses the env-server at `http://localhost:30102`; WebHarbor strict isolation recommends env-server restart between models, but this campaign keeps the shared server to honor the requested GPU-parallel schedule. Results are aggregated from per-task summaries because two-pass WebHarbor runs overwrite the root summary with the second pass only; resumed tasks may store per-sample summaries under `sample_00/summary.json`.
- **Latest event**: finished `Qwen/Qwen3-VL-8B-Instruct` `som` attempt 1 on GPU 6

## Results

| Config | Model | Finished | Mean episode return |
|---|---|---|---|
| `som` | `Qwen/Qwen3-VL-32B-Instruct` | 643/643 | 0.5117 |
| `default` | `Qwen/Qwen3.5-27B` | 643/643 | 0.5070 |
| `som` | `Qwen/Qwen3.5-27B` | 643/643 | 0.3950 |
| `som` | `Qwen/Qwen3-VL-4B-Instruct` | 643/643 | 0.3795 |
| `default` | `Qwen/Qwen3.5-9B` | 643/643 | 0.3561 |
| `som` | `Qwen/Qwen3.5-4B` | 643/643 | 0.2768 |
| `default` | `Qwen/Qwen3-VL-32B-Instruct` | 643/643 | 0.1866 |
| `default` | `Qwen/Qwen3-VL-4B-Instruct` | 643/643 | 0.1135 |
| `default` | `Qwen/Qwen3.5-4B` | 643/643 | 0.0933 |
| `som` | `Qwen/Qwen3.5-2B` | 643/643 | 0.0902 |
| `default` | `Qwen/Qwen3-VL-8B-Instruct` | 643/643 | 0.0840 |
| `som` | `Qwen/Qwen3-VL-2B-Instruct` | 643/643 | 0.0109 |
| `default` | `Qwen/Qwen3.5-2B` | 643/643 | 0.0016 |
| `default` | `Qwen/Qwen3-VL-2B-Instruct` | 643/643 | 0.0000 |
| `som` | ⚠️ `Qwen/Qwen3.5-9B` | _**317/643**_ | _**0.5678**_ |

## Highlights

- Progress: 15/16 config runs complete; 1 partial; 0 not started or no summary yet.
- Current best completed row: `Qwen/Qwen3-VL-32B-Instruct` `som` at 0.5117.
- Requested agents not run because no `scripts/configs/<agent>/default/webharbor.webvoyager/{default,som}.yaml` config/run.sh case exists: `ByteDance-Seed/UI-TARS-7B-DPO`, `ByteDance-Seed/UI-TARS-1.5-7B`, `Tongyi-MAI/MAI-UI-2B`, `MarsXL/UI-Voyager`, `stepfun-ai/GELab-Zero-4B-preview`.
