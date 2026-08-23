# osworld_2 @ 2026-07-17T19-06_551131f803 · run_0

- **Commit**: `551131f803` — `eval: update osworld run_0 qwen3-vl-2b`
- **Host / GPUs**: `gpublaze` / `6-7`
- **Artifacts**: `.exps/eval/osworld_2/2026-07-17T19-06_551131f803/run_0/`
- **Started**: `2026-07-18 01:23 PDT`
- **Last updated**: `2026-07-18 04:49 PDT`
- **Notes**: Qwen3-VL-4B is carried forward from `e2662ef099`; the only tracked `osworld_2` pipeline change since then is GPT config-only, so the Qwen3-VL artifact was copied into this commit dir. Fresh runs use `HF_HOME=/srv/share/huggingface`, `CUA_LITE_ENV_SERVER_URL=http://localhost:30103`, `CUA_LITE_ENV_SERVER_TOKEN=lhr`, and API keys from `~/env.sh`. Current `osworld_2/run.sh` only has committed config cases for `qwen3_vl`, `gpt`, and `claude`; the requested Qwen3.5 / MAI-UI / EvoCUA / UI-Voyager families have no `scripts/configs/<family>/default/osworld_2.yaml` and are not launched in this campaign.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-5.5` | 0.399 |
| `Qwen/Qwen3-VL-32B-Instruct` | 82/82 | 0.0146 |
| `Qwen/Qwen3-VL-4B-Instruct` | 82/82 | 0.0129 |
| `Qwen/Qwen3-VL-8B-Instruct` | 82/82 | 0.0079 |
| `Qwen/Qwen3-VL-2B-Instruct` | 82/82 | 0.0061 |

## Highlights


