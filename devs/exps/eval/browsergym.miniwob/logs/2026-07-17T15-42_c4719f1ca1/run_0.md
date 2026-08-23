# browsergym.miniwob @ 2026-07-17T15-42_c4719f1ca1 · run_0

- **Commit**: `f28f81402b` — `use per user log path`
- **Host / GPUs**: `gpublaze` / `0,3`
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-07-17T15-42_c4719f1ca1/run_0/`
- **Started**: `2026-07-16 18:29 PDT`
- **Last updated**: `2026-07-17 17:32 PDT`
- **Notes**: `EVAL_MODE=default`; env-server `http://localhost:30103` with token `lhr`; `HF_HOME=/srv/share/huggingface`. `Qwen3-VL-{4,8}B-Instruct` were already complete when this snapshot was created. Of the requested non-Qwen local agents, `ByteDance-Seed/UI-TARS-*`, `Tongyi-MAI/MAI-UI-2B`, `MarsXL/UI-Voyager`, and `stepfun-ai/GELab-Zero-4B-preview` do not have `scripts/configs/<agent>/default/browsergym.miniwob/default.yaml` entries in this checkout.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| gpt-5.5                         | 125/125 | 0.8400 |
| `Qwen/Qwen3.5-27B`              | 125/125 | 0.7200 |
| `Qwen/Qwen3-VL-32B-Instruct`    | 125/125 | 0.6960 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 125/125 | 0.6400 |
| `Qwen/Qwen3.5-9B`               | 125/125 | 0.5280 |
| `Qwen/Qwen3-VL-4B-Instruct`     | 125/125 | 0.5280 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 125/125 | 0.4000 |
| `Qwen/Qwen3.5-4B`               | 125/125 | 0.2480 |
| `Qwen/Qwen3.5-2B`               | 125/125 | 0.1280 |

## Highlights

- `gpt-5.5` is the current best completed default-mode run at 0.8400
- `Qwen/Qwen3-VL-Instruct` scales cleanly across completed sizes: 2B 0.4000, 4B 0.5280, 8B 0.6400, 32B 0.6960.
- Completed Qwen3.5 default runs improve from 2B 0.1280 to 4B 0.2480 to 9B 0.5280 to 27B 0.7200.
- `Qwen/Qwen3-VL-32B-Instruct` needed one mop-up relaunch after an initial 124/125 partial; the second attempt completed 125/125.
- `Qwen/Qwen3-VL-32B-Instruct` and `Qwen/Qwen3.5-27B` ran with tensor parallelism on the approved `0,3` GPU pair.
