# mobilegym @ 2026-07-08T16-14_2a40f5258e · run_0

- **Commit**: `2a40f5258e` — `eval androidworld`
- **Host / GPUs**: `gpublaze` / `6-7`
- **Artifacts**: `.exps/eval/mobilegym/2026-07-08T20-40_2a40f5258e/run_0`
- **Started**: `2026-07-07 20:13 PDT`
- **Last updated**: `2026-07-08 02:43 PDT`
- **Notes**: env-server at `http://localhost:30101`; `HF_HOME=/srv/share/huggingface` first, falling back to user cache if a model is unavailable there. Rollout eval split contains 256 tasks.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `gpt-5.5`                            | 256/256 | 0.6504 |
| `Qwen/Qwen3.5-27B`                   | 256/256 | 0.3707 |
| `Qwen/Qwen3-VL-32B-Instruct`         | 256/256 | 0.3102 |
| `Qwen/Qwen3.5-9B`                    | 256/256 | 0.2902 |
| `Qwen/Qwen3.5-4B`                    | 256/256 | 0.2615 |
| `ByteDance-Seed/UI-TARS-7B-DPO`      | 256/256 | 0.2465 |
| `Qwen/Qwen3-VL-8B-Instruct`          | 256/256 | 0.2392 |
| `Qwen/Qwen3-VL-4B-Instruct`          | 256/256 | 0.2349 |
| `ByteDance-Seed/UI-TARS-1.5-7B`      | 256/256 | 0.2219 |
| `stepfun-ai/GELab-Zero-4B-preview`   | 256/256 | 0.2179 |
| `Tongyi-MAI/MAI-UI-2B`               | 256/256 | 0.1821 |
| `Qwen/Qwen3.5-2B`                    | 256/256 | 0.1492 |
| `MarsXL/UI-Voyager`                  | 256/256 | 0.1141 |
| `Qwen/Qwen3-VL-2B-Instruct`          | 256/256 | 0.1141 |

## Highlights

- All requested models completed the full 256-task eval split. `Qwen/Qwen3.5-27B` is the clear top result at 0.3707, ahead of `Qwen/Qwen3-VL-32B-Instruct` by +0.0605 and `Qwen/Qwen3.5-9B` by +0.0805.
- Qwen3.5 scales cleanly across tested sizes: 2B -> 4B -> 9B -> 27B improves 0.1492 -> 0.2615 -> 0.2902 -> 0.3707. The largest gain is 2B->4B (+0.1123), while 9B->27B still gives a substantial +0.0805.
- Qwen3-VL also benefits from scale, but less smoothly: 2B -> 4B jumps from 0.1141 to 0.2349, 4B->8B is nearly flat (+0.0043), then 32B rises to 0.3102. The 32B VL model beats all non-Qwen3.5 models, but remains below Qwen3.5-27B.
- Within similar size ranges, specialized UI agents are competitive but do not lead: `UI-TARS-7B-DPO` reaches 0.2465, beating Qwen3-VL-8B/4B while trailing Qwen3.5-4B by 0.0150. `UI-TARS-1.5-7B` lands lower at 0.2219.
- Small-model performance varies sharply by family. `MAI-UI-2B` at 0.1821 beats both Qwen 2B variants, while `Qwen3.5-2B` at 0.1492 is ahead of `Qwen3-VL-2B-Instruct` and `UI-Voyager`, both at 0.1141.
- Operational note: `Qwen/Qwen3-VL-32B-Instruct` was paused at 133/256 for an env-server restart, then resumed in the same artifact root. `ByteDance-Seed/UI-TARS-1.5-7B` launched under `2026-07-07T22-05_015a875c99` after a doc-equivalent HEAD advance; artifacts were copied back into this run root.
