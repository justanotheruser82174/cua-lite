# browsergym.miniwob @ 2026-07-01T20-57_f57d3cbb92 · run_1_textonly

- **Commit**: `f57d3cbb92` — `update eval pipeline`
- **Host / GPUs**: `gpublaze` / `0,1,3`
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-07-01T20-57_f57d3cbb92/run_1_textonly/`
- **Started**: `2026-07-02 12:12 PDT`
- **Last updated**: `2026-07-02 17:35 PDT`
- **Notes**: `EVAL_MODE=text_only`; env-server `http://localhost:30100` with token `lhr`; `HF_HOME=/srv/share/huggingface`. One-GPU models run on one of `0,1,3`; 32B/27B models run on two GPUs. `think_off` is the default setting; `think_on` variants use `EVAL_ENABLE_THINKING=true` and artifact suffix `__think_on`.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3-VL-32B-Thinking` `think_on` | 125/125 | 0.6480 |
| `Qwen/Qwen3.5-27B` `think_off` | 125/125 | 0.6240 |
| `Qwen/Qwen3.5-4B` `think_on`    | 125/125 | 0.6080 |
| `Qwen/Qwen3.5-9B` `think_on`    | 125/125 | 0.6080 |
| `Qwen/Qwen3-VL-32B-Instruct` `think_off` | 125/125 | 0.5840 |
| `Qwen/Qwen3-VL-8B-Thinking` `think_on` | 125/125 | 0.5760 |
| `Qwen/Qwen3.5-4B` `think_off`  | 125/125 | 0.5600 |
| `Qwen/Qwen3.5-27B` `think_on`   | 125/125 | 0.5520 |
| `Qwen/Qwen3-VL-4B-Thinking` `think_on` | 125/125 | 0.5040 |
| `Qwen/Qwen3-VL-8B-Instruct` `think_off` | 125/125 | 0.4960 |
| `Qwen/Qwen3-VL-4B-Instruct` `think_off` | 125/125 | 0.4720 |
| `Qwen/Qwen3.5-9B` `think_off`  | 125/125 | 0.4480 |
| `Qwen/Qwen3.5-2B` `think_on`    | 125/125 | 0.3680 |
| `Qwen/Qwen3-VL-2B-Instruct` `think_off` | 125/125 | 0.3600 |
| `Qwen/Qwen3.5-2B` `think_off`  | 125/125 | 0.3280 |
| `Qwen/Qwen3-VL-2B-Thinking` `think_on` | 125/125 | 0.2640 |

## Highlights

- Text-only uses AXTree/bid actions instead of screenshot+coordinate grounding. This is much friendlier to Qwen3.5: `Qwen3.5-27B` `think_off` rises from 0.2000 in default to 0.6240 here.
- Top text-only agents: `Qwen3-VL-32B-Thinking` `think_on` (0.6480) > `Qwen3.5-27B` `think_off` (0.6240) > `Qwen3.5-{4B,9B}` `think_on` (0.6080).
- Qwen3-VL Thinking scales monotonically from 2B -> 32B (0.2640 -> 0.6480). Relative to Instruct, Thinking hurts at 2B but helps at 4B/8B/32B.
- Qwen3.5 thinking-on is mixed: it helps 2B (0.3680 vs 0.3280), 4B (0.6080 vs 0.5600), and 9B (0.6080 vs 0.4480), but hurts 27B (0.5520 vs 0.6240).
- Qwen3.5 text-only is not a clean size curve. For reporting, treat thinking-on/off as separate agent configs: 4B thinking-on ties 9B thinking-on, 9B thinking-off trails 4B thinking-off, and 27B is strongest only with thinking off.
- Text-only improves Qwen3.5 mainly by replacing fragile coordinate targeting with bid-level actions. Drag/resize and widget tasks remain weak spots across most agents.
