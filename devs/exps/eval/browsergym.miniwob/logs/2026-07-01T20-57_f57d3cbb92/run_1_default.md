# browsergym.miniwob @ 2026-07-01T20-57_f57d3cbb92 · run_1_default

- **Commit**: `f57d3cbb92` — `update eval pipeline`
- **Host / GPUs**: `gpublaze` / `0,1,3`
- **Artifacts**: `.exps/eval/browsergym.miniwob/2026-07-01T20-57_f57d3cbb92/run_1_default/`
- **Started**: `2026-07-02 12:12 PDT`
- **Last updated**: `2026-07-09`
- **Notes**: `EVAL_MODE=default`; env-server `http://localhost:30100` with token `lhr`; `HF_HOME=/srv/share/huggingface`. One-GPU models run on one of `0,1,3`; 32B/27B models run on two GPUs.

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3-VL-32B-Thinking`    | 125/125 | 0.6880 |
| `gpt-5.5`                       | 125/125 | 0.6320 |
| `Qwen/Qwen3-VL-32B-Instruct`    | 125/125 | 0.5920 |
| `Qwen/Qwen3-VL-8B-Thinking`     | 125/125 | 0.5840 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 125/125 | 0.5440 |
| `Qwen/Qwen3-VL-4B-Thinking`     | 125/125 | 0.5280 |
| `Qwen/Qwen3-VL-4B-Instruct`     | 125/125 | 0.4880 |
| `Qwen/Qwen3.5-9B`               | 125/125 | 0.4720 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 125/125 | 0.3840 |
| `Qwen/Qwen3-VL-2B-Thinking`     | 125/125 | 0.3520 |
| `Qwen/Qwen3.5-4B`               | 125/125 | 0.2400 |
| `Qwen/Qwen3.5-27B`              | 125/125 | 0.2000 |
| `Qwen/Qwen3.5-2B`               | 125/125 | 0.1120 |

## Highlights

- Default is a screenshot+coordinate setting. `Qwen3-VL-32B-Thinking` (0.6880) > `Qwen3-VL-32B-Instruct` (0.5920) > `Qwen3-VL-8B-Thinking` (0.5840); 32B Thinking is the clear best default agent.
- Qwen3-VL scales cleanly in default mode. Instruct is monotonic from 2B -> 32B (0.3840 -> 0.5920), and Thinking is also monotonic across 2B/4B/8B/32B (0.3520 -> 0.6880).
- Thinking is size-sensitive for Qwen3-VL: it hurts at 2B (0.3520 vs 0.3840), then helps at 4B/8B/32B, with the largest gain at 32B (+0.0960).
- Qwen3.5 default does not scale normally: `9B` (0.4720) > `4B` (0.2400) > `27B` (0.2000) > `2B` (0.1120). The weak 27B result looks specific to default-mode grounding, since the same checkpoint is strong in text-only mode.
- `Qwen3.5-27B` default failures are consistent with coordinate grounding trouble, not a proven harness conversion bug. It loses many tasks that 27B text-only solves, including click, text/form, reasoning/table, and selection/navigation cases.
