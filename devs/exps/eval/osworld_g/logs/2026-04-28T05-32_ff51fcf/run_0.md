# osworld_g @ 2026-04-28T05-32_ff51fcf · run_0

- **Commit**: `ff51fcf` — `agents(grounding): clean grounding.action route — remove _is_navigation_task gate`. Pipeline-relevant changes vs `828ea14` (the prior osworld_g snapshot at `logs/2026-04-27T23-21_828ea14/`):
  - Envs now declare `task_type="grounding.action"` (was `"navigation"`); registry routes to `<family>:desktop:grounding.action` adapters.
  - `Qwen3VLBaseAdapter._is_navigation_task` gone; `system_prompt` now a class-level field default per subclass (Navigation: pinned; GroundingAction: `None`).
  - EvoCUA gains GroundingActionAdapter; UI-TARS / UI-TARS-1.5 GroundingAction adapters pin `user_prompt_template=GROUNDING_USER_PROMPT`.
  - 10 grounding YAMLs drop the now-redundant `system_prompt: ""` / `summary_template: "{instruction}"` / `user_prompt_template:` overrides — adapter classes ship those defaults.
  - Smoke test (`.logs/grounding_smoke_new/` vs `.logs/grounding_smoke/`) confirmed prompts (system+tools+user) and per-task MERs are byte-identical across 10/10 cells: pure refactor, no semantic change.
- **Host / GPUs**: `gpublaze` / 2-GPU dispatcher on GPU 2,3 (shared with another tenant's small workload, ~2 GiB each — sglang at default mem_fraction=0.79 fit comfortably alongside).
- **Started**: `2026-04-28 12:34 UTC`
- **Last updated**: `2026-04-28 16:18 UTC` (campaign duration: 224 min)
- **Notes**: Eligible task count: **510** of 564 (54 `box_type=refusal` tasks dropped via `--filter "lambda m: not m.others.get('exclude_reason')"` + `--env-kwargs '{"use_extra_tools": false}'`). Phase 1 (9 tp=1 models, 2-way parallel) → Phase 2 (Qwen3-VL-32B + Qwen3.5-27B at tp=2 sequential on both GPUs). Qwen3.5-27B was pinned tp=2 in `lite/utils/agents.py` (down from default tp=4) to fit on the 2-GPU eval slot.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3.5-27B`              | 510/510 | **0.6843** |
| `Qwen/Qwen3-VL-32B-Instruct`    | 510/510 | 0.6824 |
| `Qwen/Qwen3-VL-8B-Instruct`     | 510/510 | 0.6314 |
| `Qwen/Qwen3-VL-4B-Instruct`     | 510/510 | 0.6294 |
| `Qwen/Qwen3.5-9B`               | 510/510 | 0.6196 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 510/510 | 0.5922 |
| `Qwen/Qwen3.5-4B`               | 510/510 | 0.5627 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 510/510 | 0.5314 |
| `meituan/EvoCUA-8B-20260105`    | 510/510 | 0.5196 |
| `Qwen/Qwen3.5-2B`               | 510/510 | 0.4922 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 510/510 | 0.4863 |

## Highlights

- **27B / 32B class wins both rankings** by a hair: Qwen3.5-27B (0.684) > Qwen3-VL-32B-Instruct (0.682). Both substantially above the rest of the pack; the next-best (Qwen3-VL-8B at 0.631) is ~5pp behind.
- **Qwen3-VL family monotonic in size**: 32B (0.682) > 8B (0.631) ≈ 4B (0.629) >> 2B (0.531). 4B sits within 0.2pp of 8B — a 2× compute saving for the same MER, worth investigating which subset 4B trades wins on.
- **Qwen3.5 family also monotonic**: 27B (0.684) > 9B (0.620) > 4B (0.563) > 2B (0.492). The 27B / 9B gap (6pp) and 9B / 4B gap (6pp) suggest scaling hasn't saturated.
- **Cross-family at matched size**: at 8B-class, Qwen3-VL-8B (0.631) > Qwen3.5-9B (0.620) > EvoCUA-8B (0.520) > UI-TARS-7B-DPO (0.486). Qwen3-VL still beats Qwen3.5 same-size on grounding (consistent with the prior round on `828ea14`), but the gap shrank from ~8pp to ~1pp this run — most of the prior gap was the YAML-vs-adapter-default plumbing now removed.
- **EvoCUA-8B sits mid-pack** (0.520) — between Qwen3-VL-2B (0.531) and Qwen3.5-2B (0.492), behind the 9B / 8B class. The CUA-specific SFT distribution doesn't translate to a single-step click advantage.
- **UI-TARS-1.5 ≫ UI-TARS-7B-DPO** (0.592 vs 0.486) — the 1.5 generation advantage holds; DPO is the weakest model in the matrix on osworld_g.
- **Pipeline-aware path key worked**: `2026-04-28T03-16_a671f2b/run_0/` was renamed to `2026-04-28T05-32_ff51fcf/run_0/` after the grounding-route refactor, and the dispatcher resumed from there — all `num_finished == 510` carried over without re-rolling tasks. Verified pre-launch via the `.logs/grounding_smoke_new/` byte-identical comparison.

## Breakdown

Two-step regen: `reaggregate_breakdown.py --axes paper_category,box_type,GUI_types .exps/eval/osworld_g/2026-04-28T05-32_ff51fcf/run_0/` → `render_breakdown.py .exps/eval/osworld_g/2026-04-28T05-32_ff51fcf/run_0/`. The wide `GUI_types` axis (33 columns) is omitted here — re-run the renderer if needed.

### Breakdown — by `paper_category`

| Model | `element_recognition` | `fine_grained_manipulation` | `layout_understanding` | `text_matching` | Avg |
|---|---:|---:|---:|---:|---:|
| `Qwen_Qwen3-VL-32B-Instruct` | 0.7091 | 0.5833 | 0.7113 | 0.7875 | 0.7121 |
| `Qwen_Qwen3.5-27B` | 0.7157 | 0.5985 | 0.6904 | 0.7833 | 0.7099 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.6275 | 0.5758 | 0.6360 | 0.7875 | 0.6641 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.6634 | 0.5076 | 0.6485 | 0.7625 | 0.6630 |
| `Qwen_Qwen3.5-9B` | 0.6536 | 0.4697 | 0.6778 | 0.7417 | 0.6565 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.5915 | 0.5303 | 0.6109 | 0.7417 | 0.6270 |
| `Qwen_Qwen3.5-4B` | 0.5850 | 0.4470 | 0.6318 | 0.7125 | 0.6107 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.5425 | 0.4470 | 0.6067 | 0.6750 | 0.5802 |
| `meituan_EvoCUA-8B-20260105` | 0.5000 | 0.5000 | 0.5314 | 0.7208 | 0.5660 |
| `Qwen_Qwen3.5-2B` | 0.5327 | 0.3788 | 0.5732 | 0.6042 | 0.5398 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4902 | 0.4015 | 0.5105 | 0.6250 | 0.5180 |

### Breakdown — by `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| `Qwen_Qwen3.5-27B` | 0.6723 | 0.8250 | 0.6843 |
| `Qwen_Qwen3-VL-32B-Instruct` | 0.6702 | 0.8250 | 0.6824 |
| `Qwen_Qwen3-VL-8B-Instruct` | 0.6149 | 0.8250 | 0.6314 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.6170 | 0.7750 | 0.6294 |
| `Qwen_Qwen3.5-9B` | 0.6043 | 0.8000 | 0.6196 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.5766 | 0.7750 | 0.5922 |
| `Qwen_Qwen3.5-4B` | 0.5426 | 0.8000 | 0.5627 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.5191 | 0.6750 | 0.5314 |
| `meituan_EvoCUA-8B-20260105` | 0.5021 | 0.7250 | 0.5196 |
| `Qwen_Qwen3.5-2B` | 0.4766 | 0.6750 | 0.4922 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4660 | 0.7250 | 0.4863 |
