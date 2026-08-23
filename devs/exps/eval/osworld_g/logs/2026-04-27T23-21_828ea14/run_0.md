# osworld_g @ 2026-04-27T23-21_828ea14 · run_0

- **Commit at campaign kickoff**: `828ea143` — `eval(grounding): bump concurrency 16 → 64 in run.sh` (chain on top of `7de99d9 feat(osworld_g): add 564-task grounding env + 10 rollout configs`).
- **Caveat — code drifted mid-campaign**: this snapshot's path is named after the *kickoff* commit, but the run produced data under a moving HEAD. Two changes landed mid-campaign and affected later rollouts (not earlier ones):
  - `9604e62 eval(grounding): use_extra_tools env_kwarg + to_thread hook fix + snapshots` — which (a) bundled `--env-kwargs '{"use_extra_tools": false}'` into `devs/exps/eval/osworld_g/run.sh` so the agent's tool list dropped `report_infeasible` for non-refusal tasks, and (b) wrapped TrajectoryLogger hooks in `asyncio.to_thread` (~4× per-slot throughput on grounding evals).
  - The earliest rollouts in the table below ran before either landed; later rollouts (the resume cycle for Qwen3-VL-8B's 3 deleted `report_infeasible` task dirs, and slot-1's Qwen3.5-{4B, 2B, 9B, EvoCUA}) ran after both. The **final per-task summaries are produced by whichever code state was live at task-run time** — meaning numbers in the table are a mix of pre- and post-fix runs, and pure 828ea14 reproducibility is not exact. Future campaigns should commit-before-launch.
- **Host / GPUs**: `gpublaze` / 4-GPU dispatcher on 0-3 then 2-GPU on 2-3
- **Notes**: First grounding-env campaign on the standardized layout. Eligible task count: **510** of 564 (54 `box_type=refusal` tasks dropped via `--filter "lambda m: not m.others.get('exclude_reason')"` + `--env-kwargs '{"use_extra_tools": false}'` to also strip `report_infeasible` from the agent's tool list). Raw artifacts: `.exps/eval/osworld_g/2026-04-27T23-21_828ea14/run_0/<slug>/`. Wave 2 (Qwen3-VL-32B, tp=2) and Wave 3 (Qwen3.5-27B, tp=2 fallback) deferred — campaign was paused mid-Wave-2 to debug the prompt-template overhead found in single-step grounding.

## Results

Sorted by MER desc.

| Model | Finished | Mean episode return |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct`     | 510/510 | **0.5784** |
| `Qwen/Qwen3-VL-4B-Instruct`     | 510/510 | 0.5667 |
| `ByteDance-Seed/UI-TARS-1.5-7B` | 510/510 | 0.5255 |
| `Qwen/Qwen3.5-9B`               | 510/510 | 0.4961 |
| `ByteDance-Seed/UI-TARS-7B-DPO` | 510/510 | 0.4843 |
| `Qwen/Qwen3-VL-2B-Instruct`     | 510/510 | 0.4765 |
| `meituan/EvoCUA-8B-20260105`    | 510/510 | 0.4725 |
| `Qwen/Qwen3.5-4B`               | 510/510 | 0.4667 |
| `Qwen/Qwen3.5-2B`               | 510/510 | 0.4196 |
| ⚠️ `Qwen/Qwen3-VL-32B-Instruct` | _**0/510**_ | _**—**_ |
| ⚠️ `Qwen/Qwen3.5-27B`           | _**0/510**_ | _**—**_ |

## Highlights

- **Qwen3-VL family wins:** 8B (0.578) > 4B (0.567) >> 2B (0.477). Clean monotonic scaling.
- **UI-TARS-1.5 > 7B-DPO** (0.526 vs 0.484) — the 1.5 generation upgrade replicates here.
- **Qwen3.5 vs Qwen3-VL on grounding:** Qwen3-VL-2B (0.477) ≈ Qwen3.5-9B (0.496) — a 4.5× larger Qwen3.5 only matches a 2B Qwen3-VL. Vision backbone matters more than text reasoning for click prediction.
- **EvoCUA-8B** (0.473) sits mid-pack — competitive with Qwen3-VL-2B and Qwen3.5-{2B,4B,9B}, behind UI-TARS and the larger Qwen3-VL.
- **Wave 2 + Wave 3 deferred:** campaign paused before Qwen3-VL-32B and Qwen3.5-27B ran; will resume after the prompt-template debug round (`scripts/configs/{qwen3_vl,qwen3_5,evocua,ui_tars,ui_tars_15_v1}/default/osworld_g.yaml` overrides to strip the navigation-flavoured `Please generate the next move / Previous actions: None` preamble that's irrelevant for T=1 single-step grounding).

## report_infeasible behavior

Both filter (`exclude_reason="refusal"` skips 54 tasks) AND `use_extra_tools=False` (drops `report_infeasible` from the agent's action list) are active for this campaign. Pre-campaign ad-hoc runs (before `use_extra_tools=False` landed) showed Qwen3-VL-8B emitting `report_infeasible` on 3/510 non-refusal tasks (false-positive give-ups, all scored 0). The flag is now in `devs/exps/eval/osworld_g/run.sh` to prevent recurrence.

## Breakdown

Two-step regen: `reaggregate_breakdown.py --axes paper_category,box_type,GUI_types .exps/eval/osworld_g/2026-04-27T23-21_828ea14/run_0/` → `render_breakdown.py .exps/eval/osworld_g/2026-04-27T23-21_828ea14/run_0/`. The wide `GUI_types` axis (33 columns) is omitted here — re-run the renderer if needed.

### Breakdown — by `paper_category`

OSWorld-G's paper-canonical 4-class split (refusal excluded by the filter): `text_matching`, `element_recognition`, `layout_understanding`, `fine_grained_manipulation`. Multi-tag — a task may belong to >1 category, so per-row sums exceed n_tasks.

| Model | `element_recognition` | `fine_grained_manipulation` | `layout_understanding` | `text_matching` | Avg |
|---|---:|---:|---:|---:|---:|
| `Qwen_Qwen3-VL-8B-Instruct` | 0.5654 | 0.5379 | 0.5732 | 0.7417 | 0.6096 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.5588 | 0.4848 | 0.6025 | 0.7417 | 0.6074 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.5163 | 0.5076 | 0.5439 | 0.6875 | 0.5671 |
| `Qwen_Qwen3.5-9B` | 0.4935 | 0.4167 | 0.5439 | 0.6833 | 0.5453 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.4804 | 0.4242 | 0.5146 | 0.6500 | 0.5256 |
| `meituan_EvoCUA-8B-20260105` | 0.4444 | 0.4242 | 0.5063 | 0.6750 | 0.5180 |
| `Qwen_Qwen3.5-4B` | 0.4608 | 0.4242 | 0.5021 | 0.6583 | 0.5180 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4869 | 0.4091 | 0.5021 | 0.6208 | 0.5147 |
| `Qwen_Qwen3.5-2B` | 0.4118 | 0.3788 | 0.4895 | 0.5625 | 0.4667 |

### Breakdown — by `box_type`

| Model | `bbox` | `polygon` | Avg |
|---|---:|---:|---:|
| `Qwen_Qwen3-VL-8B-Instruct` | 0.5553 | 0.8500 | 0.5784 |
| `Qwen_Qwen3-VL-4B-Instruct` | 0.5511 | 0.7500 | 0.5667 |
| `ByteDance-Seed_UI-TARS-1.5-7B` | 0.5021 | 0.8000 | 0.5255 |
| `Qwen_Qwen3.5-9B` | 0.4766 | 0.7250 | 0.4961 |
| `ByteDance-Seed_UI-TARS-7B-DPO` | 0.4596 | 0.7750 | 0.4843 |
| `Qwen_Qwen3-VL-2B-Instruct` | 0.4574 | 0.7000 | 0.4765 |
| `meituan_EvoCUA-8B-20260105` | 0.4532 | 0.7000 | 0.4725 |
| `Qwen_Qwen3.5-4B` | 0.4511 | 0.6500 | 0.4667 |
| `Qwen_Qwen3.5-2B` | 0.4021 | 0.6250 | 0.4196 |
