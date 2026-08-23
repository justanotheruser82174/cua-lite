# LibreOffice Calc — Synth Plan

> Keep in sync with code. Implementation: [`libreoffice_calc.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/libreoffice_calc.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/libreoffice_calc.md`](/devs/envs/lite.osworld/perturb/libreoffice_calc.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain libreoffice_calc` for live numbers. Synth N=266, eval N=46 (1 infeasibility filtered). **Calc has the most mechanical bridge fix in the entire codebase**: one config-step change closes a 96pp gap.

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `save_protocol.ctrl_s_only` | 100% | 0% | +100 | 🔴 | **Single biggest fix in the repo** — add `{"type": "open", "parameters": {"path": "/home/user/Desktop/<name>.xlsx"}}` to every `f_calc_*` config. Synth already has Ctrl+S; just missing the `open` step → no Save-As trap. |
| `source_provenance.synth_inline_openpyxl` | 100% | 0% | +100 | 🔴 | Populate `assets/synth/data/csv/` from FRED / Census / SheetCopilot; switch ~30-50% of templates from inline `openpyxl.Workbook()` → curated `.xlsx` `download` step |
| `skill_class.pivot_table` | 0% | 11% | -11 | ❌ | Add 4-6 pivot-table templates (single-field, two-field, value+row+col, subtotal) |
| `skill_class.sheet_print` | 0% | 11% | -11 | ❌ | Add 3-4 page-setup / number-format templates (landscape, fit-to-1-page, M/B unit display, decimal-format) |
| `skill_class.check_cell` | 0% | 4% | -4 | ❌ | Add 2-3 per-cell-formula templates (single-cell `=SUM()`, `=A1*B2`) |
| `eval_fn.compare_csv` | 0% | 2% | -2 | ❌ | Add 1 CSV export template (SaveAs → format-dialog handling) |
| `eval_fn.compare_pdfs` (calc-side) | 0% | 2% | -2 | ❌ | Add 1 "Resize to one page → export PDF" template |

**Quant-correction**: gap.md claimed synth lacks both `open` AND Ctrl+S — actually synth HAS Ctrl+S in postconfig already. Only `open` is missing. The bridge is ONE mechanical edit across 266 configs.

**v2 fix**: `calc_skill_class` reads `evaluator.options.rules[*].type` (v1 incorrectly read `expected.rules` → returned `compare_table_unknown` for every row).

## Current shape

| Metric | Value | Source |
|---|---|---|
| `File` instances (`F_CALC_1..90`) | **90** | §I.c — distinct source xlsx shapes |
| `FileTask` entries | **180** | §I.e — (file × task) pairs |
| `Param` rows (= max emitted templates) | **358** | per-FileTask Param lists |
| Eval anchors | 47 (calc subset of `eval.jsonl`) | calc's share input to the scaler's global cap; uncapped at the shipped `TARGET = math.inf`, so Stage B never fires and all 278 rows emit |
| Cap structure | cap-2×2 (≤2 FileTasks / File, ≤2 Params / FileTask) | §I.a |
| Real-CSV-backed files | 11 (F-CALC-27..37) | §I.c batch 4 via `_make_csv_file` |

Eval-class distribution across the 180 FileTasks:

| `eval_class` | FileTasks | Eval skill anchored |
|---|---:|---|
| `multi_sheet_aggregate` | 66 | Sheet2 + filter / groupby / aggregate / copy-col (eval #1273e544, #26a8440e, #0cecd4f3, …) |
| `apply_formula`         | 44 | derived col / total row / VLOOKUP-resolved / quarterly-summary |
| `sort_col`              | 33 | sort ± freeze / bold header |
| `conditional_format`    | 29 | predicate fill / 2-color row fill |
| `text_manipulation`     | 8  | LOWER / UPPER / PROPER / TRIM string clean |
| **total**               | **180** | |

Eval-side breakdown: `compare_table` 44, `compare_csv` 1, `check_pdf_pages` 1, `infeasible` 1 (47 total). Synth currently mirrors only the `compare_table` family — `compare_csv` / `check_pdf_pages` rows are tracked under Bridge plan below.

## Architecture / design notes

**File-first design (per AGENTS.md §3c).** Each `File` is one structurally distinct xlsx shape; the topic IS the file. Adding a new structural shape = adding a new `File` + 1-2 `FileTask`s.

**Source files (§I.c)** are grouped in three batches:

- **Batch 1 (F-CALC-1..16)** — pre-existing inline xlsx: PnL, gradebook, orders, movies, expenses, sales, inventory, sales-rep × quarter, loans, safety inspection, user-emails, product-codes, phonics-titles, questionnaire, bus-schedule, tournament. One-shot openpyxl heredocs.
- **Batch 2 (F-CALC-27..37, 11 files)** — real-CSV-backed via `_make_csv_file(file_id, setup_class, basename, csv_rel, csv_builder)`. The factory chains `_stage_asset(csv_rel, /tmp/_synth_csv_<id>.csv)` ahead of the `_csv_src_*` heredoc. Row counts are capped to land inside the eval Q-scale band (median 22 / p75 30 / max ~45). CSV inventory: us-gdp / us-population-states / us-unemployment / world-gdp-2022 / oil-wti-daily / us-fed-funds-rate / us-housing-starts / us-inflation-cpi / us-mortgage-30yr / us-state-median-income / world-population-2022.
- **Batch 3 (F-CALC-17..26, 38..90)** — inline xlsx covering perturb-orthogonal Cat 2 skills (attendance, warehouse, fitness, concerts, market-share, invoices, ops-metrics, attendance-lookup, quarterly-3-sheets, bank transactions, event schedule, product catalog, clinic visits, survey responses, etc.). No two Files share a builder.

**Structural variation axes** each new `File` should hit a unique combination of: (1) sheet count + names, (2) header row pattern, (3) column type mix, (4) numeric scale, (5) formula presence, (6) datetime cells, (7) empty/sparse rows, (8) merged cells, (9) charts/PivotTables. A `File` that varies only (4)+(7) but not (1)+(5) is **pseudo-diverse**.

**FileTask families**:

- `multi_sheet_aggregate` (66) — `_gold_sheet2_filter` / `_gold_sheet2_groupby_sum` / `_gold_sheet2_filter_score_threshold` / `_gold_sheet2_copy_col` / `_gold_sheet2_aggregate`. Sub-distribution: `_gold_sheet2_groupby_sum` ×36, `_gold_sheet2_filter` ×27, `_gold_sheet2_aggregate` ×3, `_gold_sheet2_filter_score_threshold` ×2, `_gold_sheet2_copy_col` ×2.
- `apply_formula` (44) — `_gold_derived_col` ×42, `_gold_total_row` ×1, `_gold_vlookup_late_fee` ×1. Where `number_format` is non-`None` the per-Param `rules` extends `_RULE_SHEET_DATA` with a `style:number_format` rule.
- `sort_col` (33) — `_gold_sort` — one column index + reverse boolean per FileTask. Each File carries at most one sort task.
- `conditional_format` (29) — `_gold_cell_color_by_predicate` ×15 + `_gold_two_color_by_predicate` ×13 + `_gold_merge_header` ×1.
- `text_manipulation` (8) — `_gold_string_clean` — LOWER / UPPER / PROPER / TRIM into a new "Clean" column.
- Special: F-CALC-26 (3-sheet `quarterly-rollup`) → `_gold_quarterly_summary` (multi-sheet sum into a new Summary sheet).

## Hard Constraints carried into §I.c gold-pys

- **#13 `_LO_NORMALIZE_TAIL`** — every gold-py ends with the soffice round-trip tail (insurance against datetime drift + `<v>` cache).
- **#11 (F11 merged cells / charts)** — don't load-then-modify if source has these; build from scratch.
- **#2 (F2 oracle-only artifact)** — eval must not read files only oracle creates. Oracle is `cp expected_path source_path`.
- **#12 (F12 postconfig save race)** — every row uses `LO_SAVE_POSTCONFIG` from `common.py`; never a custom Ctrl+S.

See [AGENTS.md §Recurring failure-mode taxonomy](/devs/envs/lite.osworld/synth/AGENTS.md#recurring-failure-mode-taxonomy-carry-over-from-perturb-sweeps).

## Implementation references

- `libreoffice_calc.py` §I.a–§I.f — dataclasses, source builders, gold helpers, evaluator builders, FileTask list.
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5) — per-domain volume scaler; intra-domain skill ratio is the author's responsibility per PD (4a).
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — file-shape diversity is dominant value-axis (files/task ratio 0.98).
- [AGENTS.md §Pipeline reference](/devs/envs/lite.osworld/synth/AGENTS.md#pipeline-reference) — inherits perturb-calc's 4-phase flow; only synth-specific change is the `_src_*` heredoc.

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- **Pivot-table** — eval has 4 anchors (`1954cced` / `1de60575` / `51719eea` / `535364ea`). openpyxl `PivotTable` API produces malformed `_pivots` XML (~30% failure historically). Defer until openpyxl stabilises OR a macro-based source generator is validated.
- **`check_pdf_pages`** — eval has 1 anchor. Re-add as a new `FileTask` family + per-File pdf-export gold once a stable PDF page-fit configurator exists.
- **`compare_csv` (xlsx → CSV export)** — re-add via a new FileTask family wiring `_gold_csv_from_xlsx` (soffice convert-to-csv path) when eval-side anchor count justifies budget.
- **Chart insert** — 3 chart templates worked in prior cycles but openpyxl chart XML round-trips imperfectly through LO save; defer.
- **VLOOKUP (formula form)** — `=VLOOKUP()` range-name handling unstable through LO save. The resolved-value path (F-CALC-25 `vlookup_late_fee`) is in.

Re-enable plan for each: (a) per-eval gold-py that survives LO round-trip + (b) one new FileTask per File via §I.e.

## Cycle-recurring failures to avoid (calc-specific)

- **Save-As trap**: synth omits the `open` config-step (quant snapshot row 1); pop in the standard `{"type": "open", "parameters": {"path": "..."}}`.
- **F11 (merged cells / charts)**: build from scratch via openpyxl, never load-then-modify a source carrying these.
- **F12 (postconfig save race)**: use the shared `LO_SAVE_POSTCONFIG`; never a custom Ctrl+S step.

## Pipeline reference

`pre_config_steps` builds the source xlsx via openpyxl heredoc (and optionally chains `_stage_asset` for real-CSV-backed files); agent acts in LO Calc; postconfig is `LO_SAVE_POSTCONFIG`; oracle = `cp expected_path source_path` (the source xlsx is the agent's input AND the expected output's namesake). Eval reads via `compare_table` rules. Gold-pys terminate in `_LO_NORMALIZE_TAIL` for the soffice round-trip.
