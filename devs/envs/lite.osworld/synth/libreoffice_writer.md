# LibreOffice Writer — Synth Plan

> Keep in sync with code. Implementation: [`libreoffice_writer.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/libreoffice_writer.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/libreoffice_writer.md`](/devs/envs/lite.osworld/perturb/libreoffice_writer.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain libreoffice_writer` for live numbers. Synth N=196, eval N=22 (1 infeasibility filtered).

**Writer is the canonical case where `eval_fn` matching is impossible**: synth uses `compare_docx_strict` for ~34% of rows (one fn → 5+ different skills via `examine_*` flags); eval uses ~12 distinct upstream fn names and never uses `compare_docx_strict`. The `skill_class` classifier handles a **dual pattern** — see `evaluator_pattern` dim below.

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `target_anchor.ordinal` | 37.2% | 9.1% | +28 | 🔴 | Cut ordinal-targeted templates ~30 rows; instruction-only |
| `target_anchor.quote_anchor` | 0% | 9.1% | -9 | ❌ | Add ~5 quote-anchor templates: `bold "the second sentence"`, `change "2" in "H2O" to subscript` |
| `skill_class.specialized_uncov` | 10.7% | 27.3% | -17 | 🔴 | Add `check_tabstops` + `compare_unique_train_records` templates (currently 0 synth each) |
| `skill_class.text_match` | 39.3% | 22.7% | +17 | 🔴 | Down-weight `compare_docx_strict_default` (no examine_*) and `compare_docx_files` find-replace; reinvest in specialized |
| `skill_class.tables` | 1.5% | 13.6% | -12 | 🔴 | Triple tables: add 5-8 `compare_docx_tables` (n×m blank, text→table) + `evaluate_colored_words_in_tables` |
| `evaluator_pattern.compound_multi_property` | 2.0% | 22.7% | -21 | 🔴 | Eval has 5 compound rows (mostly same-fn×N at multiple paragraphs/tables/pages); only 1 is true multi-skill (`compare_docx_files+compare_subscript_contains` H2O). Add 2-3 compound templates |
| `evaluator_pattern.compare_docx_strict_default` | 15.8% | 0% | +16 | 🔴 | 31 synth rows use compare_docx_strict with NO examine_* — synth-only super-evaluator. Convert to specific upstream fn (compare_docx_files / compare_font_names / compare_line_spacing) to match eval style |
| `evaluator_pattern.compare_docx_strict+examine_flag` | 17.9% | 0% | +18 | 🔴 | Same — synth-only style. eval never uses compare_docx_strict. Authors should prefer direct upstream fn over examine_* flag |

### Writer skill-match logic (dual pattern)

```
synth row                                ↔  eval row matched against
─────────────────────────────────────────────────────────────────────
compare_docx_strict + examine_font_name  ↔  compare_font_names
compare_docx_strict + examine_images     ↔  compare_docx_images
compare_docx_strict + examine_highlight  ↔  check_highlighted_words
compare_docx_strict + examine_font_size  ↔  check_italic_font_size_14
compare_docx_strict + examine_color      ↔  (no clean eval peer)
compare_docx_strict (no flag)            ↔  compare_docx_files
compare_line_spacing            (direct) ↔  compare_line_spacing
compare_docx_tables             (direct) ↔  compare_docx_tables
contains_page_break             (direct) ↔  contains_page_break
has_page_numbers_in_footers     (direct) ↔  has_page_numbers_in_footers
... + 5 specialized direct-fn match
```

Verified: 0 synth rows have ≥2 examine_* True — priority-order safe. Synth examine_* vocabulary covers only 5 properties (color/font_name/font_size/highlight/images); for the other 7+ specialized upstream funcs, synth uses direct-fn templates.

**Quant-correction**: gap.md earlier mis-characterised compound coverage as "multi-skill compound 23% eval-only". Reality: 4 of 5 eval compound rows are **same-fn-at-multiple-locations**; only 1 row (`0b17a146` H2O subscript) is truly multi-skill. The functional skill hole is smaller than first framed — but synth still has 0 compound rows.

## Current shape

- **Files**: 95 (`F_WRITER_1` … `F_WRITER_95`) — defined once each in §I.c.
- **FileTasks**: 188 — flat `FILE_TASKS` list at §I.f. Two tasks per file for 93 files, one task for `F_WRITER_94` (chemistry-subscript) and `F_WRITER_95` (colored-table).
- **Per-FileTask `params`**: 1 or 2 (capped by `SYNTH_CAP_PARAMS_PER_TASK = 2`). `SYNTH_CAP_TASKS_PER_FILE = 2`.
- **Emitted rows**: **249 current rows** in `train.synth.jsonl` (older ~340 estimate predated later drops/refactors).

Source AND gold docx are both materialised in `pre_config_steps` via python-docx heredocs (Hard Constraint #2); oracle `cp`s the gold over the agent's file path. Postconfig is `LO_SAVE_POSTCONFIG`; every row sets `oracle_after_postconfig=True`.

Eval-class distribution (FileTask 3rd argument):

| eval_class | # FileTasks |
|---|---:|
| `bold_text` | 39 |
| `find_replace` | 31 |
| `change_font` | 25 |
| `highlight_text` | 18 |
| `footnote_citation` | 18 |
| `insert_image` | 16 |
| `change_line_spacing` | 16 |
| `add_header_footer` | 8 |
| `add_page_break` | 7 |
| `color_table_text` | 6 |
| `pdf_export` | 2 |
| `subscript_text` | 1 |
| `blank_table_insert` | 1 |
| **total** | **188** |

## Architecture / design notes

**Evaluator builders (§I header)** — one builder per eval-kind dispatched by `_to_synth_template._eval`:

| `eval_kind` | builder | Upstream eval func |
|---|---|---|
| `strict` | `_build_strict_evaluator` | `compare_docx_strict` (+ `examine_*` flags) |
| `files` | `_build_files_evaluator` | `compare_docx_files` (with `delete_empty_lines=True`) |
| `tables` | `_build_tables_evaluator` | `compare_docx_tables` |
| `line_spacing` | `_build_line_spacing_evaluator` | `compare_line_spacing` |
| `first_centered` | `_build_first_centered_evaluator` | `is_first_line_centered` |
| `page_numbers` | `_build_page_numbers_evaluator` | `has_page_numbers_in_footers` |
| `page_break` | `_build_page_break_evaluator` | `contains_page_break` |
| `font_names` | `_build_font_names_evaluator` | `compare_font_names` |
| `default_font` | `_build_default_font_evaluator` | `find_default_font` |
| `strike_last` | `_build_strike_last_para_evaluator` | `evaluate_strike_through_last_paragraph` |
| `italic_size14` | `_build_italic_size14_evaluator` | `check_italic_font_size_14` |
| `colored_table` | `_build_colored_table_evaluator` | `evaluate_colored_words_in_tables` |
| `subscript` | `_build_subscript_contains_evaluator` | `compare_subscript_contains` |
| `pdf_export` | inline in `_to_synth_template` | `["compare_pdfs"]*4` with `conj=or` over 4 target dirs |

**Source-file builders (§I.c)** — four families by build pattern:

- **(A) Inline-body files** — `_src_genre` / `_src_long_body` / `_src_structured` / `_src_structured_with_photo` / `_src_short_memo` / `_src_qa_format` / `_src_bullet_reference` / `_src_mixed_length_essay` / `_src_wiki_structured`. Paragraphs baked as literal strings in the heredoc.
- **(B) Gutenberg files** (`_src_gutenberg`) — stage a Project Gutenberg `.txt` via `_stage_asset`, slice first N body paragraphs after a known anchor. The 11-book catalog (`_GUTENBERG_BOOKS`) covers Alice / Pride / Moby / Frankenstein / Art of War / Tale of Two Cities / Metamorphosis / Sherlock / Treasure Island / Tom Sawyer / Earnest. Files: `F_WRITER_18-25`, `F_WRITER_57-66`.
- **(C) Image-host files** (`_src_image_host` / `_src_double_image_host`) — stage one or two photo jpgs at `/home/user/Desktop/<basename>` + build a body-only docx referencing the image. Real-photo content from `assets/synth/photos/`. Files: `F_WRITER_26-31`, `F_WRITER_77-81`.
- **(D) Wikipedia structured files** (`_src_wiki_article` / `_src_wiki_structured`) — paragraphs from `assets/synth/html/wikipedia/` cached at module load. Files: `F_WRITER_82-93`.

Bespoke: `F_WRITER_94` (`_src_chemistry_notes` — for `subscript_text`), `F_WRITER_95` (`_src_colored_table` — for `color_table_text` × `evaluate_colored_words_in_tables`).

**Gold heredoc helpers (§I.d)** — one `_gold_<op>` per operation, signature `(src, gold, **gold_args) -> heredoc_body`. Coverage: `_gold_bold_para`, `_gold_italic_para`, `_gold_underline_para`, `_gold_strike_para`, `_gold_highlight_para`, `_gold_size_para`, `_gold_doc_spacing`, `_gold_para_spacing`, `_gold_doc_font`, `_gold_default_font_noop`, `_gold_append_paragraph`, `_gold_find_replace`, `_gold_insert_empty_table`, `_gold_page_break`, `_gold_first_centered`, `_gold_page_numbers_footer`, `_gold_strike_last`, `_gold_italic_size14`, `_gold_gutenberg_p0_op`, `_gold_image_insert`, `_gold_double_image_insert`, `_gold_subscript_chemistry`, `_gold_colored_table`, `_gold_pdf_export`.

`_gold_doc_font` and `_gold_<apply_style>` append an `soffice --headless --convert-to docx` round-trip so the gold's docx matches LO's emitted format on save.

**Body-focus post-open step** — `_WRITER_BODY_FOCUS_STEPS` is wired into every writer task's `_params["post_open_config_steps"]`. It activates the LO Writer window, clicks at (500, 400) in the body, then sends `Ctrl+End` to dodge the first-run Release Notes banner.

**Active row groups** (oracle-validated):

| Eval `func` (resolved via builder) | FileTask eval_class buckets |
|---|---|
| `compare_docx_strict` (+ `examine_*`) | `bold_text` (italic / underline / strike / size18 / highlight via `examine_highlight` / font_size via `examine_font_size`) |
| `compare_docx_files` (with `delete_empty_lines=True`) | `find_replace`, `footnote_citation` |
| `compare_docx_tables` | `blank_table_insert` (F-WRITER-5) |
| `compare_line_spacing` | `change_line_spacing` |
| `is_first_line_centered` | `bold_text` (centred-paragraph; F-WRITER-4) |
| `compare_docx_strict + examine_font_name` (preferred over `compare_font_names`) | `change_font` |
| `find_default_font` | `change_font` registrymodifications.xcu (gold = src) |
| `has_page_numbers_in_footers` | `add_header_footer` |
| `contains_page_break` | `add_page_break` |
| `evaluate_strike_through_last_paragraph` | `bold_text` strike-last (F-WRITER-7) |
| `check_italic_font_size_14` | `highlight_text` italic+size14 combo (F-WRITER-11, 33, 38, 45, 50, 57) |
| `evaluate_colored_words_in_tables` | `color_table_text` (F-WRITER-95) |
| `compare_subscript_contains` | `subscript_text` (F-WRITER-94) |
| `["compare_pdfs"]*4` (conj=or) | `pdf_export` (4-way over Desktop / Documents / Downloads / home) |

## Implementation references

- `libreoffice_writer.py` §I.a–§I.g — dataclasses, source builders, gold helpers, evaluator builders, FileTask list.
- `_gold_apply_style` / `_image_gold_heredoc` / `_double_image_gold_heredoc` append `soffice --headless --convert-to docx` round-trip so gold matches LO's emitted format on save.
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5) — cross-domain volume only; per-skill ratio author-controlled.
- [AGENTS.md §F6 (visible-vs-raw idx)](/devs/envs/lite.osworld/synth/AGENTS.md#recurring-failure-mode-taxonomy-carry-over-from-perturb-sweeps) — synth owns the source builder, so `body_idxs[i] == i` for inline-body files; Gutenberg/Wiki sources target paragraph 0 only, sidestepping F6.

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- **icon-decoration rows** — SVG rasterization needs `rsvg-convert` or `cairosvg`, neither baseline. JPG-photo rows cover the image-insert skill.
- **`check_highlighted_words`** — odfpy ODT-only. Substituted by `compare_docx_strict + examine_highlight=True`.
- **`check_tabstops`** — needs careful per-paragraph tab arithmetic; deferred.
- **Niche editor ops (numbered list / bulleted list / multi-section / hyperlink / column / comment-add / footnote / track-change)** — verified absent in `eval.jsonl`, so per AGENTS.md no-fabrication rule these are not implemented.
- **3 mis-labeled assets** (don't break eval): `nasa-facility.jpg` is a Mars-lander concept; `office-building.jpg` is a skatepark; `pizza-dish.jpg` is an aircraft.

## Cycle-recurring failures to avoid (writer-specific)

- **J-save trap** — agents commonly add an extra blank paragraph before appended citations; `_build_files_evaluator` carries `delete_empty_lines=True` for find-replace / footnote rows.
- **Empty-trailing-paragraph FALSE_NEG** — `compare_font_names` flakes on doc-wide font rows; use `_build_strict_evaluator(..., examine_font_name=True)` instead.
- **Triple-click for paragraph-selection** — instructions must say "triple-click ... select the entire line" so `_char_format_signature` doesn't see off-by-one drag-selections.
- **PDF-export wording** — instructions must spell out File > Export As > Export Directly as PDF + Shift+Ctrl+E + explicit Export click.
- **Image-insert byte-equality** — image gold builders append `soffice --headless --convert-to docx` round-trip so `examine_images` per-image sha256 matches LO's drawingML re-encoding.

## Pipeline reference

`_src_*` via python-docx in `pre_config_steps`; gold landed at `/tmp/expected_<template_id>_<seed>.docx` (or `.pdf` for `pdf_export`); agent edits in LO Writer; `LO_SAVE_POSTCONFIG` normalizes the agent's output; oracle `cp`s gold over the result; `compare_docx_strict` / dispatched builder reads via python-docx.
