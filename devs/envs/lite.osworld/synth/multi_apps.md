# Multi-Apps — Synth Plan

> Keep in sync with code. Implementation: [`multi_apps.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/multi_apps.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/multi_apps.md`](/devs/envs/lite.osworld/perturb/multi_apps.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain multi_apps` for live numbers. Synth N=360, eval N=100 (1 infeasibility filtered). multi_apps is the largest and most diverse domain.

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `tool_leak.tool_leak` | 26.9% | 0% | +27 | 🔴 | Strip backticked `pdftk` / `pandoc` / `convert` / `magick` / `ffmpeg` / `picard` commands from 97 instructions. Name the tool, drop the flags |
| `apps_per_task.apps_le_1` | 61.4% | 46.0% | +15 | 🔴 | Reclassify ~80 single-app terminal recipes to `os` domain (they're misnamed as multi_apps). Reinvest budget in 2-app workflows. |
| `apps_per_task.apps_3plus` | 0% | 8.0% | -8 | ❌ | Add 4-6 3-app workflow templates (chrome+writer+files, calc+chrome+writer, etc.) |
| `app_combination` missing pairs | — | — | — | ❌ | Missing 3-app and specific pairs: `pdf+writer` 0/3, `calc+thunderbird` 0/3, `gimp` (alone) 0/3, plus 6+ uncovered shapes (email→docx/Drive/wallpaper, Google-Drive I/O, Chrome state side-effects, git workflows, GIMP+VLC chains) |
| Photo-to-docx Trigger O | 106 rows | 1 row (`09a37c51` SSIM) | — | 🔴 eval-bug | 106 synth rows byte-compare embedded image blob; eval uses `compare_docx_images` SSIM tolerance. **One eval-side fix recovers ~20-25 tasks.** Port soffice round-trip / SSIM. |

**Quant-correction**: gap.md claimed apps-per-task 84% synth vs 2% eval. With combined instruction-kw + eval-fn inference, the gap shrinks to 61% vs 46% — gap.md instruction-only estimate over-stated. Real gap is moderate.

**v2 fix**: `_APP_INSTR_KEYWORDS` and `_EVAL_FN_TO_APP` expanded with `gdrive` / `git` / `gnome` / `vim` + 5 new eval-fn mappings (`is_expected_installed_extensions`, `compare_conference_city_in_order`, etc.).

## Current shape

**276 templates** (Cat A–R) emitted to `TEMPLATES`, sourced from **29 `_*_SPECS: list[dict]` spec lists** consumed by **~60 `_make_*_template(spec)` factory functions** (plus zero-arg factories for one-off rows and the photo Cartesian generators). Per-row volume is set by the cross-domain scaler in `synth/catalog.py` — multi_apps.py only declares templates and their per-template `n_rows ∈ {1, 2}` cap.

Source state lives entirely in `pre_config_steps` heredocs that materialize the input file(s) AND the gold expected file. Oracle = `cp` gold over the agent's sink (or its shell-pipeline equivalent). LO-sink rows set `oracle_after_postconfig=True` + use `LO_SAVE_POSTCONFIG`; `synth_command=""` everywhere.

| Cat | Family | Spec list(s) | Factory | Active templates |
|---|---|---|---|---|
| A | txt → docx via Writer | `_TXT_TO_DOCX_SPECS` (8) | `_make_txt_to_docx_template` | 8 |
| B | csv → xlsx via Calc | `_CSV_TO_XLSX_SPECS` (9) | `_make_csv_to_xlsx_template` | 9 |
| C | xlsx → csv via Calc | `_XLSX_TO_CSV_SPECS` (6) | `_make_xlsx_to_csv_template` | 6 |
| D | xlsx → docx-table via Writer | `_XLSX_TO_DOCX_TABLE_SPECS` (10) | `_make_xlsx_to_docx_table_template` | 10 |
| E | shell pipeline (inline text) | (inline) | `_make_shell_pipeline_template` + `_shell_pipeline_templates()` | 12 |
| F | extract zip / tar.gz | (inline) | `_make_extract_sort_template`, `_make_extract_targz_template` | 2 |
| G | zip create | (inline) | `_make_zip_create_template` | 1 |
| H | real-asset shell pipeline (`_stage_asset`) | `_ASSET_SHELL_SPECS` (9) | `_make_asset_shell_row` | 9 + 1 (`asset_csv_to_xlsx_state_income`) |
| I | real-asset CROSS-APP (sub-cats I.1–I.11) | `_CSV_TO_DOCX_TABLE_SPECS` (11), `_CSV_TO_DOCX_TEXT_SUMMARY_SPECS` (8), `_ARXIV_PDF_SPECS` (5), `_CSV_CONCAT_XLSX_SPECS` (4), `_CODE_TO_DOCX_SPECS` (7), photo Cartesian (`TOPIC_FAMILIES`) | per sub-cat | ~50 (11 CSV→docx-table + 8 CSV→docx-summary + 5 arxiv + 4 multi-sheet concat + 7 code→docx + Cartesian 5×3 + 5×1 gallery + standalone wikipedia/h2 + 3-way FRED concat) |
| J | pandoc | `_PANDOC_MD_TO_DOCX_SPECS` (7), `_PANDOC_HTML_TO_MD_SPECS` (9), 1 csv→md, 1 docx→txt | `_pandoc_*_template` | 18 |
| K | ImageMagick (`convert` / `montage`) | `_IM_PHOTO_SPECS` (20) + 1 contact sheet | `_make_im_template`, `_make_im_montage_contact_sheet` | 21 |
| L | poppler-utils (`pdftotext` / `pdfinfo`) | `_PDFTOTEXT_SPECS` (6) + 1 pagecount | `_make_pdftotext_template`, `_make_pdfinfo_pagecount_template` | 7 |
| M | poppler-utils (`pdftoppm`) | `_PDFTOPPM_SPECS` (4) | `_make_pdftoppm_template` | 4 |
| N | docx/xlsx/pptx → pdf via soffice (eval via `compare_text_file` over `pdftotext`) | `_DOCX_TO_PDF_SPECS` (5), `_XLSX_TO_PDF_SPECS` (4), 1 pptx, pdftk merge/extract | `_make_*_to_pdf_template`, `_make_pdftk_*_template` | 12 |
| O | exact_match (csv → single-string answer) | `_EXACT_MATCH_SPECS` (6) | `_make_exact_match_template` | 6 |
| P | compare_image_list (photo subset collect/rename) | `_COMPARE_IMAGE_LIST_SPECS` (5) | `_make_compare_image_list_template` | 5 |
| Q | check_python_file_by_test_suite (fix-bug) | `_PY_TEST_SPECS` (4) | `_make_py_test_template` | 4 |
| R | long-tail gap-fill (Cat R) | `_DIFF_TEXT_FILE_SPECS` (8), `_CHECK_LIST_SPECS` (8), `_CHECK_MP3_META_SPECS` (8), `_COMPARE_IMAGE_TEXT_SPECS` (6), `_COMPARE_EPUB_SPECS` (6), `_IS_EXPECTED_TABS_SPECS` (6), `_COMPARE_PDFS_SPECS` (5), `_COMPARE_TABLE_EXTRA_SPECS` (3) + 16 one-offs | per spec list / one-off | 66 |

Totals: A 8 + B 9 + C 6 + D 10 + E 12 + F 2 + G 1 + H 10 + I 50 + J 18 + K 21 + L 7 + M 4 + N 12 + O 6 + P 5 + Q 4 + R 66 = **276 templates**.

## Architecture / design notes

**Spec-dict factory pattern (different from other domains).** Unlike `libreoffice_calc.py` / `gimp.py` / `chrome.py` (which use `FileTask` + `Param` per-template authoring), multi_apps is **spec-dict + factory**:

- A `_<NAME>_SPECS: list[dict]` block declares N row variants as plain dicts (each spec carries `id`, source filename, agent-visible instruction, eval rule parameters).
- A `_make_<NAME>_template(spec: dict) -> SynthTemplate` factory consumes a single spec dict.
- `TEMPLATES` at the bottom is a list comprehension `[_make_X_template(s) for s in _X_SPECS]` plus zero-arg one-offs.

This lets cross-app rows that share a heredoc pipeline scale by appending dicts, not by copy-pasting factory bodies.

**Eval files/task ratio**: 1.85 — HIGHEST of any domain. Each multi_apps eval task averages near 2 source files. Synth must invest most heavily in **file-shape diversity** here.

**Eval evaluator-func mix**: `compare_table` (xlsx output), `compare_docx_files` / `compare_docx_strict`, `compare_pptx_files`, `compare_pdfs`, `compare_archive` (zip), `compare_images` / `check_structure_sim`, `check_include_exclude` (file presence / directory state).

**Multi_apps tier taxonomy** (from perturb side): A1 chrome → LO sink; A2 single-app file ops; A3 dual-app cross-flow; A4 real eval source → derived doc (xlsx → docx table); A6 txt concat → docx; A8 structural section modify; A10 image processing; A23 cross-app skeleton mirroring eval.

**Real-asset wiring**: `_stage_asset` host_push for `assets/synth/data/csv/*.csv` (FRED / Census / WorldBank), `assets/synth/docs/gutenberg/*.txt`, `assets/synth/docs/pdf/arxiv-*.pdf`, `assets/synth/html/wikipedia/*.html`, `assets/synth/photos/{landscape,nature,portrait,architecture,product}/*.jpg`, `assets/synth/graphics/{icon,diagram,logo}/*`, `assets/synth/code/{python,javascript,typescript,go,rust,java}/*`. Real-source ratio across active rows ~33%.

## Cat 2 templates (verified eval-grounded)

| Template | Eval task_id citation | Evaluator |
|---|---|---|
| `synth_multi_thunderbird_export_emails_to_files` | a0b9dc9c (Bills folder w/ AWS invoice mbox) — V1 vendor-invoice / V2 newsletter / V3 receipts triage | `check_list` comparing directory contents to expected file list |
| `synth_multi_thunderbird_export_contacts_csv` | c867c42d (Personal Address Book → contacts.csv) — V1 personal → CSV → xlsx / V2 business → ODS / V3 mailing-list → TSV / V4 filter @example.com | `check_csv` |
| `synth_multi_image_in_doc_to_writer` | 227d2f97 (1152×648 GIMP XCF → empty docx) — V1 XCF logo / V2 architecture PNG / V3 photo JPEG / V4 chart-export SVG | `compare_docx_files` with image-presence check |
| `synth_multi_xlsx_real_to_docx_table` | 00fa164e / 81c425f5 / 185f29bd | `compare_docx_tables` |

**Reject** `synth_multi_pdf_form_fill` — eval marks PDF form-fill tasks as infeasible; agent lacks reliable PDF annotation lib.

## Implementation references

- `multi_apps.py` — 29 `_*_SPECS` lists + ~60 `_make_*` factories; `TEMPLATES` assembly at end.
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5) — cross-domain volume rebalance only; per-skill ratio author-controlled via spec-list growth.
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — Cat 2 dominant; cross-app combinations are inherently sparse in perturb.
- [AGENTS.md §Asset placement rule](/devs/envs/lite.osworld/synth/AGENTS.md#asset-placement-rule--assetssynth-vs-inline-py-heredoc) — when to inline vs `_stage_asset`.
- `lite/gym/envs/lite/osworld/src/eval/metrics/*` — `compare_table`, `compare_docx_files`, `compare_pptx_files`, `compare_pdfs`, `compare_archive`, `compare_csv`, `check_include_exclude`, `check_list`, `compare_docx_images`, `check_thunderbird_folder`, `check_mp3_meta`, `diff_text_file`.

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- **Chrome-source content-extract rows** — UI clipboard path is brittle in scripted oracle; shell-shortcut path reads staged HTML via `cat`, exercising no chrome skill. Eval has 0 rows that read content FROM chrome into a sink file.
- **Thunderbird-source content-extract rows** — UI eml-save / sqlite-export are hard to oracle without per-message scripted pulls. Bulk export via `check_list` is covered.
- **`compare_docx_images` PIL `tobytes()`** — Iter-11 dropped these; byte-equality only matched the `cp` oracle path (any real LO Insert > Picture re-encoded JPEG). Other photo→docx rows survive via `compare_docx_strict examine_images=True`.
- **ffmpeg rows owned by `vlc.py`** — multi_apps doesn't duplicate; vlc has the deterministic media-transform pipelines.
- **Calibre rows** — NOT installed in baseline.

## Cycle-recurring failures to avoid (multi_apps-specific)

- **F2 (eval depends on oracle artifact)**: NEVER have eval read from `_ls.txt` / `_state.txt` that only oracle creates. Use `vm_command_line` computed at eval time OR have agent's app save the result.
- **F11**: multi_apps eval sources often have rich xlsx/pptx structure (charts, merged cells); avoid round-tripping through openpyxl/python-pptx if you don't need to mutate the structure.
- **F12 (postconfig save race)**: especially relevant because the agent often switches between apps; the final `Ctrl+S` may go to the wrong window. Use `LO_SAVE_POSTCONFIG` with explicit `activate_window`.
- **URL hallucination**: if synth needs an HF URL (rare; synth should self-contain), copy verbatim from eval source; don't infer UUIDs from short_id.
- **Tool-dependency contract**: pandoc / jq / pdftk / pdftoppm / ImageMagick / ffmpeg are required by Cat J–N rows. The Dockerfile installs them; per [AGENTS.md §Tool-dependency contract](/devs/envs/lite.osworld/synth/AGENTS.md), do not gate rows on tools that aren't in baseline.

## Pipeline reference

Multi_apps templates emit BOTH source files in `pre_config_steps` (e.g. `_make_source_html_*` + `_make_source_xlsx_*`), launch BOTH apps (chrome + LO), and the agent flows between them. Eval reads the LO sink file via the same channel as single-app domains.

### Step 0: inspect eval multi_apps tasks (helper)

```python
"""Enumerate multi_apps eval evaluator-func + result/expected file types."""
import json
from collections import Counter
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
ma = [r for r in rows if "_multi_apps_" in r["task_id"]]
func_count = Counter()
for r in ma:
    ev = r["metadata"].get("evaluator", {})
    if isinstance(ev, dict):
        fn = ev.get("func", "?")
        func_count[fn if isinstance(fn, str) else tuple(fn)] += 1
for fn, n in func_count.most_common():
    print(f"  {fn}: {n}")
```
