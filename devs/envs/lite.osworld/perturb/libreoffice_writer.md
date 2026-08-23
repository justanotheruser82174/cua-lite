# LibreOffice Writer — Perturbation Plan

Domain-specific plan for `libreoffice_writer`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/libreoffice_writer.py`

---

## Cycle 35a+ updates

### `_FONTS` pool trim (Calibri / Georgia removed)

Per cycle 27 audit, **Calibri** and **Georgia** are not installed in the LO VM — when the agent applies them LO silently falls back to a default font, the post-save docx records the substituted name, and the `compare_docx_strict` font-name check passes regardless of agent action → trivial pass / inverted training signal. New `_FONTS` pool: `["Arial", "Times New Roman", "Courier New", "Verdana"]`. The V4e `_FONTS` reference list elsewhere in this doc still lists 6 fonts for historical V4e parameter-coverage tracking; the runtime pool is the 4-font subset.

### New TYPE_3 archetypes (page_number, page_break)

`_TYPE3_BASES` expanded from `{0a0faba3, 0b17a146, 4bcb1253}` → **`{0a0faba3, 0b17a146, 4bcb1253, 0e47de2a, ecc2413d}`**. The two new archetypes target eval evaluators that `compare_docx_strict` does not cover:

| tid | eval evaluator | builder | Mechanic | Pool |
|---|---|---|---|---|
| `0e47de2a` | `has_page_numbers_in_footers` | `_emit_page_number_rows` | py_code injects a `PAGE` field into the document footer via `python-docx` section.footer; LO double-normalize + cp oracle | paraphrase × footer-position variants |
| `ecc2413d` | `contains_page_break` (`expected_break_count=5`) | `_emit_page_break_rows` | The source doc already has 4 pre-existing page breaks; oracle py_code inserts exactly **1** additional `<w:br type="page"/>` so the saved file totals 5 — meeting the evaluator's expected count | paraphrase × insertion-point variants |

`0e47de2a` and `ecc2413d` now produce ~5 rows each (2 strict-evaluator TYPE_2 + 3 archetype TYPE_3) instead of 2, mirroring the existing TYPE_3 pattern.

---

## Step 0: Download Files and Inspect Content

Run once before drafting the per-task plan. Source DOCXs are on HuggingFace, URLs embedded in `eval.jsonl`.

```python
"""Run from repo root: uv run python this_script.py

Produces /tmp/writer_full.json with the schema the perturb generator consumes:
keys are 8-hex source task ids (e.g. "0810415c"); each entry contains
``n_paras``, ``text_paras`` (non-empty paragraph indices + 60-char prefix used
for the n_paras_safe cap), tables, headings, and ``has_page_break``. The
perturb code reads ``text_paras`` to compute ``last_non_empty_idx + 1`` so
paragraph-index ops never emit a target beyond the last non-empty paragraph.
"""
import json, urllib.request, pathlib
from docx import Document

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
writer = [r for r in rows if "writer" in r["task_id"]]
out_dir = pathlib.Path("/tmp/writer_docx")
out_dir.mkdir(exist_ok=True)

# 1) Download all source docx (URL → /tmp/writer_docx/<8hex>.docx)
for r in writer:
    tid = r["task_id"]
    short = tid.split("_")[-1]  # 8-hex id
    for step in r["metadata"]["config"]:
        if step.get("type") == "download":
            for f in step["parameters"].get("files", []):
                if f["path"].endswith(".docx"):
                    fname = out_dir / f"{short}.docx"
                    if not fname.exists():
                        print(f"downloading {short}...")
                        urllib.request.urlretrieve(f["url"], fname)

# 2) Inspect each docx and emit the rich schema the perturb code consumes
result = {}
for r in writer:
    tid = r["task_id"]
    short = tid.split("_")[-1]
    docx_path = out_dir / f"{short}.docx"
    if not docx_path.exists():
        result[short] = {"ext": "odt" if any(
            f.get("path", "").endswith(".odt")
            for step in r["metadata"]["config"] if step.get("type") == "download"
            for f in step["parameters"].get("files", [])
        ) else "unknown", "note": "non-docx - skipped"}
        continue
    try:
        doc = Document(str(docx_path))
    except Exception as e:
        print(f"[SKIP] {short}: {e}"); continue
    text_paras = [
        [i, p.text[:60]] for i, p in enumerate(doc.paragraphs) if p.text.strip()
    ]
    headings = [
        [i, p.style.name, p.text[:50]]
        for i, p in enumerate(doc.paragraphs)
        if p.style and p.style.name and "Heading" in p.style.name
    ]
    tables = [
        {"rows": len(t.rows), "cols": len(t.columns),
         "header": [c.text[:20] for c in t.rows[0].cells] if t.rows else []}
        for t in doc.tables
    ]
    has_page_break = any(
        "<w:br" in p._p.xml and 'type="page"' in p._p.xml
        for p in doc.paragraphs
    )
    result[short] = {
        "ext": "docx",
        "n_paras": len(doc.paragraphs),
        "n_tables": len(doc.tables),
        "n_images": sum(1 for s in doc.inline_shapes),
        "text_paras": text_paras[:10],   # first 10 non-empty for readability
        "tables": tables,
        "headings": headings,
        "styles": sorted({p.style.name for p in doc.paragraphs if p.style}),
        "has_page_break": has_page_break,
    }
    print(f"{short}: {len(doc.paragraphs)} paragraphs, last_non_empty="
          f"{text_paras[-1][0] if text_paras else 'n/a'}")

pathlib.Path("/tmp/writer_full.json").write_text(json.dumps(result, indent=2))
print("wrote /tmp/writer_full.json")
```

`_load_analysis()` in the perturb code reads from `/tmp/writer_full.json` and consumes:
- `n_paras` → upper bound on paragraph-index ops (`min(4, n_paras-1)` for the 5-bucket sample range).
- `text_paras` → list of `[idx, text_prefix]` pairs for non-empty paragraphs. The perturb code derives `last_non_empty_idx = max(t[0] for t in text_paras)` and caps the start idx at `last_non_empty_idx + 1` so the runtime advance loop is guaranteed to land on a non-empty paragraph (avoids `trivial_pass` from no-op ups/lower/etc on empty trailing paragraphs in short documents like `8472fece`).

If the file is missing, all tasks fall back to `n_paras=5` and `text_paras=[]` — the latter disables the safe cap, which lets idx ops emit targets beyond the last non-empty paragraph in short docs (`8472fece` n=4, last_non_empty=1) and may cause `rc=42` from the runtime advance. **Always regenerate `/tmp/writer_full.json` before running perturb generation** so the safe cap engages.

---

## Oracle Mechanics (LibreOffice Writer)

### Ground Truth Generation: Oracle py_code

For each perturbation op, a Python snippet using `python-docx` writes the target state to `expected_path`. The snippet is stored in a `perturb_config_step` (an `execute` step) appended to the task config.

### Oracle Pattern

```
perturb_config_step: python snippet -> writes target state to expected_path
oracle:
  1. LO-normalize expected_path (LibreOffice --convert-to docx round-trip)
  2. cp expected_path -> file_path
  3. LO-normalize file_path
postconfig: LO_SAVE_POSTCONFIG (activate_window + Ctrl+S + sleep)
```

The LO-normalize step ensures both files go through identical XML serialization before comparison, preventing false-negative mismatches from round-trip whitespace differences.

### Always use `oracle_after_postconfig=True`

All ops use `oracle_after_postconfig=True` so the oracle (cp) runs after `LO_SAVE_POSTCONFIG`. This prevents Ctrl+S from overwriting the oracle-placed file.

### Evaluator (all ops)

```python
evaluator = {
    "func": "compare_docx_strict",
    "result": {"type": "vm_file", "path": file_path, "dest": file_path.split("/")[-1]},
    "expected": {"type": "vm_file", "path": expected_path, "dest": "expected_file"},
    "options": {
        "examine_font_name": op_idx == 1,  # only font op keeps strict
        "examine_font_size": False,
        "examine_color": False,
        "examine_highlight": False,
        "examine_images": False,
    },
    "postconfig": LO_SAVE_POSTCONFIG,
}
```

Character-format fields (color, font_size, highlight) are relaxed on all ops except `font` — LibreOffice's DOCX round-trip normalizes these on untouched runs, causing false-fail mismatches when the agent applies a correct change.

---

## Perturbation Strategies

**TYPE_1 + TYPE_2 + TYPE_3 (writer domain)**: Writer eval tasks use highly varied operations (PDF export, clipboard, realtime collaboration, complex formatting). For the subset of eval tasks whose operation matches one of the 5 perturb ops (currently 4 tasks), TYPE_1 resamples the same op class with different parameters — ensuring the agent sees the op at slightly different settings than the eval. All other tasks get TYPE_2 only.

**TYPE_3 archetypes** (P3-4-writer): for the 3 eval bases whose evaluators fall *outside* `compare_docx_strict`'s coverage, an additional per-base archetype emits 2-3 paraphrase-varied rows that exercise the actual eval evaluator (`check_tabstops` / `compare_subscript_contains` / `compare_pdfs`). TYPE_3 runs *in addition to* TYPE_1/TYPE_2 for those bases, so each of the 3 bases produces ~5 rows (2 strict-evaluator TYPE_2 + 3 archetype TYPE_3) instead of 2.

**TYPE_1 eligible tasks** — eval op is in the perturb pool, mapped in `_TYPE1_FNS`:

| tid | eval op | TYPE_1 constraint |
|---|---|---|
| `0810415c` | spacing of paras 0-1 to double | spacing on para ≥2, any spacing value |
| `b21acd93` | spacing on para 0 | spacing on para ≥1, any spacing value |
| `0e763496` | font → Times New Roman | font, any font except Times New Roman |
| `f178a4a9` | default font → Times New Roman | font, any font except Times New Roman |

TYPE_1 op is excluded from that task's TYPE_2 pool to prevent the same op appearing twice.

**TYPE_3 eligible bases** — eval evaluator is outside `compare_docx_strict`'s coverage, mapped via `_TYPE3_BASES` + dedicated `_emit_*_rows` builders:

| tid | eval evaluator | archetype | builder fn | Pool size |
|---|---|---|---|---|
| `0a0faba3` | `check_tabstops` | tabstops: split each non-empty paragraph at the n-th word with a tab + add right-aligned tab stop at right margin (filtered at runtime to paragraphs with ≥ n+1 words; `sys.exit(42)` if none qualify) | `_emit_tabstops_rows` → `_make_tabstops` | 9 paraphrase × 3 split-points (n ∈ {2,3,4}) |
| `0b17a146` | `compare_docx_files+compare_subscript_contains` | subscript: pick the n-th non-empty paragraph **that contains a digit** (filtered at runtime to skip digit-free headers like "Fact sheet"); split the first run containing a digit so the digit lives in its own run; set `font.subscript=True` on that run | `_emit_subscript_rows` → `_make_subscript` | 10 paraphrase × 3 paragraph indices |
| `4bcb1253` | `compare_pdfs+compare_pdfs+compare_pdfs+compare_pdfs` (conj=or) | pdf_export: oracle uses `soffice --convert-to pdf` to render the docx to all 4 standard target dirs (Desktop/Documents/Downloads/home) AND to a single `/tmp/perturb_expected_*.pdf` reused 4× as the expected (all 4 compare slots reference the same generated PDF, fuzz-ratio = 1.0) | `_emit_pdf_export_rows` → `_make_pdf_export` | 10 paraphrase |

`max_type3=3` (default) draws up to 3 distinct paraphrases per archetype. TYPE_3 evaluators bypass `_build_evaluator` (which is hard-wired to `compare_docx_strict`); each archetype declares its own evaluator dict matching the eval's func/result/expected shape exactly. The tabstops/subscript archetypes reuse `_build_oracle` (LO double-normalize + cp) so docx round-trip parity holds; the pdf_export archetype uses a pure-shell oracle (no docx normalize) because the result/expected files are PDFs.

**No-leakage guarantee**: all 5 op types (bold, font, find_replace, spacing, uppercase) are applied to the eval file at generation time, producing a new `expected_path` that differs from the eval's target state. The parameter space (font choice, replacement word, paragraph index, spacing value) is drawn from pools that don't overlap with typical eval task targets. TYPE_3 archetype paraphrases are checked against eval instructions in the dispatcher (`apply_structural_perturbation` filters `instruction in eval_instructions`) — current pools have no verbatim overlap.

---

## Instruction Paraphrase Pools

> **Keep in sync with code.** Each `_<OP>_VARIANTS` list in `libreoffice_writer.py` and the design targets below must agree. Changes to the pool size, narrative style, or polite ratio update both files together.

Each of the 8 ops owns a 12-entry paraphrase pool (D5 expansion: 4 short / 4 medium / 4 long buckets) so the emitted dataset spans p25 ≤ 14 words, p50 ≈ 28 words, and p75 ≥ 40 words simultaneously, matching eval's bimodal short/long shape. A subset of medium/long variants embed multi-step sequence keywords (then / next / first / once / after) so the multi-step ratio in the emitted dataset clears the ≥18% target. The dispatcher `_OP_POOL[op_idx](...)` calls `rng.choice(pool)` to pick one.

| Target | Value | Rationale |
|---|---|---|
| Templates per op | 12 (4 short + 4 medium + 4 long) | Trimodal length distribution matches eval p25/p50/p75 simultaneously |
| Short-bucket words | ~6–12 | Covers eval p25 (~10 words) |
| Medium-bucket words | ~24–32 | Matches eval mean (~27.7 words, V3 baseline) |
| Long-bucket words | ~42–65 | Covers eval p75/max (~47–85 words) |
| Multi-step ratio | ≥18% (achieved via ~2–4 multi-step variants per pool) | Matches eval multi-step density |
| Polite-leading templates per pool | ~20% | Aggregates to writer-target polite share 18–25% (eval=17%) |
| Lead-in styles | "I'm reviewing…", "While editing…", "For a writeup…", "Quick edit:", "Could you…/Please…", "First locate…", "Bold…/Apply…" | Mixed openings (imperative, polite, contextual, sequencing) so distribution doesn't cluster |
| TYPE_3 archetype pool sizes | 9 (tabstops) / 10 (subscript) / 10 (pdf_export) | Smaller short tier (2/3/2) since each archetype runs ≤ 3 rows per base |

`_build_instruction(phrase, rng)` is now a pass-through — no polite-prefix augmentation is layered on top of the pool, because each pool already encodes its own narrative tone at the desired ratio. (Earlier design appended `Could you help me / Please / I need to / Can you / I want to / I'd like to` at 32% — that pushed polite over the eval target and created telegraphic instructions that didn't match eval narrative density.)

**Save-suffix invariant**: no template ends with "Save the file" or contains the substring "save the file" (AGENTS.md hard constraint, V1 static check).

---

## Op Pool (8 ops)

| idx | op | What it does | Oracle (python-docx) | Evaluator flag | Feasibility |
|---|---|---|---|---|---|
| 0 | `bold` | Make the N-th paragraph bold | `r.font.bold = True` for all runs | all relaxed | ≥1 paragraph, paragraph has runs, not already bold |
| 1 | `font` | Set document-wide font | `r.font.name = font` on all paragraphs + table cells | `examine_font_name=True` | `.docx` file, any size |
| 2 | `find_replace` | Find most frequent 4+ letter word; replace with new word | regex replace per run | all relaxed | ≥1 matching word found |
| 3 | `spacing` | Set N-th paragraph line spacing to 1.5/double | `p.paragraph_format.line_spacing = val` | all relaxed | ≥1 paragraph, not already at target spacing |
| 4 | `uppercase` | Convert N-th paragraph text to uppercase | `r.text = r.text.upper()` | all relaxed | ≥1 paragraph with non-empty text |
| 5 | `strikethrough` | Apply strikethrough to the N-th paragraph | `r.font.strike = True` for all runs | all relaxed | ≥1 paragraph with runs, not already struck |
| 6 | `lowercase` | Convert N-th paragraph text to lowercase | `r.text = r.text.lower()` | all relaxed | ≥1 paragraph with non-empty text |
| 7 | `italic` | Italicize the N-th paragraph | `r.font.italic = True` for all runs | all relaxed | ≥1 paragraph with runs, not already italic via style |

### TYPE_3 Archetypes (P3-4-writer)

> **Keep in sync with code.** The 3 archetype builders (`_make_tabstops` / `_make_subscript` / `_make_pdf_export`) and their `_emit_*_rows` orchestrators must mirror the per-base evaluator/oracle shapes documented here.

Three additional archetype paths target evaluators outside `compare_docx_strict`'s coverage. They run **on top of** TYPE_2 for the 3 specific base eval tasks.

| Archetype | Eval base | Evaluator (compound shape) | Oracle | Pool |
|---|---|---|---|---|
| `tabstops` | `0a0faba3` | `check_tabstops` (single func) | LO double-normalize + cp; py_code splits each non-empty paragraph at the n-th word with a `\t`, registers a right-aligned tab stop at the right margin (collapses runs into the first run before splitting; `sys.exit(42)` if no paragraphs qualify) | 5 paraphrase × n ∈ {2,3,4} |
| `subscript` | `0b17a146` | `compare_docx_files+compare_subscript_contains` (list-form, 2 vm_file results both pointing at the same `file_path`; 2 expected at the same `expected_path`) | LO double-normalize + cp; py_code finds the n-th non-empty paragraph (n ∈ {0,1,2}), splits the first run containing a digit so the digit is its own run with `font.subscript=True`, preserves bold/italic on surrounding splits | 5 paraphrase × 3 indices |
| `pdf_export` | `4bcb1253` | `compare_pdfs+compare_pdfs+compare_pdfs+compare_pdfs` (conj=`or`, list-form 4× vm_file results at standard PDF target dirs; 4× vm_file expected all pointing at the same `/tmp/perturb_expected_*.pdf`) | Pure-shell oracle: `soffice --headless --convert-to pdf` to a tmp dir, then `cp` to all 4 target dirs (Desktop / Documents / Downloads / home) AND to `/tmp/perturb_expected_*.pdf`. No docx normalize. No `perturb_config_step` (oracle does all setup). | 5 paraphrase |

**Why archetype rows are needed**: `compare_docx_strict` (writer's TYPE_2 evaluator) implicitly covers 16 of 19 missing eval evaluator rows by checking line spacing / font name / alignment / tables / highlight / PAGE field / break count. The remaining 3 — tab stops attribute, subscript char-format, and PDF text — are *not* in strict's check set, so without TYPE_3 the perturb dataset has zero training signal for those eval evaluators.

### Feasibility Notes

- **Skip `.odt` files**: `compare_docx_files` returns 0 on extension mismatch. Only `.docx` tasks are perturbable.
- **`_RUN_FORMAT_INFEASIBLE`** (`bold`, `strikethrough`, `italic`): tasks where the target paragraph range may include only empty/no-run paragraphs. Currently only `e528b65e` — already excluded in `_T2_VARIANTS` for that task; the set is a safety net.
- **`find_replace` excluded for long docs (`n_paras > 20`)**: the agent must read the whole doc to find the most-frequent word — infeasible vision-only in 15 turns.
- **`find_replace` progressively relaxes** the word-length regex (4+ → 3+ → 2+ → any) until a word is found; exits with `rc=42` if the document is empty.
- **`bold`, `strikethrough`, `italic` check for effectively-set paragraphs** (style-inherited counts) via the style chain and skip `rc=42` to avoid vacuous pass.
- **`strikethrough` and `italic`** are always checked by `compare_docx_strict` (`bool(run.font.strike)` / `bool(run.italic)` — not in the relaxed flags). No new evaluator options needed.
- **`uppercase` / `lowercase` advance past empty / runless / already-cased paragraphs** at runtime (mirroring the `bold`/`strike`/`italic` skip-already-set pattern) and exit with `rc=42` if no eligible paragraph exists. The dispatcher additionally caps the start-idx at `last_non_empty_idx + 1` (computed from the eval doc's `text_paras` analysis) so short tasks like `8472fece` (n_paras=4, only paras 0–1 non-empty) cannot generate a variant whose target lands beyond the last non-empty paragraph and silently no-ops the change → `trivial_pass`.
- **Subscript archetype (`_make_subscript`)** picks the n-th non-empty paragraph **that contains a digit** (filtered at runtime); earlier code took the n-th non-empty paragraph blindly and `sys.exit(42)`'d on digit-free first paragraphs (e.g., the `H2O_Factsheet` "Fact sheet" header), producing oracle FAIL with no expected file.

---

## Broken/Skip Tasks

| Reason | Examples |
|---|---|
| PDF export evaluator — no file oracle | Tasks using `check_pdf_pages`, `compare_pdfs` |
| Clipboard / external content | Tasks that require paste from external source |
| Realtime collaboration features | Google-Docs-style collaborative editing ops |
| `.odt` file format | Extension mismatch → evaluator returns 0 |
| No `.docx` file in config | Tasks with only `.odt` or `.doc` |

Run the following snippet to list all skip reasons:

```python
import json

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
writer = [r for r in rows if "writer" in r["task_id"]]

for r in writer:
    tid = r["task_id"]
    file_path = None
    for step in r["metadata"]["config"]:
        if step.get("type") == "download":
            for f in step["parameters"].get("files", []):
                file_path = f.get("path", "")
        if step.get("type") == "open":
            file_path = step["parameters"].get("path", "")
    fmt = r["evaluator"]["func"] if isinstance(r["evaluator"], dict) else r["evaluator"][0]["func"]
    skip = ""
    if not file_path or not file_path.endswith(".docx"):
        skip = f"non-docx ({file_path})"
    elif fmt not in ("compare_docx_strict", "compare_docx_files"):
        skip = f"wrong evaluator ({fmt})"
    print(f"{'SKIP' if skip else 'OK  '} {tid.split('_')[-1]}  {skip or ''}")
```

---

## Per-task Plan

> **Keep in sync with code.** Every change to the op pool or feasibility rules must be reflected in `perturb/libreoffice_writer.py` immediately, and vice versa. The table and the code are the single joint source of truth.

### Variant Count Rules

- **TYPE_1 eligible tasks** (4 tasks, `_TYPE1_FNS`): up to `max_type1` (default 1) TYPE_1 rows + up to `max_type2` (default 2) TYPE_2 rows → 3 rows per task; TYPE_1 op excluded from TYPE_2 pool at design time in `_T2_VARIANTS`
- **All other `.docx` tasks**: up to `max_type2` (default 2) TYPE_2 rows drawn from the explicit per-task candidate pool in `_T2_VARIANTS` (shuffled per call)
- **TYPE_3 eligible tasks** (3 bases, `_TYPE3_BASES`): up to `max_type3` (default 3) archetype rows targeting the eval-specific evaluator; runs *in addition to* TYPE_2, so these bases produce ~5 rows total
- Each task has exactly **4 TYPE_2 candidates** in `_T2_VARIANTS`, so `max_type2` up to 4 is always meaningful
- Constraints are encoded in `_T2_VARIANTS` at design time: `find_replace` excluded for long docs, T1 op excluded for T1-eligible tasks, `bold`/`strikethrough`/`italic` excluded for `e528b65e`
- Skip entirely if file is not `.docx` (e.g. `.odt`)

### Perturbable Tasks (22 docx tasks → ~50 rows)

Ops: **0**=bold **1**=font **2**=find_replace **3**=spacing **4**=uppercase **5**=strike **6**=lower **7**=italic

> **Keep in sync with `_T2_VARIANTS` in code.** The 4-candidate pool per task encodes all feasibility constraints at design time. Changing a task's pool here requires updating `_T2_VARIANTS` and vice versa.

| tid | n_paras | eval op (brief) | TYPE_1 template | TYPE_2 pool (4 ops) | notes |
|---|---|---|---|---|---|
| `0810415c` | 67 | Line spacing of paras 0-1 → double | spacing, para ≥2 | ① bold ② font ③ uppercase ④ italic | long doc+T1(spac) |
| `0a0faba3` | 12 | Left-align first 3 words, right-align rest | — | ① font ② find_replace ③ strike ④ lower | **+ TYPE_3 tabstops archetype (3 rows)** |
| `0b17a146` | 16 | Change "2" in H2O to subscript | — | ① bold ② find_replace ③ spacing ④ italic | **+ TYPE_3 subscript archetype (3 rows)** |
| `0e47de2a` | 39 | Add page numbers at bottom left | — | ① bold ② font ③ strike ④ lower | long doc |
| `0e763496` | 19 | Change font to Times New Roman throughout | font (not TNR) | ① bold ② find_replace ③ spacing ④ strike | T1(font) |
| `3ef2b351` | 5 | Center-align the heading | — | ① font ② find_replace ③ uppercase ④ italic | |
| `4bcb1253` | 27 | Export document to PDF | — | ① bold ② font ③ spacing ④ strike | long doc; **+ TYPE_3 pdf_export archetype (3 rows)** |
| `66399b0d` | 18 | Insert 7×5 empty table at cursor | — | ① font ② find_replace ③ spacing ④ italic | |
| `6ada715d` | 31 | Copy image from desktop to cursor | — | ① bold ② spacing ③ uppercase ④ lower | long doc |
| `6f81754e` | 101 | Track trains via signaling system | — | ① font ② spacing ③ strike ④ lower | long doc |
| `72b810ef` | 5 | Strikethrough the last paragraph | — | ① bold ② find_replace ③ uppercase ④ italic | |
| `8472fece` | 4 | Color words starting with vowels red | — | ① font ② find_replace ③ uppercase ④ strike | |
| `88fe4b2d` | 28 | Separate each sentence in third paragraph | — | ① bold ② spacing ③ lower ④ italic | long doc |
| `936321ce` | 43 | Convert comma-separated text to table | — | ① font ② spacing ③ uppercase ④ strike | long doc |
| `adf5e2c3` | 37 | Add bibliography entry | — | ① bold ② font ③ strike ④ italic | long doc |
| `b21acd93` | 3 | Set paragraph to 1.5 line spacing | spacing, para ≥1 | ① bold ② find_replace ③ uppercase ④ lower | T1(spac) |
| `bb8ccc78` | 36 | Share document for real-time collaboration | — | ① font ② spacing ③ strike ④ lower | long doc; eval=infeasible but perturb generates valid rows |
| `d53ff5ee` | 11 | Convert uppercase text to lowercase | — | ① find_replace ② uppercase ③ lower ④ italic | |
| `e246f6d8` | 40 | Change italic font size to 14pt and color | — | ① bold ② spacing ③ lower ④ italic | long doc |
| `e528b65e` | 4 | Capitalize first letter of each word | — | ① font ② find_replace ③ spacing ④ uppercase | no bold/strike/italic (empty last para) |
| `ecc2413d` | 118 | Insert blank page after current | — | ① bold ② spacing ③ uppercase ④ italic | long doc |
| `f178a4a9` | 51 | Set Times New Roman as default font | font (not TNR) | ① bold ② spacing ③ strike ④ lower | long doc+T1(font) |

### Not Perturbable (1 task)

| tid | Reason |
|---|---|
| `6a33f9b9` | `.odt` file — extension mismatch causes evaluator to return 0 |

---

## V4a Op Coverage Check

```python
from collections import Counter
import json, re

REQUIRED_OPS = {"bold", "font", "find_replace", "spacing", "uppercase", "strike", "lower", "italic"}

_OP_PATTERNS = [
    ("bold",         re.compile(r'r\.font\.bold\s*=\s*True')),
    ("font",         re.compile(r'r\.font\.name\s*=')),
    ("find_replace", re.compile(r're\.sub\(')),
    ("spacing",      re.compile(r'line_spacing\s*=')),
    ("uppercase",    re.compile(r'r\.text\s*=\s*r\.text\.upper\(\)')),
    ("strike",       re.compile(r'r\.font\.strike\s*=\s*True')),
    ("lower",        re.compile(r'r\.text\s*=\s*r\.text\.lower\(\)')),
    ("italic",       re.compile(r'r\.font\.italic\s*=\s*True')),
]

def _config_cmd(r):
    for s in r["metadata"].get("config", []):
        if s.get("type") == "execute":
            return s["parameters"].get("command", "")
    return ""

def _ops(r):
    cmd = _config_cmd(r)
    found = [name for name, pat in _OP_PATTERNS if pat.search(cmd)]
    return found or ["_action"]

all_perturb  = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
perturb_rows = [r for r in all_perturb if "writer" in r["task_id"]]

perturb_op_counts = Counter(op for r in perturb_rows for op in _ops(r))
missing = REQUIRED_OPS - set(perturb_op_counts.keys())
print(f"[{'FAIL' if missing else 'OK  '}] missing ops: {missing or 'none'}")
print(f"total rows: {len(perturb_rows)}")
print()

total = sum(perturb_op_counts.values()) or 1
print(f"{'op':<20}  {'count':>6}  {'share':>7}")
for op in sorted(REQUIRED_OPS | set(perturb_op_counts.keys())):
    cnt = perturb_op_counts.get(op, 0)
    flag = " <-- MISSING" if cnt == 0 else (" <-- HIGH (>30%)" if cnt/total > 0.30 else "")
    print(f"  {op:<18}  {cnt:>6}  {cnt/total:>6.1%}{flag}")
```

Targets:
- All 5 ops present (missing = none)
- No single op exceeds 30% of action slots

---

## V4b Perturb-Eval Match Verification

V4b has three parts: **instruction clarity** (is the instruction unambiguous?), **feasibility** (does the target object exist?), and **distribution match** (do perturb parameter choices resemble reasonable variety?). All three must pass.

### Part A: Instruction Clarity

Two checks: **(1)** polite%, save%, avg_words; **(2)** manual sample review.

```python
import json, re

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
writer = [r for r in rows if "writer" in r["task_id"]]

_SAVE_PAT   = re.compile(r'\bsave\b', re.IGNORECASE)
_POLITE_PAT = re.compile(r'^(Please|Could you|Can you|I need|I want|I\'d like|I would like)\b', re.IGNORECASE)

total = len(writer)
save_count   = sum(1 for r in writer if _SAVE_PAT.search(r["instruction"]))
polite_count = sum(1 for r in writer if _POLITE_PAT.match(r["instruction"]))
avg_words    = sum(len(r["instruction"].split()) for r in writer) / total
print(f"total={total}  save={save_count} ({save_count/total:.0%})  "
      f"polite={polite_count} ({polite_count/total:.0%})  avg_words={avg_words:.1f}")
# Writer-specific targets (matching eval distribution, not AGENTS.md generic):
#   save=0%, polite=18–25% (eval=17%), avg_words=25–30 (eval=27.9)
# Note: AGENTS.md says polite in [30,40]; writer overrides because eval polite is 17%.
```

Print 20 random rows for manual inspection:

```python
import json, random

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
writer = [r for r in rows if "writer" in r["task_id"]]

def _config_cmd(r):
    for s in r["metadata"].get("config", []):
        if s.get("type") == "execute":
            return s["parameters"].get("command", "")
    return ""

rng = random.Random(0)
for r in rng.sample(writer, min(20, len(writer))):
    print(f"[{r['task_id'].split('_')[-2]}]")
    print(f"  INSTR : {r['instruction']}")
    print(f"  ORACLE: {_config_cmd(r)[:160].strip()}")
    print()
```

What to look for per row:

| Check | What to verify |
|---|---|
| Op type | Instruction verb matches what oracle does (e.g., "bold" ↔ `r.font.bold = True`) |
| Paragraph index | "first"/"second"/etc. in instruction ↔ target paragraph index in oracle |
| Font name | Quoted font name ↔ `r.font.name = '...'` |
| Replace word | Quoted replacement word ↔ `re.sub(..., 'word', ...)` |
| Spacing value | "double" ↔ `line_spacing = 2.0`; "1.5 lines" ↔ `1.5` |
| No save leak | Instruction must not say "save the file" |
| Grammar | Instruction reads as natural English; no broken phrasing |

### Part B: Feasibility

For writer, feasibility means: the target paragraph index exists and the operation is non-trivial (won't silently no-op).

Cross-check each task's paragraph count from `/tmp/writer_full.json` against the paragraph index the oracle targets:

```python
import json, re
from collections import defaultdict

analysis = json.loads(open("/tmp/writer_full.json").read())
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
writer_perturb = [r for r in all_perturb if "libreoffice_writer" in r["task_id"]]

def _config_cmd(r):
    for s in r["metadata"].get("config", []):
        if s.get("type") == "execute":
            return s["parameters"].get("command", "")
    return ""

issues = []
for r in writer_perturb:
    short = r["task_id"].split("_")[-2]  # 8-hex source task id
    info = analysis.get(short, {})
    n_paras = info.get("n_paras", 99)  # default high so checks pass if unknown
    config = _config_cmd(r)
    if not config:
        continue

    # Extract target paragraph index from oracle (min(idx, len-1) logic in _make_bold etc.)
    # Oracle uses: target = min(idx, len(doc.paragraphs) - 1)
    # Check that the hardcoded idx (before clamping) <= n_paras - 1
    idx_matches = re.findall(r'\btarget\s*=\s*min\((\d+),', config)
    for idx_str in idx_matches:
        idx = int(idx_str)
        if n_paras > 0 and idx >= n_paras:
            issues.append((r["task_id"], f"target idx {idx} >= n_paras {n_paras}"))

    # find_replace: verify it isn't applied to long docs (n_paras > 20)
    if "re.sub" in config and n_paras > 20:
        issues.append((r["task_id"], f"find_replace on long doc n_paras={n_paras}"))

    # font: always feasible (no check needed)

print(f"[{'FAIL' if issues else 'OK  '}] feasibility: {len(issues)} violations")
for tid, reason in issues[:10]:
    print(f"  {tid}: {reason}")
```

Per-op feasibility notes from the per-task plan:

| op | Feasibility condition | Code handling |
|---|---|---|
| `bold` | ≥1 paragraph, not already effectively bold | `_make_bold` skips with `sys.exit(42)` if target is empty/already-bold |
| `font` | `.docx` file | Always feasible for any non-empty doc |
| `find_replace` | `n_paras ≤ 20`, ≥1 findable word | `sys.exit(42)` if no words found; excluded when `n_paras > 20` |
| `spacing` | ≥1 paragraph, not already at target spacing | `_make_spacing` skips with `sys.exit(42)` if already correct |
| `uppercase` | ≥1 non-empty paragraph | `_make_uppercase` advances past empty paragraphs |

### Part C: Distribution Match

**Invariant — pool-level, not sampling-level.** Distribution match is a constraint on the TYPE_2 candidate pool design, not on the sampled output. `max_type1` and `max_type2` are runtime parameters that control how many rows are drawn from each task's candidate pool — varying them must never cause distribution match to fail. This means the TYPE_2 candidate pool (the ops assigned to each task before any sampling occurs) must be designed so that, in aggregate across all tasks, op type frequencies already match the eval distribution. For TYPE_1-eligible tasks where the T1 op is excluded from the T2 pool to avoid duplicating the eval op, that exclusion must be compensated by the pool design of other tasks.

Three checks: **(1)** op type balance; **(2)** per-op parameter distributions (font names, spacing values, replacement words, paragraph indices); **(3)** paragraph index coverage (ops spread across paragraphs, not always targeting only the first).

```python
"""V4b distribution match — libreoffice_writer.
Run from repo root: uv run python this_script.py
Requires: train.perturb.jsonl generated, eval.jsonl present.
"""
import json, re
from collections import Counter, defaultdict

all_perturb  = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
writer_perturb = [r for r in all_perturb if "libreoffice_writer" in r["task_id"]]

def _config_cmd(r):
    for s in r["metadata"].get("config", []):
        if s.get("type") == "execute":
            return s["parameters"].get("command", "")
    return ""

# ── op type detector ─────────────────────────────────────────────────────────
_OP_PATTERNS = [
    ("bold",         re.compile(r'r\.font\.bold\s*=\s*True')),
    ("font",         re.compile(r"r\.font\.name\s*=\s*'")),
    ("find_replace", re.compile(r're\.sub\(')),
    ("spacing",      re.compile(r'line_spacing\s*=')),
    ("uppercase",    re.compile(r'r\.text\s*=\s*r\.text\.upper\(\)')),
]

def _op_type(r):
    cmd = _config_cmd(r)
    for name, pat in _OP_PATTERNS:
        if pat.search(cmd):
            return name
    return "_unknown"

# ── parameter extractors ─────────────────────────────────────────────────────
_FONTS = ["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana", "Calibri"]

def _p_font_name(r):
    m = re.search(r"r\.font\.name\s*=\s*'([^']+)'", _config_cmd(r))
    return m.group(1) if m else None

def _p_spacing_val(r):
    m = re.search(r'line_spacing\s*=\s*(\d+\.?\d*)', _config_cmd(r))
    if m:
        v = float(m.group(1))
        return "double(2.0)" if v >= 1.9 else "1.5 lines"
    return None

def _p_replace_word(r):
    m = re.search(r"'([a-z]+)'", r["instruction"])
    return m.group(1) if m else None

def _p_para_idx(r):
    # Extract the hardcoded starting idx from oracle: min(IDX, len(doc.paragraphs)-1)
    m = re.search(r'min\((\d+),\s*len\(doc\.paragraphs\)', _config_cmd(r))
    return int(m.group(1)) if m else None

def _compare(label, ctr, warn_ratio=3.0, min_share=0.05):
    total = sum(ctr.values()) or 1
    all_keys = sorted(ctr, key=lambda k: -ctr[k])
    print(f"\n  {label}:")
    print(f"    {'value':<25}  {'count':>6}  {'share':>7}")
    for k in all_keys:
        cnt = ctr[k]
        share = cnt / total
        flag = "  ← HIGH (>40%)" if share > 0.40 else ""
        print(f"    {str(k):<25}  {cnt:>6}  {share:>6.1%}{flag}")

print("=" * 65)
print("V4b — Distribution match verification (libreoffice_writer)")
print("=" * 65)

# C1: Op type balance ─────────────────────────────────────────────────────────
print()
print("C1: Op type balance")
op_counts = Counter(_op_type(r) for r in writer_perturb)
_compare("op types", op_counts)
REQUIRED_OPS = {"bold", "font", "find_replace", "spacing", "uppercase"}
missing = REQUIRED_OPS - set(op_counts.keys())
print(f"\n  [{'FAIL' if missing else 'OK  '}] missing ops: {missing or 'none'}")

# C2: Per-op parameter distributions ─────────────────────────────────────────
print()
print("C2: Per-op parameter distributions")

font_rows = [r for r in writer_perturb if _op_type(r) == "font"]
_compare("font — font name", Counter(_p_font_name(r) for r in font_rows if _p_font_name(r)))

spacing_rows = [r for r in writer_perturb if _op_type(r) == "spacing"]
_compare("spacing — spacing value", Counter(_p_spacing_val(r) for r in spacing_rows if _p_spacing_val(r)))

fr_rows = [r for r in writer_perturb if _op_type(r) == "find_replace"]
_compare("find_replace — replacement word", Counter(_p_replace_word(r) for r in fr_rows if _p_replace_word(r)))

# C3: Paragraph index coverage ────────────────────────────────────────────────
print()
print("C3: Paragraph index coverage")
para_op_rows = [r for r in writer_perturb if _op_type(r) in ("bold", "spacing", "uppercase")]
idx_ctr = Counter(_p_para_idx(r) for r in para_op_rows if _p_para_idx(r) is not None)
_compare("paragraph index (before clamping)", idx_ctr)
if idx_ctr:
    print(f"\n  idx 0 share: {idx_ctr.get(0,0)/sum(idx_ctr.values()):.0%} — target ≤ 50%")
    print(f"  idx range: 0–{max(idx_ctr.keys())} — target ≥ 4 distinct values")
```

**Targets for C1–C3:**
- C1: All 5 ops present; no single op exceeds 40% of rows. TYPE_1 rows (spacing/font) may shift distribution slightly toward those ops for the 4 eligible tasks.
- C2: font names drawn from all 6 options; spacing 50/50 between double and 1.5; replacement words from the 6-word pool. TYPE_1 font rows exclude Times New Roman (so TNR may have lower count).
- C3: paragraph indices cover 0–4 (not always targeting only idx=0); idx=0 share ≤ 50%

---

## V4c Eval Leakage Check

Writer perturb uses TYPE_1 (for 4 eligible tasks) and TYPE_2 for all tasks. TYPE_1 resamples the eval op with deliberately different parameters (different font, different para index), so instructions cannot match the eval verbatim. TYPE_2 uses structurally different ops. Since eval tasks target specific operations (cell formatting, find-replace with a specific word, etc.) and writer perturb draws from a general pool with fresh parameters (new font, new replacement word, new paragraph index), leakage is structurally prevented.

Runtime check: verify that no instruction duplicates an eval instruction verbatim.

```python
import json

all_eval    = {r["task_id"]: r["instruction"] for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
writer_perturb = [r for r in all_perturb if "libreoffice_writer" in r["task_id"]]

import re
leakage = []
for r in writer_perturb:
    base_tid = re.sub(r"^perturb_(.+)_[0-9a-f]{8}$", r"\1", r["task_id"])
    eval_instr = all_eval.get(base_tid, "")
    if eval_instr and r["instruction"].strip() == eval_instr.strip():
        leakage.append(r["task_id"])

print(f"[{'FAIL' if leakage else 'OK  '}] verbatim eval leakage: {len(leakage)} rows")
for tid in leakage[:10]:
    print(f"  {tid}")
```

---

## V4d Inter-Variant Uniqueness

Both rows generated from the same source eval task must differ in instruction and oracle code.

```python
import json, re
from collections import defaultdict

all_perturb    = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
writer_perturb = [r for r in all_perturb if "libreoffice_writer" in r["task_id"]]

def _config_cmd(r):
    for s in r["metadata"].get("config", []):
        if s.get("type") == "execute":
            return s["parameters"].get("command", "")
    return ""

by_source = defaultdict(list)
for r in writer_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src].append(r)

instr_dups  = []
oracle_dups = []
for src, rows in by_source.items():
    instrs  = [r["instruction"] for r in rows]
    configs = [_config_cmd(r) for r in rows]
    if len(instrs) != len(set(instrs)):
        instr_dups.append((src, instrs))
    if len(configs) != len(set(configs)):
        oracle_dups.append((src, configs))

print(f"[{'FAIL' if instr_dups  else 'OK  '}] duplicate instructions: {len(instr_dups)} source tasks")
print(f"[{'FAIL' if oracle_dups else 'OK  '}] duplicate oracle code:  {len(oracle_dups)} source tasks")
```

---

## V4e Instruction-Oracle Value Consistency

The instruction names concrete parameters — the oracle must agree.

```python
import json, re

all_perturb    = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
writer_perturb = [r for r in all_perturb if "libreoffice_writer" in r["task_id"]]

_FONTS = ["Arial", "Times New Roman", "Courier New", "Georgia", "Verdana", "Calibri"]

def _config_cmd(r):
    for s in r["metadata"].get("config", []):
        if s.get("type") == "execute":
            return s["parameters"].get("command", "")
    return ""

def _check(r):
    instr  = r["instruction"]
    config = _config_cmd(r)
    if not config.strip():
        return []
    errors = []

    # Font name: "Calibri" in instruction → r.font.name = 'Calibri' in oracle
    for font in _FONTS:
        if font in instr and f"r.font.name = '{font}'" not in config:
            errors.append(f"font '{font}' in instruction but not assigned in oracle")

    # Replacement word: '"{new}"' in instruction → re.sub(..., '{new}', ...) in oracle
    quoted = re.findall(r'"([a-z]+)"', instr)
    for w in quoted:
        if f"'{w}'" not in config and f'"{w}"' not in config:
            errors.append(f"replacement word '{w}' not found in oracle")

    # Line spacing: "double" ↔ line_spacing = 2.0, "1.5 lines" ↔ 1.5
    if re.search(r'\bdouble\b', instr, re.I) and "line_spacing" in config:
        if "2.0" not in config and "2)" not in config:
            errors.append("instruction says 'double' but oracle doesn't have 2.0")
    if re.search(r'1\.5\s+lines', instr, re.I) and "line_spacing" in config:
        if "1.5" not in config:
            errors.append("instruction says '1.5 lines' but oracle doesn't have 1.5")

    # No save leak in instruction
    if re.search(r'\bsave\b', instr, re.I):
        errors.append(f"instruction contains 'save': {instr!r}")

    return errors

failures = [(r["task_id"], errs) for r in writer_perturb if (errs := _check(r))]
print(f"[{'FAIL' if failures else 'OK  '}] instruction-oracle consistency: {len(failures)} rows with mismatches")
for tid, errs in failures[:10]:
    print(f"  {tid}:")
    for e in errs: print(f"    {e}")
```

---

## Expected Output

- **Structural perturb: 43 current rows** (was ~48 pre-P3-4-writer; later audit drops changed the generated total)
- Historical pre-drop design shape:
  - 23 eval tasks, 22 perturbable `.docx` tasks (1 `.odt` skipped)
  - 4 TYPE_1-eligible tasks: 3 rows each (1 TYPE_1 + 2 TYPE_2) → 12 rows
  - 15 other `.docx` tasks (excluding the 3 TYPE_3 bases): 2 rows each (TYPE_2 only) → 30 rows
  - 3 TYPE_3-eligible bases (`0a0faba3` / `0b17a146` / `4bcb1253`): 5 rows each (2 TYPE_2 + 3 TYPE_3 archetype) → 15 rows
- All 8 op types covered; TYPE_1 contributes spacing/font rows for the 4 eligible tasks; TYPE_3 contributes `check_tabstops`, `compare_subscript_contains` (compound), and `compare_pdfs` (×4 compound) evaluator coverage
- save=0%, polite=18–25% (eval=17%), avg_words=25–30 (eval=27.9)
- V2 pass-rate target: **100%**
