# LibreOffice Impress — Perturbation Plan

Domain-specific plan for `libreoffice_impress`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/libreoffice_impress.py`

---

## Step 0: Download Files and Inspect Content

Run once before drafting the per-task plan. Source PPTXs are on HuggingFace, URLs embedded in `eval.jsonl`.

```python
"""Run from repo root: uv run python this_script.py"""
import json, urllib.request, pathlib
from pptx import Presentation

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
impress = [r for r in rows if "impress" in r["task_id"]]
out_dir = pathlib.Path("/tmp/impress_pptx")
out_dir.mkdir(exist_ok=True)

# Download
for r in impress:
    tid = r["task_id"].split("_")[-1]
    for step in r.get("metadata", {}).get("config", []):
        if step.get("type") == "download":
            for f in step["parameters"].get("files", []):
                fname = out_dir / f"{tid}.pptx"
                if not fname.exists():
                    print(f"downloading {tid}...")
                    urllib.request.urlretrieve(f["url"], fname)

# Analyze and write /tmp/impress_full.json
result = {}
for pptx_path in sorted(out_dir.glob("*.pptx")):
    tid = pptx_path.stem
    try:
        prs = Presentation(str(pptx_path))
    except Exception as e:
        print(f"[SKIP] {tid}: {e}"); continue

    slides = []
    for idx_0, sld in enumerate(prs.slides):
        texts, pics, tables = [], [], []
        for sh in sld.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                texts.append(sh.shape_id)
            if sh.shape_type == 13:
                pics.append({"id": sh.shape_id, "h_cm": round(sh.height / 914400 * 2.54, 1)})
            if sh.has_table:
                t = sh.table
                tables.append({"id": sh.shape_id, "rows": len(t.rows), "cols": len(t.columns)})
        slides.append({"idx": idx_0 + 1, "texts": texts, "pics": pics, "tables": tables})

    result[tid] = {"n_slides": len(prs.slides), "slides": slides}
    print(f"{tid}: {len(prs.slides)} slides, txt={[s['idx'] for s in slides if s['texts']]}, "
          f"pics={[(s['idx'], p['h_cm']) for s in slides for p in s['pics']]}, "
          f"tbls={[(s['idx'], t['rows']) for s in slides for t in s['tables']]}")

pathlib.Path("/tmp/impress_full.json").write_text(json.dumps(result, indent=2))
print("wrote /tmp/impress_full.json")
```

`_slide_list(info, "texts/pics/tables")` in the perturb code reads from this JSON. All sN indices in the per-task table below are 1-based (python-pptx uses 0-based internally; subtract 1 in code).

---

## Op Type Definitions

| op | Description | x space |
|---|---|---|
| `set_font_color` | font color for textbox/title | red/blue/green/yellow/black/white/orange/purple/pink |
| `set_font_size` | font size for textbox/title | 10/12/14/16/18/20/24/28/32/36/40/44/48/60/72 pt |
| `set_font_style` | font style | bold/underline/italic/strikethrough |
| `set_font_name` | font name | Arial/Times New Roman/Calibri/Georgia/Verdana/Courier New/Trebuchet MS |
| `set_background_color` | slide background color | red/blue/green/yellow/purple/orange/cyan/white/black |
| `set_title_text` | title text | Introduction/Summary/Overview/Conclusion/Highlights/References/Agenda/Results/Discussion/Background |
| `set_text_alignment` | text alignment | left/center/right/justify |
| `set_picture_size` | picture height | 5/8/10/12/15/18/20/25 cm |
| `move_object` | object position | top/bottom/left/right/center |
| `set_slide_transition` | slide transition | dissolve/fade/wipe/push/uncover |
| `insert_table` | insert table | (3,2)/(4,3)/(5,2)/(6,4) rows×cols |
| `edit_table_cell` | edit table cell | first/second/last row + content list |
| `add_speaker_note` | add speaker note | short text from word pool |
| `reorder_slides` | reorder slides | first→last / last→first / swap N M |

All 14 must appear in the training set. Current eval directly covers ~8.

---

## Oracle Mechanics (LibreOffice Impress)

### Initial State Setup: `config_py`

`config_py` is a Python script injected into `config` just before the `open` step. It opens the PPTX with python-pptx and writes a state that differs from the target, ensuring reward = 0 before the agent acts.

### Ground Truth Generation: `expected_py`

`expected_py` is a Python script in `oracle_actions`. It opens the PPTX and writes the target state, then saves to `expected_path`. This is the ground truth.

### 3-Step Oracle

```
expected_py (python-pptx) -> writes target state to expected_path
domain _lo_normalize_cmd(expected_path, "pptx")  (LibreOffice --convert-to pptx round-trip)
cp expected_path -> file_path
```

**LO normalize** = LibreOffice headless `--convert-to pptx` round-trip that normalizes XML so both files go through identical serialization before comparison.

After oracle runs, `file_path == expected_path`. The evaluator then applies one more normalize pass to both sides. Do not add a second oracle-side normalize of `file_path`: LO normalize is not idempotent for PPTX geometry and the old 4th action produced spurious reward 0.

### Evaluator

`compare_pptx_files(file_path, expected_path, **examine_flags)` — runs on the host after pulling both files from the VM via `desktop_env/evaluators/metrics/slides.py`. `examine_*` flags (e.g. `examine_font_color=True`) are set per op in `_build_evaluator`.

### V2 Failure Modes (LibreOffice Impress)

| Failure | Cause | Fix |
|---|---|---|
| `trivial_pass` | `config_py` left initial state == target | Write a different initial value in `config_py` |
| `rc=42` | `raise SystemExit(42)` in `expected_py` — object not found | Exclude invalid slide/shape indices; check analysis data |
| `compare_pptx_files=0` | `expected_py` ran but file state unchanged | Fix logic in `expected_py` snippet |

---

## Perturbation Strategies

**TYPE_1 (same ops, new params)**: keep the eval's op list unchanged; independently resample `(slide_idx, param_value)` for each op. Reject combinations identical to eval (rejection sampling, up to 10 retries). **No-leakage guarantee**: a TYPE_1 row is accepted only when at least one `(slide_idx, param_value)` pair differs from the source eval task — any row that resamples back to the exact same parameter set is discarded and regenerated.

**TYPE_2 (different ops)**: replace the entire op list with op types not present in that eval task. Expands op coverage so all 14 op types appear in training. TYPE_1 and TYPE_2 are complementary — TYPE_1 augments existing skills, TYPE_2 ensures coverage of unseen ops. **No-leakage guarantee**: TYPE_2 uses op types disjoint from the source eval task by construction — structural difference is inherent to the strategy.

**TYPE_3 (atomic eval-evaluator perturb)** (P3-4-impress): targets one of 8 impress-domain evaluators with each row constructed against the actual eval evaluator (not python-pptx round-trip via `compare_pptx_files`). Each TYPE_3 row uses the eval task's downloaded source pptx as the base, mutates initial state via a config_step (so reward=0 before agent acts), and supplies an oracle that writes the gold state via raw zip / xml / python-pptx manipulation. Resampling space: where the evaluator accepts parameter rules (`check_auto_saving_time` minutes, `evaluate_presentation_fill_to_rgb_distance` rgb, `check_page_number_colors` color), each variant chooses a different rule value; binary evaluators (`check_presenter_console_disable`, `check_image_stretch_and_center`, `check_slide_orientation_Portrait`) keep the eval rules unchanged but vary the instruction paraphrase. Covered evaluators (8): `check_presenter_console_disable`, `check_auto_saving_time`, `evaluate_presentation_fill_to_rgb_distance`, `compare_images`, `check_image_stretch_and_center`, `check_page_number_colors`, `compare_audios`, `check_slide_orientation_Portrait`. Source eval bases (8): `0f84bef9` / `2cd43775` / `3b27600c` / `455d3c66` / `5d901039` / `ac9bb6cb` / `c59742c0` / `ce88f674`. **No-leakage guarantee**: TYPE_3 rule values are sampled from a pool that excludes the eval's value (e.g. `check_auto_saving_time` resamples to {5,10,15} excluding eval's 3min); for binary evaluators the instruction paraphrase pool contains no string-equal entries to the eval task's instruction, so the dispatcher's instruction-equality filter rejects any accidental matches.

---

## Per-task Plan

> **Keep in sync with code.** Every change to TYPE_1 resampling space or TYPE_2 variants must be reflected in `perturb/libreoffice_impress.py` immediately, and vice versa. The table and the code are the single joint source of truth — divergence silently produces wrong training tasks.

### Perturbable (34 tasks)

> **D2 reduction (Phase 3)**: All `set_slide_transition + other_op` compound TYPE_2 variants have been replaced with non-transition equivalents (text-alignment / font-style / font-color / etc.) since `check_transition + compare_pptx_files` is a dead skill (eval has no compound-transition tasks). Standalone single-`set_slide_transition` TYPE_2 variants are kept only in `7dbc52a6` (1 variant total) to preserve a deterministic single-transition example; combined with TYPE_1 from `21760ecb` this yields exactly 2 single-transition rows (down from 5 before D2). The 8 compound transition rows were eliminated entirely.

| tid | n_slides | eval ops | TYPE_1 resampling space | TYPE_2 variants |
|---|---|---|---|---|
| `04578141` | 22 | set_font_color (s1: 3 textboxes → yellow/red/green) | same 3 ops, resample slide(txt=s1-s22) + color | ① edit_table_cell(**s18** first row, "Col A"/"Col B") ② reorder_slides(s11→s7, span=4) ③ add_speaker_note(**s3**) + set_font_name(**s5**, Calibri) ④ set_text_alignment(**s4**, center) + set_font_size(**s4**, 24pt) **D2: was set_slide_transition+set_font_size** |
| `05dd4c1d` | 6 | set_text_alignment (s3→right, s4→center, s5→left) | same 3 ops, resample slide(txt=s2,s3,s6 only — s1/s4/s5 lack text frames) + alignment | ① set_picture_size(**s2** pic, 15cm) + set_font_color(**s6**, red) ② set_font_name(**s3**, Verdana) + set_background_color(**s5**, blue) ③ add_speaker_note(**s1**) + set_font_style(**s6**, italic) **D2: was add_speaker_note+set_slide_transition** |
| `08aced46` | 2 | set_title_text("Note") + set_text_alignment(right) | same 2 ops, resample title text + alignment | ① add_speaker_note(**s1**) ② set_background_color(**s2**, blue) + set_font_style(**s1**, italic) ③ set_font_name(**s1**, Arial) + set_font_size(**s1**, 28pt) |
| `15aece23` | 2 | move_object (title s2 → bottom) | same 1 op, resample position(txt+tbl=s2) | ① set_text_alignment(**s2**, center) ② add_speaker_note(**s1**) ③ set_font_color(**s2**, blue) + set_font_size(**s2**, 20pt) |
| `21760ecb` | 24 | set_slide_transition("dissolve", s1) | same 1 op, resample transition + slide(txt+pic=s1-s24) | ① add_speaker_note(**s3**) + set_font_color(**s5**, blue) ② reorder_slides(s11→s15, span=4) ③ insert_table(**s2**, 3r×2c) + set_font_name(**s4**, Calibri) ④ set_background_color(**s6**, red) + set_font_size(**s8**, 20pt) |
| `2b94c692` | 2 | move_object (image s2 → right) | same 1 op, resample position(txt+pic=s2) | ① set_font_name(**s2**, Times New Roman) ② set_text_alignment(**s2**, center) ③ set_background_color(**s1**, blue) + set_picture_size(**s2** pic, 12cm) |
| `3161d64e` | 22 | set_font_size (s14: tb1→60pt, tb2→28pt) | same 2 ops, resample slide(txt+pic=s1-s22, tbl=s18) + pt | ① reorder_slides(s7→s4, span=3) + set_font_style(**s5**, italic) ② edit_table_cell(**s18** first row, "A"/"B"/"C") ③ set_font_color(**s3**, blue) + set_background_color(**s7**, green) **D2: was set_slide_transition+set_background_color** ④ add_speaker_note(**s10**) + set_font_name(**s14**, Georgia) |
| `358aa0a7` | 28 | set_font_name (all slides) | same 1 op, resample font(txt=s1-s28) | ① set_font_style(**s5**, italic) + set_background_color(**s10**, purple) **D2: was set_slide_transition+set_background_color** ② reorder_slides(s14→s10, span=4) ③ add_speaker_note(**s3**) + set_font_size(**s7**, 20pt) ④ set_picture_size(**s6** pic, 15cm) + set_font_color(**s12**, red) |
| `39be0d19` | 3 | insert_table (5r×2c in s3) | same 1 op, resample (rows,cols) or slide(txt=s1-s3) | ① set_title_text(**s1**, "Overview") + set_background_color(**s2**, blue) ② set_font_style(**s3**, bold) **D2: was standalone set_slide_transition** ③ add_speaker_note(**s2**) + set_font_color(**s3**, red) |
| `3b27600c` | 19 | set_background_color (all → blue) | same 1 op, resample color(txt=s1-s19, no pic/tbl) | ① set_text_alignment(**s3**, center) + set_font_size(**s7**, 24pt) **D2: was set_slide_transition+set_font_size** ② reorder_slides(s6→s10, span=4) + set_font_color(**s5**, blue) ③ insert_table(**s2**, 4r×3c) + set_font_name(**s4**, Calibri) ④ add_speaker_note(**s10**) + set_font_style(**s12**, italic) — also **TYPE_3** (P3-4-impress): atomic `evaluate_presentation_fill_to_rgb_distance` perturb (×2 variants resampling rgb) |
| `4ed5abd0` | 6 | set_font_color (s2,3,5 → black) + set_font_style(underline) | same 2 ops, resample slide + color/style ⚠️ s2/3/5 have no title placeholder; s6 has formal title | ① set_title_text(**s6**, "Summary") + set_font_size(**s4**, 20pt) ② set_text_alignment(**s2**, right) + set_background_color(**s5**, yellow) **D2: was set_slide_transition+set_background_color** ③ add_speaker_note(**s3**) + set_font_name(**s1**, Arial) |
| `550ce7e7` | 7 | set_font_style (strikethrough, first+second line) | same 1 op, resample style + slide(txt=s1-s7, pic=s1,s3) | ① set_font_name(**s4**, Georgia) + insert_table(**s6**, 3r×2c) ② set_picture_size(**s3** pic, 15cm) + set_font_color(**s5**, blue) ③ set_font_style(**s2**, bold) + set_background_color(**s7**, purple) **D2: was set_slide_transition+set_background_color** ④ add_speaker_note(**s1**) + set_text_alignment(**s4**, center) |
| `57667013` | 5 | set_font_color (s5, all → yellow) | same 1 op, resample slide(txt=s3,s5 — s1/s2/s4 lack text frames) + color | ① set_text_alignment(**s3**, right) + set_picture_size(**s2** pic, 12cm) ② add_speaker_note(**s4**) + set_font_size(**s5**, 20pt) ③ set_font_name(**s3**, Verdana) + set_background_color(**s1**, green) |
| `5c1a6c3d` | 5 | set_font_style(bold,s1) + set_font_size(44pt,s1) + set_font_style(underline,s1) | same 3 ops, each resample slide(txt=s1,s2,s3,s5 — s4 lacks text frames; pic=s1-s5) + param | ① add_speaker_note(**s3**) + set_font_color(**s5**, blue) ② set_picture_size(**s2** pic, 15cm) + set_background_color(**s5**, yellow) ③ set_title_text(**s3**, "Results") + set_font_name(**s5**, Arial) |
| `5cfb9197` | 5 | edit_table_cell (s4 first row → T1/T2/T3/T4) | same 1 op, resample row + content(txt=s1-s5, tbl=s4) | ① set_font_name(**s3**, Verdana) + set_background_color(**s5**, red) ② add_speaker_note(**s2**) + set_text_alignment(**s3**, right) ③ insert_table(**s1**, 3r×2c) + set_font_color(**s2**, blue) |
| `73c99fb9` | 2 | edit_textbox_content (s2 → "Page 1") | ⚠️ TYPE_1 returns [] (not in 14-op pool) | ① add_speaker_note(**s1**) ② set_font_color(**s1**, blue) + set_background_color(**s2**, green) ③ set_text_alignment(**s1**, center) |
| `7ae48c60` | 6 | set_picture_size (s3→20cm, s4→30cm, s6→25cm) | same 3 ops, resample slide(pic=s3,s4,s6) + height ⚠️ s3 has 7 decorative icons ~1.1cm; targets are s4(28.6cm)/s6(26.0cm) | ① set_text_alignment(**s2**, center) + set_font_color(**s5**, red) ② add_speaker_note(**s4**) + set_font_name(**s1**, Calibri) ③ set_font_size(**s3**, 24pt) + set_background_color(**s5**, purple) **D2: was set_slide_transition+set_background_color** |
| `7dbc52a6` | 2 | add_speaker_note(=title) + set_font_style(bold title) | same 2 ops, resample note + style(txt+pic=s2) | ① set_background_color(**s1**, blue) + set_font_color(**s2**, yellow) ② set_slide_transition(**s1**, fade) |
| `841b50aa` | 1 | add_speaker_note("APP") + set_background_color(purple) | same 2 ops, resample note + color(tbl=s1, no txt) | ① set_background_color(**s1**, blue) **D2: was standalone set_slide_transition (replaced because s1 has only a table)** ② edit_table_cell(**s1** first row, "Item A"/"Item B") |
| `8979838c` | 1 | set_background_color(purple) + add_speaker_note(title) | same 2 ops, resample color + note(txt+pic=s1) | ① set_font_style(**s1**, bold) ② set_font_name(**s1**, Verdana) ③ set_font_color(**s1**, blue) |
| `986fc832` | 1 | set_font_style(underline) + set_font_color(incl table) | same 2 ops, resample style + color(txt+tbl=s1) | ① set_background_color(**s1**, red) ② edit_table_cell(**s1** first row, "P"/"Q"/"R") ③ set_text_alignment(**s1**, center) |
| `9cf05d24` | 2 | set_background_color (s1 → green) | same 1 op, resample color + slide(txt+pic+tbl=s2) | ① set_font_size(**s2**, 20pt) + set_picture_size(**s2** pic, 12cm) ② edit_table_cell(**s2** first row, "Item 1"/"Item 2") + set_font_color(**s2**, blue) ③ set_text_alignment(**s2**, center) |
| `9ec204e4` | 24 | duplicate_slides (last 2 → alternating A B A' B') | ⚠️ TYPE_1 returns [] (complex, no oracle mode) | ① reorder_slides(s9→s13, span=4) + set_font_color(**s3**, red) ② set_font_style(**s5**, underline) + set_background_color(**s10**, blue) **D2: was set_slide_transition+set_background_color** ③ add_speaker_note(**s7**) + set_font_name(**s12**, Calibri) ④ set_picture_size(**s5** pic, 15cm) + set_font_size(**s3**, 20pt) |
| `a434992a` | 1 | set_font_size(12) + set_font_color(orange) + set_background_color | same 3 ops, each resample params(txt=s1) | ① set_font_style(**s1**, italic) ② set_text_alignment(**s1**, center) |
| `a53f80cd` | 5 | set_font_color(s2-3 title→black) + set_font_style(bold) + delete_shapes(s4) | TYPE_1 resamples first 2 ops only; delete_shapes excluded (txt=s1-s5, pic=s1,s3,s4,s5) | ① set_picture_size(**s4** pic, 15cm) + set_font_size(**s2**, 20pt) ② set_font_color(**s5**, purple) + set_font_name(**s2**, Georgia) **D2: was set_slide_transition+set_font_name** ③ add_speaker_note(**s3**) + set_background_color(**s4**, blue) |
| `a669ef01` | 10 | indent_adjust (s3 bullet, complex format) | ⚠️ TYPE_1 returns [] | ① insert_table(**s4**, 4r×3c) + set_font_color(**s6**, red) ② set_text_alignment(**s2**, center) + set_background_color(**s7**, yellow) **D2: was set_slide_transition+set_background_color** ③ set_picture_size(**s3** pic, 12cm) + set_font_size(**s5**, 24pt) |
| `ac1b39ff` | 3 | move_object (table s3 → bottom) | same 1 op, resample position only (obj=table, s3 fixed) | ① set_text_alignment(**s2**, center) **D2: was standalone set_slide_transition; s1 has no text frames** ② edit_table_cell(**s3** first row, "R1"/"R2"/"R3") + set_font_color(**s2**, blue) ③ reorder_slides(s3→s2, span=1) + set_background_color(**s2**, red) |
| `ac9bb6cb` | 15 | set_font_color (footer → red, slide master) | ⚠️ python-pptx footer color support limited, TYPE_1 returns [] | ① set_picture_size(**s5** pic, 15cm) + set_font_size(**s3**, 20pt) ② add_speaker_note(**s7**) + set_background_color(**s10**, blue) ③ reorder_slides(s8→s4, span=4) + set_font_name(**s8**, Calibri) ④ insert_table(**s2**, 3r×2c) + set_font_style(**s4**, italic) — also **TYPE_3** (P3-4-impress): atomic `check_page_number_colors` perturb (×2 variants resampling color via slideMaster1.xml srgbClr patch) |
| `af2d657a` | 6 | set_title_text("Happy Family") + set_font_name("Microsoft JhengHei") | same 2 ops, resample title text + font ⚠️ all slides have empty title placeholders (no body text frames anywhere) — set_title_text creates the text first, then set_font_name targets it | ① insert_table(**s3**, 3r×2c) ② set_background_color(**s2**, blue) + add_speaker_note(**s4**) ③ set_background_color(**s1**, orange) **D2: was standalone set_slide_transition; replaced standalone set_font_color (no body text frames anywhere) with set_background_color** |
| `b8adbc24` | 2 | set_title_text(s2 → "Online Shopping") | same 1 op, resample title text ⚠️ must copy s1 title attributes | ① edit_table_cell(**s1** first row, "A"/"B"/"C") + set_font_style(**s1**, italic) ② set_background_color(**s2**, blue) + set_picture_size(**s1** pic, 12cm) |
| `e4ef0baf` | 6 | set_picture_size(s3→20cm) + set_font_size(s6→40pt) | same 2 ops, resample slide(txt=s1-s6, pic=s1-s5) + height/pt | ① set_font_style(**s3**, italic) + set_background_color(**s5**, purple) ② set_text_alignment(**s1**, left) + set_font_name(**s6**, Georgia) **D2: was set_slide_transition+set_font_name** ③ add_speaker_note(**s4**) + set_text_alignment(**s6**, right) ④ set_font_color(**s4**, red) + set_font_size(**s6**, 24pt) |
| `ed43c15f` | 8 | move_object(pic s2→top) + set_font_style(underline, s1+s2) | same 2 ops, resample position + slide + style | ① edit_table_cell(**s4** first row, "T1"/"T2") + set_font_color(**s3**, blue) ② reorder_slides(s6→s2, span=4) + set_background_color(**s5**, green) ③ add_speaker_note(**s6**) + set_font_size(**s6**, 20pt) |
| `edb61b14` | 5 | set_font_name (last slide → "Times New Roman") | same 1 op, resample font + slide(txt=s1-s5, pic=s2, tbl=s4) | ① set_title_text(**s3**, "Summary") + set_text_alignment(**s5**, center) **D2: was set_title_text+set_slide_transition** ② edit_table_cell(**s4** first row, "V1"/"V2"/"V3") + set_font_color(**s2**, red) ③ set_picture_size(**s2** pic, 12cm) + set_background_color(**s5**, blue) |
| `f23acfd2` | 1 | add_bullet_point (slide 1) | same 1 op, resample bullet text(txt=s1) | ① set_font_color(**s1**, red) ② set_background_color(**s1**, blue) ③ set_font_style(**s1**, bold) |

### TYPE_3 Atomic Eval-Evaluator Perturb (8 tasks, P3-4-impress)

These tasks were previously "Not Perturbable" because their eval evaluators sit outside the `compare_pptx_files` family covered by TYPE_1/TYPE_2. Phase 3 (P3-4-impress) adds atomic `_t3_*` builders that target each evaluator directly, mirroring the eval evaluator's `result`/`expected` payload exactly and constructing an oracle that writes the gold state via raw zip/xml/python-pptx manipulation.

| tid | Evaluator | TYPE_3 mechanic | Variants |
|---|---|---|---|
| `0f84bef9` | `check_presenter_console_disable` | Write `registrymodifications.xcu` with `EnablePresenterScreen=false` (init: `=true`). | 1 |
| `2cd43775` | `check_auto_saving_time` | Write xcu with `AutoSaveTimeIntervall=N` minutes; resample N ∈ {5,10,15}. | 2 |
| `3b27600c` | `evaluate_presentation_fill_to_rgb_distance` | Set every slide's background fill to a target RGB; resample {red, green, yellow, purple}. (Eval task already has TYPE_1+TYPE_2; TYPE_3 adds atomic eval-evaluator coverage.) | 2 |
| `455d3c66` | `compare_images` | Render pptx → `/tmp/res_gold.png` during setup; oracle copies that rendered image to Desktop `res.png` so SSIM=1.0 without exposing a gold file on the Desktop. Initial `res.png` is absent so pre-oracle reward is exactly 0. | 1 |
| `5d901039` | `check_image_stretch_and_center` | Snapshot `file_path → expected_path` (original), then stretch slide-1 picture to slide bounds in-place. Init shrinks the image so reward=0. | 1 |
| `ac9bb6cb` | `check_page_number_colors` | Patch `slideMaster1.xml` srgbClr to a hex matching the eval's red/blue/green classifier; resample 2 colors. | 2 |
| `c59742c0` | `compare_audios` | Embed `Baseball.mp3` (already on Desktop from eval config) into pptx slide-1 via zip extract / rels patch / repack. Mirror synth `audio_insert` oracle. | 1 |
| `ce88f674` | `check_slide_orientation_Portrait` | Swap `prs.slide_width` and `prs.slide_height` so width<height. Init forces landscape. | 1 |

Total TYPE_3 rows: **11** (1+2+2+1+1+2+1+1).

### Not Perturbable (6 tasks)

| tid | Reason |
|---|---|
| `0a211154` | Requires image understanding; i space not parameterizable |
| `70bca0cc` | Background color = title color; x depends on semantic reading |
| `af23762e` | AI-generated summary; no deterministic result |
| `bf4e9888` | Depends on external PNG file set |
| `c82632a4` | Depends on external none.png; low value |
| `ef9d12bd` | UI state restore; no file parameters |
| `a097acff` | Save-as; low training value |

---

## Implementation Spec

### Function Signature

```python
def perturb_impress_per_task(
    eval_row: dict,
    rng: random.Random,
    perturb_types: tuple[str, ...] = (PERTURB_TYPE_1, PERTURB_TYPE_2, PERTURB_TYPE_3),
    max_type1: int = 1,
    max_type2: int = 2,
    max_type3: int = 2,
) -> list[dict]:
```

For TYPE_3 only tasks (`2cd43775` has no downloaded pptx), the dispatcher skips TYPE_1/TYPE_2 (they require `file_path`) and runs TYPE_3 directly.

`_build_row(eval_row, ops, file_path, expected_path, perturb_type, rng)` assembles each row: generates `config_py`/`expected_py` strings via `_build_py`, sets evaluator `examine_*` flags via `_build_evaluator`, generates NL instruction via `_build_instruction`, calls `make_perturb_row`.

### Generation Entry Point / Budgets

```python
# __main__.py
parser.add_argument("--track", choices=["synth", "perturb", "all"], default="all")
parser.add_argument("--domain", default="all")
parser.add_argument("-o", "--output", default=None)
parser.add_argument("--seed", type=int, default=42)

# perturb/dispatch.py
apply_structural_perturbation(eval_row, rng, **budget_overrides)

# libreoffice_impress.py
def perturb_impress_per_task(
    eval_row,
    rng,
    perturb_types=(PERTURB_TYPE_1, PERTURB_TYPE_2, PERTURB_TYPE_3),
    max_type1=1,
    max_type2=2,
    max_type3=2,
): ...
```

The generator CLI does not expose Impress-specific budget flags. Current
defaults live in the domain function signature; the dispatcher filters any API
level `budget_overrides` to parameters the domain accepts.

### config_py / expected_py per Op

The "initial value" in config_py must differ from the target to ensure pre-oracle reward = 0.

| op | config_py (initial state) | expected_py (target state) |
|---|---|---|
| `set_font_color` | shapes → gray(128,128,128) | shapes → (r,g,b) |
| `set_font_size` | → 12pt (or 14pt if target=12) | → target pt |
| `set_font_style` | target style → False | → True |
| `set_font_name` | → "Calibri" (or "Arial" if target=Calibri) | → target font |
| `set_background_color` | → gray(128,128,128) | → (r,g,b) |
| `set_title_text` | → "TempTitle" | → target text |
| `set_text_alignment` | → LEFT | → target alignment |
| `set_picture_size` | → 10cm (or 12cm if target=10) | → target cm, preserve ratio |
| `move_object` | → center | → target position |
| `add_speaker_note` | → "" | → target text |
| `set_slide_transition` | delete transition XML | inject target transition XML (lxml) |
| `insert_table` | (no change needed) | add_table(rows, cols) |
| `edit_table_cell` | target row → "A","B","C",... | → target content list |
| `reorder_slides` | move(src, src) — no-op | move(src, dst) |
| `add_bullet_point` (TYPE_1 only) | (no change needed) | add_paragraph(text) |

Multi-op tasks: merge all snippets into one script with a single `prs.save()`.

### Instruction Style (Paraphrase Pool + Multi-Op Fusion)

> **Keep in sync with code.** Pool sizes, fusion probabilities, and polite-prefix rate listed here must match `_op_instr` / `_build_instruction` in `perturb/libreoffice_impress.py`.

`_op_instr(op, rng)` draws from a **5-paraphrase pool per op type** (14 op types × 5 = 70 total paraphrases). Each pool mixes:

- **Imperative** (`"Change the font color of all text on slide 1 to red"`)
- **Context-then-action** (`"On slide 1, switch every text run's color over to red so the whole slide reads in one color"`)
- **Action-with-rationale** (`"For slide 2, recolor each text element to red — every textbox should pick up the same shade"`)

Per-op paraphrase length spans **8–28 words** to spread `avg_words` across the eval baseline (~22 words/instruction).

`_build_instruction(ops, rng)` fuses the per-op parts:

| Scenario | Behavior |
|---|---|
| 1 op | `"<op>."` |
| 2 ops (~85%) | `"<op1>, and <op2>."` — single-sentence "and"-fusion |
| 2 ops (~15%) | `"<op1>. <op2>."` — two new sentences (capital after `. `) |
| 3+ ops (~75%) | `"<op1>. <op2>. <op3>."` — bare new sentences |
| 3+ ops (~25%) | `". Also, "` / `". Then, "` / `". Next, "` / `". Additionally, "` connector |

**Why so few `Also/Then/Next/Additionally` connectors?** Eval baseline uses these multi-sentence connectors at only ~2% of instructions. Earlier perturb output ran at ~39% — the dominant style mismatch. The fix above keeps the connector probability ≤ 25% of multi-op rows, which lands at ~1–3% overall (most rows are 1-op or 2-op-fused).

After fusion, polite-wrapping fires at **~40% probability gated by verb-start**: only paraphrases whose first word is an imperative verb (`change`/`set`/`apply`/`move`/...) become `"Please ..."` / `"Could you help me ..."` / `"I need to ..."` / `"I'd like to ..."` / `"I want to ..."` / `"Can you ..."`. Paraphrases that start with a preposition (`"On slide N, ..."` / `"For slide N, ..."`) are skipped — wrapping those would read ungrammatically (`"Please on slide N, ..."`). The verb-gate halves the eligible pool, so realised polite% lands ~17–25% (eval baseline 17%).

**V3 distribution targets** (perturb vs eval, 20-seed sweep):

| metric | eval | perturb (mean) | perturb (range) |
|---|---|---|---|
| `polite` | 17% | ~25% | 15–35% |
| `multi_sep` | 2% | ~1% | 0–3% |
| `avg_words` | 22.6 | ~22.0 | 20.9–23.0 |
| `save the file` | 0% | 0% | 0% |

Never append `"Save the file."` — eval never says this and the agent should not save explicitly.

---

## V4a Op Coverage Check

Three checks: **(1)** all 14 required ops appear in perturb; **(2)** actions-per-task count distribution matches eval (e.g. if 80% of eval tasks have 3 actions, perturb should too); **(3)** each op's share of total action slots stays proportional to eval (prevents an op that accounts for 5% of eval slots from ballooning to 30% in perturb).

```python
from collections import Counter
import json, re

REQUIRED_OPS = {
    "set_font_color", "set_font_size", "set_font_style", "set_font_name",
    "set_background_color", "set_title_text", "set_text_alignment",
    "set_picture_size", "move_object", "add_speaker_note",
    "set_slide_transition", "insert_table", "edit_table_cell", "reorder_slides",
}

# knob_assignment is not stored in rows; extract ops from oracle code instead.
# eval rows use wget oracle — fall back to instruction sep count.
_OP_PATTERNS = [
    ("set_font_color",       re.compile(r'font\.color\.rgb\s*=')),
    ("set_font_size",        re.compile(r'font\.size\s*=\s*Pt\(')),
    ("set_font_style",       re.compile(r'font\.(bold|italic|underline)\s*=\s*True|sngStrike')),
    ("set_font_name",        re.compile(r"font\.name\s*=")),
    ("set_background_color", re.compile(r'fore_color\.rgb\s*=')),
    ("set_title_text",       re.compile(r'_tsh\.text=')),
    ("set_text_alignment",   re.compile(r'PP_ALIGN\.')),
    ("set_picture_size",     re.compile(r'int\(\d+\*360000\)')),
    ("move_object",          re.compile(r'_sh\.(top|left)\s*=')),
    ("add_speaker_note",     re.compile(r'notes_slide\.notes_text_frame\.text\s*=')),
    ("set_slide_transition", re.compile(r"<p:[a-z]+|qn\(.p:[a-z]+")),
    ("insert_table",         re.compile(r'add_table\(')),
    ("edit_table_cell",      re.compile(r'_sh\.table\.rows')),
    ("reorder_slides",       re.compile(r'_lst\.insert\(')),
]
_SEP = re.compile(r"\. (?:Also|Additionally|Then|Next),")

def _oracle_text(r):
    parts = []
    for a in (r.get("metadata", {}).get("others") or {}).get("oracle_actions", []):
        cmd = a.get("parameters", {}).get("command", "")
        if isinstance(cmd, str):
            parts.append(cmd)
    return "\n".join(parts)

def _ops(r):
    oracle = _oracle_text(r)
    if oracle:
        found = [name for name, pat in _OP_PATTERNS if pat.search(oracle)]
        if found:
            return found
    n = len(_SEP.findall(r["instruction"])) + 1
    return ["_action"] * n

all_eval = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
eval_rows = [r for r in all_eval if "impress" in r["task_id"]]
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
perturb_rows = [r for r in all_perturb if "impress" in r["task_id"]]

# --- (1) Missing ops ---
perturb_op_counts = Counter(op for r in perturb_rows for op in _ops(r))
missing = REQUIRED_OPS - set(perturb_op_counts.keys())
print(f"[{'FAIL' if missing else 'OK  '}] missing ops: {missing or 'none'}")
print()

# --- (2) Actions-per-task length distribution ---
def length_dist(rows):
    c = Counter(len(_ops(r)) for r in rows)
    total = sum(c.values())
    return {k: f"{v/total:.0%}" for k, v in sorted(c.items())}

print("Actions-per-task distribution (% of rows):")
print(f"  eval   : {length_dist(eval_rows)}")
print(f"  perturb: {length_dist(perturb_rows)}")
print()

# --- (3) Per-op count in perturb (absolute, not ratio vs eval) ---
# eval uses wget oracle so ops can't be extracted; check perturb balance only.
print(f"Per-op count in perturb (total action slots={sum(perturb_op_counts.values())}):")
print(f"  {'op':<28}  {'count':>6}  {'share':>7}")
for op in sorted(REQUIRED_OPS | set(perturb_op_counts.keys())):
    cnt = perturb_op_counts.get(op, 0)
    total = sum(perturb_op_counts.values()) or 1
    flag = " <-- MISSING" if cnt == 0 else (" <-- HIGH (>15%)" if cnt/total > 0.15 else "")
    print(f"  {op:<28}  {cnt:>6}  {cnt/total:>6.1%}{flag}")
```

Targets:
- All 14 ops present in perturb (missing = none)
- Actions-per-task distribution roughly matches eval (±10pp per bucket)
- No single op exceeds 15% of total perturb action slots (eval uses wget oracle so per-op ratio vs eval is not computable)

## V4b Perturb-Eval Match Verification

V4b has three parts: **instruction clarity** (is the instruction unambiguous?), **feasibility** (does the target object exist?), and **distribution match** (do perturb parameter choices resemble eval's?). All three must pass.

### Part A: Instruction Clarity

Manual inspection: sample ~20 rows from the generated `train.perturb.jsonl` and verify each instruction accurately and unambiguously describes the oracle action. There is no universal script for this — the check is semantic.

What to look for per row:

| Check | What to verify |
|---|---|
| Op type | Instruction verb matches what oracle does (e.g., "change font color" ↔ `font.color.rgb =`) |
| Slide index | Slide N in instruction = oracle acts on `prs.slides[N-1]` (or `_lst[N-1]` for reorder) |
| Parameter value | Color/font/size/transition named in instruction matches oracle's literal value |
| Object identity | "the picture" → slide has 1 content pic (h_cm > 3); "the table" → slide has 1 table |
| Grammar | Instruction reads as natural English; no broken connectors, no truncation |
| No save leak | Instruction must not say "save the file" |

Print sample for inspection:

```python
import json, random
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
impress = [r for r in rows if "impress" in r["task_id"]]

def oracle_text(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r.get("metadata", {}).get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

rng = random.Random(0)
for r in rng.sample(impress, 20):
    print(f"[{r['task_id'].split('_')[-2]}]")
    print(f"  INSTR : {r['instruction']}")
    # first oracle execute block (the expected_py)
    first_cmd = next(
        (a["parameters"]["command"] for a in (r.get("metadata", {}).get("others") or {}).get("oracle_actions", [])
         if "PYEOF" in a.get("parameters", {}).get("command", "")), ""
    )
    print(f"  ORACLE: {first_cmd[:200].strip()}")
    print()
```

### Part B: Feasibility

Print per-task slide structure, then manually cross-check every TYPE_2 variant in the per-task table above:

```python
import json
analysis = json.loads(open("/tmp/impress_full.json").read())

for tid, info in sorted(analysis.items()):
    n = info["n_slides"]
    txt  = [s["idx"] for s in info["slides"] if s["texts"]]
    pics = [(s["idx"], p["h_cm"]) for s in info["slides"] for p in s["pics"]]
    tbls = [(s["idx"], t["rows"]) for s in info["slides"] for t in s["tables"]]
    print(f"{tid} ({n}sl)  txt={txt}  pics={pics}  tbls={tbls}")
```

For each `(tid, slide_idx, op)` in the per-task table, verify:

| op | Check | Why |
|---|---|---|
| `set_font_color/size/style/name/text_alignment` | target slide in txt list | op applies to text shapes; slides without text will silently no-op |
| `add_bullet_point` | target slide in txt list | needs an existing text frame to append bullet into |
| `set_picture_size` | target slide in pics, h_cm > 3 | h_cm ≤ 3 = decorative icon, not a real content picture |
| `move_object` | target slide in pics or txt | needs a movable object |
| `insert_table` | target slide NOT in tbls | inserting into a slide that already has a table produces duplicate tables |
| `edit_table_cell` | target slide in tbls, row index < actual rows | out-of-range row → rc=42 |
| `reorder_slides` | n_slides >= 2 | single-slide deck has nothing to reorder |
| `set_background_color` | none | slide background op; no shape needed |
| `set_slide_transition` | none | slide property op; no shape needed |
| `add_speaker_note` | none | `notes_text_frame` exists on every slide regardless of shapes |

### Part C: Distribution Match

**Invariant — pool-level, not sampling-level.** Distribution match is a constraint on the TYPE_2 candidate pool design, not on the sampled output. `max_type1` and `max_type2` are runtime parameters that control how many rows are drawn from each task's candidate pool — varying them must never cause distribution match to fail. This means the TYPE_2 candidate pool (the ops assigned to each task before any sampling occurs) must be designed so that, in aggregate across all tasks, op type frequencies already match the eval distribution. For TYPE_1-eligible tasks where the T1 op is excluded from the T2 pool to avoid duplicating the eval op, that exclusion must be compensated by the pool design of other tasks.

Three checks: **(1)** slide indices used in perturb are spread across the deck, not clustered on early slides; **(2)** per-op parameter values (colors, fonts, alignments, transitions, table dims, etc.) appear in proportions similar to eval — no single value dominates disproportionately; **(3)** numeric ranges for `set_font_size` and `set_picture_size` cover the upper end seen in eval (eval has 44pt/60pt font sizes and 20cm/25cm picture heights — perturb must not be limited to small values).

Automated script. Extracts parameter values from perturb oracle code and eval instruction text, then compares distributions. Run after generating `train.perturb.jsonl`.

```python
"""V4b distribution match — libreoffice_impress.
Run from repo root: uv run python this_script.py
Requires: train.perturb.jsonl generated, eval.jsonl present.
"""
import json, re
from collections import Counter, defaultdict

# ── load ──────────────────────────────────────────────────────────────────────
all_eval    = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
eval_rows   = [r for r in all_eval    if "impress" in r["task_id"]]
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
perturb_rows = [r for r in all_perturb if "impress" in r["task_id"]]

# ── oracle code for perturb ───────────────────────────────────────────────────
def _oracle(r):
    parts = []
    for a in (r.get("metadata", {}).get("others") or {}).get("oracle_actions", []):
        cmd = a.get("parameters", {}).get("command", "")
        if isinstance(cmd, str):
            parts.append(cmd)
    return "\n".join(parts)

# ── RGB → color name ──────────────────────────────────────────────────────────
_RGB_MAP = {
    (255,0,0):"red",(255,80,0):"red",(220,0,0):"red",
    (0,0,255):"blue",(0,0,200):"blue",
    (0,128,0):"green",(0,255,0):"green",(0,200,0):"green",
    (255,255,0):"yellow",(255,240,0):"yellow",
    (128,0,128):"purple",(127,0,127):"purple",
    (255,165,0):"orange",(255,128,0):"orange",(255,192,0):"orange",
    (0,255,255):"cyan",(0,200,200):"cyan",
    (255,255,255):"white",
    (0,0,0):"black",
    (128,128,128):"gray",(192,192,192):"gray",
    (255,192,203):"pink",(255,180,190):"pink",
}
def _rgb_name(r, g, b):
    key = (int(r), int(g), int(b))
    if key in _RGB_MAP:
        return _RGB_MAP[key]
    # nearest by L1
    nearest = min(_RGB_MAP, key=lambda c: abs(c[0]-key[0])+abs(c[1]-key[1])+abs(c[2]-key[2]))
    return _RGB_MAP[nearest]

# ── perturb parameter extractors (from oracle Python code) ───────────────────
def _p_slides(r):
    return [int(i) for i in re.findall(r'prs\.slides\[(\d+)\]', _oracle(r))]

def _p_font_colors(r):
    return [_rgb_name(*m) for m in re.findall(
        r'font\.color\.rgb\s*=\s*RGBColor\((\d+),\s*(\d+),\s*(\d+)\)', _oracle(r))]

def _p_bg_colors(r):
    return [_rgb_name(*m) for m in re.findall(
        r'fore_color\.rgb\s*=\s*RGBColor\((\d+),\s*(\d+),\s*(\d+)\)', _oracle(r))]

def _p_font_names(r):
    return re.findall(r"font\.name\s*=\s*'([^']+)'", _oracle(r))

def _p_alignments(r):
    return re.findall(r'PP_ALIGN\.(\w+)', _oracle(r))

def _p_pic_heights_cm(r):
    return [int(n) for n in re.findall(r'int\((\d+)\*360000\)', _oracle(r))]

def _p_transitions(r):
    # Matches both _etree.fromstring('<p:fade/>') and qn('p:fade') PER_TASK_SPECS formats
    return [a or b for a, b in re.findall(r"<p:(\w+)/>|qn\('p:(\w+)'\)", _oracle(r))]

def _p_table_dims(r):
    return [f"{a}×{b}" for a, b in re.findall(r'add_table\((\d+),\s*(\d+),', _oracle(r))]

def _p_table_rows(r):
    return re.findall(r"\['(first|second|last)'\]", _oracle(r))

def _p_reorder_spans(r):
    src_list = [int(i) for i in re.findall(r'_e\s*=\s*_lst\[(\d+)\]', _oracle(r))]
    dst_list = [int(i) for i in re.findall(r'_lst\.insert\((\d+),', _oracle(r))]
    return [abs(s - d) for s, d in zip(src_list, dst_list)]

def _p_font_styles(r):
    styles = re.findall(r'font\.(bold|italic|underline)\s*=\s*True', _oracle(r))
    if re.search(r"sngStrike", _oracle(r)):
        styles.append("strikethrough")
    return styles

def _p_font_sizes_pt(r):
    # From instruction text (oracle uses lxml for font size)
    return [int(n) for n in re.findall(r'(\d+)\s*pt\b', r["instruction"], re.I)]

def _p_move_positions(r):
    return re.findall(r'\bto\s+(?:the\s+)?(top|bottom|left|right|center)\b',
                      r["instruction"], re.I)

def _p_title_texts(r):
    # Both oracle and instruction
    from_oracle = re.findall(r"\.text\s*=\s*'([^']+)'", _oracle(r))
    from_instr  = re.findall(r'"([^"]{2,30})"', r["instruction"])
    return from_oracle or from_instr

# ── eval parameter extractors (from instruction text) ────────────────────────
_COLOR_RE   = re.compile(r'\b(red|blue|green|yellow|purple|orange|cyan|white|black|pink|gray|grey)\b', re.I)
_FONT_RE    = re.compile(r'\b(Arial|Times New Roman|Calibri|Georgia|Verdana|Courier New|Trebuchet MS)\b', re.I)
_ALIGN_RE   = re.compile(r'\b(left|right|center|justify|justified)\b', re.I)
_SIZE_RE    = re.compile(r'(\d+)\s*pt\b', re.I)
_HEIGHT_RE  = re.compile(r'(\d+)\s*cm\b', re.I)
_TRANS_RE   = re.compile(r'\b(dissolve|fade|wipe|push|uncover)\b', re.I)
_TABLE_RE   = re.compile(r'(\d+)\s*rows?\D+?(\d+)\s*col', re.I)
_TROW_RE    = re.compile(r'\b(first|second|last)\s+row\b', re.I)
_SLIDE_RE   = re.compile(r'\bslide\s+(\d+)\b', re.I)
_STYLE_RE   = re.compile(r'\b(bold|italic|underline|strikethrough)\b', re.I)
_POS_RE     = re.compile(r'\bto\s+(?:the\s+)?(top|bottom|left|right|center)\b', re.I)
_REORDER_RE = re.compile(r'slide\s+(\d+).*?(?:becomes|to)\s+slide\s+(\d+)|last.*?becomes.*?first', re.I)

def _e_slides(r):
    return [int(n)-1 for n in _SLIDE_RE.findall(r["instruction"])]  # convert to 0-based

def _e_reorder_spans(r):
    spans = []
    for m in _REORDER_RE.finditer(r["instruction"]):
        if m.group(1) and m.group(2):
            spans.append(abs(int(m.group(1)) - int(m.group(2))))
        else:
            spans.append(99)  # "last → first" in large deck; use sentinel
    return spans

# ── print comparison ──────────────────────────────────────────────────────────
def _compare(label, e_ctr, p_ctr, warn_ratio=3.0, min_e_share=0.02):
    e_tot = sum(e_ctr.values()) or 1
    p_tot = sum(p_ctr.values()) or 1
    all_keys = sorted(set(e_ctr) | set(p_ctr),
                      key=lambda k: -(e_ctr.get(k,0)/e_tot + p_ctr.get(k,0)/p_tot))
    print(f"\n  {label}:")
    print(f"    {'value':<22}  {'eval':>6}  {'perturb':>8}  {'p/e':>5}")
    for k in all_keys:
        e = e_ctr.get(k, 0) / e_tot
        p = p_ctr.get(k, 0) / p_tot
        ratio = p / e if e > 0 else float("inf")
        flag = "  ← HIGH" if ratio > warn_ratio else \
               ("  ← LOW"  if e >= min_e_share and ratio < 1/warn_ratio else "")
        print(f"    {str(k):<22}  {e:>5.1%}  {p:>7.1%}  {ratio:>4.1f}x{flag}")
    if not all_keys:
        print("    (no data)")

print("=" * 65)
print("V4b — Perturb-eval match verification (libreoffice_impress)")
print("=" * 65)

# B1: Slide index distribution ------------------------------------------------
e_slide_ctr = Counter(s for r in eval_rows    for s in _e_slides(r))
p_slide_ctr = Counter(s for r in perturb_rows for s in _p_slides(r))

# Bucket into ranges for readability
def _bucket(ctr, buckets=[(0,0),(1,4),(5,9),(10,19),(20,99)]):
    bc = Counter()
    for k, v in ctr.items():
        for lo, hi in buckets:
            if lo <= k <= hi:
                bc[f"{lo}" if lo==hi else f"{lo}–{hi}"] += v
    return bc

print()
print("C1: Slide index distribution (0-based buckets)")
_compare("slides targeted", _bucket(e_slide_ctr), _bucket(p_slide_ctr))

# B2: Per-op parameter distributions -----------------------------------------
print()
print("C2: Per-op parameter distributions")

_compare("set_font_color  — color",
    Counter(c for r in eval_rows    for c in _COLOR_RE.findall(r["instruction"])),
    Counter(c for r in perturb_rows for c in _p_font_colors(r)))

_compare("set_background_color — color",
    Counter(c for r in eval_rows    for c in _COLOR_RE.findall(r["instruction"])),
    Counter(c for r in perturb_rows for c in _p_bg_colors(r)))

_compare("set_font_name   — font",
    Counter(f for r in eval_rows    for f in _FONT_RE.findall(r["instruction"])),
    Counter(f for r in perturb_rows for f in _p_font_names(r)))

_compare("set_text_alignment — alignment",
    Counter(a.lower() for r in eval_rows    for a in _ALIGN_RE.findall(r["instruction"])),
    Counter(a.lower() for r in perturb_rows for a in _p_alignments(r)))

_compare("set_font_style  — style",
    Counter(s.lower() for r in eval_rows    for s in _STYLE_RE.findall(r["instruction"])),
    Counter(s.lower() for r in perturb_rows for s in _p_font_styles(r)))

_compare("set_slide_transition — type",
    Counter(t.lower() for r in eval_rows    for t in _TRANS_RE.findall(r["instruction"])),
    Counter(t.lower() for r in perturb_rows for t in _p_transitions(r)))

_compare("move_object     — position",
    Counter(p.lower() for r in eval_rows    for p in _POS_RE.findall(r["instruction"])),
    Counter(p.lower() for r in perturb_rows for p in _p_move_positions(r)))

_compare("insert_table    — rows×cols",
    Counter(f"{a}×{b}" for r in eval_rows    for a,b in _TABLE_RE.findall(r["instruction"])),
    Counter(d          for r in perturb_rows for d in _p_table_dims(r)))

_compare("edit_table_cell — row",
    Counter(x.lower() for r in eval_rows    for x in _TROW_RE.findall(r["instruction"])),
    Counter(x.lower() for r in perturb_rows for x in _p_table_rows(r)))

# B3: Numeric range / difficulty checks ---------------------------------------
print()
print("C3: Numeric difficulty distribution")

e_sizes = [int(n) for r in eval_rows    for n in _SIZE_RE.findall(r["instruction"])]
p_sizes = [int(n) for r in perturb_rows for n in _p_font_sizes_pt(r)]
size_buckets = [(0,14),(15,20),(21,32),(33,99)]
def _bucket_num(vals, buckets):
    bc = Counter()
    for v in vals:
        for lo, hi in buckets:
            if lo <= v <= hi:
                bc[f"{lo}–{hi}pt"] += 1
    return bc
_compare("set_font_size   — pt range",
    _bucket_num(e_sizes, size_buckets),
    _bucket_num(p_sizes, size_buckets))
if e_sizes: print(f"    eval range: {min(e_sizes)}–{max(e_sizes)} pt")
if p_sizes: print(f"    perturb range: {min(p_sizes)}–{max(p_sizes)} pt")

e_heights = [int(n) for r in eval_rows    for n in _HEIGHT_RE.findall(r["instruction"])]
p_heights = [h for r in perturb_rows for h in _p_pic_heights_cm(r)]
h_buckets = [(0,9),(10,14),(15,19),(20,99)]
_compare("set_picture_size — height range",
    _bucket_num(e_heights, [(lo,hi) for lo,hi in h_buckets]),
    _bucket_num(p_heights, h_buckets))
if e_heights: print(f"    eval range: {min(e_heights)}–{max(e_heights)} cm")
if p_heights: print(f"    perturb range: {min(p_heights)}–{max(p_heights)} cm")

```

**Targets for C1–C3:**
- C1: slides 0–4 share ≤ 60% (perturb must spread ops across the full deck, not cluster on early slides)
- C2: for values with eval share ≥ 2%, perturb share must be between 1/3× and 3×
- C3: `set_font_size` must cover > 32pt (eval has 44pt, 60pt); `set_picture_size` must cover ≥ 15cm (eval has 20cm, 25cm)

---

## V4c Eval Leakage Check

Verify that no perturb row is functionally identical to its source eval task.

### Guarantees by design

- **TYPE_1 per-parameter exclusion**: each `t1_fn` passes the eval's known parameter value as `exclude` to the resampling helper (`_rc`, `_rs`, `_rn`, etc.). This ensures every generated `(op_type, param_value)` slot differs from its eval counterpart — the combined ops list can never be identical to eval. Critical rule: **every `t1_fn` that calls `_rc`/`_rb`/`_rn`/`_rs` must pass the eval's value as `exclude`**. Adding a new `t1_fn` or changing eval parameters requires updating the exclude call.
- **TYPE_2 structural disjointness**: op types in each TYPE_2 variant are chosen to be different from the source eval task's op types. The `_t2_variants` dict encodes this explicitly.

### V4c automated check

Eval rows use wget-based oracles (no extractable Python code), so leakage is checked against the perturb row's oracle and the eval instruction text.

```python
import json, re

all_eval    = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
impress_perturb = [r for r in all_perturb if "libreoffice_impress" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r.get("metadata", {}).get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

# Extract key parameter values from perturb oracle code
_PARAM_EXTRACTORS = [
    ("rgb",        re.compile(r'RGBColor\((\d+,\s*\d+,\s*\d+)\)')),
    ("font_name",  re.compile(r"font\.name\s*=\s*'([^']+)'")),
    ("font_size",  re.compile(r'Pt\((\d+)\)')),
    ("align",      re.compile(r'PP_ALIGN\.(\w+)')),
    ("transition", re.compile(r"<p:(\w+)/>|qn\('p:(\w+)'\)")),
    ("table_dim",  re.compile(r'add_table\((\d+),\s*(\d+),')),
    ("slide_idx",  re.compile(r'prs\.slides\[(\d+)\]')),
    ("note_text",  re.compile(r'notes_text_frame\.text\s*=\s*["\']([^"\']+)["\']')),
    ("title_text", re.compile(r'_tsh\.text\s*=\s*["\']([^"\']+)["\']')),
]

def _extract_params(oracle_text):
    result = {}
    for key, pat in _PARAM_EXTRACTORS:
        result[key] = set(m if isinstance(m, str) else ",".join(x for x in m if x)
                          for m in pat.findall(oracle_text))
    return result

def _eval_instr_params(instr):
    """Extract key param values from eval instruction text for comparison."""
    colors = set(re.findall(r'\b(red|blue|green|yellow|purple|orange|cyan|white|black|pink|gray)\b', instr, re.I))
    fonts  = set(re.findall(r'\b(Arial|Calibri|Georgia|Verdana|Courier New|Trebuchet MS|Times New Roman)\b', instr, re.I))
    sizes  = set(re.findall(r'(\d+)\s*pt\b', instr, re.I))
    notes  = set(re.findall(r'"([^"]+)"', instr))
    return {"colors": colors, "fonts": fonts, "sizes": sizes, "notes": notes}

_OP_PAT = re.compile(
    r'font\.color\.rgb|font\.size.*Pt\(|font\.(bold|italic|underline)|font\.name|'
    r'fore_color\.rgb|_tsh\.text=|PP_ALIGN\.|int\(\d+\*360000\)|_sh\.(top|left)\s*=|'
    r'notes_text_frame\.text|<p:[a-z]+|add_table\(|_sh\.table\.rows|_lst\.insert\('
)
_OP_TYPE_PATTERNS = [
    ("set_font_color",       re.compile(r'font\.color\.rgb\s*=')),
    ("set_font_size",        re.compile(r'font\.size\s*=\s*Pt\(')),
    ("set_font_style",       re.compile(r'font\.(bold|italic|underline)\s*=\s*True|sngStrike')),
    ("set_font_name",        re.compile(r"font\.name\s*=")),
    ("set_background_color", re.compile(r'fore_color\.rgb\s*=')),
    ("set_title_text",       re.compile(r'_tsh\.text=')),
    ("set_text_alignment",   re.compile(r'PP_ALIGN\.')),
    ("set_picture_size",     re.compile(r'int\(\d+\*360000\)')),
    ("move_object",          re.compile(r'_sh\.(top|left)\s*=')),
    ("add_speaker_note",     re.compile(r'notes_slide\.notes_text_frame\.text\s*=')),
    ("set_slide_transition", re.compile(r"<p:[a-z]+|qn\(.p:[a-z]+")),
    ("insert_table",         re.compile(r'add_table\(')),
    ("edit_table_cell",      re.compile(r'_sh\.table\.rows')),
    ("reorder_slides",       re.compile(r'_lst\.insert\(')),
]
def _op_types(oracle_text):
    return frozenset(name for name, pat in _OP_TYPE_PATTERNS if pat.search(oracle_text))

leakage = []
for r in impress_perturb:
    base_tid = re.sub(r"^perturb_(.+)_[0-9a-f]{8}$", r"\1", r["task_id"])
    eval_row = all_eval.get(base_tid)
    if eval_row is None:
        continue

    oracle = _oracle(r)
    perturb_params = _extract_params(oracle)
    eval_params    = _eval_instr_params(eval_row["instruction"])

    perturb_type = r.get("metadata", {}).get("perturb_type", "")
    if perturb_type == "type1":
        # TYPE_1: at least one extracted param value must differ from eval.
        # Check colors (RGBs), font names, font sizes, note text, title text.
        all_perturb_colors = {
            {"red": "255,0,0", "blue": "0,0,255", "green": "0,128,0",
             "yellow": "255,255,0", "purple": "128,0,128", "orange": "255,165,0",
             "cyan": "0,255,255", "white": "255,255,255", "black": "0,0,0",
             "pink": "255,192,203"}.get(c.lower(), c)
            for c in eval_params["colors"]
        }
        rgb_vals = perturb_params["rgb"]
        # If eval mentions colors, at least one RGB must not match any eval color
        if eval_params["colors"] and rgb_vals:
            overlap = {rgb.replace(" ", "") for rgb in rgb_vals} & all_perturb_colors
            if len(overlap) == len(rgb_vals):
                leakage.append((r["task_id"], f"TYPE_1 all RGBs match eval colors: {rgb_vals}"))
    elif perturb_type == "type2":
        # TYPE_2: op types must not ALL overlap with eval op types
        perturb_op_types = _op_types(oracle)
        eval_op_types    = _op_types(_oracle(eval_row))  # eval uses wget, may be empty
        if eval_op_types and perturb_op_types and perturb_op_types <= eval_op_types:
            leakage.append((r["task_id"], f"TYPE_2 op types subset of eval: {perturb_op_types}"))

print(f"[{'FAIL' if leakage else 'OK  '}] eval leakage: {len(leakage)} violations")
for tid, reason in leakage[:10]:
    print(f"  {tid}: {reason}")
```

> The TYPE_1 check above is heuristic (extracts params from oracle code, compares to eval instruction). The primary guarantee is structural: every `t1_fn` excludes its eval value via `exclude=` in `_rc`/`_rb`/`_rn`/`_rs`. This check is a secondary sanity gate, not the sole protection.

---

## V4d Inter-Variant Uniqueness

Within the N variants generated from the same source eval task, every row must be distinct — identical variants waste training budget and could bias the model toward a single solution. Check both instruction text and oracle code.

```python
import json, re
from collections import defaultdict

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
impress_perturb = [r for r in all_perturb if "libreoffice_impress" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r.get("metadata", {}).get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

# Group by source task (strip trailing 8-hex hash)
by_source = defaultdict(list)
for r in impress_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src].append(r)

instr_dups = []
oracle_dups = []
for src, rows in by_source.items():
    instrs  = [r["instruction"] for r in rows]
    oracles = [_oracle(r) for r in rows]
    if len(instrs) != len(set(instrs)):
        instr_dups.append((src, instrs))
    if len(oracles) != len(set(oracles)):
        oracle_dups.append((src, oracles))

print(f"[{'FAIL' if instr_dups  else 'OK  '}] duplicate instructions within source: {len(instr_dups)} sources")
print(f"[{'FAIL' if oracle_dups else 'OK  '}] duplicate oracle code within source:  {len(oracle_dups)} sources")
for src, rows in instr_dups[:3]:
    print(f"  {src}:")
    for i in rows: print(f"    {i!r}")
```

---

## V4e Instruction-Oracle Value Consistency

The instruction describes what the oracle will do — every key value named in the instruction must appear in the oracle code. Mismatches indicate a bug in the builder (e.g., `_op_instr` picked "red" but oracle emitted blue; instruction says "slide 3" but oracle acts on `prs.slides[4]`).

Checks per row:
- **Slide index**: every "slide N" in instruction → `prs.slides[N-1]` in oracle
- **Color name**: every color word in instruction → at least one `RGBColor(...)` consistent with that color in oracle
- **Font name**: every font name in instruction → `font.name = '...'` matching string in oracle
- **Font size**: every "N pt" in instruction → `Pt(N)` in oracle (or via lxml for SIZE ops)
- **Transition**: every transition word in instruction → `<p:word` or `qn('p:word')` in oracle
- **Table dimensions**: "N row × M col" in instruction → `add_table(N, M,` in oracle
- **Quoted text** (notes, titles): quoted strings in instruction → assigned in oracle

```python
import json, re
from collections import defaultdict

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
impress_perturb = [r for r in all_perturb if "libreoffice_impress" in r["task_id"]]

_RGB_BY_NAME = {
    "red":    {(255,0,0),(220,0,0),(255,80,0)},
    "blue":   {(0,0,255),(0,0,200)},
    "green":  {(0,128,0),(0,255,0),(0,200,0)},
    "yellow": {(255,255,0),(255,240,0)},
    "purple": {(128,0,128),(127,0,127)},
    "orange": {(255,165,0),(255,128,0),(255,192,0)},
    "cyan":   {(0,255,255),(0,200,200)},
    "white":  {(255,255,255)},
    "black":  {(0,0,0)},
    "pink":   {(255,192,203),(255,180,190)},
    "gray":   {(128,128,128),(192,192,192)},
    "grey":   {(128,128,128),(192,192,192)},
}

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r.get("metadata", {}).get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

def _check(r):
    instr  = r["instruction"]
    oracle = _oracle(r)
    if not oracle.strip():
        return []  # per_task rows have no Python oracle; skip
    errors = []

    # Slide indices: "slide N" (1-based) → prs.slides[N-1] or _lst[N-1] or _lst.insert(N-1,
    # Note: reorder destination "slide N" maps to _lst.insert(N-1, ...) not prs.slides[N-1]
    for n in map(int, re.findall(r'\bslide\s+(\d+)\b', instr, re.I)):
        idx = n - 1
        if (f"slides[{idx}]" not in oracle and
                f"_lst[{idx}]" not in oracle and
                f"_lst.insert({idx}," not in oracle and
                f"_lst.insert({idx} ," not in oracle):
            errors.append(f"slide {n} → prs.slides[{idx}] not in oracle")

    # Colors: each color word → matching RGBColor in oracle
    oracle_rgbs = {
        tuple(map(int, m.split(",")))
        for m in re.findall(r'RGBColor\(\s*(\d+\s*,\s*\d+\s*,\s*\d+)\s*\)', oracle)
    }
    for color in re.findall(r'\b(red|blue|green|yellow|purple|orange|cyan|white|black|pink|gray|grey)\b', instr, re.I):
        expected = _RGB_BY_NAME.get(color.lower(), set())
        if expected and not (oracle_rgbs & expected):
            errors.append(f"color '{color}' not found as RGBColor in oracle (oracle has {oracle_rgbs})")

    # Font names: "Calibri" / "Arial" etc. → font.name = 'Calibri'
    for font in re.findall(r'\b(Arial|Calibri|Georgia|Verdana|Courier New|Trebuchet MS|Times New Roman)\b', instr, re.I):
        if f"font.name = '{font}'" not in oracle and f'font.name = "{font}"' not in oracle:
            errors.append(f"font '{font}' not assigned in oracle")

    # Font sizes: "N pt" → Pt(N) in oracle or emu value (size in lxml uses 100*pt EMU units)
    for pt in map(int, re.findall(r'(\d+)\s*pt\b', instr, re.I)):
        emu = pt * 100  # pptx lxml stores size as pt*100 in hundredths of a point
        if f"Pt({pt})" not in oracle and str(emu) not in oracle:
            errors.append(f"font size {pt}pt not in oracle (expected Pt({pt}) or emu {emu})")

    # Transitions: "fade"/"wipe"/etc. → <p:fade or qn('p:fade')
    for tr in re.findall(r'\b(dissolve|fade|wipe|push|uncover)\b', instr, re.I):
        tr_l = tr.lower()
        if f"<p:{tr_l}" not in oracle and f"'p:{tr_l}'" not in oracle:
            errors.append(f"transition '{tr}' not in oracle")

    # Table dimensions: "N row(s)" + "M col(s)" → add_table(N, M,
    row_m = re.search(r'(\d+)\s*rows?', instr, re.I)
    col_m = re.search(r'(\d+)\s*col', instr, re.I)
    if row_m and col_m:
        r_, c_ = row_m.group(1), col_m.group(1)
        if f"add_table({r_}, {c_}," not in oracle and f"add_table({r_},{c_}," not in oracle:
            errors.append(f"table {r_}×{c_} not in oracle")

    # Quoted text (notes, titles): 'text' in instruction → assigned in oracle
    for quoted in re.findall(r'"([^"]{2,60})"', instr):
        if quoted not in oracle:
            errors.append(f"quoted text {quoted!r} not found in oracle")

    return errors

failures = [(r["task_id"], errs) for r in impress_perturb if (errs := _check(r))]
print(f"[{'FAIL' if failures else 'OK  '}] instruction-oracle consistency: {len(failures)} rows with mismatches")
for tid, errs in failures[:10]:
    print(f"  {tid}:")
    for e in errs:
        print(f"    {e}")
```

Known acceptable exceptions to annotate (add `# V4e-skip: <reason>` beside them in this section):
- `set_font_size` oracle uses lxml XML bytes (not `Pt()`), so pt-size check may produce false positives — verified manually that lxml path stores correct emu value.
- `move_object` and `reorder_slides` don't directly name numeric values in instruction in a checkable pattern.
- PER_TASK_SPECS rows (`per_task` in task_id) have no Python oracle; skipped automatically.

---

## Expected Output

- 34 perturbable tasks × (max_type1=1 + max_type2=2) ≈ 95–98 rows (some TYPE_1 return [])
- 8 TYPE_3 tasks × atomic eval-evaluator variants = **+11 rows** (P3-4-impress)
- D2 cleanup: 13 transition rows (8 compound + 5 single) → 2 single rows
- **Phase 3 total target: ~109 rows** (was 98)
- All 14 op types covered (TYPE_1+TYPE_2)
- All 8 impress-domain evaluators covered (TYPE_3)
- All `(op, slide, param)` combinations differ from eval
- V2 pass-rate target: **100%**

---

## Cycle 35a updates — `_t2_variants` skill-mix trim

3 trim cycles on `_t2_variants` to bring perturb's per-skill ratios within 0.82× – 1.60× of the eval feasibility-checked baseline. Each cycle re-measured per-skill share of TYPE_2 emissions, then adjusted the per-base TYPE_2 op pools (which slides expose which ops, and the `_T2_POOLS` skill weights) so over-represented skills shrank and under-represented skills grew, **without** introducing infeasible op-on-slide combinations.

Final ratios vs eval feasibility-checked baseline:

| skill | before | after | notes |
|---|---|---|---|
| italic | 6.4% | 1.9% NEW (then 3.7% after user feedback bump) | over-trimmed at first, restored to ~target |
| speaker_note | 11.0% | 2.8% NEW (then 6.5% after feedback) | similar restore |
| alignment | 11.9% | 2.8% | within 0.82× |
| background | 22.9% | 13.9% | high-skill cap |
| picture | 11.0% | 15.7% | bumped (was under) |
| underline | 2.8% | 7.4% | bumped (was under) |
| bold | 3.7% | 9.3% | bumped (was under) |

All 7 within **0.82×–1.60×** of eval. **GROUP-aware feasibility verified**: font ops (`set_font_*`) only emit on slides whose top-level shape is a non-group text frame — slides whose text lives inside a GROUP shape fail python-pptx font assignments (no recursive descent), so those slide indices are masked out of the per-base font op pool at spec design time. 0 infeasible perturb rows confirmed in the post-trim full-domain pass.

---

## Cross-domain status (cycle 35a+)

- **707 current total perturb rows / 0 infeasible rows** dataset-wide (architecture-level guard at `perturb/dispatch.py:70` — see [`os.md` cycle 35a section](/devs/envs/lite.osworld/perturb/os.md#cycle-35a-updates) — confirms that `apply_structural_perturbation` filters infeasible bases before emission).
