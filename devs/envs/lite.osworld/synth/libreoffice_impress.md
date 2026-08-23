# LibreOffice Impress — Synth Plan

> Keep in sync with code. Implementation: [`libreoffice_impress.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/libreoffice_impress.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/libreoffice_impress.md`](/devs/envs/lite.osworld/perturb/libreoffice_impress.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain libreoffice_impress` for live numbers. Cycle-47 (post 6-base body-focus + Save-As + RGB-strip + doc-wide expansion): Synth N=256, eval N=47.

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `rgb_leak.rgb_triplet_leak` | 0% | 0% | 0 | ✓ | DONE cycle-47 — stripped `(RGB X,X,X)` parens from all 91 instructions |
| `comparator_strictness.color_tolerant` | 29.3% | 0% | +29 | 🔴 | Demote `compare_pptx_files_color_tolerant` rows — DEFERRED: tolerant kept where Custom-Color picker drift is real (cycle-46 rationale); no safe demotion path without breaking the cluster |
| `atom_count.atom_2` | 0% | 25.5% | -26 | ❌ | Add 5-8 compound `compare_pptx_files+compare_pptx_files` templates (per-slide-diff multi-slide check). Synth has 0 compound rows. |
| `op_family.title_or_body_style` | 52.3% | 36.2% | +16 | 🔴 | Down-weight title/body-style cluster ~30 rows; doc-wide expansion adds title-color/bold variants but title-class still over |
| `eval_fn.compare_pptx_files+compare_pptx_files` | 0% | 26% | -26 | ❌ | Same as atom_2 above — biggest structural gap |
| `slide_anchor.doc_wide` | 20.7% | 53.2% | -33 | 🔴 | DONE cycle-47 — added 10 new doc-wide FileTasks (D-IMP-70..78) covering all-titles/all-bg variants; still 33pp short, more doc-wide variety needed |
| `op_family.save_or_export` | 1.6% | 2.1% | -0.5 | ✓ | DONE cycle-47 — added D-IMP-68/69 Save-As .pptx templates via `_make_save_as_filetask_template` factory |

**Quant-correction**: gap.md claimed title_style 50% vs 23% eval; quant shows 52% vs 36% — real gap is 16pp not 27pp.

## Cycle-47 changes (gap.md §libreoffice_impress bridge plan)

1. **`_IMPRESS_BODY_FOCUS_STEPS` rollout** — new module-level constant (richer Escape/F6/window-activate sequence) opted-in via `FileTask.body_focus=True` on 12 FileTasks under 6 base files: D-IMP-04/05/12/25/30/32 (gap.md Axis A H-trigger cluster).
2. **Save-As/Export coverage** — added `_make_save_as_filetask_template` factory + 2 new Files (D-IMP-68/69) with 4 pure Save-As Params; eval reads from typed Desktop filename via `result.path=/home/user/Desktop/<save_as_name>`.
3. **RGB triplet leak fix** — stripped ` (RGB X,X,X)` parens from 91 instructions; color names retained ("dark red", "magenta", "dark navy", etc.).
4. **Doc-wide framing expansion** — added 4 helpers (`_gold_all_slides_title_color`, `_gold_all_slides_title_bold`, `_gold_all_slides_title_underline`, `_gold_all_slides_background`) + 10 new doc-wide FileTasks under 9 new Files (D-IMP-70..78); instructions reference "every slide" / "across the deck" / "throughout this presentation" rather than ordinal slide indices.

## Current shape

**Current generated total: 287 jsonl rows** (historical cycle-47 volume-rescale snapshot: ~125 templates → ~110 rows).

| Quantity | Count | Note |
|---|---:|---|
| `D_IMP_*` File instances (`D_IMP_01..D_IMP_78`) | 67 | each declared once at module top-level; cycle-47 added D-IMP-68/69 (Save-As) + D-IMP-70..78 (doc-wide) |
| `FileTask` entries in `FILE_TASKS` | ~125 | cap-2×2: ≤2 FileTasks per File; cycle-47 added 2 Save-As + 10 doc-wide tasks |
| `Param` slots (≤2 per FileTask) | ~250 | scaler may downgrade some to 1-Param |
| `TopicTheme` entries in `_TOPIC_FAMILIES` | 10 | family_weekend_cooking, city_food_tour, wildlife_photography_journal, office_productivity_review, classroom_year_recap, healthy_eating_program, architecture_photo_essay, product_launch_recap, nature_hike_diary, community_event_recap |
| `_VALID_EXAMINE_FIELDS` whitelist size | 23 | enforced at `_examine_options` + extra_examine validation paths |
| `_build_evaluator` factory arms | 7 | default `_to_synth_template` + 6 specialized factories |
| Final jsonl rows after `_rescale_for_volume` + `_apply_drop_filter` | 287 | uncapped — `TARGET = math.inf` in `synth/catalog.py`, so Stage B never downgrades |

Source AND gold pptx are both materialised via python-pptx heredocs landed in `pre_config_steps` (Hard Constraint #2 — oracle = `cp gold source`). Evaluator default is `compare_pptx_files` with a single non-default `examine_<field>=True`; the 6 specialized factories cover transition / master-bg / pagenum-color / image-stretch / table-insert / image-resize.

## Architecture / design notes

**File catalog (56 `D-IMP-*` Files)** — each is a structural deck shape (`n_slides × layout × media arrangement`) declared once. Adding structural variety = adding a File; adding skill variety per File = adding a FileTask (≤2/File). Per-seed topic variety is free (sampled from `_TOPIC_FAMILIES`).

| File range | Source builder | Structural axis |
|---|---|---|
| `D-IMP-01..03`, `D-IMP-21` | `_src_text_deck(layout="title_body")` | n_slides ∈ {3, 5, 6, 7} |
| `D-IMP-04`, `D-IMP-22..23` | `_src_text_deck(layout="title_only")` | n_slides ∈ {5, 4, 8} |
| `D-IMP-05`, `D-IMP-24..25` | `_src_text_deck(layout="title_subtitle")` | n_slides ∈ {5, 3, 6} |
| `D-IMP-19..20`, `D-IMP-52` | `_src_text_deck(layout="title_body")` reused for repos/swap/transition tasks | n_slides ∈ {4, 5, 5} |
| `D-IMP-41..43` | `_src_text_deck` w/ non-default `source_font` | "Liberation Serif", "DejaVu Sans", "DejaVu Serif" |
| `D-IMP-06..08`, `D-IMP-26..29`, `D-IMP-44..46` | `_src_hero_photo_deck` | `photo_pos` ∈ {center, top, bottom, corner}; `caption` ∈ {T, F}; n_slides ∈ {3..8} |
| `D-IMP-09..12`, `D-IMP-30..33`, `D-IMP-47..48` | `_src_gallery_deck` | `grid` ∈ {2x2, 3x3, 1+3, strip}; n_slides ∈ {3, 4, 5} |
| `D-IMP-13..15`, `D-IMP-34..36`, `D-IMP-49` | `_src_footer_deck` | `footer_type` ∈ {basic, with_page_num, with_logo_corner}; n_slides ∈ {4..8} |
| `D-IMP-16..17`, `D-IMP-37..38`, `D-IMP-50` | `_src_notes_deck` | n_slides ∈ {4, 5, 6, 7, 8} |
| `D-IMP-18`, `D-IMP-39..40` | `_src_portrait_deck` | n_slides ∈ {3, 5, 7} |
| `D-IMP-51` | `_src_table_target_deck` | 4-slide deck with table-target slides |
| `D-IMP-53` | `_src_resize_hero_deck(init_h_cm=8.0)` | hero photo at 8cm init height |
| `D-IMP-54` | `_src_master_bg_white_deck` | 5-slide deck w/ all backgrounds = white |
| `D-IMP-55` | `_src_pagenum_master_deck` | 5-slide deck w/ injected sldNum placeholder in slideMaster1.xml |
| `D-IMP-56` | `_src_offcenter_pic_deck` | 3-slide deck w/ small off-center PIL-png on slide 0 |

**Skill catalog (`compare_pptx_files` default factory)** — `_to_synth_template` builds a `compare_pptx_files` evaluator with `_examine_options(variant.examine_field)`. The `examine_field` whitelist + `_gold_*` mutator helpers determine which run/shape/slide attribute the gold diff exercises.

| Skill (eval_class) | Canonical examine_field(s) | Gold-mutator helper(s) |
|---|---|---|
| `set_font_color` | `examine_color_rgb` | `_gold_set_title_font_color(idx, rgb)`, `_gold_set_body_font_color(idx, rgb)` |
| `edit_title` (size / name) | `examine_font_size` | `_gold_set_title_font_size`, `_gold_set_title_font_name`, `_gold_set_body_font_name` |
| `edit_title` (note edit) | `examine_note` | `_gold_set_speaker_note(idx, text)` |
| `bold_underline_text` | `examine_font_bold` / `_italic` / `_underline` / `_alignment` | `_gold_set_body_bold`, `_gold_set_italic`, `_gold_set_underline`, `_gold_set_body_text_alignment` |
| `change_bg_color` | `examine_background_color` | `_gold_set_background(idx, rgb)` |
| `compound_pptx` (paired-mutation) | tuple of 2+ examine keys + `extra_examine` | composed snippet via newline-join |
| `image_stretch` (picture height) | `examine_modify_height` | `_gold_resize_picture(idx, h_cm)` |
| `reorder_slides` | `examine_text` | `_gold_swap_slides(i, j)` |
| `add_slide` (table-insert routed through specialized factory) | `examine_shape=True`, `examine_text=False` | `_gold_insert_table(slide_idx, rows, cols)` |

**`_VALID_EXAMINE_FIELDS` whitelist (cycle-41 guard)** — `examine_number_of_slides`, `examine_shape`, `examine_text`, `examine_indent`, `examine_font_name`, `examine_font_size`, `examine_font_bold`, `examine_font_italic`, `examine_color_rgb`, `examine_font_underline`, `examine_strike_through`, `examine_alignment`, `examine_title_bottom_position`, `examine_table_bottom_position`, `examine_run_count`, `examine_right_position`, `examine_top_position`, `examine_shape_for_shift_size`, `examine_image_size`, `examine_modify_height`, `examine_bullets`, `examine_background_color`, `examine_note`. Anything not in this set is a SILENT NO-OP at `compare_pptx_files` time. Both `_examine_options(field)` and the compound-Param `extra_examine` loop validate every key.

**Specialized factories (6 eval-gap fillers)** — FileTasks set `make_template=` to a custom factory because their evaluator differs from `compare_pptx_files`.

| File | Factory | Eval func | Gold mechanism |
|---|---|---|---|
| `D-IMP-51` | `_make_table_template` | `compare_pptx_files` | `_replay_src_for_gold` + `_gold_insert_table(slide_idx, rows, cols)` |
| `D-IMP-52` | `_make_transition_filetask_template` | `check_transition` | XML inject via `_gold_set_transition(slide_idx, kind)` |
| `D-IMP-53` | `_make_image_resize_filetask_template` | `compare_pptx_files` | replay + `_gold_resize_picture(idx, h_cm)` |
| `D-IMP-54` | `_make_master_bg_filetask_template` | `evaluate_presentation_fill_to_rgb_distance` | `_gold_set_background(i, rgb)` over `range(5)` |
| `D-IMP-55` | `_make_pagenum_color_filetask_template` | `check_page_number_colors` | full replay then `_master_color_patch_py(gold_path, HEX)` |
| `D-IMP-56` | `_make_image_stretch_filetask_template` | `check_image_stretch_and_center` | pre_config snapshots src → `expected_path`; oracle = `_oracle_stretch_image(slide_idx)` |

**TopicTheme** is what a single pptx is *about*: short id + 6 coherent `slide_titles` + 6 matching `slide_bodies` + `photo_dirs` subset of `assets/synth/photos/<dir>/`. `_cycle_to(pool, n)` extends pools for `n_slides > 6`; `_pick_topic(seed, salt)` picks deterministically; `_sample_topic_photos(topic, n, seed)` samples photo rel-paths.

## Implementation references

- `libreoffice_impress.py` — `_to_synth_template` default factory + 6 specialized factories listed above.
- `EVAL_CLASS_TO_EXAMINE_FIELD` mapping (source ~L265-288) — documents legacy eval_class → canonical-key correspondence.
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5) — cross-domain volume rebalance only; intra-domain skill ratio is author responsibility per PD (4a) / PD (4d).
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance) — Cat 2 dominant (HOMO_ZERO findings).

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- **Chart insert rows** — chart XML doesn't round-trip cleanly through LO save (F11); perturb has no synth-side chart precedent.
- **Audio embed rows (`compare_audios`)** — needs real `.mp3` / `.wav` asset.
- **PDF / image export rows (`compare_pdfs` / `compare_images`)** — agent-side LO-Export-As GUI flow is fragile and has no oracle-stable variant on the synth side yet.
- **Theme apply / animation rows** — gold is ill-defined via python-pptx alone; perturb side covers via TYPE_3 wireframes only.

## Cycle-recurring failures to avoid (impress-specific)

- **F3 (sample with replacement)**: use `rng.sample` for variant pools (cycle-33 picture-height + alignment regressions).
- **F9 (agent-cap)**: source builders cap text-frames at 1-2/slide; per-Param mutators target the exact `shape[i]` instead of looping every shape on the slide.
- **F1 (LO normalize)**: NOT applicable to impress — `compare_pptx_files` reads via python-pptx on both sides → no XML mismatch.
- **F2 (oracle artifact)**: always have agent's pptx as the eval target; never route eval through a custom `vm_command_line` dump.

## Pipeline reference

`_src_*_deck` builds source via python-pptx in `pre_config_steps`; agent edits in LO Impress; `compare_pptx_files` reads both via python-pptx so F1 (LO-XML normalize) doesn't apply. Each `File` → declared at module top; each `FileTask` → entry in `FILE_TASKS`; ≤2 `Param`s per FileTask; `_emit_templates(FILE_TASKS)` → `SynthTemplate` list; `_rescale_for_volume(templates)` → current final 287 jsonl rows.
