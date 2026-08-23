# GIMP — Synth Plan

> Keep in sync with code. Implementation: [`gimp.py`](/lite/gym/envs/lite/osworld/src/gen/train/synth/gimp.py).
> Common workflow: [`AGENTS.md`](/devs/envs/lite.osworld/synth/AGENTS.md). Cross-reference: [`perturb/gimp.md`](/devs/envs/lite.osworld/perturb/gimp.md).

## Current quant-gap snapshot (`measure_gap.py` v2)

Run `uv run python devs/envs/lite.osworld/measure_gap.py --domain gimp` for live numbers. Synth N=100, eval N=16 (10 infeasibility filtered — unusually high; gimp has the most refusal/can't-do rows in eval, all kept out of synth per AGENTS.md).

| Dim | Synth | Eval | Δpp | Status | Bridge |
|---|---:|---:|---:|:-:|---|
| `instruction_leak.leak` | 86.0% | 0% | +86 | 🔴 | Strip Ctrl+L / "Filters → Levels" / "via the Hue-Saturation dialog" hints from 86 instructions |
| `skill_class.preferences` | 34.0% | 25.0% | +9 | ⚠️ | Down-weight prefs cluster ~5 rows; reinvest in transform / color templates |
| `skill_class.other:check_*` | — | several | ❌ | ❌ | 5 eval-only fns 0 synth: `check_image_mirror`, `check_green_background`, `check_textbox_on_leftside`, `check_triangle_position`, `check_structure_sim_resized` — add ≥1 template per |

**Quant-correction**: gap.md claimed prefs 8.5× over-weighted; post-infeasibility-filter the eval-side prefs share is 25%, so synth is only ~1.4× over. Real headline is `instruction_leak`.

## Current shape

**25 Files / 50 FileTasks / 73 current rows / 9 eval_class buckets.**

Two surfaces, both append into `TEMPLATES`:

1. **§I file-task surface (current generated count contributes to the 73-row domain total)** — file-as-topic. Each `File` materialises ONE source image on the Desktop (real photo via `_stage_asset` or PIL-built synthetic). Each `FileTask` carries up to `SYNTH_CAP_PARAMS_PER_TASK=2` `Param` variants; `_emit_templates` enforces `SYNTH_CAP_TASKS_PER_FILE=2`. Per-task `gold` callable applies a PIL transform off the source and writes the post-op gold; oracle is `cp gold out_path`. The asymmetric init→target file content is the trivial-pass guard.
2. **`config_status` surface (current generated count included above)** — gimprc / sessionrc rows. The GIMP-preference skill has no source image and the canvas is optional, so it does not fit the file-as-topic shape. Mirrors `osworld_gimp_{7767eef2, 7b7617bd, b148e375, d52d6308}` (theme / undo-levels / layer-new-name / hide-docks + tile-cache-size adjacent skill).

**Real-source coverage:** 19/25 Files are `setup_class="real_image"` (76% — well above the PD 3a ≥25% floor). Remaining 6 are `synth_image`: solid / gradient / checkerboard / alpha-disc / palette-indexed PNG / EXIF-tagged JPG — each a distinct ingestion-path probe.

| Bucket | FileTasks | Eval `func` materialised |
|---|---|---|
| `image_transform` | 15 | `check_structure_sim` (blur / sharpen / grayscale / posterize / invert / pixelize) |
| `check_brightness_decrease_and_structure_sim` | 7 | compound — expected = ORIGINAL source |
| `resize` | 6 | `check_image_size` (rule expected, dim-only) |
| `check_saturation_increase_and_structure_sim` | 6 | compound — expected = ORIGINAL source |
| `check_file_exists_and_structure_sim` | 5 | compound — expected = TRANSFORMED gold |
| `check_palette_and_structure_sim` | 4 | compound — expected = ORIGINAL source |
| `check_contrast_increase_and_structure_sim` | 4 | compound — expected = ORIGINAL source |
| `check_image_mirror` | 3 | compound — expected = ORIGINAL source |
| `check_config_status` | 12 (config surface) | gimprc / sessionrc s-expr regex |

The §I `_eval` factory selects `expected` per `_COMPOUND_EVAL_KINDS`: `image_mirror` / `brightness_decrease` / `contrast_increase` / `saturation_increase` / `palette` use `expected = ORIGINAL source` so the eval func actually measures the agent's transform delta (a cp-no-op fails because the result still equals the source). Only `file_exists` and the `structure_sim` baseline use `expected = TRANSFORMED gold`.

## Architecture / design notes

**Eval files/task ratio**: 0.85. GIMP eval revolves around image transformations + gimprc settings.

**Structural variation axes** covered by 25 Files: image format (PNG / JPG / palette PNG / EXIF-tagged JPG); content domain (wildlife / food / landscape / portrait / nature + 4 PIL-synthetic shapes); dimensions (256×256 PIL up to 4K real photos); channel mode (RGB / RGBA / palette-indexed); gimprc/sessionrc state (default vs custom theme / undo / hide-docks).

**Real-photo inventory**: `assets/synth/photos/...` shared with perturb's `_REAL_PHOTO_SOURCES`. Categories: wildlife (horse-meadow, tiger-closeup, animal-portrait, bird-perch), food (pizza-dish, coffee-latte, salad-bowl, restaurant-meal), landscape (forest-trail, beach-sunset, mars-rover-vista, jupiter-full-disk, desert-dunes, mountain-range, io-volcanic-eruption, earth-blue-marble-apollo17), portrait (person-headshot-1/2), nature (galaxy-andromeda). All staged via `_stage_asset(asset_rel, path)` host_push dispatch.

All §I rows use `synth_command=""` — GIMP is not in `DOMAIN_DEFAULT_OPEN`; the agent launches GIMP manually. Oracle is shell-only (`cp gold out`); eval opens both with PIL.

## Implementation references

- `gimp.py` §I — `File` / `Param` / `FileTask` dataclasses; `_eval` dispatches by `Param.eval_kind` (8 kinds → 8 §I eval_class buckets).
- `_make_config_status_row` writes `gimprc`/`sessionrc` s-expr via shell heredoc; postconfig is `_GIMP_QUIT_POSTCONFIG` (graceful Ctrl+Q quit); eval reads the config file via `gimp_config_file` getter + regex.
- PIL-built sources (`F_GIMP_5/6/7/8/24/25`): `_pil_src(body)` heredoc in `pre_config_steps`.
- Real-photo sources (`F_GIMP_1/2/3/4/9-23`): `_real_src(asset_rel)` → `_stage_asset(asset_rel, path)` host_push.
- Gold transforms: `_gold_blur` / `_gold_mirror` / `_gold_contrast` / `_gold_brightness` / `_gold_posterize` / `_gold_grayscale` / `_gold_sharpen` / `_gold_resize` / `_gold_crop_center` / `_gold_invert` / `_gold_pixelize` / `_gold_rotate_90` / `_gold_saturation` / `_gold_palette` / `_gold_file_exists_rename`.
- [AGENTS.md §Per-domain Cat 1 / Cat 2 allocation guidance](/devs/envs/lite.osworld/synth/AGENTS.md#per-domain-cat-1--cat-2-allocation-guidance).
- [AGENTS.md §Scaler architecture](/devs/envs/lite.osworld/synth/AGENTS.md#scaler-architecture-cycle-41--design-5).

## Bridge plan / outstanding work

The quant snapshot is the canonical bridge plan; items it does not cover:

- **xcf Script-Fu layer manipulation** — eval rows 7-18 (named-layer fill / resize / create) require GIMP's `gimp` CLI with Script-Fu console; not reachable from a python heredoc oracle. The `layer-new-name` config_status rows partially cover the skill via gimprc string-equality.
- **Multi-layer .xcf sources** — PIL cannot write .xcf. Build via `gimp -b "(create-multi-layer-xcf ...)"` once Script-Fu plug-in path is validated.
- **`check_textbox_on_leftside` / `check_triangle_position`** — bespoke eval funcs with specific geometric anchors; need a custom geometric source builder.

## Cycle-recurring failures to avoid (gimp-specific)

- **F12 (postconfig dialog button focus race)** — the Export As → "File Exists" → "Replace?" → JPEG quality dialog sequence is fragile. §I rows bypass this (oracle is `cp`, not a GIMP UI export). config_status rows reuse `_GIMP_QUIT_POSTCONFIG`.
- **F9 (multi-layer agent-cap)** — agent picks wrong layer (Background vs named). §I uses flat single-layer sources only.
- **Action-history UI side effect** — `/home/user/.config/GIMP/2.10/action-history` updates when agent uses menus; not oracle-dependent.

## Pipeline reference

§I: `pre_config_steps` stages a single image (PIL heredoc or `_stage_asset` real photo); agent edits via GIMP UI; oracle = `cp '<gold>' '<out>'` shell; eval opens both via PIL using `_eval` dispatch. config_status: shell heredoc writes gimprc/sessionrc to opposite state; agent edits via Preferences UI; postconfig graceful quit; eval reads the config file via regex.
