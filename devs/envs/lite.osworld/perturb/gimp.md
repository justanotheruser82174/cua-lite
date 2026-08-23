# GIMP — Perturbation Plan

Domain-specific plan for `gimp`. Common workflow is in [`AGENTS.md`](/devs/envs/lite.osworld/perturb/AGENTS.md).

Code: `lite/gym/envs/lite/osworld/src/gen/train/perturb/gimp.py`

> **Cycle 32 (rollout audit) — postconfig `activate_window` now matches by WM_CLASS.**
> All postconfig `activate_window` steps targeting GIMP use
> `{"window_name": "Gimp", "by_class": True}` instead of `{"window_name": "GIMP"}`.
> With no image loaded, GIMP's title is "GNU Image Manipulation Program"
> (no "GIMP" substring) so wmctrl's case-insensitive substring match silently
> failed → ctrl+q misrouted → gimprc/sessionrc never flushed. WM_CLASS "Gimp"
> is stable across image-loaded and image-less window states. Mirrored fix in
> `synth/gimp.py`. Direct trigger: HOMO_ZERO on `7767eef2`, `7b7617bd`, `d52d6308`,
> `a746add2` in the train.perturb rollout (azure_gpt-5.4).

---

## Cycle 35a — convergence note

No code change this session. The audit loop for `gimp` converged to **architectural limits**: the only remaining outlier is `check_structure_sim` running **4.57× over** eval rate, which traces to `_build_image_op_evaluator` substituting eval's op-specific evaluators (`check_brightness_decrease_*`, `check_palette_*`, `check_green_background`, etc.) with the generic `check_structure_sim` against a PIL-computed gold (see the [Image Op section](#image-op-type_1--type_2-sourcegold-pattern) for full rationale).

This substitution is **intentional design** — it lets one builder cover 7 image ops (TYPE_1 + TYPE_2 multiplexing) with a stricter pin-the-exact-transformation evaluator, instead of branching per eval func. The 4.57× ratio is therefore a derived consequence of the design choice, not a bug to fix; reducing it would require either dropping `image_op` TYPE_2 rows (regresses op coverage and over-represents check_config_status) or re-introducing per-op eval funcs (duplicates the substitution rationale). Loop is **converged** — no further iterations planned for cycle 35a.

---

## Step 0: Download + Analyze Source Images

Image-content–dependent perturbs (TYPE_1 / TYPE_2 of `image_op`) need to know per-task image properties (mode, alpha, size). Run this once before drafting per-task plans — it downloads each gimp task's source image, opens with PIL, and writes `/tmp/gimp_full.json`:

```python
"""Run from repo root: uv run python this_script.py"""
import json, urllib.request, pathlib
from PIL import Image, ImageStat

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl")]
gimp = [r for r in rows if "_gimp_" in r["task_id"]]
out_dir = pathlib.Path("/tmp/gimp_imgs"); out_dir.mkdir(exist_ok=True)

result = {}
for r in gimp:
    tid = r["task_id"].split("_")[-1][:8]
    excl = r["metadata"]["others"].get("exclude_reason", "")
    files = []
    for step in r["metadata"].get("config", []):
        if step.get("type") != "download": continue
        for f in step["parameters"].get("files", []):
            url, vm_path = f["url"], f["path"]
            ext = vm_path.lower().rsplit(".", 1)[-1] if "." in vm_path else ""
            if ext not in ("png","jpg","jpeg","bmp","gif","xcf"): continue
            local = out_dir / f"{tid}_{pathlib.Path(vm_path).name}"
            if not local.exists():
                try: urllib.request.urlretrieve(url, local)
                except Exception as e:
                    files.append({"vm_path": vm_path, "ext": ext, "error": f"download:{e}"}); continue
            entry = {"vm_path": vm_path, "ext": ext}
            if ext == "xcf":
                entry["pil_supported"] = False; files.append(entry); continue
            try:
                img = Image.open(local); w, h = img.size; mode = img.mode
                has_alpha = mode in ("RGBA","LA") or "transparency" in img.info
                alpha_zero_pct = None
                if has_alpha and mode == "RGBA":
                    alpha = img.split()[-1]
                    alpha_zero_pct = round(sum(1 for p in alpha.getdata() if p == 0) / (w*h), 3)
                rgb = img.convert("RGB")
                mean = [round(c) for c in ImageStat.Stat(rgb).mean]
                entry.update({"size": [w,h], "mode": mode, "has_alpha": has_alpha,
                              "alpha_zero_pct": alpha_zero_pct, "mean_rgb": mean,
                              "pil_supported": True})
            except Exception as e:
                entry.update({"error": f"open:{e}", "pil_supported": False})
            files.append(entry)
    result[tid] = {"files": files, "exclude_reason": excl}

pathlib.Path("/tmp/gimp_full.json").write_text(json.dumps(result, indent=2))
```

Per-task results inform `_IMAGE_TASKS` spec in code (which source path / which feasible TYPE_2 ops) and feed V4b feasibility verification.

---

## Setting / Task Type Definitions

GIMP perturb covers five task categories. The first three are TYPE_1-only (existing); `image_op` adds TYPE_1 + TYPE_2 image-content–independent perturbs (brightness/contrast/saturation/mirror/rotate/mode/fill_color); `misc_image_op` (P3-3 Phase 3-B) closes 4 gap bases that don't fit the `image_op` template:

| task_type | Perturb fn | Description | Value pool |
|---|---|---|---|
| `config_setting` | `perturb_gimp_config` | gimprc key-value pairs | Per-key pools (see below) |
| `filter_action` | `perturb_gimp_filter_action` | Filter applied to action-history | 7 GIMP filters |
| `image_resize` | `perturb_gimp_image_resize` | Image layer height target | 256/384/512/640/768/800/1024/1280 px (128 dropped — SSIM unstable) |
| `image_op` | `perturb_gimp_image_op` | PIL-computed gold (TYPE_1 + TYPE_2) for 6 base tasks: `7a4deb26` `f723c744` `554785e9` `72f83cdc` `06ca5602` `734d6579` | Per-base spec (see Image Op section); **D3-trimmed** to 12 rows total (was 32) |
| `misc_image_op` | `perturb_gimp_misc_image` | Gap-base archetypes for `77b8ab4d` (file_exists+SSIM), `e2dd0213` (textbox_left), `f4aec372` (triangle_center). `2a729ded` dropped — see audit note. | Per-base PIL gold variants (see Misc Image Op section) |

### Config Setting Keys

| gimprc key | Value pool | Notes |
|---|---|---|
| `theme` | `"Light"` / `"Dark"` / `"System"` | Must exclude orig value |
| `undo-levels` | 25/50/75/100/125/150/200/250/300 | Numeric, exclude orig |
| `layer-new-name` | Circle/Triangle/Rectangle/Diamond/Star/Oval/Square/Hexagon | Agent must use Layer→New Layer… dialog (not Preferences UI) |
| `tile-cache-size` | 512MB/1GB/2GB/4GB (bytes) | Display map in templates |
| `hide-docks` | `yes` / `no` | Boolean toggle |
| `default-grid` | 8/10/16/20/24/32/48/64 px | Grid spacing |

### Filter Pool

7 filters: `filters-vignette` / `filters-gaussian-blur` / `filters-unsharp-mask` / `filters-emboss` / `filters-edge-detect` / `filters-posterize` / `filters-pixelize`

---

## Oracle Mechanics (GIMP)

### Config Setting

`perturb_gimp_config` writes the new gimprc key-value pair (s-expression format `(key value)`) to `~/.config/GIMP/2.10/gimprc`.

- **Pre-config**: seeds gimprc with a value different from the perturb target to prevent trivial pass.
- **Oracle**: Python heredoc that strips the old key line and appends `(key new_value)\n`.
- **Postconfig**: activate GIMP window → Ctrl+Q → sleep 2 → Return (dismiss save dialog) → wait for exit loop → pkill. Ensures gimprc is flushed before evaluator reads it. (cycle 27 audit fix for SIGTERM race with save dialog.)

### Filter Action

`perturb_gimp_filter_action` replaces the filter ID in the eval's oracle (which writes to GIMP's `action-history` XML). A `killall -9 gimp` is prepended so GIMP cannot overwrite the action-history on exit.

- **Oracle**: deep-copies eval oracle and replaces old filter-ID string with new filter-ID.
- **Postconfig**: same graceful Ctrl+Q quit as config setting.

### Image Resize

`perturb_gimp_image_resize` replaces the height integer in both the evaluator and oracle commands.

- **Oracle**: deep-copies eval oracle and replaces `str(orig_height)` with `str(new_size)`.
- **Evaluator**: `rules["height"] = new_size`.
- **Pool note**: 128 px excluded — `check_structure_sim_resized` SSIM is unstable at sizes that small (cycle-30 audit fix).

### Image Op (TYPE_1 + TYPE_2 source+gold pattern)

`perturb_gimp_image_op` replaces the eval's op-specific evaluator with `check_structure_sim` against a **PIL-computed gold image**, mirroring the libreoffice_impress source+gold pattern. One internal fn handles 7 image ops via spec dict + small dispatch helpers:

```
perturb_config_step:  python3 << PYEOF: PIL gold → /tmp/perturb_expected_<short>_<t1|t2>[_<op>].png
oracle_actions:       killall gimp + mkdir -p + cp gold → result_path
evaluator:            check_structure_sim(result_path, expected=/tmp/...)
```

**Why replace eval func**: original GIMP evals are op-specific (`check_brightness_decrease_*`, `check_saturation_increase_*`, `check_image_mirror`, `check_palette_*`, `check_green_background`). They constrain to a single direction and don't pin a specific param. `check_structure_sim` against a concrete PIL gold is stricter — agent must match the specific transformation — and uniformly applicable across all image ops, enabling TYPE_1 + TYPE_2 multiplexing without per-op evaluator branching.

**SSIM threshold**: 0.9 default (upstream `structure_check_by_ssim`). Oracle output is byte-identical to gold (both produced by same PIL code) → SSIM = 1.0 → trivially passes. For real agent runs, mirror/rotate/mode/fill produce byte-equivalent results across PIL ↔ GIMP UI; brightness/contrast/saturation may need tighter SSIM tolerance — empirically PIL.enhance ↔ GIMP slider falls within 0.85–0.95 SSIM.

**PIL ↔ op mapping** (`_build_pil_expected_py`):

| op | PIL expression |
|---|---|
| `brightness` | `ImageEnhance.Brightness(img).enhance(factor)` |
| `contrast` | `ImageEnhance.Contrast(img).enhance(factor)` |
| `saturation` | `ImageEnhance.Color(img).enhance(factor)` |
| `mirror` | `img.transpose(Image.FLIP_LEFT_RIGHT \| FLIP_TOP_BOTTOM)` |
| `rotate` | `img.rotate(deg, expand=True)` |
| `mode` | `img.convert('L' \| 'P')` |
| `fill_color` | numpy mask: `(rgb != (0,0,0))` → set masked pixels to target rgb (preserves black object) |

### Misc Image Op (P3-3 Phase 3-B — gap-base archetypes)

`perturb_gimp_misc_image` covers 3 eval gap bases whose evaluators check **image content** (file existence / textbox layout / triangle position) rather than the operation itself, so they don't fit the `image_op` "PIL gold + check_structure_sim" template directly.

> **Dropped (audit, Step 0 ground truth):** `2a729ded` (transparency archetype). `dog_with_background.png` is RGBA but mean RGB [135, 141, 145] — gray-blue scene with no near-white background. Luminance thresholds 230 / 215 / 200 mask only 0.3% / 0.65% / 16% of pixels, so the PIL gold is virtually identical to the source and the instruction labels ("remove the white background", "off-white", "bright background") misdescribe the image. V2 still passes via oracle short-circuit (gold cp → result, SSIM=1.0), but a real agent doing proper background removal in GIMP produces SSIM ≪ 0.9 vs the near-original gold — inverted training signal. Removed from `_MISC_IMAGE_TASKS`; perturb_gimp_misc_image returns [] for this base.

Pattern mirrors `image_op`: PIL writes a deterministic gold to `/tmp/perturb_misc_<short>_v<idx>.png|jpg`; oracle does `killall gimp + mkdir + cp gold → result_path`; evaluator preserves the eval's original `func` and only redirects `expected` (where applicable).

| tid | eval func | source | PIL gold strategy | variants |
|---|---|---|---|---|
| `77b8ab4d` | `check_file_exists_and_structure_sim` | `The_Lost_River_Of_Dreams.jpg` | Re-save source as JPEG quality 95; vary export filename | 3 (`export.jpg` / `river_export.jpg` / `photo_final.jpg`) |
| `e2dd0213` | `check_textbox_on_leftside` | (PIL synthetic 800×600 orange canvas) | Black rectangle anchored at x_offset ≤ 12 px; evaluator passes when leftmost dark pixel < 5% width | 3 (offsets 4 / 8 / 12 px) |
| `f4aec372` | `check_triangle_position` | (PIL synthetic 800×600 white canvas) | Isoceles triangle whose **geometric centroid** lands at (W/2, H/2) — vertices (cx, cy − 2h/3), (cx ± h, cy + h/3); evaluator centroid tolerance 5% | 3 (sizes 100 / 120 / 160 px, distinct fill colors) |

**Why preserve eval func instead of replacing**: the gap evaluators encode image-content geometry (e.g., "leftmost dark pixel < 5% width" for `check_textbox_on_leftside`) — the perturb gold satisfies that geometry intrinsically, so the evaluator's behaviour transfers directly. Replacing with `check_structure_sim` would require strict pixel-level matching where the agent's natural workflow (drag a textbox in GIMP) produces variation that fails SSIM but still satisfies the original geometric predicate.

**Note on XCF sources**: `e2dd0213` and `f4aec372` download `.xcf` source files. PIL cannot read XCF, so the gold is fully synthetic (no `Image.open(src_path)`). The eval's `config` (download + GIMP launch) is preserved by `make_perturb_row`, so the agent still sees the same starting environment as eval.

**Centroid-aligned triangle** (audit fix): the previous gold used vertices `(cx, cy − h)`, `(cx ± h, cy + h)`, which yields a geometric centroid at `(cx, cy + h/3)`. For size 160 the y-offset was 26.7 px against the evaluator tolerance of 30 px — only 3 px of margin against rasterization jitter. The new vertex layout (`top_y = cy − 2h/3`, `bottom_y = cy + h/3`) places the centroid at `(cx, cy)` exactly. Empirical verification: for sizes 100/120/160 the rasterized centroid sits 0.5 px from the canvas center — two orders of magnitude inside tolerance.

### Evaluator Functions

| func | What it checks | Used by |
|---|---|---|
| `check_config_status` | gimprc key-value pair present in `gimprc` / `sessionrc` | `perturb_gimp_config` |
| `check_include_exclude` | action-history contains filter ID | `perturb_gimp_filter_action` |
| `check_image_size` (+ `check_structure_sim_resized`) | image dimensions match expected height + structure preserved | `perturb_gimp_image_resize` |
| `check_structure_sim` | result image SSIM ≥ 0.9 vs gold (PIL-computed) | `perturb_gimp_image_op` (TYPE_1 + TYPE_2) |
| `check_file_exists_and_structure_sim` | result file exists + SSIM ≥ 0.9 vs gold | `perturb_gimp_misc_image` (`77b8ab4d`) |
| `check_textbox_on_leftside` | leftmost dark pixel column < 5% of image width | `perturb_gimp_misc_image` (`e2dd0213`) |
| `check_triangle_position` | second-most-common color region centroid within 5% of image center | `perturb_gimp_misc_image` (`f4aec372`) |

---

## Perturbation Strategy

**TYPE_1 (same op, new param)**: resample from a per-task param pool excluding the eval's expected value. Three flavors:
- `config_setting` / `filter_action` / `image_resize`: legacy direct-value resample.
- `image_op`: PIL-gold + `check_structure_sim`. Pool covers **both directions** for enhance ops (brightness/contrast/saturation), since check_structure_sim is direction-agnostic — agent learns the op generally, not just one direction.

**TYPE_2 (different op)** (`image_op` only): for each base task with a source image, sample 2 ops from `feasible_t2_ops` (a per-task whitelist of image-content–independent ops different from `t1_op`); for each chosen op, sample 1 param from `_T2_POOLS[op]`. Expands op coverage so all 7 image ops appear across the training set.

**No-leakage guarantee**:
- TYPE_1 legacy: `candidates = [v for v in pool if v != orig_value]`.
- TYPE_1 image_op: pool excludes the eval's value (e.g. 7a4deb26's eval factor 0.6 not in `[0.5, 0.7, 1.3, 1.5]`).
- TYPE_2 image_op: by construction, `t1_op ∉ feasible_t2_ops` — so TYPE_2 ops are structurally different from the eval's op.

---

## Instruction Style (Domain-Specific)

GIMP eval is the **most polite** of all domains — 77% of eval instructions start with `Could you / Please / Can you / I'd like / I want` (cf. ~30% across other domains). Perturb instruction templates therefore deviate from the common-rules default of 30–40% polite and target **60–70% polite-starting** to match this domain bias. Concretely each instruction-template pool (per gimprc key, per filter, per resize, per image-op) holds **5 paraphrases**, of which **3 are polite-starting** and **2 are imperative**, plus optional motivation/context (e.g. "I'm working with bigger images now", "I want a different mood") to lift naturalness.

| metric | eval | perturb target | mechanism |
|---|---|---|---|
| polite-starting | 77% | 60–70% | 3/5 polite templates per pool |
| `save the file` | 0% | 0% | banned phrase; image-op uses `save as '<filename>'` to specify output path, not "save the file" |
| avg_words | 14.9 | 14–18 | per-template length 12–20 words; motivation kept short |

**Why no "Save the file"**: image_resize result is captured by the evaluator's postconfig (Shift+Ctrl+E export → File chooser → Return). Image_op result is captured by oracle's `cp <gold> <result_path>`. Neither requires the agent (or the agent-facing instruction) to explicitly say "save the file" — instead instructions specify the **target output filename** for image_op (e.g. `save as 'edited_darker.png' on the Desktop`) so the agent's export path matches `evaluator.result.path`.

---

## Per-task Plan

> **Keep in sync with code.** The value pool exclusions in the table must match `perturb/gimp.py`. Divergence silently produces wrong training tasks.

### Perturbable (16 tasks)

**Config/filter/resize bases** (TYPE_1 legacy):

| tid | func | eval target | Perturb pool | Rows |
|---|---|---|---|---|
| `7767eef2` | `check_config_status` | theme=`"Light"` | `"Dark"`, `"System"` (2 → 4 via R1 paraphrase fallback) | 4 |
| `7b7617bd` | `check_config_status` | undo-levels=`100` | 25/50/75/125/150/200/250/300 | 4 |
| `b148e375` | `check_config_status` | layer-new-name=`"Square"` | 7 other shape names | 4 |
| `d52d6308` | `check_config_status` | hide-docks=`yes` | `no` (1 candidate, **R1 paraphrase fallback** → 4 distinct instr variants) | 4 |
| `a746add2` | `check_include_exclude` | filter=`filters-vignette` | 6 other filters | 4 |
| `d16c99dc` | `[check_image_size, check_structure_sim_resized]` | resize h=512 | 256/384/640/768/800/1024/1280 | 4 |

**R1 paraphrase fallback** (Audit 6 fix): when `[v for v in pool if v != orig_value]` yields < 4 candidates, `perturb_gimp_config` cycles through `instruction_templates` to emit up to 4 paraphrase variants by reusing values across templates. Affected bases: `d52d6308` (1 candidate → 4 paraphrases), `7767eef2` (2 candidates → 4 rows = 2 values × 2 paraphrases). `knob_assignment["paraphrase_idx"]` is added for the 2nd-4th paraphrase of any value so each row hashes to a distinct `task_id`. This eliminates the only single-variant base in the gimp domain (was `d52d6308`) and lifts `7767eef2` from 2 → 4 rows.

**Image-op bases** (TYPE_1 + TYPE_2, source+gold) — **D3-trimmed** to control over-representation (Audit 5: was 32 / 6× eval ratio, now 12 / 12× ratio):

| tid | base eval func | source image | TYPE_1 op + pool | TYPE_2 feasible ops (n=1) | Rows |
|---|---|---|---|---|---|
| `7a4deb26` | `check_brightness_decrease_and_structure_sim` | woman_sitting_by_the_tree.png (1099×730 RGBA) | brightness ∈ {0.7} | contrast / saturation / mirror / rotate | 1+1 = 2 |
| `f723c744` | `check_contrast_increase_and_structure_sim` | berries.png (1280×851 RGB) | contrast ∈ {1.8} | brightness / saturation / mirror / rotate | 1+1 = 2 |
| `554785e9` | `check_saturation_increase_and_structure_sim` | woman_sitting_by_the_tree2.png (1099×730 RGBA) | saturation ∈ {0.5} | brightness / contrast / mirror / rotate | 1+1 = 2 |
| `72f83cdc` | `check_image_mirror` | berry.png (1280×851 RGB) | mirror ∈ {TB} (orig=LR) | brightness / contrast / saturation / rotate | 1+1 = 2 |
| `06ca5602` | `check_palette_and_structure_sim` | computer.png (5184×3888 RGB) | mode ∈ {L} (orig=P) | brightness / contrast / saturation / mirror / rotate | 1+1 = 2 |
| `734d6579` | `check_green_background` | white_background_with_object.png (800×800 RGBA) | fill_color ∈ {red} (was 6, trimmed to 1) | brightness / contrast / saturation / mirror / rotate | 1+1 = 2 |

**TYPE_2 param pools** (`_T2_POOLS`): brightness=[0.7, 1.3, 1.5] · contrast=[1.2, 1.5, 1.8, 2.0] · saturation=[0.5, 1.5, 1.8] · mirror=[LR, TB] · rotate=[90, 180, 270]. T2 sample count = **1** per base (D3, was 2).

**Misc Image Op bases** (P3-3 — gap closure, see Misc Image Op section above):

| tid | eval func | PIL gold | Rows |
|---|---|---|---|
| `77b8ab4d` | `check_file_exists_and_structure_sim` | re-saved JPEG, vary export filename | 3 |
| `e2dd0213` | `check_textbox_on_leftside` | synthetic orange canvas + left-anchored black rect | 3 |
| `f4aec372` | `check_triangle_position` | synthetic white canvas + centroid-aligned triangle | 3 |

> `2a729ded` (transparency archetype) was dropped after Step 0 audit — see the Misc Image Op section above for the full rationale.

### Not Perturbable (11 tasks: all infeasible)

| tid | Reason |
|---|---|
| `045bf3ff` | infeasible |
| `2a729ded` | infeasible — Step 0 audit: source has no white/near-white background, so PIL-gold semantics misalign with instructions and real-agent SSIM (see Misc Image Op section) |
| `2e6f678f` | infeasible |
| `38f48d40` | infeasible |
| `58d3eeeb` | infeasible |
| `5ca86c6f` | infeasible |
| `62f7fd55` | infeasible |
| `8ea73f6f` | infeasible |
| `dbbf4b99` | infeasible |
| `e19bd559` | infeasible |
| `fbb548ca` | infeasible |

---

## V4a Coverage Check

Three checks: (1) all 10 legacy ops appear in perturb output (misc_image_op rows are exempt — they don't carry a legacy op token); (2) TYPE_2 share ≥ 15% — **note**: post-D3 trim this dropped to ~13% in absolute count because D3 reduced both T1 and T2 proportionally; the threshold is informational, not blocking. (3) no single op exceeds 35% of total op-occurrences (prevents one easy op from drowning out variety).

> **Diff vs Impress**: gimp eval is 1-op-per-task across 6 image-op bases — Impress's "actions-per-task length distribution" comparison is meaningless here. Replaced with TYPE_1 / TYPE_2 split sanity check.

```python
"""V4a coverage check — gimp.
Run from repo root: uv run python this_script.py
"""
import json, re
from collections import Counter

REQUIRED_OPS = {
    "config_setting", "image_resize", "filter_action",
    "brightness", "contrast", "saturation",
    "mirror", "mode", "fill_color", "rotate",
}

# Detect ops from oracle code + perturb_config_step (PIL heredocs)
_OP_PATTERNS = [
    ("config_setting", re.compile(r"gimprc|sessionrc")),
    ("filter_action",  re.compile(r"filters-(vignette|gaussian-blur|unsharp-mask|emboss|edge-detect|posterize|pixelize)")),
    ("image_resize",   re.compile(r"\.resize\(\([^)]+\),\s*Image\.LANCZOS\)")),
    ("brightness",     re.compile(r"ImageEnhance\.Brightness")),
    ("contrast",       re.compile(r"ImageEnhance\.Contrast")),
    ("saturation",     re.compile(r"ImageEnhance\.Color")),
    ("mirror",         re.compile(r"FLIP_(LEFT_RIGHT|TOP_BOTTOM)")),
    ("rotate",         re.compile(r"\.rotate\(\d+,\s*expand=True\)")),
    ("mode",           re.compile(r"\.convert\('([LP1])'\)")),
    ("fill_color",     re.compile(r"np\.array\(img\).*?mask\s*=", re.S)),
]

def _all_text(r):
    parts = []
    for a in (r["metadata"].get("others") or {}).get("oracle_actions", []):
        c = a.get("parameters", {}).get("command", "")
        if isinstance(c, str): parts.append(c)
    for s in r["metadata"].get("config", []):
        c = s.get("parameters", {}).get("command", "")
        if isinstance(c, str): parts.append(c)
    return "\n".join(parts)

def _ops(r):
    text = _all_text(r)
    return [name for name, pat in _OP_PATTERNS if pat.search(text)]

def _perturb_type(r):
    text = _all_text(r)
    if "_t2_" in text:  return "type2"
    if "_t1." in text:  return "type1_image"
    return "type1_legacy"

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
gimp = [r for r in rows if "_gimp_" in r["task_id"]]

# (1) Op coverage
counts = Counter(op for r in gimp for op in _ops(r))
missing = REQUIRED_OPS - set(counts)
print(f"[{'FAIL' if missing else 'OK  '}] op coverage: {len(REQUIRED_OPS - missing)}/{len(REQUIRED_OPS)}, missing={missing or 'none'}")

# (2) TYPE_1 / TYPE_2 split — TYPE_2 share ≥ 15%
ptype = Counter(_perturb_type(r) for r in gimp)
t2_share = ptype.get("type2", 0) / max(len(gimp), 1)
print(f"[{'OK  ' if t2_share >= 0.15 else 'FAIL'}] TYPE_2 share: {t2_share:.0%}  splits={dict(ptype)}")

# (3) Per-op share — single op should not exceed 35%
total = sum(counts.values()) or 1
top_share = max((c / total for c in counts.values()), default=0)
print(f"[{'OK  ' if top_share <= 0.35 else 'FAIL'}] max per-op share: {top_share:.1%} (cap 35%)")
print(f"\nPer-op count (total op-occurrences={total}):")
for op in sorted(REQUIRED_OPS):
    c = counts.get(op, 0)
    pct = c / total
    flag = " ← MISSING" if c == 0 else (" ← HIGH" if pct > 0.35 else "")
    print(f"  {op:<16} {c:>3}  {pct:>5.1%}{flag}")

print(f"\nTotal gimp perturb rows: {len(gimp)}")
```

Targets:
- All 10 ops present
- TYPE_2 share ≥ 15% (currently ~13% post-D3; tracked but not blocking — D3 prioritized over-representation reduction)
- Single op share ≤ 35%
- Current generated total: **42 rows**. Historical breakdown here predates later audit drops; use `train.perturb.jsonl` as the source of truth.

---

## V4b Perturb-Eval Match Verification

### Part A: Instruction Clarity

```python
import json, random

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
gimp = [r for r in rows if "_gimp_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

rng = random.Random(0)
for r in rng.sample(gimp, min(10, len(gimp))):
    print(f"[{r['task_id'].split('_gimp_')[-1].split('_')[0]}]")
    print(f"  INSTR : {r['instruction']}")
    print(f"  ORACLE: {_oracle(r)[:200].strip()}")
    print()
```

### Part B: Feasibility (image-op rows cross-checked vs `/tmp/gimp_full.json`)

For TYPE_1 / TYPE_2 image_op rows, verify the source image's PIL-readable properties (mode, size, alpha) are compatible with the chosen op. Legacy rows (config_setting / filter_action / image_resize) are structurally guaranteed by their fn.

```python
"""V4b Part B — gimp image-op feasibility."""
import json, re, pathlib

ana = json.loads(pathlib.Path("/tmp/gimp_full.json").read_text())
rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
gimp = [r for r in rows if "_gimp_" in r["task_id"]]

def _all_text(r):
    parts = []
    for a in (r["metadata"].get("others") or {}).get("oracle_actions", []):
        c = a.get("parameters",{}).get("command","")
        if isinstance(c, str): parts.append(c)
    for s in r["metadata"].get("config", []):
        c = s.get("parameters",{}).get("command","")
        if isinstance(c, str): parts.append(c)
    return "\n".join(parts)

def _short(r):
    src = r["metadata"]["others"].get("source", "").replace("perturb:","")
    return src.rsplit("_", 1)[-1][:8]

def _op_param(r):
    text = _all_text(r)
    if "ImageEnhance.Brightness" in text:
        m = re.search(r"Brightness\([^)]+\)\.enhance\(([\d.]+)\)", text)
        return ("brightness", float(m.group(1))) if m else (None, None)
    if "ImageEnhance.Contrast" in text:
        m = re.search(r"Contrast\([^)]+\)\.enhance\(([\d.]+)\)", text)
        return ("contrast", float(m.group(1))) if m else (None, None)
    if "ImageEnhance.Color" in text:
        m = re.search(r"Color\([^)]+\)\.enhance\(([\d.]+)\)", text)
        return ("saturation", float(m.group(1))) if m else (None, None)
    if "FLIP_LEFT_RIGHT" in text:  return ("mirror", "LR")
    if "FLIP_TOP_BOTTOM" in text:  return ("mirror", "TB")
    m = re.search(r"\.rotate\((\d+),", text)
    if m and "expand=True" in text:  return ("rotate", int(m.group(1)))
    m = re.search(r"\.convert\('([LP])'\)", text)
    if m:  return ("mode", m.group(1))
    if "px[mask, 0]" in text:  return ("fill_color", None)
    return (None, None)

issues = []
for r in gimp:
    op, param = _op_param(r)
    if op is None: continue  # legacy row
    short = _short(r)
    img = next((f for f in ana.get(short, {}).get("files", [])
                if f.get("pil_supported") and "size" in f), None)
    if not img:
        issues.append((r["task_id"], op, "no PIL-readable source image"))
        continue
    mode = img["mode"]; w, h = img["size"]
    if op == "saturation" and mode not in ("RGB", "RGBA"):
        issues.append((r["task_id"], op, f"mode={mode} (saturation needs RGB/RGBA)"))
    if op == "mode" and param == mode:
        issues.append((r["task_id"], op, f"target mode == source mode ({mode}) — no-op"))
    if op == "fill_color" and w * h > 1_500_000:
        issues.append((r["task_id"], op, f"size={w}x{h} too large for fill_color"))

print(f"[{'FAIL' if issues else 'OK  '}] feasibility: {len(issues)} issues")
for tid, op, why in issues[:10]:
    print(f"  {tid} ({op}): {why}")
```

### Part C: Distribution Match (parameter coverage)

Check that each enhance op covers both directions (decrease + increase), mirror covers both axes (LR + TB), rotate covers all three angles (90/180/270). These guarantees come from pool design (`_IMAGE_TASKS[*].t1_pool` covers both directions; `_T2_POOLS[*]` provides multi-value pools); the script catches accidental drift.

```python
"""V4b Part C — gimp parameter distribution."""
import json, re
from collections import Counter

rows = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
gimp = [r for r in rows if "_gimp_" in r["task_id"]]

def _all_text(r):
    parts = []
    for a in (r["metadata"].get("others") or {}).get("oracle_actions", []):
        c = a.get("parameters",{}).get("command","")
        if isinstance(c, str): parts.append(c)
    for s in r["metadata"].get("config", []):
        c = s.get("parameters",{}).get("command","")
        if isinstance(c, str): parts.append(c)
    return "\n".join(parts)

def _params(op_pat):
    out = []
    for r in gimp:
        for m in re.findall(op_pat, _all_text(r)):
            out.append(float(m))
    return out

# C1: enhance ops cover both directions
for op, pat in [("brightness", r"Brightness\([^)]+\)\.enhance\(([\d.]+)\)"),
                ("contrast",   r"Contrast\([^)]+\)\.enhance\(([\d.]+)\)"),
                ("saturation", r"Color\([^)]+\)\.enhance\(([\d.]+)\)")]:
    vals = _params(pat)
    dec, inc = any(v < 1 for v in vals), any(v > 1 for v in vals)
    flag = "OK  " if (dec and inc and vals) else "FAIL"
    print(f"[{flag}] {op}: vals={sorted(set(vals))}  decrease={dec} increase={inc}")

# C2: mirror covers both axes
mvals = Counter()
for r in gimp:
    t = _all_text(r)
    if "FLIP_LEFT_RIGHT" in t: mvals["LR"] += 1
    if "FLIP_TOP_BOTTOM" in t: mvals["TB"] += 1
print(f"[{'OK  ' if {'LR','TB'} <= set(mvals) else 'FAIL'}] mirror: {dict(mvals)}")

# C3: rotate covers 90/180/270
rvals = Counter()
for r in gimp:
    for m in re.findall(r"\.rotate\((\d+),", _all_text(r)):
        rvals[int(m)] += 1
needed = {90, 180, 270}
print(f"[{'OK  ' if needed <= set(rvals) else 'PARTIAL'}] rotate: {dict(rvals)}  missing={needed - set(rvals)}")

# C4: filter pool coverage (filter_action)
filt = Counter()
for r in gimp:
    for f in re.findall(r"filters-(\w[\w-]*)", _all_text(r)):
        filt[f] += 1
print(f"[OK  ] filter_action: {len(filt)} distinct filters: {dict(filt)}")

# C5: gimprc key coverage (config_setting)
keys = Counter()
for r in gimp:
    for k in ["theme", "undo-levels", "layer-new-name", "tile-cache-size", "hide-docks", "default-grid"]:
        if f"({k} " in _all_text(r): keys[k] += 1
print(f"[OK  ] config keys: {dict(keys)}")
```

Targets:
- brightness / contrast / saturation each cover **both** decrease and increase
- mirror covers **LR + TB**
- rotate covers **90 + 180 + 270**
- filter_action covers ≥ 5 of 7 filter IDs (eval pool itself is 7)
- config_setting covers all 6 gimprc keys present in eval bases

---

## V4c Eval Leakage Check

```python
import json, re

all_eval = {r["task_id"]: r for r in (json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/eval.jsonl"))}
all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
gimp_perturb = [r for r in all_perturb if "_gimp_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

leakage = []
for r in gimp_perturb:
    src = r["metadata"]["others"].get("source", "")
    base_tid = src.replace("perturb:", "")
    eval_row = all_eval.get(base_tid)
    if not eval_row:
        continue
    if r["instruction"].strip() == eval_row["instruction"].strip():
        leakage.append((r["task_id"], "instruction identical to eval"))

print(f"[{'FAIL' if leakage else 'OK  '}] eval leakage: {len(leakage)} violations")
for tid, reason in leakage[:10]:
    print(f"  {tid}: {reason}")
```

---

## V4d Inter-Variant Uniqueness

```python
import json, re
from collections import defaultdict

all_perturb = [json.loads(l) for l in open("lite/gym/envs/lite/osworld/data/train.perturb.jsonl")]
gimp_perturb = [r for r in all_perturb if "_gimp_" in r["task_id"]]

def _oracle(r):
    return "\n".join(
        a.get("parameters", {}).get("command", "")
        for a in (r["metadata"].get("others") or {}).get("oracle_actions", [])
        if isinstance(a.get("parameters", {}).get("command"), str)
    )

by_source = defaultdict(list)
for r in gimp_perturb:
    src = re.sub(r"_[0-9a-f]{8}$", "", r["task_id"])
    by_source[src].append(r)

instr_dups = []
oracle_dups = []
for src, rows in by_source.items():
    if len(set(r["instruction"] for r in rows)) < len(rows):
        instr_dups.append(src)
    if len(set(_oracle(r) for r in rows)) < len(rows):
        oracle_dups.append(src)

print(f"[{'FAIL' if instr_dups  else 'OK  '}] duplicate instructions: {len(instr_dups)} sources")
print(f"[{'FAIL' if oracle_dups else 'OK  '}] duplicate oracle code:  {len(oracle_dups)} sources")
```

---

## Expected Output

- **Current generated total: 42 rows** (post Phase 3-B: P3-3 + D3 + R1 + later audit drops)
- Historical category plan before later drops:
  - config_setting: 4 bases × 4 = 16 rows (theme 4-via-paraphrase + undo 4 + shape 4 + dock 4-via-paraphrase)
  - filter_action: 1 base × 4 = 4 rows
  - image_resize: 1 base × 4 = 4 rows
  - image_op: 6 bases × (1 T1 + 1 T2) = 12 rows (D3-trimmed from 32)
  - misc_image_op: 3 bases × 3 variants = 9 rows (P3-3 gap closure; `2a729ded` dropped — see Misc Image Op section)
- **10 ops covered** (legacy): config_setting, image_resize, filter_action, brightness, contrast, saturation, mirror, mode, fill_color, rotate
- **3 new gap evaluators**: check_file_exists_and_structure_sim, check_textbox_on_leftside, check_triangle_position
- **All perturb values differ from eval target** (per-task pool constructed to exclude eval value, or paraphrased variants for single-candidate pools)
- **V2 pass-rate target**: 100% (oracle byte-identical to gold via PIL → SSIM = 1.0)
- **V3 distribution (post Phase 3-B)**: polite 69% / save 0% / multi_sep 0% / avg_words 17.0 vs eval 77% / 0% / 0% / 14.9 (within ±5pp on polite, +2.1 on avg_words)
- **R1 fix**: 0 single-variant bases (was 1: d52d6308 hide-docks → 4 paraphrase variants)
