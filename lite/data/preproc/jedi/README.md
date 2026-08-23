# Jedi — evaluated and **dropped** (quality post-mortem)

**Status: not integrated.** The Jedi dataset (`xlangai/Jedi`) was preprocessed,
audited, quality-filtered, and ultimately **dropped in its entirety** — the data
quality is too low to be worth including, even after aggressive filtering. This
file is the record of *why*, so the dataset is not re-added without re-evaluating
these issues. The adapter code, the HuggingFace repo (`cua-lite/Jedi`), and the
local processed data have all been removed.

> **All figures below are historical and UNVERIFIED against the current tree.**
> They were measured against the normalized parquets at audit time; the adapter
> and the processed data are gone (this directory now holds only this file and
> `assets/`), so nothing here is re-derivable without redoing the preproc from
> `xlangai/Jedi`.

## What it was (normalized stats, before any filtering)

A desktop GUI corpus from Figma design mockups, OS-layout screenshots, and
extracted app icons. First normalized pass produced **3,380,651 rows**:

| cohort | rows |
|---|--:|
| `desktop/understanding` · icon_caption | 331,964 |
| `desktop/understanding` · layout | 851,696 |
| `mobile/understanding` (iOS app icons) | 49,498 |
| `desktop/grounding.bbox` | 1,967,196 |
| `desktop/grounding.point` | 180,297 |
| **total** | **3,380,651** |

## Quality problems found (the categories)

### 1. Non-screenshot "blank-canvas" images
Large parts of Jedi are not screenshots at all — they are solid-color canvases
holding a single (or a few) isolated icon(s), measured at **~99.8 % blank**
(non-background pixels average **0.18 %**). Evidence (a verbatim `icon_caption`
"screenshot"):

![blank icon image](/lite/data/preproc/jedi/assets/dropped_icon_caption_blank_image_example.png)

This affects:
- the whole **`icon_caption`** family (desktop `icons_v0122`/`icons_v0222` + mac + iOS), and
- **~60 %** of **`grounding.point`** images (the `icon_v0222_grounding` source, ~162 K rows) — a blank canvas with ~8 scattered icons.

### 2. Ungrounded answers (metadata / filename hallucination)
The `icon_caption` answers are not derivable from the image:
- iOS/mac subsets answer with **App Name / Developer / Category / Price /
  Description** — none visible in a logo.
- desktop subsets describe a grounded *appearance* but assert a *function* taken
  from the **source filename** (a red hexagon → `debug-breakpoint-data.png` →
  "represents a data breakpoint in a debugging context").

Training on these teaches metadata hallucination.

### 3. Prompt-template redundancy
The layout grounding/understanding annotates each element many times with
different prompt-template wrappers but an **identical target and description**.
One screenshot held **2,450 rows over only 349 unique elements** (~7×); for
`grounding.bbox` the same `(image, bbox)` recurs ~5× → **80 % of bbox rows are
pure duplicates**. (This also exploded the HF footprint, since the viewer format
re-embeds the screenshot per row — Jedi was ~1.6 TB on HF.)

### 4. Garbled instruction text
~**14 %** of `grounding.bbox` prompts are grammatically broken from template
splicing, e.g. *"Locate the element: Here's what this **The element is a star
icon, commonly used in rating systems…** looks like:"*.

## Why dropped entirely (not just filtered)

Filtering removed the worst parts (drop `icon_caption`; dedup the template
redundancy: 3.38 M → 1.23 M rows). But the residual was still weak: the bulk of
`grounding.point` is synthetic icon-soup on blank canvases, the instructions are
verbose and frequently garbled, and what remains is heavily-augmented synthetic
grounding rather than real GUI-screen interaction. Net value did not justify
inclusion, so the dataset was removed.

## Reproducing the audit
The numbers above come from sampling the normalized parquets: blank-pixel
fraction per image, `(image, bbox)` duplicate counts, per-element annotation
counts, and prompt-text pattern matching. See the git history of this directory
for the original adapter if a re-evaluation is needed.
