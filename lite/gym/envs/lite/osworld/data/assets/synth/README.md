# assets/synth/ — real-content asset bundle for synth tasks

Real images / audio / video / document templates that synth task `pre_config_steps` materialize into the container. Hosted on HuggingFace and downloaded at install time (see Lifecycle below); referenced from `train.synth.jsonl` by stable relative path.

Sibling directories under `assets/` may be added later for other bundles (e.g., `assets/eval-reuse/` for OSWorld eval-asset copies, `assets/shared/` for cross-bundle resources). Each bundle owns its own `README.md` + `MANIFEST.csv`.

## Lifecycle

The bundle is **hosted on HuggingFace** — dataset [`cua-lite/lite.osworld-assets`](https://huggingface.co/datasets/cua-lite/lite.osworld-assets) — and is **no longer committed to git** (only this `README.md` and `MANIFEST.csv` remain in-repo).

- **Runtime / codegen**: `scripts/utils/assets.sh pull` downloads the bundle (at the pinned `data/assets.lock.yaml` revision) into `<env>/.cache/assets/pulled/synth/`. Both the runtime staging path (`src/utils/dispatch.py`) and the codegen path (`src/gen/train/synth/`) resolve it through `asset_root()` (`src/utils/assets.py`), which prefers the stamped `.cache/` download and falls back to an in-repo `data/assets/synth/` copy if one is present (dev). Runtime staging copies files into the container directly — no `file://`/HTTP fetch at task time.
- **Updating bundle content**: rebuild locally from source URLs, update the hosted bundle, then bump the pinned revision in `data/assets.lock.yaml`.

The path layout below (`photos/…`, `html/…`, …) is stable and identical in the HF repo and the local cache.

## Layout

```
assets/synth/
├── README.md                 (this file)
├── MANIFEST.csv              (every asset listed: path / license / source / intended use)
├── photos/
│   ├── landscape/            outdoor scenery, mountains, beaches
│   ├── portrait/             single-subject people photos
│   ├── architecture/         buildings, floor plans
│   ├── food/                 dishes, ingredients
│   ├── product/              consumer products on neutral bg
│   └── nature/               flora, fauna, wildlife
├── graphics/
│   ├── logo/                 small brand-style marks (transparent PNG)
│   ├── diagram/              flowcharts, architecture diagrams
│   ├── chart/                pre-rendered bar/line/pie chart images
│   └── icon/                 small UI icons
├── scans/
│   ├── document/             scanned letters, contracts, articles
│   ├── receipt/              scanned receipts (vendor / amount visible)
│   └── form/                 scanned blank forms
├── video/
│   ├── solid_color/          ffmpeg-buildable; in-repo redundant — put only special cases
│   ├── nature/               short real video clips ≤30s
│   └── interview/            talking-head clips ≤30s
├── audio/
│   ├── voice/                real spoken-word clips ≤30s
│   ├── music/                short music clips ≤30s
│   └── sine/                 ffmpeg-buildable test tones (in-repo if used as fixture)
├── html/                     mock-website source HTML
│   ├── form/                 mock forms (booking / search / etc.)
│   ├── article/              mock article pages (recipe / news / etc.)
│   └── dashboard/            mock dashboard HTML w/ widgets
└── docs/                     pre-built docx/pptx/xlsx templates (rare; mostly programmatic)
    ├── template_invoice/
    ├── template_resume/
    └── template_letter/
```

## Naming convention

**Filename describes WHAT THE FILE SHOWS, not the upstream catalog ID.** A NASA URL like `https://images-assets.nasa.gov/image/S69-31739/S69-31739~orig.jpg` should NOT become `s69_31739.jpg` in this repo — that name is opaque to anyone who doesn't know NASA's catalog. Instead use `astronaut-apollo11-crew.jpg` (because the photo shows Armstrong, Collins, Aldrin).

**Rules**:

1. **kebab-case lowercase** — `earth-blue-marble-apollo17.jpg`, not `EarthBlueMarble_AS17.jpg`
2. **Content-descriptive** — say what the image / video / icon depicts. The reader should know what they'll see without opening the file.
3. **Drop redundant type-prefix when subdir already implies it**:
   - `graphics/icon/folder.svg` ✓ (not `folder-icon.svg` — `icon/` already says it's an icon)
   - `graphics/logo/abstract.svg` ✓ (not `abstract-logo-1.svg` — `logo/` already says it's a logo)
4. **Disambiguation**:
   - Use historical year only when needed to disambiguate (`apollo11` vs `apollo17` Earth photos)
   - Use numeric suffix `-01`, `-02` only for true series of same-genre images (`floor-plan-studio-01.png`, `floor-plan-studio-02.png`)
5. **4-6 words max** — long names get truncated in plan-table cells
6. **NEVER** use catalog IDs like `pia00342`, `s69_31739`, `as17-148-22727` as the primary name. They go in the MANIFEST `source_url` column for reproducibility, not in the filename.

**Examples — good vs bad**:

| Bad (catalog ID) | Good (content-descriptive) |
|---|---|
| `s69_31739.jpg` | `astronaut-apollo11-crew.jpg` |
| `pia17944.jpg` | `mars-curiosity-panorama.jpg` |
| `as17-blue-marble.jpg` | `earth-blue-marble-apollo17.jpg` |
| `pia02570.jpg` | `europa-ice-surface.jpg` |
| `circle-logo-1.svg` | `circle.svg` (in `graphics/logo/`) |
| `folder-icon.svg` | `folder.svg` (in `graphics/icon/`) |

Stable names survive HF migration — the in-repo path layout is preserved, only the URL prefix changes (per Lifecycle section above).

## MANIFEST.csv format

One row per asset. Columns:

| asset_path | license | source | intended_use |
|---|---|---|---|
| `photos/architecture/floor-plan-studio-01.png` | CC0 | freefloorplans.net | impress real-estate property tour template |
| `photos/food/pasta-01.jpg` | Unsplash Lite | unsplash.com/photo/abc123 | impress recipe deck (slide 2 image) |

License values:
- **`CC0`** — public domain dedication, no attribution required
- **`CC-BY-4.0`** — attribution required (note in `intended_use`)
- **`unsplash`** — Unsplash License (free for commercial + noncommercial)
- **`AI-gen`** — generated via Stable Diffusion / DALL-E (own copyright)
- **`OSWorld-eval`** — copied from OSWorld eval assets; verify license per file

Files MUST have a manifest entry; un-manifested files will be flagged in V1 and removed.

## How to source new assets

Priority order:

1. **Public-domain (CC0)** — Wikimedia Commons (verify per-image; many are CC-BY rather than CC0), Unsplash, OpenClipArt, FreeFloorPlans, NASA Image Library, government open-data portals. Always preferred when truly CC0.
2. **CC-BY-4.0 / Unsplash License** — fine for our use; record attribution in MANIFEST `source` column.
3. **AI-generated** — for scarce categories (specific floor-plan styles, specific diagram shapes), use SD/DALL-E. Save the prompt in MANIFEST.csv `source` column for reproducibility.
4. **HARD RULE: NEVER reuse OSWorld eval source files** — even if license allows, copying an image from `/tmp/eval_dl/<domain>/` into train would create train/eval contamination (model sees the eval image during training, evaluation no longer measures generalization). Always source train assets independently from eval.
5. **NEVER** hotlink live external URLs (cars.com images, current Wikipedia images, etc.) — F10 instability.

**Train/eval-contamination check (V1)**: any asset added to `assets/synth/` MUST be hash-compared against every file in `/tmp/eval_dl/` (when present). Identical sha256 = contamination = V1 failure.

## Adding a new asset (workflow)

1. Place file under appropriate `<category>/<subgenre>/` with kebab-case name (in a local checkout of the bundle / this dir).
2. Append a row to `MANIFEST.csv`.
3. In the synth `{domain}.md` plan-table row, cite the path: `[mech: hf]` + `path: assets/synth/photos/architecture/floor-plan-studio-01.png`.
4. In synth Python, stage it with `_stage_asset("photos/.../file.png", dst)` — the rel path is resolved against `asset_root()`; do NOT hardcode a `file://`/repo path.
5. Update the hosted bundle and bump the pinned asset revision.

## Size budget

Now that the bundle is HF-hosted (not in git), total size is not repo-constrained, but keep it lean so the install download stays fast:
- Per-file caps: photos ≤500KB after compression, video ≤5MB per clip, audio ≤2MB per clip.
