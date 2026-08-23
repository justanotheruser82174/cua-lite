# README assets

Visual assets shown in the repo's [top-level README](/README.md). Each
subdirectory is **self-contained and reproducible** — its script regenerates the
artifact from source, so nothing here is an orphaned binary.

| Asset | Shown in | Reproduce |
|---|---|---|
| [`teaser/animation.gif`](/assets/README/teaser/) | README header | `python teaser/make_gif.py` — pixel-art animation composited from `teaser/assets/` |
| [`showcase/*.gif`](/assets/README/showcase/) | README quick-start | `uv run bash showcase/make_showcase.sh` — one `gpt-5.5` rollout per benchmark (desktop / browser / mobile) |

See each subdirectory's `README.md` for the file layout and options.
