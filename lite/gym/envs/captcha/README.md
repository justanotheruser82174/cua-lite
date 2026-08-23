# CAPTCHA

`--env-id` `captcha`

CUA-Lite in-browser CAPTCHA suite — inspired by reCAPTCHA v2 and [ASTRAL-Group/ReCAP-Agent](https://github.com/ASTRAL-Group/ReCAP-Agent)'s `dynamic_captchas` benchmark, with cua-lite extensions (rotation, math, asset-driven image grids). 603 tasks (10 train + 297 eval + 296 test) across 8 categories (text OCR, slider, rotation, icon-select, arithmetic, 3×3 image-grid, drag-match, carousel), via `gym.make("captcha@<task_id>")` with `LiteBrowserActionSpace`. See [docs/envs.md](/docs/envs.md) for the env contract.

## Task Families

The suite covers `text_captcha_4`, `slider`, `rotation`, `icon_click`,
`math`, `icon_match`, `image_select` (`crop`, `full`, and held-out Halligan
test sources), and `paged`.

`image_select_{crop,full}` train/eval uses a handmade Wikimedia pool; the
**held-out** test is 72 real reCAPTCHA v2 challenges from Halligan (Teoh et
al., USENIX '25) — same server and reward logic, just a different test source.

Each category runs a Playwright-driven Flask challenge server: the server is spawned as a subprocess on the host, a headless Chromium drives the page, and ~1 s startup keeps RL rollout cheap.

Unlike `androidworld`, captcha tasks are **single-page** with binary 0/1 reward at terminal step — designed for short-horizon GUI grounding RL signals.

## Setup

```bash
# Install deps, Chromium, and assets.
uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh

# Optional lifecycle helper:
# uv run --no-sync bash lite/gym/envs/captcha/scripts/install.sh status      # check state
```

Assets are required at registration time; run `install.sh` before direct or
env-server use. To remove the downloaded cache:

```bash
uv run --no-sync bash lite/gym/envs/captcha/scripts/uninstall.sh
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` spawns the per-instance Flask + Chromium itself:

```python
import asyncio, lite.gym as gym
env = gym.make("captcha@<task_id>", max_steps=15)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes — deps, lifecycle, cleanup</summary>

`install.sh` installs the Python deps (flask / pillow / playwright / huggingface_hub — not a pyproject extra, matching mobilegym / browsergym), the Playwright Chromium binaries, and the challenge assets (~75 MB) into git-ignored `.cache/assets/`.

Each captcha env spawns a per-instance Flask subprocess + dedicated headless Chromium (~400 MB); the env-server's L1 (host RAM/load) + L2 (in-flight cap) handle backpressure (no internal pool). No `start.sh` / shared service — each episode spawns its Flask server in `reset()` and kills it in `close()`. Leak recovery:

| Layer | When | What |
|---|---|---|
| `env.close()` | every episode end | kills the Flask proc, closes Chromium, removes `/tmp` files |
| env-server idle TTL | client never closed | server calls `env.close()` after `--idle-ttl-sec` |
| service cleanup (steady) | every ~120 s | kills `PPid==1` orphan Flask procs + chromiums, sweeps stale `/tmp/captcha_*` files |
| service cleanup (boot) | env-server boot | same sweep at startup to recover procs leaked by a killed prior server |
| `scripts/cleanup.sh` | manual | same sweep by hand |

All categories are **content + config driven** — magic numbers, vocab, and rendering hyperparameters live in `.cache/assets/<captcha_id>/train_eval.json`, not the Python source.

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("captcha@slider_local", max_steps=15)
    result = await env.reset()
    print(len(result.image), "bytes screenshot")
    print(result.text)
    # "A slider CAPTCHA puzzle is displayed in the browser. ..."

    result = await env.step([
        make_tool_call("computer", {"actions": [
            {"action": "drag",
             "start_coordinate": [120, 600],
             "coordinate":       [350, 600]},
        ]}, call_id="call_0000"),
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/captcha/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `CAPTCHA_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("captcha"))   # {"train": [...], "eval": [...], "test": [...]}
```

603 tasks (10 train + 297 eval + 296 test) across 8 categories. Metadata in `env.metadata.others`: `category`, `mode`. Splits, modes, and families are detailed below.

## Tasks, splits, modes

Three orthogonal axes select tasks. Everything below is queryable at export time and in env-server client mode (metadata travels over `GET /envs/captcha/tasks`):

| Axis | How | Meaning |
|---|---|---|
| **Challenge type** | `--filter` on `m.others["category"]` | which of the 8 categories |
| **Distribution** | `--split` | **split semantics are distribution semantics**: `train` / `eval` = in-distribution, `test` = always OOD |
| **Content mode** | `--filter` on `m.others["mode"]` | non-default content modes (see below); absent = default distribution |

### Splits

| Split | Tasks | What they are |
|---|---|---|
| `train` | `random_<family>_local` | **A distribution handle, not a fixed task**: `seed=None`, every `reset()` generates a fresh random challenge from `train_eval.json`. One handle per family = an infinite training stream. (GRPO injects a shared per-group seed so same-group rollouts see the same challenge.) |
| `eval` | `<family>_local` + `<family>_local_eval{0..31}` | A **fixed 33-question exam** from the *same* `train_eval.json` distribution, pinned to seeds `42 + N×7919`. Byte-identical across runs → comparable across checkpoints. |
| `test` | `<family>_held_out_local_test{0..31}` | **Always OOD, never touched by the training loop.** Same renderer, disjoint content (`test/held_out.json`: new vocab / disjoint background pools) on a disjoint seed range (`10,000,003 + N×7919`). image_select instead ships 72 real reCAPTCHA v2 challenges (Halligan, Teoh et al. USENIX '25; Jaccard ≥ 0.75 scoring): `image_select_halligan_local_test{0..71}`. |

Why 33 eval seeds: a single fixed-seed eval gives ±0.5 std on a binary metric at p ≈ 0.5; with 32+1 distinct deterministic challenges per family the std of the mean drops to ~±0.09 — enough to detect ~10 pp accuracy improvements between checkpoints. Same idea as androidworld's `eval_32.parquet`, but built into the registry.

### Content modes (`others["mode"]`)

| `mode` | Sub-env | Split | Meaning |
|---|---|---|---|
| `crop` | image_select | train + eval | 3×3 tiles cut from one photo (standard distribution) |
| `full` | image_select | train + eval | 3×3 grid of independent images (standard distribution) |
| `halligan` | image_select | test | real reCAPTCHA v2 OOD data |
| `easy` | rotation | train | relaxed training distribution (`train_easy.json`: 20° tolerance vs the standard 10°) for RL bootstrapping |
| *(absent)* | all others | — | default `train_eval.json` distribution |

Adding a training distribution is data-only: drop a `train_<mode>.json` next to a family's `train_eval.json` and `random_<family>_<mode>_local` registers itself with `mode=<mode>`. Note image_select has no untagged default — crop and full are both standard, so its filters always name a mode.

### Counts per category

| Category | train | eval | test |
|---|---:|---:|---:|
| `rotation` | 2 (standard + `easy`) | 33 | 32 |
| `image_select` | 2 (`crop` + `full`) | 66 (33 each) | 72 (`halligan`) |
| other 6 categories | 1 | 33 | 32 |
| **total (`captcha`)** | **10** | **297** | **296** |

```python
import lite.gym as gym
ids = gym.registry.task_ids("captcha")   # one env_id for all categories
# {"train": [...10], "eval": [...297], "test": [...296]}
gym.registry.task_metadata("captcha", "slider_local_eval3").others
# {"source": "captcha", "category": "slider", "task_id": "slider_local_eval3"}
```

### Task families

| Task family | Skill tested | max_steps |
|---|---|---:|
| `text_captcha_4` | OCR + type + click submit | 10 |
| `slider` | Drag puzzle piece into a gap | 15 |
| `rotation` | Drag slider to rotate image upright | 15 |
| `icon_click` | Click ALL icons matching a category | 15 |
| `math` | OCR an arithmetic expression + type the answer | 10 |
| `image_select` (`mode=crop`) | 3×3 tiles from one big scene; click matching tiles | 15 |
| `image_select` (`mode=full`) | 3×3 grid of independent images; click matching tiles | 15 |
| `icon_match` | Find the visually-identical pair, drag one onto the other | 10 |
| `paged` | Use arrows / dots to navigate to the target card, click submit | 15 |

### Export recipes

> Run these inside the Slime container (see [docs/grpo.md](/docs/grpo.md)) — `export_tasks` is a training-side command, so the `python -m …` invocations below are bare (no host-side `uv run` prefix).

```bash
# rotation train parquet, standard 10° tolerance:
python -m lite.train.export.export_tasks --env-id captcha --split train \
    --filter "lambda m: m.others['category'] == 'rotation' and not m.others.get('mode')" \
    -o rotation_train.parquet

# easy mode (20° tolerance, for RL bootstrapping):
python -m lite.train.export.export_tasks --env-id captcha --split train \
    --filter "lambda m: m.others.get('mode') == 'easy'" -o rotation_easy_train.parquet

# crop-only image_select eval:
python -m lite.train.export.export_tasks --env-id captcha --split eval \
    --filter "lambda m: m.others.get('mode') == 'crop'" -o crop_eval.parquet

# everything OOD (final evaluation): just the test split, no filter needed
python -m lite.train.export.export_tasks --env-id captcha --split test -o ood_test.parquet
```

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`); reward is **binary** — `1.0` iff the challenge was submitted correctly, else `0.0`. On each submit the Flask server records `{"submitted", "correct"}` (served at `GET /result`) and the terminal step reads it once; a correct submit is **locked** (a later wrong re-submit can't overwrite it). Truncation without a submit → `0.0`; mid-episode steps return `reward=None`.

All canonical browser coordinate actions are supported (see [`/lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py)). Captcha-relevant ones:

| Action | Used by |
|---|---|
| `click` | All categories (Submit button, tile selection, dot navigation, …) |
| `drag` | `slider`, `rotation`, `icon_match` |
| `type` | `text_captcha_4`, `math` |
| `terminate` / `response` | Standalone extra tools for all categories; call after the captcha shows "✓ Verified" |

## Key Paths

Reproducibility: `CAPTCHA_SEED=42 python lite/gym/envs/captcha/servers/slider.py` gives byte-identical challenges. The eval registration exports `CAPTCHA_SEED` and `PYTHONHASHSEED` into the Flask subprocess (both required — some servers iterate `dict`/`set` whose order depends on hash randomization). Training variants (`random_<id>`) use `seed=None` for fresh randomness each reset.

| Setting | Default | Env Var |
|---|---|---|
| Eval seed | `42` | (`_EVAL_SEED` constant in [main.py](/lite/gym/envs/captcha/main.py)) |
| Eval seed step | `7919` | (`_EVAL_SEED_STEP` constant) |
| Eval seeds per family | `32`+1 | (`_EVAL_VARIANTS` constant) |
| Result file (server → env) | `/tmp/captcha_result.json` (env passes the server-scoped `/tmp/captcha_<scope>_<port>.json`, where `<scope>` is the owning env-server's port or `direct`) | `CAPTCHA_RESULT_FILE` |
| Server seed | unset → random | `CAPTCHA_SEED` |
| Mode dispatch | `train_eval` | `CAPTCHA_MODE` (`train_eval` / `test/<src>` / etc.) |
| Static challenge id | `0` | `CAPTCHA_TEST_ID` (only used in static `test/<src>/<NNN>/` path) |
| Local viewport | `1920×1080` (default from [`configs/default.yaml`](/lite/gym/envs/captcha/configs/default.yaml) `env_kwargs.display_resolution`) | `display_resolution=` kwarg on `gym.make` (per-call override) |

The `CAPTCHA_*` vars above are separate from [Configuration](#configuration) — they configure the Flask challenge-server subprocess, not the env defaults.

## Architecture

```
lite/gym/envs/captcha/
├── __init__.py
├── main.py                       # LocalCaptchaEnv + registry (_CATEGORIES, instructions) + cleanup hooks
├── README.md                     # this file
│
├── servers/                      # one self-contained Flask app per category
│   ├── text_captcha_4.py  # OCR distorted text
│   ├── slider.py          # slider drag
│   ├── rotation.py        # rotation
│   ├── icon_click.py      # multi-icon click
│   ├── math.py            # math expression
│   ├── image_select.py    # 3×3 grid (crop / full / halligan modes)
│   ├── icon_match.py      # drag-to-match pair
│   └── paged.py           # carousel with arrows + dots
│
├── scripts/
│   ├── install.sh                # idempotent: deps + chromium + assets → .cache/
│   ├── cleanup.sh                # kill leftover Flask procs + /tmp files
│   ├── uninstall.sh              # cleanup.sh + remove .cache/
│   └── utils/
│       └── download_assets.py    # asset download (called by install.sh)
│
└── .cache/                       # git-ignored runtime downloads
    └── assets/
        ├── text_captcha_4/       # train_eval.json + test/held_out.json
        ├── slider/               # train_eval.json + test/held_out.json
        │   └── backgrounds/      # train/ + test/ photo pools
        ├── rotation/             # train_eval.json + train_easy.json (20° tolerance)
        │                         #   + test/held_out.json + backgrounds/{train,test}/
        ├── icon_click/           # train_eval.json + test/held_out.json
        │                         #   + backgrounds/{train,test}/
        ├── icon_match/           # train_eval.json + test/held_out.json
        ├── paged/                # train_eval.json + test/held_out.json
        ├── math/                 # train_eval.json + test/held_out.json
        ├── image_select/
        │   ├── crop/             # train_eval.json + 19 categories of <NNN>.{jpg,json}
        │   ├── full/             # train_eval.json + same categories with
        │   │                     #   {needs_review, negative, positive}/ subsets
        │   └── test/halligan/    # 72 real reCAPTCHA v2 challenges (all split=test)
        └── snapshots/            # PNGs for README gallery (artifacts only)
```

Each `servers/*.py` is self-contained — Flask + PIL + inline HTML/CSS/JS, no shared utilities. Assets (images + JSON configs) are hosted at [OnAnOrange/captcha-assets](https://huggingface.co/datasets/OnAnOrange/captcha-assets) and downloaded into `.cache/assets/` by `install.sh`.
