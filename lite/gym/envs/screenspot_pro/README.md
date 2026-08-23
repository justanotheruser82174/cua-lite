# ScreenSpot-Pro

`--env-id` `screenspot_pro`

CUA-Lite wrapper for [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) — a single-step click grounding benchmark for professional high-resolution computer use. The agent sees a screenshot + instruction and produces a single `point` coordinate, evaluated against a ground-truth bounding box. ~1581 tasks, via `gym.make("screenspot_pro@<task_id>")` with the `grounding.point` action surface. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

No Docker or emulator required. Run the installer to install `huggingface_hub`
and pre-download the dataset (images + annotations) into
`~/.cache/huggingface/hub/`.

```bash
uv run --no-sync bash lite/gym/envs/screenspot_pro/scripts/install.sh
```

If the cache is missing, first env use downloads the same snapshot.

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` loads tasks in-process (no server):

```python
import asyncio, lite.gym as gym
env = gym.make("screenspot_pro@<task_id>")
asyncio.run(env.reset())
```

<details>
<summary>Setup notes</summary>

**Manual pre-download:**

```bash
uv run python lite/gym/envs/screenspot_pro/scripts/utils/download_tasks.py
```

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids("screenspot_pro")  # {"eval": [...]}
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f"Found {len(task_ids)} tasks")  # ~1581

    env = gym.make(f"screenspot_pro@{task_ids[0]}", max_steps=1)
    result = await env.reset()
    print(result.text)

    result = await env.step([
        make_tool_call("point", {"coordinate": [500, 500]}, call_id="call_0000")
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/screenspot_pro/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `SCREENSPOT_PRO_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("screenspot_pro"))
# {"eval": ["excel_macos_0", "word_macos_42", "photoshop_windows_15", ...]}
```

~1581 tasks under the `eval` split (task IDs follow `<annotation_file>_<index>`). Metadata in `env.metadata.others`: `application`, `group`, `platform_os`.

## Evaluation

Single-step (`terminated=True` after one step): the `point` coordinate (cua-lite `[0, 1000]`, normalized to `[0, 1]`) is checked against the ground-truth bbox — `reward = 1.0` if inside, `0.0` otherwise.

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/screenspot_pro/
├── __init__.py
├── main.py                       # ScreenSpotProEnv (LiteBaseEnv subclass)
└── scripts/utils/download_tasks.py     # optional pre-download
```

**References:**

- Paper: [ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use](https://likaixin2000.github.io/papers/ScreenSpot_Pro.pdf)
- Dataset: [likaixin/ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)
- Reference eval code: [ScreenSpot-Pro-GUI-Grounding](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)

</details>
