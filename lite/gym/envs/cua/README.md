# CUABench + `cua.sandbox`

`--env-id` `cua.bench.local.basic` · `cua.bench.local.kicad` · `cua.bench.local.workflows`

CUA-Lite wrappers over the [Cua](https://github.com/trycua/cua) ecosystem — Cua provisions the machine
(local Docker/QEMU by default, or Cua cloud), so there's no cua-lite image to build. Two entry points (see [docs/envs.md](/docs/envs.md) for the env contract):

- **CUABench** (`cua.bench.<mode>.<dataset>`) — [cua-bench](https://github.com/trycua/cua/tree/main/libs/cua-bench)
  benchmark tasks via `gym.make`, scored by cua-bench's own evaluator. Uses `LiteDesktopActionSpace`.
- **`cua.sandbox`** — open-ended, instruction-driven (`CuaSandboxEnv`, not `gym.make`); `reward=None`.
  Android sandboxes use `LiteMobileActionSpace`.

| Dataset (env_id) | Tasks | Backend | Notes |
|---|---|---|---|
| `cua.bench.local.basic` | 68 | `cua-xfce` container (Docker) | GUI actions (`click-button`, `fill-form`, `drag-slider`, …); rendered via `launch_window` HTML |
| `cua.bench.local.kicad` | 25 | `cua-xfce` container (Docker) | KiCad EDA (schematic edits; reward = netlist compare). Run at `--concurrency 1` (see [Evaluation](#evaluation)) |
| `cua.bench.local.workflows` | 52 | `cua-xfce` container (Docker) | real-app workflows (`openshot-tasks`, `unity-tasks`) |

Backend is named by the env_id (lifecycle family is per-env_id): `webtop` = in-process Playwright HTML
(no Docker) · `local` = `trycua/cua-xfce` container (Docker, cleanup-managed) · `cloud` = Cua cloud
*(planned; needs `CUA_API_KEY`)*. Shipped datasets all declare `native` → `local`; the dev
`example_tasks` fixtures (mixed `webtop`/`local`) are not downloaded by `install.sh`.

## Setup

```bash
# Install Python deps, Playwright, datasets, and the trycua/cua-xfce local image.
uv run --no-sync bash lite/gym/envs/cua/scripts/install.sh

# Optional runtime configuration:
# export CUA_API_KEY=...              # only for Cua cloud; local modes need just Docker
# export CUA_BENCH_DATASET_ROOT=...   # optional: override the .cache default with one dataset dir
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("cua.bench.local.basic@click-button/0", max_steps=30)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes</summary>

`cua.bench.local.*` is a DEDICATED container backend — run it behind an env-server so leaked containers stay scoped to the serving session (see [/docs/envs.md#env-server](/docs/envs.md#env-server)). `scripts/cleanup.sh` sweeps leftover containers + image-tags; `scripts/uninstall.sh` removes the `.cache` datasets.

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("cua.bench.local.basic@click-button/0", max_steps=30)  # dataset env_id, task = env/variant
    result = await env.reset()
    print(result.text)

    result = await env.step([
        make_tool_call("computer", {"actions": [
            {"action": "click", "coordinate": [500, 300]}
        ]}, call_id="call_0000")
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

cua.bench defaults live in [`bench/configs/default.yaml`](/lite/gym/envs/cua/bench/configs/default.yaml) — per-dataset `env_kwargs` / `server_kwargs` plus carried `make_kwargs.cursor`, read via `env_config.load`; swap the whole file with `CUA_BENCH_CONFIG=<abs-path | name-under-configs/>`. cua.sandbox defaults live in [`sandbox/configs/default.yaml`](/lite/gym/envs/cua/sandbox/configs/default.yaml) and are overridden by explicit `CuaSandboxEnv(...)` kwargs. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
gym.registry.registered_env_ids()   # → ["cua.bench.local.basic", "cua.bench.local.kicad", ...]  (one per dataset)
gym.registry.task_ids("cua.bench.local.kicad")   # → {"eval": ["154d0750", "1625e97a", ...]}  (task = environment)
```

**Each dataset is one env_id** `cua.bench.<mode>.<dataset>` (the collection); a **task_id** is an
environment (single-variant, e.g. `154d0750`) or `environment/variant` (e.g. `click-button/0`), under
the `eval` split. `install.sh` downloads the datasets, so they're available with no export.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via cua-bench's per-env evaluator, matching cua-bench's own `runners.py` (`solved = reward >= 0.5`). See [`/docs/examples/cua.md`](/docs/examples/cua.md) for worked examples and score reproduction.

> **KiCad → `--concurrency 1`.** KiCad's per-task `apt install` streams through cua's `run_command`,
> which truncates under concurrent load — verified: at `--concurrency 4` the canary `154d0750` scored
> `0.0` (+ reset errors) vs `0.667` sequentially. `basic`/`workflows` have no per-task install, so they
> run concurrently fine (e.g. `--concurrency 8`).
