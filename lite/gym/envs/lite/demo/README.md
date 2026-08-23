# Lite.Demo

`--env-id` `lite.demo`

CUA-Lite's own demo tasks — a handful of simple desktop tasks (<10 steps each, deterministic evaluator + oracle), via `gym.make("lite.demo@<task_id>")` with `LiteDesktopActionSpace` on the `cua-lite/sandbox.linux` layout. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

lite.demo runs on the shared `cua-lite/sandbox.linux` image (the sandbox-family desktop base).

```bash
# Choose one install path:
# Source path: build cua-lite/sandbox.linux if missing/stale.
uv run --no-sync bash lite/gym/sandbox/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image.
# uv run --no-sync bash lite/gym/sandbox/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/sandbox/scripts/install.sh status    # image freshness
# uv run --no-sync bash lite/gym/sandbox/scripts/install.sh rebuild   # force a fresh rebuild
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("lite.demo@create_file", max_steps=10)
asyncio.run(env.reset())
```

## Quick Start

```python
import asyncio
import lite.gym as gym

async def main():
    env = gym.make("lite.demo@create_file", max_steps=10)
    result = await env.reset()
    print(result.text)

    result = await env.step([
        {
            "id": "call_0000",
            "type": "function",
            "function": {
                "name": "computer",
                "arguments": {
                    "actions": [{"action": "click", "coordinate": [500, 300]}],
                },
            },
        },
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/lite/demo/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `LITE_DEMO_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("lite.demo"))   # {"eval": ["create_file", ...]}
```

A handful of tasks under the `eval` split (e.g. `create_file`). Metadata in `env.metadata.others`.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via each task's deterministic evaluator (binary reward), with an oracle for validation.
