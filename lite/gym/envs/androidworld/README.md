# AndroidWorld

`--env-id` `androidworld`

CUA-Lite wrapper for [AndroidWorld](https://github.com/google-research/android_world). 116 multi-step mobile tasks across 20 real Android apps, via `gym.make("androidworld@<TaskClassName>")` with `LiteMobileActionSpace`. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

> **KVM required** — `/dev/kvm` must be rw-accessible (usually via the `kvm` group).

```bash
# Choose one install path:
# Source path: build cua-lite/androidworld:latest if missing/stale (~15 GB, ~30 min first run).
# The build clones google-research/android_world @ d9c569f and applies docker/patches/.
uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image.
# uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh status    # image freshness/resources
# uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh rebuild   # force a fresh rebuild
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("androidworld@<TaskClassName>", max_steps=30)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes</summary>

Each direct-mode `gym.make` starts an Android emulator container. For larger runs, prefer env-server mode so admission, ownership cleanup, and retry/recovery are centralized.

**KVM setup:**

```bash
sudo usermod -aG kvm $(whoami) && newgrp kvm     # group method
sudo setfacl -m u:$(id -u):rw /dev/kvm           # ACL method (per boot)
ls -l /dev/kvm                                   # verify: your user has rw
```

**Cleanup:** `uv run --no-sync bash lite/gym/envs/androidworld/scripts/cleanup.sh` removes lingering `lite-env-*-androidworld-*` containers for the current `SESSION_ID` (defaults to `local`). Image removal: `docker rmi cua-lite/androidworld:latest cua-lite/androidworld:base`.

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("androidworld@ContactsAddContact", max_steps=10)
    result = await env.reset()
    print(result.text)
    # e.g. "Create a new contact for Hannah da Silva. Their number is +11934683520."

    result = await env.step([
        make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": [500, 500]}]},
            call_id="call_0000",
        )
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/androidworld/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `ANDROID_WORLD_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("androidworld"))
# {"eval": ["AudioRecorderRecordAudio", ...], "train": [...]}
# Each task is registered in BOTH splits so a single --head N picks the same set.
```

116 tasks across 20 apps (Contacts, Calendar, Camera, Markor, SMS, Expense, Recipe, System, OsmAnd, Retro Music, VLC, Files, Browser, Clock, etc.). Metadata in `env.metadata.others`: `difficulty` (easy/medium/hard), `tags`, `optimal_steps`.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via `task.is_successful(env)`, which checks real device state (not UI matching) and returns a binary `reward` (`1.0` success / `0.0` fail). Each `reset()` generates new random parameters, so the same task ID yields different instances.

| Action | Description |
|--------|-------------|
| `tap` | Single tap |
| `long_press` | Touch and hold |
| `swipe` | Drag gesture |
| `type` | Text input |
| `system_button` | Home, Back, or Enter |
| `open_app` | Standalone extra tool; launch app |
| `wait` | Pause |
| `screenshot` | Observe without changing state |
| `terminate` / `response` | End episode and trigger evaluation |

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/androidworld/
├── main.py                 # env class, service registration, task registration
├── data/tasks.json         # static task manifest (split/difficulty/tags/optimal_steps)
├── container.py            # container helpers
├── docker/
│   ├── Dockerfile          # fork-free: clone google-research/android_world @ d9c569f → apply patches/ → Python + JDK + SDK + AVD
│   ├── patches/            # pyproject.toml.patch + setup.py.patch (build-config) + a11y_patch.py (a11y dep)
│   ├── apps.sh             # Build-time: boot emulator, install 20 benchmark APKs
│   └── server.py           # in-container env API
└── scripts/
    ├── install.sh          # docker build :base + run privileged builder + commit :latest (build / rebuild / pull / status)
    ├── utils/tasks.sh            # dump data/tasks.json from the image (dev step; re-run when the pin changes)
    └── cleanup.sh          # docker rm -f filter (per-session)
```

**Runtime model:** env-server mode cold-starts task instances on demand, then owns admission, cleanup, and recovery. Direct mode starts containers in-process and is best kept for smoke tests.

Key `server_kwargs` (see [Configuration](#configuration) for the file + override):

| Setting | `default.yaml` key | Default |
|---------|--------------------|---------|
| Boot timeout (per container) | `server_kwargs.boot_timeout` | 240 s |
| In-container HTTP API readiness | `server_kwargs.api_timeout` | 120 s |
| ThreadPoolExecutor worker count | `server_kwargs.max_workers` | configured default; tune per host |
| Spawn image | `env_kwargs.image` | `cua-lite/androidworld:latest` |

(`AVD name` stays a code default — emulator-immutable, baked into the image.)

**Rebuild semantics:** changes to `docker/Dockerfile`, `docker/apps.sh`, patches, or image-time pip deps are picked up by re-running `install.sh` (use `rebuild` to force). `docker/server.py` and host-side code (main.py, container.py) take effect on the next env-server restart because the server is bind-mounted at run time.

</details>
