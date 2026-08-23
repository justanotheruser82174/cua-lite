# AndroidLab

`--env-id` `androidlab`

CUA-Lite wrapper for [AndroidLab](https://github.com/THUDM/Android-Lab). 138 multi-step mobile tasks across 9 offline apps, via `gym.make("androidlab@<task_id>")` with `LiteMobileActionSpace`. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

> **KVM required** — `/dev/kvm` must be rw-accessible (usually via the `kvm` group).

```bash
# Choose one install path:
# Source path: build cua-lite/androidlab:latest if missing/stale (~45 min first run; downloads ~8.65 GB).
uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh

# Or, published-image path: adopt a matching GHCR image.
# uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh status    # image freshness/resources
# uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh rebuild   # force a fresh rebuild
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` brings the container up itself:

```python
import asyncio, lite.gym as gym
env = gym.make("androidlab@<task_id>", max_steps=30)
asyncio.run(env.reset())
```

<details>
<summary>Setup notes</summary>

**What `install.sh` does:** downloads `docker-file.zip` (~8.65 GB) + pinned emulator 34.2.15 zip (build 11906825, from Google's redirector) into `.cache/`, generates an adb keypair, then builds `cua-lite/androidlab:latest` (~32 GB) from `lite/gym/envs/androidlab/docker/Dockerfile` — `python:3.11-bullseye` base, JDK 8, Android SDK (`platforms;android-33`, `build-tools;33.0.0`, `platform-tools`), Pixel 7 Pro AVD + skins + x86_64 image unpacked from `docker-file.zip`.

By default the zip is fetched via `gdown` from AndroidLab's Google Drive. Operators behind locked-down networks (or with an internal mirror) can set `ANDROID_LAB_DOCKER_FILE_URL=<https-url>` before running `install.sh` to bypass Google Drive — `curl` then handles the fetch. Pre-staging the zip at `lite/gym/envs/androidlab/.cache/docker-file.zip` also works.

Idempotent — re-running `install.sh` no-ops only when the image is fresh; a changed Dockerfile/source triggers an automatic rebuild. `install.sh pull` adopts a matching GHCR image, `install.sh rebuild` forces a fresh build, and `install.sh status` prints what's present.

**KVM setup (if not already in the `kvm` group):**

```bash
sudo usermod -aG kvm $(whoami) && newgrp kvm     # group method
sudo setfacl -m u:$(id -u):rw /dev/kvm           # ACL method (per boot)
```

`container.py` passes `--group-add <kvm_gid>` to `docker run`.

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("androidlab@setting_0", max_steps=10)  # "Turn on airplane mode"
    result = await env.reset()
    print(result.text[:200])

    result = await env.step([
        make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": [500, 240]}]},
            call_id="call_0000",
        )
    ])
    result = await env.step([
        make_tool_call("terminate", {"status": "success"}, call_id="call_0001")
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/androidlab/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `ANDROID_LAB_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("androidlab"))
# {"eval": ["bluecoins_1", ..., "zoom_5"]}  — 138 tasks total
```

138 tasks across 9 apps: Bluecoins (15), Calendar (14), Cantook (12), Clock (27), Contacts (15), Maps.me (15), PiMusic (12), Settings (23), Zoom (5). All offline — no network/login. Eval-only (no train split); tasks aren't parameterized. Metadata in `env.metadata.others`: `app`. For RL training data, use `androidworld:perturb_*`.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via **compressed-XML inspection** — each task ships a hand-written `judge()` that walks the UI-Automator tree after every step (cached as `best_judge` when `judge_page=True`); `reward = 1.0 if best_judge["complete"] else 0.0`. The agent doesn't need to end on the right screen, only to pass through it.

| Action | AndroidLab / adb |
|----------|------------------|
| `tap` | `adb shell input tap X Y` |
| `long_press` | `adb shell input swipe X Y X Y <duration_ms>` |
| `swipe` | `adb shell input swipe SX SY EX EY 400` |
| `type` | ADBKeyboard broadcast + **auto KEYCODE_ENTER** (matches reference) |
| `system_button` | `adb shell input keyevent KEYCODE_{HOME,BACK,ENTER,MENU}` |
| `open_app` | Standalone extra tool: `find_package(app_name)` → `monkey -p <pkg> -c LAUNCHER` |
| `wait` | blocking `sleep` |
| `screenshot` | no-op (always captured post-step) |
| `terminate` / `response` | standalone extra tools; end episode |
| `pinch` | not advertised: `android_supported_actions()` subtracts it for the whole Android family, so `step` noops it before dispatch |

<details>
<summary>Observation modes</summary>

- `observation_text="none"` (default) — screenshot only, matches `androidworld`.
- `observation_text="a11y_tree:pixel"` / `"a11y_tree:norm"` — reference's compressed UI-Automator tree as JSON.
- `observation_text="a11y_list:pixel"` / `"a11y_list:norm"` — flat element list (analogous to `androidworld`'s `a11y:*`).

The internal judge always uses the compressed-XML tree regardless of this setting.

</details>

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/androidlab/
├── __init__.py
├── main.py                 # AndroidLabEnv + 138 task registrations
├── docker/Dockerfile       # Python 3.11 + JDK 8 + Android SDK + pinned emulator 34.2.15
├── scripts/
│   ├── install.sh          # build / rebuild / pull / status cua-lite/androidlab:latest
│   └── cleanup.sh          # clean orphan `lite-env-${SESSION_ID}-androidlab-*`
└── container.py      # AndroidLabContainer / AndroidLabContainerFactory
```

**Why docker-in-docker (and how it differs from androidworld's container layout):**

1. **Snapshot path hardcoding.** The Quick Boot snapshot has `/root/.android/system-images/...` and `/root/.android/avd/...` baked into `snapshot.pb`. Rewriting these on the host is non-trivial; inside a container the paths match natively and the snapshot loads cleanly with all seed data (ABC/AAA contacts, Pink-Floyd MP3s, `bluecoins.fydb`, `cantook.db`, etc.).
2. **Emulator version pinning.** Snapshot was saved with emulator 34.2.15 (build 11906825). Today's `sdkmanager 'emulator'` installs 36.5.10, which rejects 34.x snapshots. Our image swaps 34.2.15 in at build time.

Per-episode reset is a fast `adb emu avd snapshot load default_boot` (~3–5 s). Workers don't tear down containers between episodes. Port allocation uses the shared `lite.gym.utils.backend.ports` allocator (reservation file `<repo>/.tmp/sandbox-port-reservations.json`); androidlab reserves `21000–21999` and allocates one host port per container for the in-container HTTP API. Emulator console, adb, and gRPC stay container-internal. See the port-range map at the top of [`lite/gym/utils/backend/ports.py`](/lite/gym/utils/backend/ports.py) for the full layout.

**`type` auto-Enter:** reference's `TextOnlyExecutor.type()` always calls `controller.enter()` after the text broadcast. Many tasks implicitly depend on this. We preserve it so scores stay comparable with the paper. If the agent also emits `system_button: Enter`, Enter fires twice — a no-op in almost every modal flow.

**adb runs in-container, RPC from host:** the container hosts a FastAPI server (`docker/server.py`) that owns the local adb client. Host's `AndroidLabEnv` sends per-step actions to that API. Boot-time setup still uses `docker exec` for snapshot load, adb root, geo fix, and date pin; those calls run once per env reset, not per step.

| Setting | Default | Source |
|---------|---------|--------|
| Docker image | `cua-lite/androidlab:latest` | (fixed — rebuild to change) |
| AVD name (inside container) | `Pixel_7_Pro_API_33` | (fixed — baked into image) |
| Boot timeout | 600 s | [`configs/default.yaml`](/lite/gym/envs/androidlab/configs/default.yaml) `server_kwargs.boot_timeout` |
| Executor threads | 256 | [`configs/default.yaml`](/lite/gym/envs/androidlab/configs/default.yaml) `server_kwargs.max_workers` |
| Cache dir | `lite/gym/envs/androidlab/.cache/` | `ANDROID_LAB_CACHE` |
| Host port range | 21000–21999 | (hardcoded in `container.py`) |

</details>
