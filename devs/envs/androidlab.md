See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Reference implementation:** `${CUA_LITE_REFERENCES_ROOT}/Android-Lab`

**Goal:**
Wrap [AndroidLab](https://github.com/THUDM/Android-Lab) (THUDM's Android agent benchmark, 138 tasks across 9 offline apps) as a cua-lite gym environment. Multi-step mobile benchmark with **XML-state evaluation** — each task ships a handwritten `judge()` that inspects the compressed UI-Automator XML tree (and occasional `adb shell settings get` / SQLite queries) after every step.

**Difficulty:** Hard — depends on Android SDK + emulator (AVD `Pixel_7_Pro_API_33`) pre-loaded with 9 offline apps (Bluecoins, Calendar, Cantook, Clock, Contacts, Maps.me, PiMusic, Settings, Zoom). The AVD snapshot bakes in seeded data (ABC/AAA contacts, Pink-Floyd MP3s in `/data/media/0/Download/Music/`, `bluecoins.fydb`, `cantook.db`) that every `judge()` assumes. Each worker emulator runs inside its own Docker container (`cua-lite/androidlab:latest`) — same per-worker-container shape as `androidworld`'s `cua-lite/androidworld:latest`, but the two images differ substantially because AndroidLab requires a Quick Boot snapshot.

## Setup

```bash
# one-time: create the unified venv (model stack + env-server runtime)
uv sync --all-extras

# 1. Install Python deps (pulls androidlab from the cua-lite fork).
uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh

# 2. Build cua-lite/androidlab:latest (~45 min first run, one-time):
#    install.sh auto-fetches docker-file.zip (~8.65 GB) on first run.
uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh
```

`install.sh` is idempotent. `install.sh rebuild` forces a fresh build; `install.sh status` prints what's present.

> A later bare `uv sync` evicts the ad-hoc env deps `install.sh` adds (e.g. webgym's host judge). After setup, use `uv run --no-sync`; if you re-sync, re-run the env's `install.sh` to re-add them.

The zip download goes through gdown against AndroidLab's Google Drive id by default. Operators on locked-down networks (or with an internal mirror of the zip) can bypass Google Drive entirely by exporting `ANDROID_LAB_DOCKER_FILE_URL=<https-url>` before running `install.sh` — `curl` then handles the fetch. Pre-staging the zip at `lite/gym/envs/androidlab/.cache/docker-file.zip` also works.

KVM must be accessible to your user (`/dev/kvm` rw — usually means membership in the `kvm` group).

The `android-lab` pip package is consumed from the cua-lite fork:

```toml
# pyproject.toml
android-lab = [
    "android-lab @ git+https://github.com/cua-lite/Android-Lab.git@cua-lite",
]
```

## Design Decisions

**Docker-per-worker.** AndroidLab's Quick Boot snapshot bakes absolute paths (`/root/.android/system-images/...`, `/root/.android/avd/...`) into `snapshot.pb`. On the host those paths belong to root and rewriting them is non-trivial. Running each worker emulator inside its own `cua-lite/androidlab:latest` container makes the paths match natively — the only straightforward way to load the bundled snapshot with all seeded data intact. (`androidworld` uses the same one-container-per-worker shape but for different reasons; see its README.)

**Emulator version pinning.** The snapshot was saved with emulator build 11906825 (34.2.15). Today's `sdkmanager 'emulator'` installs 36.5.10, which rejects 34.x snapshots ("incompatible version"). `cua-lite/androidlab:latest` swaps 34.2.15 in at build time.

**Episode flow.** `reset()` → `AndroidLabContainerFactory.acquire()` spawns a fresh container, boots the emulator, waits for "Successfully loaded snapshot" in `/tmp/emu.log` (up to `server_kwargs.boot_timeout: 600` in [configs/default.yaml](/lite/gym/envs/androidlab/configs/default.yaml), read as `CFG.server_kwargs["boot_timeout"]` — empirically 300–500 s on an idle host). Between episodes on the same env, `reset_snapshot()` hot-reloads `default_boot` via `adb emu avd snapshot load default_boot` (~3-5s) — workers don't tear down containers between episodes.

**`judge()` runs every step.** After each `step()`, `judge()` is called with a `line` dict containing `parsed_action`, `target`, `current_activity`, `window`, and `command` (populated for tasks with `adb_query:` in their YAML). When `judge_page=True`, the result is cached as `_best_judge`. On `terminate`/`response` or `max_steps`, reward = `1.0 if _best_judge.get("complete") else 0.0`. Only binary SR — no sub-SR, RRR, or ROR metrics.

**Action translation.** Each per-step action is a localhost HTTP POST from the host-side `AndroidLabContainer` into the in-container env-server (`POST /env/tap`, `POST /env/swipe_precise`, …); the in-container handler then runs `adb -s emulator-5554 …` directly against the local adb daemon (no `docker exec` on the hot path). The legacy host-side `docker exec adb` path is gone — the in-container RPC avoids the docker daemon's per-call serialization cost, which used to push per-step latency to 30+ s under N≥16 concurrent workers. See [`lite/gym/envs/androidlab/docker/server.py`](/lite/gym/envs/androidlab/docker/server.py) for the full RPC surface.

| CUA-Lite | adb |
|----------|-----|
| `tap` | `input tap X Y` |
| `long_press` | `input swipe X Y X Y <duration_ms>` |
| `swipe` | `input swipe SX SY EX EY 400` |
| `type` | ADBKeyboard broadcast + auto `KEYCODE_ENTER` (matches reference `TextOnlyExecutor`) |
| `system_button` | `input keyevent KEYCODE_{HOME,BACK,ENTER,MENU}` |
| `open_app` | Standalone extra tool: `monkey -p <pkg> -c LAUNCHER` |
| `wait` | blocking `sleep` |
| `terminate`/`response` | Standalone extra tools; end episode |

Coordinates use `[0,1000]` normalized space (`LiteMobileActionSpace`).

**Port allocation.** Uses `lite.gym.utils.backend.ports` (random-offset scan + port-bind-check prune, reservation file `<repo>/.tmp/sandbox-port-reservations.json`). androidlab reserves the `21000-21999` lane. See the port-range map at the top of [`lite/gym/utils/backend/ports.py`](/lite/gym/utils/backend/ports.py) for the full layout across the cua-lite stack. Each container reserves one host port for the in-container HTTP API; emulator console, adb, and gRPC stay container-internal.

**Task IDs.** 138 tasks across 9 apps: Bluecoins (15), Calendar (14), Cantook (12), Clock (27), Contacts (15), Maps.me (15), PiMusic (12), Settings (23), Zoom (5). Tasks are eval-only — instructions reference literal strings like `"John"` and `"12345678"`, so tasks aren't parameterized for RL training. For RL training data use `androidworld:perturb_*` instead.

**Observation text.** Configurable via `observation_text=`:
- `"none"` (default) — screenshot only
- `"a11y_tree:pixel"` / `"a11y_tree:norm"` — compressed UI-Automator XML tree (reference's own text mode), pixel or [0,1000] normalized coords
- `"a11y_list:pixel"` / `"a11y_list:norm"` — flat element list (shorter, compatible with `androidworld` a11y-style agents)

The internal judge path always uses the compressed-XML tree regardless of this setting.

## Verification

```bash
uv run python - <<'PY'
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids('androidlab')
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f'Found {len(task_ids)} tasks')  # expect 138
    # terminate is a standalone extra tool and extra_tools defaults to []
    # (configs/default.yaml) — opt in here so env.step treats it as the
    # terminal channel instead of unsupported feedback.
    env = gym.make('androidlab@setting_0', max_steps=10, extra_tools=['terminate'])
    obs = await env.reset()
    print(obs.text[:200])
    result = await env.step([
        make_tool_call('terminate', {'status': 'failure'}, call_id='call_0000')
    ])
    print(f'reward={result.reward}, best_judge={result.info[\"best_judge\"]}')
    await env.close()

asyncio.run(main())
PY
```
