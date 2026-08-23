See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Reference implementation:** `${CUA_LITE_REFERENCES_ROOT}/android_world`

**Goal:**
Wrap [AndroidWorld](https://github.com/google-research/android_world) (Google's Android benchmark, 116 tasks across 20 apps) as a cua-lite gym environment. Multi-step mobile benchmark with **semantic evaluation** — tasks are evaluated by checking real device state (SQLite, file system, app state) rather than UI matching.

**Difficulty:** Hard — depends on android_env, Android SDK + emulator (AVD), gRPC accessibility forwarding. No root access available.

## Setup

1. Install deps + provision the emulator: `android_env==1.2.3`, `dm_env`, `grpcio-tools`, `protobuf`, `fuzzywuzzy`, etc., plus the Android SDK and an AVD named `AndroidWorldAvd` — see [README.md](/lite/gym/envs/androidworld/README.md) for the concrete setup.
2. Start the emulator (gRPC): `~/Android/Sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554`
3. First run installs 20+ apps via `env_launcher.load_and_setup_env(console_port=5554, adb_path=..., emulator_setup=True)`

## Design Decisions

**Episode flow:** `reset()` → connect to emulator, instantiate `TaskEval` with random params, call `task.initialize_task(env)`. `step()` → translate CUA-Lite actions to AndroidWorld's `json_action` format. Only call `task.is_successful(env)` on the final step (terminate/response or max_steps).

**Action translation.** The CUA-Lite side is exactly the eight actions this env accepts —
re-derive with:

```bash
uv run python -c "from lite.gym.utils.feedback.surface import android_supported_actions; print(sorted(android_supported_actions()))"
```

`drag`, `long_press`, `screenshot`, `swipe`, `system_button`, `tap`, `type`, `wait`.

Names like `click`, `scroll`, `input_text`, `navigate_back`, `navigate_home`, `keyboard_enter` and
`status` are **not** CUA-Lite action names — `tap` covers the first, `system_button` covers the
navigation ones. `double_tap` exists only on the model-emit side (see
`lite.core.tools.action_space.is_action_name_or_action_batch_tool_name`); no CUA-Lite action space declares it.

**Task IDs:** Each task class → `androidworld:<TaskClassName>` (e.g., `androidworld:ContactsAddContact`). Dynamic random params — each `reset()` generates a new instance.

**Optional dep:** `androidworld` package.

**Emulator sharing with `android_env`:** Both envs need a running emulator — keep config (port, ADB path) consistent.

## Verification

```bash
uv run python -c "
import asyncio, lite.gym as gym
async def main():
    splits = gym.registry.task_ids('androidworld')
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f'Found {len(task_ids)} tasks')
    env = gym.make('androidworld@ContactsAddContact', max_steps=10)
    obs = await env.reset()
    print(f'Instruction: {obs.text}')
    await env.close()
asyncio.run(main())
"
```

## Boot readiness gating (current state)

`/init` used to fail under host contention because the in-container `/healthz` only checked uvicorn liveness — not the Android system services (`PackageManager`, `AccessibilityService`, `a11y_grpc` reverse channel) that `env_launcher.load_and_setup_env()` depends on. Two compounding root causes were fixed:

1. **System-service readiness gate.** `AndroidWorldContainer._wait_until_android_ready` (`lite/gym/envs/androidworld/container.py`) polls `pm path com.android.settings` AND `service check accessibility` for up to `server_kwargs.android_ready_timeout: 120` in [configs/default.yaml](/lite/gym/envs/androidworld/configs/default.yaml) — read as `CFG.server_kwargs["android_ready_timeout"]`, **not** an env var — before `/init` may fire. (The only env var the loader reads is `ANDROID_WORLD_CONFIG`, which replaces the whole yaml; see [docs/envs.md#config](/docs/envs.md#config).) A sibling probe in `lite/gym/envs/androidlab/container.py` checks `PackageManager` only — androidlab doesn't need the accessibility channel.

2. **DNS race on a11y APK download.** `a11y_grpc_wrapper._get_accessibility_forwarder_apk()` used to HTTPS-fetch a 4.5 MB APK from `storage.googleapis.com` on every `/init`, saturating slirp4netns's single-thread DNS forwarder under N≥16 concurrent /init storms. The APK is now baked into the docker image at build time, and the wrapper is patched in `lite/gym/envs/androidworld/docker/Dockerfile` to read the local file. Runtime is fully offline for the a11y forwarder install.

### Adjacent (separate) failure mode — KVM group race

```
acquire attempt 1/3 failed: Emulator in <name> can't access /dev/kvm despite --group-add.
Intermittent rootless-docker race; caller should destroy + retry.
```

Fires when rootless docker (`DOCKER_HOST=unix:///run/user/<uid>/docker.sock`) drops `--group-add <kvm-gid>` under concurrent `docker run` storms. Already auto-retried at the acquire layer; not related to the readiness gate above.

---

## History / discovery trail (pre-fix, kept for context)

The original investigation lives below. Numbers and reproduction steps reflect the **pre-fix** state on commit `e2e9300b` (2026-05-26) and will no longer reproduce against current `main`.

### Symptom (pre-fix)

Under host contention (other users running 30+ KVM emulators concurrently on the same shared host), eval rollouts saw large unfinished-task counts:

| Run | host time | eval-set unfinished | env-server `/init 500` count |
|---|---|---|---|
| run 1 (baseline, low load) | 02:48 AM PT | **0 / 86** | ~0 |
| run 3 (same code, low load) | 02:24 AM PT | **1 / 86** | ~0 |
| run 4 (same code, high load) | 07:58 AM PT | **50 / 86** | ~40 |
| run 2 (same code, high load) | 09:14 AM PT | **41 / 86** | ~30 |

Failing rollouts surfaced as `httpx.HTTPStatusError 500` on the client (slime or `scripts/rollout.py`), traced back to `RuntimeError: server /init returned 500: Internal Server Error` in env-server. Reproduced cleanly **without slime**, so slime was ruled out.

### Root cause (pre-fix)

The in-container `/healthz` only checked that uvicorn was up:

```python
# lite/gym/envs/androidworld/docker/server.py — pre-fix /healthz handler
@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "env_ready": _env is not None,          # set by /init only
        "task_ready": _task is not None,
    }
```

The container's emulator-side boot wait DID poll `getprop sys.boot_completed == 1` before logging "emulator booted", but `sys.boot_completed=1` only marks zygote startup — **PackageManager / AccessibilityService / a11y_grpc reverse channel** became available significantly later.

Measured timing on `e2e9300b`, host load 60-95:

| Phase | Median |
|---|---|
| `docker run` → `sys.boot_completed=1` | 33–36 s |
| `sys.boot_completed=1` → env-server calls `/init` | 2–3 s |
| Android system services fully ready under host contention | several extra seconds (variable) |

Under low load these settled within the same `/init` call's first internal adb attempts — under high load they didn't, so `env_launcher.load_and_setup_env()` raised inside the container and the handler returned 500. Not a reused `_env` lifecycle issue: with `server_kwargs.max_resets_per_container: 0` (the shipped default) every episode reset destroys and cold-spawns, so the failing `/init` always landed on a brand-new container.

### Fix directions considered

| Fix | Effort | Eliminates | Status |
|---|---|---|---|
| **A**. Strengthen `/healthz`: probe `service check accessibility` AND `pm path com.android.settings`; extend container.py poll to wait for system services | ~10 lines in `lite/gym/envs/androidworld/docker/server.py` + ~5 lines in container.py | Root cause | **Landed** — `_wait_until_android_ready` |
| **A'**. Bake a11y APK into image, patch `a11y_grpc_wrapper` to read local file | Dockerfile change | Compounding DNS race | **Landed** |
| B. Host-side `/init` 500 → destroy-and-respawn retry wrapper | ~30 lines in `main.py` | Symptom only | Not landed (A + A' were sufficient) |
| C. Hard sleep N seconds after `sys.boot_completed=1` | 1 line | Symptom — wastes time when system is fast | Rejected |
| D. Drop concurrency (32 → 16) | 0 code lines | Reduces race rate, doesn't fix | Rejected (workaround) |
