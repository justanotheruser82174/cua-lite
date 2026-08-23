# Shared Guidelines for Adding New Environments

This document defines the shared requirements for all environments under `lite/gym/envs/`. Environment-specific details go in each env's own spec file (e.g., `android_env.md`).

## Worktree-Based Development

All new env development should happen in a **git worktree** on a dedicated branch:

```bash
git worktree add ../cua-lite-<env_id> <env_id>
cd ../cua-lite-<env_id>
git submodule update --init
uv sync --extra quick-start  # then run per-env scripts/install.sh
```

Do NOT modify the main working directory. Perform all edits, commits, and test runs inside the worktree.

## Development Steps

### 1. Implement

Create the required files under `lite/gym/envs/<env_id>/`. See [File Structure](#file-structure) and [Coding Guidelines](#coding-guidelines) below for requirements.

### 2. Test

Add test cases under `tests/gym/envs/<env_id>/test_<env_id>_*.py`, or the nearest
existing env-owner subdirectory. See [Test Cases](#test-cases) below for
requirements.

### 3. Verification

Run a quick sanity check with small `max_steps`:

```bash
CUDA_VISIBLE_DEVICES=<gpu> uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3-VL-8B-Instruct --env-id <ENV> --head 1 --env-kwargs '{"max_steps": 3}'
```

Inspect logs and verify observations, action dispatch, and reward computation per the env's spec file.

#### 3.1 Log Validation

Inspect per-turn artifacts under `<log-root>/.../sample_NN/turn_NNNN/`:

- `prompt_images/*.png` — optional debug prompt-image cache; absent when debug artifacts were disabled
- `prompt_images_annotated/*.png` — action overlays with the same basename as the prompt image they annotate; written only for turns with coordinate actions, and only with debug artifacts on
- `01_prompt.txt` — full prompt sent to model
- `02_response.txt` — raw model output
- `03_actions.json` — agent_message, lite_message, executed_actions
- `04_results.json` — step result: reward, terminated, truncated, results, info
- `05_timing.json` — per-turn timing data; absent when the turn recorded no timings

Canonical trajectory images live at `sample_NN/images/*.png` and are referenced
from `trajectory.parquet`; visual result images in `04_results.json` must point
back to that store with `image_index`. Legacy readers may accept per-turn
`images/`, `annotated/`, `05_results.json`, `06_timing.json`, or
`result_images/`, but do not treat those names as current layout.

When a batch is used as a reference log, pair by config path, env id, agent
family, runtime mode, and task id. Keep the complete log root and sample
directory; `summary.json` alone is only a score aggregate.

### 4. Stress Test

Run parallel evaluation through a dedicated env-server owned by the coordinator.
Direct rollout mode is only for small smoke checks; full stress batches must use
env-server so reset and container lifecycle behavior matches production
concurrency.

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run python scripts/serve_env.py \
  --port <PORT> \
  --env-ids <ENV> \
  --token <TOKEN> \
  --idle-ttl-sec 900
```

Do not set `--max-live-envs` by default. The server auto-derives an admission
cap from host capacity; pass `--max-live-envs <N>` only as an advanced override
for a constrained repro or tenant-specific cap, and record the reason in the
validation notes.

```bash
HOST_IP=$(hostname -I | awk '{print $1}')
CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:<PORT> \
CUA_LITE_ENV_SERVER_TOKEN=<TOKEN> \
CUDA_VISIBLE_DEVICES=<gpu> uv run python scripts/rollout.py \
    --model-id Qwen/Qwen3-VL-8B-Instruct --env-id <ENV> \
    --config-path scripts/configs/qwen3_vl/default/<env>.yaml \
    --sample 128 --concurrency 32 --env-kwargs '{"max_steps": 15}'
```

> **Always pass `--config-path scripts/configs/qwen3_vl/default/<env>.yaml`.** The rollout config pins the SFT-trained sampling kwargs and agent / env defaults; without it, rollout falls back to `scripts/rollout.py`'s generic `{"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 2048}` that's off-distribution and shifts the eval result. For non-Qwen3-VL agents, swap to the matching `scripts/configs/<agent>/default/<env>.yaml`.

Review all logs and screenshots following the sub-steps below. Fix any errors, warnings, or action space conversion gaps that surface.

#### 4.1 General Notes

- **Coordinator/subagent boundary:** The coordinator owns persistent repo edits, commits, shared artifact regeneration, env-server lifecycle, and rerun decisions. Subagents may audit completed logs, inspect artifacts, run probes/replays, and return fix proposals in parallel; they must not start env-servers, write persistent files, or claim a fix landed before the coordinator applies and verifies it.
- **Batch gate discipline:** Do not mark a stress batch complete until the env-server run has finished cleanly and the blocking checks in [Action Space Verification](#42-action-space-verification) pass for every audited image-bearing turn.
- **Env-specific verification:** No deadlocks, file-lock contention, or resource leaks under concurrency. Port allocation, emulator pool, and Docker container management work correctly at scale. Memory and GPU usage remain stable across all parallel instances. All instances produce valid observations and clean up on close.
- **Scope of fixes:** Env-interface problems are fixed in the owning env or shared env helpers first. From the
  agent's view, every env must obey the same low-level tool-result model: parsed invalid/unsupported/execution
  failures return paired observation plus `LiteToolResult.error` from `env.step()` when the env can observe.
  Touch adapter/action-space code only when that layer's parsing or canonicalization is itself wrong; do not
  add provider-specific harness workarounds, wrapper caches, or one-off monkey patches for env behavior.
  RL training requires massive parallel sampling, so any issue that surfaces during parallel runs is worth
  solving at the correct ownership layer.
- **Counter reset rule:** When a task fails, fix the issue and re-run the exact failed case(s) to confirm the fix. Do not simply start a fresh batch of random tasks — the previously failed cases must pass after the fix before continuing.
- **Zombie cleanup rule:** Periodically clean up zombie or dead processes (sglang, envs, containers, emulators, etc.) between runs. Lingering processes waste resources and can cause port conflicts or OOM in subsequent tasks.

#### 4.1.1 Cursor Rendering Verification

Cursor rendering is env-owned capture behavior, configured with
`make_kwargs.cursor`, not a registry wrapper or retired cursor wrapper key.
When a change touches screenshot capture, cursor state, action dispatch, env
registration, remote bytes, or browser/CUA/WAA images, run a cursor smoke before
claiming the lane complete.

Required matrix for cursor-capable envs:

- `cursor=True` and `cursor=False` in direct mode (`CUA_LITE_ENV_SERVER_URL` /
  `CUA_LITE_ENV_SERVER_TOKEN` unset).
- `cursor=True` and `cursor=False` through a cold env-server started after the
  source/image change, without `--warm-singleton`.
- For shared-backend envs, repeat through a singleton-backed server
  (`scripts/serve_env.py --env-ids <ENV> --warm-singleton`) after the backend
  reports available. This applies to BrowserGym shared stacks, WebGym,
  WebHarbor, Online-Mind2Web, and MobileGym-style singleton services.
  Dedicated/per-trajectory envs such as WAA and CUA local modes stop at direct +
  cold-server.

Required evidence:

- Rebuild or validate image freshness before the smoke (`install.sh status`, and
  `install.sh rebuild` when the Docker patch/source changed).
- Capture reset and one post-`mouse_move`/`click` screenshot for both
  `cursor=True` and `cursor=False`.
- Compare a small pixel crop around the expected cursor position: `cursor=True`
  must show the shared Linux cursor sprite, `cursor=False` must not show any
  synthetic cursor, and neither path may use synthetic point markers.
- Record whether the screenshot bytes were produced by the env/server. Remote
  clients must not repaint or mutate image bytes after decoding the server frame.
- For raw/no-cursor envs such as CAPTCHA and static grounding screenshots,
  explicitly record that no cursor make kwarg is supported and no cursor
  compositing path ran.

#### 4.2 Action Space Verification

> **NON-NEGOTIABLE — this is a blocking gate.** No batch is "done" until all three checks below pass for every turn in that batch. "Looks fine" without opening the actual artifacts does not count. Skipping this step lets action conversion bugs silently corrupt RL training data and waste GPU hours.
> **Parallelism:** Use subagents to analyze completed logs in parallel, but do not launch the next batch until the blocking checks for the current batch pass.

For every image-bearing debug turn, open matching `prompt_images/<name>.png` and `prompt_images_annotated/<name>.png` pairs when present and verify:

1. **Visual correctness** — each action produces the expected on-screen effect (scroll direction/units, drag vectors, key combos, click targets, etc.)
2. **Action conversion chain** — open `03_actions.json` to trace each action through `agent_message → lite_message → executed_actions`. Verify both legs of the conversion:

   **Leg 1 — Adapter output normalizes to CUA-Lite spec** (`agent_message → lite_message`):
   - Inspect `lite_message`: canonical stored key actions carry `keys: list[str]`; raw model strings are ingress only. Every field matches the type declared in `LiteDesktopActionSpace` / `LiteBrowserActionSpace` / `LiteMobileActionSpace` (e.g., `coordinate` must be `list[int]` within `[0, 1000]`; `amount` must be `int > 0`).
   - Watch for models outputting shorthand that the adapter passes through unchecked. Known edge cases:
     - `key(keys="enter")` — model outputs string instead of list; the shared action-space owner must normalize it to `["enter"]`.
     - `key(keys="ctrl+F")` — combo key as single string; the shared action-space owner must split on `+` and normalize casing → `["ctrl", "f"]`.
     - `key(keys="ctrl++")` — trailing literal plus key must normalize through the shared action-space owner to `["ctrl", "+"]`.
     - `key(keys=["ctrl+F"])` is not a canonical shortcut form; reject it or fix
       it at the raw source boundary instead of treating lists as chord strings.
   - No extra or missing required fields.

   **Leg 2 — CUA-Lite action maps to correct env command** (`lite_message → env execution`):
   - Units are consistent across the boundary (e.g., CUA-Lite `scroll.amount` is in wheel-click units; if the env API expects pixels, the env must convert using the correct factor).
   - Semantic intent is preserved (e.g., `type(text)` should only type text — it must not implicitly press Enter if the agent controls Enter via a separate `key(["enter"])` call).
   - All CUA-Lite parameters are forwarded (e.g., `scroll.amount` must not be silently dropped, `click.button` / `click.clicks` must map to the correct env-side variant).
   - Coordinate systems are converted correctly (`[0, 1000]` normalized → env-native pixel coordinates).

   If the on-screen result doesn't match intent, debug through both the adapter and the env implementation (either may contain bugs).
3. **Multi-action parsing** — when the model outputs multiple GUI actions in a single turn, they stay nested under one top-level `computer` / `mobile` wrapper in `arguments.actions[]`, in order. The env-side unpacking path expands them into the individual backend `executed_actions`.

### 5. Docs and Deps

- Add the new env to the "Available Environments" table in `/docs/envs.md`.
- Add any env-specific Python packages to `[project.optional-dependencies] envs` in `/pyproject.toml`.

### 6. Commit

Commit on the env's branch inside the worktree. Do NOT merge into `zzh`.

---

## File Structure

Every environment must have:

```
lite/gym/envs/<env_id>/
├── __init__.py
├── main.py          # env implementation + task registration
├── README.md        # setup instructions + usage examples
└── utils/           # (optional) download scripts, helpers
```

- `main.py` is the entry point — the registry auto-discovers and imports it via `lite.gym.envs.<env_id>.main`.
- `README.md` should contain prerequisites, setup steps, a quick usage example so others can reproduce the setup, and solutions to common pitfalls.

## Coding Guidelines

### Registry and Lazy Import

The registry (`lite/gym/registry.py`) uses **lazy import**: each env's `main.py` is only imported when first needed (via `gym.make("env_id@...")` or `gym.registry.task_ids("env_id")`).

- `_import_env(name)`: imports a single env; missing optional deps should surface as `EnvDepsMissingError` with an install hint. Plain `ImportError` is reserved for genuine import bugs.
- `_import_all()`: imports all envs for listing; catches per-env failures for best-effort catalog probes.

This means:
- `gym.registry.env_ids()` is a cheap directory scan and does not import envs.
- `gym.registry.registered_env_ids()` is the import-aware best-effort list.
- `gym.make("env_id@task_id")` raises the env's typed setup/dependency error when required packages or assets are missing.

### Optional Dependencies — Fail Fast on Use

If an env depends on an optional package or external runtime setup (e.g., `android_env`, `androidworld`, Docker image assets), **raise `EnvDepsMissingError`** at the use boundary with `what`, `install`, and `see` fields. Do NOT silently degrade or register partial tasks.

```python
# In main.py — in the prereq probe used by make/reset/envserver setup
from lite.gym.errors import EnvDepsMissingError

try:
    from android_world import registry as aw_registry
    ...
except ImportError as e:
    raise EnvDepsMissingError(
        what="androidworld package not found",
        install="uv run --no-sync bash lite/gym/envs/<env>/scripts/install.sh",
        see="lite/gym/envs/androidworld/README.md",
    ) from e
```

Do NOT use `warnings.warn` + fallback for missing required packages. The user should get a clear error when they try to use an env without its deps installed.

### Error Messages — Be Friendly and Actionable

Every user-facing error must tell the user **what went wrong**, **how to fix it**, and **where to find more details** (README.md). This applies to:

- `EnvDepsMissingError` for missing packages, task data, Docker images, or services
- `EnvDepsMissingError` for a service that is not running (a setup problem the user must fix)
- `CapacityExhausted.warming(...)` for a service that IS running but not ready yet (transient,
  `retryable=True`, HTTP 503) — never a bare `ConnectionError`; see below
- Warnings for missing data or misconfigured evaluators
- Any place that could silently fail (e.g., returning `0.0` reward when the evaluator is broken)
  — and the fix there is to **not return one**; see *Never coalesce a non-verdict* below

```python
# Good: a TYPED error, actionable message, README reference
raise EnvDepsMissingError(
    what=f"Cannot connect to server at {url}.",
    install="Start the server first: ...",
    see="lite/gym/envs/<env_id>/README.md",
)

# Bad: raw exception with no guidance
r.raise_for_status()  # httpx.ConnectError: Connection refused

# Bad: a friendly message on an UNTYPED error. `lite/gym/errors.py` forbids a bare
# `RuntimeError`/`ConnectionError` on a served surface: the server maps an untyped
# exception to a TERMINAL HTTP 500, so the caller never retries a transient failure.
raise ConnectionError(f"Cannot connect to server at {url}. Start it first: ...")
```

**Never coalesce a non-verdict into a reward.** An evaluator that could not answer —
misconfigured VLM judge, rate-limit after retries, timeout, auth failure — must
**propagate**. Do not substitute `reward=0.0` and log a `WARNING`: a warning scrolls
past, the reward does not, and a fabricated `0.0` poisons RL data with a label the
checker never produced. Let the terminal step ERR and let the rollout retry it. The
three browser envs all implement exactly this — see
`lite/gym/envs/online_mind2web/main.py::_evaluate` (*"We must never coalesce a
non-verdict into reward=0.0: that silently poisons RL data"*), and `webgym` /
`webharbor.webvoyager`, which re-raise a judge runtime error for the same reason.
A verdict the evaluator genuinely reached — "the agent gave no answer" — is a
different thing and may of course be scored.

Dependency and config problems the user can fix still belong at **startup**, as an
`EnvDepsMissingError` with the install command (above), not as a warning attached to
a substituted value.

### Data and Service Setup

Task data (images, annotations, APKs, textprotos, Docker images, browser assets, etc.) belongs in the env's `scripts/install.sh` / README setup flow, not in silent module-import downloads. Import-time side effects make registry probes slow and hard to reproduce. If required data is absent when the env is used, raise `EnvDepsMissingError` with the exact install command and README path.

Optionally provide a standalone download script under `utils/download_tasks.py` for manual pre-download.

### Subclassing

- Directly subclass `LiteBaseEnv` (from `lite/gym/base.py`).
- Do NOT use `SandboxBaseEnv` (which wraps Docker/Computer) unless the env specifically needs Docker.

### Metadata

Every env must return `LiteBaseMetadata`. CUA envs use
`LiteCUAMetadata(dims=(platform, task_type))`, where `platform` is
`"desktop"`, `"browser"`, or `"mobile"` and `task_type` is a concrete current
literal: `"use"`, `"grounding.action"`, `"grounding.point"`,
`"grounding.bbox"`, or `"understanding"`. Generic envs may use
`LiteGenericMetadata(dims=...)` or empty dims. Routing uses
`compose_key(agent_id, *metadata.dims)`; never collapse CUA grounding tasks to a
generic `"grounding"` dimension.

The task instruction is **not** stored in `LiteBaseMetadata`. Instead, `reset()`
returns `LiteEnvObservation(image, text, metadata)`: `text` is the task
instruction and `image` is the initial screenshot as raw PNG bytes. `step()`
returns `LiteEnvStepResult(results, reward, terminated, truncated, info)`, with
one `LiteToolResult` per executed or classified tool call in `results`, paired
by the result's `tool_call_id` to the tool call's `id`, including terminal and
max-step calls when a pairable call id exists.

## Test Cases

Tests go under `tests/gym/envs/<env_id>/test_<env_id>_*.py`, or the nearest
existing env-owner subdirectory.

### Skip tests when optional deps are missing

If the env depends on an optional package, use `pytest.importorskip` at the top of the test file so the entire file is skipped when the package isn't installed:

```python
import pytest

pytest.importorskip("android_env", reason="android_env not installed (run per-env scripts/install.sh)")
```

If the env depends on downloaded data instead of a package (e.g., `screenspot_pro`), skip based on data availability:

```python
_has_data = _DATA_DIR is not None
pytestmark = pytest.mark.skipif(not _has_data, reason="ScreenSpot-Pro data not available")
```

### Testing without a real device/emulator

Envs that require external infrastructure (emulators, Docker) should support a `use_fake=True` config for testing. Tests create fake envs **directly via the factory function**, not through the registry:

```python
from lite.gym.envs.<env_id>.main import <Config>, <EnvClass>, _make_env

def _make_fake(max_steps=50):
    return _make_env(config=<Config>(use_fake=True, ...), max_steps=max_steps)
```

Do NOT register fake tasks in the registry (no `register(id="env:fake", ...)`).

### Minimum test coverage

- `reset()` returns a valid raw-PNG-bytes screenshot
- `step()` accepts the env's action space (tap, click, swipe, etc.)
- Truncation at `max_steps`
- `close()` cleans up without errors
- Task registration and listing (if applicable)
