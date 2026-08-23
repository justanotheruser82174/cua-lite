# WindowsAgentArena Developer Smokes

Use this page for action-contract checks that sit between the unit tests and a
full rollout batch. Run these after changing WAA action translation, env-server
routing, or direct/server mode plumbing.

## Non-Live Guard

This check does not boot Windows. It verifies bounded duration coercion for the
WAA translation layer, rejected-action feedback, and config-pair matrix coverage.

```bash
uv run pytest \
  tests/gym/envs/waa/test_waa.py::test_action_translation_coerces_drag_duration \
  tests/gym/envs/waa/test_waa.py::test_action_translation_rejects_bad_model_durations \
  tests/gym/utils/feedback/test_batch_abort_feedback.py::test_waa_capped_hold_key_duration_refuses_only_that_action \
  tests/gym/utils/feedback/test_batch_abort_feedback.py::test_waa_short_coordinate_refuses_only_that_action \
  tests/gym/matrix/test_agent_env_pair_matrix.py \
  -q
```

## Live Action Contract

The live smoke sends one batched `computer` call containing `wait`,
`hold_key`, and `drag`. It should produce three executed actions and no
model-action error result. Keep the duration values small so the check only
proves forwarding and bounds, not task progress.

Direct mode:

```bash
env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
  uv run python - <<'PY'
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    task_id = gym.registry.task_ids("waa", split="eval")[0]
    env = gym.make(f"waa@{task_id}", max_steps=5)
    try:
        await env.reset()
        result = await env.step([
            make_tool_call("computer", {"actions": [
                {"action": "wait", "duration": 0.1},
                {"action": "hold_key", "keys": ["ctrl"], "duration": 0.1},
                {
                    "action": "drag",
                    "start_coordinate": [500, 500],
                    "coordinate": [520, 520],
                    "duration": 0.1,
                },
            ]}, call_id="call_waa_action_smoke")
        ])
        errors = [r.error for r in result.results if (r.metadata or {}).get("is_error")]
        assert not errors, errors
        print(result.info["executed_actions"])
    finally:
        await env.close()

asyncio.run(main())
PY
```

Server mode:

```bash
uv run python scripts/serve_env.py \
  --port 30110 --env-ids waa \
  --token waa-smoke
```

Leave `--max-live-envs` unset for the normal smoke so env-server admission uses
its host-capacity default. Add `--max-live-envs <N>` only for an intentional
constrained-cap repro.

In another shell, run the same Python snippet through the env server:

```bash
HOST_IP=$(hostname -I | awk '{print $1}')
CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30110 \
CUA_LITE_ENV_SERVER_TOKEN=waa-smoke \
  uv run python - <<'PY'
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    task_id = gym.registry.task_ids("waa", split="eval")[0]
    env = gym.make(f"waa@{task_id}", max_steps=5)
    try:
        await env.reset()
        result = await env.step([
            make_tool_call("computer", {"actions": [
                {"action": "wait", "duration": 0.1},
                {"action": "hold_key", "keys": ["ctrl"], "duration": 0.1},
                {
                    "action": "drag",
                    "start_coordinate": [500, 500],
                    "coordinate": [520, 520],
                    "duration": 0.1,
                },
            ]}, call_id="call_waa_action_smoke")
        ])
        errors = [r.error for r in result.results if (r.metadata or {}).get("is_error")]
        assert not errors, errors
        print(result.info["executed_actions"])
    finally:
        await env.close()

asyncio.run(main())
PY
```

Direct/server parity is satisfied when the same task reaches reset, the
executed-action names are `wait`, `hold_key`, and `drag` in order, and neither
mode returns an `is_error` tool result for the smoke call.

## Cursor Rendering Smoke

WAA screenshots do not include a guest cursor, so cursor rendering is
env-owned and controlled by `make_kwargs.cursor`. After changing WAA screenshot
capture, cursor tracking, action translation, image prep, or env-server routing,
run the live action contract twice in direct mode and twice through a fresh
cold env-server:

- `gym.make(f"waa@{task_id}", max_steps=5, cursor=True)`
- `gym.make(f"waa@{task_id}", max_steps=5, cursor=False)`

Use a `mouse_move` or `drag` to place the cursor at a known coordinate, save the
reset and post-step screenshots, and compare a small crop around that coordinate.
The true path must show the shared Linux cursor sprite and the false path must
remain raw. WAA is dedicated/per-trajectory, so no singleton-backed cursor smoke
is required.
