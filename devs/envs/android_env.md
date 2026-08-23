See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Reference implementation:** `${CUA_LITE_REFERENCES_ROOT}/android_env`

**Goal:**
Wrap [android_env](https://github.com/deepmind/android_env) (DeepMind's Android RL environment) as a cua-lite gym environment. Multi-step mobile RL — agent interacts with a live Android emulator, executing touch/swipe/type actions and receiving screenshots + rewards.

**Difficulty:** Hard — depends on Android SDK + emulator (AVD), no root access available.

## Setup

Get android_env running (Android SDK, AVD, emulator, ADB) without root. Document steps in `README.md`.

## Design Decisions

**Episode flow:** `reset()` starts/reuses emulator, loads `.textproto` task → `step()` translates CUA-Lite mobile actions to android_env touch/lift actions, accumulates rewards → `close()`.

**Coordinate conversion:** `[0, 1000]` → `[0, 1]` for android_env.

**Task IDs:** Each `.textproto` file → `android_env:<task_id>` (e.g., `android_env:classic_2048`).

**Optional dep:** `android_env` package.

## Verification

```bash
uv run python -c "
import asyncio, lite.gym as gym
async def main():
    splits = gym.registry.task_ids('android_env')
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f'Found {len(task_ids)} tasks')
    env = gym.make('android_env@classic_2048', max_steps=5)
    obs = await env.reset()
    print(f'Instruction: {obs.text}')
    await env.close()
asyncio.run(main())
"
```
