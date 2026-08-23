See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Goal:**
Wrap [OSWorld](https://github.com/xlang-ai/OSWorld) (369 real-world desktop tasks across Ubuntu) as a cua-lite gym environment with `LiteDesktopActionSpace`.

## Design Decisions

**Episode flow:** `reset()` → launch VM container, load task config, take initial screenshot. `step()` → execute desktop actions, take screenshot, evaluate on termination.

**Evaluation:** OSWorld's built-in evaluators (file-based, state-based, etc.) are called on the last step. `reward = 1.0` if the evaluator passes, `0.0` otherwise.

**Task IDs:** `osworld@<example_id>` (e.g., `osworld@94d95f96-9699-4208-98ba-3c3119edf9c2`).

**Container management:** All DesktopEnv calls are blocking, run in a ThreadPoolExecutor. `reset()` cancellation is leak-safe — orphaned containers are cleaned up by a daemon thread.

**Data dep:** Uses OSWorld's `evaluation_examples/` directory. Requires `desktop_env` package (`uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh`).

## Verification

```bash
uv run python -c "
import asyncio, lite.gym as gym
async def main():
    splits = gym.registry.task_ids('osworld')
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f'Found {len(task_ids)} tasks')
    env = gym.make(f'osworld@{task_ids[0]}', max_steps=3)
    obs = await env.reset()
    print(f'Instruction: {obs.text}')
    result = await env.step([
        {
            'id': 'call_0000',
            'type': 'function',
            'function': {
                'name': 'computer',
                'arguments': {
                    'actions': [{'action': 'click', 'coordinate': [500, 500]}],
                },
            },
        },
    ])
    print(f'Reward: {result.reward}, Terminated: {result.terminated}')
    await env.close()
asyncio.run(main())
"
```
