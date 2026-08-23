See [AGENTS.md](/devs/envs/AGENTS.md) for shared requirements.

**Reference implementation:** `${CUA_LITE_REFERENCES_ROOT}/ScreenSpot-Pro-GUI-Grounding`

**Goal:**
Wrap [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) as a cua-lite gym environment. Single-step click grounding benchmark — agent sees screenshot + instruction, produces a single click, evaluated against ground-truth bounding box.

## Design Decisions

**Data format:** Each annotation entry has `img_filename`, `img_size`, `instruction`, `bbox` (pixels), `platform`, `application`, `group`, `ui_type`.

**Episode flow:** `reset()` → load screenshot + instruction. `step()` → evaluate click against bbox, `terminated=True`.

**Evaluation:** Normalize CUA-Lite coordinates (`[0, 1000]`) and bbox (by `img_size`) to `[0, 1]`. Click inside bbox → `reward = 1.0`, otherwise `0.0`.

**Task IDs:** `screenspot_pro:<annotation_filename>_<index>` (e.g., `screenspot_pro:Word_0`).

**Data dep:** Uses `huggingface_hub.snapshot_download()`. Tests skip based on data availability (no package dep).

## Verification

```bash
uv run python - <<'PY'
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids('screenspot_pro')
    task_ids = [tid for ids in splits.values() for tid in ids]
    print(f'Found {len(task_ids)} tasks')
    env = gym.make(f'screenspot_pro@{task_ids[0]}', max_steps=1)
    obs = await env.reset()
    print(f'Instruction: {obs.text}')
    # grounding.point surface: top-level 'point', NOT the 'computer' wrapper
    # (LiteScreenSpotProEnv.step rejects computer/mobile as an invalid action).
    result = await env.step([
        make_tool_call('point', {'coordinate': [500, 500]}, call_id='call_0000')
    ])
    print(f'Reward: {result.reward}, Terminated: {result.terminated}')
    await env.close()
asyncio.run(main())
PY
```
