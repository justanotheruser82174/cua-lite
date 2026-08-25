# OSWorld-G

`--env-id` `osworld_g`

CUA-Lite wrapper for [OSWorld-G](https://github.com/xlang-ai/OSWorld-G) — single-step click grounding via UI decomposition ([Xie et al. 2025](https://arxiv.org/abs/2505.13227)). 564 tasks across 3 box-type modes, via `gym.make("osworld_g@<task_id>")` with `LiteDesktopActionSpace`. See [docs/envs.md](/docs/envs.md) for the env contract.

Box-type modes:

| `box_type` | count | judgement |
|---|---|---|
| `bbox` | 470 | click center ∈ rect (`[x, y, w, h]` → `[x1, y1, x2, y2]`) |
| `polygon` | 40 | click center ∈ polygon (ray-casting) |
| `refusal` | 54 | model must call `report_infeasible` (no element to click) |

## Setup

```bash
# Auto-clones the OSWorld-G tasks from GitHub into .cache/OSWorld-G/ (no HF mirror).
uv run python lite/gym/envs/osworld_g/scripts/utils/download_tasks.py   # or --force to re-clone
```

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` loads tasks in-process (no server):

```python
import asyncio, lite.gym as gym
env = gym.make("osworld_g@<task_id>")
asyncio.run(env.reset())
```

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    env = gym.make("osworld_g@<task_id>")
    result = await env.reset()
    print(result.text)

    result = await env.step([
        make_tool_call("computer", {"actions": [
            {"action": "click", "coordinate": [500, 300]}
        ]}, call_id="call_0000")
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/osworld_g/configs/default.yaml) — `env_kwargs` (per-instance) + `server_kwargs` (per-deployment infra), read via `env_config.load`. Swap the whole file with `OSWORLD_G_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("osworld_g"))   # {"eval": [...]}
```

564 tasks under the `eval` split, across 3 box-type modes. Metadata in `env.metadata.others`: `box_type` (`bbox` / `polygon` / `refusal`), `exclude_reason`, `image_size`, `instruction_style`.

## Two instruction variants

Each task has both `OSWorld-G.json` (original short label) and
`OSWorld-G_refined.json` (longer visually-grounded description). Pick at env
construction time:

```yaml
env_kwargs:
  instruction_style: "original"   # default
  # or "refined" — explains the target's visual context more
```

## Refusal handling

Refusal tasks (`box_type: "refusal"`) want the model to recognize an absent
element. cua-lite exposes `report_infeasible` as an extra tool (mirrors the
OSWorld convention) — calling it on a refusal task scores **1.0**, calling
it on a bbox/polygon task scores **0.0**. Plain `click` is symmetric.

## Filtering by mode

Two metadata fields, by intent:

| field | values | purpose |
|---|---|---|
| `box_type` | `"bbox"` / `"polygon"` / `"refusal"` | descriptive eval-shape label; filter mode-by-mode |
| `exclude_reason` | `"refusal"` (only) | OSWorld-style "skip-by-default" marker — set on refusal tasks only |

Common filters:

| filter | tasks |
|---|---|
| (none) | all 564 |
| `--filter "lambda m: not m.others.get('exclude_reason')"` | bbox + polygon (510) — default grounding |
| `--filter "lambda m: m.others.get('box_type') == 'bbox'"` | bbox only (470) |
| `--filter "lambda m: m.others.get('box_type') == 'refusal'"` | refusal only (54) |

Example skipping refusals (matches OSWorld exclude_reason convention):

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 --env-id osworld_g \
  --filter "lambda m: not m.others.get('exclude_reason')"
```

## Evaluation

Single-step (`terminated=True` after one step), identical scoring shape to `screenspot_pro`: the click is de-normalized from cua-lite `[0, 1000]` to native pixels via `image_size`, then compared to ground truth (bbox / polygon containment, or `report_infeasible` for refusal tasks). `info.executed_actions` records the de-normalized pixel click for comparison with `box_coordinates` in the trajectory log.

`info.executed_actions` example:
```json
[{"call": "click", "args": {"x": 1436, "y": 340}}]
[{"call": "report_infeasible", "args": {"reason": "no such element"}}]
```

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@inproceedings{xie2025osworldg,
  author    = {Tianbao Xie and Jiaqi Deng and Xiaochuan Li and Junlin Yang and Haoyuan Wu and Jixuan Chen and Wenjing Hu and Xinyuan Wang and Yuhui Xu and Zekun Wang and Yiheng Xu and Junli Wang and Doyen Sahoo and Tao Yu and Caiming Xiong},
  title     = {Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis},
  booktitle = {Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025},
  year      = {2025},
  url       = {http://papers.nips.cc/paper\_files/paper/2025/hash/22c868099177ee278eb7baccec649f35-Abstract-Datasets\_and\_Benchmarks\_Track.html}
}
```
