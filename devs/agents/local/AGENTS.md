# Local VLM Agent Guidelines

See [shared guidelines](/devs/agents/AGENTS.md) for the full development workflow. This doc covers local-specific setup.

## Prerequisites

- GPU with sufficient VRAM. Check with `nvidia-smi` and select a free GPU via `CUDA_VISIBLE_DEVICES`.
- sglang or HuggingFace transformers backend.

## Files to Create

```
lite/agents/models/<agent>/action_space.py
lite/agents/models/<agent>/adapter.py
lite/agents/models/<agent>/agent.py
tests/agents/models/<agent>/test_<agent>_action_space.py
tests/agents/models/<agent>/test_<agent>_adapter.py
tests/agents/models/<agent>/test_<agent>_agent.py
```

## Launch Entry

Add the agent to the `LOCAL_AGENTS` dict in [`lite/agents/factory.py`](/lite/agents/factory.py). Only model-intrinsic properties live here (`agent_id`, optional `engine_kwargs`, optional `backends`):

```python
LOCAL_AGENTS = {
    "org/MyAgent-7B":  {"agent_id": "myagent"},
    "org/MyAgent-32B": {"agent_id": "myagent", "engine_kwargs": {"tp_size": 4}},
}
```

Per-model `protocol_kwargs` defaults belong on the **adapter/protocol class** via `default_factory=lambda: MyProtocol(full_history_size=4)`, not in `LOCAL_AGENTS`. Per-env overrides belong in `scripts/configs/<agent>/default/<env>.yaml`. CLI `--agent-kwargs` overrides last.

## Verification

```bash
CUDA_VISIBLE_DEVICES=<gpu> uv run python scripts/rollout.py \
    --model-id <AgentName> --env-id <ENV> --head 1 --env-kwargs '{"max_steps": 3}'
```

`--task-id ID` pins a single task; `--head 1` (above) runs the first task as a quick smoke test. Omit both to run all tasks. See **Key flags** below for the full list.

## Stress Test

```bash
CUDA_VISIBLE_DEVICES=<gpu> uv run python scripts/rollout.py \
    --model-id <AgentName> --env-id <ENV> \
    --config-path scripts/configs/<agent>/default/<env>.yaml \
    --sample 128 --concurrency 16 --env-kwargs '{"max_steps": 15}'
```

> **Always pass `--config-path scripts/configs/<agent>/default/<env>.yaml`.** Each agent's rollout config pins the SFT-trained sampling kwargs (`temperature`, `top_p`, `max_new_tokens`), agent-specific protocol kwargs (history window, summary template, etc.), and env-specific defaults. Running without it falls back to `scripts/rollout.py`'s generic `{"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 2048}` that's off-distribution for most agents and shifts the eval result. Browse `scripts/configs/` for the per-agent / per-env YAMLs (`{qwen3_vl, mai_ui, step_gui, ui_tars, ui_tars_15_v1, evocua, claude, gpt}/default/{androidworld, lite.osworld, osworld, webgym, ...}.yaml`).

Key flags (`scripts/rollout.py`):
- `--model-id NAME` — model key from the `LOCAL_AGENTS` dict (**required**; `choices=list(AGENTS)`, no default)
- `--model-path PATH` — override model path
- `--backend sglang|hf` — inference backend (default: `sglang`)
- `--sglang-server-url URL` — connect to a running sglang server instead of auto-starting one (default: `$SGLANG_SERVER_URL`)
- `--env-id ENV` — environment (required)
- `--task-id ID` — pin a single task (sample mode); omit to run all tasks
- `--sample N` — randomly sample N tasks (omit to run all)
- `--head N` — keep first N tasks only
- `--splits S [S ...]` — splits to evaluate (default: all)
- `--prompt-data FILE` — parquet task list (overrides `--head`/`--sample`)
- `--group-size G` — rollouts per task for GRPO variance analysis (default: 1)
- `--concurrency C` — max parallel envs (default: 16)
- `--config-path FILE` — YAML config providing base agent/env kwargs
- `--sampling-kwargs JSON` — sampling params (default: `{"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 2048}`)
- `--engine-kwargs JSON` — sglang Engine overrides, e.g. `'{"tp_size": 4}'`
- `--agent-kwargs JSON` — agent overrides, e.g. `'{"protocol_kwargs": {"full_history_size": 2}}'`
- `--env-kwargs JSON` — env overrides (default: `{}` — the env's own defaults apply)

**Maximize GPU parallelism:** launch rollout runs on different envs across free GPUs concurrently. RL training requires massive parallel sampling, so stress testing should mirror that workload.
