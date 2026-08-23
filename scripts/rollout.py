#!/usr/bin/env python3
r"""Run an agent (local or API) on an env's tasks — the one inference entrypoint.

``--task-id T`` runs a single task (sample mode); omit it to run all tasks
(rollout mode). ``--model-id`` auto-routes local (sglang/hf) vs API (gpt/claude).

Config precedence (highest → lowest): prompt_data > CLI args > yaml.

Usage:
    # Single task (quick sample), local model, auto-start sglang
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/rollout.py \
        --model-id Qwen/Qwen3-VL-4B-Instruct --env-id osworld --task-id <task_id>

    # All tasks (rollout), connect to a running sglang server + config
    uv run python scripts/rollout.py --model-id Qwen/Qwen3-VL-4B-Instruct --env-id webgym \
        --sglang-server-url http://localhost:30002 \
        --config-path scripts/configs/qwen3_vl/default/webgym.yaml

    # API model (gpt/claude — no server)
    uv run python scripts/rollout.py --model-id claude-opus-4-6 --env-id osworld \
        --api-kwargs '{"max_tokens": 4096}'

    # hf backend (in-process, no server)
    uv run python scripts/rollout.py --model-id <id> --env-id <env> --backend hf

    # Multi-GPU data parallelism: auto-started sglang reads dp/tp from LOCAL_AGENTS
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/rollout.py \
        --model-id Qwen/Qwen3-VL-4B-Instruct --env-id osworld

    # Resume a previous run (re-runs only missing samples)
    uv run python scripts/rollout.py --model-id Qwen/Qwen3-VL-8B-Instruct --env-id osworld \
        --log-root .logs/rollout/Qwen_Qwen3-VL-8B-Instruct/osworld/20260325_145829
"""
from __future__ import annotations

import asyncio

from lite.infer.cli import make_infer_parser, run_infer
from lite.utils.logging import setup_logging

setup_logging()

if __name__ == "__main__":
    asyncio.run(run_infer(make_infer_parser().parse_args()))
