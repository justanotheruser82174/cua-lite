#!/usr/bin/env python3
"""Launcher for an sglang server serving a local VLM model.

Starts an OpenAI-compatible inference server that rollout/eval code can connect
to via ``--sglang-server-url``. Sglang launch policy lives here; request
handling lives in sglang itself.

``CUDA_VISIBLE_DEVICES`` controls the visible GPU count; the launcher derives
``dp_size = visible_gpus // tp_size`` and rejects ``tp_size`` larger than the
visible GPU set. ``--port 0`` prints ``PORT=<actual>`` before subprocess launch
for automation; that line is not a readiness signal. The default host is
``0.0.0.0``; use ``--host 127.0.0.1`` for local-only binding.

Usage::
    # 8B model on 2 GPUs (dp_size=2)
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/serve_sglang.py \\
        --model-id Qwen/Qwen3-VL-8B-Instruct

    # 32B model with tp=2 from LOCAL_AGENTS config
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/serve_sglang.py \\
        --model-id Qwen/Qwen3-VL-32B-Instruct

    # Custom port
    uv run python scripts/serve_sglang.py \\
        --model-id Qwen/Qwen3-VL-4B-Instruct --port 30001

    # Override sglang engine kwargs
    uv run python scripts/serve_sglang.py \\
        --model-id Qwen/Qwen3-VL-8B-Instruct \\
        --engine-kwargs '{"tp_size": 2}' --port 30000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shlex
import socket
import subprocess
import sys

from lite.agents.factory import LOCAL_AGENTS
from lite.utils.logging import setup_logging

_SGLANG_BACKEND = "sglang"


def _sglang_model_ids() -> list[str]:
    """Model IDs whose local config permits sglang serving."""
    return [
        model_id
        for model_id, cfg in LOCAL_AGENTS.items()
        if _SGLANG_BACKEND in (cfg.get("backends") or (_SGLANG_BACKEND, "hf"))
    ]


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _find_free_port(start: int = 30000, end: int = 31000) -> int:
    """Return a currently bindable TCP port in ``[start, end)``."""
    ports = list(range(start, end))
    random.shuffle(ports)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range [{start}, {end})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an sglang server for a local VLM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        required=True,
        choices=_sglang_model_ids(),
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model path, for example a local checkpoint.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help=(
            "Server port. Use 0 to auto-pick a free port; the launcher prints "
            "PORT=<actual> before starting sglang for automation."
        ),
    )
    parser.add_argument(
        "--engine-kwargs",
        type=_json_object,
        default=None,
        help=(
            "JSON sglang overrides, for example '{\"tp_size\": 4}'. "
            "dp_size is derived from visible GPUs."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Build sglang launch config, print PORT, and hand off to subprocess."""
    setup_logging()
    logger = logging.getLogger("cua-lite.sglang-server")

    cfg = LOCAL_AGENTS[args.model_id]
    allowed_backends = cfg.get("backends") or (_SGLANG_BACKEND, "hf")
    if _SGLANG_BACKEND not in allowed_backends:
        raise SystemExit(
            f"Model '{args.model_id}' does not support sglang; "
            f"allowed backends: {list(allowed_backends)}"
        )
    model_path = args.model_path or args.model_id
    engine_kwargs = {**cfg.get("engine_kwargs", {}), **(args.engine_kwargs or {})}
    tp_size = engine_kwargs.pop("tp_size", 1)
    if "dp_size" in engine_kwargs:
        raise SystemExit("dp_size is derived from visible GPUs; set tp_size instead")
    if not isinstance(tp_size, int) or tp_size < 1:
        raise SystemExit("tp_size must be a positive integer")

    # Auto dp_size: number of visible GPUs divided by tensor parallelism.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_visible = len(cvd.split(",")) if cvd else 1
    if tp_size > num_visible:
        raise SystemExit(
            f"tp_size={tp_size} exceeds visible GPUs ({num_visible}); "
            "reduce tp_size or expose more GPUs."
        )
    dp_size = max(num_visible // tp_size, 1)

    # Auto-pick port if --port 0.
    port = args.port if args.port != 0 else _find_free_port()

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        args.host,
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
        "--dp-size",
        str(dp_size),
    ]

    # Pass remaining engine_kwargs using sglang's argparse conventions.
    for key, value in engine_kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, (list, tuple)):
            cmd.append(flag)
            cmd.extend(str(item) for item in value)
        else:
            cmd.extend([flag, str(value)])

    logger.info(
        "sglang server: model=%s host=%s port=%d tp_size=%d dp_size=%d visible_gpus=%d",
        model_path,
        args.host,
        port,
        tp_size,
        dp_size,
        num_visible,
    )
    logger.info("sglang command: %s", shlex.join(cmd))
    print(f"PORT={port}", flush=True)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(_parse_args())
