#!/usr/bin/env python3
"""RegionFocus grounding rollout with custom debug logging.

Differences from ``scripts/rollout.py``:
  - pre-registers the RegionFocus adapter+agent entries
  - patches trajectory logging to dump RegionFocus artifacts
  - defaults log_root under ``.logs/mt_rollout``
  - augments top-level summary with self-judge metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from examples.grounding import adapter  # noqa: F401
from examples.grounding.utils.debug_helpers import augment_summary, patch_trajectory_logger
from lite.agents.factory import LOCAL_AGENTS
from lite.infer.rollout import (
    _merge_agent_kwargs,
    check_ci_gates,
    make_rollout_parser,
    run_rollout,
)
from lite.infer.serving import (
    extract_sampling_kwargs,
    make_server_generate_fn,
    start_server,
    stop_server,
)
from lite.utils.config import load_config
from lite.utils.logging import setup_logging

setup_logging()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        parents=[make_rollout_parser()],
        description="Rollout RegionFocus grounding agent on all tasks of an environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        choices=list(LOCAL_AGENTS),
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sampling-kwargs", type=json.loads, default=None)
    parser.add_argument("--engine-kwargs", type=json.loads, default=None)
    parser.add_argument("--sglang-server-url", default=None)
    parser.set_defaults(concurrency=16, save_video=True)
    return parser.parse_args(argv)


async def _main(args: argparse.Namespace) -> None:
    from transformers import AutoProcessor

    model_path = args.model_path or args.model_id
    os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
    os.environ.pop("CUA_LITE_ENV_SERVER_TOKEN", None)
    patch_trajectory_logger()

    file_cfg = load_config(args.config_path) if args.config_path else {}
    agent_kwargs = dict(args.agent_kwargs or {})
    merged_agent_kwargs = _merge_agent_kwargs(
        file_cfg.get("agent_kwargs", {}),
        dict(agent_kwargs),
    )
    sampling_kwargs = extract_sampling_kwargs(
        merged_agent_kwargs,
        args.sampling_kwargs,
    )

    sglang_server_proc = None
    generate_fn = None
    if args.sglang_server_url:
        sglang_server_url = args.sglang_server_url
    else:
        sglang_server_proc, sglang_server_url = await start_server(
            args.model_id, model_path=args.model_path, engine_kwargs=args.engine_kwargs,
        )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    generate_fn = make_server_generate_fn(sglang_server_url, sampling_kwargs)
    merged_agent_kwargs["processor"] = processor
    merged_agent_kwargs["generate_fn"] = generate_fn

    if not args.log_root:
        args.log_root = (
            f".logs/mt_rollout/{args.model_id.replace('/', '_')}/"
            f"{args.env_id}/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )

    try:
        for attempt in range(1, args.max_attempts + 1):
            if attempt > 1:
                print(f"\n=== Retry attempt {attempt}/{args.max_attempts} ===\n")
            all_done, resolved_log_root = await run_rollout(
                model_id=args.model_id,
                model_path=model_path,
                env_id=args.env_id,
                agent_kwargs=merged_agent_kwargs,
                env_kwargs=args.env_kwargs,
                seed=args.seed,
                group_size=args.group_size,
                concurrency=args.concurrency,
                log_root=args.log_root,
                prompt_data=args.prompt_data,
                splits=args.splits,
                head=args.head,
                sample=args.sample,
                filter_expr=args.filter_expr,
                config_path=args.config_path,
                save_data=args.save_data,
                save_video=args.save_video,
                save_gif=args.save_gif,
                debug=args.debug,
                render_instruction_banner=args.render_instruction_banner,
                group_shared_seed=args.group_shared_seed,
            )
            augment_summary(Path(resolved_log_root))
            args.log_root = str(resolved_log_root)
            if all_done:
                print(f"\n=== All tasks complete (attempt {attempt}/{args.max_attempts}) ===\n")
                break
            if attempt == args.max_attempts:
                print(
                    "\n=== Max attempts "
                    f"({args.max_attempts}) reached, some tasks still incomplete ===\n"
                )

        if args.min_valid_frac is not None or args.min_mean_return is not None:
            reasons = check_ci_gates(
                Path(args.log_root) / "summary.json",
                min_valid_frac=args.min_valid_frac,
                min_mean_return=args.min_mean_return,
            )
            if reasons:
                print("\n=== CI gate FAILED ===", file=sys.stderr)
                for r in reasons:
                    print(f"  {r}", file=sys.stderr)
                sys.exit(1)
            print("\n=== CI gates pass ===\n")
    finally:
        if sglang_server_proc is not None:
            await stop_server(sglang_server_proc, generate_fn=generate_fn)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_main(_parse_args(argv)))


if __name__ == "__main__":
    main()
