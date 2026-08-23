"""CLI wrapper for exporting Geo3K prompt-data parquets.

The canonical exporter reads the gym registry. Geo3K lives under ``examples``,
so this wrapper imports the example registration module first, then delegates to
``lite.train.export.export_tasks`` unchanged. The example env is pure in-process,
so inherited env-server variables are ignored here just like in the Geo3K GRPO
wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from examples.geo3k.env import geo3k_source_fingerprint, register_geo3k
from lite.train.export.export_tasks import main as _export_tasks_main


def _replace_env_kwargs_arg(argv: list[str], env_kwargs: dict[str, object]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--env-kwargs":
            skip_next = True
            continue
        if arg.startswith("--env-kwargs="):
            continue
        cleaned.append(arg)
    return [
        *cleaned,
        "--env-kwargs",
        json.dumps(env_kwargs, sort_keys=True),
    ]


def _prepare_export_argv(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split", default="train")
    parser.add_argument("--env-kwargs", type=json.loads, default=None)
    args, _ = parser.parse_known_args(argv)

    source = os.environ.get("GEO3K_SOURCE")
    register_geo3k(source_path=source, split=args.split)
    if not source:
        return argv

    env_kwargs = dict(args.env_kwargs or {})
    fingerprint = geo3k_source_fingerprint(source)
    existing = env_kwargs.get("source_fingerprint")
    if existing is not None and existing != fingerprint:
        raise SystemExit(
            "GEO3K_SOURCE fingerprint disagrees with --env-kwargs.source_fingerprint"
        )
    env_kwargs["source_fingerprint"] = fingerprint
    return _replace_env_kwargs_arg(argv, env_kwargs)


def main() -> None:
    os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
    os.environ.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

    if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        sys.argv = [sys.argv[0], *_prepare_export_argv(sys.argv[1:])]
    _export_tasks_main()

if __name__ == "__main__":
    main()
