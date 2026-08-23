"""CLI wrapper for exporting Wordle prompt-data parquets.

The canonical exporter reads the gym registry. Wordle lives under ``examples``,
so this wrapper registers the example env first, then delegates to
``lite.train.export.export_tasks`` unchanged. The example env is pure
in-process, so inherited env-server variables are ignored here just like in the
Wordle GRPO wrapper.

Registration is split-aware: the canonical exporter looks tasks up per split, so
tasks registered under ``train`` are invisible to ``--split test``.
"""

from __future__ import annotations

import argparse
import os
import sys

from examples.wordle.env import register_wordle
from lite.train.export.export_tasks import main as _export_tasks_main


def main() -> None:
    os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
    os.environ.pop("CUA_LITE_ENV_SERVER_TOKEN", None)

    if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--split", default="train")
        args, _ = parser.parse_known_args(sys.argv[1:])
        register_wordle(split=args.split)
    _export_tasks_main()


if __name__ == "__main__":
    main()
