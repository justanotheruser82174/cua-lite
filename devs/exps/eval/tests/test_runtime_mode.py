"""Eval runbook runtime-mode guards.

Official eval run.sh entrypoints must not silently fall back to direct mode.
They source devs/exps/eval/utils/runtime_mode.sh, which requires a reachable
env-server unless EVAL_ALLOW_DIRECT=1 is set explicitly.

Run:
    uv run pytest devs/exps/eval/tests/test_runtime_mode.py -q
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / "devs/exps/eval/utils/runtime_mode.sh"


def test_runtime_mode_helper_requires_server_vars_without_direct_override():
    env = os.environ.copy()
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    env.pop("CUA_LITE_ENV_SERVER_TOKEN", None)
    proc = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail; "
            "EVAL_ENV_ID=screenspot_pro; "
            f"source {HELPER}",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "env-server required for official eval" in proc.stderr


def test_runtime_mode_helper_direct_override_clears_remote_vars():
    proc = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail; "
            "EVAL_ENV_ID=screenspot_pro; "
            "EVAL_ALLOW_DIRECT=1; "
            "CUA_LITE_ENV_SERVER_URL=http://example.invalid; "
            "CUA_LITE_ENV_SERVER_TOKEN=secret; "
            f"source {HELPER}; "
            'test -z "${CUA_LITE_ENV_SERVER_URL:-}"; '
            'test -z "${CUA_LITE_ENV_SERVER_TOKEN:-}"; '
            'test "${CUA_LITE_EVAL_RUNTIME_MODE}" = direct',
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
