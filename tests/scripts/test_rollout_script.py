from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rollout.py"


def test_rollout_help_exits_zero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["scripts/rollout.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert exc_info.value.code == 0
