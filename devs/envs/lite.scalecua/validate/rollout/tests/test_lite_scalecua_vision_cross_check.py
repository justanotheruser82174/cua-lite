"""Tests for the ScaleCUA rollout vision cross-check helper.

Run:
    uv run pytest \
      devs/envs/lite.scalecua/validate/rollout/tests/test_lite_scalecua_vision_cross_check.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_HERE = Path(__file__).resolve().parents[1]


def _load_vision_cross_check() -> ModuleType:
    path = _HERE / "vision_cross_check.py"
    spec = importlib.util.spec_from_file_location("scalecua_vision_cross_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalecua_vision_cross_check_uses_shared_metadata_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cross = _load_vision_cross_check()
    monkeypatch.setattr(cross, "_traj_details", lambda *_args: ("task", ["frame.png"]))
    monkeypatch.setattr(cross, "_judge", lambda *_args: (True, "visually done"))

    out = cross._process(
        {
            "trajectory_path": str(tmp_path / "missing.parquet"),
            "frame_paths": {},
            "reported_reward": 0.0,
        },
        config=None,
    )

    assert out["vision_done"] is True
    assert out["disagree"] is True
