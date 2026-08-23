from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_lite_osworld_filter():
    path = Path(__file__).parents[1] / "filter.py"
    spec = importlib.util.spec_from_file_location("lite_osworld_filter_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trajectory_level_exclude_reason_registry_is_separate():
    flt = _load_lite_osworld_filter()

    for reason in (
        "incomplete",
        "dependency_install",
        "complex_shell",
        "footgun:loop",
        "footgun:undo_storm",
        "footgun:no_submit",
        "reward_vision_disagree",
    ):
        assert flt.validate_trajectory_reason(reason) == reason

    for reason in (
        "upstream_live_site_drift",
        "google_auth",
        "block: live-site redirect",
        "incomplete:any",
        "reward_vision_disagree:any",
        "footgun",
        "footgun:any",
        "footgun_loop",
        "footgun_undo_storm",
        "footgun_no_submit",
        "oob_coordinate",
        "made_up_category",
    ):
        with pytest.raises(ValueError):
            flt.validate_trajectory_reason(reason)
