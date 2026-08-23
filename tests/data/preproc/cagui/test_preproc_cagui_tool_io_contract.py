"""CAGUI preproc tool/result contract tests."""

from __future__ import annotations

import json

import pytest
from data.preproc._tool_io_helpers import (
    _actions,
    _all_calls,
    _assert_first_action_result_is_tool,
    _assert_structural_done_row,
    _assert_terminate_outcome,
)

from lite.core.tools.calls import tool_call_name
from lite.data.preproc.cagui import use as cagui_use
from lite.data.utils.rows import validate_canonical_rows


def test_cagui_post_action_screenshot_is_tool(monkeypatch):
    monkeypatch.setattr(cagui_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = cagui_use.build_episode(
        [
            {
                "episode_id": "ep",
                "step_id": 0,
                "instruction": "finish",
                "image_path": "domestic/ep/0.jpeg",
                "image_width": 1000,
                "image_height": 1000,
                "result_action_type": 4,
                "result_touch_yx": "[0.5, 0.25]",
                "result_lift_yx": "[0.5, 0.25]",
                "ui_positions": "[]",
            },
            {
                "episode_id": "ep",
                "step_id": 1,
                "instruction": "finish",
                "image_path": "domestic/ep/1.jpeg",
                "image_width": 1000,
                "image_height": 1000,
                "result_action_type": 4,
                "result_touch_yx": "[0.5, 0.75]",
                "result_lift_yx": "[0.5, 0.75]",
                "ui_positions": "[]",
            },
        ],
        "OpenBMB/CAGUI/CAGUI_agent",
    )
    _assert_first_action_result_is_tool(row)


def test_cagui_ui_positions_skip_bad_boxes_without_dropping_good_boxes() -> None:
    boxes = json.dumps([
        [0.1, 0.2, 0.3, 0.4],
        None,
        ["bad", 0.2, 0.3, 0.4],
        [0.1, "nan", 0.3, 0.4],
        [0.1, "1e400", 0.3, 0.4],
        [0.5, 0.6, 0.1, 0.2],
    ])

    assert cagui_use._convert_ui_positions(boxes) == [
        [200, 100, 600, 400],
        [600, 500, 800, 600],
    ]


def _cagui_step(step_id: int, action_type: int, x: float = 0.5) -> dict:
    return {
        "episode_id": "ep",
        "step_id": step_id,
        "instruction": "finish",
        "image_path": f"domestic/ep/{step_id}.jpeg",
        "image_width": 1000,
        "image_height": 1000,
        "result_action_type": action_type,
        "duration": 1000.0,
        "result_touch_yx": f"[0.5, {x}]",
        "result_lift_yx": f"[0.5, {x}]",
        "ui_positions": "[]",
    }


def test_cagui_status_task_complete_becomes_done(monkeypatch):
    """Code 10 (STATUS_TASK_COMPLETE) is structural → ``Done.``, no terminate."""
    monkeypatch.setattr(cagui_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = cagui_use.build_episode(
        [_cagui_step(0, 4, 0.25), _cagui_step(1, 4, 0.75), _cagui_step(2, 10)],
        "OpenBMB/CAGUI/CAGUI_agent",
    )

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile", "mobile"]
    # The terminal step's image closes the last real action's role:tool result.
    assert len(row["images"]) == 3


def test_cagui_rejects_steps_after_terminal(monkeypatch):
    monkeypatch.setattr(cagui_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    with pytest.raises(cagui_use.SkipEpisodeError, match="steps after terminal"):
        cagui_use.build_episode(
            [_cagui_step(0, 4), _cagui_step(1, 10), _cagui_step(2, 4)],
            "OpenBMB/CAGUI/CAGUI_agent",
        )


@pytest.mark.parametrize("code", [0, 1])
def test_cagui_rejects_malformed_duration(code: int) -> None:
    step = _cagui_step(0, code)
    step["duration"] = "not-a-duration"
    with pytest.raises(cagui_use.SkipEpisodeError, match="invalid duration"):
        cagui_use.step_to_tool_calls(step)


def test_cagui_episode_loader_fails_loud_on_malformed_annotation(tmp_path):
    episode_dir = tmp_path / "episodes" / "broken"
    episode_dir.mkdir(parents=True)
    (episode_dir / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="broken.json"):
        list(cagui_use.iter_episodes({"episode_glob": "episodes/*"}, str(tmp_path), None))


def test_cagui_status_task_impossible_moves_to_metadata_others(monkeypatch):
    """Code 11 (STATUS_TASK_IMPOSSIBLE) asserts the demonstration failed.

    That label is not recoverable from a ``Done.`` final, so it moves to
    ``metadata.others``; the row itself ends exactly like code 10's.
    """
    monkeypatch.setattr(cagui_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = cagui_use.build_episode(
        [_cagui_step(0, 4, 0.25), _cagui_step(1, 4, 0.75), _cagui_step(2, 11)],
        "OpenBMB/CAGUI/CAGUI_agent",
    )

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile", "mobile"]
    # cagui's terminator carries no authored text, so only the status survives.
    _assert_terminate_outcome(row, status="failure")
    validate_canonical_rows([row], "cagui")


def test_cagui_rejects_touch_oob_before_normalization_clamp() -> None:
    with pytest.raises(cagui_use.SkipEpisodeError, match="outside \\[0, 1\\]"):
        cagui_use.step_to_tool_calls({
            "episode_id": "ep",
            "step_id": 0,
            "result_action_type": 4,
            "result_touch_yx": "[0.5, 1.2]",
            "result_lift_yx": "[0.5, 1.2]",
        })

    assert cagui_use._norm_yx_to_xy_1000(1 + 1e-7, -1e-7) == [0, 1000]


def test_cagui_wait_and_long_press_preserve_source_duration() -> None:
    assert _actions(cagui_use.step_to_tool_calls(_cagui_step(0, 1))) == [
        {"action": "wait", "duration": 1.0}
    ]
    assert _actions(cagui_use.step_to_tool_calls(_cagui_step(0, 0))) == [
        {"action": "long_press", "coordinate": [500, 500], "duration": 1.0}
    ]
