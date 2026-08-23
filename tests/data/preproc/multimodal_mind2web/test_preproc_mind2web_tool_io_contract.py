"""Mind2Web preproc tool/result contract tests."""

from __future__ import annotations

import json

import pytest
from data.preproc._tool_io_helpers import (
    _assert_final_action_row,
    _assert_first_action_result_is_tool,
)

from lite.core.tools.calls import tool_call_id
from lite.data.preproc.multimodal_mind2web import use as mind2web_use


def _mind2web_row(step: int, op: str, value: str = "") -> dict:
    return {
        "annotation_id": "ann",
        "target_action_index": str(step),
        "target_action_reprs": op,
        "confirmed_task": "finish",
        "operation": json.dumps({"original_op": op, "value": value}),
        "pos_candidates": [
            json.dumps({
                "attributes": json.dumps({"bounding_box_rect": "100,200,50,50"})
            })
        ],
    }


@pytest.fixture
def mind2web_env(monkeypatch):
    monkeypatch.setattr(mind2web_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(mind2web_use, "_read_image_size", lambda _path: (1000, 1000))


@pytest.mark.parametrize("n_steps", [2, 3, 5])
def test_mind2web_emits_rows_with_final_action_at_eof(mind2web_env, n_steps):
    """A multi-step episode must PUBLISH, not vanish.

    The source has no screenshot after the final executable action, so that
    action remains as the EOF target. Guards the total-drop regression, where
    the per-episode raise sat inside the loop and fired on ``i == 0`` for every
    episode, emitting zero rows dataset-wide.
    """
    rows = [_mind2web_row(0, "TYPE", "hello")] + [
        _mind2web_row(i, "CLICK") for i in range(1, n_steps)
    ]
    row = mind2web_use.episode_to_entry("ann", rows)

    # Every screenshot and every action is published; only the final action has
    # no tool result.
    assert len(row["images"]) == n_steps
    action_turns = [m for m in row["messages"] if m.get("tool_calls")]
    assert len(action_turns) == n_steps

    # Each action is answered by the NEXT step's screenshot as role:"tool".
    tool_results = [m for m in row["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == n_steps - 1
    assert [m["content"] for m in tool_results] == [
        [{"type": "image", "index": i}] for i in range(1, n_steps)
    ]
    assert [m["tool_call_id"] for m in tool_results] == [
        tool_call_id(turn["tool_calls"][0]) for turn in action_turns[:-1]
    ]

    # No role:"user" turn survives past the goal turn.
    assert [m["role"] for m in row["messages"]].count("user") == 1
    _assert_first_action_result_is_tool(row)
    _assert_final_action_row(row)


def test_mind2web_single_step_episode_publishes_final_action(mind2web_env):
    """A single-step episode still contributes one supervised target."""
    row = mind2web_use.episode_to_entry("ann", [_mind2web_row(0, "CLICK")])

    assert len(row["images"]) == 1
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    _assert_final_action_row(row)


def test_mind2web_rejects_oob_bbox_center_before_normalization_clamp() -> None:
    # ``_bbox_center`` takes an already-parsed rect, so its ``None`` means one
    # thing only: the centre falls outside the captured screenshot. An
    # unparseable rect is ``_parse_bbox_rect``'s ``None`` and a different skip
    # reason -- the two must not collapse back into one message.
    assert mind2web_use._bbox_center((1000, 10, 20, 20), 1000, 1000) is None
    assert mind2web_use._bbox_center((1000, 1000, 0, 0), 1000, 1000) == [1000, 1000]
    assert mind2web_use._parse_bbox_rect("1000,10,20,20") == (1000, 10, 20, 20)
    assert mind2web_use._parse_bbox_rect("not,a,rect") is None
    assert mind2web_use._parse_bbox_rect(None) is None
