"""GUIAct use preproc action-batch and final-answer publish tests."""

from __future__ import annotations

import pytest

from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.preproc.guiact import use as guiact_use
from lite.data.preproc.guiact.utils import (
    SkipTrajectoryError,
    convert_mobile_action,
    convert_web_actions,
)
from lite.data.utils.rows import validate_canonical_rows

# Single-action GUI turns are canonical length-1 action-batch calls.
_WEB_CLICK_STEP = [
    {"name": "click", "point": {"related": "<point>0.25, 0.75</point>"}}
]

# GUIAct web ``input`` action → focus-click + type = TWO GUI actions in one turn.
_WEB_INPUT_STEP = [
    {"name": "input", "text": "hello", "point": {"related": "<point>0.5, 0.5</point>"}}
]

def test_convert_web_actions_wraps_single_action_turn():
    out = convert_web_actions(_WEB_CLICK_STEP, is_terminal=False)
    assert out == [
        make_tool_call(
            "computer",
            {
                "actions": [{"action": "click", "coordinate": [250, 750]}],
            },
        )
    ]

def test_convert_web_actions_emits_batched_computer_for_multi_action_turn():
    out = convert_web_actions(_WEB_INPUT_STEP, is_terminal=False)
    assert out == [
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [500, 500]},
                    {"action": "type", "text": "hello"},
                ],
            },
        )
    ]

def test_convert_web_actions_rejects_json_string_arguments_in_batch(monkeypatch):
    from lite.data.preproc.guiact import utils as guiact_utils

    monkeypatch.setitem(
        guiact_utils._WEB_CONVERTERS,
        "bad_json_args",
        lambda _action: [
            {
                "type": "function",
                "function": {
                    "name": "computer",
                    "arguments": "{\"actions\":[{\"action\":\"click\"}]}",
                },
            },
        ],
    )

    with pytest.raises(ValueError, match="arguments must be a dict"):
        convert_web_actions([{"name": "bad_json_args"}], is_terminal=False)

def test_convert_web_actions_terminal_answer_stays_response_not_terminate():
    out = convert_web_actions([{"name": "answer", "text": "completed"}], is_terminal=True)
    assert out == [make_tool_call("response", {"text": "completed"})]

def test_convert_mobile_action_terminal_answer_stays_response_not_terminate():
    out = convert_mobile_action({"name": "answer", "text": "completed"}, is_terminal=True)
    assert out == [make_tool_call("response", {"text": "completed"})]

def _patch_guiact_image_resolution(monkeypatch):
    monkeypatch.setattr(
        guiact_use,
        "resolve_path",
        lambda rel, _env: f"/raw/{rel}",
    )

def _guiact_web_step(
    image_id: str,
    actions_label: list[dict],
    *,
    question: str = "finish",
) -> dict:
    return {
        "image_id": image_id,
        "image_size": {"width": 1000, "height": 800},
        "question": question,
        "actions_label": actions_label,
    }

def _all_tool_calls(messages: list[dict]) -> list[dict]:
    return [
        tc
        for msg in messages
        for tc in msg.get("tool_calls", [])
    ]

def test_guiact_terminal_answer_is_carried_by_the_content_only_final(monkeypatch):
    """The final turn's text IS the answer; no ``terminate``, no schema.

    ``Done.`` is the trajectory-end SIGNAL, used only when the source produced
    no information. GUIAct's terminal ``answer`` IS the information, and it
    survives nowhere else -- ``terminate_outcome_others`` maps a success outcome
    to ``{}``.
    """
    _patch_guiact_image_resolution(monkeypatch)
    row = guiact_use.episode_to_example(
        "42",
        [
            _guiact_web_step(
                "uid_record_42_step_00",
                [{"name": "click", "point": {"related": "<point>0.5, 0.5</point>"}}],
            ),
            _guiact_web_step(
                "uid_record_42_step_01",
                [{"name": "answer", "text": "completed"}],
            ),
        ],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert row["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "completed"}],
    }
    assert "tool_calls" not in row["messages"][-1]
    assert row["metadata"]["extra_tool_schemas"] == []
    calls = _all_tool_calls(row["messages"])
    assert [tool_call_name(tc) for tc in calls] == ["computer"]
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "click", "coordinate": [500, 500]},
    ]
    assert "terminate" not in {tool_call_name(tc) for tc in calls}

def test_guiact_single_step_episode_without_answer_publishes_final_action(monkeypatch):
    """One action with no post-action screenshot is still an EOF SFT label."""
    _patch_guiact_image_resolution(monkeypatch)
    row = guiact_use.episode_to_example(
        "43",
        [
            _guiact_web_step(
                "uid_record_43_step_00",
                [{"name": "click", "point": {"related": "<point>0.25, 0.75</point>"}}],
            ),
        ],
        config=guiact_use.SUBSETS["web_multi"],
    )

    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert tool_call_name(row["messages"][-1]["tool_calls"][0]) == "computer"
    validate_canonical_rows([row], "guiact-single-step-final-action")

def test_guiact_nonterminal_answer_is_skipped_because_it_cannot_be_paired(monkeypatch):
    """A mid-trajectory ``answer`` publishes a ``response`` no screenshot answers.

    The turn also holds (or is followed by) an action-batch call, and a screenshot
    answers at most ONE call, so the ``response`` is left unpaired and
    ``validate_canonical_rows`` rejects the row -- 7 already-published web-multi
    train rows are invalid for exactly this reason.
    """
    _patch_guiact_image_resolution(monkeypatch)
    with pytest.raises(SkipTrajectoryError, match="non-terminal answer"):
        guiact_use.episode_to_example(
            "44",
            [
                _guiact_web_step(
                    "uid_record_44_step_00",
                    [{"name": "answer", "text": "not yet"}],
                ),
                _guiact_web_step(
                    "uid_record_44_step_01",
                    [{"name": "click", "point": {"related": "<point>0.25, 0.75</point>"}}],
                ),
                _guiact_web_step(
                    "uid_record_44_step_02",
                    [{"name": "answer", "text": "completed"}],
                ),
            ],
            config=guiact_use.SUBSETS["web_multi"],
        )
