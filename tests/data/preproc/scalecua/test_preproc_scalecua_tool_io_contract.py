"""ScaleCUA preproc tool/result contract tests."""

from __future__ import annotations

import pytest
from data.preproc._tool_io_helpers import (
    _actions,
    _all_calls,
    _assert_final_action_row,
    _assert_first_action_result_is_tool,
    _assert_no_terminate_outcome,
    _assert_structural_done_row,
    _assert_terminate_outcome,
)

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.preproc.scalecua import use as scalecua_use
from lite.data.utils.rows import validate_canonical_rows


def test_scalecua_post_action_screenshot_is_tool(monkeypatch):
    monkeypatch.setattr(scalecua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")

    def rec(step: int, prev: str, action: str) -> dict:
        return {
            "_line_num": step,
            "id": f"r{step}",
            "image": f"s{step}.png",
            "width": 1000,
            "height": 1000,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nTask: finish\n\nPrevious operations:\n{prev}",
                },
                {"from": "gpt", "value": f"Action: {action}"},
            ],
        }

    row = scalecua_use.merge_trajectory_steps(
        [
            rec(0, "None", "click(x=0.1, y=0.2)"),
            rec(1, "Step 1: click", "click(x=0.3, y=0.4)"),
        ],
        "android_unit",
        "OpenGVLab/ScaleCUA-Data/images",
        "mobile",
        {},
        0,
    )
    _assert_first_action_result_is_tool(row)


def test_scalecua_lone_final_terminate_becomes_done_without_schema(monkeypatch):
    monkeypatch.setattr(scalecua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")

    def rec(step: int, prev: str, action: str) -> dict:
        return {
            "_line_num": step,
            "id": f"r{step}",
            "image": f"s{step}.png",
            "width": 1000,
            "height": 1000,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nTask: finish\n\nPrevious operations:\n{prev}",
                },
                {"from": "gpt", "value": f"Action: {action}"},
            ],
        }

    row = scalecua_use.merge_trajectory_steps(
        [
            rec(0, "None", "click(x=0.1, y=0.2)"),
            rec(1, "Step 1: click", "terminate(status='success')"),
        ],
        "android_unit",
        "OpenGVLab/ScaleCUA-Data/images",
        "mobile",
        {},
        0,
    )

    _assert_structural_done_row(row)
    calls = _all_calls(row)
    assert [tool_call_name(call) for call in calls] == ["mobile"]
    assert tool_call_arguments(calls[0])["actions"][0]["action"] == "tap"
    assert row["metadata"]["extra_tool_schemas"] == []


def _scalecua_row(monkeypatch, final_action: str) -> dict:
    monkeypatch.setattr(scalecua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")

    def rec(step: int, prev: str, action: str) -> dict:
        return {
            "_line_num": step,
            "id": f"r{step}",
            "image": f"s{step}.png",
            "width": 1000,
            "height": 1000,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nTask: finish\n\nPrevious operations:\n{prev}",
                },
                {"from": "gpt", "value": f"Action: {action}"},
            ],
        }

    return scalecua_use.merge_trajectory_steps(
        [rec(0, "None", "click(x=0.1, y=0.2)"), rec(1, "Step 1: click", final_action)],
        "android_unit",
        "OpenGVLab/ScaleCUA-Data/images",
        "mobile",
        {},
        0,
    )


def test_scalecua_failure_reason_survives_into_metadata_others(monkeypatch):
    """ScaleCUA authors explanatory text on failures; it must survive the drop.

    The source writes it under ``info=``, so the terminate pattern accepts that
    spelling — otherwise the call misses the pattern entirely and the keyword
    fallback stores the raw call syntax as the ``reason``. This is the case the
    uniform ``Done.`` rewrite would silently destroy if the text were not moved
    to ``metadata.others``.
    """
    row = _scalecua_row(
        monkeypatch,
        "terminate(status='failure', info='The Video tab does not respond.')",
    )

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile"]
    _assert_terminate_outcome(
        row,
        status="failure",
        reason="The Video tab does not respond.",
    )
    validate_canonical_rows([row], "scalecua")


def test_scalecua_normalizes_fail_alias_to_failure(monkeypatch):
    """``status='fail'`` appears 161x in the raw corpus; normalize to the enum."""
    row = _scalecua_row(monkeypatch, "terminate(status='fail')")

    _assert_terminate_outcome(row, status="failure")
    validate_canonical_rows([row], "scalecua_fail_alias")


@pytest.mark.parametrize("status", ["failure", "fail"])
def test_scalecua_unquoted_info_none_yields_no_reason(monkeypatch, status):
    """``info=None`` is unquoted, so a quoted-only pattern misses the whole call.

    It is the dominant corpus spelling (120 failure occurrences across the
    navigation annotations). When it misses, the keyword fallback stores
    ``action_text[:200]`` -- i.e. the raw call syntax -- as the ``reason``, which
    the terminal-policy rewrite then publishes into
    ``metadata.others.terminate_reason`` as if it were the demonstrator's
    explanation. It carries no text, so the correct result is a recorded status
    and NO reason at all.
    """
    row = _scalecua_row(monkeypatch, f"terminate(status='{status}', info=None)")

    _assert_structural_done_row(row)
    _assert_terminate_outcome(row, status="failure")
    assert "terminate_reason" not in row["metadata"]["others"]


def test_scalecua_success_with_info_none_records_nothing(monkeypatch):
    """The same spelling on the success side must stay a plain structural end."""
    row = _scalecua_row(monkeypatch, "terminate(status='success', info=None)")

    _assert_structural_done_row(row)
    _assert_no_terminate_outcome(row)


def test_scalecua_success_info_becomes_final_text(monkeypatch):
    row = _scalecua_row(
        monkeypatch, "terminate(status='success', info='The file has been saved.')"
    )

    assert row["messages"][-1]["content"] == [
        {"type": "text", "text": "The file has been saved."}
    ]
    _assert_no_terminate_outcome(row)
    validate_canonical_rows([row], "scalecua_success_info")


def _scalecua_trajectory(monkeypatch, *actions: str, platform: str = "browser") -> dict:
    """Build a ScaleCUA ``use`` row from one action per source step."""
    monkeypatch.setattr(scalecua_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")

    def rec(step: int, action: str) -> dict:
        prev = "None" if step == 0 else "\n".join(
            f"Step {i + 1}: earlier" for i in range(step)
        )
        return {
            "_line_num": step,
            "id": f"r{step}",
            "image": f"s{step}.png",
            "width": 1000,
            "height": 1000,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nTask: answer me\n\nPrevious operations:\n{prev}",
                },
                {"from": "gpt", "value": f"Action: {action}"},
            ],
        }

    return scalecua_use.merge_trajectory_steps(
        [rec(step, action) for step, action in enumerate(actions)],
        "web_unit",
        "OpenGVLab/ScaleCUA-Data/images",
        platform,
        {},
        0,
    )


def test_scalecua_terminal_answer_becomes_the_final_turn_text(monkeypatch):
    """A terminal ``response`` is the row's ANSWER (the content-only final's text,
    as ``guiact`` emits), never a persisted finish call: persisting it appends the
    structural final AFTER a finish call, which ``validate_role_tool_pairing``
    rejects.
    """
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "response(answer='42')",
        "terminate(status='success')",
    )

    messages = row["messages"]
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "42"}],
    }
    assert all(tool_call_name(call) != "response" for call in _all_calls(row))
    assert row["metadata"]["extra_tool_schemas"] == []
    # The answer step keeps its screenshot (as the click's result); the shed
    # terminate step contributes neither a message nor an image.
    assert [message["role"] for message in messages] == [
        "user", "assistant", "tool", "assistant",
    ]
    assert len(row["images"]) == 2
    validate_canonical_rows([row], "scalecua")


def test_scalecua_success_info_overrides_earlier_terminal_response(monkeypatch):
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "response(answer='intermediate total')",
        "terminate(status='success', info='final total')",
    )

    assert row["messages"][-1]["content"] == [
        {"type": "text", "text": "final total"}
    ]
    validate_canonical_rows([row], "scalecua_success_info_priority")


def test_scalecua_blank_terminal_answer_falls_back_to_done(monkeypatch):
    """An empty answer is not information, so the structural marker stands."""
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "response(answer='')",
        "terminate(status='success')",
    )

    _assert_structural_done_row(row)
    assert all(tool_call_name(call) != "response" for call in _all_calls(row))


def test_scalecua_non_terminal_response_is_removed_without_dropping_actions(monkeypatch):
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "response(answer='an aside')",
        "click(x=0.3, y=0.4)",
        "terminate(status='success')",
    )
    assert [tool_call_name(call) for call in _all_calls(row)] == ["computer", "computer"]
    assert len(row["images"]) == 3
    assert any(
        part == {"type": "inline_reasoning", "text": "an aside"}
        for message in row["messages"]
        for part in message.get("content") or []
    )
    validate_canonical_rows([row], "scalecua_nonterminal_response")


def test_scalecua_non_terminal_mixed_response_never_silently_drops_its_action(monkeypatch):
    with pytest.raises(scalecua_use.SkipTrajectoryError, match="nonterminal_answer"):
        _scalecua_trajectory(
            monkeypatch,
            "click(x=0.1, y=0.2)\nresponse(answer='aside')",
            "click(x=0.3, y=0.4)",
            "terminate(status='success')",
        )


def test_scalecua_only_final_response_becomes_the_answer(monkeypatch):
    row = _scalecua_trajectory(
        monkeypatch,
        "response(answer='first')",
        "click(x=0.3, y=0.4)",
        "response(answer='second')",
        "terminate(status='success')",
    )
    assert row["messages"][-1]["content"] == [{"type": "text", "text": "second"}]
    assert row["messages"][0]["content"][-1] == {"type": "text", "text": "answer me"}
    validate_canonical_rows([row], "scalecua_multiple_response")


def test_scalecua_only_last_consecutive_terminal_response_becomes_answer(monkeypatch):
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.3, y=0.4)",
        "response(answer='partial result')",
        "response(answer='combined final result')",
        "terminate(status='success')",
    )
    assert row["messages"][-1]["content"] == [
        {"type": "text", "text": "combined final result"}
    ]
    assert len(row["images"]) == 2
    validate_canonical_rows([row], "scalecua_consecutive_terminal_responses")


def test_scalecua_answer_sharing_a_turn_keeps_the_actions(monkeypatch):
    """Only the answer call is shed; its turn's real actions still publish.

    Their result is the terminate step's screenshot, which is exactly what keeps
    that step's image in the row here and drops it in the answer-only case.
    """
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "click(x=0.3, y=0.4)\nresponse(answer='42')",
        "terminate(status='success')",
    )

    messages = row["messages"]
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "42"}],
    }
    assert [tool_call_name(call) for call in _all_calls(row)] == ["computer", "computer"]
    assert [message["role"] for message in messages] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    assert len(row["images"]) == 3
    validate_canonical_rows([row], "scalecua")


def test_scalecua_answerless_trajectory_can_end_on_final_action(monkeypatch):
    """No terminate screenshot is needed to train the final action itself."""
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "click(x=0.3, y=0.4)",
    )

    assert [message["role"] for message in row["messages"]] == [
        "user", "assistant", "tool", "assistant",
    ]
    _assert_final_action_row(row)
    assert len(row["images"]) == 2


def test_scalecua_terminal_answer_can_share_final_action_turn(monkeypatch):
    """A final answer beside a final action stays on the same EOF assistant turn."""
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "click(x=0.3, y=0.4)\nresponse(answer='42')",
    )

    assert row["messages"][-1]["content"][-1] == {"type": "text", "text": "42"}
    assert tool_call_name(row["messages"][-1]["tool_calls"][0]) == "computer"
    _assert_final_action_row(row)


def test_scalecua_answerless_trajectory_still_ends_on_done(monkeypatch):
    """The ordinary ending is untouched: no answer -> structural ``Done.``."""
    row = _scalecua_trajectory(
        monkeypatch,
        "click(x=0.1, y=0.2)",
        "click(x=0.3, y=0.4)",
        "terminate(status='success')",
    )

    _assert_structural_done_row(row)
    assert len(row["images"]) == 3
    assert [tool_call_name(call) for call in _all_calls(row)] == ["computer", "computer"]


def test_scalecua_preserves_source_line_order_and_press_count() -> None:
    calls = scalecua_use.parse_action_to_tool_calls(
        "moveTo(x=0.1, y=0.2)\ndragTo(x=0.3, y=0.4)",
        "moveTo(x=0.1, y=0.2)\ndragTo(x=0.3, y=0.4)",
        "desktop", "k", 1, 0,
    )
    assert [action["action"] for action in _actions(calls)] == ["mouse_move", "drag"]
    repeated = scalecua_use.parse_action_to_tool_calls(
        "press(keys='up', presses=3)", "press(keys='up', presses=3)",
        "desktop", "k", 1, 0,
    )
    assert [action["keys"] for action in _actions(repeated)] == [["up"], ["up"], ["up"]]

    with pytest.raises(scalecua_use.SkipTrajectoryError, match="invalid_click_count"):
        scalecua_use.parse_action_to_tool_calls(
            "click(x=0.3, y=0.4, clicks=40)",
            "click(x=0.3, y=0.4, clicks=40)",
            "desktop", "k", 1, 0,
        )

    with pytest.raises(scalecua_use.SkipTrajectoryError, match="unsupported_key"):
        scalecua_use.parse_action_to_tool_calls(
            "press(keys='Insert a blank line between the two lines of text.')",
            "press(keys='Insert a blank line between the two lines of text.')",
            "desktop", "k", 1, 0,
        )

    with pytest.raises(
        scalecua_use.SkipTrajectoryError, match="mobile_dragto_missing_start"
    ):
        scalecua_use.parse_action_to_tool_calls(
            "dragTo(x=0.3, y=0.4)",
            "dragTo(x=0.3, y=0.4)",
            "mobile", "k", 1, 0,
        )


def _scalecua_actions(code: str) -> list[dict]:
    return _actions(
        scalecua_use.parse_action_to_tool_calls(code, code, "desktop", "k", 1, 0)
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("press(keys='ctrl++')", [{"action": "key", "keys": ["ctrl", "+"]}]),
        ("press(keys='ctrl+-')", [{"action": "key", "keys": ["ctrl", "-"]}]),
        ("press(keys='ctrl+=')", [{"action": "key", "keys": ["ctrl", "="]}]),
        ("key('ctrl++')", [{"action": "key", "keys": ["ctrl", "+"]}]),
        ("key('ctrl+-')", [{"action": "key", "keys": ["ctrl", "-"]}]),
        ("key('ctrl+=')", [{"action": "key", "keys": ["ctrl", "="]}]),
        ("key('plus')", [{"action": "key", "keys": ["+"]}]),
        ("key('minus')", [{"action": "key", "keys": ["-"]}]),
        ("key('equal')", [{"action": "key", "keys": ["="]}]),
        ("key(keys=[','])", [{"action": "key", "keys": [","]}]),
        ("key(keys=['+', '-', '='])", [{"action": "key", "keys": ["+", "-", "="]}]),
        ("key(keys=['plus','minus','equal'])", [{"action": "key", "keys": ["+", "-", "="]}]),
        ("press(keys='plus')", [{"action": "key", "keys": ["+"]}]),
        ("press(keys='minus')", [{"action": "key", "keys": ["-"]}]),
        ("press(keys='equal')", [{"action": "key", "keys": ["="]}]),
        ("press(keys=[','])", [{"action": "key", "keys": [","]}]),
        ("press(keys=['+', '-', '='])", [{"action": "key", "keys": ["+", "-", "="]}]),
        ("press(keys=['plus','minus','equal'])", [{"action": "key", "keys": ["+", "-", "="]}]),
        ("hotkey('ctrl', '+')", [{"action": "key", "keys": ["ctrl", "+"]}]),
        ("hotkey('-', '=')", [{"action": "key", "keys": ["-", "="]}]),
        ("hotkey(keys=['ctrl', 'plus'])", [{"action": "key", "keys": ["ctrl", "+"]}]),
        ("hotkey(keys=['minus', 'equal'])", [{"action": "key", "keys": ["-", "="]}]),
        ("keyDown(key='ctrl++')", [{"action": "key_down", "keys": ["ctrl", "+"]}]),
        ("keyDown(key='minus')", [{"action": "key_down", "keys": ["-"]}]),
        ("keyDown(key=['ctrl', ','])", [{"action": "key_down", "keys": ["ctrl", ","]}]),
        (
            "keyDown(key=['plus', 'minus', 'equal'])",
            [{"action": "key_down", "keys": ["+", "-", "="]}],
        ),
        ("keyUp(key='ctrl+=')", [{"action": "key_up", "keys": ["ctrl", "="]}]),
        ("keyUp(key='plus')", [{"action": "key_up", "keys": ["+"]}]),
        ("keyUp(key=[','])", [{"action": "key_up", "keys": [","]}]),
        ("keyUp(key=['plus', 'minus', 'equal'])", [{"action": "key_up", "keys": ["+", "-", "="]}]),
    ],
)
def test_scalecua_key_parsers_emit_canonical_glyphs(code: str, expected: list[dict]) -> None:
    assert _scalecua_actions(code) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("hotkey('ctrl', 's')", [{"action": "key", "keys": ["ctrl", "s"]}]),
        ("hotkey(keys='ctrl+s')", [{"action": "key", "keys": ["ctrl", "s"]}]),
        ("press(keys='enter')", [{"action": "key", "keys": ["enter"]}]),
        ("keyDown(key='ctrl++')", [{"action": "key_down", "keys": ["ctrl", "+"]}]),
        ("keyUp(key=['ctrl', 's'])", [{"action": "key_up", "keys": ["ctrl", "s"]}]),
        (
            "hotkey('ctrl', 'a')\npress(keys='backspace')",
            [
                {"action": "key", "keys": ["ctrl", "a"]},
                {"action": "key", "keys": ["backspace"]},
            ],
        ),
    ],
)
def test_scalecua_direct_keyboard_actions_are_detected(
    response: str,
    expected: list[dict],
) -> None:
    _content_text, _reasoning_content, tool_calls = scalecua_use.parse_thought_action(
        response,
        "desktop",
        "k",
        1,
        0,
    )

    assert _actions(tool_calls) == expected


def test_scalecua_direct_non_keyboard_fallback_keeps_first_call_only() -> None:
    _content_text, _reasoning_content, tool_calls = scalecua_use.parse_thought_action(
        "click(x=0.1, y=0.2)\nresponse(answer='ignored')",
        "desktop",
        "k",
        1,
        0,
    )

    assert _actions(tool_calls) == [{"action": "click", "coordinate": [100, 200]}]


@pytest.mark.parametrize(
    "code",
    [
        "key(keys=['Insert a blank line'])",
        "key(keys=['ctrl+s'])",
        "key(keys=[])",
        "key(keys=[',', 1])",
        "press(keys=' ')",
        "press(keys=['ctrl+s'])",
        "press(keys=[])",
        "hotkey(keys=['ctrl', 1])",
        "hotkey(keys=['ctrl+s'])",
        "hotkey(keys=[])",
        "hotkey(keys=[' '])",
        "keyDown(key=['ctrl+s'])",
        "keyDown(key=[])",
        "keyDown(key=['\\n'])",
        "keyUp(key=['ctrl+s'])",
        "keyUp(key=[])",
    ],
)
def test_scalecua_key_parsers_reject_malformed_or_unknown_keys(code: str) -> None:
    with pytest.raises(scalecua_use.SkipTrajectoryError, match="unsupported_key"):
        scalecua_use.parse_action_to_tool_calls(code, code, "desktop", "k", 1, 0)


@pytest.mark.parametrize(
    "code",
    [
        "key('plus')",
        "key(keys=['+'])",
        "press(keys='plus')",
        "press(keys=['+'])",
        "press()",
        "hotkey('ctrl', '+')",
        "hotkey(keys=['ctrl', 'plus'])",
        "hotkey()",
        "key(keys=[])",
        "hotkey(keys=[])",
        "keyDown(key='plus')",
        "keyDown(key=['ctrl', 'plus'])",
        "keyDown(key=[])",
        "keyUp(key='plus')",
        "keyUp(key=['ctrl', 'plus'])",
        "keyUp(key=[])",
    ],
)
def test_scalecua_mobile_key_parsers_are_counted_source_skips(code: str) -> None:
    with pytest.raises(scalecua_use.SkipTrajectoryError, match="mobile_keyboard") as exc:
        scalecua_use.parse_action_to_tool_calls(code, code, "mobile", "k", 1, 0)

    assert exc.value.reason == "mobile_keyboard"
