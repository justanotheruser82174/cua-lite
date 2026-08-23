"""Tests for canonical Lite tool-call id stamping."""

from __future__ import annotations

import pytest

from lite.core.errors import ToolCallValidationError
from lite.core.tools.calls import (
    RUNTIME_RESULT_CALL_ID_KEY,
    make_tool_call,
    stamp_message_tool_call_ids,
    stamp_messages_tool_call_ids,
    stamp_tool_call_list_ids,
)


def _call(
    name: str = "computer",
    arguments: dict | None = None,
    call_id: str | None = None,
) -> dict:
    return make_tool_call(name, arguments or {}, call_id=call_id)


def test_stamp_tool_call_list_ids_restamps_explicit_call_list() -> None:
    calls = [
        _call("computer", call_id="provider_1"),
        _call("bash", {"command": "pwd"}),
    ]

    assert stamp_tool_call_list_ids(calls, preserve=False, start=7) is calls
    assert [call["id"] for call in calls] == ["call_0007", "call_0008"]
    assert [list(call) for call in calls] == [["id", "type", "function"]] * 2


def test_stamp_message_tool_call_ids_keeps_single_message_contract() -> None:
    message = {"role": "assistant", "tool_calls": [_call("computer")]}

    assert stamp_message_tool_call_ids(message, preserve=False) is message
    assert message["tool_calls"][0]["id"] == "call_0000"
    assert list(message["tool_calls"][0]) == ["id", "type", "function"]


def test_stamp_message_tool_call_ids_rejects_tool_call_dict() -> None:
    with pytest.raises(TypeError, match="Lite tool-call dict"):
        stamp_message_tool_call_ids(_call("computer"), preserve=False)


def test_stamp_messages_tool_call_ids_fills_missing_id_after_preserved_id() -> None:
    messages = [
        {"role": "user", "content": []},
        {"role": "assistant", "tool_calls": [_call("computer", call_id="call_0003")]},
        {"role": "assistant", "tool_calls": [_call("bash", {"command": "pwd"})]},
    ]

    assert stamp_messages_tool_call_ids(messages, preserve=True, start=3) is messages
    assert messages[1]["tool_calls"][0]["id"] == "call_0003"
    assert messages[2]["tool_calls"][0]["id"] == "call_0004"
    assert [list(call) for msg in messages[1:] for call in msg["tool_calls"]] == [
        ["id", "type", "function"],
        ["id", "type", "function"],
    ]


def test_stamp_messages_tool_call_ids_rejects_mixed_non_message_entries() -> None:
    messages = [
        {"role": "assistant", "tool_calls": []},
        {"name": "bash", "arguments": {"command": "pwd"}},
    ]

    with pytest.raises(TypeError, match=r"messages\[1\].*message dict with a role"):
        stamp_messages_tool_call_ids(messages, preserve=False)  # type: ignore[arg-type]


def test_explicit_call_list_rejects_message_shape() -> None:
    with pytest.raises(TypeError, match=r"tool_calls\[0\] must be a Lite tool-call dict"):
        stamp_tool_call_list_ids(
            [{"role": "assistant", "tool_calls": []}],  # type: ignore[list-item]
            preserve=False,
        )


def test_stamp_tool_call_list_ids_overwrites_existing_ids() -> None:
    calls = [
        _call("computer", call_id="provider_1"),
        _call("bash", {"command": "pwd"}),
    ]

    assert stamp_tool_call_list_ids(calls, start=3, preserve=False) is calls
    assert [call["id"] for call in calls] == ["call_0003", "call_0004"]
    assert [list(call) for call in calls] == [["id", "type", "function"]] * 2


def test_stamp_message_tool_call_ids_rejects_bare_agent_wire_dict() -> None:
    with pytest.raises(TypeError, match="message dict with a role"):
        stamp_message_tool_call_ids(
            {"name": "computer", "arguments": {}},  # type: ignore[arg-type]
            preserve=False,
        )


def test_stamp_tool_call_list_ids_rejects_bare_agent_wire_call_list_before_stamping() -> None:
    calls = [{"name": "computer", "arguments": {}}]

    with pytest.raises(ToolCallValidationError, match="bare model-function projection"):
        stamp_tool_call_list_ids(calls, preserve=False)  # type: ignore[arg-type]


def test_stamp_messages_tool_call_ids_preserves_existing_ids_without_collision() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _call("computer", call_id="call_0000"),
                _call("bash", {"command": "pwd"}),
            ],
        },
    ]

    assert stamp_messages_tool_call_ids(messages, preserve=True) is messages
    calls = messages[0]["tool_calls"]
    assert [call["id"] for call in calls] == ["call_0000", "call_0001"]
    assert [list(call) for call in calls] == [["id", "type", "function"]] * 2


@pytest.mark.parametrize("preserve", [False, True])
def test_stamp_tool_call_list_ids_rejects_result_pairing_id_on_calls(
    preserve: bool,
) -> None:
    calls = [
        {
            "tool_call_id": "result_1",
            "type": "function",
            "function": {"name": "computer", "arguments": {}},
        }
    ]

    with pytest.raises(ToolCallValidationError, match="tool_call_id"):
        stamp_tool_call_list_ids(calls, preserve=preserve)  # type: ignore[arg-type]


@pytest.mark.parametrize("tool_calls", [{}, "", 0, None])
def test_explicit_message_stampers_reject_non_list_tool_calls(tool_calls) -> None:
    message = {"role": "assistant", "tool_calls": tool_calls}
    messages = [message]

    with pytest.raises(TypeError, match="tool_calls"):
        stamp_message_tool_call_ids(message, preserve=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tool_calls"):
        stamp_messages_tool_call_ids(messages, preserve=True)  # type: ignore[arg-type]


def test_stamp_tool_call_list_ids_rejects_duplicate_preserved_ids() -> None:
    calls = [
        _call("computer", call_id="dup"),
        _call("bash", {"command": "pwd"}, call_id="dup"),
    ]

    with pytest.raises(ToolCallValidationError, match="duplicate id"):
        stamp_tool_call_list_ids(calls, preserve=True)


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("call_id", ["", None, 123])
def test_stamp_tool_call_list_ids_rejects_bad_present_id_before_stamping(
    preserve: bool,
    call_id,
) -> None:
    call = _call("computer")
    call["id"] = call_id

    with pytest.raises(ToolCallValidationError, match="id"):
        stamp_tool_call_list_ids([call], preserve=preserve)


def test_stamp_tool_call_list_ids_overwrites_duplicate_present_ids() -> None:
    calls = [
        _call("computer", call_id="provider_dup"),
        _call("bash", {"command": "pwd"}, call_id="provider_dup"),
    ]

    assert stamp_tool_call_list_ids(calls, preserve=False, start=9) is calls
    assert [call["id"] for call in calls] == ["call_0009", "call_0010"]


def test_stamp_tool_call_list_ids_accepts_canonical_call_before_overwrite() -> None:
    calls = [_call("computer", call_id="provider_1")]

    assert stamp_tool_call_list_ids(calls, preserve=False) is calls
    assert calls[0]["id"] == "call_0000"


def test_stamp_tool_call_list_ids_rejects_string_arguments_before_overwrite() -> None:
    calls = [
        {
            "id": "provider_1",
            "type": "function",
            "function": {"name": "computer", "arguments": "{}"},
        }
    ]

    with pytest.raises(ToolCallValidationError, match="arguments must be a dict"):
        stamp_tool_call_list_ids(calls, preserve=False)


def test_stamp_tool_call_list_ids_rejects_reserved_result_call_id_before_overwrite() -> None:
    call = _call("computer")
    call[RUNTIME_RESULT_CALL_ID_KEY] = "call_private"

    with pytest.raises(ToolCallValidationError, match="noncanonical outer keys"):
        stamp_tool_call_list_ids([call], preserve=False)


def test_core_tool_call_id_helpers_stay_on_calls_owner_not_tools_facade() -> None:
    import lite.core as core
    import lite.core.tools as core_tools
    import lite.core.tools.calls as core_calls

    expected = {
        "stamp_message_tool_call_ids",
        "stamp_messages_tool_call_ids",
        "stamp_tool_call_list_ids",
    }

    assert expected.isdisjoint(set(core.__all__))
    assert expected <= set(core_calls.__all__)
    assert expected.isdisjoint(set(core_tools.__all__))
    assert "stamp_tool_call_ids" not in set(core_calls.__all__)
    assert "stamp_tool_call_ids" not in set(core_tools.__all__)
    assert not hasattr(core_tools, "stamp_tool_call_ids")
    assert not hasattr(core_calls, "stamp_tool_call_ids")
    for name in expected:
        assert not hasattr(core, name)
        assert not hasattr(core_tools, name)
        assert getattr(core_calls, name)
