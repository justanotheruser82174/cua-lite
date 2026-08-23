from __future__ import annotations

import pytest

from lite.core.tools.action_space import validate_lite_action_batch_structure
from lite.core.tools.calls import make_tool_call
from lite.data.utils.rows import validate_action_batches, validate_tool_calls


def _tc(name: str, arguments: dict | None = None, *, call_id: str) -> dict:
    return make_tool_call(name, arguments, call_id=call_id)


def test_validate_tool_calls_accepts_canonical_calls():
    """Canonical no-arg tools carry an explicit empty arguments dict."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _tc("back", {}, call_id="call_0000"),
                _tc("goto", {"url": "x"}, call_id="call_0001"),
            ],
        },
    ]
    assert validate_tool_calls(messages) is messages


def test_validate_tool_calls_no_tool_calls_is_noop():
    """Messages without tool_calls pass through unchanged."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert validate_tool_calls(messages) == messages


@pytest.mark.parametrize("tool_calls", [None, []])
def test_validate_tool_calls_tolerates_non_assistant_empty_padding(tool_calls):
    """Storage padding can leave empty/null tool_calls on non-assistant rows."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi"}],
            "tool_calls": tool_calls,
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": "ok"}],
            "tool_calls": tool_calls,
        },
    ]

    assert validate_tool_calls(messages) == messages


@pytest.mark.parametrize("role", ["user", "tool", "system", None])
def test_validate_tool_calls_rejects_non_assistant_non_empty_tool_calls(role):
    message = {
        "content": [{"type": "text", "text": "hi"}],
        "tool_calls": [_tc("response", {"text": "x"}, call_id="call_0000")],
    }
    if role is not None:
        message["role"] = role

    with pytest.raises(ValueError, match="non-assistant"):
        validate_tool_calls([message])


@pytest.mark.parametrize(
    "messages,match",
    [
        ({}, "messages must be a list"),
        (["not-a-message"], "messages\\[0\\] must be a dict"),
    ],
)
def test_validate_tool_calls_rejects_noncanonical_messages_container(messages, match):
    with pytest.raises(ValueError, match=match):
        validate_tool_calls(messages)


@pytest.mark.parametrize(
    "call,match",
    [
        (
            {
                "call_id": "call_0000",
                "name": "back",
                "arguments": {},
                "type": "function_call",
            },
            "provider function-call payload",
        ),
        ({"call_id": "call_0000", "name": "back", "arguments": {}, "id": "native"}, "noncanonical"),
        (
            {
                "function": {"name": "back", "arguments": {}},
                "call_id": "call_0000",
                "name": "back",
            },
            "noncanonical",
        ),
        (
            {"type": "function", "function": {"name": "back", "arguments": {}}},
            "missing non-empty id",
        ),
        ({"call_id": "call_0000", "name": "back"}, "noncanonical"),
        ({"call_id": "call_0000", "name": "back", "arguments": None}, "noncanonical"),
        ({"call_id": "call_0000", "name": "back", "arguments": "null"}, "noncanonical"),
        ({"name": "back", "arguments": {}}, "bare model-function projection"),
        ({"call_id": "", "name": "back", "arguments": {}}, "noncanonical"),
        ({"call_id": "call_0000", "arguments": {}}, "noncanonical"),
        ({"call_id": "call_0000", "name": "", "arguments": {}}, "noncanonical"),
    ],
)
def test_validate_tool_calls_rejects_noncanonical_calls(call, match):
    messages = [{"role": "assistant", "tool_calls": [call]}]
    with pytest.raises(ValueError, match=match):
        validate_tool_calls(messages)


@pytest.mark.parametrize(
    "tool_calls,match",
    [
        ({"call_id": "call_0000", "name": "back", "arguments": {}}, "must be a list"),
        (None, "must be a list"),
        (["back"], "must be a dict"),
    ],
)
def test_validate_tool_calls_rejects_noncanonical_tool_calls_container(tool_calls, match):
    messages = [{"role": "assistant", "tool_calls": tool_calls}]
    with pytest.raises(ValueError, match=match):
        validate_tool_calls(messages)


def test_validate_action_batches_rejects_empty_actions():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [_tc("computer", {"actions": []}, call_id="call_computer")],
        }
    ]

    with pytest.raises(ValueError, match="non-empty"):
        validate_action_batches(messages)


def test_validate_action_batches_raises_shared_action_batch_structure_reason():
    arguments = {"actions": [{"coordinate": [1, 2]}]}
    _children, error = validate_lite_action_batch_structure("computer", arguments)
    assert error is not None
    messages = [
        {
            "role": "assistant",
            "tool_calls": [_tc("computer", arguments, call_id="call_computer")],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        validate_action_batches(messages)

    assert error.reason in str(exc_info.value)


def test_validate_action_batches_rejects_child_argument_errors():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _tc("computer", {"actions": [{"action": "key"}]}, call_id="call_computer")
            ],
        }
    ]
    match = (
        r"messages\[0\]\.tool_calls\[0\]\.computer\.arguments\.actions"
        r"\[0\]\.keys is required for key"
    )

    with pytest.raises(ValueError, match=match):
        validate_action_batches(messages)
