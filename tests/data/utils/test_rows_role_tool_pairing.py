from __future__ import annotations

import pytest

from lite.core.tools.calls import make_tool_call
from lite.data.utils.rows import validate_role_tool_pairing


def _tc(name: str, arguments: dict | None = None, *, call_id: str) -> dict:
    return make_tool_call(name, arguments, call_id=call_id)


def test_validate_role_tool_pairing_rejects_legacy_tool_message_call_id():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )
            ],
        },
        {"role": "tool", "call_id": "call_0000", "content": [{"type": "text", "text": "ok"}]},
    ]

    with pytest.raises(ValueError, match="must use tool_call_id, not call_id"):
        validate_role_tool_pairing(messages, is_rollout=True)


def test_validate_role_tool_pairing_rejects_partial_multi_call_completion():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    **_tc(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="call_0000",
                    ),
                },
                {
                    **_tc(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [3, 4]}]},
                        call_id="call_0001",
                    ),
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "text", "text": "ok"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]

    with pytest.raises(ValueError, match="call_0001"):
        validate_role_tool_pairing(messages, is_rollout=True)


@pytest.mark.parametrize("call_id", [None, "", 1, 1.5, False, b"x", ["x"], {"x": 1}])
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("computer", {"actions": [{"action": "click", "coordinate": [1, 2]}]}),
        ("response", {"text": "done"}),
    ],
)
def test_validate_role_tool_pairing_rejects_malformed_assistant_call_id_directly(
    call_id,
    name,
    arguments,
):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "text", "text": "ok"}]},
    ]

    with pytest.raises(ValueError, match="assistant tool_call missing non-empty string id"):
        validate_role_tool_pairing(messages, is_rollout=True)


def test_validate_role_tool_pairing_rejects_duplicate_assistant_call_id_same_turn():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _tc("response", {"text": "a"}, call_id="call_0000"),
                _tc("response", {"text": "b"}, call_id="call_0000"),
            ],
        }
    ]

    with pytest.raises(ValueError, match="duplicate assistant tool_call id 'call_0000'"):
        validate_role_tool_pairing(messages, is_rollout=True)


def test_validate_role_tool_pairing_rejects_reused_assistant_call_id_later_turn():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )
            ],
        },
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "text", "text": "ok"}]},
        {
            "role": "assistant",
            "tool_calls": [
                _tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [3, 4]}]},
                    call_id="call_0000",
                )
            ],
        },
    ]

    with pytest.raises(ValueError, match="reused assistant tool_call id 'call_0000'"):
        validate_role_tool_pairing(messages, is_rollout=True)


@pytest.mark.parametrize("name", ["terminate", "response"])
def test_validate_role_tool_pairing_allows_terminal_finish_tool_call_at_eof(name):
    messages = [
        {
            "role": "assistant",
            "tool_calls": [_tc(name, {"text": "Done."}, call_id="call_0000")],
        },
    ]

    validate_role_tool_pairing(messages, is_rollout=True)


def test_validate_role_tool_pairing_allows_finish_call_unpaired_at_eof():
    """Finishing produces no observation to pair with, so its result is optional.

    Unconditionally: whether an env screenshots after ``terminate`` is its own
    choice and both answers are legitimate, so this cannot depend on metadata.
    """
    messages = [
        {
            "role": "assistant",
            "tool_calls": [_tc("response", {"text": "Done."}, call_id="call_0000")],
        },
    ]

    validate_role_tool_pairing(messages, is_rollout=True)
