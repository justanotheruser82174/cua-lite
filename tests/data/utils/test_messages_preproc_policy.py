from __future__ import annotations

import pytest

from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_id
from lite.core.tools.schemas import tool_schema_name
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    finalize_use_messages,
)

_DONE = [{"type": "text", "text": "Done."}]


def _image_user(index: int) -> dict:
    return {"role": "user", "content": [{"type": "image", "index": index}]}


def test_message_policy_pairs_structural_action_wrapper_result() -> None:
    messages = [
        _image_user(0),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "future_action", "value": 1}]},
                )
            ],
        },
        _image_user(1),
    ]

    out = finalize_use_messages(messages)

    call = out[1]["tool_calls"][0]
    assert out[2] == {
        "role": "tool",
        "tool_call_id": tool_call_id(call),
        "content": [{"type": "image", "index": 1}],
    }


def test_message_policy_does_not_pair_standalone_gui_action_by_global_whitelist() -> None:
    messages = [
        _image_user(0),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [make_tool_call("click", {"coordinate": [1, 2]})],
        },
        _image_user(1),
    ]

    out = finalize_use_messages(messages)

    assert out[2] == _image_user(1)


def test_message_policy_pairs_explicit_standalone_result_boundary() -> None:
    messages = [
        _image_user(0),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [make_tool_call("goto", {"url": "https://example.com"})],
        },
        _image_user(1),
    ]

    out = finalize_use_messages(messages, result_boundary_tools={"goto"})

    call = out[1]["tool_calls"][0]
    assert out[2] == {
        "role": "tool",
        "tool_call_id": tool_call_id(call),
        "content": [{"type": "image", "index": 1}],
    }


@pytest.mark.parametrize("name", ["bash", "response", "terminate", "future_text_tool"])
def test_message_policy_does_not_pair_text_or_finish_tools_by_omission(name: str) -> None:
    messages = [
        _image_user(0),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [make_tool_call(name, {"text": "x"})],
        },
        _image_user(1),
    ]

    out = finalize_use_messages(messages)

    assert out[2] == _image_user(1)


def test_extra_tool_schemas_are_message_derived_not_structural() -> None:
    messages = [
        _image_user(0),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                ),
                make_tool_call("open_app", {"app_name": "Settings"}),
                make_tool_call("response", {"text": "done"}),
            ],
        },
        {"role": "assistant", "content": _DONE},
    ]

    assert [tool_schema_name(schema) for schema in extra_tool_schemas_for_messages(messages)] == [
        "open_app",
        "response",
    ]


def test_extra_tool_schemas_include_only_persisted_standalone_finish_calls() -> None:
    messages = [
        _image_user(0),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [make_tool_call("terminate", {"status": "failure"})],
        },
    ]

    assert [
        tool_schema_name(schema)
        for schema in extra_tool_schemas_for_messages(messages)
    ] == ["terminate"]
