"""Message utility contract tests for data rows."""

from __future__ import annotations

import re

import pytest

from lite.core import LiteCUAMetadata
from lite.core.tools.action_space import validate_lite_action_batch_structure
from lite.core.tools.calls import make_tool_call
from lite.core.tools.extra_tools import APP_LAUNCH_TOOL_NAME, FINISH_TOOL_ORDER
from lite.core.tools.schemas import tool_schema_name
from lite.data.utils import messages as messages_module
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    validate_content_only_finals,
)


def _tc(name: str, arguments: dict | None = None, *, call_id: str) -> dict:
    return make_tool_call(name, arguments, call_id=call_id)


def test_persisted_extra_tool_schema_catalog_tracks_core_finish_order():
    """``extra_tool_schemas_for_messages`` emits core's names in core's order."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call("terminate", {"status": "success"}),
                make_tool_call("open_app", {"app_name": "Settings"}),
                make_tool_call("response", {"text": "done"}),
            ],
        }
    ]

    schemas = extra_tool_schemas_for_messages(messages)

    assert [tool_schema_name(schema) for schema in schemas] == [
        APP_LAUNCH_TOOL_NAME,
        *FINISH_TOOL_ORDER,
    ]


def test_finalize_use_messages_emits_role_tool_key_order():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                _tc(
                    "computer",
                    {"actions": [{"action": "screenshot"}]},
                    call_id="call_0000",
                )
            ],
        },
        {"role": "user", "content": [{"type": "image", "index": 0}]},
    ]

    out = messages_module.finalize_use_messages(messages)

    assert list(out[1]) == ["role", "tool_call_id", "content"]
    assert out[1] == {
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": [{"type": "image", "index": 0}],
    }


def test_action_wrapper_pairing_delegates_batch_shape_but_not_child_membership():
    """Shape errors come from core; an unknown child action still pairs.

    Pairing keys on the wrapper's structure so a batch naming an action the
    catalog does not yet know still consumes its screenshot; publication
    rejects that call later, at ``validate_canonical_rows``.
    """
    unknown_child = _tc(
        "computer",
        {"actions": [{"action": "future_action", "value": 1}]},
        call_id="call_computer",
    )
    assert messages_module._is_action_wrapper_result_boundary(unknown_child) is True

    for arguments in ("x", {"actions": []}, {"actions": ["click"]}, {"actions": [{}]}):
        call = {
            "id": "call_computer",
            "type": "function",
            "function": {"name": "computer", "arguments": arguments},
        }
        expected = validate_lite_action_batch_structure("computer", arguments)[1].reason
        with pytest.raises(ValueError, match=re.escape(expected)):
            messages_module._is_action_wrapper_result_boundary(call)


def test_validate_content_only_finals_rejects_bad_shape_even_with_raw_response():
    metadata = LiteCUAMetadata.from_dict(
        LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict()
    )
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Done."}],
            "raw_response": {"adapter_key": "gpt@desktop@use", "text": "Done."},
        }
    ]

    with pytest.raises(ValueError, match="content-only final assistant turn"):
        validate_content_only_finals(messages, metadata)
