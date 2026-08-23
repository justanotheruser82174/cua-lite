"""Qwen2.5-VL valid-actions render delegation rows."""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    agent_adapter_for,
)

from lite.core.tools.calls import make_tool_call, tool_call_name


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        ("qwen2_5_vl@desktop@use", "desktop"),
        ("qwen2_5_vl@mobile@use", "mobile"),
    ],
)
@pytest.mark.parametrize(
    "call,schema",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA),
    ],
)
def test_qwen2_5_finish_render_delegates_to_family_converter(
    adapter_key,
    platform,
    call,
    schema,
) -> None:
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    adapter = agent_adapter_for(adapter_key, platform)
    if tool_call_name(call) == "response":
        with pytest.raises(ValueError, match="cannot render canonical tool 'response'"):
            adapter.convert_message_to_agent(message)
    else:
        adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(adapter_key, platform, extra_tool_schemas=[schema])
    if tool_call_name(call) == "response":
        with pytest.raises(ValueError, match="cannot render canonical tool 'response'"):
            adapter.convert_message_to_agent(message)
    else:
        adapter.convert_message_to_agent(message)


def test_qwen2_5_mobile_open_app_render_does_not_require_matching_extra_schema() -> None:
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [make_tool_call("open_app", {"app_name": "Settings"})],
    }

    adapter = agent_adapter_for("qwen2_5_vl@mobile@use", "mobile")
    adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(
        "qwen2_5_vl@mobile@use",
        "mobile",
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    adapter.convert_message_to_agent(message)
