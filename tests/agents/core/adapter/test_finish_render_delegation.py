"""Adapter render delegates canonical tool calls to the owning family converter."""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    ASK_USER_SCHEMA,
    GOTO_SCHEMA,
    NO_ANSWER_CHANNEL_ADAPTER_PREFIXES,
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    UNRELATED_SCHEMA,
    agent_adapter_for,
)

from lite.core.tools.calls import make_tool_call, tool_call_name


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        ("qwen3_vl@desktop@use", "desktop"),
        ("qwen3_vl@mobile@use", "mobile"),
        ("qwen3_5@desktop@use", "desktop"),
        ("qwen3_5@mobile@use", "mobile"),
        ("evocua@desktop@use", "desktop"),
        ("fara@desktop@use", "desktop"),
        ("ui_tars@desktop@use", "desktop"),
        ("ui_tars@mobile@use", "mobile"),
    ],
)
@pytest.mark.parametrize(
    "call,schema",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA),
    ],
)
def test_open_source_finish_render_delegates_to_family_converter(
    adapter_key,
    platform,
    call,
    schema,
) -> None:
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    adapter = agent_adapter_for(adapter_key, platform)
    if tool_call_name(call) == "response" and adapter_key.startswith(
        NO_ANSWER_CHANNEL_ADAPTER_PREFIXES
    ):
        with pytest.raises(ValueError, match="cannot render canonical tool 'response'"):
            adapter.convert_message_to_agent(message)
    else:
        adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(adapter_key, platform, extra_tool_schemas=[schema])
    if tool_call_name(call) == "response" and adapter_key.startswith(
        NO_ANSWER_CHANNEL_ADAPTER_PREFIXES
    ):
        with pytest.raises(ValueError, match="cannot render canonical tool 'response'"):
            adapter.convert_message_to_agent(message)
    else:
        adapter.convert_message_to_agent(message)


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        ("qwen3_vl@desktop@use", "desktop"),
        ("qwen3_vl@mobile@use", "mobile"),
        ("qwen3_5@desktop@use", "desktop"),
        ("qwen3_5@mobile@use", "mobile"),
    ],
)
def test_open_app_render_does_not_require_matching_extra_schema(adapter_key, platform) -> None:
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [make_tool_call("open_app", {"app_name": "Settings"})],
    }

    adapter = agent_adapter_for(adapter_key, platform)
    adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(
        adapter_key,
        platform,
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    adapter.convert_message_to_agent(message)


@pytest.mark.parametrize(
    "adapter_key,platform,call,schema",
    [
        (
            "qwen3_vl@desktop@use",
            "desktop",
            make_tool_call("goto", {"url": "https://example.com"}),
            GOTO_SCHEMA,
        ),
        (
            "qwen3_5@desktop@use",
            "desktop",
            make_tool_call("bash", {"command": "pwd"}),
            UNRELATED_SCHEMA,
        ),
        (
            "qwen3_vl@mobile@use",
            "mobile",
            make_tool_call("ask_user", {"question": "Continue?"}),
            ASK_USER_SCHEMA,
        ),
        (
            "qwen3_5@mobile@use",
            "mobile",
            make_tool_call("ask_user", {"question": "Continue?"}),
            ASK_USER_SCHEMA,
        ),
    ],
)
def test_standalone_extra_render_does_not_require_matching_schema(
    adapter_key,
    platform,
    call,
    schema,
) -> None:
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    adapter = agent_adapter_for(adapter_key, platform)
    adapter.convert_message_to_agent(message)

    adapter = agent_adapter_for(adapter_key, platform, extra_tool_schemas=[schema])
    adapter.convert_message_to_agent(message)


@pytest.mark.parametrize(
    "adapter_key,platform,call",
    [
        ("qwen3_vl@desktop@use", "desktop", make_tool_call("click", {"coordinate": [1, 2]})),
        ("qwen3_vl@mobile@use", "mobile", make_tool_call("tap", {"coordinate": [1, 2]})),
        ("qwen3_vl@desktop@use", "desktop", make_tool_call("point", {"coordinate": [1, 2]})),
        (
            "qwen3_vl@desktop@use",
            "desktop",
            make_tool_call("bbox", {"coordinate": [1, 2, 3, 4]}),
        ),
    ],
)
def test_bare_actions_do_not_gate_adapter_render(
    adapter_key,
    platform,
    call,
) -> None:
    adapter = agent_adapter_for(adapter_key, platform)
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    adapter.convert_message_to_agent(message)
