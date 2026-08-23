"""Qwen3.5 valid-actions and active-extra gates."""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    BASH_SCHEMA,
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    action_enum,
    agent_adapter_for,
    assemble_for,
    computer_use_enum,
    mobile_use_enum,
    rendered_schemas_for,
    rendered_system_text_for,
    tool_names,
    wrapper_name,
)

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name

QWEN3_5_DESKTOP_KEY = "qwen3_5@desktop@use"
QWEN3_5_MOBILE_KEY = "qwen3_5@mobile@use"


def test_qwen3_5_finish_schemas_are_native_wire_only_when_rendered() -> None:
    schemas = rendered_schemas_for(
        QWEN3_5_DESKTOP_KEY,
        "desktop",
        None,
        [RESPONSE_SCHEMA, TERMINATE_SCHEMA],
    )
    names = tool_names(schemas)
    assert "computer_use" in names
    assert "response" not in names
    assert "terminate" not in names

    adapter = agent_adapter_for(
        QWEN3_5_DESKTOP_KEY,
        "desktop",
        extra_tool_schemas=[RESPONSE_SCHEMA, TERMINATE_SCHEMA],
    )
    rendered = adapter._tool_calls_to_agent_ordered(
        [
            make_tool_call("response", {"text": "42"}),
            make_tool_call("terminate", {"status": "success"}),
        ]
    )
    assert rendered == [
        {"name": "computer_use", "arguments": {"action": "answer", "text": "42"}},
        {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}},
    ]


def test_qwen3_5_finish_schemas_survive_empty_action_as_standalone_tools() -> None:
    schemas = rendered_schemas_for(
        QWEN3_5_DESKTOP_KEY,
        "desktop",
        [],
        [RESPONSE_SCHEMA, TERMINATE_SCHEMA],
    )
    assert tool_names(schemas) == {"response", "terminate"}

    adapter = agent_adapter_for(
        QWEN3_5_DESKTOP_KEY,
        "desktop",
        valid_actions=[],
        extra_tool_schemas=[RESPONSE_SCHEMA],
    )
    assert adapter._tool_calls_to_agent_ordered(
        [
            make_tool_call("response", {"text": "42"}),
        ]
    ) == [{"name": "response", "arguments": {"text": "42"}}]


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        (QWEN3_5_DESKTOP_KEY, "desktop"),
        (QWEN3_5_MOBILE_KEY, "mobile"),
    ],
)
def test_qwen3_5_finish_system_prompt_guidance_is_extra_schema_driven(
    adapter_key,
    platform,
) -> None:
    default_prompt = rendered_system_text_for(adapter_key, platform)
    assert "If finishing" not in default_prompt

    native_prompt = rendered_system_text_for(
        adapter_key,
        platform,
        extra_tool_schemas=[RESPONSE_SCHEMA, TERMINATE_SCHEMA],
    )
    assert (
        "- If finishing, use action=answer for a question-answering task, "
        "otherwise use action=terminate in the tool call."
    ) in native_prompt

    response_only_prompt = rendered_system_text_for(
        adapter_key,
        platform,
        extra_tool_schemas=[RESPONSE_SCHEMA],
    )
    assert (
        "- If finishing a question-answering task, use action=answer in the tool call."
    ) in response_only_prompt
    assert "action=terminate" not in response_only_prompt
    assert "otherwise" not in response_only_prompt

    terminate_only_prompt = rendered_system_text_for(
        adapter_key,
        platform,
        extra_tool_schemas=[TERMINATE_SCHEMA],
    )
    assert "- If finishing, use action=terminate in the tool call." in terminate_only_prompt
    assert "action=answer" not in terminate_only_prompt
    assert "otherwise" not in terminate_only_prompt

    standalone_prompt = rendered_system_text_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[RESPONSE_SCHEMA, TERMINATE_SCHEMA],
    )
    assert "If finishing" not in standalone_prompt
    assert "action=answer" not in standalone_prompt
    assert "action=terminate" not in standalone_prompt

    standalone_response_only_prompt = rendered_system_text_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[RESPONSE_SCHEMA],
    )
    assert "If finishing" not in standalone_response_only_prompt
    assert "action=answer" not in standalone_response_only_prompt

    standalone_terminate_only_prompt = rendered_system_text_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[TERMINATE_SCHEMA],
    )
    assert "If finishing" not in standalone_terminate_only_prompt
    assert "action=terminate" not in standalone_terminate_only_prompt


@pytest.mark.parametrize(
    "adapter_key,platform,wrapper_enum",
    [
        (QWEN3_5_DESKTOP_KEY, "desktop", computer_use_enum),
        (QWEN3_5_MOBILE_KEY, "mobile", mobile_use_enum),
    ],
)
def test_qwen3_5_empty_valid_actions_preserves_extra_schemas(
    adapter_key,
    platform,
    wrapper_enum,
) -> None:
    schemas = assemble_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[BASH_SCHEMA],
    )
    assert tool_names(schemas) == {"bash"}
    assert wrapper_enum(schemas) is None


@pytest.mark.parametrize(
    "adapter_key,platform,wrapper_enum",
    [
        (QWEN3_5_DESKTOP_KEY, "desktop", computer_use_enum),
        (QWEN3_5_MOBILE_KEY, "mobile", mobile_use_enum),
    ],
)
@pytest.mark.parametrize(
    "extra_schema,native_entry",
    [(RESPONSE_SCHEMA, "answer"), (TERMINATE_SCHEMA, "terminate")],
)
def test_qwen3_5_empty_valid_actions_preserves_finish_schemas_at_assembly(
    adapter_key,
    platform,
    wrapper_enum,
    extra_schema,
    native_entry,
) -> None:
    schemas = assemble_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[extra_schema],
    )
    assert tool_names(schemas) == {tool_schema_name(extra_schema), wrapper_name(platform)}
    assert wrapper_enum(schemas) == [native_entry]


def test_qwen3_5_mobile_open_app_extra_schema_enables_native_open() -> None:
    schemas = assemble_for(
        QWEN3_5_MOBILE_KEY,
        "mobile",
        valid_actions=[],
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    assert tool_names(schemas) == {"mobile_use", "open_app"}
    assert mobile_use_enum(schemas) == ["open"]

    rendered = rendered_schemas_for(
        QWEN3_5_MOBILE_KEY,
        "mobile",
        valid_actions=[],
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    assert tool_names(rendered) == {"mobile_use"}
    assert mobile_use_enum(rendered) == ["open"]


def test_qwen3_5_mobile_valid_actions_open_app_does_not_enable_native_open() -> None:
    schemas = assemble_for(
        QWEN3_5_MOBILE_KEY,
        "mobile",
        valid_actions=["open_app"],
        extra_tool_schemas=[],
    )
    assert tool_names(schemas) == set()
    assert mobile_use_enum(schemas) is None


@pytest.mark.parametrize(
    "adapter_key,platform,wrapper",
    [
        (QWEN3_5_DESKTOP_KEY, "desktop", "computer_use"),
        (QWEN3_5_MOBILE_KEY, "mobile", "mobile_use"),
    ],
)
@pytest.mark.parametrize(
    "call,schema,expected_action",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA, "answer"),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA, "terminate"),
    ],
)
def test_qwen3_5_native_finish_render_is_independent_of_matching_extra_schema(
    adapter_key,
    platform,
    wrapper,
    call,
    schema,
    expected_action,
) -> None:
    adapter = agent_adapter_for(adapter_key, platform)
    rendered = adapter._tool_calls_to_agent_ordered([call])
    assert rendered == [
        {
            "name": wrapper,
            "arguments": {"action": expected_action, **tool_call_arguments(call)},
        }
    ]
    enum = action_enum(
        rendered_schemas_for(adapter_key, platform, None),
        wrapper,
    )
    assert expected_action not in (enum or [])

    adapter = agent_adapter_for(adapter_key, platform, extra_tool_schemas=[schema])
    rendered = adapter._tool_calls_to_agent_ordered([call])
    assert rendered == [
        {
            "name": wrapper,
            "arguments": {"action": expected_action, **tool_call_arguments(call)},
        }
    ]
    enum = action_enum(
        rendered_schemas_for(adapter_key, platform, None, [schema]),
        wrapper,
    )
    assert expected_action in (enum or [])


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        (QWEN3_5_DESKTOP_KEY, "desktop"),
        (QWEN3_5_MOBILE_KEY, "mobile"),
    ],
)
@pytest.mark.parametrize(
    "call,schema",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA),
    ],
)
def test_qwen3_5_finish_render_stays_standalone_when_empty_action_hides_wrapper(
    adapter_key,
    platform,
    call,
    schema,
) -> None:
    adapter = agent_adapter_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[schema],
    )

    assert adapter._tool_calls_to_agent_ordered([call]) == [
        {
            "name": tool_call_name(call),
            "arguments": tool_call_arguments(call),
        }
    ]


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        (QWEN3_5_DESKTOP_KEY, "desktop"),
        (QWEN3_5_MOBILE_KEY, "mobile"),
    ],
)
def test_qwen3_5_finish_render_stays_standalone_when_native_wrapper_hidden(
    adapter_key,
    platform,
) -> None:
    rendered = rendered_schemas_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[BASH_SCHEMA],
    )
    assert tool_names(rendered) == {"bash"}

    adapter = agent_adapter_for(
        adapter_key,
        platform,
        valid_actions=[],
        extra_tool_schemas=[BASH_SCHEMA],
    )
    assert adapter._tool_calls_to_agent_ordered(
        [
            make_tool_call("terminate", {"status": "success"}),
        ]
    ) == [
        {
            "name": wrapper_name(platform),
            "arguments": {"action": "terminate", "status": "success"},
        }
    ]


def test_qwen3_5_content_done_ignores_source_yaml_terminate_without_metadata_schema() -> None:
    source_env_kwargs = {"extra_tools": ["terminate"]}
    parquet_metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=None,
        others={"source_env_kwargs": source_env_kwargs},
    )
    adapter = AgentAdapterRegistry.get(QWEN3_5_DESKTOP_KEY, metadata=parquet_metadata)

    done = adapter.convert_message_to_agent(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
        }
    )

    assert done == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Done."}],
    }
