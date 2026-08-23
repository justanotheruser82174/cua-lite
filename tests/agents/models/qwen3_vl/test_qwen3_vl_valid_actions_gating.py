"""Qwen3-VL ``valid_actions`` and active-extra tool-surface gates."""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    BASH_SCHEMA,
    CLICK_TYPE_ENUM,
    FINISH_ENUM,
    OPEN_APP_SCHEMA,
    QWEN3_VL_DESKTOP_KEY,
    QWEN3_VL_MOBILE_KEY,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    action_enum,
    agent_adapter_for,
    assemble_for,
    assemble_qwen3_vl_desktop,
    assemble_qwen3_vl_mobile,
    computer_use_enum,
    mobile_use_enum,
    qwen3_vl_adapter,
    rendered_qwen3_vl_schemas,
    rendered_schemas_for,
    rendered_system_text_for,
    tool_names,
    wrapper_name,
)

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name


def test_qwen_valid_actions_none_exposes_full_enum() -> None:
    """``valid_actions=None`` exposes every GUI enum entry."""
    schemas = assemble_qwen3_vl_desktop(None)
    assert tool_names(schemas) == {"computer_use"}
    enum = computer_use_enum(schemas)
    assert enum is not None
    assert FINISH_ENUM.isdisjoint(enum)
    assert CLICK_TYPE_ENUM <= set(enum)
    assert len(enum) == 12

    with_finish = computer_use_enum(
        assemble_qwen3_vl_desktop(None, [RESPONSE_SCHEMA, TERMINATE_SCHEMA])
    )
    assert with_finish is not None
    assert FINISH_ENUM <= set(with_finish)
    assert len(with_finish) == 14


def test_qwen_valid_actions_subset_trims_enum_exactly() -> None:
    schemas = assemble_qwen3_vl_desktop(["click", "type"])
    assert tool_names(schemas) == {"computer_use"}
    enum = computer_use_enum(schemas)
    assert enum is not None
    assert set(enum) == CLICK_TYPE_ENUM

    enum = computer_use_enum(assemble_qwen3_vl_desktop(["click", "type"], [TERMINATE_SCHEMA]))
    assert enum is not None
    assert set(enum) == CLICK_TYPE_ENUM | {"terminate"}


def test_qwen_valid_actions_public_hook_trims_wrapper_enum() -> None:
    space = type(qwen3_vl_adapter(None).action_space)
    filtered = space.filter_tool_schemas_for_valid_actions(
        space.get_tool_schemas(),
        ["click"],
    )
    enum = set(computer_use_enum(filtered) or [])
    assert enum == (CLICK_TYPE_ENUM - {"type"}) | FINISH_ENUM


def test_qwen_valid_actions_empty_suppresses_wrapper_tool() -> None:
    """``valid_actions=[]`` drops the ``computer_use`` wrapper entirely."""
    schemas = assemble_qwen3_vl_desktop([])
    assert schemas == []
    assert computer_use_enum(schemas) is None


def test_qwen_mobile_valid_actions_empty_suppresses_wrapper_tool() -> None:
    schemas = assemble_qwen3_vl_mobile([])
    assert schemas == []
    assert mobile_use_enum(schemas) is None


def test_qwen_empty_drop_evaluated_before_finish_reinject() -> None:
    """Active finish extras stay visible when ``valid_actions=[]`` hides the wrapper."""
    schemas = rendered_qwen3_vl_schemas([], [RESPONSE_SCHEMA, TERMINATE_SCHEMA])
    assert tool_names(schemas) == {"response", "terminate"}
    assert computer_use_enum(schemas) is None

    schemas = rendered_qwen3_vl_schemas([], [BASH_SCHEMA])
    assert tool_names(schemas) == {"bash"}
    assert computer_use_enum(schemas) is None


@pytest.mark.parametrize(
    "extra_tool_schemas",
    [
        [],
        [BASH_SCHEMA],
    ],
)
def test_qwen_native_finish_enum_is_extra_schema_driven(extra_tool_schemas) -> None:
    enum = set(computer_use_enum(assemble_qwen3_vl_desktop(None, extra_tool_schemas)) or [])
    assert FINISH_ENUM.isdisjoint(enum)
    assert CLICK_TYPE_ENUM <= enum


@pytest.mark.parametrize(
    "extra_tool_schemas,expected_entry",
    [
        ([RESPONSE_SCHEMA], "answer"),
        ([TERMINATE_SCHEMA], "terminate"),
    ],
)
def test_qwen_finish_schemas_drive_native_enum(extra_tool_schemas, expected_entry) -> None:
    schemas = assemble_qwen3_vl_desktop(None, extra_tool_schemas)
    enum = set(computer_use_enum(schemas) or [])
    assert FINISH_ENUM & enum == {expected_entry}
    assert CLICK_TYPE_ENUM <= enum
    assert tool_names(schemas) >= {tool_schema_name(extra_tool_schemas[0])}


def test_qwen_finish_schemas_are_not_rendered_as_standalone_tools() -> None:
    schemas = rendered_qwen3_vl_schemas(None, [RESPONSE_SCHEMA, TERMINATE_SCHEMA])
    names = tool_names(schemas)
    assert "computer_use" in names
    assert "response" not in names
    assert "terminate" not in names


def test_qwen_finish_schemas_do_not_overlay_nonempty_action_subset() -> None:
    enum = set(
        computer_use_enum(rendered_qwen3_vl_schemas(["click"], [RESPONSE_SCHEMA, TERMINATE_SCHEMA]))
        or []
    )
    assert CLICK_TYPE_ENUM - {"type"} <= enum
    assert FINISH_ENUM <= enum


def test_qwen_finish_schemas_survive_empty_action_as_standalone_tools() -> None:
    """``valid_actions=[]`` renders canonical standalone finish schemas."""
    schemas = rendered_qwen3_vl_schemas([], [RESPONSE_SCHEMA, TERMINATE_SCHEMA])
    assert tool_names(schemas) == {"response", "terminate"}


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        (QWEN3_VL_DESKTOP_KEY, "desktop"),
        (QWEN3_VL_MOBILE_KEY, "mobile"),
    ],
)
def test_qwen_finish_system_prompt_guidance_is_extra_schema_driven(
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


def test_qwen_action_valid_actions_without_finish_schemas_expose_no_standalone_finish() -> None:
    schemas = assemble_qwen3_vl_desktop(["click"], [])
    assert tool_names(schemas) == {"computer_use"}
    assert tool_names(schemas).isdisjoint({"response", "terminate"})


def test_extra_tools_surfaces_bash_schema() -> None:
    schemas = assemble_qwen3_vl_desktop(None, extra_tool_schemas=[BASH_SCHEMA])
    assert tool_names(schemas) == {"computer_use", "bash"}
    assert len(computer_use_enum(schemas)) == 12


def test_no_extra_tools_surfaces_no_bash() -> None:
    schemas = assemble_qwen3_vl_desktop(None)
    assert "bash" not in tool_names(schemas)


def test_extra_tools_survive_empty_valid_actions() -> None:
    schemas = assemble_qwen3_vl_desktop([], extra_tool_schemas=[BASH_SCHEMA])
    assert tool_names(schemas) == {"bash"}
    assert computer_use_enum(schemas) is None


def test_mobile_extra_tools_survive_empty_valid_actions() -> None:
    schemas = assemble_qwen3_vl_mobile([], extra_tool_schemas=[BASH_SCHEMA])
    assert tool_names(schemas) == {"bash"}
    assert mobile_use_enum(schemas) is None


@pytest.mark.parametrize(
    "adapter_key,platform,wrapper_enum",
    [
        (QWEN3_VL_DESKTOP_KEY, "desktop", computer_use_enum),
        (QWEN3_VL_MOBILE_KEY, "mobile", mobile_use_enum),
    ],
)
def test_qwen_family_empty_valid_actions_preserves_extra_schemas(
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
        (QWEN3_VL_DESKTOP_KEY, "desktop", computer_use_enum),
        (QWEN3_VL_MOBILE_KEY, "mobile", mobile_use_enum),
    ],
)
@pytest.mark.parametrize(
    "extra_schema,native_entry",
    [(RESPONSE_SCHEMA, "answer"), (TERMINATE_SCHEMA, "terminate")],
)
def test_qwen_family_empty_valid_actions_preserves_finish_schemas_at_assembly(
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


def test_qwen_mobile_open_app_extra_schema_enables_native_open() -> None:
    schemas = assemble_for(
        QWEN3_VL_MOBILE_KEY,
        "mobile",
        valid_actions=[],
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    assert tool_names(schemas) == {"mobile_use", "open_app"}
    assert mobile_use_enum(schemas) == ["open"]

    rendered = rendered_schemas_for(
        QWEN3_VL_MOBILE_KEY,
        "mobile",
        valid_actions=[],
        extra_tool_schemas=[OPEN_APP_SCHEMA],
    )
    assert tool_names(rendered) == {"mobile_use"}
    assert mobile_use_enum(rendered) == ["open"]


def test_qwen_mobile_valid_actions_open_app_does_not_enable_native_open() -> None:
    """``open_app`` is injected through extra tools, never ``valid_actions``."""
    schemas = assemble_for(
        QWEN3_VL_MOBILE_KEY,
        "mobile",
        valid_actions=["open_app"],
        extra_tool_schemas=[],
    )
    assert tool_names(schemas) == set()
    assert mobile_use_enum(schemas) is None


@pytest.mark.parametrize(
    "adapter_key,platform,wrapper",
    [
        (QWEN3_VL_DESKTOP_KEY, "desktop", "computer_use"),
        (QWEN3_VL_MOBILE_KEY, "mobile", "mobile_use"),
    ],
)
@pytest.mark.parametrize(
    "call,schema,expected_action",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA, "answer"),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA, "terminate"),
    ],
)
def test_qwen_native_finish_render_is_independent_of_matching_extra_schema(
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
        (QWEN3_VL_DESKTOP_KEY, "desktop"),
        (QWEN3_VL_MOBILE_KEY, "mobile"),
    ],
)
@pytest.mark.parametrize(
    "call,schema",
    [
        (make_tool_call("response", {"text": "42"}), RESPONSE_SCHEMA),
        (make_tool_call("terminate", {"status": "success"}), TERMINATE_SCHEMA),
    ],
)
def test_qwen_finish_render_stays_standalone_when_empty_action_hides_wrapper(
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
        (QWEN3_VL_DESKTOP_KEY, "desktop"),
        (QWEN3_VL_MOBILE_KEY, "mobile"),
    ],
)
def test_qwen_finish_render_stays_standalone_when_native_wrapper_hidden(
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


def test_qwen_content_done_ignores_source_yaml_terminate_without_metadata_schema() -> None:
    source_env_kwargs = {"extra_tools": ["terminate"]}
    parquet_metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=None,
        others={"source_env_kwargs": source_env_kwargs},
    )
    adapter = AgentAdapterRegistry.get(QWEN3_VL_DESKTOP_KEY, metadata=parquet_metadata)

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
    enum = action_enum(rendered_schemas_for(QWEN3_VL_DESKTOP_KEY, "desktop", None), "computer_use")
    assert "terminate" not in (enum or [])
    assert adapter._tool_calls_to_agent_ordered(
        [
            make_tool_call("terminate", {"status": "success"}),
        ]
    ) == [
        {
            "name": "computer_use",
            "arguments": {"action": "terminate", "status": "success"},
        }
    ]
