"""Shared fixtures for the valid-actions gating split tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents.models._support.provider_fakes import FakeEnv

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools.schemas import make_tool_schema, tool_schema_name, tool_schema_parameters

register_all()

QWEN3_VL_DESKTOP_KEY = "qwen3_vl@desktop@use"
QWEN3_VL_MOBILE_KEY = "qwen3_vl@mobile@use"

ENV_REJECTED = "is rendering a canonical action-batch the env already rejected: "

FINISH_ENUM = {"terminate", "answer"}
CLICK_TYPE_ENUM = {
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "type",
}

RESPONSE_SCHEMA = make_tool_schema(
    "response",
    description="Submit the final answer.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

TERMINATE_SCHEMA = make_tool_schema(
    "terminate",
    description="Stop the task.",
    parameters={
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    },
)

OPEN_APP_SCHEMA = make_tool_schema(
    "open_app",
    description="Open an installed app.",
    parameters={
        "type": "object",
        "properties": {"app_name": {"type": "string"}},
        "required": ["app_name"],
    },
)

GOTO_SCHEMA = make_tool_schema(
    "goto",
    description="Navigate to a URL.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
)

ASK_USER_SCHEMA = make_tool_schema(
    "ask_user",
    description="Ask the user for clarification.",
    parameters={
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
)

UNRELATED_SCHEMA = make_tool_schema(
    "bash",
    description="Run a bash command.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)

BASH_SCHEMA = make_tool_schema(
    "bash",
    description="Run a bash command.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)

NO_ANSWER_CHANNEL_ADAPTER_PREFIXES = (
    "qwen2_5_vl@",
    "fara@",
    "evocua@",
    "ui_tars@desktop",
    "ui_tars@browser",
)

RESPONSE_CALL = {
    "type": "function",
    "function": {
        "name": "response",
        "arguments": {"text": "42"},
    },
}
RESPONSE_WIRE_CALL = {"name": "response", "arguments": {"text": "42"}}

SCHEMA_SURFACE_ACTIVE_EXTRA_GATES = {
    "evocua": "filter_qwen_action_values_for_active_extra_tools",
    "fara": "filter_fara_action_values_for_active_extra_tools",
    "qwen2_5_vl": "filter_qwen_action_values_for_active_extra_tools",
    "qwen3_5": "filter_qwen_action_values_for_active_extra_tools",
    "qwen3_vl": "filter_qwen_action_values_for_active_extra_tools",
}


def extra_tool_names_table(action_space: object) -> dict:
    """Read a family's provider-value -> Lite extra-tool-name table."""
    for name in dir(action_space):
        if name.endswith("_TO_EXTRA_TOOL_NAMES"):
            return getattr(action_space, name) or {}
    return {}


def family_active_extra_gate(adapter_key: str, action_space_cls):
    """Return the adapter family's active-extras gate, or None outside schema surfaces."""
    gate_name = SCHEMA_SURFACE_ACTIVE_EXTRA_GATES.get(adapter_key.split("@", 1)[0])
    if gate_name is None:
        return None
    return getattr(action_space_cls, gate_name)


def agent_adapter_for(
    adapter_key: str,
    platform: str,
    extra_tool_schemas=None,
    valid_actions=None,
    task_type: str = "use",
):
    meta = LiteCUAMetadata(
        dims=(platform, task_type),
        valid_actions=valid_actions,
        extra_tool_schemas=extra_tool_schemas or [],
    )
    return AgentAdapterRegistry.get(adapter_key, metadata=meta)


def qwen3_vl_adapter(valid_actions, extra_tool_schemas=None):
    return agent_adapter_for(
        QWEN3_VL_DESKTOP_KEY,
        "desktop",
        valid_actions=valid_actions,
        extra_tool_schemas=extra_tool_schemas,
    )


def assemble_for(
    adapter_key: str,
    platform: str,
    valid_actions,
    extra_tool_schemas=None,
):
    meta = LiteCUAMetadata(
        dims=(platform, "use"),
        valid_actions=valid_actions,
        extra_tool_schemas=extra_tool_schemas or [],
    )
    adapter = AgentAdapterRegistry.get(adapter_key, metadata=meta)
    schemas = adapter._assemble_tool_schemas()
    gate = family_active_extra_gate(adapter_key, type(adapter.action_space))
    if gate is not None:
        schemas = gate(schemas, adapter.active_extra_tool_names())
    return schemas


def assemble_qwen3_vl_desktop(valid_actions, extra_tool_schemas=None):
    return assemble_for(QWEN3_VL_DESKTOP_KEY, "desktop", valid_actions, extra_tool_schemas)


def assemble_qwen3_vl_mobile(valid_actions, extra_tool_schemas=None):
    return assemble_for(QWEN3_VL_MOBILE_KEY, "mobile", valid_actions, extra_tool_schemas)


def rendered_schemas_for(
    adapter_key: str,
    platform: str,
    valid_actions,
    extra_tool_schemas=None,
):
    """Schemas emitted inside a rendered ``<tools>`` block."""
    lines = (
        agent_adapter_for(
            adapter_key,
            platform,
            valid_actions=valid_actions,
            extra_tool_schemas=extra_tool_schemas,
        )
        ._build_tools_section()
        .splitlines()
    )
    start = lines.index("<tools>") + 1
    end = lines.index("</tools>")
    return [json.loads(line) for line in lines[start:end] if line.strip()]


def rendered_qwen3_vl_schemas(valid_actions, extra_tool_schemas=None):
    return rendered_schemas_for(
        QWEN3_VL_DESKTOP_KEY,
        "desktop",
        valid_actions,
        extra_tool_schemas,
    )


def rendered_system_text_for(
    adapter_key: str,
    platform: str,
    valid_actions=None,
    extra_tool_schemas=None,
) -> str:
    adapter = agent_adapter_for(
        adapter_key,
        platform,
        valid_actions=valid_actions,
        extra_tool_schemas=extra_tool_schemas,
    )
    sample = LiteSample(
        metadata=adapter.metadata,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Complete the task."}],
            }
        ],
    )
    step = adapter.render_step(sample, 1, [])
    assert step[0]["role"] == "system"
    return "\n".join(
        part["text"]
        for part in step[0].get("content") or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


def computer_use_enum(schemas) -> list[str] | None:
    """Return the ``computer_use`` wrapper's action enum, or None if absent."""
    return action_enum(schemas, "computer_use")


def mobile_use_enum(schemas) -> list[str] | None:
    return action_enum(schemas, "mobile_use")


def tool_names(schemas) -> set[str]:
    return {tool_schema_name(s) for s in schemas}


def openai_provider_tool_names(schemas) -> set[str]:
    """Names from OpenAI request-tool schemas."""
    names: set[str] = set()
    for schema in schemas:
        if schema["type"] == "function":
            names.add(schema["name"])
        else:
            names.add(schema["type"])
    return names


def anthropic_provider_tool_names(schemas) -> set[str]:
    """Names from Anthropic request-tool schemas."""
    names: set[str] = set()
    for schema in schemas:
        if "input_schema" in schema:
            names.add(schema["name"])
        else:
            names.add(schema["function"]["name"])
    return names


def env_dims(env) -> tuple[int, int]:
    w, h = env.metadata.others["resolution"]
    return int(w), int(h)


async def openai_tools_sent(agent, monkeypatch, env=None) -> list[dict]:
    """The OpenAI Responses ``tools`` payload sent by ``agent.sample()``."""
    env = env if env is not None else FakeEnv(terminate_after=1)
    monkeypatch.setattr(
        "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
        AsyncMock(return_value=[env_dims(env)]),
    )
    mock = AsyncMock(
        return_value={
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
            "id": "resp_test",
            "usage": {},
        }
    )
    monkeypatch.setattr("litellm.aresponses", mock)
    await agent.sample(env, max_steps=2)
    return mock.call_args.kwargs["tools"] or []


async def anthropic_tools_sent(agent, monkeypatch, env=None) -> list[dict]:
    """The Anthropic ``tools`` payload sent by ``agent.sample()``."""
    mock = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=[], role="assistant"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            model_dump=lambda: {"choices": []},
        )
    )
    monkeypatch.setattr("litellm.acompletion", mock)
    await agent.sample(env if env is not None else FakeEnv(terminate_after=1), max_steps=2)
    return mock.call_args.kwargs["tools"]


def wrapper_name(platform: str) -> str:
    """Qwen's provider-native wrapper tool name for *platform*."""
    return "mobile_use" if platform == "mobile" else "computer_use"


def action_enum(schemas, wrapper: str) -> list[str] | None:
    """The ``action`` enum of the named wrapper schema, or None if absent."""
    for schema in schemas:
        if tool_schema_name(schema) == wrapper:
            return tool_schema_parameters(schema).get("properties", {}).get("action", {}).get(
                "enum"
            )
    return None


def response_message() -> dict:
    return {"role": "assistant", "content": [], "tool_calls": [dict(RESPONSE_CALL)]}


def enum_of(schemas: list[dict], wrapper: str):
    for schema in schemas:
        if tool_schema_name(schema) != wrapper:
            continue
        props = tool_schema_parameters(schema)["properties"]
        items = props.get("actions", {}).get("items", {}).get("properties", {})
        if "action" in items:
            return items["action"].get("enum")
        if "action" in props:
            return props["action"].get("enum")
    return None


def wrapper_item_props(schema: dict) -> dict:
    return tool_schema_parameters(schema)["properties"]["actions"]["items"]["properties"]
