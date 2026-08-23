"""Tests for Qwen3-VL-owned tool prompt serialization."""

from __future__ import annotations

import json

import pytest

from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.errors import LiteContractError
from lite.core.tools.schemas import make_tool_schema

_SCHEMA = make_tool_schema(
    "terminate",
    description="Finish the task.",
    parameters={
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": [],
    },
)


def _json_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


def test_qwen_schema_is_emitted_with_load_bearing_key_order() -> None:
    line = json.dumps(_SCHEMA)
    assert line.startswith('{"type": "function", "function":')
    assert '"name": "terminate"' in line
    assert _json_lines(line) == [_SCHEMA]


def test_qwen_tools_json_is_one_object_per_line() -> None:
    lines = "\n".join(
        json.dumps(schema) for schema in [_SCHEMA, make_tool_schema("noop")]
    ).splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('{"type": "function", "function":')
    assert [schema["function"]["name"] for schema in _json_lines("\n".join(lines))] == [
        "terminate",
        "noop",
    ]


def test_qwen_rejects_top_level_schema_fields() -> None:
    schema = {
        "type": "function",
        "name": "terminate",
        "description": "Finish the task.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    adapter = Qwen3VLDesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
        ),
    )
    adapter.action_space.get_tool_schemas = lambda: [schema]  # type: ignore[method-assign]

    with pytest.raises(LiteContractError, match="noncanonical outer keys"):
        adapter._build_tools_section()


def test_qwen_tool_json_escapes_non_ascii() -> None:
    schema = make_tool_schema("unicode_tool", description="Resume - check")
    schema["function"]["description"] = "Ré­sumé — ✓"
    qwen = Qwen3VLDesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
        ),
    )
    qwen.action_space.get_tool_schemas = lambda: [schema]  # type: ignore[method-assign]

    assert "\\u" in qwen._build_tools_section()
