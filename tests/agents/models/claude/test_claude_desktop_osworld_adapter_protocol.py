"""Claude desktop OSWorld-family adapter/protocol rows."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent.base import AgentRegistry
from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.agents.models.claude.action_space import ClaudeDesktopActionSpace
from lite.agents.models.claude.utils.parse import parse_response_with_provenance
from lite.core.messages import no_tool_call_final_text
from lite.core.messages.final import pop_model_output_error
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.utils.image import encode_png

register_all()

FINAL_TEXT = "The answer is 42."
CURRENT_RESULT_TEXT = "CURRENT_RESULT_SENTINEL"


@dataclasses.dataclass(frozen=True)
class Row:
    env_id: str

    @property
    def adapter_key(self) -> str:
        return "claude@desktop@use"

    @property
    def row_id(self) -> str:
        return f"claude:{self.env_id}"


ROWS = (
    Row("lite.osworld"),
    Row("osworld"),
    Row("osworld_2"),
)


def _row_id(row: Row) -> str:
    return row.row_id


def _claude_tool_call(name: str, arguments: dict[str, Any] | str) -> Any:
    raw_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ChatCompletionMessageToolCall(
        id=f"toolu_{name}",
        type="function",
        function=Function(name=name, arguments=raw_arguments),
    )


def _claude_response(*, content: str | list[dict[str, Any]] | None = None, tool_calls=()) -> Any:
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls))
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _parse_claude(response: Any) -> tuple[dict[str, Any], str | None]:
    message = parse_response_with_provenance(
        response,
        scale_x=1.0,
        scale_y=1.0,
        action_space=ClaudeDesktopActionSpace(),
        resolution=(1024, 768),
        call_id_start=0,
    ).message
    return message, pop_model_output_error(message)


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_desktop_osworld_partial_rows_resolve_to_supported_boundary(row: Row) -> None:
    assert AgentRegistry.contains(row.adapter_key)
    assert not AgentAdapterRegistry.contains(row.adapter_key)


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_active_known_tool_projects_to_canonical_action(row: Row) -> None:
    del row
    message, error = _parse_claude(
        _claude_response(
            tool_calls=[
                _claude_tool_call(
                    "computer",
                    {"action": "left_click", "coordinate": [100, 200]},
                )
            ]
        )
    )

    assert error is None
    assert tool_call_name(message["tool_calls"][0]) == "computer"
    assert tool_call_arguments(message["tool_calls"][0])["actions"][0]["action"] == "click"


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_malformed_known_action_is_model_output_error(row: Row) -> None:
    del row
    message, error = _parse_claude(
        _claude_response(tool_calls=[_claude_tool_call("computer", "{not json")])
    )

    assert message["tool_calls"] == []
    assert error and "malformed tool_call arguments" in error


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_literal_unknown_tool_is_model_output_error(row: Row) -> None:
    del row
    message, error = _parse_claude(
        _claude_response(tool_calls=[_claude_tool_call("literal_unknown_tool", {})])
    )

    assert message["tool_calls"] == []
    assert error and "undeclared tool_call literal_unknown_tool" in error


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_content_only_final_text_survives_native_parser(row: Row) -> None:
    del row
    message, error = _parse_claude(_claude_response(content=FINAL_TEXT))

    assert error is None
    assert not message.get("tool_calls")
    assert no_tool_call_final_text(message) == FINAL_TEXT


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_image_data_binding_uses_current_result_image_index(row: Row) -> None:
    del row
    current_png = encode_png(Image.new("RGB", (8, 8), color="blue"))
    step_result = SimpleNamespace(
        results=[
            LiteToolResult(
                tool_call_id="call_0000",
                images=[current_png],
                text=CURRENT_RESULT_TEXT,
            )
        ]
    )

    [result] = step_result.results
    message = build_tool_result_message(
        result.tool_call_id,
        (1,),
        result.text,
        result.metadata,
        error=result.error,
    )

    assert message == {
        "role": "tool",
        "tool_call_id": "call_0000",
        "content": [
            {"type": "image", "index": 1},
            {"type": "text", "text": CURRENT_RESULT_TEXT},
        ],
    }
