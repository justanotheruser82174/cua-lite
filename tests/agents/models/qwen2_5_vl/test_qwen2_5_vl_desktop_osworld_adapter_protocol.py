"""Qwen2.5-VL desktop OSWorld-family adapter/protocol rows."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent.base import AgentRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages import keep_model_visible_content, no_tool_call_final_text
from lite.core.messages.final import pop_model_output_error
from lite.core.messages.image_refs import (
    referenced_image_indices_in_message_order,
    referenced_images_in_message_order,
)
from lite.core.tools import make_tool_call
from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
from lite.core.tools.calls import tool_call_arguments, tool_call_name

register_all()

FAMILY = "qwen2_5_vl"
EXPECTED_PROTOCOL = "qwen3_vl.history"
FINAL_TEXT = "The answer is 42."
CURRENT_RESULT_TEXT = "CURRENT_RESULT_SENTINEL"
SHARED_ZERO_CONVERSION_ERROR = (
    "model emitted tool_call(s), but none converted to canonical Lite tool_calls"
)
STRICT_NATIVE_SCHEMA_ERROR = (
    "tool call did not satisfy the active tool schema or native action grammar"
)
RAW_ACTIVE = (
    "Action: Click the button.\n"
    "<tool_call>\n"
    '{"name":"computer_use","arguments":{"action":"left_click","coordinate":[500,300]}}\n'
    "</tool_call>"
)
RAW_MALFORMED = (
    "Action: Use the bad action.\n"
    "<tool_call>\n"
    '{"name":"computer_use","arguments":{"action":"not_a_real_action","coordinate":[500,300]}}\n'
    "</tool_call>"
)
RAW_UNKNOWN = (
    "Action: Use a literal unknown tool.\n"
    "<tool_call>\n"
    '{"name":"totally_unknown_tool","arguments":{}}\n'
    "</tool_call>"
)
CANONICAL_EXECUTABLE_NAMES = {"computer", "terminate", "response"}
KNOWN_ACTION_WRAPPERS = {"computer", "mobile", "point", "bbox"}


@dataclasses.dataclass(frozen=True)
class Row:
    env_id: str

    @property
    def adapter_key(self) -> str:
        return f"{FAMILY}@desktop@use"

    @property
    def row_id(self) -> str:
        return f"{FAMILY}:{self.env_id}"


ROWS = (Row("lite.osworld"),)


def _row_id(row: Row) -> str:
    return row.row_id


def _parse_local(row: Row, raw: str) -> tuple[dict[str, Any], str | None]:
    adapter = AgentAdapterRegistry.get(row.adapter_key)
    agent_message = adapter.parse_raw_assistant_response(raw)
    lite_message = adapter.convert_message_from_agent(agent_message)
    error = pop_model_output_error(lite_message) or pop_model_output_error(agent_message)
    if not error and agent_message.get("tool_calls") and not lite_message.get("tool_calls"):
        error = SHARED_ZERO_CONVERSION_ERROR
    return lite_message, error


def _text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        part["text"]
        for message in messages
        for part in (message.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _sample_with_current_result_image(images: list[Image.Image]) -> LiteSample:
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
        images=images,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open the target app."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click."}],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [100, 200]}]},
                        call_id="call_0000",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [
                    {"type": "image", "index": 1},
                    {"type": "text", "text": CURRENT_RESULT_TEXT},
                ],
            },
        ],
    )


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_desktop_osworld_partial_rows_resolve_to_supported_boundary(row: Row) -> None:
    assert AgentRegistry.contains(row.adapter_key)
    assert AgentAdapterRegistry.contains(row.adapter_key)
    adapter = AgentAdapterRegistry.get(row.adapter_key)
    assert adapter.protocol.get_registry_key() == EXPECTED_PROTOCOL


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_active_known_tool_projects_to_canonical_action(row: Row) -> None:
    lite_message, error = _parse_local(row, RAW_ACTIVE)

    assert error is None
    assert lite_message.get("tool_calls")
    assert tool_call_name(lite_message["tool_calls"][0]) in CANONICAL_EXECUTABLE_NAMES


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_malformed_known_action_is_not_a_clean_content_only_final(row: Row) -> None:
    """A malformed action must never read as a clean content-only final."""
    lite_message, error = _parse_local(row, RAW_MALFORMED)
    calls = lite_message.get("tool_calls") or []

    if calls:
        assert len(calls) == 1
        assert tool_call_name(calls[0]) in LITE_ACTION_BATCH_TOOL_NAMES
        children = tool_call_arguments(calls[0])["actions"]
        assert [child["action"] for child in children] == ["not_a_real_action"]
        return

    assert error
    if no_tool_call_final_text(lite_message):
        assert error in {SHARED_ZERO_CONVERSION_ERROR, STRICT_NATIVE_SCHEMA_ERROR}


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_literal_unknown_tool_stays_literal_or_parse_error(row: Row) -> None:
    lite_message, error = _parse_local(row, RAW_UNKNOWN)
    calls = lite_message.get("tool_calls") or []

    if calls:
        assert tool_call_name(calls[0]) not in KNOWN_ACTION_WRAPPERS
        assert tool_call_name(calls[0]) not in CANONICAL_EXECUTABLE_NAMES
        assert error is None
    else:
        assert error
        assert not no_tool_call_final_text(lite_message)


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_content_only_final_text_survives_local_adapter(row: Row) -> None:
    lite_message, error = _parse_local(row, FINAL_TEXT)

    assert error is None
    assert not lite_message.get("tool_calls")
    assert no_tool_call_final_text(lite_message) == FINAL_TEXT


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_image_data_binding_uses_rendered_prompt_indices(row: Row) -> None:
    adapter = AgentAdapterRegistry.get(row.adapter_key)
    raw_images = [
        Image.new("RGB", (640, 360), color="red"),
        Image.new("RGB", (640, 360), color="blue"),
    ]
    processed = [adapter.process_image(image) for image in raw_images]
    sample = _sample_with_current_result_image(raw_images)

    rendered = keep_model_visible_content(adapter.render_step(sample, 2, processed))
    image_indices = referenced_image_indices_in_message_order(rendered)
    ordered_images = referenced_images_in_message_order(rendered, processed)

    assert image_indices
    assert image_indices[-1] == 1
    assert len(ordered_images) == len(image_indices)
    for image, index in zip(ordered_images, image_indices):
        assert image is processed[index]
    assert CURRENT_RESULT_TEXT in _text(rendered)
