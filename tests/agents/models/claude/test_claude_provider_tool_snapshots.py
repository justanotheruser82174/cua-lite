"""Claude provider tool-list snapshots."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from lite.agents.models.claude.agent import (
    ClaudeDesktopGroundingPointAgent,
    ClaudeDesktopUseAgent,
    ClaudeMobileUseAgent,
)
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_schema

_CLAUDE_OPUS_4_6_TOOL_CONFIG = {
    "tool_version": "computer_20251124",
    "beta_flag": "computer-use-2025-11-24",
}


def _lookup_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }


def _metadata(valid_actions: list[str] | None) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        valid_actions=valid_actions,
        extra_tool_schemas=[
            make_tool_schema(
                "lookup",
                description="Look up a short value.",
                parameters=_lookup_parameters(),
            )
        ],
    )


def _anthropic_lookup_tool() -> dict[str, Any]:
    return {
        "name": "lookup",
        "description": "Look up a short value.",
        "input_schema": _lookup_parameters(),
    }


def _litellm_lookup_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a short value.",
            "parameters": _lookup_parameters(),
        },
    }


def _payload_bytes(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def _claude_desktop_tools() -> list[dict[str, Any]]:
    return ClaudeDesktopUseAgent(
        model_id="claude-opus-4-6",
        metadata=_metadata(["click"]),
    )._build_tools(
        _CLAUDE_OPUS_4_6_TOOL_CONFIG,
        display_w=1024,
        display_h=768,
    )


def _claude_grounding_tools() -> list[dict[str, Any]]:
    return ClaudeDesktopGroundingPointAgent(
        model_id="claude-opus-4-6",
        metadata=_metadata(None),
    )._build_tools(
        _CLAUDE_OPUS_4_6_TOOL_CONFIG,
        display_w=1024,
        display_h=768,
    )


def _claude_mobile_tools() -> list[dict[str, Any]]:
    return ClaudeMobileUseAgent(
        model_id="claude-opus-4-6",
        metadata=_metadata(["tap"]),
    )._build_tools()


@pytest.mark.parametrize(
    ("build_tools", "expected"),
    [
        pytest.param(
            _claude_desktop_tools,
            [
                {
                    "type": "computer_20251124",
                    "function": {
                        "name": "computer",
                        "parameters": {
                            "display_height_px": 768,
                            "display_width_px": 1024,
                            "display_number": 1,
                        },
                    },
                },
                _anthropic_lookup_tool(),
            ],
            id="claude-desktop",
        ),
        pytest.param(
            _claude_grounding_tools,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "left_click",
                        "description": (
                            "Click the left mouse button at the specified pixel "
                            "coordinates of the target UI element."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "coordinate": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": (
                                        "[x, y] pixel coordinates of the click "
                                        "target on the screenshot."
                                    ),
                                    "minItems": 2,
                                    "maxItems": 2,
                                }
                            },
                            "required": ["coordinate"],
                        },
                    },
                },
                _anthropic_lookup_tool(),
            ],
            id="claude-grounding",
        ),
        pytest.param(
            _claude_mobile_tools,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "tap",
                        "description": (
                            "Single tap at the given pixel coordinate on an Android device.\n"
                            "Use for most UI interactions (buttons, list items, text fields).\n"
                            "Prefer over long_press unless you need a context menu."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "coordinate": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": (
                                        "[x, y] pixel coordinates on the current "
                                        "screenshot, origin top-left."
                                    ),
                                },
                                "clicks": {
                                    "type": "integer",
                                    "enum": [1, 2],
                                    "description": "Number of taps: 1=single, 2=double.",
                                },
                            },
                            "required": ["coordinate"],
                        },
                    },
                },
                _litellm_lookup_tool(),
            ],
            id="claude-mobile",
        ),
    ],
)
def test_claude_build_tools_snapshots(
    build_tools: Callable[[], list[dict[str, Any]]],
    expected: list[dict[str, Any]],
) -> None:
    tools = build_tools()

    assert tools == expected
    assert _payload_bytes(tools) == _payload_bytes(expected)
