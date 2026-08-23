"""GPT provider tool-list snapshots."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from lite.agents.models.gpt.agent import (
    GPTDesktopGroundingPointAgent,
    GPTDesktopUseAgent,
    GPTMobileUseAgent,
)
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_schema


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


def _responses_lookup_tool(*, strict: bool = False) -> dict[str, Any]:
    tool = {
        "type": "function",
        "name": "lookup",
        "description": "Look up a short value.",
        "parameters": _lookup_parameters(),
    }
    if strict:
        tool["strict"] = True
    return tool


def _payload_bytes(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def _gpt_desktop_tools() -> list[dict[str, Any]]:
    return GPTDesktopUseAgent(
        model_id="gpt-5.5",
        metadata=_metadata(["click"]),
    )._build_tools()


def _gpt_grounding_tools() -> list[dict[str, Any]]:
    return GPTDesktopGroundingPointAgent(
        model_id="gpt-5.5",
        metadata=_metadata(None),
    )._build_tools()


def _gpt_mobile_tools() -> list[dict[str, Any]]:
    return GPTMobileUseAgent(
        model_id="gpt-5.5",
        metadata=_metadata(["tap"]),
    )._build_tools()


@pytest.mark.parametrize(
    ("build_tools", "expected"),
    [
        pytest.param(
            _gpt_desktop_tools,
            [
                {"type": "computer"},
                _responses_lookup_tool(),
            ],
            id="gpt-desktop",
        ),
        pytest.param(
            _gpt_grounding_tools,
            [
                {
                    "type": "function",
                    "name": "click",
                    "description": (
                        "Click the left mouse button at the specified pixel "
                        "coordinates of the target UI element."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "integer",
                                "description": "X pixel coordinate of the click target.",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y pixel coordinate of the click target.",
                            },
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False,
                    },
                },
                _responses_lookup_tool(),
            ],
            id="gpt-grounding",
        ),
        pytest.param(
            _gpt_mobile_tools,
            [
                {
                    "type": "function",
                    "name": "tap",
                    "description": (
                        "Tap the screen at pixel coordinates (x, y).\n"
                        "Aim for the CENTER of the target element."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "integer",
                                "description": "X pixel coordinate (0 = left edge of screen).",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y pixel coordinate (0 = top edge of screen).",
                            },
                            "clicks": {
                                "type": "integer",
                                "enum": [1, 2],
                                "description": "Number of taps: 1=single, 2=double.",
                            },
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False,
                    },
                },
                _responses_lookup_tool(strict=True),
            ],
            id="gpt-mobile",
        ),
    ],
)
def test_gpt_build_tools_snapshots(
    build_tools: Callable[[], list[dict[str, Any]]],
    expected: list[dict[str, Any]],
) -> None:
    tools = build_tools()

    assert tools == expected
    assert _payload_bytes(tools) == _payload_bytes(expected)
