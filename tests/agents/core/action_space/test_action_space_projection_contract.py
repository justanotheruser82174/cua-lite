"""Projection invariants for action-space tool-call conversion.

The canonical Lite side of the contract is nested
``{"type": "function", "function": {"name": ..., "arguments": ...}}``. The
agent/provider side may be a bare action, one wrapper dialect, or
provider-native action objects. Spaces whose agent side differs from Lite
project out of canonical nested shape, and parse restores a pre-stamp canonical
Lite call. Native Lite spaces use bare Lite function-call agent wire, not
canonical pass-through.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/action_space/test_action_space_projection_contract.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

from typing import Any

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import ActionSpaceRegistry
from lite.core.tools.calls import make_tool_call

register_all()


def _canonical_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


DESKTOP_ACTIONS = [
    {"action": "click", "coordinate": [500, 300]},
    {"action": "type", "text": "hello"},
]
DESKTOP_BATCH = _canonical_call(
    "computer",
    {"actions": DESKTOP_ACTIONS},
)

MOBILE_ACTIONS = [
    {"action": "tap", "coordinate": [500, 300], "clicks": 1},
    {"action": "type", "text": "hello"},
]
MOBILE_BATCH = _canonical_call(
    "mobile",
    {"actions": MOBILE_ACTIONS},
)

POINT_CALL = _canonical_call("point", {"coordinate": [500, 300]})


def _is_nested_canonical_call(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and call.get("type") == "function"
        and isinstance(call.get("function"), dict)
        and isinstance(call["function"].get("name"), str)
        and isinstance(call["function"].get("arguments"), dict)
    )


def _is_bare_function_call(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and set(call) == {"name", "arguments"}
        and isinstance(call["name"], str)
        and isinstance(call["arguments"], dict)
    )


def _assert_restored_canonical_surface(
    restored: list[dict[str, Any]],
    *,
    top_name: str,
    arguments: dict[str, Any],
) -> None:
    assert len(restored) == 1
    call = restored[0]
    assert _is_nested_canonical_call(call)
    assert "id" not in call
    assert call["function"]["name"] == top_name
    assert call["function"]["arguments"] == arguments


@pytest.mark.parametrize(
    ("key", "canonical_calls", "expected_projected", "top_name", "arguments", "kwargs"),
    [
        pytest.param(
            "lite@desktop",
            [DESKTOP_BATCH],
            [
                {
                    "name": "computer",
                    "arguments": {"actions": DESKTOP_ACTIONS},
                }
            ],
            "computer",
            {"actions": DESKTOP_ACTIONS},
            {},
            id="lite-desktop-base-fanout",
        ),
        pytest.param(
            "qwen3_vl@desktop",
            [DESKTOP_BATCH],
            [
                {
                    "name": "computer_use",
                    "arguments": {"action": "left_click", "coordinate": [500, 300]},
                },
                {"name": "computer_use", "arguments": {"action": "type", "text": "hello"}},
            ],
            "computer",
            {"actions": DESKTOP_ACTIONS},
            {},
            id="qwen-wrapper",
        ),
        pytest.param(
            "gpt@desktop",
            [DESKTOP_BATCH],
            [
                {"type": "click", "x": 1000, "y": 300},
                {"type": "type", "text": "hello"},
            ],
            "computer",
            {"actions": DESKTOP_ACTIONS},
            {"resolution": (2000, 1000)},
            id="gpt-desktop-provider-native",
        ),
        pytest.param(
            "claude@desktop",
            [DESKTOP_BATCH],
            [
                {"action": "left_click", "coordinate": [1000, 300]},
                {"action": "type", "text": "hello"},
            ],
            "computer",
            {"actions": DESKTOP_ACTIONS},
            {"resolution": (2000, 1000)},
            id="claude-desktop-provider-native",
        ),
        pytest.param(
            "gpt@mobile",
            [MOBILE_BATCH],
            [
                {"name": "tap", "arguments": {"x": 1000, "y": 300, "clicks": 1}},
                {"name": "type", "arguments": {"text": "hello"}},
            ],
            "mobile",
            {"actions": MOBILE_ACTIONS},
            {"resolution": (2000, 1000)},
            id="gpt-mobile-provider-flat",
        ),
        pytest.param(
            "claude@mobile",
            [MOBILE_BATCH],
            [
                {
                    "name": "tap",
                    "arguments": {"coordinate": [1000, 300], "clicks": 1},
                },
                {"name": "type", "arguments": {"text": "hello"}},
            ],
            "mobile",
            {"actions": MOBILE_ACTIONS},
            {"resolution": (2000, 1000)},
            id="claude-mobile-provider-flat",
        ),
        pytest.param(
            "qwen3_5@desktop@point",
            [POINT_CALL],
            [
                {
                    "name": "computer_use",
                    "arguments": {"action": "left_click", "coordinate": [500, 300]},
                }
            ],
            "point",
            {"coordinate": [500, 300]},
            {},
            id="grounding-plural-override",
        ),
    ],
)
def test_render_projection_leaves_canonical_surface_then_parse_restores_it(
    key: str,
    canonical_calls: list[dict[str, Any]],
    expected_projected: list[dict[str, Any]],
    top_name: str,
    arguments: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    space = ActionSpaceRegistry.get(key)

    projected = space.convert_tool_calls_to_agent(canonical_calls, **kwargs)

    assert projected
    assert projected == expected_projected
    assert all(not _is_nested_canonical_call(call) for call in projected)
    if key.startswith("lite@"):
        assert all(_is_bare_function_call(call) for call in projected)

    restored = space.convert_tool_calls_from_agent(projected, **kwargs)

    _assert_restored_canonical_surface(
        restored,
        top_name=top_name,
        arguments=arguments,
    )


@pytest.mark.parametrize(
    ("key", "kwargs", "expected_projected"),
    [
        pytest.param(
            "gpt@mobile",
            {"resolution": (1000, 1000)},
            [{"name": "bash", "arguments": {"command": "pwd"}}],
            id="gpt-mobile",
        ),
        pytest.param(
            "claude@mobile",
            {"resolution": (1000, 1000)},
            [{"name": "bash", "arguments": {"command": "pwd"}}],
            id="claude-mobile",
        ),
    ],
)
def test_provider_extra_tools_unwrap_then_restore_canonical_storage(
    key: str,
    kwargs: dict[str, Any],
    expected_projected: list[dict[str, Any]],
) -> None:
    space = ActionSpaceRegistry.get(key)
    canonical = make_tool_call("bash", {"command": "pwd"})

    projected = space.convert_tool_calls_to_agent([canonical], **kwargs)

    assert projected == expected_projected

    restored = space.convert_tool_calls_from_agent(projected, **kwargs)

    _assert_restored_canonical_surface(
        restored,
        top_name="bash",
        arguments={"command": "pwd"},
    )
