"""Tests for protocol-owned history-window content policy."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import lite.agents.core.protocol as protocol_api
from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.agents.core.protocol import window as protocol_window
from lite.core import messages as core_messages
from lite.core.messages import content as content_module

ROOT = Path(__file__).resolve().parents[4]

CORE_CONTENT_HELPERS = {
    "first_image_content_part",
    "extract_first_text",
    "keep_model_visible_content",
    "message_has_error_feedback",
    "message_has_image",
    "tool_message_text_parts",
    "set_or_append_text",
}

WINDOW_HELPERS = {
    "append_with_boundary_tool_projection",
    "collapse_image_observation",
    "evicted_tool_result_summary_text",
    "filter_history_content",
    "inject_summary_text",
    "join_role_tool_text",
    "keep_images_and_text",
    "keep_images_and_tool_text",
}

PROTOCOL_FACADE_WINDOW_HELPERS = {
    "append_with_boundary_tool_projection",
    "filter_history_content",
}


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_protocol_window_owns_only_protocol_window_helpers() -> None:
    path = ROOT / "lite" / "agents" / "core" / "protocol" / "window.py"
    assert _defined_functions(path) == WINDOW_HELPERS

    for name in CORE_CONTENT_HELPERS:
        assert hasattr(content_module, name)
        assert not hasattr(protocol_window, name)
    for name in WINDOW_HELPERS:
        assert hasattr(protocol_window, name)
        assert not hasattr(core_messages, name)
    for name in PROTOCOL_FACADE_WINDOW_HELPERS:
        assert getattr(protocol_api, name) is getattr(protocol_window, name)


def test_protocol_package_facade_does_not_reexport_core_message_helpers() -> None:
    for name in CORE_CONTENT_HELPERS:
        assert not hasattr(protocol_api, name)
    for name in WINDOW_HELPERS - PROTOCOL_FACADE_WINDOW_HELPERS:
        assert not hasattr(protocol_api, name)
    for name in PROTOCOL_FACADE_WINDOW_HELPERS:
        assert getattr(protocol_api, name) is getattr(protocol_window, name)


def test_window_policy_is_not_owned_by_core_messages() -> None:
    core_path = ROOT / "lite" / "core" / "messages" / "content.py"
    window_path = ROOT / "lite" / "agents" / "core" / "protocol" / "window.py"

    assert "filter_history_content" not in _defined_functions(core_path)
    assert "inject_summary_text" not in _defined_functions(core_path)
    assert "join_role_tool_text" not in _defined_functions(core_path)
    assert "keep_images_and_text" not in _defined_functions(core_path)
    assert "keep_images_and_tool_text" not in _defined_functions(core_path)
    assert _defined_functions(window_path) == WINDOW_HELPERS


def test_window_content_helper_keeps_tool_empty_text_without_mutating() -> None:
    message = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": [
            {"type": "text", "text": ""},
            {"type": "metadata", "data": {"returncode": 0}},
            {"type": "other", "value": "drop"},
        ],
    }
    original = copy.deepcopy(message)

    assert protocol_window.keep_images_and_text(message)["content"] == [
        {"type": "text", "text": ""},
        {"type": "metadata", "data": {"returncode": 0}},
    ]
    assert message == original


def test_protocol_window_helpers_handle_string_content_without_crashing() -> None:
    message = {"role": "user", "content": "plain instruction"}

    assert protocol_window.keep_images_and_tool_text(message) == message
    assert protocol_window.keep_images_and_text(message) == message


def test_inject_summary_text_preserves_tool_string_content() -> None:
    message = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "tool result",
    }

    protocol_window.inject_summary_text(message, "history")

    assert message["content"] == [
        {"type": "text", "text": "history"},
        {"type": "text", "text": "tool result"},
    ]


def test_window_policy_strips_ordinary_image_text_but_keeps_tool_payloads() -> None:
    user = {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "ambient observation"},
            {"type": "metadata", "data": {"page_title": "Example"}},
        ],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": [
            {"type": "image", "index": 1},
            {"type": "text", "text": ""},
            {"type": "metadata", "data": {"returncode": 0}},
        ],
    }

    assert protocol_window.filter_history_content(
        user,
        strip_regular_image_text=True,
    )["content"] == [
        {"type": "image", "index": 0},
        {"type": "metadata", "data": {"page_title": "Example"}},
    ]
    assert protocol_window.filter_history_content(
        tool,
        strip_regular_image_text=True,
    )["content"] == [
        {"type": "image", "index": 1},
        {"type": "text", "text": ""},
        {"type": "metadata", "data": {"returncode": 0}},
    ]


def test_window_policy_ignores_non_dict_content_parts_when_stripping() -> None:
    message = {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            "legacy opaque part",
            {"type": "metadata", "data": {"page_title": "Example"}},
            {"type": "text", "text": "ambient observation"},
        ],
    }

    filtered = protocol_window.filter_history_content(
        message,
        strip_regular_image_text=True,
    )

    assert filtered["content"] == [
        {"type": "image", "index": 0},
        {"type": "metadata", "data": {"page_title": "Example"}},
    ]
    assert message["content"][1] == "legacy opaque part"


def test_window_collapse_image_observation_preserves_protocol_payloads() -> None:
    user = {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "ambient observation"},
            {"type": "metadata", "data": {"page_title": "Example"}},
        ],
    }
    tool = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": [
            {"type": "image", "index": 1},
            {"type": "text", "text": ""},
            {"type": "metadata", "data": {"is_error": True}},
        ],
    }
    original_tool = copy.deepcopy(tool)

    assert protocol_window.collapse_image_observation(
        user,
        "collapsed",
        strip_regular_image_text=True,
    )["content"] == [
        {"type": "text", "text": "collapsed"},
        {"type": "metadata", "data": {"page_title": "Example"}},
    ]
    assert protocol_window.collapse_image_observation(
        user,
        "collapsed",
        strip_regular_image_text=False,
    )["content"] == [
        {"type": "text", "text": "collapsed"},
        {"type": "text", "text": "ambient observation"},
        {"type": "metadata", "data": {"page_title": "Example"}},
    ]
    assert protocol_window.collapse_image_observation(
        tool,
        "collapsed",
        strip_regular_image_text=True,
    )["content"] == [
        {"type": "text", "text": "collapsed"},
        {"type": "text", "text": ""},
        {"type": "metadata", "data": {"is_error": True}},
    ]
    assert tool == original_tool


def test_window_boundary_projection_rewrites_only_leading_tool_results() -> None:
    tool = {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": [{"type": "text", "text": "tool result"}],
    }

    result = [{"role": "assistant", "content": []}]
    protocol_window.append_with_boundary_tool_projection(result, [tool])
    assert result[-1]["role"] == "tool"
    assert result[-1]["tool_call_id"] == "call_0"

    result = []
    protocol_window.append_with_boundary_tool_projection(result, [tool])
    assert result[-1]["role"] == "user"
    assert "call_id" not in result[-1]
    assert "tool_call_id" not in result[-1]
    assert result[-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "tool result"}],
    }
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_0"


def test_window_evicted_tool_result_summary_keeps_text_only_and_errors() -> None:
    text_only = {
        "role": "tool",
        "tool_call_id": "call_text",
        "content": [{"type": "text", "text": "stdout"}],
    }
    ordinary_image = {
        "role": "tool",
        "tool_call_id": "call_image",
        "content": [
            {"type": "image", "index": 1},
            {"type": "text", "text": "ordinary image observation"},
        ],
    }
    image_error = build_tool_result_message(
        "call_error",
        (2,),
        metadata={"kind": "gui"},
        error="click failed",
    )

    assert protocol_window.evicted_tool_result_summary_text(
        [text_only, ordinary_image, image_error]
    ) == "stdout\n## Error from previous action:\nclick failed"

    metadata_only_marker = {
        "role": "tool",
        "tool_call_id": "call_error",
        "content": [
            {"type": "image", "index": 2},
            {"type": "text", "text": "click failed"},
            {"type": "metadata", "data": {"is_error": True}},
        ],
    }
    assert content_module.message_metadata(metadata_only_marker).get("is_error") is True
    assert core_messages.message_has_error_feedback(metadata_only_marker) is False
    assert protocol_window.evicted_tool_result_summary_text(
        [metadata_only_marker]
    ) == ""
