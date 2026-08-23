"""Tests for core message content helper ownership and edge cases."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lite.core import messages as core_messages
from lite.core.messages import content as content_module
from lite.core.messages import selectors as selectors_module
from lite.core.tools.results import TOOL_RESULT_ERROR_SECTION_HEADER

ROOT = Path(__file__).resolve().parents[3]

CORE_CONTENT_HELPERS = {
    "first_image_content_part",
    "extract_first_text",
    "keep_model_visible_content",
    "message_has_error_feedback",
    "message_has_image",
    "tool_message_text_parts",
    "set_or_append_text",
}

PRIVATE_CONTENT_HELPERS = {"first_image_content_part", "tool_message_text_parts"}
PUBLIC_CORE_CONTENT_HELPERS = CORE_CONTENT_HELPERS - PRIVATE_CONTENT_HELPERS


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_core_content_module_owns_provider_free_content_helpers() -> None:
    content_path = ROOT / "lite" / "core" / "messages" / "content.py"
    defined = _defined_functions(content_path)

    for name in CORE_CONTENT_HELPERS:
        assert hasattr(content_module, name)
        assert name in defined


def test_core_messages_facade_exports_provider_free_content_helpers() -> None:
    for name in PUBLIC_CORE_CONTENT_HELPERS:
        assert getattr(core_messages, name) is getattr(content_module, name)

    for name in PRIVATE_CONTENT_HELPERS:
        assert not hasattr(core_messages, name)


def test_core_content_helpers_handle_string_content_without_crashing() -> None:
    message = {"role": "user", "content": "plain instruction"}

    assert core_messages.extract_first_text(message) == "plain instruction"
    assert content_module.first_image_content_part(message) is None
    assert core_messages.message_has_image(message) is False
    assert core_messages.message_has_error_feedback(message) is False


@pytest.mark.parametrize(
    ("window_helper", "empty_answer"),
    [
        (content_module.peel_system_message, (None, [])),
        (selectors_module.first_user_message, None),
        (selectors_module.instruction_text, ""),
    ],
)
def test_message_window_helpers_reject_none_instead_of_spelling_empty_twice(
    window_helper,
    empty_answer,
) -> None:
    assert window_helper([]) == empty_answer

    with pytest.raises(TypeError, match=r"expects a list of messages, got NoneType"):
        window_helper(None)

    assert content_module.require_message_list([], where="unit") is None


def test_set_or_append_text_replaces_string_content_with_text_part() -> None:
    message = {"role": "user", "content": "plain instruction"}

    core_messages.set_or_append_text(message, "summary")

    assert message["content"] == [{"type": "text", "text": "summary"}]


def test_legacy_projected_error_fallback_uses_owner_projection_helper(monkeypatch) -> None:
    sentinel_header = "## Owner error header sentinel:"
    monkeypatch.setattr(
        content_module,
        "text_has_projected_tool_result_error",
        lambda text: any(line == sentinel_header for line in text.splitlines()),
    )

    legacy_projected_message = {
        "role": "tool",
        "content": [{"type": "text", "text": f"{sentinel_header}\nclick failed"}],
    }
    old_literal_message = {
        "role": "tool",
        "content": [
            {
                "type": "text",
                "text": f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nclick failed",
            }
        ],
    }

    assert content_module.message_has_error_feedback(legacy_projected_message) is True
    assert content_module.message_has_error_feedback(old_literal_message) is False
