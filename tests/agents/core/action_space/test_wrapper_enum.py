"""
Structural tests for the shared provider wrapper action-enum owner.

``filter_wrapper_action_enum`` is the ONE mechanical helper model families call
to narrow a provider-native wrapper tool (``computer_use`` / ``mobile_use``).
These tests pin the surgery only — enum narrowing, wrapper drop on empty enum,
untouched siblings, no caller mutation, bullet pruning, and the
"Required only by `action=...`" prose rewrite. Family policy (which action
values survive) lives in the family action spaces and is tested there.

Run:
    uv run pytest tests/agents/core/action_space/test_wrapper_enum.py -v
"""

from __future__ import annotations

import copy
from typing import Any

from lite.agents.core.action_space.utils.wrapper_enum import filter_wrapper_action_enum


def _wrapper_schema(
    name: str = "computer_use",
    *,
    enum: list[str] | None = None,
    action_description: str | None = None,
    extra_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical Lite tool schema shaped like a provider wrapper tool."""
    enum = enum if enum is not None else ["left_click", "type", "scroll", "terminate"]
    if action_description is None:
        action_description = "\n".join(
            [
                "The action to perform. The available actions are:",
                "* `left_click`: Click the left mouse button.",
                "* `type`: Type a string of text.",
                "* `scroll`: Scroll the mouse wheel.",
                "* `terminate`: Terminate the current task.",
            ]
        )
    properties: dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": enum,
            "description": action_description,
        }
    }
    properties.update(extra_properties or {})
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Provider wrapper tool.",
            "parameters": {"type": "object", "properties": properties},
        },
    }


def _action_property(schema: dict[str, Any]) -> dict[str, Any]:
    return schema["function"]["parameters"]["properties"]["action"]


# -----------------------------------------------------------------------------
# Enum narrowing
# -----------------------------------------------------------------------------

def test_enum_keeps_only_predicate_matches_in_original_order():
    schemas = [_wrapper_schema()]

    filtered = filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value in {"scroll", "left_click"},
    )

    assert len(filtered) == 1
    assert _action_property(filtered[0])["enum"] == ["left_click", "scroll"]


def test_wrapper_schema_is_dropped_when_no_action_value_survives():
    schemas = [_wrapper_schema(), _wrapper_schema("open_app", enum=["a"])]

    filtered = filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: False,
    )

    assert [schema["function"]["name"] for schema in filtered] == ["open_app"]


def test_schema_without_action_enum_is_passed_through():
    schema = _wrapper_schema()
    del _action_property(schema)["enum"]

    filtered = filter_wrapper_action_enum(
        [schema],
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: False,
    )

    assert filtered == [schema]


# -----------------------------------------------------------------------------
# Blast radius: other schemas, caller dicts
# -----------------------------------------------------------------------------

def test_non_wrapper_schemas_are_untouched_and_keep_their_order():
    other_before = {
        "type": "function",
        "function": {
            "name": "report_infeasible",
            "parameters": {
                "type": "object",
                "properties": {"action": {"enum": ["left_click", "type"]}},
            },
        },
    }
    schemas = [copy.deepcopy(other_before), _wrapper_schema(), _wrapper_schema("mobile_use")]

    filtered = filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value == "type",
    )

    assert [schema["function"]["name"] for schema in filtered] == [
        "report_infeasible",
        "computer_use",
        "mobile_use",
    ]
    # Same-named ``action`` enums on other tools are not this helper's business.
    assert filtered[0] == other_before
    assert _action_property(filtered[2])["enum"] == ["left_click", "type", "scroll", "terminate"]


def test_input_schemas_are_not_mutated():
    schemas = [_wrapper_schema()]
    before = copy.deepcopy(schemas)

    filtered = filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value == "type",
    )

    assert schemas == before
    assert _action_property(filtered[0])["enum"] == ["type"]


def test_unreadable_schema_is_passed_through_untouched():
    schemas = [{"no_function_key": True}, _wrapper_schema()]

    filtered = filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value == "type",
    )

    assert filtered[0] == {"no_function_key": True}
    assert _action_property(filtered[1])["enum"] == ["type"]


# -----------------------------------------------------------------------------
# Description bullet pruning
# -----------------------------------------------------------------------------

def test_bullet_lines_for_removed_action_values_are_pruned():
    filtered = filter_wrapper_action_enum(
        [_wrapper_schema()],
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value in {"left_click", "terminate"},
    )

    assert _action_property(filtered[0])["description"] == "\n".join(
        [
            "The action to perform. The available actions are:",
            "* `left_click`: Click the left mouse button.",
            "* `terminate`: Terminate the current task.",
        ]
    )


def test_non_bullet_description_lines_survive_pruning():
    description = "\n".join(
        [
            "The action to perform.",
            "* `left_click`: Click the left mouse button.",
            "* `type`: Type a string of text.",
            "Coordinates are absolute.",
        ]
    )
    filtered = filter_wrapper_action_enum(
        [_wrapper_schema(enum=["left_click", "type"], action_description=description)],
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value == "left_click",
    )

    assert _action_property(filtered[0])["description"] == "\n".join(
        [
            "The action to perform.",
            "* `left_click`: Click the left mouse button.",
            "Coordinates are absolute.",
        ]
    )


# -----------------------------------------------------------------------------
# "Required only by `action=...`" prose rewriting
# -----------------------------------------------------------------------------

def _text_description(filtered: list[dict[str, Any]]) -> str:
    return filtered[0]["function"]["parameters"]["properties"]["text"]["description"]


def _filter_with_text_prose(prose: str, kept: set[str]) -> list[dict[str, Any]]:
    schemas = [
        _wrapper_schema(
            enum=["type", "answer", "open", "scroll"],
            action_description="The action to perform.",
            extra_properties={"text": {"type": "string", "description": prose}},
        )
    ]
    return filter_wrapper_action_enum(
        schemas,
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value in kept,
    )


def test_required_only_by_prose_keeps_a_single_surviving_ref():
    filtered = _filter_with_text_prose(
        "Required only by `action=type`, `action=open`, and `action=answer`.",
        {"type"},
    )

    assert _text_description(filtered) == "Required only by `action=type`."


def test_required_only_by_prose_joins_two_surviving_refs_with_and():
    filtered = _filter_with_text_prose(
        "Required only by `action=type`, `action=open`, and `action=answer`.",
        {"type", "answer"},
    )

    assert _text_description(filtered) == "Required only by `action=type` and `action=answer`."


def test_required_only_by_prose_joins_three_surviving_refs_with_oxford_comma():
    filtered = _filter_with_text_prose(
        "Required only by `action=type`, `action=open`, and `action=answer`.",
        {"type", "open", "answer"},
    )

    assert _text_description(filtered) == (
        "Required only by `action=type`, `action=open`, and `action=answer`."
    )


def test_required_only_by_clause_is_dropped_when_no_ref_survives():
    filtered = _filter_with_text_prose(
        "The text to enter. Required only by `action=type` and `action=answer`. Keep it short.",
        {"scroll"},
    )

    assert _text_description(filtered) == "The text to enter. Keep it short."


def test_prose_becomes_empty_when_the_clause_was_the_whole_description():
    filtered = _filter_with_text_prose(
        "Required only by `action=type` and `action=answer`.",
        {"scroll"},
    )

    assert _text_description(filtered) == ""


def test_prose_without_action_refs_is_left_alone():
    filtered = _filter_with_text_prose("The x,y coordinates for mouse actions.", {"type"})

    assert _text_description(filtered) == "The x,y coordinates for mouse actions."


def test_action_property_prose_is_not_rewritten_as_a_sibling():
    # The ``action`` description owns the bullet list; it must not also be run
    # through the sibling "Required only by" rewrite.
    description = "Required only by `action=scroll`.\n* `type`: Type a string of text."
    filtered = filter_wrapper_action_enum(
        [_wrapper_schema(enum=["type", "scroll"], action_description=description)],
        wrapper_tool_name="computer_use",
        keep_action_value=lambda value: value == "type",
    )

    assert _action_property(filtered[0])["description"] == description
