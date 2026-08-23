"""Cross-family schema surfaces keep ``valid_actions`` and extras orthogonal."""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    action_enum,
    assemble_for,
    extra_tool_names_table,
    wrapper_name,
)

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core.tools.schemas import tool_schema_name

# (adapter_key, platform, action valid_action, its native enum entry,
#  native semantic entry, the canonical extra schema that opens it)
SEMANTIC_FAMILY_MATRIX = [
    ("qwen3_vl@desktop@use", "desktop", "click", "left_click", "terminate", TERMINATE_SCHEMA),
    ("qwen3_vl@mobile@use", "mobile", "tap", "click", "open", OPEN_APP_SCHEMA),
    ("qwen3_5@desktop@use", "desktop", "click", "left_click", "answer", RESPONSE_SCHEMA),
    ("qwen3_5@mobile@use", "mobile", "tap", "click", "open", OPEN_APP_SCHEMA),
    ("qwen2_5_vl@desktop@use", "desktop", "click", "left_click", "terminate", TERMINATE_SCHEMA),
    ("qwen2_5_vl@mobile@use", "mobile", "tap", "click", "open", OPEN_APP_SCHEMA),
    ("evocua@desktop@use", "desktop", "click", "left_click", "terminate", TERMINATE_SCHEMA),
    ("fara@desktop@use", "desktop", "click", "left_click", "terminate", TERMINATE_SCHEMA),
]


@pytest.mark.parametrize(
    "adapter_key,platform,action,action_entry,semantic_entry,extra_schema",
    SEMANTIC_FAMILY_MATRIX,
    ids=[row[0] for row in SEMANTIC_FAMILY_MATRIX],
)
def test_valid_actions_and_extra_tools_are_orthogonal(
    adapter_key,
    platform,
    action,
    action_entry,
    semantic_entry,
    extra_schema,
) -> None:
    wrapper = wrapper_name(platform)

    def enum(valid_actions, extras):
        return action_enum(
            assemble_for(adapter_key, platform, valid_actions, extras),
            wrapper,
        )

    q1 = enum([action], [])
    assert action_entry in q1
    assert semantic_entry not in q1

    q2 = enum([action], [extra_schema])
    assert set(q2) == set(q1) | {semantic_entry}

    assert enum([], []) is None
    assert enum([], [extra_schema]) == [semantic_entry]


@pytest.mark.parametrize(
    "adapter_key,platform,action,action_entry,semantic_entry,extra_schema",
    SEMANTIC_FAMILY_MATRIX,
    ids=[row[0] for row in SEMANTIC_FAMILY_MATRIX],
)
def test_valid_actions_never_touches_native_semantic_entries(
    adapter_key,
    platform,
    action,
    action_entry,
    semantic_entry,
    extra_schema,
) -> None:
    """Holding ``extra_tools`` fixed, ``valid_actions`` never changes native semantic entries."""
    del action_entry
    wrapper = wrapper_name(platform)
    action_space_cls = type(AgentAdapterRegistry.get(adapter_key).action_space)
    semantic = frozenset(extra_tool_names_table(action_space_cls))
    assert semantic_entry in semantic

    for extras, expected in (([], set()), ([extra_schema], {semantic_entry})):
        for valid_actions in (None, [action], []):
            enum = action_enum(
                assemble_for(adapter_key, platform, valid_actions, extras),
                wrapper,
            )
            assert (set(enum or []) & semantic) == expected, (
                f"{adapter_key}: valid_actions={valid_actions} "
                f"extras={[tool_schema_name(s) for s in extras]}"
            )
