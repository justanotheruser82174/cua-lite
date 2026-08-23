"""Lite action-batch filtering through ``metadata.valid_actions``."""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import wrapper_item_props

from lite.agents.core.action_space.base import LiteDesktopActionSpace, LiteMobileActionSpace


@pytest.mark.parametrize(
    ("space", "valid_actions"),
    [
        (LiteDesktopActionSpace, ["click"]),
        (LiteMobileActionSpace, ["tap"]),
    ],
    ids=["desktop", "mobile"],
)
def test_lite_core_public_hook_narrows_the_batch_enum(space, valid_actions) -> None:
    schemas = space.get_tool_schemas()

    public = space.filter_tool_schemas_for_valid_actions(schemas, valid_actions)

    props = wrapper_item_props(public[0])
    assert props["action"]["enum"] == valid_actions


@pytest.mark.parametrize(
    ("space", "valid_actions"),
    [
        (LiteDesktopActionSpace, []),
        (LiteMobileActionSpace, []),
    ],
    ids=["desktop_empty", "mobile_empty"],
)
def test_lite_core_public_hook_drops_the_batch_when_no_action_survives(
    space,
    valid_actions,
) -> None:
    """An empty enum drops the whole wrapper schema; ``"enum": []`` is unusable."""
    schemas = space.get_tool_schemas()

    assert space.filter_tool_schemas_for_valid_actions(schemas, valid_actions) == []


def test_valid_actions_prunes_parameters_of_filtered_out_desktop_actions() -> None:
    space = LiteDesktopActionSpace
    full = space.get_tool_schemas()[0]
    assert {"keys", "text", "direction"} <= set(wrapper_item_props(full))

    filtered = space.filter_tool_schemas_for_valid_actions([full], ["click"])[0]
    props = wrapper_item_props(filtered)

    assert props["action"]["enum"] == ["click"]
    assert set(props) == {"action", "coordinate", "button", "clicks"}
    for orphan in ("keys", "text", "direction", "amount", "duration", "start_coordinate"):
        assert orphan not in props


def test_valid_actions_prunes_parameters_of_filtered_out_mobile_actions() -> None:
    space = LiteMobileActionSpace
    full = space.get_tool_schemas()[0]

    filtered = space.filter_tool_schemas_for_valid_actions([full], ["tap"])[0]
    props = wrapper_item_props(filtered)

    assert props["action"]["enum"] == ["tap"]
    assert "text" not in props
    assert "button" not in props


def test_full_valid_actions_leaves_the_wrapper_schema_byte_identical() -> None:
    """The pruning rebuild must be a no-op when nothing is filtered out."""
    for space in (LiteDesktopActionSpace, LiteMobileActionSpace):
        full = space.get_tool_schemas()[0]
        assert space.filter_tool_schemas_for_valid_actions(
            [full],
            sorted(space.get_declared_action_schema_names()),
        ) == [full]


def test_valid_actions_pruning_preserves_an_earlier_include_narrowing() -> None:
    """Filtering must intersect with the incoming enum, never widen it back."""
    space = LiteDesktopActionSpace
    narrowed = space.get_tool_schemas(include=["click", "type"])[0]

    out = space.filter_tool_schemas_for_valid_actions(
        [narrowed],
        ["click", "type", "scroll"],
    )[0]

    assert set(wrapper_item_props(out)["action"]["enum"]) == {"click", "type"}
