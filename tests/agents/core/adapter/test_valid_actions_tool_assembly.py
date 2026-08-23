"""Adapter assembly applies action-space ``valid_actions`` filtering."""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import LiteDesktopActionSpace, LiteMobileActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata

register_all()


@pytest.mark.parametrize(
    ("adapter_key", "platform", "space", "valid_actions"),
    [
        ("lite@desktop@use", "desktop", LiteDesktopActionSpace, ["click"]),
        ("lite@mobile@use", "mobile", LiteMobileActionSpace, ["tap"]),
    ],
    ids=["desktop_adapter", "mobile_adapter"],
)
def test_lite_core_adapter_assembly_uses_public_hook_parity(
    adapter_key,
    platform,
    space,
    valid_actions,
) -> None:
    adapter = AgentAdapterRegistry.get(
        adapter_key,
        metadata=LiteCUAMetadata(dims=(platform, "use"), valid_actions=valid_actions),
    )

    expected = space.filter_tool_schemas_for_valid_actions(
        space.get_tool_schemas(),
        valid_actions,
    )

    assert adapter._assemble_tool_schemas() == expected
