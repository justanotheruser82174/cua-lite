"""Browser navigation stays catalog-owned and outside browser action spaces."""

from __future__ import annotations

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.action_space.base import LiteBrowserActionSpace
from lite.core.tools.extra_tools import LiteBrowserNavToolSet
from lite.core.tools.schemas import tool_schema_name

register_all()


class TestLiteBrowserNavNotInActionSpace:
    """The six nav verbs are not native ``lite@browser`` action-space tools."""

    def test_registry_resolves_to_browser_action_space(self):
        assert isinstance(ActionSpaceRegistry.get("lite@browser"), LiteBrowserActionSpace)

    def test_all_six_nav_verbs_are_not_native_actions(self):
        actions = LiteBrowserActionSpace.get_declared_action_schema_names()
        assert LiteBrowserNavToolSet.get_tool_names().isdisjoint(actions)
        assert "click" in actions

    def test_nav_schemas_remain_catalog_only(self):
        browser_schemas = LiteBrowserActionSpace.get_tool_schemas()
        browser_schema_names = {tool_schema_name(s) for s in browser_schemas}
        for name in ("goto", "back"):
            assert name not in browser_schema_names
            assert LiteBrowserNavToolSet.get_tool_schema(name) is not None
