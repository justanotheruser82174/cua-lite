"""Qwen3-VL browser navigation wire checks for extra tool schemas."""

from __future__ import annotations

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteBrowserNavToolSet
from lite.core.tools.schemas import tool_schema_name

register_all()

_GOTO_SCHEMA, _BACK_SCHEMA = LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"])
_NAV_EXTRA_TOOLS = [_GOTO_SCHEMA, _BACK_SCHEMA]


def _md_browser(extra_tool_schemas=None) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tool_schemas or [],
        others={"id": "nav-wire"},
    )


class TestQwen3VLBrowserNavViaExtraTools:
    """qwen3_vl keeps nav out of its action space and surfaces it via extra tools."""

    def _adapter(self):
        return Qwen3VLDesktopUseAdapter(metadata=_md_browser(_NAV_EXTRA_TOOLS))

    def test_nav_absent_from_model_action_space(self):
        actions = ActionSpaceRegistry.get("qwen3_vl@browser").get_declared_action_schema_names()
        assert "goto" not in actions and "back" not in actions

    def test_extra_tools_surface_in_assembled_schemas(self):
        names = [tool_schema_name(s) for s in self._adapter()._assemble_tool_schemas()]
        assert "computer_use" in names
        assert "goto" in names and "back" in names

    def test_nav_renders_into_tools_wire_block(self):
        tools_text = self._adapter()._build_tools_section()
        assert '"goto"' in tools_text and '"back"' in tools_text
        assert '"url"' in tools_text
