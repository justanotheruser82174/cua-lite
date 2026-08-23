"""
Tests for the extra_tools mechanism.

Covers:
  1. Browser registry keys resolve to the desktop action_space class
  2. Qwen3VL adapter: extra tool schemas appear in <tools> section
  3. Qwen3VL adapter: extra tool calls pass through conversion unchanged (both directions)
  4. Name collision guard: ValueError on overlap with standard action names

Run:
    uv run pytest tests/agents/models/qwen3_vl/test_extra_tools.py -v
"""

from __future__ import annotations

import json

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.qwen3_vl.action_space import Qwen3VLDesktopActionSpace
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

register_all()


def _md(extra_tools=None, valid_actions=None):
    """Test helper — build a desktop-navigation LiteCUAMetadata."""
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tools or [],
        valid_actions=valid_actions,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BACK_SCHEMA = make_tool_schema(
    "back",
    description="Navigate to the previous page in browser history.",
    parameters={"type": "object", "properties": {}, "required": []},
)

GOTO_SCHEMA = make_tool_schema(
    "goto",
    description="Navigate to a URL.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL to navigate to."}},
        "required": ["url"],
    },
)

EXTRA_TOOLS = [BACK_SCHEMA, GOTO_SCHEMA]

CLICK_INDEX_SCHEMA = make_tool_schema(
    "click",
    description="Click a DOM element by index.",
    parameters={
        "type": "object",
        "properties": {"index": {"type": "integer"}},
        "required": ["index"],
    },
)

# ---------------------------------------------------------------------------
# 1. Browser registry keys resolve to the desktop action_space class
# ---------------------------------------------------------------------------


class TestBrowserRegistryRoutesToDesktop:
    def test_registry_lite_browser(self):
        instance = ActionSpaceRegistry.get("lite@browser")
        assert isinstance(instance, LiteDesktopActionSpace)

    def test_registry_qwen3_vl_browser(self):
        instance = ActionSpaceRegistry.get("qwen3_vl@browser")
        assert isinstance(instance, Qwen3VLDesktopActionSpace)

    def test_no_back_goto_in_browser_actions(self):
        # ``back`` / ``goto`` only land in extra_tools — they are not part
        # of the standard desktop action vocabulary.
        actions = LiteDesktopActionSpace.get_declared_action_schema_names()
        assert "back" not in actions
        assert "goto" not in actions


# ---------------------------------------------------------------------------
# 2. Extra tools in system prompt / tools section
# ---------------------------------------------------------------------------


class TestExtraToolsInToolsSection:
    def test_extra_tools_appear_in_tools_section(self):
        adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=EXTRA_TOOLS))
        tools_text = adapter._build_tools_section()
        assert '"back"' in tools_text
        assert '"goto"' in tools_text

    def test_standard_tools_still_present(self):
        adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=EXTRA_TOOLS))
        tools_text = adapter._build_tools_section()
        assert '"computer_use"' in tools_text

    def test_no_extra_tools_same_as_before(self):
        with_extra = Qwen3VLDesktopUseAdapter()
        without_extra = Qwen3VLDesktopUseAdapter()
        assert with_extra._build_tools_section() == without_extra._build_tools_section()

    def test_render_closure_uses_resolved_extra_tool_schemas_only(self):
        """Adapter render closure: persisted LiteSample metadata carries
        resolved schemas, not a separate ``extra_tools`` selector. A saved row
        without ``extra_tool_schemas`` must not surface env extras offline."""
        with_schema = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=[GOTO_SCHEMA]))
        without_schema = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=[]))

        assert not hasattr(with_schema.metadata, "extra_tools")
        assert '"goto"' in with_schema._build_tools_section()
        assert '"goto"' not in without_schema._build_tools_section()

    def test_extra_schemas_are_valid_json(self):
        adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=EXTRA_TOOLS))
        tools_text = adapter._build_tools_section()
        # Extract lines between <tools> and </tools>
        in_tools = False
        for line in tools_text.splitlines():
            if line.strip() == "<tools>":
                in_tools = True
                continue
            if line.strip() == "</tools>":
                break
            if in_tools and line.strip():
                json.loads(line)  # should not raise


# ---------------------------------------------------------------------------
# 4. convert_message_to_agent: extra tool calls pass through
# ---------------------------------------------------------------------------


class TestConvertMessageToAgent:
    def setup_method(self):
        self.adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=EXTRA_TOOLS))

    def test_standard_tool_call_converted(self):
        """Standard computer batch should be converted to computer_use."""
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "clicking"}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                ),
            ],
        }
        result = self.adapter.convert_message_to_agent(msg)
        tc = result["tool_calls"][0]
        assert tc["name"] == "computer_use"
        assert tc["arguments"]["action"] == "left_click"

    def test_extra_tool_call_passed_through(self):
        """Extra 'back' tool call should pass through unchanged."""
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "going back"}],
            "tool_calls": [
                make_tool_call("back", {}),
            ],
        }
        result = self.adapter.convert_message_to_agent(msg)
        tc = result["tool_calls"][0]
        assert tc["name"] == "back"
        assert tc["arguments"] == {}

    def test_mixed_standard_and_extra(self):
        """Mixed standard batch + extra tool calls: standard converted, extra passed through."""
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "do stuff"}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [100, 200]}]},
                ),
                make_tool_call("goto", {"url": "https://example.com"}),
            ],
        }
        result = self.adapter.convert_message_to_agent(msg)
        assert len(result["tool_calls"]) == 2
        # First: click -> computer_use
        assert result["tool_calls"][0]["name"] == "computer_use"
        # Second: goto passed through
        assert result["tool_calls"][1]["name"] == "goto"
        assert result["tool_calls"][1]["arguments"]["url"] == "https://example.com"

    def test_no_extra_tools_same_behavior(self):
        """Without extra_tools, all tool calls go through action space."""
        adapter = Qwen3VLDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [100, 200]}]},
                ),
            ],
        }
        result = adapter.convert_message_to_agent(msg)
        assert result["tool_calls"][0]["name"] == "computer_use"


# ---------------------------------------------------------------------------
# 5. convert_message_from_agent: extra tool calls pass through
# ---------------------------------------------------------------------------


class TestConvertMessageFromAgent:
    def setup_method(self):
        self.adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=EXTRA_TOOLS))

    def test_standard_tool_call_converted(self):
        """computer_use -> length-1 computer batch in CUA-lite."""
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "computer_use",
                    "arguments": {"action": "left_click", "coordinate": [500, 300]},
                },
            ],
        }
        result = self.adapter.convert_message_from_agent(msg)
        tc = result["tool_calls"][0]
        assert tool_call_name(tc) == "computer"
        assert tool_call_arguments(tc)["actions"] == [
            {"action": "click", "coordinate": [500, 300]},
        ]

    def test_extra_tool_call_passed_through(self):
        """'back' passes through from_agent unchanged."""
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"name": "back", "arguments": {}},
            ],
        }
        result = self.adapter.convert_message_from_agent(msg)
        tc = result["tool_calls"][0]
        assert tool_call_name(tc) == "back"

    def test_mixed_standard_and_extra(self):
        """Mixed: computer_use converted, goto passed through."""
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"name": "computer_use", "arguments": {"action": "type", "text": "hello"}},
                {"name": "goto", "arguments": {"url": "https://test.com"}},
            ],
        }
        result = self.adapter.convert_message_from_agent(msg)
        assert tool_call_name(result["tool_calls"][0]) == "computer"
        assert tool_call_arguments(result["tool_calls"][0])["actions"] == [
            {"action": "type", "text": "hello"},
        ]
        assert tool_call_name(result["tool_calls"][1]) == "goto"


# ---------------------------------------------------------------------------
# 6. Name collision guard
# ---------------------------------------------------------------------------


class TestNameCollisionGuard:
    def test_collision_with_standard_action_raises(self):
        """Extra tool named 'computer_use' collides with Qwen3VL standard action."""
        collision_schema = make_tool_schema(
            "computer_use",
            description="Colliding name.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        with pytest.raises(ValueError, match="collide"):
            Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=[collision_schema]))

    def test_no_collision_ok(self):
        """Non-colliding extra tools should work fine."""
        adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=EXTRA_TOOLS))
        assert len(adapter.metadata.extra_tool_schemas) == 2

    def test_action_child_action_name_does_not_collide_with_lite_wrapper(self):
        """Only model-facing top-level tool names collide with extras."""
        adapter = AgentAdapterRegistry.get(
            "lite@desktop@use",
            metadata=_md(extra_tools=[CLICK_INDEX_SCHEMA]),
        )
        msg = {
            "role": "assistant",
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [10, 20]}]},
                ),
            ],
        }

        assert adapter.convert_message_to_agent(msg)["tool_calls"] == msg["tool_calls"]

    def test_same_name_action_extra_routes_by_schema_shape_on_render(self):
        adapter = Qwen3VLDesktopUseAdapter(metadata=_md(extra_tools=[CLICK_INDEX_SCHEMA]))

        standalone = adapter.convert_message_to_agent(
            {
                "role": "assistant",
                "tool_calls": [make_tool_call("click", {"index": 7})],
            }
        )
        action = adapter.convert_message_to_agent(
            {
                "role": "assistant",
                "tool_calls": [make_tool_call("click", {"coordinate": [10, 20]})],
            }
        )

        assert standalone["tool_calls"] == [{"name": "click", "arguments": {"index": 7}}]
        assert action["tool_calls"] == [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [10, 20]},
            }
        ]


# ---------------------------------------------------------------------------
# 7. valid_actions filtering
# ---------------------------------------------------------------------------


class TestValidActions:
    """Test valid_actions filtering for Qwen3VL adapters.

    Qwen3VL uses a single ``computer_use`` tool with an ``action`` enum.
    valid_actions uses GUI CUA-Lite names (click, type, scroll, ...) and filters
    ONLY those. The native finish spellings (answer/terminate) are orthogonal:
    they appear iff the matching canonical ``extra_tool_schemas`` entry is
    active, so with no finish extra they are absent from every render below.
    """

    def test_valid_actions_filters_enum(self):
        """Only specified actions (mapped to Qwen3VL names) appear in enum."""
        adapter = Qwen3VLDesktopUseAdapter(
            metadata=_md(valid_actions=["click", "type"]),
        )
        tools_text = adapter._build_tools_section()
        # click -> left_click + right_click + ...
        # No finish extra active → no native finish spelling advertised.
        assert '"left_click"' in tools_text
        assert '"type"' in tools_text
        assert '"answer"' not in tools_text
        assert '"terminate"' not in tools_text
        assert '"scroll"' not in tools_text

    def test_valid_actions_none_exposes_all(self):
        """valid_actions=None exposes every GUI action; the native finish
        spellings stay closed until a finish extra activates them."""
        adapter = Qwen3VLDesktopUseAdapter()
        tools_text = adapter._build_tools_section()
        assert '"left_click"' in tools_text
        assert '"terminate"' not in tools_text
        assert '"answer"' not in tools_text
        assert '"scroll"' in tools_text

    def test_valid_actions_with_extra_tools(self):
        """valid_actions filters action enum; extra_tools still appended."""
        adapter = Qwen3VLDesktopUseAdapter(
            metadata=_md(valid_actions=["click"], extra_tools=EXTRA_TOOLS),
        )
        tools_text = adapter._build_tools_section()
        # Standard: click -> left_click etc.
        assert '"left_click"' in tools_text
        # back/goto are nav extras, not finish extras → finish stays closed.
        assert '"answer"' not in tools_text
        assert '"terminate"' not in tools_text
        # Extra: back and goto still present as separate tools
        assert '"back"' in tools_text
        assert '"goto"' in tools_text

    def test_valid_actions_filters_descriptions(self):
        """Action descriptions are also filtered to match the enum."""
        adapter = Qwen3VLDesktopUseAdapter(
            metadata=_md(valid_actions=["click", "type"]),
        )
        tools_text = adapter._build_tools_section()
        assert "left_click" in tools_text
        assert "* `terminate`" not in tools_text
        assert "* `answer`" not in tools_text
        assert "* `scroll`" not in tools_text

    def test_filter_uses_action_space_class(self):
        """The adapter dispatches through the public action-space hook."""
        from unittest.mock import patch

        adapter = Qwen3VLDesktopUseAdapter(
            metadata=_md(valid_actions=["click"]),
        )
        # Patch the classmethod on the actual action space class to verify dispatch
        hook_name = "filter_tool_schemas_for_valid_actions"
        original = getattr(type(adapter.action_space), hook_name)
        with patch.object(type(adapter.action_space), hook_name, wraps=original) as mock:
            adapter._build_tools_section()
            mock.assert_called_once()

    def test_adapter_uses_copy_module(self):
        """Adapter helpers use module-level copy, not a local import."""
        import lite.agents.models.qwen3_vl.adapter as mod

        assert hasattr(mod, "copy"), "copy should be imported at module level"

    def test_mobile_adapter_valid_actions_filters(self):
        """Mobile action space supports the public valid-actions hook."""
        from lite.agents.models.qwen3_vl.adapter import Qwen3VLMobileUseAdapter

        adapter = Qwen3VLMobileUseAdapter(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
                valid_actions=["tap", "type"],
            ),
        )
        tools_text = adapter._build_tools_section()
        # tap -> click.
        assert '"click"' in tools_text
        assert '"type"' in tools_text
        assert '"answer"' not in tools_text
        assert '"terminate"' not in tools_text
        assert '"swipe"' not in tools_text

    def test_get_tool_schemas_include(self):
        """BaseActionSpace.get_tool_schemas(include=...) filters correctly."""
        from lite.agents.core.action_space.base import LiteDesktopActionSpace

        all_schemas = LiteDesktopActionSpace.get_tool_schemas()
        filtered = LiteDesktopActionSpace.get_tool_schemas(include=["click", "type"])
        assert len(filtered) == 1
        assert tool_schema_name(filtered[0]) == "computer"
        action_enum = tool_schema_parameters(filtered[0])["properties"]["actions"]["items"][
            "properties"
        ]["action"]["enum"]
        assert set(action_enum) == {"click", "type"}
        assert len(all_schemas) == 1
        all_enum = tool_schema_parameters(all_schemas[0])["properties"]["actions"]["items"][
            "properties"
        ]["action"]["enum"]
        assert set(action_enum) < set(all_enum)
