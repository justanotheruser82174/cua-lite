"""
Tests for BaseActionSpace and ActionSpaceRegistry.

Covers:
  1. Registry: get, get_class, list, list_patterns
  2. get_tool_schemas, get_declared_action_schema_names, get_tool_schema
  3. get_registry_key
  4. format_tool_call_as_text, format_tool_calls_as_text, format_message_as_text

Run:
    uv run pytest tests/agents/core/action_space/test_action_space_base.py -v
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from agents._support.valid_actions_gating import (
    CLICK_TYPE_ENUM,
    FINISH_ENUM,
    RESPONSE_SCHEMA,
    UNRELATED_SCHEMA,
    qwen3_vl_adapter,
    tool_names,
)

import lite.agents.core.action_space.base as base_module
from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    BaseActionSpace,
    LiteBBoxActionSpace,
    LiteBrowserActionSpace,
    LiteBrowserNavToolSet,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.gpt.action_space import GPTMobileActionSpace
from lite.core.tools.action_space.batches import merge_adjacent_lite_action_batches
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.schemas import (
    make_tool_schema,
    tool,
    tool_schema_name,
    tool_schema_parameters,
)

# =============================================================================
# 1. Registry
# =============================================================================


class TestRegistry:
    """Test ActionSpaceRegistry get/list/patterns."""

    # ``lite@point`` / ``lite@bbox`` carry no ``@<platform>`` key segment, so the
    # registry never overwrites their class default — that default IS their
    # declared platform, and it must be a real ``LiteCUAMetadata.Platform`` value.
    # It used to be ``None``, which made ``LiteCUAMetadata.Platform(None)`` raise in
    # ``adapter/base.py::_default_metadata_for_adapter`` and reach "desktop" only
    # through that method's ValueError rescue.
    @pytest.mark.parametrize(
        "key,expected_platform",
        [
            ("lite@desktop", "desktop"),
            ("lite@browser", "browser"),
            ("lite@mobile", "mobile"),
            ("lite@point", "desktop"),
            ("lite@bbox", "desktop"),
        ],
    )
    def test_get_returns_action_space(self, key, expected_platform):
        space = ActionSpaceRegistry.get(key)
        assert space.platform == expected_platform

    def test_stale_web_key_is_not_registered(self):
        with pytest.raises(KeyError):
            ActionSpaceRegistry.get("lite@web")

    def test_list_includes_all_keys(self):
        all_keys = ActionSpaceRegistry.list_expanded()
        for key in ["lite@desktop", "lite@browser", "lite@mobile", "lite@point", "lite@bbox"]:
            assert key in all_keys

    def test_get_class_returns_class(self):
        cls = ActionSpaceRegistry.get_class("lite@desktop")
        assert cls is LiteDesktopActionSpace
        inst = cls()
        assert inst.platform == "desktop"


# =============================================================================
# 2. Tool Schemas
# =============================================================================


class TestToolSchemas:
    """Test get_tool_schemas, get_declared_action_schema_names, get_tool_schema."""

    def test_desktop_returns_list(self):
        schemas = LiteDesktopActionSpace.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0
        for s in schemas:
            assert s["type"] == "function"
            function = s["function"]
            assert "name" in function
            assert "parameters" in function

    def test_desktop_includes_click_type_key(self):
        actions = LiteDesktopActionSpace.get_declared_action_schema_names()
        assert "click" in actions
        assert "type" in actions
        assert "key" in actions
        assert "screenshot" in actions

    def test_point_has_point_only(self):
        assert LitePointActionSpace.get_declared_action_schema_names() == frozenset({"point"})

    def test_bbox_has_bbox_only(self):
        assert LiteBBoxActionSpace.get_declared_action_schema_names() == frozenset({"bbox"})

    def test_get_tool_schema_by_name(self):
        assert LiteDesktopActionSpace.get_tool_schema("click") is None
        schema = LiteDesktopActionSpace.get_tool_schema("computer")
        assert schema is not None
        assert tool_schema_name(schema) == "computer"
        action_enum = tool_schema_parameters(schema)["properties"]["actions"]["items"][
            "properties"
        ]["action"]["enum"]
        assert "click" in action_enum
        assert LiteDesktopActionSpace.get_tool_schema("nonexistent") is None

    def test_schemas_are_deep_copies(self):
        s1 = LiteDesktopActionSpace.get_tool_schemas()
        s2 = LiteDesktopActionSpace.get_tool_schemas()
        s1[0]["function"]["name"] = "MUTATED"
        assert tool_schema_name(s2[0]) != "MUTATED"

    def test_merge_adjacent_action_batches_respects_extra_and_surface_boundaries(self):
        calls = [
            LiteDesktopActionSpace.click(coordinate=[1, 2]),
            LiteDesktopActionSpace.type(text="a"),
            make_tool_call("bash", {"command": "pwd"}),
            LiteDesktopActionSpace.wait(duration=1),
            LiteMobileActionSpace.tap(coordinate=[3, 4]),
            LiteMobileActionSpace.type(text="b"),
            LiteDesktopActionSpace.screenshot(),
        ]

        merged = merge_adjacent_lite_action_batches(calls)

        assert [tool_call_name(call) for call in merged] == [
            "computer",
            "bash",
            "computer",
            "mobile",
            "computer",
        ]
        assert [a["action"] for a in tool_call_arguments(merged[0])["actions"]] == ["click", "type"]
        assert tool_call_arguments(merged[1]) == {"command": "pwd"}
        assert [a["action"] for a in tool_call_arguments(merged[2])["actions"]] == ["wait"]
        assert [a["action"] for a in tool_call_arguments(merged[3])["actions"]] == ["tap", "type"]
        assert [a["action"] for a in tool_call_arguments(merged[4])["actions"]] == ["screenshot"]


class TestValidActionFilteringOwners:
    def test_valid_actions_filters_nested_wrapper_function_parameters(self) -> None:
        """Post-migration nested schemas still take the wrapper-enum path."""
        qwen_space = type(qwen3_vl_adapter(None).action_space)

        class NestedQwenDesktopActionSpace(qwen_space):
            @classmethod
            def get_tool_schemas(cls, include=None):
                return super().get_tool_schemas(include)

        schemas = NestedQwenDesktopActionSpace.get_tool_schemas() + [UNRELATED_SCHEMA]

        filtered = NestedQwenDesktopActionSpace.filter_tool_schemas_for_valid_actions(
            schemas,
            ["click"],
        )

        wrapper = next(s for s in filtered if tool_schema_name(s) == "computer_use")
        action_prop = tool_schema_parameters(wrapper)["properties"]["action"]
        assert set(action_prop["enum"]) == (CLICK_TYPE_ENUM - {"type"}) | FINISH_ENUM
        assert "parameters" not in wrapper
        assert next(s for s in filtered if tool_schema_name(s) == "bash") == UNRELATED_SCHEMA

    def test_valid_actions_passes_through_function_schema_without_properties(self) -> None:
        qwen_space = type(qwen3_vl_adapter(None).action_space)
        no_properties = make_tool_schema(
            "noop",
            description="No-op.",
            parameters={"type": "object"},
        )

        filtered = qwen_space.filter_tool_schemas_for_valid_actions(
            qwen_space.get_tool_schemas() + [no_properties],
            ["click"],
        )

        assert next(s for s in filtered if tool_schema_name(s) == "noop") == no_properties

    def test_valid_actions_filters_function_schema_without_action_property(self) -> None:
        """Function-tool families are gated by name without requiring action."""
        schemas = GPTMobileActionSpace.get_tool_schemas()
        schemas.append(RESPONSE_SCHEMA)

        filtered = GPTMobileActionSpace.filter_tool_schemas_for_valid_actions(
            schemas,
            ["tap"],
        )

        assert tool_names(filtered) == {"tap", "response"}
        tap = next(s for s in filtered if tool_schema_name(s) == "tap")
        assert "action" not in tool_schema_parameters(tap).get("properties", {})

    def test_base_filter_has_no_wrapper_owner_cache(self):
        class _WrapperShapedActionSpace(BaseActionSpace):
            @staticmethod
            @tool(action="Native wrapper action.")
            def computer_use(action: Literal["left_click"]) -> dict[str, Any]:
                return make_tool_call("computer_use", {"action": action})

            LITE_ACTION_NAME_TO_QWEN_ACTION_VALUES = {"click": ["left_click"]}

        foreign_extra = make_tool_schema(
            "foreign_extra",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["external"]},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        )

        assert _WrapperShapedActionSpace.filter_tool_schemas_for_valid_actions(
            [foreign_extra],
            [],
        ) == [foreign_extra]
        assert _WrapperShapedActionSpace.filter_tool_schemas_for_valid_actions(
            _WrapperShapedActionSpace.get_tool_schemas(),
            [],
        ) == []

    def test_base_valid_actions_does_not_own_opaque_provider_schema_strings(self) -> None:
        """Opaque native ``computer*`` grammar belongs to provider action spaces."""
        native = [{"type": "computer"}]

        assert BaseActionSpace.filter_tool_schemas_for_valid_actions(native, []) == native

    @pytest.mark.parametrize(
        "retired_hook",
        ["_SCHEMA_VALID_ACTION_ALIASES", "_is_opaque_native_schema"],
    )
    def test_base_valid_actions_carries_no_retired_filtering_hook(self, retired_hook) -> None:
        """The base gate keeps no provider-shaped escape hatch."""
        assert not hasattr(BaseActionSpace, retired_hook)
        for space_cls in (LiteDesktopActionSpace, LiteMobileActionSpace):
            assert not hasattr(space_cls, retired_hook)

    def test_action_space_base_public_exports_exclude_extra_admission_helpers(self):
        assert "extra_tool_name_and_arguments_are_admitted" not in base_module.__all__
        assert "tool_call_satisfies_active_extra_schema" not in base_module.__all__

    @pytest.mark.parametrize(
        "space_cls,valid_action",
        [
            (LitePointActionSpace, "point"),
            (LiteBBoxActionSpace, "bbox"),
        ],
    )
    def test_grounding_valid_actions_gate_their_public_schemas(
        self,
        space_cls,
        valid_action,
    ):
        schemas = space_cls.get_tool_schemas()

        assert space_cls.filter_tool_schemas_for_valid_actions(schemas, [valid_action]) == schemas
        assert space_cls.filter_tool_schemas_for_valid_actions(schemas, []) == []


# =============================================================================
# 3. get_registry_key
# =============================================================================


class TestRegistryKey:
    def test_lite_desktop(self):
        # Desktop and browser are separate first-class platforms.
        assert LiteDesktopActionSpace.get_registry_key() == r"lite@desktop"

    def test_lite_browser(self):
        assert LiteBrowserActionSpace.get_registry_key() == r"lite@browser"
        assert LiteBrowserActionSpace().platform == "browser"
        # Browser = desktop action surface; nav/tab tools are resolved via
        # metadata.extra_tool_schemas from the LiteBrowserNavToolSet catalog.
        acts = LiteBrowserActionSpace.get_declared_action_schema_names()
        assert acts == LiteDesktopActionSpace.get_declared_action_schema_names()
        assert LiteBrowserNavToolSet.get_tool_names().isdisjoint(acts)
        assert LiteBrowserActionSpace.get_tool_schema("goto") is None
        assert LiteBrowserNavToolSet.get_tool_schema("goto") is not None
        assert "response" not in acts  # finish tools are extra_tool_schemas, not GUI actions

    def test_mobile(self):
        assert LiteMobileActionSpace.get_registry_key() == "lite@mobile"

    def test_point(self):
        assert LitePointActionSpace.get_registry_key() == "lite@point"

    def test_bbox(self):
        assert LiteBBoxActionSpace.get_registry_key() == "lite@bbox"

    def test_absolute_desktop_is_not_top_level_agents_api(self):
        import lite.agents as agents

        assert "AbsoluteDesktopActionSpace" not in agents.__all__
        assert not hasattr(agents, "AbsoluteDesktopActionSpace")


# =============================================================================
# 4. Canonical / agent-wire seam
# =============================================================================


class TestConversionSeam:
    def test_base_to_agent_projects_nested_canonical_to_bare_model_function(self):
        space = BaseActionSpace()
        call = make_tool_call("response", {"text": "done"})

        agent_calls = space.convert_tool_calls_to_agent([call])

        assert agent_calls == [{"name": "response", "arguments": {"text": "done"}}]
        assert agent_calls[0]["arguments"] is not tool_call_arguments(call)

    def test_base_from_agent_wraps_bare_model_function_as_prestamp_canonical(self):
        space = BaseActionSpace()
        agent_calls = [{"name": "response", "arguments": {"text": "done"}}]

        calls = space.convert_tool_calls_from_agent(agent_calls)

        assert calls == [make_tool_call("response", {"text": "done"})]
        assert "id" not in calls[0]
        assert validate_lite_tool_call(calls[0], "tool_calls[0]", require_id=False) is None

    def test_base_from_agent_rejects_canonical_input_on_bare_projection_side(self):
        space = BaseActionSpace()

        with pytest.raises(ValueError, match="non-bare-call keys"):
            space.convert_tool_calls_from_agent([make_tool_call("response", {"text": "done"})])

    @pytest.mark.parametrize(
        "bad_call,expected_type",
        [
            ("not-a-dict", TypeError),
            ({"name": "response", "arguments": {}, "call_id": "provider_1"}, ValueError),
            ({"name": "response"}, ValueError),
            ({"name": "", "arguments": {}}, ValueError),
            ({"name": "response", "arguments": '{"text": "done"}'}, ValueError),
        ],
    )
    def test_base_from_agent_bare_shape_violations_are_caller_bugs_not_model_output(
        self, bad_call, expected_type
    ):
        """The default parse entry rejects CALLER-layer mistakes, loudly and unowned.

        ``ModelToolCallParseError`` subclasses ``ValueError``, so nothing else in
        the suite can tell the two apart. This is the closure guard for that
        classification: a non-dict element, a provider ``call_id`` on a bare GUI
        call, or a canonical envelope on the parse side is built by the calling
        parser, not chosen by the model. Spelling them as the model-output type
        would let ``AdapterBasedAgent._parse_generation_response()`` rewrite an
        adapter bug into a terminal model parse-failure final.
        """
        space = BaseActionSpace()

        with pytest.raises(expected_type) as excinfo:
            space.convert_tool_calls_from_agent([bad_call])

        assert type(excinfo.value) is expected_type
        assert not isinstance(excinfo.value, ModelToolCallParseError)


# =============================================================================
# 5. Text Format (BaseActionSpace generic)
# =============================================================================


class TestTextFormat:
    """Test generic BaseActionSpace text formatting via LiteDesktopActionSpace."""

    def setup_method(self):
        self.space = LiteDesktopActionSpace()

    def test_format_tool_call_as_text_click(self):
        tc = LiteDesktopActionSpace.click(coordinate=[500, 300])
        text = self.space.format_tool_call_as_text(tc)
        assert "click" in text
        assert "500" in text and "300" in text

    def test_format_tool_call_as_text_type(self):
        tc = LiteDesktopActionSpace.type(text="hello")
        text = self.space.format_tool_call_as_text(tc)
        assert "type" in text
        assert "hello" in text

    def test_format_tool_calls_as_text_joins(self):
        tcs = [
            LiteDesktopActionSpace.click(coordinate=[100, 200]),
            LiteDesktopActionSpace.type(text="test"),
        ]
        text = self.space.format_tool_calls_as_text(tcs)
        assert "\n\n" in text
        assert "click" in text
        assert "type" in text

    def test_renderer_takes_a_canonical_call_not_a_bare_projection(self):
        """Input shape is the base renderer's contract, not an incidental.

        The UI-TARS / Step-GUI renderers are the mirror image (bare
        ``{name, arguments}`` projection only), so the projection must fail here
        rather than render something plausible. It fails loudly at the first key
        access; no shape sniffing is added to make both work.
        """
        with pytest.raises(KeyError):
            self.space.format_tool_call_as_text({
                "name": "click",
                "arguments": {"coordinate": [500, 300]},
            })

    def test_plural_path_carries_the_same_shape(self):
        """``format_tool_calls_as_text`` adds no shape handling of its own."""
        # ``@final``: a family narrows the shape by overriding the SINGULAR
        # renderer, never by growing a second plural decision.
        assert BaseActionSpace.format_tool_calls_as_text.__final__ is True

        with pytest.raises(KeyError):
            self.space.format_tool_calls_as_text([
                {"name": "click", "arguments": {"coordinate": [500, 300]}},
            ])

    def test_format_message_as_text_with_content_and_tool_calls(self):
        # ``action_description`` is the prose describing the action; it
        # renders as-is (no ``Thought:`` relabeling — ``Thought:`` is
        # reserved for the prompted ``InlineReasoningContent`` channel).
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Click the button."}],
            "tool_calls": [LiteDesktopActionSpace.click(coordinate=[500, 300])],
        }
        text = self.space.format_message_as_text(msg)
        assert "Thought:" not in text  # no inline_reasoning → no Thought: line
        assert "Click the button." in text  # action_description rendered as-is
        assert "Action:" in text
        assert "click" in text

    def test_format_message_as_text_with_reasoning(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "I need to click the button."},
                {"type": "action_description", "text": "Click it."},
            ],
            "tool_calls": [LiteDesktopActionSpace.click(coordinate=[500, 300])],
        }
        text = self.space.format_message_as_text(msg)
        assert "Thought: I need to click the button." in text
        assert "Click it." in text
        assert "Action:" in text

    def test_format_message_as_text_no_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Just thinking."}],
        }
        text = self.space.format_message_as_text(msg)
        # No inline_reasoning → no Thought: line. action_description is
        # rendered as the raw action prose.
        assert "Thought:" not in text
        assert "Just thinking." in text
        assert "Action:" not in text

    def test_format_message_as_text_action_description_content(self):
        # Assistant content is a list of structured parts (per the LiteMessage
        # contract — string content is reserved for system / user messages).
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Click the button."}],
            "tool_calls": [LiteDesktopActionSpace.click(coordinate=[500, 300])],
        }
        text = self.space.format_message_as_text(msg)
        assert "Click the button." in text
