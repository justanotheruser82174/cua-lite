"""
Tests for MAI-UI adapter + agent.

Covers:
  1. Registry resolution
  2. System prompt assembly (native prompt + app-name hint only)
  3. parse_raw_assistant_response (<thinking> + <tool_call>)
  4. convert_message_to_agent / from_agent (with and without extra_tools)
  5. convert_sample_to_agent (no-history, with-history, beyond-window)
  6. MAIUIHistoryProtocol shape (split first user, beyond-window windowing)
  7. MAIUIMobileAgent.build_generation_prompt — reasoning_content folding
  8. End-to-end byte-comparison against MAI-UI reference _build_messages

Run:
    uv run pytest tests/agents/models/mai_ui/test_mai_ui_adapter.py -v
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any

import pytest
from lite_samples import build_lite_trajectory

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AgentRegistry
from lite.agents.models.mai_ui.action_space import MAIUIMobileActionSpace
from lite.agents.models.mai_ui.adapter import (
    _MAI_MOBILE_SYS_PROMPT_APPS_LINE,
    _MAI_MOBILE_SYS_PROMPT_FOOTER,
    MAI_MOBILE_SYS_PROMPT,
    MAIUIMobileUseAdapter,
)
from lite.agents.models.mai_ui.agent import MAIUIMobileAgent
from lite.agents.models.mai_ui.protocol import MAIUIHistoryProtocol
from lite.core import (
    LiteCUAMetadata,
    LiteSample,
)
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.action_space import (
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet, make_open_app_tool
from lite.core.tools.schemas import tool_schema_parameters


def _md(
    extra_tools: list[dict] | None = None,
    valid_actions: list[str] | None = None,
    **others: Any,
) -> LiteCUAMetadata:
    """Test helper — build a mobile-navigation LiteCUAMetadata with extras."""
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tools or [],
        valid_actions=valid_actions,
        others=others,
    )


def _mai_natively_rendered_extra_schemas() -> list[dict[str, Any]]:
    return [
        make_open_app_tool(["Chrome"]),
        LiteFinishToolSet.get_tool_schema("response"),
        LiteFinishToolSet.get_tool_schema("terminate"),
    ]


_MAI_STANDALONE_EXTRA_NAMES = ("open_app", "response", "terminate")


def _mai_extra_schema(name: str) -> dict[str, Any]:
    if name == "open_app":
        return make_open_app_tool(["Chrome"])
    return LiteFinishToolSet.get_tool_schema(name)


# Byte goldens for MAI-UI's ``## Action Space`` rows, in SFT order. This block is
# the model's fine-tuning distribution, not a rendered tool surface, so the tests
# below pin its exact bytes rather than asking whether a substring is present.
_MAI_ANSWER_ROW = (
    r'{"action": "answer", "text": "xxx"} # Use escape characters \', \", and \n'
    r" in text part to ensure we can parse the text in normal python string format."
)
_MAI_TRAINED_ACTION_ROWS = (
    '{"action": "click", "coordinate": [x, y]}',
    '{"action": "long_press", "coordinate": [x, y]}',
    '{"action": "type", "text": ""}',
    '{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]}'
    ' # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.',
    '{"action": "open", "text": "app_name"}',
    '{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}',
    '{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter',
    '{"action": "wait"}',
    '{"action": "terminate", "status": "success or fail"}',
    _MAI_ANSWER_ROW,
)
# The rows the pre-fix active-extras gate deleted whenever the sample advertised
# no matching standalone extra tool — i.e. at every env's shipped
# ``extra_tools: []`` default.
_MAI_ROWS_THE_EXTRAS_GATE_USED_TO_DELETE = (
    '{"action": "open", "text": "app_name"}',
    '{"action": "terminate", "status": "success or fail"}',
    _MAI_ANSWER_ROW,
)


def _mai_adapter_with_natively_rendered_extras() -> MAIUIMobileUseAdapter:
    return MAIUIMobileUseAdapter(
        metadata=_md(extra_tools=_mai_natively_rendered_extra_schemas()),
    )


def _single_mobile_action(tool_call: dict[str, Any]) -> dict[str, Any]:
    assert tool_call_name(tool_call) == "mobile"
    actions = tool_call_arguments(tool_call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _mobile_call(action: str, **arguments: Any) -> dict[str, Any]:
    return make_tool_call("mobile", {"actions": [{"action": action, **arguments}]})

# =============================================================================
# 1. Registry resolution
# =============================================================================


class TestRegistry:
    def test_adapter_registry(self):
        adapter = AgentAdapterRegistry.get("mai_ui@mobile@use")
        assert isinstance(adapter, MAIUIMobileUseAdapter)

    def test_agent_registry(self):
        agent = AgentRegistry.get(
            "mai_ui@mobile@use",
            generate_fn=lambda **kw: {"response": ""},
        )
        assert isinstance(agent, MAIUIMobileAgent)
        assert isinstance(agent.adapter, MAIUIMobileUseAdapter)
        assert isinstance(agent.adapter.action_space, MAIUIMobileActionSpace)
        assert isinstance(agent.adapter.protocol, MAIUIHistoryProtocol)
        assert agent.adapter.protocol.full_history_size == 3

    def test_default_system_prompt(self):
        adapter = MAIUIMobileUseAdapter()
        assert adapter.system_prompt == MAI_MOBILE_SYS_PROMPT
        assert "<thinking>" in adapter.system_prompt
        assert "<tool_call>" in adapter.system_prompt
        # The action space JSON examples are baked into the system prompt
        assert '{"action": "click"' in adapter.system_prompt
        assert '{"action": "swipe"' in adapter.system_prompt


# =============================================================================
# 2. System prompt assembly (native prompt filtering + MCP block)
# =============================================================================


class TestSystemPromptAssembly:
    def test_system_text_is_base_prompt(self):
        adapter = MAIUIMobileUseAdapter()
        assert adapter._build_system_text() == MAI_MOBILE_SYS_PROMPT

    def test_native_action_enum_source_descends_nested_schema(self):
        schemas = MAIUIMobileActionSpace.get_tool_schemas()

        assert set(schemas[0]) == {"type", "function"}
        assert "parameters" not in schemas[0]
        action_enum = tool_schema_parameters(schemas[0])["properties"]["action"]["enum"]
        assert {"click", "wait", "answer"} <= set(action_enum)

    def test_non_native_extra_tools_append_mcp_block(self):
        extra = [
            make_tool_schema(
                "search_web",
                description="Search the web for a query.",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))
        full = adapter._build_system_text()
        section = adapter._build_tools_section()
        assert section
        assert full.startswith(MAI_MOBILE_SYS_PROMPT)
        assert full.endswith(section)
        assert "\n\n## MCP Tools" in full
        assert '"name": "search_web"' in full

    def test_non_native_extra_tool_mcp_block_matches_byte_golden(self):
        extra = [
            make_tool_schema(
                "translate_label",
                description="Translate café labels into 日本語.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Source text, e.g. mañana or crème brûlée.",
                        },
                        "locale": {
                            "type": "string",
                            "enum": ["ja-JP", "es-ES"],
                        },
                    },
                    "required": ["text"],
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))

        expected = (
            "## MCP Tools\n"
            "You are also provided with MCP tools, you can use them to complete the task.\n"
            '{"type": "function", "function": {"name": "translate_label", '
            '"description": "Translate café labels into 日本語.", "parameters": '
            '{"type": "object", "properties": {"text": {"type": "string", '
            '"description": "Source text, e.g. mañana or crème brûlée."}, '
            '"locale": {"type": "string", "enum": ["ja-JP", "es-ES"]}}, '
            '"required": ["text"]}}}\n'
            "\n"
            "If you want to use MCP tools, you must output as the following format:\n"
            "<thinking>\n"
            "...\n"
            "</thinking>\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )
        assert adapter._build_tools_section() == expected

    def test_open_app_schema_does_not_render_schema_and_apps_hint_is_appended(self):
        adapter = MAIUIMobileUseAdapter(
            metadata=_md(
                extra_tools=[make_open_app_tool(["Chrome", "Settings"])],
            )
        )
        full = adapter._build_system_text()
        assert '"open_app"' not in full
        assert "## MCP Tools" not in full
        assert '"action": "open"' in adapter.system_prompt
        assert 'Available Apps: `["Chrome","Settings"]`' in full

    def test_apps_hint_requires_open_app_schema(self):
        """The apps hint is env DATA, so it needs the ``open_app`` enum. The
        ``open`` action ROW is trained text and stays either way."""
        adapter = MAIUIMobileUseAdapter(metadata=_md(apps=["Chrome", "Settings"]))

        full = adapter._build_system_text()

        assert "Available Apps" not in full
        assert '{"action": "open", "text": "app_name"}' in full

    def test_valid_actions_does_not_reach_the_action_space_rows(self):
        """``valid_actions`` gates the ``mobile_use`` SCHEMA, which this family
        never renders — its Action Space rows are prompt bytes, so the GUI gate
        narrows what the env admits and nothing the model sees."""
        adapter = MAIUIMobileUseAdapter(metadata=_md(valid_actions=["tap", "type"]))
        full = adapter._build_system_text()

        for row in _MAI_TRAINED_ACTION_ROWS:
            assert row in full, row

    def test_natively_advertised_extras_do_not_duplicate_into_mcp_block(self):
        """open_app / response / terminate are already on the model's surface as
        Action Space rows, so activating them adds no MCP block."""
        adapter = MAIUIMobileUseAdapter(
            metadata=_md(extra_tools=_mai_natively_rendered_extra_schemas()),
        )

        assert "## MCP Tools" not in adapter._build_system_text()

    def test_empty_valid_actions_keeps_every_row_and_still_routes_the_mcp_block(self):
        """``valid_actions=[]`` is the strictest gate an env can set and it does
        not touch a single Action Space row. Only non-native extras go to the
        MCP block."""
        non_native_extra = make_tool_schema(
            "search_web",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        adapter = MAIUIMobileUseAdapter(
            metadata=_md(
                extra_tools=[*_mai_natively_rendered_extra_schemas(), non_native_extra],
                valid_actions=[],
            )
        )
        full = adapter._build_system_text()

        for row in _MAI_TRAINED_ACTION_ROWS:
            assert row in full, row
        # Natively advertised extras stay OUT of the MCP block; the rest go in.
        assert "## MCP Tools" in full
        assert '"name": "open_app"' not in full
        assert '"name": "response"' not in full
        assert '"name": "search_web"' in full

    def test_empty_valid_actions_leaves_the_full_trained_row_set_in_order(self):
        """The GUI-off, extras-off quadrant — the state that used to hand the
        model a 1005-byte prompt with 7 of the 10 rows gone. Both gates off, and
        the block is still the model's fine-tuning distribution, in SFT order."""
        adapter = MAIUIMobileUseAdapter(metadata=_md(valid_actions=[]))
        full = adapter._build_system_text()

        rows = [line for line in full.splitlines() if line.startswith('{"action": ')]
        assert tuple(rows) == _MAI_TRAINED_ACTION_ROWS

    def test_extra_tool_name_collision_raises(self):
        # Collide with a standard mobile_use action enum value
        extra = [make_tool_schema("mobile_use", description="x", parameters={})]
        with pytest.raises(ValueError, match="collide"):
            MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))


class TestTrainedPromptIsInvariantAcrossTheActiveSurface:
    """MAI-UI's ``## Action Space`` block is hand-baked SFT text, not a rendered
    tools section, so neither ``metadata.extra_tool_schemas`` nor
    ``metadata.valid_actions`` may edit it.

    Withholding a trained row moves the model off its fine-tuning distribution;
    reachability is env ingress's question to answer, and it already does —
    a call to an inactive tool comes back as model-visible feedback keyed to
    its call id.

    These assert byte IDENTITY of the rendered prompt. An ``"x" in text`` check
    is what let this same defect ship in the sibling text-grammar families.
    """

    @staticmethod
    def _render(combo: tuple[str, ...], valid_actions: list[str] | None) -> str:
        return MAIUIMobileUseAdapter(
            metadata=_md(
                extra_tools=[_mai_extra_schema(name) for name in combo],
                valid_actions=valid_actions,
            )
        )._build_system_text()

    @pytest.mark.parametrize("valid_actions", [None, ["tap", "type"], []])
    def test_prompt_is_byte_identical_across_all_eight_extras_combinations(
        self,
        valid_actions,
    ):
        combos = [
            combo
            for size in range(len(_MAI_STANDALONE_EXTRA_NAMES) + 1)
            for combo in itertools.combinations(_MAI_STANDALONE_EXTRA_NAMES, size)
        ]
        assert len(combos) == 8

        rendered = {combo: self._render(combo, valid_actions) for combo in combos}

        # The one thing an extras combination may still move is the two
        # "Available Apps" lines. Those are not a finish form and not trained
        # text -- they are the ``open_app`` enum the env advertises, genuinely
        # per-sample DATA -- so they render iff that enum exists.
        without_apps = {c: t for c, t in rendered.items() if "open_app" not in c}
        with_apps = {c: t for c, t in rendered.items() if "open_app" in c}

        assert len(set(without_apps.values())) == 1, sorted(without_apps)
        assert len(set(with_apps.values())) == 1, sorted(with_apps)

        base = next(iter(without_apps.values()))
        apps_lines = _MAI_MOBILE_SYS_PROMPT_APPS_LINE.format(apps_list='["Chrome"]')
        assert next(iter(with_apps.values())) == base.replace(
            _MAI_MOBILE_SYS_PROMPT_FOOTER,
            f"{apps_lines}\n{_MAI_MOBILE_SYS_PROMPT_FOOTER}",
        )

    @pytest.mark.parametrize(
        "valid_actions",
        [None, [], ["tap"], ["tap", "type"]],
    )
    def test_prompt_is_byte_identical_across_valid_actions(self, valid_actions):
        """The GUI gate does not reach the trained rows either.

        The defect this replaces handed the model a 1005-byte prompt under
        ``valid_actions=[]`` with 7 of the 10 rows deleted — ``click``,
        ``long_press``, ``type``, ``swipe``, ``drag``, ``system_button`` and
        ``wait``, i.e. every GUI verb the model can actually emit.
        """
        assert self._render((), valid_actions) == MAI_MOBILE_SYS_PROMPT

    def test_default_prompt_carries_the_full_trained_row_set_in_order(self):
        full = MAIUIMobileUseAdapter(metadata=_md())._build_system_text()

        assert full == MAI_MOBILE_SYS_PROMPT
        rows = [line for line in full.splitlines() if line.startswith('{"action": ')]
        assert tuple(rows) == _MAI_TRAINED_ACTION_ROWS

    @pytest.mark.parametrize("row", _MAI_ROWS_THE_EXTRAS_GATE_USED_TO_DELETE)
    def test_rows_the_extras_gate_used_to_delete_survive_with_no_extras_active(self, row):
        """The negative half, named. Before this fix, ``extra_tools: []`` — the
        value every mobile env's ``configs/default.yaml`` ships, hence what a
        config-less ``scripts/rollout.py`` run gets — deleted exactly these three
        rows (and the two "Available Apps" lines) from MAI-UI's SFT text.
        """
        assert row in MAIUIMobileUseAdapter(metadata=_md())._build_system_text()


# =============================================================================
# 3. parse_raw_assistant_response
# =============================================================================


class TestParseResponse:
    def setup_method(self):
        self.adapter = MAIUIMobileUseAdapter()

    def test_parse_extracts_tool_call_and_keeps_thinking_as_text(self):
        """parse_raw_assistant_response extracts <tool_call> into tool_calls
        and keeps the remaining text (including <thinking>) as plain text."""
        response = (
            "<thinking>\nI should tap the search bar at the top.\n</thinking>\n"
            '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", '
            '"coordinate": [500, 100]}}\n</tool_call>'
        )
        msg = self.adapter.parse_raw_assistant_response(response)
        assert msg["role"] == "assistant"
        # <thinking> is NOT parsed at this stage — it stays in the text content.
        text = msg["content"][0]["text"]
        assert "<thinking>" in text
        assert "I should tap the search bar at the top." in text
        assert "tool_calls" in msg
        assert msg["tool_calls"][0]["name"] == "mobile_use"
        assert msg["tool_calls"][0]["arguments"]["action"] == "click"
        assert msg["tool_calls"][0]["arguments"]["coordinate"] == [500, 100]

    def test_full_chain_extracts_inline_reasoning(self):
        """Chaining parse_raw_assistant_response + convert_message_from_agent
        produces InlineReasoningContent from <thinking>."""
        response = (
            "<thinking>\nI should tap the search bar at the top.\n</thinking>\n"
            '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", '
            '"coordinate": [500, 100]}}\n</tool_call>'
        )
        parsed = self.adapter.parse_raw_assistant_response(response)
        msg = self.adapter.convert_message_from_agent(parsed)
        assert msg["role"] == "assistant"
        reasoning = next(
            (c["text"] for c in msg["content"] if c.get("type") == "inline_reasoning"),
            None,
        )
        assert reasoning == "I should tap the search bar at the top."
        assert _single_mobile_action(msg["tool_calls"][0])["action"] == "tap"

    def test_parses_tool_call_without_thinking(self):
        response = (
            '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "wait"}}\n</tool_call>'
        )
        msg = self.adapter.parse_raw_assistant_response(response)
        # No <thinking> → no text content after stripping <tool_call>.
        assert "reasoning_content" not in msg
        assert "content" not in msg
        assert msg["tool_calls"][0]["arguments"]["action"] == "wait"

    def test_handles_think_shorthand(self):
        """Reference parser tolerates </think> as a synonym for </thinking>.
        parse_raw keeps <thinking> as text; convert_message_from_agent extracts it."""
        response = (
            "I'm reasoning here.</think>\n"
            '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "wait"}}\n</tool_call>'
        )
        parsed = self.adapter.parse_raw_assistant_response(response)
        # At parse stage, the thinking text is in plain text content.
        assert parsed["content"][0]["text"] == "I'm reasoning here.</think>"
        # After full chain, convert_message_from_agent extracts it.
        msg = self.adapter.convert_message_from_agent(parsed)
        reasoning = next(
            (c["text"] for c in msg.get("content", []) if c.get("type") == "inline_reasoning"),
            None,
        )
        assert reasoning == "I'm reasoning here."

    def test_drops_invalid_json_tool_call(self):
        response = "<thinking>\nbad\n</thinking>\n<tool_call>\nNOT JSON\n</tool_call>"
        parsed = self.adapter.parse_raw_assistant_response(response)
        # At parse stage, <thinking> stays in the text content.
        assert "<thinking>" in parsed["content"][0]["text"]
        assert "bad" in parsed["content"][0]["text"]
        assert "tool_calls" not in parsed
        # After full chain, convert_message_from_agent extracts reasoning.
        msg = self.adapter.convert_message_from_agent(parsed)
        reasoning = next(
            (c["text"] for c in msg.get("content", []) if c.get("type") == "inline_reasoning"),
            None,
        )
        assert reasoning == "bad"
        assert "tool_calls" not in msg

    def test_repairs_malformed_coordinate_json_tool_call(self, caplog):
        """MAI-UI closes ``"coordinate":[N,N`` with ``}`` instead of ``]}}``.

        That quirk is documented in ``_parse_mai_tool_call_json`` and the repair
        pass exists for it. An earlier revision deleted the repair and this test
        was written against the deleted behaviour, freezing a regression as the
        expectation -- a click the agent used to execute was being dropped.
        """
        caplog.set_level(logging.INFO, logger="lite.agents.models.mai_ui.adapter")
        response = (
            "<thinking>\nbad coordinate\n</thinking>\n"
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,100}}\n</tool_call>'
        )
        parsed = self.adapter.parse_raw_assistant_response(response)
        assert "tool_calls" in parsed, "the repair pass should recover this call"
        (call,) = parsed["tool_calls"]
        assert call["arguments"]["coordinate"] == [500, 100]
        assert "Repaired MAI-UI tool_call JSON" in caplog.text

    def test_ignores_provider_envelope_tool_call_json(self):
        response = (
            "<tool_call>\n"
            '{"type": "function", "function": {"name": "mobile_use", '
            '"arguments": {"action": "wait"}}}\n'
            "</tool_call>"
        )
        parsed = self.adapter.parse_raw_assistant_response(response)
        assert "tool_calls" not in parsed

    def test_multiple_tool_calls(self):
        response = (
            '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", '
            '"coordinate": [1, 1]}}\n</tool_call>\n'
            '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "wait"}}\n</tool_call>'
        )
        msg = self.adapter.parse_raw_assistant_response(response)
        assert len(msg["tool_calls"]) == 2


# =============================================================================
# 4. convert_message_to_agent / from_agent
# =============================================================================


class TestMessageConversion:
    def setup_method(self):
        self.adapter = MAIUIMobileUseAdapter()

    def test_assistant_tool_call_translated_to_mai_ui(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Tap home."}],
            "tool_calls": [_mobile_call("tap", coordinate=[500, 500])],
        }
        agent_msg = self.adapter.convert_message_to_agent(msg)
        # convert_message_to_agent folds tool_calls into the wire-format text.
        # The translated MAI-UI form (``mobile_use`` + ``action=click``) appears
        # inside the ``<tool_call>{json}</tool_call>`` block.
        text = agent_msg["content"][0]["text"]
        assert '"name":"mobile_use"' in text
        assert '"action":"click"' in text

    def test_canonical_nested_input_renders(self):
        msg = {
            "role": "assistant",
            "tool_calls": [_mobile_call("tap", coordinate=[1, 2])],
        }
        agent_msg = self.adapter.convert_message_to_agent(msg)
        text = agent_msg["content"][0]["text"]
        assert '"name":"mobile_use"' in text
        assert '"action":"click"' in text
        assert '"coordinate":[1,2]' in text

    def test_assistant_tool_call_translated_from_mai_ui(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "mobile_use",
                    "arguments": {"action": "click", "coordinate": [499, 499]},
                }
            ],
        }
        cua_msg = self.adapter.convert_message_from_agent(msg)
        tc = cua_msg["tool_calls"][0]
        assert _single_mobile_action(tc)["action"] == "tap"

    def _from_raw(self, body: str, adapter=None):
        adapter = adapter or self.adapter
        return adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(f"<tool_call>\n{body}\n</tool_call>")
        )

    def test_top_level_native_function_name_reads_as_the_native_action(self):
        """Flat output — the ``mobile_use`` wrapper dropped and the native action
        value used AS the tool name — converts through the same dispatch as the
        nested form. Handing ``click`` to the env as a standalone tool instead
        wasted the turn.
        """
        flat = self._from_raw('{"name": "click", "arguments": {"coordinate": [499, 499]}}')
        nested = self._from_raw(
            '{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [499, 499]}}'
        )

        assert flat["tool_calls"] == nested["tool_calls"]
        assert _single_mobile_action(flat["tool_calls"][0])["action"] == "tap"
        assert MODEL_OUTPUT_ERROR_KEY not in flat

    def test_top_level_name_outside_the_native_vocabulary_stays_standalone(self):
        """The negative half: only the names the ``mobile_use`` action enum
        spells are read as actions. Anything else keeps its wire name and reaches
        env ingress, which answers it with feedback keyed to the call id.
        """
        cua_msg = self._from_raw('{"name": "summarize", "arguments": {"text": "all done"}}')

        assert cua_msg["tool_calls"] == [make_tool_call("summarize", {"text": "all done"})]
        assert MODEL_OUTPUT_ERROR_KEY not in cua_msg

    def test_top_level_answer_shorthand_still_maps_to_response(self):
        response = '<tool_call>\n{"name": "answer", "arguments": {"text": "42"}}\n</tool_call>'

        parsed = self.adapter.parse_raw_assistant_response(response)
        cua_msg = self.adapter.convert_message_from_agent(parsed)

        tc = cua_msg["tool_calls"][0]
        assert tool_call_name(tc) == "response"
        assert tool_call_arguments(tc) == {"text": "42"}

    def test_action_shaped_non_wrapper_without_matching_extra_shape_uses_action_switch(self):
        extra = [
            make_tool_schema(
                "loose_extra",
                description="x",
                parameters={
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                    "required": ["payload"],
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "loose_extra",
                    "arguments": {"action": "click", "coordinate": [499, 499]},
                },
                {"name": "loose_extra", "arguments": {"payload": "env"}},
            ],
        }

        cua_msg = adapter.convert_message_from_agent(msg)

        assert _single_mobile_action(cua_msg["tool_calls"][0])["action"] == "tap"
        assert cua_msg["tool_calls"][1] == make_tool_call("loose_extra", {"payload": "env"})

    def test_active_extra_schema_shape_wins_over_action_shaped_non_wrapper(self):
        extra = [
            make_tool_schema(
                "report_action",
                description="Report an action-shaped payload.",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "coordinate": {"type": "array"},
                    },
                    "required": ["action", "coordinate"],
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "report_action",
                    "arguments": {"action": "click", "coordinate": [499, 499]},
                }
            ],
        }

        cua_msg = adapter.convert_message_from_agent(msg)

        tc = cua_msg["tool_calls"][0]
        assert tool_call_name(tc) == "report_action"
        assert tool_call_arguments(tc) == {
            "action": "click",
            "coordinate": [499, 499],
        }

    def test_active_extra_with_off_schema_arguments_reaches_env_ingress(self):
        """An advertised extra called with an argument value its schema rejects
        keeps its wire name. Env ingress owns argument validation and answers it
        with feedback keyed to the call id; dropping it here turned the turn into
        a parse-failure final instead.
        """
        extra = [
            make_tool_schema(
                "choose_mode",
                description="Choose mode.",
                parameters={
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["yes"]}},
                    "required": ["mode"],
                    "additionalProperties": False,
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))
        msg = {
            "role": "assistant",
            "tool_calls": [{"name": "choose_mode", "arguments": {"mode": "no"}}],
        }

        cua_msg = adapter.convert_message_from_agent(msg)

        assert cua_msg["tool_calls"] == [make_tool_call("choose_mode", {"mode": "no"})]
        assert MODEL_OUTPUT_ERROR_KEY not in cua_msg

    def test_extra_named_like_a_native_action_is_claimed_by_its_arguments(self):
        """A name collision between an active extra and a native action value is
        resolved by the ARGUMENTS: browsergym's ``click(bid=...)`` is the env's
        tool, while ``click(coordinate=...)`` is still the native action.
        """
        extra = [
            make_tool_schema(
                "click",
                description="Click an element by bid.",
                parameters={
                    "type": "object",
                    "properties": {"bid": {"type": "string"}},
                    "required": ["bid"],
                    "additionalProperties": False,
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))

        by_bid = adapter.convert_message_from_agent(
            {
                "role": "assistant",
                "tool_calls": [{"name": "click", "arguments": {"bid": "a12"}}],
            }
        )
        by_coordinate = adapter.convert_message_from_agent(
            {
                "role": "assistant",
                "tool_calls": [{"name": "click", "arguments": {"coordinate": [499, 499]}}],
            }
        )
        nested = adapter.convert_message_from_agent(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "mobile_use",
                        "arguments": {"action": "click", "coordinate": [499, 499]},
                    }
                ],
            }
        )

        assert by_bid["tool_calls"] == [make_tool_call("click", {"bid": "a12"})]
        assert by_coordinate["tool_calls"] == nested["tool_calls"]

    def test_active_natively_rendered_extra_top_level_call_wins_before_action_switch(self):
        adapter = _mai_adapter_with_natively_rendered_extras()
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"name": "open_app", "arguments": {"app_name": "Chrome"}},
                {"name": "response", "arguments": {"text": "done"}},
                {"name": "terminate", "arguments": {"status": "success"}},
            ],
        }

        cua_msg = adapter.convert_message_from_agent(msg)

        assert cua_msg["tool_calls"] == [
            make_tool_call("open_app", {"app_name": "Chrome"}),
            make_tool_call("response", {"text": "done"}),
            make_tool_call("terminate", {"status": "success"}),
        ]

    def test_non_native_extra_tool_renders_as_standalone_mcp_call(self):
        extra = [
            make_tool_schema(
                "search_web",
                description="x",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))
        msg = {
            "role": "assistant",
            "tool_calls": [
                _mobile_call("tap", coordinate=[100, 100]),
                make_tool_call("search_web", {"q": "weather"}),
            ],
        }
        agent_msg = adapter.convert_message_to_agent(msg)
        text = agent_msg["content"][0]["text"]
        assert '"name":"mobile_use"' in text
        assert '"action":"click"' in text
        assert '"name":"search_web"' in text
        assert '"q":"weather"' in text

    def test_wrapper_embedded_extra_tool_becomes_invalid_action_batch(self):
        """Extra tools must be standalone, not hidden inside
        ``mobile_use(action=<extra>)``. The embedded call takes the
        unknown-action path: an INVALID ``mobile`` batch env ingress rejects by
        name, so the model sees the rejection instead of a silent drop."""
        extra = [
            make_tool_schema(
                "search_web",
                description="x",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "mobile_use",
                    "arguments": {"action": "search_web", "q": "weather"},
                }
            ],
        }
        cua_msg = adapter.convert_message_from_agent(msg)

        [tc] = cua_msg["tool_calls"]
        assert tool_call_name(tc) == "mobile"
        assert tool_call_arguments(tc) == {
            "actions": [{"action": "search_web", "q": "weather"}],
        }
        children, error = validate_lite_action_batch_structure(
            "mobile",
            tool_call_arguments(tc),
        )
        assert len(children) == 1
        error = lite_action_batch_child_name_errors("mobile", children).get(0)
        assert error is not None
        assert error.child_action_name == "search_web"

    def test_unknown_top_level_tool_call_uses_non_wrapper_fallback(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{"name": "unknown_tool", "arguments": {"x": 1}}],
        }
        cua_msg = self.adapter.convert_message_from_agent(msg)
        assert cua_msg["tool_calls"] == [make_tool_call("unknown_tool", {"x": 1})]
        assert MODEL_OUTPUT_ERROR_KEY not in cua_msg

    def test_from_agent_preserves_declared_extra_tool_order(self):
        extra = [
            make_tool_schema(
                "ask_user",
                description="Ask the user.",
                parameters={
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            )
        ]
        adapter = MAIUIMobileUseAdapter(metadata=_md(extra_tools=extra))
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "mobile_use",
                    "arguments": {"action": "click", "coordinate": [499, 499]},
                },
                {"name": "ask_user", "arguments": {"question": "Continue?"}},
                {
                    "name": "mobile_use",
                    "arguments": {"action": "type", "text": "ok"},
                },
            ],
        }

        cua_msg = adapter.convert_message_from_agent(msg)

        assert _single_mobile_action(cua_msg["tool_calls"][0])["action"] == "tap"
        assert tool_call_name(cua_msg["tool_calls"][1]) == "ask_user"
        assert _single_mobile_action(cua_msg["tool_calls"][2])["action"] == "type"
        assert tool_call_arguments(cua_msg["tool_calls"][1]) == {"question": "Continue?"}


# =============================================================================
# 5. convert_sample_to_agent
# =============================================================================


def _make_mobile_sample(num_steps: int, *, with_reasoning: bool = False) -> LiteSample:
    """Build a mobile-shape LiteSample with `num_steps` complete turns + 1 in-progress."""
    messages: list[dict[str, Any]] = []
    for i in range(num_steps):
        if i == 0:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": "Search for cats."},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": [{"type": "image", "index": i}]})
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": [{"type": "text", "text": f"Action step {i + 1}."}],
            "tool_calls": [_mobile_call("tap", coordinate=[100 + i * 50, 100])],
        }
        if with_reasoning:
            assistant["reasoning_content"] = f"Reasoning at step {i + 1}."
        messages.append(assistant)
    # Current step (in-progress, image only)
    messages.append({"role": "user", "content": [{"type": "image", "index": num_steps}]})
    return LiteSample(
        images=[f"img{i}.png" for i in range(num_steps + 1)],
        messages=messages,
        metadata=None,
    )


class TestSampleConversion:
    def test_no_history_first_step(self):
        """Just the very first user with instruction + the current image."""
        sample = LiteSample(
            images=["img0.png"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": "Open Settings."},
                    ],
                },
            ],
            metadata=None,
        )
        adapter = MAIUIMobileUseAdapter()
        # Last-turn rendering is the predict-time view of the partial trailing
        # user turn — same semantics the old per-step convert_sample_to_agent had.
        msgs = adapter.unroll(sample).steps[-1]

        # Expected: [system, user(text instr), user(image)]  (split, like reference)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "user"

        # First user message: text only, NO image
        first_user_content = msgs[1]["content"]
        assert all(c.get("type") != "image" for c in first_user_content)
        assert any(
            c.get("type") == "text" and "Open Settings" in c.get("text", "")
            for c in first_user_content
        )

        # Second user message: image only, NO text
        second_user_content = msgs[2]["content"]
        assert any(c.get("type") == "image" for c in second_user_content)
        assert all(c.get("type") != "text" or not c.get("text") for c in second_user_content)

    def test_within_window_history(self):
        """1 prior turn + current image, fits in full_history_size=3."""
        sample = _make_mobile_sample(num_steps=1)
        adapter = MAIUIMobileUseAdapter()
        msgs = adapter.unroll(sample).steps[-1]
        # [system, user(text), user(img0), assistant_a0, user(img_curr)]
        assert len(msgs) == 5
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "user", "assistant", "user"]
        # First user is text only
        assert all(c.get("type") != "image" for c in msgs[1]["content"])
        # Second user is image only
        assert any(c.get("type") == "image" for c in msgs[2]["content"])
        # Last user is image only
        assert any(c.get("type") == "image" for c in msgs[4]["content"])

    def test_beyond_window_history(self):
        """5 prior turns + current image — windowing kicks in (full_history_size=3)."""
        sample = _make_mobile_sample(num_steps=5)
        adapter = MAIUIMobileUseAdapter()
        msgs = adapter.unroll(sample).steps[-1]

        roles = [m["role"] for m in msgs]
        # [system, user(text), a0, a1, user(img2), a2, user(img3),
        #  a3, user(img4), a4, user(img_curr)]
        # full_history_size=3 means 3 most-recent turns keep image+assistant pairs.
        # 5 prior + 1 current = 6 turns. window_start = 6 - 3 = 3.
        # Old: turns 0,1,2 (assistant only). Recent: turns 3,4,5.
        # Wait — turn 0 is user(img0+instr), turn 1 is user(img1) etc, current is turn 5.
        # First user with text-only is turn 0's user (image stripped).
        # i=0,1,2: only assistant.
        # i=3,4: user(img only) + assistant.
        # i=5: user(img only) + (no assistant — current).
        # So: [system, user(text), a0, a1, a2, user(img3), a3, user(img4), a4, user(img_curr)]
        assert roles == [
            "system",
            "user",
            "assistant",
            "assistant",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]


# =============================================================================
# 6. MAIUIHistoryProtocol shape
# =============================================================================


class TestHistoryProtocol:
    def test_default_window_size_is_3(self):
        proto = MAIUIHistoryProtocol()
        assert proto.full_history_size == 3

    def test_no_history_splits_first_user_text_and_image(self):
        """The fits-in-window branch must still split text from image."""
        proto = MAIUIHistoryProtocol(full_history_size=3)
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open Settings."},
                ],
            },
        ]
        out = proto.process_messages(msgs)
        # Expected: [user(text only), user(image only)]
        assert len(out) == 2
        assert all(c.get("type") != "image" for c in out[0]["content"])
        assert any(c.get("type") == "image" for c in out[1]["content"])
        assert all(c.get("type") != "text" or not c.get("text") for c in out[1]["content"])


# =============================================================================
# 7. MAIUIMobileAgent.build_generation_prompt — reasoning_content fold
# =============================================================================


class _FakeProcessor:
    """Captures messages passed to apply_chat_template for inspection."""

    def __init__(self):
        self.captured_messages = None
        self.captured_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.captured_messages = messages
        self.captured_kwargs = kwargs
        return "RENDERED"


class TestBuildGenerationPrompt:
    def test_reasoning_content_and_tool_calls_folded_into_text(self):
        """``adapter.convert_message_to_agent`` folds reasoning + tool_calls
        into a single text block matching upstream mem2response byte-for-byte."""
        adapter = MAIUIMobileUseAdapter()
        out = adapter.convert_message_to_agent(
            {
                "role": "assistant",
                "content": [
                    {"type": "inline_reasoning", "text": "I should tap the icon at the top."},
                ],
                "tool_calls": [_mobile_call("tap", coordinate=[499, 499])],
            }
        )
        assert "reasoning_content" not in out
        assert "tool_calls" not in out
        assert len(out["content"]) == 1
        # Byte-exact match for upstream mem2response format
        expected = (
            "<thinking>\nI should tap the icon at the top.\n</thinking>\n"
            '<tool_call>\n{"name":"mobile_use","arguments":{"action":"click","coordinate":[499,499]}}\n</tool_call>'
        )
        assert out["content"][0]["text"] == expected

    def test_no_reasoning_content_passes_through(self):
        proc = _FakeProcessor()
        agent = AgentRegistry.get(
            "mai_ui@mobile@use",
            generate_fn=lambda **kw: {"response": ""},
            processor=proc,
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "x"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
        ]
        agent.build_generation_prompt(messages)
        assert "reasoning_content" not in proc.captured_messages[1]
        assert proc.captured_messages[1]["content"] == [{"type": "text", "text": "y"}]

    def test_does_not_pass_tools_kwarg_to_template(self):
        """The tools=kwarg path is forbidden — would inject Qwen-style <tools>."""
        proc = _FakeProcessor()
        agent = AgentRegistry.get(
            "mai_ui@mobile@use",
            generate_fn=lambda **kw: {"response": ""},
            processor=proc,
        )
        agent.build_generation_prompt([{"role": "user", "content": "x"}])
        assert "tools" not in proc.captured_kwargs


# =============================================================================
# 9. Characterization tests — MAI-UI data-flow goldens
# =============================================================================
#
# Goldens below were captured by running the current cua-lite code (adapter +
# action_space + protocol + agent) on each MAI-UI action shape and recording
# the observed output verbatim. They lock in CURRENT behavior so a planned
# refactor trips if anything changes.
#
# # OBSERVED notes document round-trip lossiness that the current code exhibits.
# They are NOT bugs being xfail'd — they are intended behavior (see the
# adapter/action_space docstrings) captured so refactors can't silently
# change them.


# --- shared raw-wire goldens per action ----------------------------------
# One representative raw for each of the 10 MAI-UI actions.
# Wire format: <thinking>...</thinking>\n<tool_call>{json}</tool_call>
_RAW_PER_ACTION: dict[str, str] = {
    "click": (
        "<thinking>\nTap the icon.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"click","coordinate":[499,499]}}\n</tool_call>'
    ),
    "long_press": (
        "<thinking>\nHold the item.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"long_press","coordinate":[100,200],"time":2.0}}\n</tool_call>'
    ),
    "type": (
        "<thinking>\nEnter text.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"type","text":"hello"}}\n</tool_call>'
    ),
    "swipe": (
        "<thinking>\nScroll up.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"swipe","coordinate":[500,800],"direction":"up"}}\n</tool_call>'
    ),
    "drag": (
        "<thinking>\nDrag across.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"drag","start_coordinate":[100,200],"end_coordinate":[700,200]}}\n</tool_call>'
    ),
    "open": (
        "<thinking>\nOpen Chrome.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"open","text":"Chrome"}}\n</tool_call>'
    ),
    "system_button": (
        "<thinking>\nGo home.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"system_button","button":"home"}}\n</tool_call>'
    ),
    "wait": (
        "<thinking>\nWait a moment.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":{"action":"wait"}}\n</tool_call>'
    ),
    "terminate": (
        "<thinking>\nDone.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"terminate","status":"success"}}\n</tool_call>'
    ),
    "answer": (
        "<thinking>\nProvide answer.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"answer","text":"42"}}\n</tool_call>'
    ),
}


def _render_assistant_as_raw(msg: dict[str, Any]) -> str:
    """Mirror the body of ``MAIUIMobileAgent.build_generation_prompt`` that
    folds reasoning + tool_calls into raw wire text. We do this without
    instantiating the agent so the tests stay adapter-scoped.

    Mirrors the production code's use of ``get_inline_reasoning`` so it
    transparently handles BOTH the legacy top-level ``reasoning_content``
    field AND ``InlineReasoningContent`` parts inside ``content``.
    """
    import json as _json

    from lite.core.messages import get_inline_reasoning as _get_inline_reasoning

    msg = copy.deepcopy(msg)
    thinking = _get_inline_reasoning(msg)
    tool_calls = msg.get("tool_calls")
    content = msg.get("content")
    if isinstance(content, str):
        existing_text = content
    elif isinstance(content, list):
        # Skip inline_reasoning parts (already captured via thinking).
        existing_text = "".join(
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") in ("text", "action_description")
        )
    else:
        existing_text = ""
    blocks: list[str] = []
    if thinking:
        blocks.append(f"<thinking>\n{thinking}\n</thinking>")
    if existing_text:
        blocks.append(existing_text)
    if tool_calls:
        for tc in tool_calls:
            tc_dict = {
                "name": tc["name"],
                "arguments": tc["arguments"],
            }
            tc_json = _json.dumps(tc_dict, separators=(",", ":"))
            blocks.append(f"<tool_call>\n{tc_json}\n</tool_call>")
    return "\n".join(blocks)


def _build_lite_trajectory(raws: list[str]) -> LiteSample:
    """Mobile trajectory via the shared builder, pinned to the MAI-UI
    adapter, with schemas for native semantic calls produced by these raws."""
    sample = build_lite_trajectory(
        _mai_adapter_with_natively_rendered_extras(),
        raws,
        platform="mobile",
        others={},
    )
    sample.metadata.extra_tool_schemas = _mai_natively_rendered_extra_schemas()
    return sample


# =============================================================================
# 9.1. Raw -> parse -> LiteMessage fields (per action)
# =============================================================================


class TestRawToLiteMessage:
    """For each action, assert the LiteMessage (post convert_from_agent)
    captures the canonical cua-lite fields."""

    adapter = _mai_adapter_with_natively_rendered_extras()

    @pytest.mark.parametrize("action", list(_RAW_PER_ACTION.keys()))
    def test_parse_then_convert_from_agent(self, action):
        raw = _RAW_PER_ACTION[action]
        parsed = self.adapter.parse_raw_assistant_response(raw)
        lite = self.adapter.convert_message_from_agent(parsed)
        assert lite["role"] == "assistant"
        # MAI-UI captures <thinking> as InlineReasoningContent inside content
        # (not Qwen3-VL's native <think> channel — different special tokens).
        assert any(
            isinstance(c, dict) and c.get("type") == "inline_reasoning"
            for c in lite.get("content", [])
        )
        # one tool_call per raw
        assert len(lite["tool_calls"]) == 1

    def test_click_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["click"]),
        )
        action = _single_mobile_action(lite["tool_calls"][0])
        assert action["action"] == "tap"
        # [499, 499] MAI-UI -> cua-lite = [int(round(499 * 1000 / 999))] = [499, 499]
        # (499 * 1000 / 999 = 499.499... rounds to 499 under banker's rounding).
        assert action["coordinate"] == [499, 499]

    def test_long_press_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["long_press"]),
        )
        action = _single_mobile_action(lite["tool_calls"][0])
        assert action["action"] == "long_press"
        assert action["duration"] == 2.0

    def test_type_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["type"]),
        )
        action = _single_mobile_action(lite["tool_calls"][0])
        assert action["action"] == "type"
        assert action["text"] == "hello"

    def test_swipe_lite_fields(self):
        """Direction-only swipe reverses to a (start, end) pair with fixed offset."""
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["swipe"]),
        )
        args = _single_mobile_action(lite["tool_calls"][0])
        assert args["action"] == "swipe"
        # anchor [500, 800] direction "up" -> start=anchor_rescaled, end above start.
        assert args["start_coordinate"][1] > args["coordinate"][1]

    def test_drag_lite_fields(self):
        """drag parses back as cua-lite `drag` with start/end coordinates."""
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["drag"]),
        )
        args = _single_mobile_action(lite["tool_calls"][0])
        assert args["action"] == "drag"
        # Rescaled by 1000/999.
        assert args["start_coordinate"] == [100, 200]
        # end_coordinate [700, 200] MAI -> 701 after rescale (700 * 1000 / 999)
        assert args["coordinate"] == [701, 200]

    def test_open_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["open"]),
        )
        tc = lite["tool_calls"][0]
        assert tool_call_name(tc) == "open_app"
        assert tool_call_arguments(tc)["app_name"] == "Chrome"

    def test_system_button_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["system_button"]),
        )
        action = _single_mobile_action(lite["tool_calls"][0])
        assert action["action"] == "system_button"
        assert action["button"] == "Home"

    def test_wait_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["wait"]),
        )
        action = _single_mobile_action(lite["tool_calls"][0])
        assert action["action"] == "wait"
        # OBSERVED (lite/agents/models/mai_ui/action_space.py:463):
        # reverse path for wait injects duration=1.0 when MAI-UI time is absent.
        assert action["duration"] == 1.0

    def test_terminate_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["terminate"]),
        )
        tc = lite["tool_calls"][0]
        assert tool_call_name(tc) == "terminate"
        assert tool_call_arguments(tc)["status"] == "success"

    def test_answer_lite_fields(self):
        lite = self.adapter.convert_message_from_agent(
            self.adapter.parse_raw_assistant_response(_RAW_PER_ACTION["answer"]),
        )
        tc = lite["tool_calls"][0]
        assert tool_call_name(tc) == "response"
        assert tool_call_arguments(tc)["text"] == "42"


# =============================================================================
# 9.2. LiteMessage -> MAI-UI agent -> raw
# =============================================================================


class TestLiteMessageToRaw:
    """For each action, build a LiteMessage by parsing + convert_from,
    then run it through convert_to_agent + render and assert the golden
    raw string byte-for-byte."""

    adapter = _mai_adapter_with_natively_rendered_extras()

    # The raws below are the OBSERVED output of the current pipeline
    # (parse -> convert_from -> convert_to -> render). For most actions they
    # equal _RAW_PER_ACTION (round-trip exact); for the lossy actions they
    # differ in documented ways.
    _EXPECTED_RENDERED: dict[str, str] = {
        "click": _RAW_PER_ACTION["click"],
        "long_press": _RAW_PER_ACTION["long_press"],
        "type": _RAW_PER_ACTION["type"],
        "swipe": _RAW_PER_ACTION["swipe"],
        "drag": _RAW_PER_ACTION["drag"],
        "open": _RAW_PER_ACTION["open"],
        "system_button": _RAW_PER_ACTION["system_button"],
        "terminate": _RAW_PER_ACTION["terminate"],
        "answer": _RAW_PER_ACTION["answer"],
        # OBSERVED: wait round-trip adds "time":1 (reverse path default —
        # lite/agents/models/mai_ui/action_space.py:463 injects duration=1.0,
        # compact_number() then renders as int on the convert_to_agent path).
        "wait": (
            "<thinking>\nWait a moment.\n</thinking>\n"
            '<tool_call>\n{"name":"mobile_use","arguments":'
            '{"action":"wait","time":1}}\n</tool_call>'
        ),
    }

    @pytest.mark.parametrize("action", list(_RAW_PER_ACTION.keys()))
    def test_render_matches_golden(self, action):
        raw = _RAW_PER_ACTION[action]
        parsed = self.adapter.parse_raw_assistant_response(raw)
        lite = self.adapter.convert_message_from_agent(parsed)
        agent_msg = self.adapter.convert_message_to_agent(lite)
        rendered = _render_assistant_as_raw(agent_msg)
        assert rendered == self._EXPECTED_RENDERED[action]


# =============================================================================
# 9.3. Raw round-trip (raw -> parse -> from -> to -> render)
# =============================================================================


class TestRawRoundTrip:
    """Round-trip each action raw through the full conversion pipeline."""

    adapter = _mai_adapter_with_natively_rendered_extras()

    # Exact byte-match for these actions.
    _EXACT_ACTIONS = [
        "click",
        "long_press",
        "type",
        "swipe",
        "drag",
        "open",
        "system_button",
        "terminate",
        "answer",
    ]

    @pytest.mark.parametrize("action", _EXACT_ACTIONS)
    def test_raw_round_trip_exact(self, action):
        raw = _RAW_PER_ACTION[action]
        parsed = self.adapter.parse_raw_assistant_response(raw)
        lite = self.adapter.convert_message_from_agent(parsed)
        agent_msg = self.adapter.convert_message_to_agent(lite)
        rendered = _render_assistant_as_raw(agent_msg)
        assert rendered == raw

    @staticmethod
    def _restored_tool_call_args(text: str) -> dict[str, Any]:
        """Parse the tool_call JSON out of MAI-UI's wire-format text."""
        import json as _json
        import re as _re

        match = _re.search(r"<tool_call>\n(.+?)\n</tool_call>", text, _re.DOTALL)
        assert match, f"no <tool_call> block in: {text!r}"
        return _json.loads(match.group(1))["arguments"]

    def test_wait_round_trip_injects_default_time(self):
        """OBSERVED: reverse path injects wait.time=1.0 when absent."""
        raw = _RAW_PER_ACTION["wait"]
        parsed = self.adapter.parse_raw_assistant_response(raw)
        lite = self.adapter.convert_message_from_agent(parsed)
        agent_msg = self.adapter.convert_message_to_agent(lite)
        args = self._restored_tool_call_args(agent_msg["content"][0]["text"])
        assert args["action"] == "wait"
        assert args["time"] == 1.0

    def test_coordinate_drift_bounded(self):
        """The 999<->1000 rescale may drift by ±1 on each axis per round-trip."""
        raw = _RAW_PER_ACTION["click"]  # [499, 499]
        parsed = self.adapter.parse_raw_assistant_response(raw)
        lite = self.adapter.convert_message_from_agent(parsed)
        agent_msg = self.adapter.convert_message_to_agent(lite)
        args = self._restored_tool_call_args(agent_msg["content"][0]["text"])
        for orig, new in zip([499, 499], args["coordinate"]):
            assert abs(orig - new) <= 1


# =============================================================================
# 9.4. Unroll structure (3-turn trajectory)
# =============================================================================


class TestUnrollStructure:
    """Validate top-level shape of unroll on a 3-turn trajectory."""

    adapter = _mai_adapter_with_natively_rendered_extras()

    def _sample(self):
        return _build_lite_trajectory(
            [
                _RAW_PER_ACTION["click"],
                _RAW_PER_ACTION["type"],
                _RAW_PER_ACTION["terminate"],
            ]
        )

    def test_num_steps_equals_num_turns(self):
        steps = self.adapter.unroll(self._sample()).steps
        assert len(steps) == 3

    def test_per_step_role_structure(self):
        """OBSERVED: for a 3-turn trajectory (fits inside full_history_size=3),
        the protocol always splits the first user into (text-only, image-only),
        so step i has 2 + 2*i messages after the system."""
        steps = self.adapter.unroll(self._sample()).steps
        expected_roles = [
            ["system", "user", "user", "assistant"],
            ["system", "user", "user", "assistant", "user", "assistant"],
            ["system", "user", "user", "assistant", "user", "assistant", "user", "assistant"],
        ]
        actual = [[m["role"] for m in step] for step in steps]
        assert actual == expected_roles

    def test_per_step_image_counts(self):
        """MAI-UI keeps ALL images within the 3-turn window (no summarization)."""
        steps = self.adapter.unroll(self._sample()).steps
        expected_counts = [1, 2, 3]
        actual = []
        for step in steps:
            n = sum(
                1
                for m in step
                for c in (m.get("content") if isinstance(m.get("content"), list) else [])
                if isinstance(c, dict) and c.get("type") == "image"
            )
            actual.append(n)
        assert actual == expected_counts


# =============================================================================
# 9.5. Unroll byte-exact target per action
# =============================================================================


class TestUnrollByteExactTargetPerAction:
    """For each action, a 1-turn trajectory's unrolled target must render
    to the action's expected raw (accounting for documented lossy fields)."""

    adapter = _mai_adapter_with_natively_rendered_extras()

    @pytest.mark.parametrize("action", list(_RAW_PER_ACTION.keys()))
    def test_target_renders_to_expected(self, action):
        raw = _RAW_PER_ACTION[action]
        sample = _build_lite_trajectory([raw])
        steps = self.adapter.unroll(sample).steps
        assert len(steps) == 1
        target = [m for m in steps[0] if m.get("role") == "assistant"][-1]
        rendered = _render_assistant_as_raw(target)
        assert rendered == TestLiteMessageToRaw._EXPECTED_RENDERED[action]


# =============================================================================
# 9.6. Unroll leakage check
# =============================================================================


class TestUnrollNoLeakage:
    """Target i's distinguishing info must NOT appear in sample i's prompt."""

    adapter = _mai_adapter_with_natively_rendered_extras()

    # Each raw carries a unique token so we can grep for leakage.
    _RAW_0 = (
        "<thinking>\nThink0.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"click","coordinate":[111,222]}}\n</tool_call>'
    )
    _RAW_1 = (
        "<thinking>\nThink1.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"type","text":"UNIQUETEXT1"}}\n</tool_call>'
    )
    _RAW_2 = (
        "<thinking>\nThink2.\n</thinking>\n"
        '<tool_call>\n{"name":"mobile_use","arguments":'
        '{"action":"answer","text":"UNIQUETEXT2"}}\n</tool_call>'
    )

    def _sample(self):
        return _build_lite_trajectory([self._RAW_0, self._RAW_1, self._RAW_2])

    @staticmethod
    def _message_text_blob(msg: dict[str, Any]) -> str:
        """Flatten a message to text for substring search."""
        import json as _json

        parts: list[str] = []
        if msg.get("reasoning_content"):
            parts.append(msg["reasoning_content"])
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
        for tc in msg.get("tool_calls", []) or []:
            parts.append(_json.dumps(tc, ensure_ascii=False))
        return "\n".join(parts)

    def test_target_markers_absent_from_prompt(self):
        steps = self.adapter.unroll(self._sample()).steps
        # markers that should appear ONLY in target i, never in earlier messages
        markers = {0: "[111, 222]", 1: "UNIQUETEXT1", 2: "UNIQUETEXT2"}
        for i, step in enumerate(steps):
            prompt_msgs = step[:-1]  # everything except the target
            prompt_blob = "\n".join(self._message_text_blob(m) for m in prompt_msgs)
            assert markers[i] not in prompt_blob, (
                f"step {i} leaked marker {markers[i]!r}:\n{prompt_blob}"
            )


# =============================================================================
# 9.7. Protocol windowing — boundary at full_history_size=3
# =============================================================================


class TestProtocolWindowing:
    """Exercise MAIUIHistoryProtocol at the boundary where windowing engages."""

    @staticmethod
    def _build_input(num_turns: int) -> list[dict[str, Any]]:
        adapter = MAIUIMobileUseAdapter()
        msgs: list[dict[str, Any]] = []
        raw = _RAW_PER_ACTION["click"]
        for i in range(num_turns):
            if i == 0:
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "index": 0},
                            {"type": "text", "text": "task"},
                        ],
                    }
                )
            else:
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "index": i},
                        ],
                    }
                )
            parsed = adapter.parse_raw_assistant_response(raw)
            msgs.append(adapter.convert_message_from_agent(parsed))
        # trailing user bubble
        msgs.append({"role": "user", "content": [{"type": "image", "index": num_turns}]})
        return msgs

    def _roles_and_counts(self, num_turns: int):
        proto = MAIUIHistoryProtocol(full_history_size=3)
        out = proto.process_messages(self._build_input(num_turns))
        roles = [m["role"] for m in out]
        imgs = []
        for m in out:
            c = m.get("content")
            if isinstance(c, list):
                imgs.append(sum(1 for x in c if isinstance(x, dict) and x.get("type") == "image"))
            else:
                imgs.append(0)
        return roles, imgs

    def test_input_3_turns(self):
        """3 assistant turns + trailing user = 4 turns. window=3 => turn 0
        becomes assistant-only. OBSERVED (captured live)."""
        roles, imgs = self._roles_and_counts(3)
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
        assert imgs == [0, 0, 1, 0, 1, 0, 1]

    def test_input_4_turns(self):
        """4 assistant turns + trailing user = 5 turns. window=3 => turns 0,1
        assistant-only; turn 0's user image dropped."""
        roles, imgs = self._roles_and_counts(4)
        assert roles == [
            "user",
            "assistant",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert imgs == [0, 0, 0, 1, 0, 1, 0, 1]

    def test_input_5_turns(self):
        """5 assistant turns + trailing user = 6 turns. window=3 => turns 0,1,2
        assistant-only."""
        roles, imgs = self._roles_and_counts(5)
        assert roles == [
            "user",
            "assistant",
            "assistant",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert imgs == [0, 0, 0, 0, 1, 0, 1, 0, 1]


# =============================================================================
# 9.8. Mutation purity — inputs must not be mutated
# =============================================================================


class TestMutationPurity:
    adapter = MAIUIMobileUseAdapter()

    def _assistant_lite(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "reasoning_content": "think",
            "content": [{"type": "text", "text": "hi"}],
            "tool_calls": [_mobile_call("tap", coordinate=[500, 500])],
        }

    def _assistant_agent(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "reasoning_content": "think",
            "content": [{"type": "text", "text": "hi"}],
            "tool_calls": [
                {
                    "name": "mobile_use",
                    "arguments": {"action": "click", "coordinate": [499, 499]},
                }
            ],
        }

    def test_convert_to_agent_pure(self):
        msg = self._assistant_lite()
        snap = copy.deepcopy(msg)
        _ = self.adapter.convert_message_to_agent(msg)
        assert msg == snap

    def test_convert_from_agent_pure(self):
        msg = self._assistant_agent()
        snap = copy.deepcopy(msg)
        _ = self.adapter.convert_message_from_agent(msg)
        assert msg == snap

    def test_protocol_process_messages_pure(self):
        proto = MAIUIHistoryProtocol(full_history_size=3)
        msgs = TestProtocolWindowing._build_input(3)
        snap = copy.deepcopy(msgs)
        _ = proto.process_messages(msgs)
        assert msgs == snap

    def test_unroll_sample_pure(self):
        sample = _build_lite_trajectory(
            [
                _RAW_PER_ACTION["click"],
                _RAW_PER_ACTION["type"],
            ]
        )
        snap_messages = copy.deepcopy(sample.messages)
        snap_images = copy.deepcopy(sample.images)
        _ = self.adapter.unroll(sample)
        assert sample.messages == snap_messages
        assert sample.images == snap_images


# =============================================================================
# 9.9. Sample independence
# =============================================================================


class TestSampleIndependence:
    """unroll twice returns independent step outputs; mutating one must
    not affect the other."""

    adapter = MAIUIMobileUseAdapter()

    def _sample(self) -> LiteSample:
        return _build_lite_trajectory(
            [
                _RAW_PER_ACTION["click"],
                _RAW_PER_ACTION["type"],
            ]
        )

    def test_two_unroll_calls_are_independent(self):
        sample = self._sample()
        first = self.adapter.unroll(sample).steps
        second = self.adapter.unroll(sample).steps
        # Mutate first's target text. After convert_message_to_agent the
        # target is a single text block (tool_calls dropped post-fold).
        first[-1][-1]["content"][0]["text"] = "MUTATED"
        assert second[-1][-1]["content"][0]["text"] != "MUTATED"

    def test_siblings_are_independent(self):
        steps = self.adapter.unroll(self._sample()).steps
        # Mutate step 0's target text; step 1 must be untouched.
        steps[0][-1]["content"][0]["text"] = "MUTATED"
        assert steps[1][-1]["content"][0]["text"] != "MUTATED"


# =============================================================================
# 9.10. Edge cases
# =============================================================================


class TestEdgeCases:
    adapter = MAIUIMobileUseAdapter()

    def test_empty_messages_unrolls_to_nothing(self):
        from lite.core import LiteCUAMetadata as _Meta

        empty = LiteSample(
            metadata=_Meta(
                dims=(_Meta.Platform.MOBILE, _Meta.TaskType.USE),
            ),
            messages=[],
            images=[],
        )
        assert self.adapter.unroll(empty).steps == []

    def test_single_turn_trajectory(self):
        """1-turn sample: system + user(text) + user(image) + target assistant."""
        sample = _build_lite_trajectory([_RAW_PER_ACTION["click"]])
        steps = self.adapter.unroll(sample).steps
        assert len(steps) == 1
        roles = [m["role"] for m in steps[0]]
        assert roles == ["system", "user", "user", "assistant"]

    def test_10_turn_trajectory(self):
        """Long trajectory: 10 steps, one per turn; windowing kicks in
        from turn 4 onward (full_history_size=3 => keep 3 most recent with
        images, earlier become assistant-only)."""
        sample = _build_lite_trajectory([_RAW_PER_ACTION["click"]] * 10)
        steps = self.adapter.unroll(sample).steps
        assert len(steps) == 10
        # Last step has 10 turns; windowing keeps exactly 3 user(image) msgs
        # in the prompt (trailing turn is the target, no user bubble for it
        # because the trajectory ends after the assistant).
        last = steps[-1]
        n_user_with_image = sum(
            1
            for m in last
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(c.get("type") == "image" for c in m["content"])
        )
        # windowed turns are the LAST 3 turns; they keep user(image) + assistant.
        assert n_user_with_image == 3

    def test_assistant_without_tool_calls(self):
        """An assistant message with only ``action_description`` content and
        no tool_calls: convert_message_to_agent drops it (no prose slot in
        MAI-UI's SFT distribution — ``<thinking>``/``<tool_call>`` only),
        producing empty content."""
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "text only"}],
        }
        agent = self.adapter.convert_message_to_agent(msg)
        # Action-description is dropped — MAI-UI format has no prose slot.
        assert agent["content"] == []
        # Round-trip from_agent on empty content stays empty.
        back = self.adapter.convert_message_from_agent(agent)
        assert back["content"] == []
