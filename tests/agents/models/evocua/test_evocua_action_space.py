"""
Tests for EvoCUADesktopActionSpace.

Covers:
  1. Registry and available actions
  2. Tool call creation (computer_use with key_down/key_up)
  3. Tool call conversion: CUA-lite ↔ EvoCUA (round-trip)

Run:
    uv run pytest tests/agents/models/evocua/test_evocua_action_space.py -v
"""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import agent_adapter_for, enum_of

from lite.agents.core.action_space.base import ActionSpaceRegistry, LiteDesktopActionSpace
from lite.agents.models.evocua.action_space import EvoCUADesktopActionSpace
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.action_space import lite_action_batch_child_name_errors
from lite.core.tools.action_space.batches import validate_lite_action_batch_structure
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

_TERMINATE_SCHEMA = make_tool_schema(
    "terminate",
    description="Finish the task.",
    parameters={
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["success", "failure"]}},
        "required": ["status"],
    },
)
_RESPONSE_SCHEMA = make_tool_schema(
    "response",
    description="Answer the task.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
_OPEN_APP_SCHEMA = make_tool_schema(
    "open_app",
    description="Open an app.",
    parameters={
        "type": "object",
        "properties": {"app_name": {"type": "string", "enum": ["Settings"]}},
        "required": ["app_name"],
    },
)


def _only_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == "computer"
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _assert_bare_model_function_projection(tool_call, name="computer_use"):
    assert set(tool_call) == {"name", "arguments"}
    assert tool_call["name"] == name
    return tool_call


# =============================================================================
# 1. Registry and available actions
# =============================================================================


class TestRegistryAndActions:
    def test_registry_key(self):
        space = ActionSpaceRegistry.get("evocua@desktop")
        assert isinstance(space, EvoCUADesktopActionSpace)

    def test_browser_alias(self):
        space = ActionSpaceRegistry.get("evocua@browser")
        assert isinstance(space, EvoCUADesktopActionSpace)

    def test_single_tool(self):
        schemas = EvoCUADesktopActionSpace.get_tool_schemas()
        assert len(schemas) == 1
        assert tool_schema_name(schemas[0]) == "computer_use"

    def test_action_enum(self):
        schema = EvoCUADesktopActionSpace.get_tool_schemas()[0]
        actions = set(tool_schema_parameters(schema)["properties"]["action"]["enum"])
        expected = {
            "key",
            "key_down",
            "key_up",
            "type",
            "mouse_move",
            "left_click",
            "left_click_drag",
            "right_click",
            "middle_click",
            "double_click",
            "triple_click",
            "scroll",
            "wait",
            "terminate",
        }
        assert actions == expected

    def test_no_hscroll_or_answer_in_enum(self):
        """hscroll/answer are in description text but NOT in the action enum."""
        schema = EvoCUADesktopActionSpace.get_tool_schemas()[0]
        actions = set(tool_schema_parameters(schema)["properties"]["action"]["enum"])
        assert "hscroll" not in actions
        assert "answer" not in actions

    def test_native_semantic_scope_is_terminate_only(self):
        assert EvoCUADesktopActionSpace.QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES == {
            "terminate": frozenset({"terminate"})
        }

    def test_schema_is_canonical_function_tool_schema(self):
        schema = EvoCUADesktopActionSpace.get_tool_schemas()[0]
        assert schema["type"] == "function"
        assert "name_for_human" not in schema
        assert "args_format" not in schema


# =============================================================================
# 2. Tool call creation
# =============================================================================


class TestToolCalls:
    def test_key_down(self):
        tc = EvoCUADesktopActionSpace.computer_use(action="key_down", keys=["shift"])
        assert tool_call_name(tc) == "computer_use"
        assert tool_call_arguments(tc)["action"] == "key_down"
        assert tool_call_arguments(tc)["keys"] == ["shift"]

    def test_key_up(self):
        tc = EvoCUADesktopActionSpace.computer_use(action="key_up", keys=["shift"])
        assert tool_call_arguments(tc)["action"] == "key_up"
        assert tool_call_arguments(tc)["keys"] == ["shift"]

    def test_left_click(self):
        tc = EvoCUADesktopActionSpace.computer_use(action="left_click", coordinate=[500, 300])
        assert tool_call_arguments(tc)["action"] == "left_click"
        assert tool_call_arguments(tc)["coordinate"] == [500, 300]

    def test_none_args_omitted(self):
        tc = EvoCUADesktopActionSpace.computer_use(action="left_click")
        assert "coordinate" not in tool_call_arguments(tc)


# =============================================================================
# 3. Tool call conversion
# =============================================================================


class TestToolCallConversion:
    def setup_method(self):
        self.space = EvoCUADesktopActionSpace()

    def test_key_down_to_agent(self):
        tc = [LiteDesktopActionSpace.key_down(keys=["shift"])]
        result = self.space.convert_tool_calls_to_agent(tc)
        agent_call = _assert_bare_model_function_projection(result[0])
        assert agent_call["arguments"]["action"] == "key_down"
        assert agent_call["arguments"]["keys"] == ["shift"]

    def test_key_up_to_agent(self):
        tc = [LiteDesktopActionSpace.key_up(keys=["shift"])]
        result = self.space.convert_tool_calls_to_agent(tc)
        agent_call = _assert_bare_model_function_projection(result[0])
        assert agent_call["arguments"]["action"] == "key_up"
        assert agent_call["arguments"]["keys"] == ["shift"]

    def test_key_down_from_agent(self):
        tc = [{"name": "computer_use", "arguments": {"action": "key_down", "keys": ["shift"]}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_action(result) == {"action": "key_down", "keys": ["shift"]}

    def test_key_up_from_agent(self):
        tc = [{"name": "computer_use", "arguments": {"action": "key_up", "keys": ["ctrl"]}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_action(result) == {"action": "key_up", "keys": ["ctrl"]}

    @pytest.mark.parametrize("action", ["key", "key_down", "key_up"])
    @pytest.mark.parametrize(
        "raw_keys,expected",
        [
            ("ctrl++", ["ctrl", "+"]),
            ("ctrl+-", ["ctrl", "-"]),
            ("ctrl+=", ["ctrl", "="]),
        ],
    )
    def test_key_actions_from_agent_string_chord_use_core_normalizer(
        self, action, raw_keys, expected
    ):
        """Model string chords are normalized once by the Lite key owner."""
        tc = [{"name": "computer_use", "arguments": {"action": action, "keys": raw_keys}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_action(result) == {"action": action, "keys": expected}

    @pytest.mark.parametrize("action", ["key", "key_down", "key_up"])
    @pytest.mark.parametrize("raw_keys", ["ctrl left", "ctrl -"])
    def test_key_actions_from_agent_reject_phrase_like_strings(self, action, raw_keys):
        tc = [{"name": "computer_use", "arguments": {"action": action, "keys": raw_keys}}]

        with pytest.raises(ValueError, match="unknown key token"):
            self.space.convert_tool_calls_from_agent(tc)

    def test_click_to_left_click(self):
        tc = [LiteDesktopActionSpace.click(coordinate=[500, 300])]
        result = self.space.convert_tool_calls_to_agent(tc)
        agent_call = _assert_bare_model_function_projection(result[0])
        assert agent_call["arguments"]["action"] == "left_click"

    def test_computer_batch_to_agent_unwraps(self):
        tc = make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [500, 300]},
                    {"action": "type", "text": "hello"},
                ]
            },
        )
        result = self.space.convert_tool_calls_to_agent([tc])
        for agent_call in result:
            _assert_bare_model_function_projection(agent_call)
        assert [r["arguments"]["action"] for r in result] == ["left_click", "type"]
        assert result[1]["arguments"]["text"] == "hello"

    def test_left_click_from_agent(self):
        tc = [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [500, 300]},
            }
        ]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert _only_action(result) == {"action": "click", "coordinate": [500, 300]}

    def test_terminate_from_agent(self):
        tc = [{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}]
        result = self.space.convert_tool_calls_from_agent(tc)
        assert tool_call_name(result[0]) == "terminate"
        assert tool_call_arguments(result[0])["status"] == "failure"

    def test_active_native_terminate_from_agent(self):
        tc = [{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}]
        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"terminate"},
            active_extra_tool_schemas=[_TERMINATE_SCHEMA],
        )
        assert result == [make_tool_call("terminate", {"status": "success"})]

    def test_inactive_native_terminate_from_agent_is_canonicalized_for_env_feedback(self):
        tc = [{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}]
        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names=set(),
            active_extra_tool_schemas=[],
        )
        assert result == [make_tool_call("terminate", {"status": "success"})]

    @pytest.mark.parametrize("arguments", [{"status": "bogus"}, {}])
    def test_active_native_terminate_invalid_arguments_reach_env_feedback(self, arguments):
        tc = [{"name": "computer_use", "arguments": {"action": "terminate", **arguments}}]
        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={"terminate"},
            active_extra_tool_schemas=[_TERMINATE_SCHEMA],
        )
        assert result == [make_tool_call("terminate", arguments)]

    @pytest.mark.parametrize(
        "native_action,schema",
        [
            ("answer", _RESPONSE_SCHEMA),
            ("open", _OPEN_APP_SCHEMA),
        ],
    )
    def test_answer_and_open_are_not_evocua_native_semantics(self, native_action, schema):
        """EvoCUA narrows its parent's enum: ``answer``/``open`` are NOT its dialect.

        Qwen3-VL would lower these to canonical ``response``/``open_app``; EvoCUA
        must not, even when those extras are active. But "not my dialect" means
        the value goes to env ingress as the model spelled it -- not that the
        call disappears, which would end the episode on a single word.
        """
        tc = [{"name": "computer_use", "arguments": {"action": native_action, "text": "Settings"}}]
        result = self.space.convert_tool_calls_from_agent(
            tc,
            active_extra_tool_names={tool_schema_name(schema)},
            active_extra_tool_schemas=[schema],
        )
        assert result, "zero tool calls would end the episode"
        assert tool_call_name(result[0]) == "computer"
        actions = tool_call_arguments(result[0])["actions"]
        assert actions[0]["action"] == native_action
        # Specifically NOT lowered to the canonical extra the parent would use.
        assert tool_call_name(result[0]) != tool_schema_name(schema)

    def test_standalone_extra_to_agent_projects_to_bare_model_function(self):
        tc = make_tool_call("bash", {"command": "pwd"})
        assert self.space.convert_tool_calls_to_agent([tc]) == [
            {"name": "bash", "arguments": {"command": "pwd"}}
        ]

    def test_standalone_extra_from_agent_canonicalizes_for_lite(self):
        tc = {"name": "bash", "arguments": {"command": "pwd"}}
        assert self.space.convert_tool_calls_from_agent([tc]) == [
            make_tool_call("bash", {"command": "pwd"})
        ]

    def test_unknown_wrapped_action_from_agent_reaches_env_ingress(self):
        """An off-enum wrapper action must cost a TURN, never the EPISODE.

        Returning ``[]`` here left the assistant turn with zero tool calls, and
        ``AgentBase.sample`` treats that as terminal -- so one action name EvoCUA
        does not know ended the rollout. Emitting the invalid batch instead lets
        env ingress answer it with feedback keyed to the call id, which is what
        every sibling wrapper family does.
        """
        tc = [{"name": "computer_use", "arguments": {"action": "bash", "command": "pwd"}}]
        out = self.space.convert_tool_calls_from_agent(tc)
        assert out, "zero tool calls would end the episode"
        assert tool_call_name(out[0]) == "computer"
        # The batch names the model's OWN value, so the rejection tells the model
        # what it actually said rather than a generic parse error.
        assert tool_call_arguments(out[0])["actions"][0]["action"] == "bash"
        children, error = validate_lite_action_batch_structure(
            "computer", tool_call_arguments(out[0])
        )
        assert len(children) == 1
        error = lite_action_batch_child_name_errors("computer", children).get(0)
        assert error is not None and error.child_action_name == "bash"

    @pytest.mark.parametrize(
        "cua_tc",
        [
            LiteDesktopActionSpace.click(coordinate=[500, 300]),
            LiteDesktopActionSpace.click(coordinate=[100, 200], button="right"),
            LiteDesktopActionSpace.click(coordinate=[100, 200], clicks=2),
            LiteDesktopActionSpace.type(text="hello"),
            LiteDesktopActionSpace.key(keys=["ctrl", "v"]),
            LiteDesktopActionSpace.key(keys=["ctrl", "+"]),
            LiteDesktopActionSpace.key_down(keys=["shift"]),
            LiteDesktopActionSpace.key_down(keys=["ctrl", "+"]),
            LiteDesktopActionSpace.key_up(keys=["shift"]),
            LiteDesktopActionSpace.wait(duration=1.5),
            make_tool_call("terminate", {"status": "success"}),
        ],
    )
    def test_round_trip(self, cua_tc):
        space = EvoCUADesktopActionSpace()
        agent_tcs = space.convert_tool_calls_to_agent([cua_tc])
        restored = space.convert_tool_calls_from_agent(agent_tcs)
        if tool_call_name(cua_tc) == "terminate":
            assert restored == [cua_tc]
        else:
            assert len(restored) == 1
            assert tool_call_name(restored[0]) == tool_call_name(cua_tc)
            assert tool_call_arguments(restored[0]) == tool_call_arguments(cua_tc)


# =============================================================================
# 4. Malformed-but-recoverable provider output
# =============================================================================


class TestMalformedProviderOutput:
    def setup_method(self):
        self.space = EvoCUADesktopActionSpace()

    @pytest.mark.parametrize(
        "name,args",
        [
            ("left_click", {"coordinate": [500, 300]}),
            ("triple_click", {"coordinate": [500, 300]}),
            ("left_click_drag", {"coordinate": [500, 300]}),
            ("key_down", {"keys": ["shift"]}),
            ("scroll", {"pixels": -300}),
        ],
    )
    def test_flat_native_action_value_used_as_tool_name_converts(self, name, args):
        """The model dropped ``computer_use`` and used the native action VALUE
        as the tool name. Native vocabulary is read as ``action`` so flat output
        runs the same dispatch branch as nested output."""
        flat = self.space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
        nested = self.space.convert_tool_calls_from_agent(
            [{"name": "computer_use", "arguments": {"action": name, **args}}]
        )
        assert flat == nested

    def test_wrong_wrapper_name_with_nested_action_converts(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "computer", "arguments": {"action": "left_click", "coordinate": [5, 6]}}]
        )
        assert _only_action(out) == {"action": "click", "coordinate": [5, 6]}

    def test_flat_terminate_becomes_canonical_terminate(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "terminate", "arguments": {"status": "failure"}}]
        )
        assert out == [make_tool_call("terminate", {"status": "failure"})]

    def test_flat_answer_stays_a_standalone_tool(self):
        """EvoCUA dropped Qwen3-VL's ``answer`` enum member, so ``answer`` is
        NOT native vocabulary here: it stays a by-name call for env ingress."""
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "answer", "arguments": {"text": "42"}}]
        )
        assert out == [make_tool_call("answer", {"text": "42"})]

    def test_non_native_name_stays_a_standalone_tool(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "summarize", "arguments": {"text": "hi"}}]
        )
        assert out == [make_tool_call("summarize", {"text": "hi"})]

    def test_canonical_action_this_enum_cannot_spell_still_rebuilds(self):
        """``hold_key`` leaves ``to_agent`` as a bare model-function projection;
        parsing through the action owner keeps it in the ``computer`` batch."""
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "hold_key", "arguments": {"keys": ["shift"], "duration": 2}}]
        )
        assert _only_action(out) == {"action": "hold_key", "keys": ["shift"], "duration": 2}

    def test_active_extra_tool_outranks_a_colliding_canonical_action_name(self):
        """A browsergym ``click(bid=...)`` is the env's tool. Without the
        admission gate it reached ``LiteDesktopActionSpace.click(bid=...)`` and
        raised TypeError instead of ever reaching env ingress."""
        schema = make_tool_schema(
            "click",
            description="Click an element by bid.",
            parameters={
                "type": "object",
                "properties": {"bid": {"type": "string"}},
                "required": ["bid"],
            },
        )
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "click", "arguments": {"bid": "a51"}}],
            active_extra_tool_names={"click"},
            active_extra_tool_schemas=[schema],
        )
        assert out == [make_tool_call("click", {"bid": "a51"})]


def test_evocua_response_render_fails_loudly() -> None:
    """Canonical ``response`` must not leak onto EvoCUA's native wire."""
    response_call = make_tool_call("response", {"text": "42"})
    with pytest.raises(ValueError, match=r"^EvoCUA cannot render canonical tool 'response'$"):
        EvoCUADesktopActionSpace().convert_tool_calls_to_agent([response_call])

    adapter = agent_adapter_for(
        "evocua@desktop@use",
        "desktop",
        extra_tool_schemas=[_RESPONSE_SCHEMA],
    )
    message = {"role": "assistant", "content": [], "tool_calls": [response_call]}
    with pytest.raises(ValueError, match=r"^EvoCUA cannot render canonical tool 'response'$"):
        adapter.convert_message_to_agent(message)


def test_inherited_qwen_families_narrow_their_enum() -> None:
    adapter = agent_adapter_for("evocua@desktop@use", "desktop")
    space = type(adapter.action_space)

    full = space.get_tool_schemas()
    trimmed = space.filter_tool_schemas_for_valid_actions(full, ["click"])

    full_enum = enum_of(full, "computer_use")
    trimmed_enum = enum_of(trimmed, "computer_use")
    assert trimmed_enum is not None
    assert set(trimmed_enum) < set(full_enum)
