"""Focused Qwen2.5-VL action-space guards."""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.models.qwen2_5_vl.action_space import (
    Qwen2_5VLDesktopActionSpace,
    Qwen2_5VLMobileActionSpace,
)
from lite.core.tools import make_tool_call
from lite.core.tools.action_space import (
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_name,
    validate_lite_tool_call,
)
from lite.core.tools.schemas import tool_schema_parameters


def _action_enum(action_space_cls, active_extra_tool_names: set[str]) -> list[str]:
    schemas = action_space_cls.filter_qwen_action_values_for_active_extra_tools(
        action_space_cls.get_tool_schemas(),
        active_extra_tool_names,
    )
    assert len(schemas) == 1
    return tool_schema_parameters(schemas[0])["properties"]["action"]["enum"]


def _canonical_args(tool_call, name):
    assert validate_lite_tool_call(tool_call, "tool_call", require_id=False) is None
    assert tool_call_name(tool_call) == name
    return tool_call_arguments(tool_call)


def test_desktop_public_constructor_returns_canonical_lite_tool_call():
    tc = Qwen2_5VLDesktopActionSpace.computer_use(
        action="left_click",
        coordinate=[100, 200],
    )
    assert _canonical_args(tc, "computer_use") == {
        "action": "left_click",
        "coordinate": [100, 200],
    }


def test_mobile_public_constructor_returns_canonical_lite_tool_call():
    tc = Qwen2_5VLMobileActionSpace.mobile_use(
        action="click",
        coordinate=[100, 200],
    )
    assert _canonical_args(tc, "mobile_use") == {
        "action": "click",
        "coordinate": [100, 200],
    }


def test_mobile_top_level_extra_from_agent_canonicalizes_for_lite():
    space = Qwen2_5VLMobileActionSpace()
    tc = {"name": "report_infeasible", "arguments": {"reason": "blocked"}}
    assert space.convert_tool_calls_from_agent([tc]) == [
        make_tool_call("report_infeasible", {"reason": "blocked"})
    ]


def test_mobile_open_app_to_agent_uses_native_open():
    space = Qwen2_5VLMobileActionSpace()
    tc = make_tool_call("open_app", {"app_name": "Settings"})
    assert space.convert_tool_calls_to_agent([tc]) == [
        {"name": "mobile_use", "arguments": {"action": "open", "text": "Settings"}}
    ]


def test_desktop_qwen_action_values_gate_terminate_by_active_extra():
    assert "terminate" not in _action_enum(Qwen2_5VLDesktopActionSpace, set())
    assert "terminate" in _action_enum(
        Qwen2_5VLDesktopActionSpace,
        {"terminate"},
    )


def test_mobile_qwen_action_values_gate_open_and_terminate_by_active_extras():
    inactive = set(_action_enum(Qwen2_5VLMobileActionSpace, set()))
    assert "open" not in inactive
    assert "terminate" not in inactive

    open_only = set(_action_enum(Qwen2_5VLMobileActionSpace, {"open_app"}))
    assert "open" in open_only
    assert "terminate" not in open_only

    terminate_only = set(_action_enum(Qwen2_5VLMobileActionSpace, {"terminate"}))
    assert "open" not in terminate_only
    assert "terminate" in terminate_only


def test_desktop_computer_batch_to_agent_unwraps_without_parse_grouping():
    space = Qwen2_5VLDesktopActionSpace()
    tc = make_tool_call(
        "computer",
        {
            "actions": [
                {"action": "click", "coordinate": [100, 200]},
                {"action": "type", "text": "hello"},
            ]
        },
    )

    rendered = space.convert_tool_calls_to_agent([tc])

    assert [call["name"] for call in rendered] == ["computer_use", "computer_use"]
    assert rendered[0]["arguments"] == {"action": "left_click", "coordinate": [100, 200]}
    assert rendered[1]["arguments"] == {"action": "type", "text": "hello"}


def test_desktop_native_wrappers_from_agent_use_action_batch():
    space = Qwen2_5VLDesktopActionSpace()
    calls = [
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [100, 200]}},
        {"name": "computer_use", "arguments": {"action": "type", "text": "hello"}},
    ]

    parsed = space.convert_tool_calls_from_agent(calls)

    assert [tool_call_name(call) for call in parsed] == ["computer"]
    assert tool_call_arguments(parsed[0])["actions"] == [
        {"action": "click", "coordinate": [100, 200]},
        {"action": "type", "text": "hello"},
    ]


@pytest.mark.parametrize(
    "raw_keys,expected",
    [
        ("ctrl++", ["ctrl", "+"]),
        ("ctrl+-", ["ctrl", "-"]),
        ("ctrl+=", ["ctrl", "="]),
    ],
)
def test_desktop_key_string_chord_uses_core_normalizer(raw_keys, expected):
    space = Qwen2_5VLDesktopActionSpace()

    parsed = space.convert_tool_calls_from_agent(
        [{"name": "computer_use", "arguments": {"action": "key", "keys": raw_keys}}]
    )

    assert tool_call_arguments(parsed[0])["actions"] == [{"action": "key", "keys": expected}]


def test_desktop_key_punctuation_round_trips_through_native_wire():
    space = Qwen2_5VLDesktopActionSpace()
    call = LiteDesktopActionSpace.key(keys=["ctrl", "+"])

    assert space.convert_tool_calls_from_agent(space.convert_tool_calls_to_agent([call])) == [
        call
    ]


@pytest.mark.parametrize("raw_keys", ["ctrl left", "ctrl -"])
def test_desktop_key_string_rejects_phrase_like_tokens(raw_keys):
    space = Qwen2_5VLDesktopActionSpace()

    with pytest.raises(ValueError, match="unknown key token"):
        space.convert_tool_calls_from_agent(
            [{"name": "computer_use", "arguments": {"action": "key", "keys": raw_keys}}]
        )


def test_desktop_bare_canonical_action_name_stays_standalone():
    """Closure guard: the flat-name recovery reads NATIVE vocabulary only.

    Matching canonical Lite names instead would both miss ``left_click`` and
    feed native argument names into Lite constructors, so a canonical-only name
    must stay a by-name call for env ingress to judge.
    """
    space = Qwen2_5VLDesktopActionSpace()

    assert space.convert_tool_calls_from_agent(
        [
            {"name": "click", "arguments": {"coordinate": [100, 200]}},
            {"name": "screenshot", "arguments": {}},
        ]
    ) == [
        make_tool_call("click", {"coordinate": [100, 200]}),
        make_tool_call("screenshot", {}),
    ]


def test_desktop_native_terminate_from_agent_parses_standalone():
    space = Qwen2_5VLDesktopActionSpace()
    tc = {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}

    assert space.convert_tool_calls_from_agent([tc]) == [
        make_tool_call("terminate", {"status": "success"})
    ]


def test_mobile_batch_to_agent_unwraps_without_parse_grouping():
    space = Qwen2_5VLMobileActionSpace()
    tc = make_tool_call(
        "mobile",
        {
            "actions": [
                {"action": "tap", "coordinate": [100, 200]},
                {"action": "type", "text": "hello"},
            ]
        },
    )

    rendered = space.convert_tool_calls_to_agent([tc])

    assert [call["name"] for call in rendered] == ["mobile_use", "mobile_use"]
    assert rendered[0]["arguments"] == {"action": "click", "coordinate": [100, 200]}
    assert rendered[1]["arguments"] == {"action": "type", "text": "hello"}


def test_mobile_native_wrappers_from_agent_use_action_batch():
    space = Qwen2_5VLMobileActionSpace()
    calls = [
        {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [100, 200]}},
        {"name": "mobile_use", "arguments": {"action": "type", "text": "hello"}},
    ]

    parsed = space.convert_tool_calls_from_agent(calls)

    assert [tool_call_name(call) for call in parsed] == ["mobile"]
    assert tool_call_arguments(parsed[0])["actions"] == [
        {"action": "tap", "coordinate": [100, 200], "clicks": 1},
        {"action": "type", "text": "hello"},
    ]


def test_mobile_bare_canonical_action_name_stays_standalone():
    """Closure guard, mobile side — see the desktop twin."""
    space = Qwen2_5VLMobileActionSpace()

    assert space.convert_tool_calls_from_agent(
        [
            {"name": "tap", "arguments": {"coordinate": [100, 200]}},
            {"name": "screenshot", "arguments": {}},
        ]
    ) == [
        make_tool_call("tap", {"coordinate": [100, 200]}),
        make_tool_call("screenshot", {}),
    ]


def test_mobile_native_terminate_from_agent_parses_standalone():
    space = Qwen2_5VLMobileActionSpace()
    tc = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}

    assert space.convert_tool_calls_from_agent([tc]) == [
        make_tool_call("terminate", {"status": "success"})
    ]


def test_mobile_native_open_from_agent_parses_standalone_when_inactive():
    space = Qwen2_5VLMobileActionSpace()
    tc = {"name": "mobile_use", "arguments": {"action": "open", "text": "Settings"}}

    assert space.convert_tool_calls_from_agent([tc]) == [
        make_tool_call("open_app", {"app_name": "Settings"})
    ]


def _assert_invalid_action_batch(out, action_batch_tool_name, child):
    """The unknown-action policy: one deliberately INVALID action-batch.

    Env ingress rejects it by name, so the model sees the rejection instead of
    the action vanishing.
    """
    assert len(out) == 1
    assert tool_call_name(out[0]) == action_batch_tool_name
    assert tool_call_arguments(out[0]) == {"actions": [child]}
    children, error = validate_lite_action_batch_structure(
        action_batch_tool_name,
        tool_call_arguments(out[0]),
    )
    assert len(children) == 1
    error = lite_action_batch_child_name_errors(action_batch_tool_name, children).get(0)
    assert error is not None
    assert error.child_action_name == child["action"]


def test_desktop_unknown_wrapper_action_from_agent_is_invalid_action_batch():
    space = Qwen2_5VLDesktopActionSpace()
    tc = {"name": "computer_use", "arguments": {"action": "report_infeasible", "reason": "blocked"}}
    _assert_invalid_action_batch(
        space.convert_tool_calls_from_agent([tc]),
        "computer",
        {"action": "report_infeasible", "reason": "blocked"},
    )


def test_mobile_unknown_wrapper_action_from_agent_is_invalid_action_batch():
    space = Qwen2_5VLMobileActionSpace()
    tc = {"name": "mobile_use", "arguments": {"action": "open_app", "app_name": "Settings"}}
    _assert_invalid_action_batch(
        space.convert_tool_calls_from_agent([tc]),
        "mobile",
        {"action": "open_app", "app_name": "Settings"},
    )


# ---------------------------------------------------------------------------
# Native mobile ``key`` (adb keyevent) — cua-lite mobile has NO ``key``
# action, so this must degrade, never raise (an AttributeError escapes the
# env's model-action-error handling and burns the turn as a parse error).
# Keyevents that name a lite ``system_button`` map; the rest take the same
# unknown-action path as any other inexpressible wrapper action.
# ---------------------------------------------------------------------------


def test_mobile_key_keyevent_maps_to_system_button():
    space = Qwen2_5VLMobileActionSpace()
    for text, button in [
        ("back", "Back"),
        ("HOME", "Home"),
        ("KEYCODE_MENU", "Menu"),
        ("enter", "Enter"),
        ("app_switch", "Recent"),
    ]:
        out = space.convert_tool_calls_from_agent(
            [
                {"name": "mobile_use", "arguments": {"action": "key", "text": text}},
            ]
        )
        assert out == [
            make_tool_call(
                "mobile",
                {
                    "actions": [
                        {"action": "system_button", "button": button},
                    ]
                },
            )
        ], text


def test_mobile_key_unmappable_keyevent_is_invalid_action_batch_not_raised():
    space = Qwen2_5VLMobileActionSpace()
    for text in ["volume_up", "power", "camera", "clear", ""]:
        _assert_invalid_action_batch(
            space.convert_tool_calls_from_agent(
                [
                    {"name": "mobile_use", "arguments": {"action": "key", "text": text}},
                ]
            ),
            "mobile",
            {"action": "key", "text": text},
        )
    # ``make_tool_call`` drops ``None`` arguments, so the child keeps only the
    # unknown action name — still rejected by name, still not raised.
    _assert_invalid_action_batch(
        space.convert_tool_calls_from_agent(
            [{"name": "mobile_use", "arguments": {"action": "key", "text": None}}]
        ),
        "mobile",
        {"action": "key"},
    )


def test_mobile_key_is_advertised_so_it_must_be_parseable():
    """The native enum offers ``key``; every offered entry must parse without
    raising (this pairs the two guards above with the schema surface)."""
    schema = Qwen2_5VLMobileActionSpace.get_tool_schemas()[0]
    assert "key" in tool_schema_parameters(schema)["properties"]["action"]["enum"]


# =============================================================================
# Malformed-but-recoverable provider output
# =============================================================================
#
# Qwen2.5-VL exposes its whole GUI surface as ONE wrapper tool whose ``action``
# argument carries the native action value. Models do not always obey that
# nesting. Both recoverable shapes must convert exactly like the nested shape,
# and active env extras must outrank a colliding native action value.


@pytest.mark.parametrize(
    "name,args",
    [
        ("left_click", {"coordinate": [500, 300]}),
        ("right_click", {"coordinate": [500, 300]}),
        ("double_click", {"coordinate": [500, 300]}),
        ("left_click_drag", {"coordinate": [500, 300]}),
        ("scroll", {"pixels": -300}),
        ("wait", {"time": 2}),
    ],
)
def test_desktop_flat_native_action_value_used_as_tool_name_converts(name, args):
    """The model dropped ``computer_use`` and used the native action VALUE as
    the tool name. The name is native vocabulary (``left_click``), so it is read
    as ``action`` and runs the same dispatch branch — otherwise the env is
    handed a tool it does not have and the turn is wasted."""
    space = Qwen2_5VLDesktopActionSpace()
    flat = space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
    nested = space.convert_tool_calls_from_agent(
        [{"name": "computer_use", "arguments": {"action": name, **args}}]
    )
    assert flat == nested
    assert tool_call_name(flat[0]) == "computer"


def test_desktop_wrong_wrapper_name_with_nested_action_converts():
    """The other malformed-but-recoverable shape: nesting kept, name wrong."""
    space = Qwen2_5VLDesktopActionSpace()
    out = space.convert_tool_calls_from_agent(
        [{"name": "computer", "arguments": {"action": "left_click", "coordinate": [5, 6]}}]
    )
    assert out == [
        make_tool_call("computer", {"actions": [{"action": "click", "coordinate": [5, 6]}]})
    ]


def test_desktop_flat_answer_becomes_response():
    """``answer`` is parse-only vocabulary (the thought-format prompt names it),
    so it must recover flat as well as nested."""
    space = Qwen2_5VLDesktopActionSpace()
    out = space.convert_tool_calls_from_agent([{"name": "answer", "arguments": {"text": "42"}}])
    assert out == [make_tool_call("response", {"text": "42"})]


def test_desktop_non_native_name_stays_a_standalone_tool():
    """The recovery reaches only NATIVE action values. A name outside that
    vocabulary is a real standalone tool and its Lite name IS its wire name."""
    space = Qwen2_5VLDesktopActionSpace()
    out = space.convert_tool_calls_from_agent([{"name": "summarize", "arguments": {"text": "hi"}}])
    assert out == [make_tool_call("summarize", {"text": "hi"})]


def test_desktop_active_extra_tool_outranks_a_colliding_native_action_value():
    """A browsergym ``scroll(delta_y=...)`` is the env's tool, not the native
    ``scroll`` the dispatch would read as a pixel-carrying wheel action."""
    schema = {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"delta_y": {"type": "number"}},
                "required": ["delta_y"],
            },
        },
    }
    space = Qwen2_5VLDesktopActionSpace()
    out = space.convert_tool_calls_from_agent(
        [{"name": "scroll", "arguments": {"delta_y": 100}}],
        active_extra_tool_names={"scroll"},
        active_extra_tool_schemas=[schema],
    )
    assert out == [make_tool_call("scroll", {"delta_y": 100})]


@pytest.mark.parametrize(
    "name,args",
    [
        ("click", {"coordinate": [500, 300]}),
        ("long_press", {"coordinate": [500, 300], "time": 2}),
        ("swipe", {"coordinate": [1, 2], "coordinate2": [3, 4]}),
        ("open", {"text": "Chrome"}),
    ],
)
def test_mobile_flat_native_action_value_used_as_tool_name_converts(name, args):
    space = Qwen2_5VLMobileActionSpace()
    flat = space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
    nested = space.convert_tool_calls_from_agent(
        [{"name": "mobile_use", "arguments": {"action": name, **args}}]
    )
    assert flat == nested


def test_mobile_wrong_wrapper_name_with_nested_action_converts():
    space = Qwen2_5VLMobileActionSpace()
    out = space.convert_tool_calls_from_agent(
        [{"name": "mobile", "arguments": {"action": "click", "coordinate": [5, 6]}}]
    )
    assert tool_call_name(out[0]) == "mobile"
    assert tool_call_arguments(out[0])["actions"] == [
        {"action": "tap", "coordinate": [5, 6], "clicks": 1}
    ]


def test_mobile_flat_answer_becomes_response():
    space = Qwen2_5VLMobileActionSpace()
    out = space.convert_tool_calls_from_agent([{"name": "answer", "arguments": {"text": "42"}}])
    assert out == [make_tool_call("response", {"text": "42"})]


def test_mobile_non_native_name_stays_a_standalone_tool():
    space = Qwen2_5VLMobileActionSpace()
    out = space.convert_tool_calls_from_agent([{"name": "summarize", "arguments": {"text": "hi"}}])
    assert out == [make_tool_call("summarize", {"text": "hi"})]


def test_mobile_active_extra_tool_outranks_a_colliding_native_action_value():
    """A browsergym ``click(bid=...)`` is the env's tool, not the native
    ``click`` the dispatch would read as a coordinate tap."""
    schema = {
        "type": "function",
        "function": {
            "name": "click",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"bid": {"type": "string"}},
                "required": ["bid"],
            },
        },
    }
    space = Qwen2_5VLMobileActionSpace()
    out = space.convert_tool_calls_from_agent(
        [{"name": "click", "arguments": {"bid": "a51"}}],
        active_extra_tool_names={"click"},
        active_extra_tool_schemas=[schema],
    )
    assert out == [make_tool_call("click", {"bid": "a51"})]
