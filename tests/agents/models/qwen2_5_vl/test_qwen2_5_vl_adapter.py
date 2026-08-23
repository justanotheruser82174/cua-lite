"""Qwen2.5-VL adapter tests.

Run:
    uv run pytest tests/agents/models/qwen2_5_vl/test_qwen2_5_vl_adapter.py -v

Pins the ``Thought:`` / ``Action:`` from-agent parse regexes on the
``qwen2_5_vl`` navigation adapter. Its ``convert_message_from_agent`` parse
override (``lite/agents/models/qwen2_5_vl/adapter.py``) carries the SAME greedy
last-``Action:`` anchor and lookahead-``Thought:`` regexes as the qwen3_vl /
qwen3_5 adapters, but the parametrized qwen3_vl test does NOT cover it — so the
identical change would otherwise go untested here.
"""

from __future__ import annotations

import json

import pytest

from lite.agents.models.qwen2_5_vl.adapter import (
    Qwen2_5VLDesktopUseAdapter,
    Qwen2_5VLMobileUseAdapter,
    Qwen2_5VLUseAdapter,
)
from lite.core import LiteCUAMetadata
from lite.core.messages import no_tool_call_final_text
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.action_space import (
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_parameters

_TERMINATE_SCHEMA = make_tool_schema(
    "terminate",
    description="Finish the task.",
    parameters={"type": "object", "properties": {"status": {"type": "string"}}, "required": []},
)

_OPEN_APP_SCHEMA = make_tool_schema(
    "open_app",
    description="Open an app.",
    parameters={"type": "object", "properties": {"app_name": {"type": "string"}}, "required": []},
)
_ASK_USER_SCHEMA = make_tool_schema(
    "ask_user",
    description="Ask the user.",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["action", "question"],
    },
)


QWEN2_5_VL_USE_ADAPTERS = (
    Qwen2_5VLUseAdapter,
    Qwen2_5VLDesktopUseAdapter,
    Qwen2_5VLMobileUseAdapter,
)

NO_TOOL_FINAL_TEXTS = (
    "Done.",
    "The answer is 42.",
)


def test_rescale_coords_leaves_nonfinite_coordinates_raw():
    args = {
        "coordinate": ["1e400", 0],
        "coordinate2": ["10.9", "20.1"],
    }

    Qwen2_5VLDesktopUseAdapter._rescale_coords_in_args(
        args,
        scale=lambda x, y: [x, y],
    )

    assert args["coordinate"] == ["1e400", 0]
    assert args["coordinate2"] == [10, 20]


def test_valid_actions_empty_drops_native_wrapper_and_hides_native_semantic_extra():
    adapter = Qwen2_5VLDesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("desktop", "use"),
            valid_actions=[],
            extra_tool_schemas=[
                make_tool_schema(
                    "response",
                    description="Submit a final answer.",
                    parameters={"type": "object", "properties": {}, "required": []},
                )
            ],
        )
    )

    text = adapter._build_tools_section(image_size=(1280, 720))
    assert '"name": "computer_use"' not in text
    assert '"name": "response"' not in text


def test_parse_multiline_thought_and_nested_action_round_trips():
    """Two regex contracts in one round-trip:

    1. A MULTI-LINE ``Thought:`` body must be captured in full — the
       ``Thought:\\s*(.*?)(?:\\n(?=Action:)|\\Z)`` lookahead must not clip at
       the first newline.
    2. An ``Action:`` substring NESTED in the Thought body must NOT
       short-circuit the action capture — the greedy last-``Action:`` anchor
       (``(?:.*\\n)?Action:\\s*(.*)\\Z``) makes the REAL trailing ``Action:``
       line win.
    """
    adapter = Qwen2_5VLDesktopUseAdapter(enable_inline_reasoning=True)

    # (1) multi-line Thought body — both lines preserved, action intact.
    response = (
        "Thought: line1\n"
        "line2\n"
        "Action: Click the settings menu.\n"
        '<tool_call>\n{"name": "computer_use", "arguments": {"action": '
        '"left_click", "coordinate": [980, 100]}}\n</tool_call>'
    )
    agent_msg = adapter.parse_raw_assistant_response(response)
    lite_msg = adapter.convert_message_from_agent(agent_msg)
    reasoning = next(p["text"] for p in lite_msg["content"] if p["type"] == "inline_reasoning")
    action = next(p["text"] for p in lite_msg["content"] if p["type"] == "action_description")
    assert reasoning == "line1\nline2", (
        "multi-line Thought body must keep both lines (regex must not clip at the first newline)"
    )
    assert action == "Click the settings menu."

    # (2) Action: nested inside the Thought body — the LAST Action: wins.
    response_nested = (
        "Thought: I considered Action: foo\n"
        "Action: real_action.\n"
        '<tool_call>\n{"name": "computer_use", "arguments": {"action": '
        '"left_click", "coordinate": [980, 100]}}\n</tool_call>'
    )
    agent_msg_nested = adapter.parse_raw_assistant_response(response_nested)
    lite_msg_nested = adapter.convert_message_from_agent(agent_msg_nested)
    action_nested = next(
        p["text"] for p in lite_msg_nested["content"] if p["type"] == "action_description"
    )
    assert action_nested == "real_action.", (
        "the LAST Action: line must win, not an Action: nested in Thought"
    )


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["done", "prose"])
@pytest.mark.parametrize("adapter_cls", QWEN2_5_VL_USE_ADAPTERS, ids=lambda c: c.__name__)
def test_no_tool_call_raw_text_stays_plain_text(adapter_cls, final_text):
    """No-tool-call qwen2_5_vl output is a final text turn, not action narration."""
    adapter = adapter_cls()
    msg = adapter.parse_raw_assistant_response(final_text)
    out = adapter.convert_message_from_agent(msg)

    assert not out.get("tool_calls")
    assert out.get("content") == [{"type": "text", "text": final_text}]
    assert no_tool_call_final_text(out) == final_text


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["done", "prose"])
@pytest.mark.parametrize("adapter_cls", QWEN2_5_VL_USE_ADAPTERS, ids=lambda c: c.__name__)
def test_no_tool_call_text_round_trips_through_agent_wire(adapter_cls, final_text):
    """``from_agent(to_agent(text final))`` must preserve qwen2_5_vl text finals."""
    source = {
        "role": "assistant",
        "content": [{"type": "text", "text": final_text}],
    }
    adapter = adapter_cls()

    rendered = adapter.convert_message_to_agent(source)
    out = adapter.convert_message_from_agent(rendered)

    assert not out.get("tool_calls")
    assert out.get("content") == [{"type": "text", "text": final_text}]
    assert no_tool_call_final_text(out) == final_text


def test_desktop_native_terminate_render_does_not_require_matching_schema():
    msg = {
        "role": "assistant",
        "tool_calls": [make_tool_call("terminate", {"status": "success"})],
    }

    rendered = Qwen2_5VLDesktopUseAdapter().convert_message_to_agent(msg)
    assert '"action": "terminate"' in rendered["content"][0]["text"]

    rendered = Qwen2_5VLDesktopUseAdapter(
        metadata=LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=[_TERMINATE_SCHEMA])
    ).convert_message_to_agent(msg)
    assert '"action": "terminate"' in rendered["content"][0]["text"]


def test_mobile_open_app_standalone_schema_controls_passthrough_not_render_availability():
    open_msg = {
        "role": "assistant",
        "tool_calls": [make_tool_call("open_app", {"app_name": "Settings"})],
    }
    terminate_msg = {
        "role": "assistant",
        "tool_calls": [make_tool_call("terminate", {"status": "success"})],
    }

    adapter = Qwen2_5VLMobileUseAdapter()
    open_text = adapter.convert_message_to_agent(open_msg)["content"][0]["text"]
    assert '"name": "mobile_use"' in open_text
    assert '"action": "open"' in open_text
    assert '"text": "Settings"' in open_text
    assert '"name": "open_app"' not in open_text
    terminate_text = adapter.convert_message_to_agent(terminate_msg)["content"][0]["text"]
    assert '"action": "terminate"' in terminate_text

    adapter = Qwen2_5VLMobileUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("mobile", "use"), extra_tool_schemas=[_OPEN_APP_SCHEMA, _TERMINATE_SCHEMA]
        )
    )
    tools_section = adapter._build_tools_section(image_size=(1080, 2408))
    assert '"name": "open_app"' in tools_section

    open_text = adapter.convert_message_to_agent(open_msg)["content"][0]["text"]
    assert '"name": "open_app"' in open_text
    assert '"app_name": "Settings"' in open_text
    assert '"action": "open"' not in open_text
    terminate_text = adapter.convert_message_to_agent(terminate_msg)["content"][0]["text"]
    assert '"action": "terminate"' in terminate_text


def test_mobile_tools_section_descriptions_follow_active_native_semantics():
    import json
    import re

    adapter = Qwen2_5VLMobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), valid_actions=["tap"])
    )

    tools_section = adapter._build_tools_section(image_size=(1080, 2408))
    match = re.search(r"<tools>\n(.*?)\n</tools>", tools_section, re.DOTALL)
    assert match is not None
    schema = json.loads(match.group(1).splitlines()[0])
    properties = tool_schema_parameters(schema)["properties"]

    assert properties["action"]["enum"] == ["click"]
    assert properties["text"]["description"] == ""
    assert properties["status"]["description"] == "The status of the task."
    rendered_schema = json.dumps(schema)
    assert "action=open" not in rendered_schema
    assert "action=terminate" not in rendered_schema


def test_mobile_native_open_from_agent_reaches_env_owned_open_app():
    adapter = Qwen2_5VLMobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), extra_tool_schemas=[])
    )
    raw = (
        "Action: Open Settings.\n"
        '<tool_call>\n{"name": "mobile_use", "arguments": '
        '{"action": "open", "text": "Settings"}}\n</tool_call>'
    )

    lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert pop_model_output_error(lite_msg.copy()) is None
    assert lite_msg.get("tool_calls") == [make_tool_call("open_app", {"app_name": "Settings"})]


@pytest.mark.parametrize(
    "adapter,tool_name,arguments,action_batch_tool_name",
    [
        pytest.param(
            Qwen2_5VLDesktopUseAdapter(),
            "computer_use",
            {"action": "not_a_real_action", "coordinate": [500, 300]},
            "computer",
            id="desktop-unknown-action",
        ),
        pytest.param(
            Qwen2_5VLDesktopUseAdapter(),
            "computer_use",
            {"action": "report_infeasible", "reason": "blocked"},
            "computer",
            id="desktop-embedded-extra",
        ),
        pytest.param(
            Qwen2_5VLMobileUseAdapter(),
            "mobile_use",
            {"action": "not_a_real_action", "coordinate": [500, 300]},
            "mobile",
            id="mobile-unknown-action",
        ),
        pytest.param(
            Qwen2_5VLMobileUseAdapter(),
            "mobile_use",
            {"action": "open_app", "app_name": "Settings"},
            "mobile",
            id="mobile-embedded-extra",
        ),
    ],
)
def test_unknown_native_wrapper_action_becomes_invalid_action_batch(
    adapter,
    tool_name,
    arguments,
    action_batch_tool_name,
):
    """A wrapper action this family cannot express stays env-visible.

    The model's own action value becomes the batch child name, so env ingress
    rejects it and pairs a result back to the model. Nothing is dropped, so the
    adapter's empty-conversion ``model_output_error`` never fires.
    """
    raw = (
        "Action: Use the bad action.\n"
        f'<tool_call>\n{{"name": "{tool_name}", "arguments": '
        f"{json.dumps(arguments)}}}\n</tool_call>"
    )

    lite_msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert pop_model_output_error(lite_msg.copy()) is None
    [tc] = lite_msg["tool_calls"]
    assert tool_call_name(tc) == action_batch_tool_name
    assert tool_call_arguments(tc) == {"actions": [arguments]}
    children, error = validate_lite_action_batch_structure(
        action_batch_tool_name,
        tool_call_arguments(tc),
    )
    assert len(children) == 1
    error = lite_action_batch_child_name_errors(action_batch_tool_name, children).get(0)
    assert error is not None
    assert error.child_action_name == arguments["action"]


def test_from_agent_preserves_declared_extra_tool_order():
    adapter = Qwen2_5VLMobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), extra_tool_schemas=[_ASK_USER_SCHEMA])
    )
    msg = {
        "role": "assistant",
        "tool_calls": [
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [100, 100]}},
            {"name": "ask_user", "arguments": {"action": "confirm", "question": "Continue?"}},
            {"name": "mobile_use", "arguments": {"action": "type", "text": "ok"}},
        ],
    }

    lite_msg = adapter.convert_message_from_agent(msg)

    assert [tool_call_name(tc) for tc in lite_msg["tool_calls"]] == ["mobile", "ask_user", "mobile"]
    assert tool_call_arguments(lite_msg["tool_calls"][0])["actions"] == [
        {"action": "tap", "coordinate": [100, 100], "clicks": 1}
    ]
    assert tool_call_arguments(lite_msg["tool_calls"][1]) == {
        "action": "confirm",
        "question": "Continue?",
    }
    assert tool_call_arguments(lite_msg["tool_calls"][2])["actions"] == [
        {"action": "type", "text": "ok"}
    ]


def test_desktop_from_agent_preserves_declared_extra_tool_order():
    adapter = Qwen2_5VLDesktopUseAdapter(
        metadata=LiteCUAMetadata(dims=("desktop", "use"), extra_tool_schemas=[_ASK_USER_SCHEMA])
    )
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "name": "computer_use",
                "arguments": {"action": "left_click", "coordinate": [100, 100]},
            },
            {"name": "ask_user", "arguments": {"action": "confirm", "question": "Continue?"}},
            {"name": "computer_use", "arguments": {"action": "type", "text": "ok"}},
        ],
    }

    lite_msg = adapter.convert_message_from_agent(msg)

    assert [tool_call_name(tc) for tc in lite_msg["tool_calls"]] == [
        "computer",
        "ask_user",
        "computer",
    ]
    assert tool_call_arguments(lite_msg["tool_calls"][0])["actions"] == [
        {"action": "click", "coordinate": [100, 100]}
    ]
    assert tool_call_arguments(lite_msg["tool_calls"][1]) == {
        "action": "confirm",
        "question": "Continue?",
    }
    assert tool_call_arguments(lite_msg["tool_calls"][2])["actions"] == [
        {"action": "type", "text": "ok"}
    ]


def test_to_agent_preserves_declared_extra_tool_with_action_argument():
    adapter = Qwen2_5VLMobileUseAdapter(
        metadata=LiteCUAMetadata(dims=("mobile", "use"), extra_tool_schemas=[_ASK_USER_SCHEMA])
    )
    msg = {
        "role": "assistant",
        "tool_calls": [
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [100, 100]}]},
            ),
            make_tool_call(
                "ask_user",
                {"action": "confirm", "question": "Continue?"},
            ),
        ],
    }

    text = adapter.convert_message_to_agent(msg)["content"][0]["text"]

    assert '"name": "mobile_use"' in text
    assert '"name": "ask_user"' in text
    assert '"action": "confirm"' in text


def test_tools_section_byte_policy_is_one_ascii_escaped_json_object_per_line():
    """Qwen2.5-VL owns its ``<tools>`` byte policy inline in its own adapter.

    There is no shared root Qwen prompt helper: each Qwen family serialises at
    its own ``_build_tools_section``. The policy is one JSON object per line
    with ``json.dumps`` defaults, so non-ASCII escapes as ``\\uXXXX``. Fara
    reuses the same nested schema shape but deliberately dumps its own bytes
    with ``ensure_ascii=False``.
    """
    import re

    adapter = Qwen2_5VLMobileUseAdapter(
        metadata=LiteCUAMetadata(
            dims=("mobile", "use"),
            extra_tool_schemas=[
                make_tool_schema(
                    "open_app",
                    description="Résumé — ✓",
                    parameters={
                        "type": "object",
                        "properties": {"app_name": {"type": "string"}},
                        "required": ["app_name"],
                    },
                )
            ],
        )
    )

    section = adapter._build_tools_section(image_size=(1080, 2408))
    match = re.search(r"<tools>\n(.*?)\n</tools>", section, re.DOTALL)
    assert match is not None
    lines = match.group(1).splitlines()
    # One JSON object per line — one line per advertised tool.
    assert [json.loads(line)["function"]["name"] for line in lines] == [
        "mobile_use",
        "open_app",
    ]
    # Default ASCII escaping: the rendered prompt carries no raw non-ASCII.
    assert "\\u2713" in lines[1]
    assert "✓" not in section
