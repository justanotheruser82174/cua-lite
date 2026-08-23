"""F8 — adapter parse↔render round-trip + batching grouper.

Canonical multi-action GUI turns are batched Lite tool calls:

* desktop: ``{"type":"function","function":{"name":"computer","arguments":{"actions":[...]}}}``
* mobile: ``{"type":"function","function":{"name":"mobile","arguments":{"actions":[...]}}}``

Raw/native model outputs may expose multiple adjacent GUI action emissions in
one assistant turn (for example ``computer_use`` / ``mobile_use`` calls, or text
wire action lines). Those boundaries must parse to canonical batches and render
back without moving standalone extra tools across action-batch calls.

Hermetic: registry + action-space conversions are pure Python (no model
download, no network).

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/adapter/test_adapter_batching_roundtrip.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import (
    stamp_message_tool_call_ids,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

register_all()


# =============================================================================
# Shared helpers
# =============================================================================


def _tc(name: str, **arguments: Any) -> dict[str, Any]:
    """A model/provider wire tool_call dict."""
    return {"name": name, "arguments": arguments}


def _name(tool_call: dict[str, Any]) -> str:
    if tool_call.get("type") == "function":
        return tool_call_name(tool_call)
    return tool_call["name"]


def _args(tool_call: dict[str, Any]) -> dict[str, Any]:
    if tool_call.get("type") == "function":
        return tool_call_arguments(tool_call)
    return tool_call["arguments"]


def _id(tool_call: dict[str, Any]) -> str | None:
    if tool_call.get("type") == "function":
        return tool_call_id(tool_call)
    return tool_call.get("call_id")


_BASH_SCHEMA = make_tool_schema(
    "bash",
    description="Run a bash command.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)
_ASK_USER_SCHEMA = make_tool_schema(
    "ask_user",
    description="Ask the user for clarification.",
    parameters={
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
)
_RESPONSE_SCHEMA = LiteFinishToolSet.get_tool_schema("response")
_TERMINATE_SCHEMA_FOR_ACTIVE = LiteFinishToolSet.get_tool_schema("terminate")
_GOTO_SCHEMA = make_tool_schema(
    "goto",
    description="Navigate to a URL.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
)
_OPEN_APP_SCHEMA = make_tool_schema(
    "open_app",
    description="Open an app.",
    parameters={
        "type": "object",
        "properties": {"app_name": {"type": "string"}},
        "required": ["app_name"],
    },
)


def _computer_use(action: str, **arguments: Any) -> dict[str, Any]:
    """A qwen/fara-style ``computer_use`` wire tool_call."""
    return _tc("computer_use", action=action, **arguments)


def _mobile_use(action: str, **arguments: Any) -> dict[str, Any]:
    """A qwen-style ``mobile_use`` wire tool_call (mobile platform)."""
    return _tc("mobile_use", action=action, **arguments)


def _actions(tool_calls: list[dict[str, Any]], name: str) -> list[Any]:
    """The ``actions`` list of the (single expected) action-batch call named
    ``name`` (``computer`` or ``mobile``)."""
    call = next(tc for tc in tool_calls if _name(tc) == name)
    return _args(call)["actions"]


def _names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [_name(tc) for tc in tool_calls]


def _action_name(action: Any) -> str | None:
    """Name of one canonical entry inside ``computer/mobile.arguments.actions``.

    The target inner shape is strict: ``{"action": <action>, ...}``. Do not
    accept a nested LiteToolCall/function envelope here, or non-canonical new data
    can pass the batching tests.
    """
    if not isinstance(action, dict):
        return None
    return action.get("action")


def _computer_actions(tool_calls: list[dict[str, Any]]) -> list[Any]:
    """The ``actions`` list of the (single expected) batched ``computer`` call."""
    comp = next(tc for tc in tool_calls if _name(tc) == "computer")
    return _args(comp)["actions"]


def _assert_single_expected_action(
    tcs: list[dict[str, Any]], expected_name: str, expected_args: dict[str, Any]
) -> str:
    assert len(tcs) == 1, f"{expected_name}: expected one canonical GUI call, got {_names(tcs)}"
    tc = tcs[0]
    assert _name(tc) in {"computer", "mobile"}
    actions = _args(tc)["actions"]
    assert len(actions) == 1
    assert actions[0] == {"action": expected_name, **expected_args}
    return _name(tc)


# =============================================================================
# Single-action parse → render → parse survives per-action
# =============================================================================


def _adapter_roundtrip(
    adapter: Any,
    lite_msg: dict[str, Any],
    *,
    to_kwargs: dict[str, Any],
    from_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """render (``convert_message_to_agent``) then re-parse back to canonical.

    Auto-detects the provider wire format: structured families keep ``tool_calls`` on the
    agent message (re-parse via ``convert_message_from_agent``); text-wire
    families fold the action into an assistant text blob (re-parse via
    ``parse_raw_assistant_response`` → ``convert_message_from_agent``)."""
    agent = adapter.convert_message_to_agent(lite_msg, **to_kwargs)
    if agent.get("tool_calls"):
        back = adapter.convert_message_from_agent(agent, **from_kwargs)
    else:
        text = next(
            (c["text"] for c in agent.get("content", []) if c.get("type") == "text"),
            "",
        )
        back = adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(text), **from_kwargs
        )
    return back.get("tool_calls", [])


def _metadata_for_adapter_key(
    adapter_key: str, extra_tool_schemas: list[dict[str, Any]]
) -> LiteCUAMetadata:
    platform = "mobile" if "@mobile" in adapter_key else "desktop"
    return LiteCUAMetadata(dims=(platform, "use"), extra_tool_schemas=extra_tool_schemas)


def _rendered_agent_tool_calls(
    adapter_key: str,
    lite_msg: dict[str, Any],
    *,
    extra_tool_schemas: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    adapter = AgentAdapterRegistry.get(
        adapter_key,
        metadata=_metadata_for_adapter_key(adapter_key, extra_tool_schemas or []),
    )
    agent = adapter.convert_message_to_agent(lite_msg)
    if agent.get("tool_calls"):
        return agent["tool_calls"]
    text = next(
        (c["text"] for c in agent.get("content", []) if c.get("type") == "text"),
        "",
    )
    return adapter.parse_raw_assistant_response(text).get("tool_calls", [])


def _qwen35_xml_tool_call(name: str, **arguments: Any) -> str:
    params = "\n".join(
        f"<parameter={key}>\n{value}\n</parameter>" for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>"


class _AdapterCase:
    """A raw-grammar adapter family (parse via ``parse_raw_assistant_response``)."""

    def __init__(
        self,
        key: str,
        raw: str,
        expected_name: str,
        expected_args: dict[str, Any],
        *,
        to_kwargs: dict[str, Any] | None = None,
        from_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.key = key
        self.raw = raw
        self.expected_name = expected_name
        self.expected_args = expected_args
        self.to_kwargs = to_kwargs or {}
        self.from_kwargs = from_kwargs or {}

    def lite_msg(self) -> dict[str, Any]:
        adapter = AgentAdapterRegistry.get(self.key)
        return adapter.convert_message_from_agent(
            adapter.parse_raw_assistant_response(self.raw), **self.from_kwargs
        )

    def roundtrip(self, lite_msg: dict[str, Any]) -> list[dict[str, Any]]:
        adapter = AgentAdapterRegistry.get(self.key)
        return _adapter_roundtrip(
            adapter, lite_msg, to_kwargs=self.to_kwargs, from_kwargs=self.from_kwargs
        )


class _LiteCase(_AdapterCase):
    """``lite`` is the canonical mid-format — it has no raw wire grammar, so the
    canonical assistant message is constructed directly."""

    def __init__(self) -> None:
        super().__init__(
            key="lite@desktop@use",
            raw="",
            expected_name="click",
            expected_args={"coordinate": [500, 300]},
        )

    def lite_msg(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Click."}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                )
            ],
        }


class _SpaceCase(_AdapterCase):
    """Native-API *teacher* action spaces (no render_step adapter).

    The "raw" is a single native action dict; parse/render happen at the
    action-space level (``convert_tool_calls_from_agent`` / ``_to_agent``).
    """

    def __init__(
        self,
        space_key: str,
        native: dict[str, Any],
        expected_name: str,
        expected_args: dict[str, Any],
        resolution: tuple[int, int] = (1280, 768),
    ) -> None:
        self.space_key = space_key
        self.native = native
        self.expected_name = expected_name
        self.expected_args = expected_args
        self.resolution = resolution

    def lite_msg(self) -> dict[str, Any]:
        space = ActionSpaceRegistry.get(self.space_key)
        tcs = space.convert_tool_calls_from_agent([self.native], resolution=self.resolution)
        return {"role": "assistant", "tool_calls": tcs}

    def roundtrip(self, lite_msg: dict[str, Any]) -> list[dict[str, Any]]:
        space = ActionSpaceRegistry.get(self.space_key)
        wire = space.convert_tool_calls_to_agent(lite_msg["tool_calls"], resolution=self.resolution)
        return space.convert_tool_calls_from_agent(wire, resolution=self.resolution)


# Real per-family raw grammars (lifted from each family's own adapter test).
_GREEN_CASES: list[_AdapterCase] = [
    _AdapterCase(
        "qwen3_vl@desktop@use",
        "Action: Click.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click", '
        '"coordinate": [500, 300]}}\n'
        "</tool_call>",
        expected_name="click",
        expected_args={"coordinate": [500, 300]},
    ),
    _AdapterCase(
        "qwen3_5@desktop@use",
        "Action: Click.\n"
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=action>\nleft_click\n</parameter>\n"
        "<parameter=coordinate>\n[491, 91]\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        expected_name="click",
        expected_args={"coordinate": [491, 91]},
    ),
    _AdapterCase(
        "ui_tars_15_v1@desktop@use",
        "Thought: Click the button.\nAction: click(start_box='(500,300)')",
        expected_name="click",
        expected_args={"coordinate": [500, 300]},
    ),
    _SpaceCase(
        "claude@desktop",
        {"action": "type", "text": "hello"},
        expected_name="type",
        expected_args={"text": "hello"},
    ),
    _LiteCase(),
]

_GREEN_IDS = [
    c.key.split("@")[0] if not isinstance(c, _SpaceCase) else c.space_key.split("@")[0]
    for c in _GREEN_CASES
]


@pytest.mark.parametrize("case", _GREEN_CASES, ids=_GREEN_IDS)
def test_parse_render_roundtrip_single_action(case: _AdapterCase) -> None:
    """A single raw GUI action round-trips without drifting its action.

    Adapter/action-space parsers return the canonical length-1
    ``computer``/``mobile`` wrapper for GUI actions. Multi-action qwen cases are
    covered below.
    """
    lite_msg = case.lite_msg()
    tcs = lite_msg["tool_calls"]

    shape = _assert_single_expected_action(tcs, case.expected_name, case.expected_args)

    # Canonical round-trip identity: render then re-parse recovers the same action
    # in the same canonical shape.
    rt = case.roundtrip(lite_msg)
    assert _assert_single_expected_action(rt, case.expected_name, case.expected_args) == shape


# =============================================================================
# GREEN: real batch sources group / unwrap losslessly
# =============================================================================


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
def test_batched_computer_group_unwrap_lossless(adapter_key: str) -> None:
    """A multi-action GUI turn (click + type) groups into ONE canonical
    ``computer{actions:[click, type]}`` on parse, and unwraps back to the
    identical per-action wire sequence on render — a LOSSLESS list wrap."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    agent_msg = {
        "role": "assistant",
        "tool_calls": [
            _computer_use("left_click", coordinate=[500, 300]),
            _computer_use("type", text="hi"),
        ],
    }
    lite = adapter.convert_message_from_agent(agent_msg)
    tcs = lite["tool_calls"]

    # Grouped shape: one batched ``computer`` carrying both actions in order.
    assert len(tcs) == 1
    assert _name(tcs[0]) == "computer"
    assert [_action_name(a) for a in _computer_actions(tcs)] == ["click", "type"]

    # Render then parse returns the same canonical batch. qwen3-vl renders
    # structured tool_calls; qwen3.5 renders XML text, so assert semantic
    # round-trip rather than one intermediate wire container.
    rt = _adapter_roundtrip(adapter, lite, to_kwargs={}, from_kwargs={})
    assert len(rt) == 1
    assert _name(rt[0]) == "computer"
    assert [_action_name(a) for a in _computer_actions(rt)] == ["click", "type"]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@mobile@use", "qwen3_5@mobile@use"])
def test_batched_mobile_group_unwrap_lossless(adapter_key: str) -> None:
    """Mobile batches IDENTICALLY to desktop (owner decision, section 1/section 3.4): a
    multi-action mobile turn (click + type) groups into ONE canonical
    ``mobile{actions:[click, type]}`` on parse and unwraps back to the two
    per-``mobile_use`` wire calls on render — symmetric with the computer path.
    Also guards the mobile wrapper-name routing: the grouped call MUST be named
    ``mobile`` (NOT ``computer``)."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    agent_msg = {
        "role": "assistant",
        "tool_calls": [
            _mobile_use("click", coordinate=[500, 300]),
            _mobile_use("type", text="hi"),
        ],
    }
    lite = adapter.convert_message_from_agent(agent_msg)
    tcs = lite["tool_calls"]

    # Grouped shape: ONE batched ``mobile`` (NOT ``computer``).
    assert len(tcs) == 1
    assert _name(tcs[0]) == "mobile"
    assert [_action_name(a) for a in _actions(tcs, "mobile")] == ["tap", "type"]

    # Render then parse returns the same canonical batch. qwen3-vl renders
    # structured tool_calls; qwen3.5 renders XML text.
    rt = _adapter_roundtrip(adapter, lite, to_kwargs={}, from_kwargs={})
    assert len(rt) == 1
    assert _name(rt[0]) == "mobile"
    assert [_action_name(a) for a in _actions(rt, "mobile")] == ["tap", "type"]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
def test_canonical_batched_computer_renders_as_native_qwen_wrappers(adapter_key: str) -> None:
    rendered = _rendered_agent_tool_calls(
        adapter_key,
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Do it."}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "click", "coordinate": [100, 100]},
                            {"action": "type", "text": "hi"},
                        ]
                    },
                )
            ],
        },
    )

    assert _names(rendered) == ["computer_use", "computer_use"]
    assert [tc["arguments"]["action"] for tc in rendered] == ["left_click", "type"]
    assert rendered[0]["arguments"]["coordinate"] == [100, 100]
    assert rendered[1]["arguments"]["text"] == "hi"


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@mobile@use", "qwen3_5@mobile@use"])
def test_canonical_batched_mobile_renders_as_native_qwen_wrappers(adapter_key: str) -> None:
    rendered = _rendered_agent_tool_calls(
        adapter_key,
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Do it."}],
            "tool_calls": [
                make_tool_call(
                    "mobile",
                    {
                        "actions": [
                            {"action": "tap", "coordinate": [100, 100]},
                            {"action": "type", "text": "hi"},
                        ]
                    },
                )
            ],
        },
    )

    assert _names(rendered) == ["mobile_use", "mobile_use"]
    assert [tc["arguments"]["action"] for tc in rendered] == ["click", "type"]
    assert rendered[0]["arguments"]["coordinate"] == [100, 100]
    assert rendered[1]["arguments"]["text"] == "hi"


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
def test_single_qwen_action_roundtrips_as_length1_wrapper(adapter_key: str) -> None:
    """The sacred case: one qwen GUI action stays one canonical wrapper."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    agent_msg = {
        "role": "assistant",
        "tool_calls": [_computer_use("left_click", coordinate=[500, 300])],
    }
    lite = adapter.convert_message_from_agent(agent_msg)
    tcs = lite["tool_calls"]

    assert len(tcs) == 1
    assert tcs[0] == make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [500, 300]}]},
    )

    # Render then parse returns the same canonical action.
    rt = _adapter_roundtrip(adapter, lite, to_kwargs={}, from_kwargs={})
    assert len(rt) == 1
    assert rt[0] == tcs[0]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@mobile@use", "qwen3_5@mobile@use"])
def test_single_qwen_mobile_action_roundtrips_as_length1_wrapper(adapter_key: str) -> None:
    """Mobile single actions get the same length-1 wrapper treatment."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    agent_msg = {
        "role": "assistant",
        "tool_calls": [_mobile_use("click", coordinate=[500, 300])],
    }
    lite = adapter.convert_message_from_agent(agent_msg)
    tcs = lite["tool_calls"]

    assert len(tcs) == 1
    assert tcs[0] == make_tool_call(
        "mobile",
        {"actions": [{"action": "tap", "coordinate": [500, 300], "clicks": 1}]},
    )

    rt = _adapter_roundtrip(adapter, lite, to_kwargs={}, from_kwargs={})
    assert len(rt) == 1
    assert rt[0] == tcs[0]


def _qwen_active_extra_tool_schemas(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for call in tool_calls:
        name = call.get("name")
        arguments = call.get("arguments") or {}
        action = arguments.get("action") if isinstance(arguments, dict) else None
        if name == "computer_use" and action == "answer":
            schemas.append(_RESPONSE_SCHEMA)
        elif name == "computer_use" and action == "terminate":
            schemas.append(_TERMINATE_SCHEMA_FOR_ACTIVE)
        elif name == "bash":
            schemas.append(_BASH_SCHEMA)
        elif name == "ask_user":
            schemas.append(_ASK_USER_SCHEMA)
        elif name == "goto":
            schemas.append(_GOTO_SCHEMA)
        elif name == "open_app":
            schemas.append(_OPEN_APP_SCHEMA)
    return schemas


def _qwen_from_agent(adapter_key: str, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapter = AgentAdapterRegistry.get(
        adapter_key,
        metadata=_metadata_for_adapter_key(
            adapter_key,
            _qwen_active_extra_tool_schemas(tool_calls),
        ),
    )
    return adapter.convert_message_from_agent({"role": "assistant", "tool_calls": tool_calls})[
        "tool_calls"
    ]


def test_adapter_rejects_provider_call_id_on_action_space_call() -> None:
    adapter = AgentAdapterRegistry.get("lite@desktop@use")
    tool_call = _tc("click", coordinate=[100, 100])
    tool_call["call_id"] = "provider-gui"

    with pytest.raises(ValueError, match="non-bare-call keys \\['call_id'\\]"):
        adapter._route_agent_tool_calls_to_lite([tool_call])


def test_adapter_preserves_provider_call_id_for_active_extra() -> None:
    adapter = AgentAdapterRegistry.get(
        "qwen3_vl@desktop@use",
        metadata=_metadata_for_adapter_key("qwen3_vl@desktop@use", [_BASH_SCHEMA]),
    )
    tool_call = _tc("bash", command="pwd")
    tool_call["call_id"] = "provider-bash"
    msg = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [tool_call],
        }
    )

    assert _names(msg["tool_calls"]) == ["bash"]
    assert _id(msg["tool_calls"][0]) == "provider-bash"


# Category-aware segmentation cases: qwen can emit multiple ``computer_use`` calls
# in one turn, and ``answer`` must stay standalone rather than being swallowed by
# the action-batch call.
_GROUPER_CASES: list[
    tuple[str, Callable[..., list[dict[str, Any]]], list[dict[str, Any]], str, str, list[str]]
] = [
    (
        "qwen3_vl_answer_response",
        lambda tool_calls: _qwen_from_agent("qwen3_vl@desktop@use", tool_calls),
        [
            _computer_use("left_click", coordinate=[500, 300]),
            _computer_use("type", text="hi"),
            _computer_use("answer", text="42"),
        ],
        "response",
        "computer",
        ["click", "type"],
    ),
    (
        "qwen3_5_answer_response",
        lambda tool_calls: _qwen_from_agent("qwen3_5@desktop@use", tool_calls),
        [
            _computer_use("left_click", coordinate=[500, 300]),
            _computer_use("type", text="hi"),
            _computer_use("answer", text="42"),
        ],
        "response",
        "computer",
        ["click", "type"],
    ),
    (
        "qwen3_vl_terminate",
        lambda tool_calls: _qwen_from_agent("qwen3_vl@desktop@use", tool_calls),
        [
            _computer_use("left_click", coordinate=[500, 300]),
            _computer_use("type", text="hi"),
            _computer_use("terminate", status="failure"),
        ],
        "terminate",
        "computer",
        ["click", "type"],
    ),
    (
        "qwen3_5_bash_extra",
        lambda tool_calls: _qwen_from_agent("qwen3_5@desktop@use", tool_calls),
        [
            _computer_use("left_click", coordinate=[500, 300]),
            _computer_use("type", text="hi"),
            _tc("bash", command="pwd"),
        ],
        "bash",
        "computer",
        ["click", "type"],
    ),
    (
        "qwen3_vl_nav_extra",
        lambda tool_calls: _qwen_from_agent("qwen3_vl@desktop@use", tool_calls),
        [
            _computer_use("left_click", coordinate=[500, 300]),
            _computer_use("type", text="hi"),
            _tc("goto", url="https://example.com"),
        ],
        "goto",
        "computer",
        ["click", "type"],
    ),
    (
        "qwen3_vl_mobile_open_app_extra",
        lambda tool_calls: _qwen_from_agent("qwen3_vl@mobile@use", tool_calls),
        [
            _mobile_use("click", coordinate=[500, 300]),
            _mobile_use("type", text="hi"),
            _tc("open_app", app_name="Chrome"),
        ],
        "open_app",
        "mobile",
        ["tap", "type"],
    ),
    (
        "qwen3_5_mobile_ask_user_extra",
        lambda tool_calls: _qwen_from_agent("qwen3_5@mobile@use", tool_calls),
        [
            _mobile_use("click", coordinate=[500, 300]),
            _mobile_use("type", text="hi"),
            _tc("ask_user", question="Continue?"),
        ],
        "ask_user",
        "mobile",
        ["tap", "type"],
    ),
]


@pytest.mark.parametrize(
    "from_agent,wire,standalone,action_batch_tool,expected_actions",
    [(c[1], c[2], c[3], c[4], c[5]) for c in _GROUPER_CASES],
    ids=[c[0] for c in _GROUPER_CASES],
)
def test_grouper_does_not_swallow_standalone_tools(
    from_agent: Callable[..., list[dict[str, Any]]],
    wire: list[dict[str, Any]],
    standalone: str,
    action_batch_tool: str,
    expected_actions: list[str],
) -> None:
    """A batched turn ``[computer(click), <standalone>]`` must keep the
    standalone canonical tool (``goto`` / ``back`` / ``response``) OUT of the
    ``computer.actions`` action-batch call. The grouper segments by canonical
    name-category, not by wire wrapper."""
    tcs = from_agent(wire)
    names = _names(tcs)

    # The GUI click is batched into a ``computer`` action-batch call...
    assert action_batch_tool in names
    batch = [_action_name(a) for a in _actions(tcs, action_batch_tool)]
    assert batch == expected_actions

    # ...but the standalone stays a first-class top-level tool, NOT swallowed.
    assert standalone in names
    assert standalone not in batch


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
def test_qwen_grouper_does_not_merge_across_standalone_extra(adapter_key: str) -> None:
    tcs = _qwen_from_agent(
        adapter_key,
        [
            _computer_use("left_click", coordinate=[100, 100]),
            _tc("bash", command="pwd"),
            _computer_use("type", text="after"),
        ],
    )

    assert _names(tcs) == ["computer", "bash", "computer"]
    assert _args(tcs[0])["actions"] == [{"action": "click", "coordinate": [100, 100]}]
    assert _args(tcs[1]) == {"command": "pwd"}
    assert _args(tcs[2])["actions"] == [{"action": "type", "text": "after"}]

    rendered = _rendered_agent_tool_calls(
        adapter_key, {"role": "assistant", "tool_calls": tcs}, extra_tool_schemas=[_BASH_SCHEMA]
    )
    assert _names(rendered) == ["computer_use", "bash", "computer_use"]
    assert [tc["arguments"].get("action") for tc in rendered] == ["left_click", None, "type"]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
def test_qwen_grouper_splits_batch_extra_batch_exactly(adapter_key: str) -> None:
    tcs = _qwen_from_agent(
        adapter_key,
        [
            _computer_use("left_click", coordinate=[100, 100]),
            _computer_use("type", text="before"),
            _tc("bash", command="pwd"),
            _computer_use("left_click", coordinate=[200, 200]),
            _computer_use("type", text="after"),
        ],
    )

    assert _names(tcs) == ["computer", "bash", "computer"]
    assert [_action_name(a) for a in _args(tcs[0])["actions"]] == ["click", "type"]
    assert _args(tcs[1]) == {"command": "pwd"}
    assert [_action_name(a) for a in _args(tcs[2])["actions"]] == ["click", "type"]


@pytest.mark.parametrize(
    "adapter_key,raw",
    [
        (
            "qwen3_vl@desktop@use",
            "Action: Do it.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [100, 100]}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "type", "text": "before"}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "bash", "arguments": {"command": "pwd"}}\n'
            "</tool_call>",
        ),
        (
            "qwen3_5@desktop@use",
            "Action: Do it.\n"
            "<tool_call>\n"
            "<function=computer_use>\n"
            "<parameter=action>\nleft_click\n</parameter>\n"
            "<parameter=coordinate>\n[100, 100]\n</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=computer_use>\n"
            "<parameter=action>\ntype\n</parameter>\n"
            "<parameter=text>\nbefore\n</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>\npwd\n</parameter>\n"
            "</function>\n"
            "</tool_call>",
        ),
    ],
)
def test_qwen_raw_parser_groups_action_run_before_standalone_extra(
    adapter_key: str, raw: str
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert _names(msg["tool_calls"]) == ["computer", "bash"]
    assert [_action_name(a) for a in _args(msg["tool_calls"][0])["actions"]] == ["click", "type"]
    assert _args(msg["tool_calls"][1]) == {"command": "pwd"}


@pytest.mark.parametrize(
    "adapter_key,raw",
    [
        (
            "qwen3_vl@desktop@use",
            "Action: Do it.\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "left_click", '
            '"coordinate": [100, 100]}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "bash", "arguments": {"command": "pwd"}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "computer_use", "arguments": {"action": "type", "text": "after"}}\n'
            "</tool_call>",
        ),
        (
            "qwen3_5@desktop@use",
            "Action: Do it.\n"
            "<tool_call>\n"
            "<function=computer_use>\n"
            "<parameter=action>\nleft_click\n</parameter>\n"
            "<parameter=coordinate>\n[100, 100]\n</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>\npwd\n</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=computer_use>\n"
            "<parameter=action>\ntype\n</parameter>\n"
            "<parameter=text>\nafter\n</parameter>\n"
            "</function>\n"
            "</tool_call>",
        ),
    ],
)
def test_qwen_raw_parser_splits_single_action_runs_around_standalone_extra(
    adapter_key: str, raw: str
) -> None:
    adapter = AgentAdapterRegistry.get(
        adapter_key, metadata=_metadata_for_adapter_key(adapter_key, [_BASH_SCHEMA])
    )
    msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert _names(msg["tool_calls"]) == ["computer", "bash", "computer"]
    assert _args(msg["tool_calls"][0])["actions"] == [{"action": "click", "coordinate": [100, 100]}]
    assert _args(msg["tool_calls"][1]) == {"command": "pwd"}
    assert _args(msg["tool_calls"][2])["actions"] == [{"action": "type", "text": "after"}]

    rt = _adapter_roundtrip(adapter, msg, to_kwargs={}, from_kwargs={})
    assert _names(rt) == ["computer", "bash", "computer"]
    assert _args(rt[0])["actions"] == [{"action": "click", "coordinate": [100, 100]}]
    assert _args(rt[1]) == {"command": "pwd"}
    assert _args(rt[2])["actions"] == [{"action": "type", "text": "after"}]


@pytest.mark.parametrize(
    "adapter_key,batch_name,raw,expected_actions,expected_routed",
    [
        (
            "qwen3_5@desktop@use",
            "computer",
            "Action: Click and type.\n"
            + _qwen35_xml_tool_call(
                "computer_use",
                action="left_click",
                coordinate=[321, 654],
            )
            + "\n"
            + _qwen35_xml_tool_call("computer_use", action="type", text="done"),
            [
                {"action": "click", "coordinate": [321, 654]},
                {"action": "type", "text": "done"},
            ],
            [
                (
                    {"name": "click", "arguments": {"coordinate": [321, 654]}},
                    "call_0000",
                ),
                ({"name": "type", "arguments": {"text": "done"}}, "call_0000"),
            ],
        ),
        (
            "qwen3_5@mobile@use",
            "mobile",
            "Action: Tap and type.\n"
            + _qwen35_xml_tool_call(
                "mobile_use",
                action="click",
                coordinate=[123, 456],
            )
            + "\n"
            + _qwen35_xml_tool_call("mobile_use", action="type", text="done"),
            [
                {"action": "tap", "coordinate": [123, 456], "clicks": 1},
                {"action": "type", "text": "done"},
            ],
            [
                (
                    {
                        "name": "tap",
                        "arguments": {"coordinate": [123, 456], "clicks": 1},
                    },
                    "call_0000",
                ),
                ({"name": "type", "arguments": {"text": "done"}}, "call_0000"),
            ],
        ),
    ],
)
def test_qwen35_raw_multi_action_parses_to_env_executable_action_batch(
    adapter_key: str,
    batch_name: str,
    raw: str,
    expected_actions: list[dict[str, Any]],
    expected_routed: list[tuple[dict[str, Any], str]],
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    msg = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert _names(msg["tool_calls"]) == [batch_name]
    assert _args(msg["tool_calls"][0])["actions"] == expected_actions

    stamp_message_tool_call_ids(msg, preserve=False)
    routed, feedback = prepare_env_tool_calls(
        msg["tool_calls"],
        _metadata_for_adapter_key(adapter_key, []),
    )
    assert routed == expected_routed
    assert feedback == {}


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
def test_qwen_adjacent_action_run_then_bash_has_one_computer_result_slot(adapter_key: str) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    msg = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [
                _computer_use("left_click", coordinate=[100, 100]),
                _computer_use("type", text="before"),
                _tc("bash", command="pwd"),
            ],
        }
    )

    assert _names(msg["tool_calls"]) == ["computer", "bash"]
    assert [_action_name(a) for a in _args(msg["tool_calls"][0])["actions"]] == ["click", "type"]
    stamp_message_tool_call_ids(msg, preserve=False)
    assert [(_id(tc), _name(tc)) for tc in msg["tool_calls"]] == [
        ("call_0000", "computer"),
        ("call_0001", "bash"),
    ]

    rendered = _rendered_agent_tool_calls(adapter_key, msg, extra_tool_schemas=[_BASH_SCHEMA])
    assert _names(rendered) == ["computer_use", "computer_use", "bash"]
    assert [tc["arguments"].get("action") for tc in rendered] == ["left_click", "type", None]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
def test_qwen_malformed_active_extra_reaches_env_feedback(adapter_key: str) -> None:
    """Mixed valid GUI plus a type-wrong active extra: the extra is not dropped.

    The parse direction routes an active extra by key shape, so a bad argument
    VALUE stays a canonical call and ``prepare_env_tool_calls`` names it back to
    the model. Full schema satisfaction here would send it to GUI conversion
    instead, where the model never learns what was wrong.
    """
    metadata = _metadata_for_adapter_key(adapter_key, [_BASH_SCHEMA])
    adapter = AgentAdapterRegistry.get(adapter_key, metadata=metadata)
    msg = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [
                _computer_use("left_click", coordinate=[100, 100]),
                _tc("bash", command=42),
            ],
        }
    )

    assert _names(msg["tool_calls"]) == ["computer", "bash"]
    stamp_message_tool_call_ids(msg, preserve=False)

    routed, feedback = prepare_env_tool_calls(msg["tool_calls"], metadata)
    assert all(call["name"] != "bash" for call, _ in routed)
    assert set(feedback) == {"call_0001"}
    assert "bash" in feedback["call_0001"].message


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@mobile@use", "qwen3_5@mobile@use"])
def test_qwen_mobile_adjacent_action_run_batches_and_restamps_one_top_level_call(
    adapter_key: str,
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    msg = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [
                _mobile_use("click", coordinate=[100, 100]),
                _mobile_use("type", text="hello"),
            ],
        }
    )

    assert _names(msg["tool_calls"]) == ["mobile"]
    assert [_action_name(a) for a in _args(msg["tool_calls"][0])["actions"]] == ["tap", "type"]
    stamp_message_tool_call_ids(msg, preserve=False)
    assert [(_id(tc), _name(tc)) for tc in msg["tool_calls"]] == [("call_0000", "mobile")]

    rendered = _rendered_agent_tool_calls(adapter_key, msg)
    assert _names(rendered) == ["mobile_use", "mobile_use"]
    assert [tc["arguments"].get("action") for tc in rendered] == ["click", "type"]


def test_qwen3_5_mobile_left_click_alias_still_batches() -> None:
    """Qwen3.5 mobile can leak desktop ``left_click``; aliasing must happen before
    the adjacent ``mobile_use`` run is grouped."""
    adapter = AgentAdapterRegistry.get("qwen3_5@mobile@use")
    msg = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [
                _mobile_use("left_click", coordinate=[100, 100]),
                _mobile_use("type", text="hello"),
            ],
        }
    )

    assert _names(msg["tool_calls"]) == ["mobile"]
    assert [_action_name(a) for a in _args(msg["tool_calls"][0])["actions"]] == ["tap", "type"]

    rendered = _rendered_agent_tool_calls("qwen3_5@mobile@use", msg)
    assert _names(rendered) == ["mobile_use", "mobile_use"]
    assert [tc["arguments"].get("action") for tc in rendered] == ["click", "type"]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@mobile@use", "qwen3_5@mobile@use"])
def test_qwen_mobile_grouper_does_not_merge_across_standalone_extra(adapter_key: str) -> None:
    tcs = _qwen_from_agent(
        adapter_key,
        [
            _mobile_use("click", coordinate=[100, 100]),
            _tc("ask_user", question="Continue?"),
            _mobile_use("type", text="after"),
        ],
    )

    assert _names(tcs) == ["mobile", "ask_user", "mobile"]
    assert _args(tcs[0])["actions"] == [{"action": "tap", "coordinate": [100, 100], "clicks": 1}]
    assert _args(tcs[1]) == {"question": "Continue?"}
    assert _args(tcs[2])["actions"] == [{"action": "type", "text": "after"}]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@mobile@use", "qwen3_5@mobile@use"])
def test_qwen_mobile_canonical_batch_render_keeps_standalone_extra_top_level(
    adapter_key: str,
) -> None:
    rendered = _rendered_agent_tool_calls(
        adapter_key,
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Do it."}],
            "tool_calls": [
                make_tool_call(
                    "mobile", {"actions": [{"action": "tap", "coordinate": [100, 100]}]}
                ),
                make_tool_call("ask_user", {"question": "Continue?"}),
                make_tool_call("mobile", {"actions": [{"action": "type", "text": "after"}]}),
            ],
        },
        extra_tool_schemas=[_ASK_USER_SCHEMA],
    )

    assert _names(rendered) == ["mobile_use", "ask_user", "mobile_use"]
    assert rendered[0]["arguments"] == {"action": "click", "coordinate": [100, 100]}
    assert rendered[1]["arguments"] == {"question": "Continue?"}
    assert rendered[2]["arguments"] == {"action": "type", "text": "after"}


def test_qwen_call_ids_are_stamped_after_grouping() -> None:
    adapter = AgentAdapterRegistry.get(
        "qwen3_vl@desktop@use",
        metadata=_metadata_for_adapter_key("qwen3_vl@desktop@use", [_RESPONSE_SCHEMA]),
    )
    msg = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [
                _computer_use("left_click", coordinate=[100, 100]),
                _computer_use("type", text="hello"),
                _computer_use("answer", text="done"),
            ],
        }
    )

    assert _names(msg["tool_calls"]) == ["computer", "response"]
    stamp_message_tool_call_ids(msg, preserve=False)
    assert [(_id(tc), _name(tc)) for tc in msg["tool_calls"]] == [
        ("call_0000", "computer"),
        ("call_0001", "response"),
    ]


@pytest.mark.parametrize("adapter_key", ["qwen3_vl@desktop@use", "qwen3_5@desktop@use"])
@pytest.mark.parametrize(
    "terminal_call,terminal_name",
    [
        (_computer_use("answer", text="done"), "response"),
        (_computer_use("terminate", status="failure"), "terminate"),
    ],
)
def test_qwen_terminal_tool_order_and_call_ids_after_grouping(
    adapter_key: str,
    terminal_call: dict[str, Any],
    terminal_name: str,
) -> None:
    schema = _RESPONSE_SCHEMA if terminal_name == "response" else _TERMINATE_SCHEMA_FOR_ACTIVE
    adapter = AgentAdapterRegistry.get(
        adapter_key,
        metadata=_metadata_for_adapter_key(adapter_key, [schema]),
    )
    msg = adapter.convert_message_from_agent(
        {
            "role": "assistant",
            "tool_calls": [
                _computer_use("left_click", coordinate=[100, 100]),
                _computer_use("type", text="hello"),
                terminal_call,
            ],
        }
    )

    assert _names(msg["tool_calls"]) == ["computer", terminal_name]
    stamp_message_tool_call_ids(msg, preserve=False)
    assert [(_id(tc), _name(tc)) for tc in msg["tool_calls"]] == [
        ("call_0000", "computer"),
        ("call_0001", terminal_name),
    ]


def test_evocua_multiple_native_wrapper_calls_merge_to_computer_batch() -> None:
    space = ActionSpaceRegistry.get("evocua@desktop")
    tcs = space.convert_tool_calls_from_agent(
        [
            _computer_use("left_click", coordinate=[500, 300]),
            _computer_use("type", text="hello"),
        ]
    )

    assert _names(tcs) == ["computer"]
    assert _args(tcs[0])["actions"] == [
        {"action": "click", "coordinate": [500, 300]},
        {"action": "type", "text": "hello"},
    ]


@pytest.mark.parametrize(
    "adapter_key,agent_tool_calls,wrapper_name,expected_actions",
    [
        (
            "evocua@desktop@use",
            [
                _computer_use("left_click", coordinate=[500, 300]),
                _computer_use("type", text="hello"),
            ],
            "computer",
            [
                {"action": "click", "coordinate": [500, 300]},
                {"action": "type", "text": "hello"},
            ],
        ),
        (
            "mai_ui@mobile@use",
            [
                _mobile_use("click", coordinate=[499, 300]),
                _mobile_use("type", text="hello"),
            ],
            "mobile",
            [
                {"action": "tap", "coordinate": [499, 300], "clicks": 1},
                {"action": "type", "text": "hello"},
            ],
        ),
        (
            "step_gui@mobile@use",
            [
                _mobile_use("CLICK", point=[500, 300]),
                _mobile_use("TYPE", value="hello"),
            ],
            "mobile",
            [
                {"action": "tap", "coordinate": [500, 300], "clicks": 1},
                {"action": "type", "text": "hello"},
            ],
        ),
    ],
)
def test_non_qwen3_native_wrapper_families_merge_adjacent_action_calls(
    adapter_key: str,
    agent_tool_calls: list[dict[str, Any]],
    wrapper_name: str,
    expected_actions: list[dict[str, Any]],
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    lite = adapter.convert_message_from_agent({"role": "assistant", "tool_calls": agent_tool_calls})

    assert _names(lite["tool_calls"]) == [wrapper_name]
    assert _args(lite["tool_calls"][0])["actions"] == expected_actions


@pytest.mark.parametrize(
    "adapter_key,agent_tool_calls,wrapper_name,expected_actions",
    [
        (
            "qwen2_5_vl@desktop@use",
            [
                _computer_use("left_click", coordinate=[500, 300]),
                _computer_use("type", text="hello"),
            ],
            "computer",
            [
                {"action": "click", "coordinate": [500, 300]},
                {"action": "type", "text": "hello"},
            ],
        ),
        (
            "qwen2_5_vl@mobile@use",
            [
                _mobile_use("click", coordinate=[500, 300]),
                _mobile_use("type", text="hello"),
            ],
            "mobile",
            [
                {"action": "tap", "coordinate": [500, 300], "clicks": 1},
                {"action": "type", "text": "hello"},
            ],
        ),
        (
            "fara@desktop@use",
            [
                _computer_use("left_click", coordinate=[500, 300]),
                _computer_use("type", text="hello"),
            ],
            "computer",
            [
                {"action": "click", "coordinate": [500, 300]},
                {"action": "type", "text": "hello"},
            ],
        ),
    ],
)
def test_qwen2_fara_native_calls_merge_adjacent_action_wrappers(
    adapter_key: str,
    agent_tool_calls: list[dict[str, Any]],
    wrapper_name: str,
    expected_actions: list[dict[str, Any]],
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    lite = adapter.convert_message_from_agent({"role": "assistant", "tool_calls": agent_tool_calls})

    assert _names(lite["tool_calls"]) == [wrapper_name]
    assert _args(lite["tool_calls"][0])["actions"] == expected_actions


@pytest.mark.parametrize(
    "adapter_key,raw,expected_actions",
    [
        (
            "ui_tars@desktop@use",
            "Thought: Do two things.\n"
            "Action: click(start_box='(500,300)')\n\n"
            "type(content='hello')",
            [
                {"action": "click", "coordinate": [500, 300]},
                {"action": "type", "text": "hello"},
            ],
        ),
        (
            "ui_tars_15_v1@desktop@use",
            "Thought: Do two things.\n"
            "Action: click(start_box='(500,300)')\n\n"
            "type(content='hello')",
            [
                {"action": "click", "coordinate": [500, 300]},
                {"action": "type", "text": "hello"},
            ],
        ),
    ],
)
def test_text_wire_parsers_merge_adjacent_actions(
    adapter_key: str,
    raw: str,
    expected_actions: list[dict[str, Any]],
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    lite = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))

    assert _names(lite["tool_calls"]) == ["computer"]
    assert _args(lite["tool_calls"][0])["actions"] == expected_actions


# =============================================================================
# len>1 batch: render expansion / roundtrip / ordering hazard for the remaining
# open-source families
# =============================================================================
#
# ``qwen3_vl`` / ``qwen3_5`` get the full treatment above. The table below
# extends the SAME three guarantees to every other SFT'd open-source family, all
# of which are trained to emit a LIST of flat native GUI calls per turn:
#
#   1. **render expansion** — one canonical ``computer/mobile{actions:[a,b,c]}``
#      renders to THREE adjacent native emissions, in order.
#   2. **roundtrip** — ``parse(render(batch)) == batch`` (canonical identity).
#   3. **ordering hazard** — ``[GUI, <standalone extra>, GUI]`` stays THREE
#      canonical calls; the extra is never swallowed by, nor merged across, the
#      neighbouring action-batch calls.
#
# ``terminate`` is the standalone extra everywhere: it is the one canonical
# non-GUI tool every one of these families has a native spelling for
# (qwen wrappers ``action="terminate"``, UI-TARS ``finished()``, Step-GUI
# ``action:COMPLETE``). ``response`` is deliberately NOT used — several of these
# families have no native answer channel and raise on render
# (fara / evocua / ui_tars desktop).

_TERMINATE_SCHEMA = LiteFinishToolSet.get_tool_schema("terminate")
_TERMINATE_CALL = make_tool_call("terminate", {"status": "success"})

# One ``fn(arg=...)`` call per line — UI-TARS's ``Action:`` grammar.
_CALL_GRAMMAR_RE = re.compile(r"(?m)^(?:Action:\s*)?([A-Za-z_]+)\(")
# Step-GUI: tab-separated ``key:value`` records, one record per line.
_STEP_GUI_ACTION_RE = re.compile(r"action:([A-Z]+)")


@dataclasses.dataclass(frozen=True)
class _BatchFamilyCase:
    """One family/platform slice of the len>1 batching contract.

    ``wire`` selects how native emissions are counted off the rendered turn:

    * ``wrapper`` — qwen-style ``computer_use`` / ``mobile_use`` calls, either
      structured on the agent message (evocua) or as ``<tool_call>`` JSON inside
      the rendered text (qwen2_5_vl, mai_ui, fara).
    * ``call_grammar`` — one ``fn(...)`` call per line (ui_tars*).
    * ``step_gui`` — tab-separated ``action:NAME`` records, one per line.
    """

    id: str
    key: str
    action_batch_tool: str  # canonical action-batch tool: computer | mobile
    wire: str
    batch: list[dict[str, Any]]  # canonical children, len 3
    native_batch: list[str]  # expected native emissions for ``batch``
    native_hazard: list[str]  # expected emissions for [GUI, extra, GUI]
    hazard_action: list[dict[str, Any]]  # the two single-action action-batch calls
    render_xfail: str | None = None
    roundtrip_xfail: str | None = None
    hazard_xfail: str | None = None


def _agent_text(agent: dict[str, Any]) -> str:
    return next((c["text"] for c in agent.get("content", []) if c.get("type") == "text"), "")


def _wrapper_emissions(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Native action name of each qwen-style wrapper call (fall back to the
    tool name for extras rendered as their own top-level tool)."""
    return [tc["arguments"].get("action") or tc["name"] for tc in tool_calls]


def _family_render_and_reparse(
    case: _BatchFamilyCase,
    tool_calls: list[dict[str, Any]],
    *,
    extra_tool_schemas: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Render canonical ``tool_calls`` to the family's native wire, then parse
    straight back. Returns ``(native emissions, canonical tool_calls)``."""
    adapter = AgentAdapterRegistry.get(
        case.key,
        metadata=_metadata_for_adapter_key(case.key, extra_tool_schemas or []),
    )
    agent = adapter.convert_message_to_agent(
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Do it."}],
            "tool_calls": tool_calls,
        }
    )

    if agent.get("tool_calls"):
        # Structured wire (evocua): the native calls are already on the message.
        native = _wrapper_emissions(agent["tool_calls"])
        back = adapter.convert_message_from_agent(agent)
        return native, back.get("tool_calls", [])

    text = _agent_text(agent)
    reparsed = adapter.parse_raw_assistant_response(text)
    if case.wire == "wrapper":
        native = _wrapper_emissions(reparsed["tool_calls"])
    elif case.wire == "call_grammar":
        native = _CALL_GRAMMAR_RE.findall(text)
    elif case.wire == "step_gui":
        native = _STEP_GUI_ACTION_RE.findall(text)
    else:
        raise AssertionError(f"unknown wire {case.wire!r}")
    back = adapter.convert_message_from_agent(reparsed)
    return native, back.get("tool_calls", [])


# Desktop batch: click + type + key. Mobile batch: tap + type + tap (mobile
# grammars normalize ``swipe``/``scroll`` args per family, which is orthogonal
# to batching, so the third slot stays a action every family renders 1:1).
_DESKTOP_BATCH = [
    {"action": "click", "coordinate": [100, 100]},
    {"action": "type", "text": "hi"},
    {"action": "key", "keys": ["ctrl", "s"]},
]
# Mobile ``tap`` canonically carries ``clicks: 1`` (see
# ``test_single_qwen_mobile_action_roundtrips_as_length1_wrapper``), so the
# input batch spells it out to keep the roundtrip an exact identity.
_MOBILE_BATCH = [
    {"action": "tap", "coordinate": [100, 100], "clicks": 1},
    {"action": "type", "text": "hi"},
    {"action": "tap", "coordinate": [200, 200], "clicks": 1},
]
_DESKTOP_HAZARD_ACTION = [
    [{"action": "click", "coordinate": [100, 100]}],
    [{"action": "type", "text": "after"}],
]
_MOBILE_HAZARD_ACTION = [
    [{"action": "tap", "coordinate": [100, 100], "clicks": 1}],
    [{"action": "type", "text": "after"}],
]


def _desktop_case(
    fam: str,
    wire: str,
    native_batch: list[str],
    native_hazard: list[str],
    **xfails: str,
) -> _BatchFamilyCase:
    return _BatchFamilyCase(
        id=f"{fam}@desktop",
        key=f"{fam}@desktop@use",
        action_batch_tool="computer",
        wire=wire,
        batch=_DESKTOP_BATCH,
        native_batch=native_batch,
        native_hazard=native_hazard,
        hazard_action=_DESKTOP_HAZARD_ACTION,
        **xfails,
    )


def _mobile_case(
    fam: str,
    wire: str,
    native_batch: list[str],
    native_hazard: list[str],
    **xfails: str,
) -> _BatchFamilyCase:
    return _BatchFamilyCase(
        id=f"{fam}@mobile",
        key=f"{fam}@mobile@use",
        action_batch_tool="mobile",
        wire=wire,
        batch=_MOBILE_BATCH,
        native_batch=native_batch,
        native_hazard=native_hazard,
        hazard_action=_MOBILE_HAZARD_ACTION,
        **xfails,
    )


_BATCH_FAMILY_CASES: list[_BatchFamilyCase] = [
    # -- qwen2.5-VL: qwen ``<tool_call>`` JSON wrappers, both platforms -------
    _desktop_case(
        "qwen2_5_vl", "wrapper", ["left_click", "type", "key"], ["left_click", "terminate", "type"]
    ),
    _mobile_case(
        "qwen2_5_vl", "wrapper", ["click", "type", "click"], ["click", "terminate", "type"]
    ),
    # -- MAI-UI: mobile only --------------------------------------------------
    _mobile_case("mai_ui", "wrapper", ["click", "type", "click"], ["click", "terminate", "type"]),
    # -- Fara: desktop/browser only ------------------------------------------
    _desktop_case(
        "fara", "wrapper", ["left_click", "type", "key"], ["left_click", "terminate", "type"]
    ),
    # -- EvoCUA: desktop/browser only, structured tool_calls on the wire ------
    _desktop_case(
        "evocua", "wrapper", ["left_click", "type", "key"], ["left_click", "terminate", "type"]
    ),
    # -- UI-TARS families: ``Action:`` call grammar ---------------------------
    _desktop_case(
        "ui_tars", "call_grammar", ["click", "type", "hotkey"], ["click", "finished", "type"]
    ),
    _mobile_case(
        "ui_tars", "call_grammar", ["click", "type", "click"], ["click", "finished", "type"]
    ),
    _desktop_case(
        "ui_tars_15_v1", "call_grammar", ["click", "type", "hotkey"], ["click", "finished", "type"]
    ),
    _mobile_case(
        "ui_tars_15_v1", "call_grammar", ["click", "type", "click"], ["click", "finished", "type"]
    ),
    # -- Step-GUI: mobile only, tab-separated KV records ----------------------
    _mobile_case(
        "step_gui",
        "step_gui",
        ["CLICK", "TYPE", "CLICK"],
        ["CLICK", "COMPLETE", "TYPE"],
    ),
]


def _batch_params(xfail_field: str) -> list[Any]:
    """One ``pytest.param`` per family, carrying that case's strict xfail (if
    the family has a real defect on this axis)."""
    params = []
    for case in _BATCH_FAMILY_CASES:
        reason = getattr(case, xfail_field)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(case, id=case.id, marks=marks))
    return params


@pytest.mark.parametrize("case", _batch_params("render_xfail"))
def test_family_canonical_batch_renders_as_n_adjacent_native_emissions(
    case: _BatchFamilyCase,
) -> None:
    """A canonical batch of THREE child actions renders to THREE adjacent
    native emissions, in the same order — no collapsing, no reordering."""
    native, _ = _family_render_and_reparse(
        case, [make_tool_call(case.action_batch_tool, {"actions": case.batch})]
    )

    assert len(native) == len(case.batch), (
        f"{case.id}: {len(case.batch)} canonical actions rendered to "
        f"{len(native)} native emissions ({native})"
    )
    assert native == case.native_batch


@pytest.mark.parametrize("case", _batch_params("roundtrip_xfail"))
def test_family_canonical_batch_render_parse_roundtrip(case: _BatchFamilyCase) -> None:
    """``parse(render(batch)) == batch`` for a len-3 canonical action-batch call."""
    batch = make_tool_call(case.action_batch_tool, {"actions": case.batch})
    _, back = _family_render_and_reparse(case, [batch])

    assert back == [batch]


@pytest.mark.parametrize("case", _batch_params("hazard_xfail"))
def test_family_ordering_hazard_keeps_standalone_extra_unmerged(
    case: _BatchFamilyCase,
) -> None:
    """``[GUI, terminate, GUI]`` renders to three adjacent native emissions and
    parses back to THREE canonical calls — the two GUI runs must NOT merge
    across the standalone extra, and the extra must stay top-level."""
    before, after = case.hazard_action
    tool_calls = [
        make_tool_call(case.action_batch_tool, {"actions": before}),
        _TERMINATE_CALL,
        make_tool_call(case.action_batch_tool, {"actions": after}),
    ]

    native, back = _family_render_and_reparse(
        case, tool_calls, extra_tool_schemas=[_TERMINATE_SCHEMA]
    )

    assert native == case.native_hazard
    assert _names(back) == [
        case.action_batch_tool,
        "terminate",
        case.action_batch_tool,
    ]
    assert back == tool_calls
