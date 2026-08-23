"""Raw adapter calls that must survive into env-owned routing.

These tests sit at the adapter/env-ingress seam: parse the model family's raw
wire format, convert to canonical Lite tool calls, stamp runtime call_ids, then
run the shared env ingress helpers that concrete envs use before dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
)
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import stamp_message_tool_call_ids
from lite.gym.utils.feedback.errors import (
    current_feedback,
    unavailable_action_message,
)
from lite.gym.utils.feedback.ingress import (
    is_inactive_tool_call,
    prepare_env_tool_calls,
)

register_all()

_KNOWN_STANDALONE = frozenset({"open_app", "ask_user", "response", "terminate"})


def _schema(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return make_tool_schema(
        name,
        description=f"{name} tool",
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


def _mobile_metadata(
    *,
    extra_tool_schemas: list[dict[str, Any]] | None = None,
) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tool_schemas or [],
        valid_actions=None,
    )


def _desktop_metadata(
    *,
    extra_tool_schemas: list[dict[str, Any]] | None = None,
) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tool_schemas or [],
        valid_actions=None,
    )


def _click_index_schema() -> dict[str, Any]:
    return _schema("click", {"index": {"type": "integer"}}, ["index"])


def _response_schema() -> dict[str, Any]:
    return _schema("response", {"text": {"type": "string"}}, ["text"])


def _terminate_schema(*, statuses: list[str] | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {"type": "string"}
    if statuses is not None:
        status["enum"] = statuses
    return _schema("terminate", {"status": status}, ["status"])


def _open_app_schema(*, apps: list[str] | None = None) -> dict[str, Any]:
    app_name: dict[str, Any] = {"type": "string"}
    if apps is not None:
        app_name["enum"] = apps
    return _schema("open_app", {"app_name": app_name}, ["app_name"])


def _assert_parse_error_no_tool_calls(message: LiteMessage) -> None:
    assert not message.get("tool_calls")
    assert pop_model_output_error(message)


def _qwen_json_tool(name: str, arguments: dict[str, Any]) -> str:
    import json

    return (
        "Action: use tool.\n"
        "<tool_call>\n"
        f"{json.dumps({'name': name, 'arguments': arguments})}\n"
        "</tool_call>"
    )


def _qwen_xml_tool(name: str, arguments: dict[str, Any]) -> str:
    body = "\n".join(
        f"<parameter={key}>\n{value}\n</parameter>" for key, value in arguments.items()
    )
    return f"Action: use tool.\n<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"


def _mai_tool(name: str, arguments: dict[str, Any]) -> str:
    import json

    return (
        "<thinking>\nUse tool.\n</thinking>\n"
        "<tool_call>\n"
        f"{json.dumps({'name': name, 'arguments': arguments}, separators=(',', ':'))}\n"
        "</tool_call>"
    )


def _parse_lite_message(
    adapter_key: str,
    raw: str,
    *,
    metadata: LiteCUAMetadata | None = None,
) -> LiteMessage:
    adapter = AgentAdapterRegistry.get(adapter_key, metadata=metadata)
    parsed = adapter.parse_raw_assistant_response(raw)
    return adapter.convert_message_from_agent(parsed)


def _stamp_and_route_inactive(message: LiteMessage, metadata: LiteCUAMetadata) -> str:
    assert message.get("tool_calls"), message
    stamp_message_tool_call_ids(message, preserve=False)
    routed, feedback = prepare_env_tool_calls(message["tool_calls"], metadata)
    assert not feedback

    for action, parent_call_id in routed:
        if is_inactive_tool_call(
            action,
            _KNOWN_STANDALONE,
            metadata.extra_tool_schemas,
        ):
            assert parent_call_id
            feedback[parent_call_id] = current_feedback(unavailable_action_message(action["name"]))

    assert len(feedback) == 1
    [item] = feedback.values()
    return item.message


@pytest.mark.parametrize(
    "adapter_key,raw_factory",
    [
        (
            "qwen2_5_vl@mobile@use",
            lambda: _qwen_json_tool(
                "mobile_use",
                {"action": "open", "text": "Settings"},
            ),
        ),
        (
            "mai_ui@mobile@use",
            lambda: _mai_tool(
                "mobile_use",
                {"action": "open", "text": "Settings"},
            ),
        ),
    ],
)
def test_qwen25_and_mai_native_mobile_open_reaches_env_owned_open_app(
    adapter_key: str,
    raw_factory: Callable[[], str],
) -> None:
    raw = raw_factory()

    without_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(),
    )
    assert without_schema["tool_calls"] == [make_tool_call("open_app", {"app_name": "Settings"})]

    with_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(extra_tool_schemas=[_open_app_schema(apps=["Settings"])]),
    )
    assert with_schema["tool_calls"] == [make_tool_call("open_app", {"app_name": "Settings"})]

    invalid_enum = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(extra_tool_schemas=[_open_app_schema(apps=["Chrome"])]),
    )
    assert invalid_enum["tool_calls"] == [make_tool_call("open_app", {"app_name": "Settings"})]


@pytest.mark.parametrize(
    "adapter_key,raw_factory",
    [
        (
            "qwen3_vl@mobile@use",
            lambda: _qwen_json_tool(
                "mobile_use",
                {"action": "open", "text": "Settings"},
            ),
        ),
        (
            "qwen3_5@mobile@use",
            lambda: _qwen_xml_tool(
                "mobile_use",
                {"action": "open", "text": "Settings"},
            ),
        ),
    ],
)
def test_qwen3_native_mobile_open_raw_reaches_env_owned_open_app(
    adapter_key: str,
    raw_factory: Callable[[], str],
) -> None:
    raw = raw_factory()

    without_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(),
    )
    assert without_schema["tool_calls"] == [make_tool_call("open_app", {"app_name": "Settings"})]

    with_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(extra_tool_schemas=[_open_app_schema(apps=["Settings"])]),
    )
    assert with_schema["tool_calls"] == [make_tool_call("open_app", {"app_name": "Settings"})]

    invalid_enum = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(extra_tool_schemas=[_open_app_schema(apps=["Chrome"])]),
    )
    assert invalid_enum["tool_calls"] == [make_tool_call("open_app", {"app_name": "Settings"})]
    stamp_message_tool_call_ids(invalid_enum, preserve=False)
    routed, feedback = prepare_env_tool_calls(
        invalid_enum["tool_calls"],
        _mobile_metadata(extra_tool_schemas=[_open_app_schema(apps=["Chrome"])]),
    )
    assert routed == []
    assert feedback == {
        "call_0000": current_feedback(
            "invalid arguments for open_app: "
            "open_app.arguments.app_name 'Settings' is not one of ['Chrome']"
        )
    }


@pytest.mark.parametrize(
    "adapter_key,raw_factory",
    [
        (
            "qwen3_vl@mobile@use",
            lambda: _qwen_json_tool(
                "mobile_use",
                {"action": "answer", "text": "Done."},
            ),
        ),
        (
            "qwen3_5@mobile@use",
            lambda: _qwen_xml_tool(
                "mobile_use",
                {"action": "answer", "text": "Done."},
            ),
        ),
        (
            "qwen2_5_vl@mobile@use",
            lambda: _qwen_json_tool(
                "mobile_use",
                {"action": "answer", "text": "Done."},
            ),
        ),
        (
            "mai_ui@mobile@use",
            lambda: _mai_tool(
                "mobile_use",
                {"action": "answer", "text": "Done."},
            ),
        ),
    ],
)
def test_native_answer_raw_reaches_env_owned_response(
    adapter_key: str,
    raw_factory: Callable[[], str],
) -> None:
    raw = raw_factory()

    without_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(),
    )
    assert without_schema["tool_calls"] == [make_tool_call("response", {"text": "Done."})]

    with_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(extra_tool_schemas=[_response_schema()]),
    )
    assert with_schema["tool_calls"] == [make_tool_call("response", {"text": "Done."})]


@pytest.mark.parametrize(
    "adapter_key,raw_factory",
    [
        (
            "qwen3_vl@mobile@use",
            lambda: _qwen_json_tool(
                "mobile_use",
                {"action": "terminate", "status": "success"},
            ),
        ),
        (
            "qwen3_5@mobile@use",
            lambda: _qwen_xml_tool(
                "mobile_use",
                {"action": "terminate", "status": "success"},
            ),
        ),
        (
            "qwen2_5_vl@mobile@use",
            lambda: _qwen_json_tool(
                "mobile_use",
                {"action": "terminate", "status": "success"},
            ),
        ),
        (
            "mai_ui@mobile@use",
            lambda: _mai_tool(
                "mobile_use",
                {"action": "terminate", "status": "success"},
            ),
        ),
    ],
)
def test_native_mobile_terminate_reaches_env_owned_terminate(
    adapter_key: str,
    raw_factory: Callable[[], str],
) -> None:
    raw = raw_factory()

    without_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(),
    )
    assert without_schema["tool_calls"] == [make_tool_call("terminate", {"status": "success"})]

    with_schema = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(extra_tool_schemas=[_terminate_schema(statuses=["success"])]),
    )
    assert with_schema["tool_calls"] == [make_tool_call("terminate", {"status": "success"})]

    invalid_enum = _parse_lite_message(
        adapter_key,
        raw,
        metadata=_mobile_metadata(extra_tool_schemas=[_terminate_schema(statuses=["failure"])]),
    )
    assert invalid_enum["tool_calls"] == [make_tool_call("terminate", {"status": "success"})]


def test_qwen2_5_desktop_raw_answer_reaches_env_owned_response() -> None:
    raw = _qwen_json_tool(
        "computer_use",
        {"action": "answer", "text": "Done."},
    )

    without_schema = _parse_lite_message(
        "qwen2_5_vl@desktop@use",
        raw,
        metadata=_desktop_metadata(),
    )
    assert without_schema["tool_calls"] == [make_tool_call("response", {"text": "Done."})]

    with_schema = _parse_lite_message(
        "qwen2_5_vl@desktop@use",
        raw,
        metadata=_desktop_metadata(extra_tool_schemas=[_response_schema()]),
    )
    assert with_schema["tool_calls"] == [make_tool_call("response", {"text": "Done."})]


def test_fara_native_visit_url_reaches_env_owned_goto() -> None:
    raw = _qwen_json_tool(
        "computer_use",
        {"action": "visit_url", "url": "https://example.com"},
    )

    without_schema = _parse_lite_message(
        "fara@desktop@use",
        raw,
        metadata=_desktop_metadata(),
    )
    assert without_schema["tool_calls"] == [make_tool_call("goto", {"url": "https://example.com"})]

    with_schema = _parse_lite_message(
        "fara@desktop@use",
        raw,
        metadata=_desktop_metadata(
            extra_tool_schemas=[
                _schema(
                    "goto", {"url": {"type": "string", "enum": ["https://example.com"]}}, ["url"]
                )
            ]
        ),
    )
    assert with_schema["tool_calls"] == [make_tool_call("goto", {"url": "https://example.com"})]

    invalid_enum = _parse_lite_message(
        "fara@desktop@use",
        raw,
        metadata=_desktop_metadata(
            extra_tool_schemas=[
                _schema(
                    "goto", {"url": {"type": "string", "enum": ["https://other.example"]}}, ["url"]
                )
            ]
        ),
    )
    assert invalid_enum["tool_calls"] == [make_tool_call("goto", {"url": "https://example.com"})]


@pytest.mark.parametrize(
    "adapter_key,raw_factory",
    [
        (
            "qwen3_vl@mobile@use",
            lambda name, args: _qwen_json_tool(name, args),
        ),
        (
            "qwen3_5@mobile@use",
            lambda name, args: _qwen_xml_tool(name, args),
        ),
        (
            "qwen2_5_vl@mobile@use",
            lambda name, args: _qwen_json_tool(name, args),
        ),
    ],
)
@pytest.mark.parametrize(
    "name,args",
    [
        ("ask_user", {"question": "Continue?"}),
        ("response", {"text": "Done."}),
        ("terminate", {"status": "success"}),
    ],
)
def test_top_level_raw_standalone_calls_reach_env_owned_inactive_routing(
    adapter_key: str,
    raw_factory: Callable[[str, dict[str, Any]], str],
    name: str,
    args: dict[str, Any],
) -> None:
    metadata = _mobile_metadata()
    lite_msg = _parse_lite_message(
        adapter_key,
        raw_factory(name, args),
        metadata=metadata,
    )

    assert lite_msg["tool_calls"] == [make_tool_call(name, args)]
    assert _stamp_and_route_inactive(lite_msg, metadata) == (
        f"{name} is not available in this task."
    )


def test_mai_ui_top_level_unadvertised_extra_reaches_inactive_routing() -> None:
    metadata = _mobile_metadata()
    lite_msg = _parse_lite_message(
        "mai_ui@mobile@use",
        _mai_tool("ask_user", {"question": "Continue?"}),
        metadata=metadata,
    )

    assert lite_msg["tool_calls"] == [make_tool_call("ask_user", {"question": "Continue?"})]
    assert _stamp_and_route_inactive(lite_msg, metadata) == (
        "ask_user is not available in this task."
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("open_app", {"app_name": "Settings"}),
        ("ask_user", {"question": "Continue?"}),
        ("response", {"text": "Done."}),
        ("terminate", {"status": "success"}),
    ],
)
def test_lite_raw_replay_standalone_calls_reach_env_owned_inactive_routing(
    name: str,
    args: dict[str, Any],
) -> None:
    metadata = _mobile_metadata()
    raw_replay: LiteMessage = {
        "role": "assistant",
        "tool_calls": [make_tool_call(name, args)],
    }

    lite_msg = AsIsAdapter(metadata=metadata).convert_message_from_agent(raw_replay)

    assert lite_msg["tool_calls"] == [make_tool_call(name, args)]
    assert _stamp_and_route_inactive(lite_msg, metadata) == (
        f"{name} is not available in this task."
    )


def test_lite_raw_invalid_batch_child_uses_child_action_in_feedback() -> None:
    metadata = _mobile_metadata()
    message: LiteMessage = {
        "role": "assistant",
        "tool_calls": [
            make_tool_call(
                "mobile",
                {"actions": [{"action": "left_click", "coordinate": [10, 20]}]},
            )
        ],
    }
    lite_msg = AsIsAdapter(metadata=metadata).convert_message_from_agent(message)
    stamp_message_tool_call_ids(lite_msg, preserve=False)

    routed, feedback = prepare_env_tool_calls(lite_msg["tool_calls"], metadata)

    # R4: the bad child keeps its slot so the env can answer it per action and
    # still frame it. What this test guards is the WORDING -- that the message
    # names the CHILD action, not the batch tool -- and that is unchanged.
    assert [a["name"] for a, _ in routed] == ["left_click"]
    assert routed[0][0]["_rejected_reason"] == (
        "invalid action: left_click; mobile.actions cannot contain left_click"
    )
    assert feedback == {}


def test_active_extra_bad_arguments_use_visible_tool_name() -> None:
    ask_user_schema = _schema(
        "ask_user",
        {"text": {"type": "string"}},
        ["text"],
    )
    metadata = _mobile_metadata(extra_tool_schemas=[ask_user_schema])
    message: LiteMessage = {
        "role": "assistant",
        "tool_calls": [make_tool_call("ask_user", {"question": "Continue?"})],
    }
    stamp_message_tool_call_ids(message, preserve=False)

    routed, feedback = prepare_env_tool_calls(message["tool_calls"], metadata)

    assert routed == []
    assert feedback == {
        "call_0000": current_feedback(
            "invalid arguments for ask_user: ask_user.arguments.text is required"
        )
    }


@pytest.mark.parametrize(
    "adapter_key,raw_factory",
    [
        ("qwen3_vl@desktop@use", lambda name, args: _qwen_json_tool(name, args)),
        ("qwen3_5@desktop@use", lambda name, args: _qwen_xml_tool(name, args)),
        ("qwen2_5_vl@desktop@use", lambda name, args: _qwen_json_tool(name, args)),
    ],
)
def test_same_name_action_extra_raw_routes_by_schema_shape(
    adapter_key: str,
    raw_factory: Callable[[str, dict[str, Any]], str],
) -> None:
    metadata = _desktop_metadata(extra_tool_schemas=[_click_index_schema()])

    standalone = _parse_lite_message(
        adapter_key,
        raw_factory("click", {"index": 7}),
        metadata=metadata,
    )
    action = _parse_lite_message(
        adapter_key,
        raw_factory(
            "computer_use",
            {"action": "left_click", "coordinate": [10, 20]},
        ),
        metadata=metadata,
    )

    assert standalone["tool_calls"] == [make_tool_call("click", {"index": 7})]
    stamp_message_tool_call_ids(standalone, preserve=False)
    routed, feedback = prepare_env_tool_calls(standalone["tool_calls"], metadata)
    assert routed == [
        (
            {"call_id": "call_0000", "name": "click", "arguments": {"index": 7}},
            "call_0000",
        )
    ]
    assert feedback == {}

    assert action["tool_calls"] == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [10, 20]}]},
        )
    ]
