"""Rendering a canonical action-batch call the env rejected must not be fatal.

The env deliberately KEEPS a malformed ``computer/mobile.actions`` child in the
action-batch call and answers it with per-call feedback instead of failing the step
(``prepare_env_tool_calls`` in :mod:`lite.gym.utils.feedback.ingress`; pinned by
the campaign tool I/O regression tests). The next turn therefore
re-renders that turn, and the adapter must be able to render what the env
preserved. ``BaseAgent._predict_with_details`` renders OUTSIDE its try, so a
raise here costs the whole trajectory — 4 of 32 webgym episodes in the live
rollout matrix, with zero environment failures.

Both live cases from that matrix are pinned here:

  * ``computer.actions cannot contain mouse_event`` — a hallucinated child name
    (``mouse_event`` occurs nowhere in ``lite/``), Qwen3.5 desktop.
  * ``computer.actions cannot contain back`` — ``back`` is a REAL, ACTIVE
    ``extra_tools`` entry that the model put one layer too deep, Qwen3-VL
    desktop.

A payload with no renderable shape at all still raises: that is the part of the
check with real value, and it is the same branch the env side keys off
(``LiteActionBatchValidationError.child_action_name`` is set only for the
vocabulary failure).

Hermetic: pure adapter/dict assertions — no model, no network, no env.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/adapter/test_render_invalid_batch_child.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import logging
import re

import pytest
from agents._support.valid_actions_gating import agent_adapter_for
from PIL import Image

from lite.agents.core.adapter import base as adapter_base
from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.samples import LiteSample
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments


def _schema(name: str, props: dict) -> dict:
    return make_tool_schema(
        name,
        description=name,
        parameters={"type": "object", "properties": props, "required": []},
    )


BACK_SCHEMA = _schema("back", {})
GOTO_SCHEMA = _schema("goto", {"url": {"type": "string"}})
RESPONSE_SCHEMA = _schema("response", {"text": {"type": "string"}})

_DESKTOP_ADAPTERS = (Qwen3_5DesktopUseAdapter, Qwen3VLDesktopUseAdapter)


def _browser_md(extras: list[dict]) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extras,
        valid_actions=None,
    )


def _assistant_message(child: dict) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "go back"}],
        "tool_calls": [make_tool_call("computer", {"actions": [child]})],
    }


def _sample(metadata: LiteCUAMetadata, child: dict) -> LiteSample:
    return LiteSample(
        messages=[
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            _assistant_message(child),
        ],
        images=[Image.new("RGB", (64, 64))],
        metadata=metadata,
    )


def _render(adapter, sample: LiteSample):
    processed = [adapter.process_image(img) for img in sample.images]
    return adapter.render_step(sample, 1, processed)


# The adapter reports per-call action-batch feedback by warning; that warning is
# the only place the wording is observable from outside the render.
_ENV_REJECTED = "is rendering a canonical action-batch the env already rejected: "


@pytest.fixture(autouse=True)
def _fresh_action_batch_warning_dedup():
    """The warning is deduped per ``(adapter, feedback)`` pair, so a test that
    reads feedback off it needs the dedup cache to start empty."""
    adapter_base._warn_invalid_action_batch_child.cache_clear()


def _reported_feedback(adapter, child: dict, caplog) -> list[str]:
    """Feedback the adapter reports while rendering ``child``, in order.

    ``convert_message_to_agent`` is the public render entry the agent loop calls;
    feedback for a preserved, env-rejected action-batch child surfaces there as a
    warning rather than a return value.
    """
    with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter.base"):
        adapter.convert_message_to_agent(_assistant_message(child))
    return [
        message.split(_ENV_REJECTED, 1)[1]
        for message in (record.getMessage() for record in caplog.records)
        if _ENV_REJECTED in message
    ]


# =============================================================================
# Case 1 — a hallucinated child name
# =============================================================================

@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_render_step_survives_hallucinated_action_batch_child(adapter_cls) -> None:
    """Pre-fix: ValueError '… action-batch contains invalid actions: computer.actions
    cannot contain mouse_event' out of render_step, killing the trajectory."""
    metadata = _browser_md([BACK_SCHEMA, GOTO_SCHEMA, RESPONSE_SCHEMA])
    adapter = adapter_cls(metadata=metadata)
    sample = _sample(metadata, {"action": "mouse_event", "coordinate": [10, 20]})

    step = _render(adapter, sample)

    assert [m["role"] for m in step][-1] == "assistant"
    # The canonical action-batch call is preserved for the env, exactly as ingress left it.
    assert sample.messages[1]["tool_calls"] == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "mouse_event", "coordinate": [10, 20]}]},
        )
    ]


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_hallucinated_action_batch_child_feedback_matches_env_wording(
    adapter_cls, caplog
) -> None:
    adapter = adapter_cls(metadata=_browser_md([BACK_SCHEMA, GOTO_SCHEMA]))

    assert _reported_feedback(adapter, {"action": "mouse_event"}, caplog) == [
        "invalid action: mouse_event; computer.actions cannot contain mouse_event"
    ]


# =============================================================================
# Case 2 — a valid, ACTIVE extra tool placed inside the action-batch call
# =============================================================================

@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_render_step_survives_misplaced_active_extra_tool(adapter_cls) -> None:
    """Pre-fix: ValueError '… computer.actions cannot contain back' out of
    render_step, even though ``back`` is an offered extra tool."""
    metadata = _browser_md([BACK_SCHEMA, GOTO_SCHEMA, RESPONSE_SCHEMA])
    adapter = adapter_cls(metadata=metadata)
    assert "back" in adapter.active_extra_tool_names()
    sample = _sample(metadata, {"action": "back"})

    step = _render(adapter, sample)

    assert [m["role"] for m in step][-1] == "assistant"
    # NOT re-homed to the top level: the layer confusion stays visible, and the
    # canonical action-batch call stays intact for the env to answer.
    assert sample.messages[1]["tool_calls"] == [
        make_tool_call("computer", {"actions": [{"action": "back"}]})
    ]


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_misplaced_active_extra_tool_is_named_a_standalone_tool(adapter_cls, caplog) -> None:
    """An ACTIVE extra tool reads differently from a hallucination, matching
    ``nested_extra_tool_action_batch_child_message`` on the env side."""
    adapter = adapter_cls(metadata=_browser_md([BACK_SCHEMA, GOTO_SCHEMA]))

    assert _reported_feedback(adapter, {"action": "back"}, caplog) == [
        "invalid action: back; computer.actions cannot contain standalone extra tool back"
    ]


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_inactive_extra_tool_in_action_batch_is_not_called_standalone(
    adapter_cls, caplog
) -> None:
    """``back`` not offered → it is just an unknown name, like any hallucination."""
    adapter = adapter_cls(metadata=_browser_md([GOTO_SCHEMA]))

    assert _reported_feedback(adapter, {"action": "back"}, caplog) == [
        "invalid action: back; computer.actions cannot contain back"
    ]


# =============================================================================
# The structural half of the check is NOT downgraded
# =============================================================================

@pytest.mark.parametrize(
    "arguments,reason",
    [
        ({"actions": "click"}, "computer.arguments.actions must be a list"),
        ({"actions": []}, "computer.arguments.actions must be a non-empty list"),
        ({"actions": ["click"]}, "computer.arguments.actions[0] must be a dict"),
        (
            {"actions": [{}]},
            "computer.arguments.actions[0].action must be a non-empty string",
        ),
    ],
)
def test_unrenderable_action_batch_payload_still_raises(arguments: dict, reason: str) -> None:
    adapter = Qwen3VLDesktopUseAdapter(metadata=_browser_md([BACK_SCHEMA]))
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [make_tool_call("computer", arguments)],
    }

    with pytest.raises(
        ValueError,
        match=re.escape(f"action-batch is not renderable: {reason}"),
    ):
        adapter.convert_message_to_agent(message)


@pytest.mark.parametrize("adapter_cls", _DESKTOP_ADAPTERS)
def test_wellformed_action_batch_reports_no_feedback(adapter_cls, caplog) -> None:
    adapter = adapter_cls(metadata=_browser_md([BACK_SCHEMA]))

    assert _reported_feedback(adapter, {"action": "click", "coordinate": [1, 2]}, caplog) == []


@pytest.mark.parametrize(
    "adapter_key,platform,call,expected",
    [
        (
            "qwen3_vl@desktop@use",
            "desktop",
            make_tool_call(
                "computer",
                {"actions": [{"action": "terminate", "status": "success"}]},
            ),
            "computer.actions cannot contain terminate",
        ),
        (
            "qwen3_vl@mobile@use",
            "mobile",
            make_tool_call(
                "mobile",
                {"actions": [{"action": "open_app", "app_name": "Settings"}]},
            ),
            "mobile.actions cannot contain open_app",
        ),
    ],
)
def test_action_wrappers_do_not_make_non_actions_schema_free(
    adapter_key,
    platform,
    call,
    expected,
    caplog,
) -> None:
    """A non-action nested in an action wrapper is still rejected via feedback."""
    adapter = agent_adapter_for(adapter_key, platform)
    message = {"role": "assistant", "content": [], "tool_calls": [call]}
    child = tool_call_arguments(call)["actions"][0]["action"]

    adapter_base._warn_invalid_action_batch_child.cache_clear()
    with caplog.at_level(logging.WARNING, logger="lite.agents.core.adapter.base"):
        adapter.convert_message_to_agent(message)

    reported = [
        record.getMessage().split(_ENV_REJECTED, 1)[1]
        for record in caplog.records
        if _ENV_REJECTED in record.getMessage()
    ]
    assert reported == [f"invalid action: {child}; {expected}"]
