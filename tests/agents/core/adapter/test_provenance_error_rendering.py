"""Adapter conversion provenance for model-visible env error wording.

These tests cover the source->canonical labels that can otherwise get lost when
an adapter rewrites a model-family action into the Lite/env surface:

* ``mobile_use(open)`` -> ``open_app``
* ``mobile_use(click)`` -> ``tap``
* ``computer_use(left_click)`` -> ``click``
* ``answer`` / StepGUI ``INFO`` -> ``response``

The assertion boundary is intentionally the rendered role:tool text, not only
the converted ``LiteToolCall``: this is the string the next model turn sees.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/adapter/test_provenance_error_rendering.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

from typing import Any

import pytest

from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.agents.core.agent.utils.provenance import (
    ProviderCallProvenance,
    provider_call_provenance_from_merge,
)
from lite.agents.models.mai_ui.adapter import MAIUIMobileUseAdapter
from lite.agents.models.qwen3_5.adapter import (
    Qwen3_5DesktopUseAdapter,
    Qwen3_5MobileUseAdapter,
)
from lite.agents.models.qwen3_vl.adapter import (
    Qwen3VLDesktopUseAdapter,
    Qwen3VLMobileUseAdapter,
)
from lite.agents.models.step_gui.adapter import STEPGUIMobileUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.action_space.batches import (
    merge_adjacent_lite_action_batches_with_provenance,
)
from lite.core.tools.calls import (
    stamp_tool_call_list_ids,
    tool_call_arguments,
    tool_call_name,
)
from lite.gym.types import LiteEnvStepResult
from lite.gym.utils.feedback.errors import append_feedback, current_feedback
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)


def _schema(name: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return make_tool_schema(
        name,
        description=f"{name} tool",
        parameters={
            "type": "object",
            "properties": props,
            "required": required,
        },
    )


OPEN_APP_SCHEMA = _schema("open_app", {"app_name": {"type": "string"}}, ["app_name"])
RESPONSE_SCHEMA = _schema("response", {"text": {"type": "string"}}, ["text"])
LAX_OPEN_APP_SCHEMA = _schema("open_app", {"app_name": {}}, ["app_name"])
LAX_RESPONSE_SCHEMA = _schema("response", {"text": {}}, ["text"])


def _md(
    platform: str,
    *,
    extra_tool_schemas: list[dict[str, Any]] | None = None,
    valid_actions: list[str] | None = None,
) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(platform, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tool_schemas or [],
        valid_actions=valid_actions,
    )


def _assistant_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [],
        "tool_calls": [{"name": tool_name, "arguments": arguments}],
    }


def _stamp(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stamp_tool_call_list_ids(calls, preserve=False)
    return calls


def _render_error(calls: list[dict[str, Any]], metadata: LiteCUAMetadata) -> str:
    """Run shared env ingress feedback through model-visible role:tool text."""
    ordered = ordered_tool_call_ids(calls)
    actions, feedback = prepare_env_tool_calls(
        calls,
        metadata,
        validate_top_level_action=True,
    )
    # A rejected batch CHILD keeps its slot and carries its reason on the action
    # (R4), so it never reaches ``feedback``. Every env replays it onto the call
    # id; do the same here so this file keeps testing WORDING rather than where
    # the string happens to live.
    feedback = dict(feedback)
    for action, result_call_id in actions:
        reason = action.get("_rejected_reason")
        assert reason, f"unrejected action reached the env: {action['name']}"
        if result_call_id:
            append_feedback(feedback, result_call_id, current_feedback(reason))
    assert set(feedback) == {call_id for call_id in ordered if call_id}

    step = build_tool_results_from_decisions(
        LiteEnvStepResult(),
        ordered_call_ids=ordered,
        continue_call_ids=ordered,
        images=[b"png"],
        text="current observation",
        metadata={"screen": "after"},
        feedback=feedback,
    )
    assert len(step.results) == 1
    result = step.results[0]
    msg = build_tool_result_message(
        result.tool_call_id,
        (1,) if result.images else (),
        result.text,
        result.metadata,
        error=result.error,
    )
    text_parts = [part["text"] for part in msg["content"] if part.get("type") == "text"]
    assert len(text_parts) == 1
    return text_parts[0]


@pytest.mark.parametrize(
    "adapter_cls",
    [Qwen3VLMobileUseAdapter, Qwen3_5MobileUseAdapter],
)
def test_mobile_open_error_renders_open_app_after_adapter_conversion(adapter_cls) -> None:
    adapter_metadata = _md("mobile", extra_tool_schemas=[LAX_OPEN_APP_SCHEMA])
    render_metadata = _md("mobile", extra_tool_schemas=[OPEN_APP_SCHEMA])
    adapter = adapter_cls(metadata=adapter_metadata)

    lite = adapter.convert_message_from_agent(
        _assistant_call("mobile_use", {"action": "open", "text": 123})
    )

    calls = _stamp(lite["tool_calls"])
    assert [(tool_call_name(call), tool_call_arguments(call)) for call in calls] == [
        ("open_app", {"app_name": 123})
    ]
    text = _render_error(calls, render_metadata)
    assert text == (
        "current observation\n\n"
        "## Error from previous action:\n"
        "invalid arguments for open_app: "
        "open_app.arguments.app_name must be a string"
    )


@pytest.mark.parametrize(
    "adapter_cls",
    [Qwen3VLMobileUseAdapter, Qwen3_5MobileUseAdapter],
)
def test_mobile_click_error_renders_tap_after_adapter_conversion(adapter_cls) -> None:
    metadata = _md("mobile", valid_actions=["swipe"])
    adapter = adapter_cls(metadata=metadata)

    lite = adapter.convert_message_from_agent(
        _assistant_call("mobile_use", {"action": "click", "coordinate": [10, 20]})
    )

    calls = _stamp(lite["tool_calls"])
    assert tool_call_name(calls[0]) == "mobile"
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "tap", "coordinate": [10, 20], "clicks": 1}
    ]
    text = _render_error(calls, metadata)
    assert text.endswith(
        "## Error from previous action:\n"
        "invalid action: tap; choose an available action for this task"
    )


@pytest.mark.parametrize(
    "adapter_cls",
    [Qwen3VLDesktopUseAdapter, Qwen3_5DesktopUseAdapter],
)
def test_desktop_left_click_error_renders_click_after_adapter_conversion(adapter_cls) -> None:
    metadata = _md("desktop", valid_actions=["type"])
    adapter = adapter_cls(metadata=metadata)

    lite = adapter.convert_message_from_agent(
        _assistant_call(
            "computer_use",
            {"action": "left_click", "coordinate": [30, 40]},
        )
    )

    calls = _stamp(lite["tool_calls"])
    assert tool_call_name(calls[0]) == "computer"
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "click", "coordinate": [30, 40]}
    ]
    text = _render_error(calls, metadata)
    assert text.endswith(
        "## Error from previous action:\n"
        "invalid action: click; choose an available action for this task"
    )


@pytest.mark.parametrize(
    "adapter,agent_message",
    [
        pytest.param(
            Qwen3VLMobileUseAdapter(
                metadata=_md("mobile", extra_tool_schemas=[LAX_RESPONSE_SCHEMA])
            ),
            _assistant_call("mobile_use", {"action": "answer", "text": 123}),
            id="qwen3_vl-answer",
        ),
        pytest.param(
            Qwen3_5MobileUseAdapter(
                metadata=_md("mobile", extra_tool_schemas=[LAX_RESPONSE_SCHEMA])
            ),
            _assistant_call("mobile_use", {"action": "answer", "text": 123}),
            id="qwen3_5-answer",
        ),
        pytest.param(
            MAIUIMobileUseAdapter(
                metadata=_md("mobile", extra_tool_schemas=[LAX_RESPONSE_SCHEMA])
            ),
            _assistant_call("mobile_use", {"action": "answer", "text": 123}),
            id="mai_ui-answer",
        ),
        pytest.param(
            STEPGUIMobileUseAdapter(
                metadata=_md("mobile", extra_tool_schemas=[LAX_RESPONSE_SCHEMA])
            ),
            _assistant_call("mobile_use", {"action": "INFO", "value": 123}),
            id="step_gui-INFO",
        ),
    ],
)
def test_answer_and_info_error_render_response_after_adapter_conversion(
    adapter,
    agent_message: dict[str, Any],
) -> None:
    render_metadata = _md("mobile", extra_tool_schemas=[RESPONSE_SCHEMA])

    lite = adapter.convert_message_from_agent(agent_message)

    calls = _stamp(lite["tool_calls"])
    assert [(tool_call_name(call), tool_call_arguments(call)) for call in calls] == [
        ("response", {"text": 123})
    ]
    text = _render_error(calls, render_metadata)
    assert text.endswith(
        "## Error from previous action:\n"
        "invalid arguments for response: response.arguments.text must be a string"
    )


def test_invalid_action_batch_child_wording_names_the_child_action() -> None:
    metadata = _md("mobile")
    calls = _stamp([
        make_tool_call(
            "mobile",
            {"actions": [{"action": "open_app", "app_name": "Clock"}]},
        )
    ])

    text = _render_error(calls, metadata)

    assert text.endswith(
        "## Error from previous action:\n"
        "invalid action: open_app; mobile.actions cannot contain open_app"
    )


def test_provider_merge_provenance_keeps_none_for_unparsed_provider_calls() -> None:
    draft = [make_tool_call("computer", {"actions": [{"action": "click"}]})]
    merge = merge_adjacent_lite_action_batches_with_provenance(draft)
    _stamp(merge.tool_calls)

    provenance = provider_call_provenance_from_merge(
        [None, [0]],
        merge,
    )

    assert provenance == (
        ProviderCallProvenance(canonical_call_id=None, is_final_for_canonical=False),
        ProviderCallProvenance(canonical_call_id="call_0000", is_final_for_canonical=True),
    )


def test_provider_merge_provenance_marks_only_last_provider_call_final() -> None:
    draft = [
        make_tool_call("computer", {"actions": [{"action": "click"}]}),
        make_tool_call("computer", {"actions": [{"action": "type", "text": "ok"}]}),
    ]
    merge = merge_adjacent_lite_action_batches_with_provenance(draft)
    _stamp(merge.tool_calls)

    provenance = provider_call_provenance_from_merge([[0], [1]], merge)

    assert provenance == (
        ProviderCallProvenance(canonical_call_id="call_0000", is_final_for_canonical=False),
        ProviderCallProvenance(canonical_call_id="call_0000", is_final_for_canonical=True),
    )


def test_provider_merge_provenance_rejects_unstamped_canonical_call() -> None:
    draft = [make_tool_call("computer", {"actions": [{"action": "click"}]})]
    merge = merge_adjacent_lite_action_batches_with_provenance(draft)

    with pytest.raises(ValueError, match="no canonical id"):
        provider_call_provenance_from_merge([[0]], merge)
