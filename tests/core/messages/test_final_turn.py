"""Unit tests for neutral final-turn helpers."""

from __future__ import annotations

import pytest

import lite.core.messages as core_messages
import lite.core.messages.final as final_messages
import lite.core.samples as core_samples
from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    CONTENT_ONLY_FINAL_REASON,
    CONTENT_ONLY_FINAL_TEXT,
    EMPTY_FINAL_REASON,
    MODEL_OUTPUT_ERROR_KEY,
    PARSE_FAILURE_FINAL_REASON,
    canonicalize_no_tool_call_final_message,
    is_canonical_no_tool_call_final_message,
    make_no_tool_call_final_actions,
    no_tool_call_final_content,
    no_tool_call_final_text,
    summarize_no_tool_call_final,
)
from lite.core.tools.calls import RUNTIME_INTERNAL_STOP_REASON_KEY


@pytest.mark.parametrize(
    "name",
    # TERMINAL_MODEL_OUTPUT_STOP_REASONS was deleted by N6.1: it was a singleton of the
    # same value as PERSISTED_FINAL_STOP_REASONS, so no test could tell them apart.
    ["PERSISTED_FINAL_STOP_REASONS"],
)
def test_final_publication_policy_is_not_core_message_api(name):
    """Which stop reasons get published is a property of the durable record.

    ``lite.core.messages.final`` owns the stop-reason *vocabulary*; the two
    partitions that decide what survives into the durable record belong to the
    durable containers in ``lite.core.samples`` and must not leak back into the
    message API.
    """
    assert not hasattr(final_messages, name)
    assert name not in final_messages.__all__
    assert not hasattr(core_messages, name)
    assert name not in core_messages.__all__
    assert hasattr(core_samples, name)
    assert name in core_samples.__all__


def test_only_exceptional_no_tool_final_stop_reasons_are_persisted():
    assert core_samples.PERSISTED_FINAL_STOP_REASONS == frozenset({
        PARSE_FAILURE_FINAL_REASON,
    })
    assert final_messages.CONTENT_ONLY_FINAL_REASON not in (
        core_samples.PERSISTED_FINAL_STOP_REASONS
    )
    assert final_messages.EMPTY_FINAL_REASON not in (
        core_samples.PERSISTED_FINAL_STOP_REASONS
    )


def test_stop_reason_vocabulary_has_one_owner():
    """Every spelling an episode can end with resolves through one module."""
    assert final_messages.STOP_REASON_INFO_KEY == "stop_reason"
    assert final_messages.INTERNAL_STOP_REASONS == frozenset({
        final_messages.LOOP_DETECT_TERMINATE_REASON,
        final_messages.CONTENT_ONLY_FINAL_REASON,
        final_messages.ENV_INTERNAL_TERMINATE_REASON,
    })
    assert final_messages.MAX_STEPS_STOP_REASON == "max_steps"


def test_content_only_final_actions_build_unpersisted_internal_response() -> None:
    actions = make_no_tool_call_final_actions("final answer")

    assert actions == [
        {
            "type": "function",
            "function": {"name": "response", "arguments": {"text": "final answer"}},
            RUNTIME_INTERNAL_STOP_REASON_KEY: CONTENT_ONLY_FINAL_REASON,
        }
    ]
    assert "id" not in actions[0]


@pytest.mark.parametrize(
    ("message", "stop_reason", "visible_text"),
    [
        ({"content": []}, EMPTY_FINAL_REASON, ""),
        (
            {"content": [{"type": "inline_reasoning", "text": "thinking"}]},
            EMPTY_FINAL_REASON,
            "",
        ),
        (
            {"content": [{"type": "action_description", "text": "Done."}]},
            EMPTY_FINAL_REASON,
            "",
        ),
        (
            {"content": [{"type": "text", "text": "Done."}]},
            CONTENT_ONLY_FINAL_REASON,
            "Done.",
        ),
        (
            {"content": [], MODEL_OUTPUT_ERROR_KEY: "bad json"},
            PARSE_FAILURE_FINAL_REASON,
            "",
        ),
    ],
)
def test_classifier_source_kind_is_the_durable_stop_reason(
    message, stop_reason, visible_text
):
    """The diagnostic answers "why no tool call?" in the stop-reason vocabulary.

    Invisible channels (``inline_reasoning``, ``action_description``) leave the
    turn textless, so they classify like any other textless turn; the part counts
    are what say which channel was used.
    """
    assert no_tool_call_final_text(message) == visible_text
    diagnostic = summarize_no_tool_call_final(message).diagnostic()
    assert diagnostic["stop_reason"] == stop_reason
    assert diagnostic["visible_text"] is bool(visible_text)


def test_summary_is_single_source_for_text_and_diagnostic():
    message = {
        "content": [
            {"type": "inline_reasoning", "text": "thinking"},
            {"type": "text", "text": " final answer "},
            {"type": "metadata", "data": {"debug": True}},
            "opaque",
        ],
        "raw_response": {"id": "resp_1"},
    }

    summary = summarize_no_tool_call_final(message)
    diagnostic = summary.diagnostic()

    assert summary.visible_text == "final answer"
    assert no_tool_call_final_text(message) == summary.visible_text
    assert diagnostic["stop_reason"] == summary.stop_reason == CONTENT_ONLY_FINAL_REASON
    assert diagnostic["content_types"] == [
        "inline_reasoning",
        "text",
        "metadata",
        "str",
    ]
    assert diagnostic["text_part_count"] == 1
    assert diagnostic["inline_reasoning_part_count"] == 1
    assert diagnostic["has_raw_response"] is True


def test_no_tool_text_final_canonicalizes_without_replay_sidecars():
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "The answer is 42."}],
        "raw_response": {"adapter_key": "x", "text": "The answer is 42."},
    }

    canonical = canonicalize_no_tool_call_final_message(message)

    assert canonical["content"] == no_tool_call_final_content()
    assert is_canonical_no_tool_call_final_message(canonical)
    assert "raw_response" not in canonical
    diagnostic = canonical[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]
    assert diagnostic["stop_reason"] == CONTENT_ONLY_FINAL_REASON
    assert diagnostic["visible_text"] is True


def test_parse_failure_final_diagnostic_is_durable_but_transient_key_is_not():
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Action: <tool_call>{bad json}</tool_call>"}],
        MODEL_OUTPUT_ERROR_KEY: "malformed <tool_call> JSON",
    }

    canonical = canonicalize_no_tool_call_final_message(message)

    assert canonical["content"] == [{"type": "text", "text": CONTENT_ONLY_FINAL_TEXT}]
    assert MODEL_OUTPUT_ERROR_KEY not in canonical
    diagnostic = canonical[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]
    assert diagnostic["stop_reason"] == PARSE_FAILURE_FINAL_REASON
    assert diagnostic["has_model_output_error"] is True


def test_action_description_only_final_stays_recoverable_from_part_counts():
    """An action-description-only final classifies as any other textless turn; the
    part counts are what distinguish it."""
    message = {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "click Save"}],
    }

    canonical = canonicalize_no_tool_call_final_message(message)

    assert no_tool_call_final_text(message) == ""
    assert canonical["content"] == [{"type": "text", "text": CONTENT_ONLY_FINAL_TEXT}]
    diagnostic = canonical[CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]
    assert diagnostic["stop_reason"] == EMPTY_FINAL_REASON
    assert diagnostic["action_description_part_count"] == 1
    assert diagnostic["text_part_count"] == 0
    assert diagnostic["inline_reasoning_part_count"] == 0
    assert diagnostic["history_summary_part_count"] == 0
    assert diagnostic["has_native_reasoning"] is False
