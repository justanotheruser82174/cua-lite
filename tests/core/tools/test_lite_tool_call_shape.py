"""Tests for the canonical nested Lite tool-call shape owner.

Run:
    uv run pytest tests/core/tools/test_lite_tool_call_shape.py -q
"""

from __future__ import annotations

import pytest

import lite.core.tools.calls as core_calls
import lite.core.tools.results as core_results
from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    make_tool_call,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
    validate_lite_tool_call,
)

#: Names that must not reappear on the canonical call owner or the tools facade.
#: The unprefixed annotation spellings are retired because they read as canonical
#: ``LiteToolCall`` fields; the runtime layer spells them ``RUNTIME_*`` instead.
#: ``_RuntimeToolCall`` is retired because a dict subclass hiding the sidecars
#: from iteration gave one logical action two answers -- see
#: :class:`lite.core.tools.calls.RuntimeEnvAction`.
RETIRED_CALL_HELPERS = frozenset(
    {
        "INTERNAL_STOP_REASON_KEY",
        "_RuntimeToolCall",
        "LITE_TOOL_CALL_KEYS",
        "RESULT_CALL_ID_KEY",
        "internal_stop_reason",
        "stamp_tool_call_ids",
        "tool_call_function",
        "with_internal_stop_reason",
    }
)

#: Runtime env-ingress annotations. They live in the calls module because they
#: describe tool-call-shaped runtime actions, but they are not canonical shape,
#: so they stay out of ``__all__`` and off the ``lite.core.tools`` facade.
RUNTIME_ANNOTATION_NAMES = frozenset(
    {
        "RUNTIME_INTERNAL_STOP_REASON_KEY",
        "RUNTIME_RESULT_CALL_ID_KEY",
        "RUNTIME_SIDECAR_KEYS",
        "RuntimeEnvAction",
        "canonical_lite_tool_call",
        "runtime_internal_stop_reason",
        "with_runtime_internal_stop_reason",
    }
)


def test_calls_owner_exports_only_the_canonical_call_surface() -> None:
    """Exhaustive by design: a new export must be decided, not drift in.

    ``EnvAction`` is the one non-envelope name on this surface, and it is here
    on purpose: it is the single spelling of the BARE ``{"name", "arguments"}``
    layer that action-batch children and post-ingress env action lists share,
    so the alternative is one alias per env module. Being importable from the
    calls owner does not promote it to the ``lite.core.tools`` facade, which
    carries canonical shape only.
    """
    import lite.core.tools as core_tools

    assert not hasattr(core_tools, "EnvAction")
    assert set(core_calls.__all__) == {
        "CANONICAL_TOOL_CALL_ERROR",
        "EnvAction",
        "LiteToolCall",
        "make_tool_call",
        "validate_lite_tool_call",
        "stamp_message_tool_call_ids",
        "stamp_messages_tool_call_ids",
        "stamp_tool_call_list_ids",
        "tool_call_arguments",
        "tool_call_id",
        "tool_call_name",
    }


def test_retired_call_helper_names_stay_absent() -> None:
    import lite.core.tools as core_tools

    for name in RETIRED_CALL_HELPERS:
        assert name not in core_calls.__all__
        assert not hasattr(core_calls, name)
        assert not hasattr(core_tools, name)


def test_runtime_annotations_are_not_canonical_call_surface() -> None:
    """Runtime env annotations are reachable, but never through the call contract."""
    import lite.core.tools as core_tools

    for name in RUNTIME_ANNOTATION_NAMES:
        assert getattr(core_calls, name)
        assert name not in core_calls.__all__
        assert not hasattr(core_tools, name)


def test_results_owner_exports_only_carrier_and_projection_surface() -> None:
    assert set(core_results.__all__) == {
        "LiteToolResult",
        "TOOL_RESULT_ERROR_SECTION_HEADER",
        "extract_projected_tool_result_error",
        "make_tool_result",
        "project_tool_result_text",
        "text_has_projected_tool_result_error",
    }
    assert not hasattr(core_results, "tool_result_metadata")
    assert not hasattr(core_results, "result_images_for_next_decision")
    assert not hasattr(core_results, "result_image_indices_for_next_decision")


def test_make_tool_call_uses_section_2_6_key_order() -> None:
    call = make_tool_call("goto", {"url": "x"}, call_id="call_0000")

    assert list(call) == ["id", "type", "function"]
    assert list(call["function"]) == ["name", "arguments"]
    assert call == {
        "id": "call_0000",
        "type": "function",
        "function": {"name": "goto", "arguments": {"url": "x"}},
    }


def test_tool_call_accessors_read_the_nested_canonical_owner_shape() -> None:
    arguments = {"url": "x"}
    call = make_tool_call("goto", arguments, call_id="call_0000")

    assert tool_call_id(call) == "call_0000"
    assert tool_call_name(call) == "goto"
    assert tool_call_arguments(call) is call["function"]["arguments"]
    assert tool_call_arguments(call) == arguments
    assert "name" not in call
    assert "arguments" not in call


def test_accepts_a_canonical_stamped_call() -> None:
    call = make_tool_call("goto", {"url": "x"}, call_id="call_0000")
    assert validate_lite_tool_call(call, "where", require_id=True) is None


def test_accepts_a_pre_stamp_call_only_when_id_is_not_required() -> None:
    """``id`` is the post-stamp pairing key, not part of the pre-stamp shape."""
    call = make_tool_call("goto", {"url": "x"})

    assert validate_lite_tool_call(call, "where", require_id=False) is None
    assert (
        validate_lite_tool_call(call, "where", require_id=True)
        == "where missing non-empty id"
    )


@pytest.mark.parametrize(
    "call,expected_fragment",
    [
        ("not-a-dict", "where must be a dict"),
        (
            {
                "id": "c",
                "type": "function",
                "function": {"name": "back", "arguments": {}},
                "name": "back",
            },
            "where has noncanonical outer keys ['name']",
        ),
        (
            {
                "tool_call_id": "c",
                "type": "function",
                "function": {"name": "back", "arguments": {}},
            },
            "where must use id on tool calls, not tool_call_id",
        ),
        (
            {"id": "", "type": "function", "function": {"name": "back", "arguments": {}}},
            "where missing non-empty id",
        ),
        (
            {"id": "c", "function": {"name": "back", "arguments": {}}},
            "where.type must be 'function'",
        ),
        ({"id": "c", "type": "function"}, "where.function must be an object"),
        (
            {"id": "c", "type": "function", "function": {"arguments": {}}},
            "where.function.name must be a non-empty string",
        ),
        (
            {"id": "c", "type": "function", "function": {"name": "", "arguments": {}}},
            "where.function.name must be a non-empty string",
        ),
        (
            {"id": "c", "type": "function", "function": {"name": "back"}},
            "where.function missing arguments dict",
        ),
        (
            {
                "id": "c",
                "type": "function",
                "function": {"name": "back", "arguments": None},
            },
            "where.function missing arguments dict",
        ),
        (
            {
                "id": "c",
                "type": "function",
                "function": {"name": "back", "arguments": "null"},
            },
            "where.function.arguments must be a dict per §1.4, not a JSON string",
        ),
    ],
)
def test_rejects_noncanonical_calls_with_a_positional_message(
    call,
    expected_fragment,
) -> None:
    reason = validate_lite_tool_call(call, "where", require_id=True)

    assert reason is not None
    assert reason.startswith("where")
    assert expected_fragment in reason


def test_bare_function_call_is_agent_wire_not_canonical_call_shape() -> None:
    call = {"name": "goto", "arguments": {"url": "x"}}

    reason = validate_lite_tool_call(call, "where", require_id=False)
    assert reason == (
        "where is a bare model-function projection, not a canonical Lite tool call"
    )


def test_provider_function_call_payload_is_not_canonical_call_shape() -> None:
    call = {
        "type": "function_call",
        "call_id": "call_0000",
        "name": "goto",
        "arguments": {"url": "x"},
    }

    reason = validate_lite_tool_call(call, "where", require_id=False)

    assert reason == (
        "where is a provider function-call payload, not a canonical Lite tool call"
    )


def test_outer_keys_are_reported_before_missing_inner_fields() -> None:
    """A malformed outer shape reads as outer-key debt, not as a missing inner field.

    A checker that tested inner fields first would tell a dataset author to add
    ``arguments`` while leaving an invalid outer representation intact.
    """
    call = {
        "id": "call_0000",
        "type": "function",
        "function": {"name": "back"},
        "arguments": {},
    }

    reason = validate_lite_tool_call(call, "where", require_id=True)

    assert reason == "where has noncanonical outer keys ['arguments']"


def test_rejects_internal_annotation_on_nested_canonical_call() -> None:
    """The transient internal side channel must never become persisted shape.

    The assertion deliberately pins only rejection, not the first error: this
    should fail whether the validator reports the transient annotation or
    another canonical-shape violation first.
    """
    call = {
        "id": "call_0000",
        "type": "function",
        "function": {"name": "response", "arguments": {"text": "done"}},
        RUNTIME_INTERNAL_STOP_REASON_KEY: "CONTENT_ONLY_FINAL",
    }

    assert validate_lite_tool_call(call, "where", require_id=True) is not None
