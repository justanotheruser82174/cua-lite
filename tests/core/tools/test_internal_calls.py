"""Tests for internal runtime env-action annotations.

Run:
    uv run pytest tests/core/tools/test_internal_calls.py -q
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    RUNTIME_SIDECAR_KEYS,
    LiteToolCall,
    RuntimeEnvAction,
    canonical_lite_tool_call,
    make_tool_call,
    runtime_internal_stop_reason,
    validate_lite_tool_call,
    with_runtime_internal_stop_reason,
)
from lite.core.tools.extra_tools import is_internal_finish_tool_call
from lite.core.tools.results import LiteToolResult, make_tool_result


def test_internal_stop_reason_round_trips_on_runtime_action() -> None:
    action = make_tool_call("terminate", {"status": "success"})

    assert runtime_internal_stop_reason(action) is None
    stamped = with_runtime_internal_stop_reason(action, "REPETITIVE_LOOP")

    assert stamped is not action
    assert RUNTIME_INTERNAL_STOP_REASON_KEY not in action
    assert stamped[RUNTIME_INTERNAL_STOP_REASON_KEY] == "REPETITIVE_LOOP"
    assert runtime_internal_stop_reason(stamped) == "REPETITIVE_LOOP"


def test_runtime_sidecars_are_ordinary_keys_that_survive_json_transport() -> None:
    """The remote env-server reads the sidecars off ``/step`` JSON."""
    stamped = with_runtime_internal_stop_reason(
        make_tool_call("terminate", {"status": "success"}),
        "REPETITIVE_LOOP",
    )

    round_tripped = json.loads(json.dumps(stamped))

    assert round_tripped == stamped
    assert runtime_internal_stop_reason(round_tripped) == "REPETITIVE_LOOP"


def test_runtime_sidecar_keys_are_derived_from_the_runtime_envelope() -> None:
    assert RUNTIME_SIDECAR_KEYS == {
        RUNTIME_INTERNAL_STOP_REASON_KEY,
        "_result_call_id",

        "_rejected_reason",
    }
    assert RUNTIME_SIDECAR_KEYS.isdisjoint(
        LiteToolCall.__required_keys__ | LiteToolCall.__optional_keys__
    )
    assert RUNTIME_SIDECAR_KEYS < RuntimeEnvAction.__optional_keys__


def test_runtime_action_is_not_a_canonical_call_but_projects_to_one() -> None:
    """One shape, one answer: the sidecar keys are visible to the validator."""
    call = make_tool_call("terminate", {"status": "success"})
    stamped = with_runtime_internal_stop_reason(call, "REPETITIVE_LOOP")

    assert validate_lite_tool_call(stamped, "action", require_id=False) is not None
    assert validate_lite_tool_call(
        canonical_lite_tool_call(stamped), "action", require_id=False
    ) is None
    assert canonical_lite_tool_call(stamped) == call
    assert canonical_lite_tool_call(call) is call


@pytest.mark.parametrize("reason", [None, True, 3, ["content_only_final"]])
def test_internal_stop_reason_ignores_non_string_annotations(reason: object) -> None:
    action = with_runtime_internal_stop_reason(
        make_tool_call("response", {"text": "done"}),
        "content_only_final",
    )
    action[RUNTIME_INTERNAL_STOP_REASON_KEY] = reason

    assert runtime_internal_stop_reason(action) is None


def test_internal_stop_reason_reads_the_same_key_any_producer_writes() -> None:
    """A plain dict literal and the stamper are one shape, not two."""
    action = {
        **make_tool_call("response", {"text": "done"}),
        RUNTIME_INTERNAL_STOP_REASON_KEY: "content_only_final",
    }

    assert runtime_internal_stop_reason(action) == "content_only_final"
    assert action == with_runtime_internal_stop_reason(
        make_tool_call("response", {"text": "done"}),
        "content_only_final",
    )


def test_internal_finish_is_recognized_by_the_runtime_sidecar_not_the_name() -> None:
    """A content-only final reaches env.step as an internal finish, not model output."""
    model_finish = make_tool_call("response", {"text": "done"}, call_id="call_0000")
    internal_finish = with_runtime_internal_stop_reason(
        make_tool_call("response", {"text": "done"}),
        "content_only_final",
    )

    assert is_internal_finish_tool_call(internal_finish)
    # Same name, no sidecar: an ordinary model finish call.
    assert not is_internal_finish_tool_call(model_finish)
    # Same sidecar, but a persisted id means the model emitted it.
    assert not is_internal_finish_tool_call({**internal_finish, "id": "call_0000"})
    # Sidecar on a non-finish name is not a finish.
    assert not is_internal_finish_tool_call(
        with_runtime_internal_stop_reason(make_tool_call("goto", {"url": "x"}), "REPETITIVE_LOOP")
    )
    # A stop reason outside the runtime vocabulary is not an internal finish.
    assert not is_internal_finish_tool_call(
        with_runtime_internal_stop_reason(
            make_tool_call("terminate", {"status": "success"}), "max_steps"
        )
    )


def test_lite_tool_result_owner_uses_tool_call_id_pairing_field() -> None:
    result = make_tool_result("call_0000", text="ok")
    field_names = {field.name for field in fields(LiteToolResult)}

    assert result.tool_call_id == "call_0000"
    assert "tool_call_id" in field_names
    assert "call_id" not in field_names
    assert not hasattr(result, "call_id")


def test_make_tool_result_uses_images_as_the_only_image_carrier() -> None:
    images = [b"first-shot"]

    result = make_tool_result("call_0000", images=images)
    images.append(b"later-shot")
    field_names = {field.name for field in fields(LiteToolResult)}

    assert result.images == [b"first-shot"]
    assert "images" in field_names
    assert "image" not in field_names
    assert not hasattr(result, "image")


def test_lite_tool_result_rejects_old_image_constructor_field() -> None:
    with pytest.raises(TypeError, match="image"):
        LiteToolResult("call_0000", image=b"shot")  # type: ignore[call-arg]


def test_make_tool_result_derives_error_metadata_without_mutating_input() -> None:
    metadata = {"source": "env"}

    result = make_tool_result("call_0000", metadata=metadata, error="bad action")

    assert result.metadata == {"source": "env", "is_error": True}
    assert metadata == {"source": "env"}
    assert result.error == "bad action"
