"""``normalize_content_only_final`` -- the collect-side half of the final-turn contract.

Run: uv run pytest tests/data/utils/test_content_only_final_policy.py

POLICY. A final assistant turn with no ``tool_calls`` has its content replaced
wholesale by ``structural_final_message()`` -- ``[{"type": "text", "text": "Done."}]``
-- regardless of what it held. **Unconditional, no opt-out.**

The reason it must not branch on the old content: any rule of the form "keep the
prose when there is prose" re-creates the downstream question *does this turn have a
trainable target?*, which is exactly the empty-SFT-target bug. One branch removed,
one bug family removed.

``action_description`` is by definition the "narration accompanying an action"
channel, so a turn with no action has nothing to narrate and the field is invalid
there -- the reciprocal contract is already stated in
``lite.core.messages.no_tool_call_final_text``'s docstring. This function
is how the ``devs/data/**`` collect filters honour it; the ``lite/data/preproc``
builders already append ``structural_final_message`` directly.
"""

from __future__ import annotations

import pytest

from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    CONTENT_ONLY_FINAL_REASON,
    EMPTY_FINAL_REASON,
)
from lite.core.tools import make_tool_call
from lite.data.utils.messages import (
    normalize_content_only_final,
    structural_final_message,
)

CANONICAL = structural_final_message()["content"]


def _final(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return [{"role": "user", "content": [{"type": "text", "text": "go"}]}, msg]


@pytest.mark.parametrize(
    ("content", "stop_reason"),
    [
        pytest.param(
            [{"type": "action_description", "text": "clicked"}],
            EMPTY_FINAL_REASON,
            id="action_description",
        ),
        pytest.param(
            [{"type": "inline_reasoning", "text": "thinking"}],
            EMPTY_FINAL_REASON,
            id="inline_reasoning",
        ),
        pytest.param(
            [
                {"type": "inline_reasoning", "text": "t"},
                {"type": "action_description", "text": "c"},
            ],
            EMPTY_FINAL_REASON,
            id="both",
        ),
        pytest.param(
            [{"type": "text", "text": "I found 3 items."}],
            CONTENT_ONLY_FINAL_REASON,
            id="other_prose",
        ),
        pytest.param(
            [{"type": "history_summary", "text": "..."}],
            EMPTY_FINAL_REASON,
            id="history_summary",
        ),
        pytest.param([], EMPTY_FINAL_REASON, id="empty"),
    ],
)
def test_every_shape_becomes_the_canonical_text_final(content, stop_reason):
    out, changed = normalize_content_only_final(_final(content))
    assert changed is True
    assert out[-1]["content"] == CANONICAL
    diagnostic = out[-1][CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]
    assert diagnostic["stop_reason"] == stop_reason
    assert diagnostic["visible_text"] is (stop_reason == CONTENT_ONLY_FINAL_REASON)


def test_already_canonical_is_a_no_op():
    out, changed = normalize_content_only_final(_final(list(CANONICAL)))
    assert changed is False
    assert out[-1]["content"] == CANONICAL
    assert CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY not in out[-1]


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"tool_calls": []}, id="empty_tool_calls"),
        pytest.param({"raw_response": {"adapter_key": "x", "text": ""}}, id="sidecar"),
    ],
)
def test_canonical_content_with_noncanonical_shape_keeps_only_diagnostic(extra):
    messages = _final(list(CANONICAL))
    messages[-1].update(extra)

    out, changed = normalize_content_only_final(messages)

    assert changed is True
    assert out[-1]["content"] == structural_final_message()["content"]
    assert "tool_calls" not in out[-1]
    assert "raw_response" not in out[-1]
    assert out[-1][CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY]["stop_reason"] == (
        CONTENT_ONLY_FINAL_REASON
    )
    assert out[-1] is not messages[-1]


def test_idempotent():
    once, _ = normalize_content_only_final(_final([{"type": "inline_reasoning", "text": "x"}]))
    twice, changed = normalize_content_only_final(once)
    assert changed is False
    assert twice == once


def test_a_final_turn_with_tool_calls_is_untouched():
    """The QA answer of a web task is submitted through the ``response`` TOOL.

    Normalizing it would destroy the graded answer, so a final turn carrying any
    tool_calls is out of scope by construction.
    """
    calls = [make_tool_call("response", {"text": "42"}, call_id="call_0001")]
    messages = _final([{"type": "text", "text": "42"}], tool_calls=calls)
    out, changed = normalize_content_only_final(messages)
    assert changed is False
    assert out[-1]["content"] == [{"type": "text", "text": "42"}]
    assert out[-1]["tool_calls"] == calls


def test_does_not_mutate_its_input():
    messages = _final([{"type": "action_description", "text": "clicked"}])
    snapshot = [dict(m) for m in messages]
    normalize_content_only_final(messages)
    assert messages == snapshot


def test_only_the_last_assistant_turn_is_normalized():
    """An earlier action turn keeps its narration; only the terminal turn changes."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click"}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0001",
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0001",
            "content": [{"type": "text", "text": "ok"}],
        },
        {"role": "assistant", "content": [{"type": "inline_reasoning", "text": "finished"}]},
    ]
    out, changed = normalize_content_only_final(messages)
    assert changed is True
    assert out[1]["content"] == [{"type": "action_description", "text": "click"}]
    assert out[-1]["content"] == CANONICAL


def test_no_assistant_turn_is_a_no_op():
    messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    out, changed = normalize_content_only_final(messages)
    assert changed is False
    assert out == messages
