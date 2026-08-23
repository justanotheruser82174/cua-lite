"""Every ``use`` adapter must render a content-only final turn without losing it.

Run: uv run pytest tests/agents/core/adapter/test_content_only_final_render.py

CONTRACT. A turn with no ``tool_calls`` is the termination signal (see
``lite/agents/core/agent/base.py`` "N3: a turn with no parseable tool call IS the
termination signal"), and every dataset-building path normalizes such a turn to a
single plain ``text`` part via
``lite.data.utils.messages.normalize_content_only_final`` -- canonically
``[{"type": "text", "text": "Done."}]``.

An adapter that drops that part renders an EMPTY SFT target: the row survives every
count-based check (``image_pad`` counts still match, message counts still match) and
trains the model on nothing. This test is the reason such an adapter cannot ship.

WHY IT IS PARAMETRIZED OVER DISCOVERED SUBCLASSES rather than a hand-written list:
when this contract was first measured, roughly half of the ``use`` adapters dropped
the part, and a grep-based guess at which ones named three adapters that all turned
out to be *correct*. Enumerating the live class tree is the only way the list cannot
drift, and it makes a newly added adapter fail here instead of silently in training.

The shape every adapter must implement is Qwen3_5UseAdapter's three branches:
renderable lines -> render them; nothing renderable AND no tool calls -> keep the
original content; nothing renderable but tool calls present -> empty. Only the
middle branch is what this test pins.
"""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter.base import BaseAgentAdapter
from lite.agents.models.mai_ui.adapter import MAIUIMobileUseAdapter
from lite.agents.models.step_gui.adapter import STEPGUIMobileUseAdapter
from lite.agents.models.ui_tars.adapter import (
    UITarsDesktopUseAdapter,
    UITarsMobileUseAdapter,
)
from lite.agents.models.ui_tars_15_v1.adapter import (
    UITars15V1DesktopUseAdapter,
    UITars15V1MobileUseAdapter,
)
from lite.core.messages import no_tool_call_final_text
from lite.core.messages.final import CONTENT_ONLY_FINAL_TEXT, pop_model_output_error
from lite.core.tools import make_tool_call
from lite.data.utils.messages import structural_final_message

register_all()


def _use_adapter_classes() -> list[type]:
    """Every concrete ``use`` adapter reachable from the base class tree."""
    seen: list[type] = []

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            seen.append(sub)
            walk(sub)

    walk(BaseAgentAdapter)
    return sorted({c for c in seen if "Use" in c.__name__}, key=lambda c: c.__name__)


USE_ADAPTERS = _use_adapter_classes()
RAW_PARSER_USE_ADAPTERS = [
    c for c in USE_ADAPTERS if c.__name__ not in {"LiteDesktopUseAdapter", "LiteMobileUseAdapter"}
]
_NO_TOOL_FINAL_TEXT = "The answer is 42."
NO_TOOL_FINAL_TEXTS = (
    CONTENT_ONLY_FINAL_TEXT,
    _NO_TOOL_FINAL_TEXT,
)
TARGETED_PROMPT_PARSER_USE_ADAPTERS = (
    UITarsDesktopUseAdapter,
    UITarsMobileUseAdapter,
    UITars15V1DesktopUseAdapter,
    UITars15V1MobileUseAdapter,
    STEPGUIMobileUseAdapter,
    MAIUIMobileUseAdapter,
)
PROTOCOL_LOOKING_NO_TOOL_FINALS = (
    (UITarsDesktopUseAdapter, "Action: Done."),
    (UITarsMobileUseAdapter, "Thought: finished\nAction: Done."),
    (UITars15V1DesktopUseAdapter, "Action: Done."),
    (UITars15V1MobileUseAdapter, "Thought: finished\nAction: Done."),
    (STEPGUIMobileUseAdapter, "<THINK>Done.</THINK>\naction:DONE"),
)


def test_the_discovery_actually_found_adapters():
    """Guard the guard: a broken walk would make every test below vacuously pass."""
    assert len(USE_ADAPTERS) >= 16, f"only found {len(USE_ADAPTERS)} use adapters"


@pytest.mark.parametrize("adapter_cls", USE_ADAPTERS, ids=lambda c: c.__name__)
def test_content_only_final_text_survives_rendering(adapter_cls):
    """The canonical "Done." final must appear in the rendered agent message."""
    rendered = adapter_cls().convert_message_to_agent(dict(structural_final_message()))
    assert CONTENT_ONLY_FINAL_TEXT in str(rendered), (
        f"{adapter_cls.__name__} dropped the content-only final turn; "
        f"rendered={rendered!r}. This produces an empty SFT target that no "
        f"count-based check can detect."
    )


@pytest.mark.parametrize("adapter_cls", USE_ADAPTERS, ids=lambda c: c.__name__)
def test_content_only_final_is_not_replaced_by_an_empty_skeleton(adapter_cls):
    """Non-empty output carrying no words is worse than empty output.

    A rendering can be non-empty and still say nothing:
    ``"# Step 0:\\n## Thought:\\n\\n\\n## Action:\\n\\n"`` is a template with every
    slot blank. It passes an ``if text:`` check and any non-empty assertion while
    carrying nothing at all, so it is pinned separately.
    """
    rendered = adapter_cls().convert_message_to_agent(dict(structural_final_message()))
    parts = rendered.get("content") if isinstance(rendered, dict) else None
    for part in parts or []:
        if isinstance(part, dict) and part.get("type") == "text":
            stripped = part.get("text", "").replace("#", " ").replace(":", " ")
            assert stripped.split(), (
                f"{adapter_cls.__name__} emitted a blank template skeleton: {part['text']!r}"
            )


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["done", "prose"])
@pytest.mark.parametrize("adapter_cls", RAW_PARSER_USE_ADAPTERS, ids=lambda c: c.__name__)
def test_raw_no_tool_call_text_survives_from_agent_parse(adapter_cls, final_text):
    """Bare no-tool-call prose from the model must remain plain text.

    This is the inference-side half of the content-only final contract. A model
    trained on ``{"type": "text", "text": ...}`` may later emit that prose
    directly. The adapter must not retag it as ``action_description`` /
    ``inline_reasoning`` or replace it with ``[]``; those shapes make
    ``no_tool_call_final_text`` read an empty final answer.
    """
    adapter = adapter_cls()
    parsed = adapter.parse_raw_assistant_response(final_text)
    lite = adapter.convert_message_from_agent(parsed)

    assert not lite.get("tool_calls")
    assert lite.get("content") == [{"type": "text", "text": final_text}]
    assert no_tool_call_final_text(lite) == final_text


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["done", "prose"])
@pytest.mark.parametrize("adapter_cls", USE_ADAPTERS, ids=lambda c: c.__name__)
def test_content_only_final_text_round_trips_through_agent_wire(adapter_cls, final_text):
    """DAgger round-trip must preserve the terminal text shape for every adapter."""
    source = {
        "role": "assistant",
        "content": [{"type": "text", "text": final_text}],
    }

    adapter = adapter_cls()
    rendered = adapter.convert_message_to_agent(source)
    lite = adapter.convert_message_from_agent(rendered)

    assert not lite.get("tool_calls")
    assert lite.get("content") == [{"type": "text", "text": final_text}]
    assert no_tool_call_final_text(lite) == final_text


@pytest.mark.parametrize("final_text", NO_TOOL_FINAL_TEXTS, ids=["done", "prose"])
@pytest.mark.parametrize(
    "adapter_cls", TARGETED_PROMPT_PARSER_USE_ADAPTERS, ids=lambda c: c.__name__
)
def test_prompt_parser_families_keep_no_tool_final_text(adapter_cls, final_text):
    """Prompt-parser wire formats must not hide no-tool finals as narration."""
    adapter = adapter_cls()
    raw = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(final_text))
    wire = adapter.convert_message_from_agent(
        adapter.convert_message_to_agent(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": final_text}],
            }
        )
    )

    for phase, lite_message in (("raw", raw), ("wire", wire)):
        assert not lite_message.get("tool_calls"), (
            f"{adapter_cls.__name__} {phase}: {lite_message!r}"
        )
        assert lite_message.get("content") == [{"type": "text", "text": final_text}]
        assert no_tool_call_final_text(lite_message) == final_text


@pytest.mark.parametrize(
    ("adapter_cls", "raw_text"),
    PROTOCOL_LOOKING_NO_TOOL_FINALS,
    ids=lambda value: value.__name__ if isinstance(value, type) else value.splitlines()[0],
)
def test_protocol_looking_no_tool_final_text_is_not_parse_failure(adapter_cls, raw_text):
    """A terminal answer can resemble the model's text protocol without being a tool."""
    adapter = adapter_cls()
    parsed = adapter.parse_raw_assistant_response(raw_text)
    lite_message = adapter.convert_message_from_agent(parsed)

    assert pop_model_output_error(lite_message) is None
    assert not lite_message.get("tool_calls")
    assert lite_message.get("content") == [{"type": "text", "text": raw_text}]
    assert no_tool_call_final_text(lite_message) == raw_text


@pytest.mark.parametrize("adapter_cls", USE_ADAPTERS, ids=lambda c: c.__name__)
def test_a_turn_with_tool_calls_is_unaffected(adapter_cls):
    """The fix must touch ONLY the no-tool-call branch.

    A turn that carries an action keeps its old behaviour: ``action_description``
    is the narration channel there, and extra plain ``text`` stays dropped.
    """
    mobile = "Mobile" in adapter_cls.__name__
    name = "mobile" if mobile else "computer"
    action = {"action": "tap" if mobile else "click", "coordinate": [10, 20]}
    message = {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "press it"}],
        "tool_calls": [make_tool_call(name, {"actions": [action]}, call_id="call_0001")],
    }
    rendered = adapter_cls().convert_message_to_agent(dict(message))
    assert rendered.get("role") == "assistant"
