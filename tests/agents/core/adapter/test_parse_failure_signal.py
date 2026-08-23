"""Parse-failure signalling for the open-model adapter families.

Two terminal-adjacent situations must stay distinguishable (the shared loop
checks ``model_output_error`` before the clean content-only final stop reason at
``lite/agents/core/agent/base.py``):

  * **Content-only final** — the model deliberately emitted only prose, with no
    tool-call grammar at all. This TERMINATES the episode, so the adapter must
    NOT set ``model_output_error``.
  * **Parse failure** — the family's tool-call MARKER was present but parsing
    yielded zero tool calls. The adapter MUST set ``model_output_error`` via
    :func:`mark_model_output_error` so the zero-call turn is recorded as a
    terminal parse-failure final instead of a clean content-only final.

Per-family markers under test:

  ==================  ================================================
  family              marker
  ==================  ================================================
  qwen2_5_vl          ``<tool_call>`` / ``</tool_call>``
  fara                ``<tool_call>`` / ``</tool_call>`` (inherited)
  ui_tars             ``Action:``
  ui_tars_15_v1       ``Action:``
  ==================  ================================================

Run:
    uv run pytest tests/agents/core/adapter/test_parse_failure_signal.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core.messages import no_tool_call_final_text
from lite.core.messages.final import pop_model_output_error

register_all()

# (adapter key, malformed-grammar response, plain-prose response)
#
# Every ``malformed`` string carries its family's marker but violates the
# grammar; every ``prose`` string is marker-free deliberate final text.
_FAMILIES = [
    pytest.param(
        "qwen2_5_vl@desktop@use",
        "<tool_call>{not valid json,,,}</tool_call>",
        id="qwen2_5_vl",
    ),
    pytest.param(
        "fara@desktop@use",
        "<tool_call>{not valid json,,,}</tool_call>",
        id="fara",
    ),
    pytest.param(
        "ui_tars@desktop@use",
        "Thought: I will click it\nAction: click(start_box=",
        id="ui_tars",
    ),
    pytest.param(
        "ui_tars_15_v1@desktop@use",
        "Thought: I will click it\nAction: click(start_box=",
        id="ui_tars_15_v1",
    ),
]

# Marker-free prose: the same deliberate content-only final for every family.
_PROSE = "The total price shown on the page is 42 dollars, so the task is done."

# Legacy coverage selector for this file's stronger text assertion. The broad
# no-tool-prose-as-``text`` contract is pinned for every use adapter in
# ``test_content_only_final_render.py``; this parse-failure test keeps its older
# narrow assertion shape and focuses on the error marker split.
_TEXT_FINAL_FAMILIES = {"fara@desktop@use"}


def _parse(key: str, response: str) -> tuple[dict, str | None]:
    """Run a family's full parse path, returning (lite_message, error)."""
    adapter = AgentAdapterRegistry.get(key)
    agent_message = adapter.parse_raw_assistant_response(response)
    lite_message = adapter.convert_message_from_agent(agent_message)
    # Same order as the shared loop (``agent/base.py``): lite first, agent second.
    error = pop_model_output_error(lite_message) or pop_model_output_error(agent_message)
    return lite_message, error


@pytest.mark.parametrize("key,malformed", _FAMILIES)
def test_malformed_grammar_sets_model_output_error(key: str, malformed: str):
    """Marker present + nothing parsed → terminal parse-failure marker."""
    lite_message, error = _parse(key, malformed)

    actions = lite_message.get("tool_calls") or []
    assert not actions
    assert error, f"{key}: malformed grammar must set model_output_error"
    # Mirrors the shared loop: ``model_output_error`` wins over the clean
    # content-only-final stop reason for a zero-call turn.
    assert bool(error and not actions)


@pytest.mark.parametrize("key,malformed", _FAMILIES)
def test_plain_prose_stays_a_content_only_final(key: str, malformed: str):
    """No marker anywhere → deliberate content-only final, NOT a parse error."""
    lite_message, error = _parse(key, _PROSE)

    assert error is None, f"{key}: marker-free prose must not set model_output_error"
    assert not lite_message.get("tool_calls")
    if key in _TEXT_FINAL_FAMILIES:
        assert no_tool_call_final_text(lite_message).strip() == _PROSE
