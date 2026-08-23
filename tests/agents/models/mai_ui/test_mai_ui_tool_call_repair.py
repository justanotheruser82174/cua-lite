"""MAI-UI ``<tool_call>`` JSON repair pass.

Run:
    uv run pytest tests/agents/models/mai_ui/test_mai_ui_tool_call_repair.py -q

The repair pass was silently deleted by ``3c1c13efe`` and nothing went red,
because nothing pinned it. MAI-UI closes ``"coordinate":[N,N`` with ``}`` plus
trailing junk instead of ``]}}``; without the repair the whole call parses to
``None`` and the agent drops a click it used to execute.
"""

from __future__ import annotations

import pytest

from lite.agents.models.mai_ui.adapter import _parse_mai_tool_call_json

_BROKEN = [
    ('{"name":"mobile_use","arguments":{"action":"click","coordinate": [412, 733}"}}', [412, 733]),
    (
        '{"name":"mobile_use","arguments":{"action":"click","coordinate": [ 100 , 250 })}',
        [100, 250],
    ),
    ('{"name":"mobile_use","arguments":{"action":"click","coordinate": [55,66}}|}', [55, 66]),
    ('{"name":"mobile_use","arguments":{"action":"click","coordinate":[-3,-4}xx', [-3, -4]),
]


@pytest.mark.parametrize(("raw", "expected"), _BROKEN)
def test_truncated_coordinate_array_is_repaired(raw: str, expected: list[int]) -> None:
    parsed = _parse_mai_tool_call_json(raw)
    assert parsed is not None, "repair pass dropped a recoverable call"
    assert parsed["arguments"]["coordinate"] == expected


def test_well_formed_json_is_untouched() -> None:
    raw = '{"name":"mobile_use","arguments":{"action":"click","coordinate":[1,2]}}'
    assert _parse_mai_tool_call_json(raw) == {
        "name": "mobile_use",
        "arguments": {"action": "click", "coordinate": [1, 2]},
    }


def test_unrepairable_body_still_returns_none() -> None:
    assert _parse_mai_tool_call_json("not json at all") is None


# -----------------------------------------------------------------------------
# The ``dict | None`` return type is a CONTRACT, not a hint. A bare
# ``json.loads`` accepts every JSON top level, so ``<tool_call>[1,2]</tool_call>``
# used to hand the caller a ``list`` through a ``dict``-annotated return —
# which is why the caller had to re-check ``isinstance(tc_json, dict)`` twice.
# Pinned at the producer so those caller-side rescues stay deleted.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", ["[1,2]", '["a","b"]', '"a string"', "3", "3.5", "true", "null", "[]"]
)
def test_non_object_json_top_level_is_none_not_the_parsed_value(raw: str) -> None:
    assert _parse_mai_tool_call_json(raw) is None


def test_repair_pass_also_narrows_to_an_object() -> None:
    """The repair branch goes through the same object narrowing — a repaired
    body that parses to a non-object is still ``None``."""
    raw = '["coordinate": [1, 2}'
    parsed = _parse_mai_tool_call_json(raw)
    assert parsed is None or isinstance(parsed, dict)


def test_parse_never_returns_a_non_dict_for_any_body() -> None:
    bodies = [b for b, _ in _BROKEN] + [
        '{"name":"mobile_use","arguments":{}}',
        "[1,2]",
        "null",
        "junk",
        "",
    ]
    for body in bodies:
        out = _parse_mai_tool_call_json(body)
        assert out is None or isinstance(out, dict), body


def test_response_parser_ignores_a_non_object_tool_call_body() -> None:
    """End-to-end through the sole caller: a JSON-array body yields no tool
    call and is flagged as a model-output error, never a crash."""
    from lite.agents.core.adapter import AgentAdapterRegistry
    from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY

    adapter = AgentAdapterRegistry.get("mai_ui@mobile@use")
    msg = adapter.parse_raw_assistant_response("<tool_call>[1,2]</tool_call>")
    assert "tool_calls" not in msg
    assert MODEL_OUTPUT_ERROR_KEY in msg


def test_response_parser_ignores_an_object_without_name_and_arguments() -> None:
    """The SECOND, still-live caller check: the parser guarantees an object, not
    that the object is a flat ``{name, arguments}`` call."""
    from lite.agents.core.adapter import AgentAdapterRegistry
    from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY

    adapter = AgentAdapterRegistry.get("mai_ui@mobile@use")
    msg = adapter.parse_raw_assistant_response('<tool_call>{"foo": 1}</tool_call>')
    assert "tool_calls" not in msg
    assert MODEL_OUTPUT_ERROR_KEY in msg
