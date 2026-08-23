"""Core history predicates for image-budget windows.

Run: uv run pytest tests/agents/core/protocol/test_image_budget_history.py
"""

from __future__ import annotations

import pytest

from lite.agents.core.protocol import BaseProtocol
from lite.core.messages import group_into_turns
from lite.core.tools import make_tool_call


def _action(desc: str, i: int) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": desc}],
        "tool_calls": [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id=f"g{i}",
            )
        ],
    }


def _bash(cmd: str, i: int) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": f"run {cmd}"}],
        "tool_calls": [make_tool_call("bash", {"command": cmd}, call_id=f"b{i}")],
    }


def _img_result(idx: int) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "x",
        "content": [{"type": "image", "index": idx}],
    }


def _txt_result(text: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "x",
        "content": [{"type": "text", "text": text}],
    }


def _turn0() -> dict:
    return {
        "role": "user",
        "content": [{"type": "image", "index": 0}, {"type": "text", "text": "goal"}],
    }


def _action_only(n_turns: int) -> list[dict]:
    msgs = [_turn0()]
    for i in range(n_turns):
        msgs += [_action(f"click {i}", i), _img_result(i + 1)]
    return msgs


def test_the_predicate_asks_the_observation_not_the_tool_name() -> None:
    """``turn_has_image`` must inspect content, never a hard-coded tool list."""
    turns = group_into_turns(
        [
            _turn0(),
            _bash("ls", 0),
            _txt_result("total 0"),
            _action("click", 1),
            _img_result(1),
        ]
    )

    assert BaseProtocol.turn_has_image(turns[0]) is True
    assert BaseProtocol.turn_has_image(turns[1]) is False
    assert BaseProtocol.image_turn_indices(turns) == [0, 2]


@pytest.mark.parametrize("n_turns", [1, 3, 4, 5, 8, 9, 12, 20])
def test_action_only_image_indices_are_the_full_range(n_turns: int) -> None:
    turns = group_into_turns(_action_only(n_turns))

    assert BaseProtocol.image_turn_indices(turns) == list(range(len(turns)))
