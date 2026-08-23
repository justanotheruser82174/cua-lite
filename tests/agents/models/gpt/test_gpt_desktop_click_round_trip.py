"""GPT desktop click provider-wire round-trip regressions.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_desktop_click_round_trip.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.core.tools.calls import tool_call_arguments, tool_call_name

RESOLUTION = (1000, 1000)
_SPACE_KEY = "gpt@desktop"


def _single_action(tool_calls):
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert tool_call_name(call) == "computer"
    actions = tool_call_arguments(call)["actions"]
    assert len(actions) == 1
    return actions[0]


def _round_trip(cua_tc):
    space = GPTDesktopActionSpace()
    native = space.convert_tool_calls_to_agent([cua_tc], resolution=RESOLUTION)
    restored = space.convert_tool_calls_from_agent(native, resolution=RESOLUTION)
    return native, _single_action(restored)


class TestGPTMiddleButton:
    def test_middle_button_survives_round_trip(self):
        native, action = _round_trip(
            LiteDesktopActionSpace.click(coordinate=[10, 20], button="middle")
        )
        assert native == [{"type": "click", "x": 10, "y": 20, "button": "middle"}]
        assert action == {"action": "click", "coordinate": [10, 20], "button": "middle"}

    def test_left_click_wire_shape_unchanged(self):
        """The default button stays implicit - no stray ``button`` key."""
        native, action = _round_trip(LiteDesktopActionSpace.click(coordinate=[10, 20]))
        assert native == [{"type": "click", "x": 10, "y": 20}]
        assert action == {"action": "click", "coordinate": [10, 20]}

    def test_right_click_keeps_dedicated_native_type(self):
        native, action = _round_trip(
            LiteDesktopActionSpace.click(coordinate=[10, 20], button="right")
        )
        assert native == [{"type": "right_click", "x": 10, "y": 20}]
        assert action == {"action": "click", "coordinate": [10, 20], "button": "right"}


# -----------------------------------------------------------------------------
# The full (button, clicks) matrix, both directions.
#
# ``_SPELLABLE[key]`` maps every ``(button, clicks)`` pair the provider wire can
# express to its native emission; every pair NOT in the table must raise rather
# than silently degrade to a different action.
# -----------------------------------------------------------------------------

_SPELLABLE: dict[str, dict[tuple[str, int], dict]] = {
    _SPACE_KEY: {
        ("left", 1): {"type": "click", "x": 10, "y": 20},
        ("left", 2): {"type": "double_click", "x": 10, "y": 20},
        ("right", 1): {"type": "right_click", "x": 10, "y": 20},
        ("middle", 1): {"type": "click", "x": 10, "y": 20, "button": "middle"},
    },
}

_ALL_PAIRS = [(b, c) for b in ("left", "right", "middle") for c in (1, 2, 3)]


def _matrix(spellable: bool):
    return [
        pytest.param(key, button, clicks, id=f"{key.split('@')[0]}-{button}-x{clicks}")
        for key, table in _SPELLABLE.items()
        for button, clicks in _ALL_PAIRS
        if ((button, clicks) in table) is spellable
    ]


class TestClickMatrix:
    """Every pair the wire CAN spell survives lite -> agent -> lite intact."""

    @pytest.mark.parametrize("key,button,clicks", _matrix(spellable=True))
    def test_spellable_pair_round_trips(self, key, button, clicks):
        native, action = _round_trip(
            LiteDesktopActionSpace.click(coordinate=[10, 20], button=button, clicks=clicks)
        )
        assert native == [_SPELLABLE[key][(button, clicks)]]
        assert action["action"] == "click"
        assert action.get("button", "left") == button
        assert action.get("clicks", 1) == clicks

    @pytest.mark.parametrize("key,button,clicks", _matrix(spellable=True))
    def test_native_spelling_parses_back(self, key, button, clicks):
        """agent -> lite: the native spelling names the same (button, clicks)."""
        space = GPTDesktopActionSpace()
        restored = space.convert_tool_calls_from_agent(
            [dict(_SPELLABLE[key][(button, clicks)])], resolution=RESOLUTION
        )
        action = _single_action(restored)
        assert action.get("button", "left") == button
        assert action.get("clicks", 1) == clicks


class TestUnspellableClicks:
    """An unrepresentable pair raises; it never becomes a different action."""

    @pytest.mark.parametrize("key,button,clicks", _matrix(spellable=False))
    def test_unspellable_pair_raises(self, key, button, clicks):
        space = GPTDesktopActionSpace()
        call = LiteDesktopActionSpace.click(coordinate=[10, 20], button=button, clicks=clicks)
        with pytest.raises(ValueError, match="cannot express click"):
            space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)

    @pytest.mark.parametrize("key", sorted(_SPELLABLE))
    def test_matrix_partitions_all_nine_pairs(self, key):
        """The table IS the capability claim - no pair is left unclassified."""
        table = _SPELLABLE[key]
        assert set(table) <= set(_ALL_PAIRS)
        assert len(_ALL_PAIRS) == 9
