"""Claude action-space round-trip fidelity.

Run:
    uv run pytest tests/agents/models/claude/test_claude_round_trip_fidelity.py
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
)

RESOLUTION = (1000, 1000)


@pytest.fixture(autouse=True, scope="module")
def _registered():
    from lite.agents.bootstrap import register_all

    register_all()


DESKTOP_CASES: dict[str, dict] = {
    "click": LiteDesktopActionSpace.click(coordinate=[500, 300]),
    "click_right": LiteDesktopActionSpace.click(coordinate=[500, 300], button="right"),
    "click_double": LiteDesktopActionSpace.click(coordinate=[500, 300], clicks=2),
    "type": LiteDesktopActionSpace.type(text="hello"),
    "key": LiteDesktopActionSpace.key(keys=["ctrl", "c"]),
    "key_plus": LiteDesktopActionSpace.key(keys=["+"]),
    "key_minus": LiteDesktopActionSpace.key(keys=["-"]),
    "key_equal": LiteDesktopActionSpace.key(keys=["="]),
    "key_comma": LiteDesktopActionSpace.key(keys=[","]),
    "key_ctrl_plus": LiteDesktopActionSpace.key(keys=["ctrl", "+"]),
    "key_ctrl_minus": LiteDesktopActionSpace.key(keys=["ctrl", "-"]),
    "key_ctrl_equal": LiteDesktopActionSpace.key(keys=["ctrl", "="]),
    "key_ctrl_comma": LiteDesktopActionSpace.key(keys=["ctrl", ","]),
    "key_down": LiteDesktopActionSpace.key_down(keys=["shift"]),
    "key_up": LiteDesktopActionSpace.key_up(keys=["shift"]),
    "hold_key": LiteDesktopActionSpace.hold_key(keys=["shift"], duration=2.0),
    "hold_key_plus": LiteDesktopActionSpace.hold_key(keys=["+"], duration=2.0),
    "hold_key_minus": LiteDesktopActionSpace.hold_key(keys=["-"], duration=2.0),
    "hold_key_equal": LiteDesktopActionSpace.hold_key(keys=["="], duration=2.0),
    "hold_key_comma": LiteDesktopActionSpace.hold_key(keys=[","], duration=2.0),
    "hold_key_ctrl_plus": LiteDesktopActionSpace.hold_key(keys=["ctrl", "+"], duration=2.0),
    "hold_key_ctrl_minus": LiteDesktopActionSpace.hold_key(keys=["ctrl", "-"], duration=2.0),
    "hold_key_ctrl_equal": LiteDesktopActionSpace.hold_key(keys=["ctrl", "="], duration=2.0),
    "hold_key_ctrl_comma": LiteDesktopActionSpace.hold_key(keys=["ctrl", ","], duration=2.0),
    "scroll_down": LiteDesktopActionSpace.scroll(direction="down", amount=3, coordinate=[400, 400]),
    "scroll_up": LiteDesktopActionSpace.scroll(direction="up", amount=3, coordinate=[400, 400]),
    "scroll_left": LiteDesktopActionSpace.scroll(direction="left", amount=3, coordinate=[400, 400]),
    "scroll_right": LiteDesktopActionSpace.scroll(
        direction="right", amount=3, coordinate=[400, 400]
    ),
    "drag": LiteDesktopActionSpace.drag(coordinate=[600, 700], start_coordinate=[100, 200]),
    "mouse_move": LiteDesktopActionSpace.mouse_move(coordinate=[123, 456]),
    "mouse_down": LiteDesktopActionSpace.mouse_down(),
    "mouse_up": LiteDesktopActionSpace.mouse_up(),
    "screenshot": LiteDesktopActionSpace.screenshot(),
    "cursor_position": LiteDesktopActionSpace.cursor_position(),
    "wait": LiteDesktopActionSpace.wait(duration=2),
}

MOBILE_CASES: dict[str, dict] = {
    "tap": LiteMobileActionSpace.tap(coordinate=[500, 300]),
    "tap_double": LiteMobileActionSpace.tap(coordinate=[500, 300], clicks=2),
    "long_press": LiteMobileActionSpace.long_press(coordinate=[500, 300], duration=2),
    "swipe": LiteMobileActionSpace.swipe(start_coordinate=[100, 200], coordinate=[300, 400]),
    "drag": LiteMobileActionSpace.drag(start_coordinate=[100, 200], coordinate=[300, 400]),
    "pinch": LiteMobileActionSpace.pinch(coordinate=[500, 300], direction="out", amount=25),
    "type": LiteMobileActionSpace.type(text="hello"),
    "system_button": LiteMobileActionSpace.system_button(button="Back"),
    "screenshot": LiteMobileActionSpace.screenshot(),
    "wait": LiteMobileActionSpace.wait(duration=2),
}

SPACE_KEYS: tuple[str, ...] = ("claude@desktop", "claude@mobile")

CANNOT_EXPRESS: dict[str, dict[str, str]] = {
    "claude@desktop": {
        "key_down": "the Claude computer wire has no half-press verb",
        "key_up": "the Claude computer wire has no half-press verb",
    },
}

DECLARED_GAPS: dict[str, dict[str, str]] = {}

RAISES_ON_PARSE: dict[str, dict[str, str]] = {
    "claude@desktop": {
        "cursor_position": "no wire verb; emitted as an unknown wire action, "
        "then parse raises on the unrecognized action",
    },
}


def _cases_for(space) -> dict[str, dict]:
    return MOBILE_CASES if space.platform == "mobile" else DESKTOP_CASES


def _cells() -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for key in SPACE_KEYS:
        space = ActionSpaceRegistry.get(key)
        cells.extend((key, case) for case in _cases_for(space))
    return cells


def _round_trip(space, call: dict) -> list[dict]:
    wire = space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
    return space.convert_tool_calls_from_agent(wire, resolution=RESOLUTION)


from lite.agents.bootstrap import register_all as _register_all  # noqa: E402

_register_all()
_CELLS = _cells()


@pytest.mark.parametrize(("key", "case"), _CELLS, ids=[f"{k}-{c}" for k, c in _CELLS])
def test_claude_round_trip_is_exact_or_the_serializer_raised(key: str, case: str) -> None:
    space = ActionSpaceRegistry.get(key)
    call = _cases_for(space)[case]

    if case in CANNOT_EXPRESS.get(key, {}):
        with pytest.raises(ValueError):
            space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
        return

    if case in RAISES_ON_PARSE.get(key, {}):
        wire = space.convert_tool_calls_to_agent([call], resolution=RESOLUTION)
        with pytest.raises(ValueError):
            space.convert_tool_calls_from_agent(wire, resolution=RESOLUTION)
        return

    restored = _round_trip(space, call)

    if case in DECLARED_GAPS.get(key, {}):
        assert restored != [call], (
            f"{key}/{case} now round-trips exactly, but it is still listed in "
            f"DECLARED_GAPS ({DECLARED_GAPS[key][case]!r}). Delete the row."
        )
        return

    assert restored == [call], (
        f"{key}/{case} does not round-trip and is not declared. Either fix the "
        f"conversion, make the serializer raise and declare it in "
        f"CANNOT_EXPRESS, make the parser raise and declare it in "
        f"RAISES_ON_PARSE, or pin it in DECLARED_GAPS with the fact that is lost."
        f"\n  sent {call}\n  got  {restored}"
    )


@pytest.mark.parametrize("case", ["key_down", "key_up"])
def test_claude_mobile_still_carries_the_desktop_half_press_verbs(case: str) -> None:
    """Desktop-only losses travel as flat pass-through calls on Claude mobile."""
    space = ActionSpaceRegistry.get("claude@mobile")
    call = DESKTOP_CASES[case]
    assert _round_trip(space, call) == [call]
