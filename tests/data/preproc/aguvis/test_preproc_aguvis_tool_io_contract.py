"""Aguvis preproc tool/result contract tests."""

from __future__ import annotations

import pytest
from data.preproc._tool_io_helpers import (
    _actions,
    _all_calls,
    _assert_final_action_row,
    _assert_first_action_result_is_tool,
    _assert_no_terminate_outcome,
    _assert_structural_done_row,
    _assert_terminate_outcome,
    _load_preproc_script,
)

from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.data.preproc.aguvis import use as aguvis_use
from lite.data.preproc.aguvis import utils as aguvis_utils
from lite.data.utils.rows import validate_canonical_rows

aguvis_grounding = _load_preproc_script(
    "lite/data/preproc/aguvis/grounding-action.py",
    "cua_lite_test_aguvis_grounding_action",
)


def test_aguvis_grounding_action_derives_extra_tool_schemas() -> None:
    row = aguvis_grounding._build_row(
        "/raw/image.png",
        "finish",
        [
            make_tool_call("open_app", {"app_name": "Settings"}),
            make_tool_call("response", {"text": "done"}),
            make_tool_call("terminate", {"status": "success"}),
        ],
        {"platform": "mobile", "os": "android"},
        "row_1",
        "source_1",
    )

    assert [tool_schema_name(schema) for schema in row["metadata"]["extra_tool_schemas"]] == [
        "open_app",
        "response",
        "terminate",
    ]


def _aguvis_step(image: str, code: str, *, human: str = "Instruction: finish\n\n") -> dict:
    return {
        "image": image,
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": "Action: act"},
            {"from": "gpt", "value": code},
        ],
    }


def test_aguvis_rejects_steps_after_structural_terminator(monkeypatch):
    """A post-terminal suffix must not be silently published as a shorter row."""
    monkeypatch.setattr(aguvis_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    cfg = {"platform": "mobile", "image_dir": "images", "os": "android", "_name": "unit"}

    with pytest.raises(aguvis_use.SkipEpisodeError, match="post_terminal_steps"):
        aguvis_use.build_episode(
            "ep",
            [
                (0, _aguvis_step("ep_0.png", "pyautogui.click(x=0.1, y=0.2)")),
                (1, _aguvis_step("ep_1.png", "pyautogui.click(x=0.2, y=0.3)")),
                (2, _aguvis_step("ep_2.png", "terminate(status='success')")),
                (3, _aguvis_step("ep_3.png", "pyautogui.click(x=0.9, y=0.9)")),
            ],
            cfg,
            "/raw",
        )


def test_aguvis_failure_terminator_moves_to_metadata_others(monkeypatch):
    """``status='failure'`` is a source-asserted outcome label, so it is kept.

    Not as a call, though: this terminal-marker row ends on ``Done.``, and the
    label moves to ``metadata.others``. Dropping it outright would silently
    relabel a failed demonstration as a completed one.
    """
    monkeypatch.setattr(aguvis_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    cfg = {"platform": "mobile", "image_dir": "images", "os": "android", "_name": "unit"}

    row = aguvis_use.build_episode(
        "ep",
        [
            (0, _aguvis_step("ep_0.png", "pyautogui.click(x=0.1, y=0.2)")),
            (1, _aguvis_step("ep_1.png", "pyautogui.click(x=0.2, y=0.3)")),
            (2, _aguvis_step("ep_2.png", "terminate(status='failure', reason='blocked')")),
        ],
        cfg,
        "/raw",
    )

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile", "mobile"]
    _assert_terminate_outcome(row, status="failure", reason="blocked")
    validate_canonical_rows([row], "aguvis")


@pytest.mark.parametrize("n_steps", [2, 3, 5])
def test_aguvis_without_terminator_keeps_final_action_at_eof(monkeypatch, n_steps):
    """No terminator is the NORMAL ending here; the episode must still publish.

    android_control / coat / guide / miniwob carry no explicit terminator at
    all, so a skip on "no terminal terminate" emptied four of the five stage-2
    subsets. The source simply has no screenshot after the last action, so that
    final action stays as the EOF SFT label.
    """
    monkeypatch.setattr(aguvis_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    cfg = {"platform": "mobile", "image_dir": "images", "os": "android", "_name": "unit"}

    row = aguvis_use.build_episode(
        "ep",
        [
            (i, _aguvis_step(f"ep_{i}.png", f"pyautogui.click(x=0.{i + 1}, y=0.2)"))
            for i in range(n_steps)
        ],
        cfg,
        "/raw",
    )

    assert len(row["images"]) == n_steps
    action_turns = [m for m in row["messages"] if m.get("tool_calls")]
    assert len(action_turns) == n_steps
    tool_results = [m for m in row["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == n_steps - 1
    assert [m["content"] for m in tool_results] == [
        [{"type": "image", "index": i}] for i in range(1, n_steps)
    ]
    assert [m["tool_call_id"] for m in tool_results] == [
        tool_call_id(turn["tool_calls"][0]) for turn in action_turns[:-1]
    ]
    assert [m["role"] for m in row["messages"]].count("user") == 1
    _assert_first_action_result_is_tool(row)
    _assert_final_action_row(row)
    _assert_no_terminate_outcome(row)


def test_aguvis_single_step_episode_publishes_final_action(monkeypatch):
    """A lone action is still a valid supervised label."""
    monkeypatch.setattr(aguvis_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    cfg = {"platform": "mobile", "image_dir": "images", "os": "android", "_name": "unit"}

    row = aguvis_use.build_episode(
        "ep",
        [(0, _aguvis_step("ep_0.png", "pyautogui.click(x=0.1, y=0.2)"))],
        cfg,
        "/raw",
    )

    assert len(row["images"]) == 1
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    _assert_final_action_row(row)


def test_aguvis_hotkey_accepts_keyword_keys() -> None:
    """MiniWoB spells 600 valid actions as ``hotkey(keys=[...])``."""
    calls = aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey(keys=['ctrl', 'a'])",
        "browser",
    )

    assert [tool_call_name(c) for c in calls] == ["computer"]
    assert [tool_call_arguments(c) for c in calls] == [
        {"actions": [{"action": "key", "keys": ["ctrl", "a"]}]}
    ]


def _aguvis_swipe(code: str) -> dict:
    """The lone ``swipe`` action a mobile scroll renders to."""
    calls = aguvis_utils.pyautogui_to_tool_calls(code, "mobile")
    assert len(calls) == 1 and tool_call_name(calls[0]) == "mobile"
    actions = tool_call_arguments(calls[0])["actions"]
    assert len(actions) == 1
    return actions[0]


@pytest.mark.parametrize(
    ("code", "start", "end"),
    [
        # page < 0 -> content scrolls DOWN -> the finger travels UP. Cross-checked
        # against the source's own low-level instruction on all 9,833 android_control
        # scroll steps: page=-0.1 reads "swipe up" 6,146 times vs "down" 1,384.
        ("pyautogui.scroll(page=-0.1)", [500, 700], [500, 300]),
        ("pyautogui.scroll(page=0.1)", [500, 300], [500, 700]),
        # hscroll page > 0 -> content scrolls RIGHT -> the finger travels LEFT
        # ("swipe left" 674 vs "right" 131 on the same corpus).
        ("pyautogui.hscroll(page=0.1)", [700, 500], [300, 500]),
        ("pyautogui.hscroll(page=-0.1)", [300, 500], [700, 500]),
    ],
)
def test_aguvis_direction_only_scroll_is_a_full_size_swipe(code, start, end):
    """``page=+-0.1`` carries a direction only, and renders one plausible fling.

    ``page`` is a signed viewport fraction upstream, but ``0.1`` is not a
    measurement: over every raw file this adapter reads it takes exactly two
    values (7,929 x -0.1, 1,904 x +0.1, all in ``android_control.json``), because
    AndroidControl's ``scroll`` is a bare direction with no distance field. The
    old ``min(400, max(1, int(|page| * 10)) * 40)`` therefore pinned every one of
    them to its floor of 40/1000 -- a 4%-of-screen twitch a model learns as a
    no-op. Both axes now render
    ``DIRECTION_ONLY_SWIPE_TRAVEL`` units through the screen centre, and the
    polarity above is unchanged.
    """
    assert _aguvis_swipe(code) == {
        "action": "swipe", "start_coordinate": start, "coordinate": end,
    }
    assert aguvis_utils.DIRECTION_ONLY_SWIPE_TRAVEL == 400


def test_aguvis_scroll_with_a_real_viewport_fraction_is_refused() -> None:
    """A ``page`` this corpus never spells is refused, not approximated.

    A real fraction (aguvis emits ``page=-0.95`` for the GUIAct subsets, which
    this adapter does not read) cannot be expressed by either rendering, so the
    step is dropped the way every other unrenderable action here is dropped.
    """
    with pytest.raises(aguvis_utils.AguvisParseError, match="viewport fraction"):
        aguvis_utils.pyautogui_to_tool_calls("pyautogui.scroll(page=-0.95)", "mobile")


def test_aguvis_wheel_click_scroll_has_no_mobile_rendering() -> None:
    """A touchscreen has no wheel, so a click count is refused like ``moveTo``.

    The positional form occurs only in ``omniact_fix.json``, which is
    ``platform="desktop"``, and there it passes straight through as the wheel
    clicks upstream wrote.
    """
    with pytest.raises(aguvis_utils.AguvisParseError, match="no mobile equivalent"):
        aguvis_utils.pyautogui_to_tool_calls("pyautogui.scroll(30)", "mobile")

    calls = aguvis_utils.pyautogui_to_tool_calls("pyautogui.scroll(-30)", "desktop")
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "scroll", "direction": "down", "amount": 30}
    ]


@pytest.mark.parametrize(
    "description, direction",
    [
        # the two shapes that make up all 386 resolvable cases
        ("the next step is to scroll down the page to explore more options", "down"),
        ("we have already scrolled down, so continue scrolling down the page", "down"),
        # never observed in the corpus, but the axis is symmetric
        ("scroll up to get back to the top of the results", "up"),
        # "download" must not be read as "down"
        ("scroll down the page and then download the logo in SVG format", "down"),
        ("the next step is to download the logo in SVG format", None),
        # no direction of its own: 2 of the 392 -- refused, not inherited
        ("the next step is to continue scrolling to find the design elements", None),
        # names both: refused rather than resolved by position
        ("scroll up and then back down", None),
    ],
)
def test_aguvis_scroll_direction_is_read_from_the_step_description(description, direction):
    """``guide.json``'s THIRD scroll encoding: a bare, argument-less ``scroll()``.

    392 of ``guide.json``'s 13,544 records spell the whole call as
    ``pyautogui.scroll()`` -- no ``page``, no wheel-click count -- and it appears
    in none of the other 13 raw files. The direction is in the same step's
    ``Action:`` prose, which the adapter already publishes as an
    ``action_description`` content part; over all 392 that prose resolves to
    ``{"down"}`` 386 times, ``{}`` 6 times and ``{"up"}`` or both **zero** times.
    """
    assert aguvis_utils.scroll_direction_from_description(description) == direction


def test_aguvis_bare_scroll_is_refused_without_a_direction_and_across_axes() -> None:
    """The bare form is refused when the prose cannot name ITS axis.

    Two refusals, because a resolved direction is not a licence: no direction at
    all (the 6 residual records), and a vertical word offered to a horizontal
    call -- a bare ``hscroll()`` occurs 0 times in the corpus and must not be
    able to borrow one.
    """
    for kwargs in ({}, {"scroll_direction": None}):
        with pytest.raises(aguvis_utils.AguvisParseError, match="carries no argument"):
            aguvis_utils.pyautogui_to_tool_calls("pyautogui.scroll()", "mobile", **kwargs)
    with pytest.raises(aguvis_utils.AguvisParseError, match="carries no argument"):
        aguvis_utils.pyautogui_to_tool_calls(
            "pyautogui.hscroll()", "mobile", scroll_direction="down")

    # Resolved, the bare form renders the SAME gesture as the ``page=+-0.1``
    # cohort -- one direction-only scroll, not two.
    assert aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.scroll()", "mobile", scroll_direction="down",
    ) == aguvis_utils.pyautogui_to_tool_calls("pyautogui.scroll(page=-0.1)", "mobile")
    assert tool_call_arguments(
        aguvis_utils.pyautogui_to_tool_calls(
            "pyautogui.scroll()", "browser", scroll_direction="down",
        )[0]
    )["actions"] == [{"action": "scroll", "direction": "down", "amount": 1}]


def test_aguvis_rejects_norm01_oob_before_normalization_clamp() -> None:
    with pytest.raises(aguvis_utils.AguvisParseError, match="outside \\[0, 1\\]"):
        aguvis_utils.norm01_to_1000(1.2, 0.5)

    assert aguvis_utils.norm01_to_1000(-1e-7, 1 + 1e-7) == [0, 1000]


def test_aguvis_batches_actions_that_share_one_source_observation(monkeypatch) -> None:
    monkeypatch.setattr(aguvis_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    cfg = {**aguvis_use.SUBSETS["miniwob"], "_name": "miniwob"}
    row = aguvis_use.build_episode(
        "ep",
        [
            (0, _aguvis_step("ep_step0.png", "pyautogui.click(x=0.1, y=0.2)")),
            (0, _aguvis_step(
                "ep_step0.png",
                "pyautogui.click(x=0.3, y=0.4)",
                human="Instruction: finish\n\nPrevious actions:\nStep 1: first click",
            )),
            (1, _aguvis_step("ep_step1.png", "pyautogui.click(x=0.5, y=0.6)")),
        ],
        cfg,
        "/raw",
    )
    assert len(row["images"]) == 2
    actions = tool_call_arguments(row["messages"][1]["tool_calls"][0])["actions"]
    assert [action["coordinate"] for action in actions] == [[100, 200], [300, 400]]
    assert row["messages"][2]["content"] == [{"type": "image", "index": 1}]

    reused_path = aguvis_use.build_episode(
        "ep",
        [
            (0, _aguvis_step("ep_step0.png", "pyautogui.click(x=0.1, y=0.2)")),
            (1, _aguvis_step("ep_step0.png", "pyautogui.click(x=0.3, y=0.4)")),
        ],
        cfg,
        "/raw",
    )
    assert len(reused_path["images"]) == 2
    assert reused_path["messages"][2]["role"] == "tool"


def test_aguvis_guide_is_browser_and_recovers_cursor_double_click() -> None:
    assert aguvis_use.SUBSETS["guide"]["platform"] == "browser"
    calls = aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.doubleClick()", "browser", cursor_coordinate=[811, 376]
    )
    assert _actions(calls) == [
        {"action": "click", "coordinate": [811, 376], "clicks": 2}
    ]


def test_aguvis_mobile_dragto_does_not_invent_a_start_coordinate() -> None:
    with pytest.raises(aguvis_utils.AguvisParseError, match="no source start coordinate"):
        aguvis_utils.pyautogui_to_tool_calls(
            "pyautogui.dragTo(x=0.3, y=0.4)", "mobile"
        )


def test_aguvis_desktop_keys_use_the_canonical_vocabulary() -> None:
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey('winleft', 's')", "desktop"
    )) == [{"action": "key", "keys": ["meta", "s"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey('ctrl', '+')", "desktop"
    )) == [{"action": "key", "keys": ["ctrl", "+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey('ctrl', 'plus')", "desktop"
    )) == [{"action": "key", "keys": ["ctrl", "+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey(keys=['ctrl', '+'])", "desktop"
    )) == [{"action": "key", "keys": ["ctrl", "+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey(keys=['ctrl', 'plus'])", "desktop"
    )) == [{"action": "key", "keys": ["ctrl", "+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.press('+')", "desktop"
    )) == [{"action": "key", "keys": ["+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.press('plus')", "desktop"
    )) == [{"action": "key", "keys": ["+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.press(keys='+')", "desktop"
    )) == [{"action": "key", "keys": ["+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.press(key='plus')", "desktop"
    )) == [{"action": "key", "keys": ["+"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey('ctrlleft', 'optionright', 'a')", "desktop"
    )) == [{"action": "key", "keys": ["ctrl", "alt", "a"]}]
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.hotkey('ctrl', '+')", "browser"
    )) == [{"action": "key", "keys": ["ctrl", "+"]}]
    with pytest.raises(aguvis_utils.AguvisParseError, match="Unsupported key token"):
        aguvis_utils.pyautogui_to_tool_calls(
            "pyautogui.hotkey('ctrl', 'ac')", "desktop"
        )
    with pytest.raises(aguvis_utils.AguvisParseError, match="no mobile equivalent"):
        aguvis_utils.pyautogui_to_tool_calls("pyautogui.hotkey('ctrl', '+')", "mobile")
    assert _actions(aguvis_utils.pyautogui_to_tool_calls(
        "pyautogui.press('+')", "mobile"
    )) == [{"action": "type", "text": "+"}]
    for code in ("pyautogui.press(' ')", "pyautogui.press('\\n')"):
        with pytest.raises(aguvis_utils.AguvisParseError, match="Unsupported key token"):
            aguvis_utils.pyautogui_to_tool_calls(code, "desktop")


def test_aguvis_rejects_contaminated_continuation_fragment(monkeypatch) -> None:
    monkeypatch.setattr(aguvis_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    step = _aguvis_step(
        "ep_1.png",
        "pyautogui.click(x=0.1, y=0.2)",
        human="Instruction: old task\n\nPrevious actions:\nStep 1: unrelated task",
    )
    with pytest.raises(aguvis_utils.SkipEpisodeError, match="partial_episode"):
        aguvis_use.build_episode(
            "ep", [(1, step)], {**aguvis_use.SUBSETS["coat"], "_name": "coat"}, "/raw"
        )
