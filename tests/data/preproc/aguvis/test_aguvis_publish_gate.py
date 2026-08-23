"""Aguvis producer publish-gate regressions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.data.preproc.aguvis.utils import (
    AguvisParseError,
    pyautogui_to_tool_calls,
)
from lite.data.utils.rows import validate_canonical_rows

_ROOT = Path(__file__).resolve().parents[4]
_PREPROC = _ROOT / "lite" / "data" / "preproc"


def _load_preproc_script(path: Path):
    """Import a hyphenated preproc script (not importable as a module name)."""
    name = f"cua_lite_pg_{path.parent.name}_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aguvis_grounding_row_passes_publish_gate() -> None:
    """Regression: this row used to be rejected for a missing canonical ``id``."""
    module = _load_preproc_script(_PREPROC / "aguvis" / "grounding-action.py")
    row = module._build_row(
        "/tmp/img.png",
        "click the OK button",
        pyautogui_to_tool_calls("pyautogui.click(x=0.5, y=0.5)", "browser"),
        {"platform": "browser", "os": None},
        "row-1",
        "seeclick",
    )
    call = row["messages"][1]["tool_calls"][0]
    assert call["type"] == "function"
    assert tool_call_id(call)
    validate_canonical_rows([row], "aguvis/grounding.action")


_MOBILE_MAPPING = [
    ("pyautogui.click(x=0.5, y=0.5)", "tap"),
    ("pyautogui.doubleClick(x=0.5, y=0.5)", "tap"),
    ("pyautogui.dragTo(x=0.5, y=0.5)", None),
    ("pyautogui.scroll(page=0.1)", "swipe"),
    ("pyautogui.scroll(page=-0.1)", "swipe"),
    ("pyautogui.hscroll(page=0.1)", "swipe"),
    ("pyautogui.hscroll(page=-0.1)", "swipe"),
    ("pyautogui.write(message='hi')", "type"),
    ("pyautogui.rightClick(x=0.5, y=0.5)", None),
    ("pyautogui.middleClick(x=0.5, y=0.5)", None),
    ("pyautogui.tripleClick(x=0.5, y=0.5)", None),
    ("pyautogui.moveTo(x=0.5, y=0.5)", None),
]


@pytest.mark.parametrize("code,expected_action", _MOBILE_MAPPING, ids=lambda v: str(v)[:40])
def test_mobile_never_emits_desktop_wrapper(code: str, expected_action: str | None) -> None:
    if expected_action is None:
        with pytest.raises(AguvisParseError, match="no mobile equivalent"):
            pyautogui_to_tool_calls(code, "mobile")
        return
    calls = pyautogui_to_tool_calls(code, "mobile")
    assert [tool_call_name(c) for c in calls] == ["mobile"], (
        f"{code} emitted a non-mobile wrapper on platform=mobile"
    )
    actions = tool_call_arguments(calls[0])["actions"]
    assert [a["action"] for a in actions] == [expected_action]


def test_mobile_hscroll_is_a_horizontal_swipe() -> None:
    """``hscroll`` must mirror ``scroll``: same synthesis, rotated 90 degrees.

    The two branches previously disagreed - only the vertical one knew about
    mobile. Both source forms carry a direction and a magnitude and no anchor
    point, so both render as a screen-centered swipe.
    """
    right = pyautogui_to_tool_calls("pyautogui.hscroll(page=0.1)", "mobile")
    left = pyautogui_to_tool_calls("pyautogui.hscroll(page=-0.1)", "mobile")
    up = pyautogui_to_tool_calls("pyautogui.scroll(page=0.1)", "mobile")

    r = tool_call_arguments(right[0])["actions"][0]
    assert r["start_coordinate"][1] == r["coordinate"][1] == 500, "must be horizontal"
    assert r["start_coordinate"][0] > r["coordinate"][0], "scroll right = swipe left"

    lft = tool_call_arguments(left[0])["actions"][0]
    assert lft["start_coordinate"][0] < lft["coordinate"][0], "scroll left = swipe right"

    u = tool_call_arguments(up[0])["actions"][0]
    assert u["start_coordinate"][0] == u["coordinate"][0] == 500, "must stay vertical"


@pytest.mark.parametrize("code", [c for c, _ in _MOBILE_MAPPING])
def test_desktop_mapping_still_emits_computer(code: str) -> None:
    """The mobile fixes must not have perturbed the desktop branch."""
    calls = pyautogui_to_tool_calls(code, "desktop")
    assert [tool_call_name(c) for c in calls] == ["computer"]
