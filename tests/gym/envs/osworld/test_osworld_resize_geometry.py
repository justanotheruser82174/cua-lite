"""OSWorld live-step coordinate geometry tests."""

from __future__ import annotations

W = 1920
H = 1080


def test_osworld_live_step_path_uses_the_truncating_mode() -> None:
    from lite.gym.envs.osworld.main import to_pyautogui

    assert to_pyautogui("click", {"coordinate": [999, 999]}, W, H) == (
        "pyautogui.click(1918, 1078, button='left')"
    )
    assert to_pyautogui("mouse_move", {"coordinate": [333, 667]}, W, H) == (
        "pyautogui.moveTo(639, 720)"
    )
    assert to_pyautogui("click", {"coordinate": [1000, 1000]}, W, H) == (
        "pyautogui.click(1920, 1080, button='left')"
    )
