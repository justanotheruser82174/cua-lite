"""GUI-Odyssey preproc tool/result contract tests."""

from __future__ import annotations

import pytest
from data.preproc._tool_io_helpers import (
    _all_calls,
    _assert_first_action_result_is_tool,
    _assert_structural_done_row,
    _assert_terminate_outcome,
)

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.preproc.guiodyssey import use as guiodyssey_use
from lite.data.utils.rows import validate_canonical_rows


def test_guiodyssey_post_action_screenshot_is_tool(monkeypatch):
    monkeypatch.setattr(guiodyssey_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = guiodyssey_use.build_episode(
        "ep",
        {
            "device_info": {"w": 1000, "h": 1000, "device_name": "phone"},
            "task_info": {"instruction": "finish", "category": "unit", "app": ["app"]},
            "steps": [
                {
                    "step": 0,
                    "action": "CLICK",
                    "info": [[100, 200], [100, 200]],
                    "low_level_instruction": "tap",
                },
                {
                    "step": 1,
                    "action": "CLICK",
                    "info": [[300, 400], [300, 400]],
                    "low_level_instruction": "tap",
                },
            ],
        },
    )
    _assert_first_action_result_is_tool(row)


def test_guiodyssey_coordinates_are_already_normalized_not_device_scaled(monkeypatch):
    monkeypatch.setattr(guiodyssey_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = guiodyssey_use.build_episode(
        "ep",
        {
            "device_info": {"w": 1080, "h": 2400, "device_name": "phone"},
            "task_info": {"instruction": "finish", "category": "unit", "app": ["app"]},
            "steps": [
                {
                    "step": 0,
                    "action": "CLICK",
                    "info": [[900, 800], [900, 800]],
                    "low_level_instruction": "tap",
                },
                {"step": 1, "action": "COMPLETE", "info": ""},
            ],
        },
    )

    calls = _all_calls(row)
    assert [tool_call_name(call) for call in calls] == ["mobile"]
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "tap", "coordinate": [900, 800], "clicks": 1}
    ]


def _guiodyssey_episode(terminal: dict) -> dict:
    return {
        "device_info": {"w": 1000, "h": 1000, "device_name": "phone"},
        "task_info": {"instruction": "finish", "category": "unit", "app": ["app"]},
        "steps": [
            {
                "step": 0,
                "action": "CLICK",
                "info": [[100, 200], [100, 200]],
                "low_level_instruction": "tap",
            },
            {
                "step": 1,
                "action": "CLICK",
                "info": [[300, 400], [300, 400]],
                "low_level_instruction": "tap",
            },
            terminal,
        ],
    }


def test_guiodyssey_complete_becomes_done(monkeypatch):
    """``COMPLETE`` is a structural episode-end marker → ``Done.``, no terminate."""
    monkeypatch.setattr(guiodyssey_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = guiodyssey_use.build_episode(
        "ep", _guiodyssey_episode({"step": 2, "action": "COMPLETE", "info": ""})
    )

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile", "mobile"]


def test_guiodyssey_rejects_steps_after_terminal(monkeypatch):
    monkeypatch.setattr(guiodyssey_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    episode = _guiodyssey_episode({"step": 2, "action": "COMPLETE", "info": ""})
    episode["steps"].append(
        {"step": 3, "action": "CLICK", "info": [[1, 1], [1, 1]], "low_level_instruction": "tap"}
    )
    with pytest.raises(guiodyssey_use.SkipEpisodeError, match="steps after terminal"):
        guiodyssey_use.build_episode("ep", episode)


def test_guiodyssey_incomplete_reason_comes_from_ps_not_info(monkeypatch):
    """``INCOMPLETE`` asserts the episode did NOT achieve the goal.

    That outcome label plus the annotator's authored reason is real supervision
    signal, so both move to ``metadata.others``. The reason lives in **``ps``**, not
    ``info``: the upstream README says ``info`` is ``""`` for any action other than
    CLICK/LONG_PRESS/SCROLL and puts the why-it-was-impossible note in ``ps``, and the
    corpus agrees (472 INCOMPLETE steps, 0 with non-blank ``info``, 472 with non-blank
    ``ps``). Reading ``info`` made ``others.terminate_reason`` unreachable for this
    source, so the raw shape below is the real one -- blank ``info``, prose in ``ps``.
    """
    monkeypatch.setattr(guiodyssey_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = guiodyssey_use.build_episode(
        "ep",
        _guiodyssey_episode({"step": 2, "action": "INCOMPLETE", "info": "", "ps": "No flights"}),
    )

    _assert_structural_done_row(row)
    assert [tool_call_name(call) for call in _all_calls(row)] == ["mobile", "mobile"]
    _assert_terminate_outcome(row, status="failure", reason="No flights")
    validate_canonical_rows([row], "guiodyssey")


def test_guiodyssey_blank_ps_emits_no_reason(monkeypatch):
    """A blank ``ps`` is not information -- only the status is recorded.

    Unreachable in today's corpus (472/472 INCOMPLETE steps author a ``ps``), but the
    ``terminate_outcome_others`` contract is "reason only when non-blank", and every
    layer must agree on that or the byte-equality bar breaks.
    """
    monkeypatch.setattr(guiodyssey_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = guiodyssey_use.build_episode(
        "ep", _guiodyssey_episode({"step": 2, "action": "INCOMPLETE", "info": "", "ps": ""})
    )

    _assert_structural_done_row(row)
    _assert_terminate_outcome(row, status="failure")
    assert "terminate_reason" not in row["metadata"]["others"]


def test_guiodyssey_incomplete_ignores_stray_info_text(monkeypatch):
    """``info`` is never the reason, even if a future annotation puts prose there.

    The README pins ``info`` to ``""`` for INCOMPLETE and the whole corpus honours it,
    so this is a replacement, not a fallback -- pinned so a "helpful" fallback branch
    cannot be reintroduced without a measurement that justifies it.
    """
    monkeypatch.setattr(guiodyssey_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = guiodyssey_use.build_episode(
        "ep",
        _guiodyssey_episode(
            {"step": 2, "action": "INCOMPLETE", "info": "app crashed", "ps": "Item not found"}
        ),
    )

    _assert_structural_done_row(row)
    _assert_terminate_outcome(row, status="failure", reason="Item not found")
    validate_canonical_rows([row], "guiodyssey_stray_info")
