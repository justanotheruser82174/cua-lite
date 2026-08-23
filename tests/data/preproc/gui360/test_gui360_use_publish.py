"""GUI-360 use preproc publish tests."""

from __future__ import annotations

from contextlib import nullcontext

from lite.core.tools.calls import tool_call_name
from lite.data.preproc.gui360 import use as gui360_use
from lite.data.utils.rows import validate_canonical_rows


def test_gui360_single_step_trajectory_publishes_final_action(monkeypatch):
    """A one-step terminal GUI action is a publishable EOF label."""
    class _FakeImage:
        size = (1000, 800)

    monkeypatch.setattr(gui360_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    monkeypatch.setattr(gui360_use.Image, "open", lambda _path: nullcontext(_FakeImage()))
    monkeypatch.setattr(
        gui360_use,
        "load_steps",
        lambda _path: [
            {
                "execution_id": "exec_1",
                "request": "finish the task",
                "complete": "yes",
                "status": "OVERALL_FINISH",
                "thought": "done",
                "subtask": "click finish",
                "screenshot": "0.png",
                "action": {
                    "function": "click",
                    "coordinate_x": 500,
                    "coordinate_y": 400,
                    "args": {},
                },
            }
        ],
    )

    row = gui360_use.build_trajectory("fake.jsonl", "word", "forms")

    assert row is not None
    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert tool_call_name(row["messages"][-1]["tool_calls"][0]) == "computer"
    validate_canonical_rows([row], "gui360-single-step-final-action")
