"""CAGUI use preproc publish tests."""

from __future__ import annotations

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.data.preproc.cagui import use as cagui_use


def _all_tool_calls(messages: list[dict]) -> list[dict]:
    return [
        tc
        for msg in messages
        for tc in msg.get("tool_calls", [])
    ]

def test_cagui_explicit_terminate_moves_to_metadata_others(monkeypatch):
    """Code 11 (STATUS_TASK_IMPOSSIBLE) ends on ``Done.`` like every other row.

    The source-asserted failure label survives in ``metadata.others``, not as a
    tool call: no ``use`` row persists a ``terminate`` or its schema.
    """
    monkeypatch.setattr(cagui_use, "resolve_path", lambda rel, _env: f"/raw/{rel}")
    row = cagui_use.build_episode(
        [
            {
                "episode_id": "ep_1",
                "step_id": 0,
                "instruction": "finish",
                "image_path": "domestic/ep_1/0.jpeg",
                "image_width": 1000,
                "image_height": 1000,
                "result_action_type": 4,
                "result_touch_yx": "[0.5, 0.25]",
                "result_lift_yx": "[0.5, 0.25]",
                "ui_positions": "[]",
            },
            {
                "episode_id": "ep_1",
                "step_id": 1,
                "instruction": "finish",
                "image_path": "domestic/ep_1/1.jpeg",
                "image_width": 1000,
                "image_height": 1000,
                "result_action_type": 11,
            },
        ],
        "OpenBMB/CAGUI/CAGUI_agent",
    )

    calls = _all_tool_calls(row["messages"])
    assert [tool_call_name(tc) for tc in calls] == ["mobile"]
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "tap", "coordinate": [250, 500], "clicks": 1},
    ]
    assert row["metadata"]["extra_tool_schemas"] == []
    assert row["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Done."}],
    }
    assert row["metadata"]["others"]["terminate_status"] == "failure"
