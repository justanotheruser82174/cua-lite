from __future__ import annotations

from lite.agents.extensions.browsergym.goal_image import _goal_image_indices


def test_goal_image_indices_reads_turn0_metadata_carrier() -> None:
    """The goal-image indices travel on the turn-0 user message metadata part."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 3},
                {"type": "image", "index": 4},
                {"type": "image", "index": 0},
                {"type": "text", "text": "Match the reference image."},
                {"type": "metadata", "data": {"goal_image_indices": [3, 4]}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "s0"}],
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "click", "arguments": {"coordinate": [1, 2]}},
                }
            ],
        },
    ]

    assert _goal_image_indices(messages) == [3, 4]
