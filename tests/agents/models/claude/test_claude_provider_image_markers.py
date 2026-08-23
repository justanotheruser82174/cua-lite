from __future__ import annotations

from lite.agents.core.agent.utils.image_markers import (
    PROVIDER_VISIBLE_IMAGE_INDEX_MARKER,
    mark_provider_visible_image_index,
)
from lite.agents.models.claude.agent import (
    _provider_visible_image_indices_from_claude_messages,
    _strip_claude_image_markers,
)


def test_claude_request_marker_traversal_is_message_content_shaped() -> None:
    non_image_marker = mark_provider_visible_image_index(
        {"payload": "not-a-claude-image-block"},
        99,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "task"},
                mark_provider_visible_image_index(
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,first"},
                    },
                    3,
                ),
                {"nested": [non_image_marker]},
            ],
        },
        {
            "role": "tool",
            "content": [
                mark_provider_visible_image_index(
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,tool"},
                    },
                    4,
                )
            ],
        },
        {"role": "assistant", "content": "done"},
    ]

    assert _provider_visible_image_indices_from_claude_messages(messages) == (3, 4)

    stripped = _strip_claude_image_markers(messages)

    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in stripped[0]["content"][1]
    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in stripped[1]["content"][0]
    assert stripped[0]["content"][2]["nested"][0][PROVIDER_VISIBLE_IMAGE_INDEX_MARKER] == 99
