from __future__ import annotations

from lite.agents.core.agent.utils.image_markers import (
    PROVIDER_VISIBLE_IMAGE_INDEX_MARKER,
    mark_provider_visible_image_index,
)
from lite.agents.models.gpt.agent import (
    _provider_visible_image_indices_from_gpt_input_items,
    _strip_gpt_image_markers,
)


def test_gpt_request_marker_traversal_is_responses_shaped() -> None:
    non_image_marker = mark_provider_visible_image_index(
        {"payload": "not-a-gpt-image-block"},
        99,
    )
    request_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "task"},
                mark_provider_visible_image_index(
                    {"type": "input_image", "image_url": "data:image/png;base64,first"},
                    0,
                ),
                {"nested": [non_image_marker]},
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "fn_1",
            "output": [
                {"type": "input_text", "text": "ok"},
                mark_provider_visible_image_index(
                    {"type": "input_image", "image_url": "data:image/png;base64,tool"},
                    1,
                ),
            ],
        },
        {
            "type": "computer_call_output",
            "call_id": "computer_1",
            "output": mark_provider_visible_image_index(
                {
                    "type": "computer_screenshot",
                    "image_url": "data:image/png;base64,computer",
                },
                2,
            ),
        },
    ]

    assert _provider_visible_image_indices_from_gpt_input_items(request_input) == (0, 1, 2)

    stripped = _strip_gpt_image_markers(request_input)

    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in stripped[0]["content"][1]
    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in stripped[1]["output"][1]
    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in stripped[2]["output"]
    assert stripped[0]["content"][2]["nested"][0][PROVIDER_VISIBLE_IMAGE_INDEX_MARKER] == 99
