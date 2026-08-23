from __future__ import annotations

from lite.agents.core.agent.utils.image_markers import (
    PROVIDER_VISIBLE_IMAGE_INDEX_MARKER,
    mark_provider_visible_image_index,
    provider_visible_image_indices_from_marked_blocks,
    strip_provider_visible_image_index_marker,
)


def test_provider_visible_image_marker_helpers_are_single_block_mechanics() -> None:
    raw_block = {"payload": "image-like"}
    marked_block = mark_provider_visible_image_index(raw_block, 7)

    assert marked_block[PROVIDER_VISIBLE_IMAGE_INDEX_MARKER] == 7
    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in raw_block

    nested_marked_block = mark_provider_visible_image_index({"payload": "nested"}, 8)
    marked_container = mark_provider_visible_image_index({"nested": [nested_marked_block]}, 9)

    stripped = strip_provider_visible_image_index_marker(marked_container)

    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in stripped
    assert stripped["nested"][0][PROVIDER_VISIBLE_IMAGE_INDEX_MARKER] == 8


def test_none_image_index_does_not_create_marker() -> None:
    block = {"type": "input_image", "image_url": "image"}

    marked_block = mark_provider_visible_image_index(block, None)

    assert marked_block == block
    assert PROVIDER_VISIBLE_IMAGE_INDEX_MARKER not in marked_block


def test_provider_visible_image_indices_from_marked_blocks_collects_marked_blocks() -> None:
    assert provider_visible_image_indices_from_marked_blocks(
        [
            mark_provider_visible_image_index({"type": "image_url"}, 1),
            {"type": "image_url"},
            "not-a-block",
            mark_provider_visible_image_index({"type": "input_image"}, 3),
        ]
    ) == (1, 3)
