"""Tests for shared devs/data utility helpers.

Run:
    uv run --extra dev pytest devs/data/tests/test_utils.py -q
"""

from __future__ import annotations

import json

import pytest

from devs.data.utils import compact_row_images
from lite.core.tools import make_tool_call


def _pictures_addressed(images: list[str], messages: list[dict]) -> list[str]:
    """The picture each image part actually addresses, in message order."""
    return [
        images[part["index"]]
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    ]


def _row_with_mid_sequence_orphan() -> tuple[list[str], list[dict]]:
    """Four pictures, but nothing references ``b.png`` after a dropped turn."""
    images = ["a.png", "b.png", "c.png", "d.png"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "goal"},
            ],
        },
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call("computer", {"actions": []}, call_id="call_0000"),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [
                {"type": "image", "index": 2},
                {"type": "text", "text": "clicked"},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0001",
            "content": [{"type": "image", "index": 3}],
        },
    ]
    return images, messages


def test_compact_row_images_drops_a_mid_sequence_orphan_and_keeps_the_pictures() -> None:
    images, messages = _row_with_mid_sequence_orphan()
    before = _pictures_addressed(images, messages)

    out_images, out_messages = compact_row_images(images, messages)

    assert out_images == ["a.png", "c.png", "d.png"]
    indices = [
        part["index"]
        for message in out_messages
        for part in message["content"]
        if part.get("type") == "image"
    ]
    assert indices == [0, 1, 2]
    assert sorted(indices) == list(range(len(out_images)))
    assert _pictures_addressed(out_images, out_messages) == before
    assert before == ["a.png", "c.png", "d.png"]


def test_compact_row_images_is_a_noop_on_an_already_dense_row() -> None:
    images = ["a.png", "b.png"]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "image", "index": 1}],
        },
    ]

    out_images, out_messages = compact_row_images(images, messages)

    assert out_images == images
    assert out_messages == messages


def test_compact_row_images_leaves_a_row_with_no_image_parts_alone() -> None:
    """No image parts means the images are addressed some other way."""
    images = ["a.png", "b.png"]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]

    out_images, out_messages = compact_row_images(images, messages)

    assert out_images == images
    assert out_messages == messages


def test_compact_row_images_rejects_an_out_of_range_index() -> None:
    """A broken reference must raise, not be renumbered into range."""
    images = ["a.png", "b.png"]
    messages = [{"role": "user", "content": [{"type": "image", "index": 2}]}]

    with pytest.raises(ValueError, match="out of range before compaction"):
        compact_row_images(images, messages)


def test_compact_row_images_remaps_repeated_references_consistently() -> None:
    images = ["a.png", "b.png", "c.png"]
    messages = [
        {"role": "user", "content": [{"type": "image", "index": 2}]},
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "image", "index": 2}],
        },
    ]
    before = _pictures_addressed(images, messages)

    out_images, out_messages = compact_row_images(images, messages)

    assert out_images == ["c.png"]
    assert _pictures_addressed(out_images, out_messages) == before == [
        "c.png",
        "c.png",
    ]


def test_compact_row_images_reads_json_string_images() -> None:
    """Parquet/HF rows hand it a JSON string; the read boundary is shared."""
    images = json.dumps(["a.png", "b.png"])
    messages = [{"role": "user", "content": [{"type": "image", "index": 1}]}]

    out_images, out_messages = compact_row_images(images, messages)

    assert out_images == ["b.png"]
    assert out_messages[0]["content"][0]["index"] == 0
