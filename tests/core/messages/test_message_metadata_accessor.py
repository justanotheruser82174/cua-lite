"""The one metadata content-part accessor.

Run:
    uv run pytest tests/core/messages/test_message_metadata_accessor.py -q
"""

from __future__ import annotations

from lite.core.messages import METADATA_PART, message_metadata


def _msg(*parts: dict) -> dict:
    return {"role": "user", "content": list(parts)}


def test_returns_the_metadata_payload():
    msg = _msg({"type": "text", "text": "hi"}, {"type": METADATA_PART, "data": {"page_title": "T"}})
    assert message_metadata(msg) == {"page_title": "T"}


def test_missing_part_and_missing_message_are_the_same_empty_answer():
    assert message_metadata(None) == {}
    assert message_metadata({"role": "user", "content": "a bare string"}) == {}
    assert message_metadata(_msg({"type": "text", "text": "hi"})) == {}


def test_merges_across_parts_rather_than_stopping_at_the_first():
    """First-part semantics would silently drop ``web_text``.

    Every hand-rolled copy this accessor replaced kept scanning until it found
    its own key; ``webharbor.webvoyager`` pins exactly this shape.
    """
    msg = _msg(
        {"type": METADATA_PART, "data": {"url": "https://example.test"}},
        {"type": METADATA_PART, "data": {"web_text": "[1] Search"}},
    )
    assert message_metadata(msg) == {"url": "https://example.test", "web_text": "[1] Search"}


def test_later_parts_win_on_a_key_collision():
    msg = _msg(
        {"type": METADATA_PART, "data": {"url": "first"}},
        {"type": METADATA_PART, "data": {"url": "second"}},
    )
    assert message_metadata(msg) == {"url": "second"}


def test_non_dict_payload_is_ignored_not_raised():
    assert message_metadata(_msg({"type": METADATA_PART, "data": "not-a-dict"})) == {}
    assert message_metadata(_msg({"type": METADATA_PART})) == {}
