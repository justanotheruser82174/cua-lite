"""Golden-lock tests for the core message-content helpers, plus
EDGE cases for ``peel_system_message``
(basic cases already live in tests/data/utils/test_messages_contract.py; only
edges live here).

Every expected value was computed by running the live code and is hardcoded
so consolidation stays byte-identical.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/core/messages/test_message_helpers.py -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

from lite.core import messages as core_messages
from lite.core.messages import (
    extract_first_text,
    get_action_description,
    get_inline_reasoning,
    make_assistant_content,
    peel_system_message,
)

# ---------------------------------------------------------------------------
# make_assistant_content: ordered content list.
# Order (confirmed live): inline_reasoning -> action_description -> text ->
# history_summary. Empty kwargs skipped; all-empty -> [].
# ---------------------------------------------------------------------------

def test_make_assistant_content_all_fields_ordered() -> None:
    assert make_assistant_content(
        text="t",
        inline_reasoning="r",
        action_description="a",
        history_summary="h",
    ) == [
        {"type": "inline_reasoning", "text": "r"},
        {"type": "action_description", "text": "a"},
        {"type": "text", "text": "t"},
        {"type": "history_summary", "text": "h"},
    ]


def test_make_assistant_content_skips_empty() -> None:
    assert make_assistant_content(text="t", action_description="a") == [
        {"type": "action_description", "text": "a"},
        {"type": "text", "text": "t"},
    ]


def test_make_assistant_content_all_empty() -> None:
    assert make_assistant_content() == []


# ---------------------------------------------------------------------------
# extract_first_text: THE first-text-part accessor. ``get_text`` was a second
# exported answer to the same question and was retired into this one; these
# cases are its old golden set, re-pinned on the survivor.
# ---------------------------------------------------------------------------

def test_extract_first_text_ignores_action_description() -> None:
    msg = {
        "content": [
            {"type": "action_description", "text": "act"},
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "second"},
        ]
    }
    assert extract_first_text(msg) == "hello"


def test_extract_first_text_empty_dict_and_none_content() -> None:
    assert extract_first_text({}) == ""
    assert extract_first_text({"content": None}) == ""


def test_get_text_is_retired() -> None:
    """One exported answer to "the first text part of this message"."""
    import lite.core as core

    for module in (core, core_messages, core_messages.content):
        assert not hasattr(module, "get_text")
        assert "get_text" not in module.__all__


# ---------------------------------------------------------------------------
# get_action_description: first non-empty action_description; ignores text.
# ---------------------------------------------------------------------------

def test_get_action_description_first_ignores_text() -> None:
    msg = {
        "content": [
            {"type": "text", "text": "x"},
            {"type": "action_description", "text": "ad1"},
            {"type": "action_description", "text": "ad2"},
        ]
    }
    assert get_action_description(msg) == "ad1"


def test_get_action_description_empty_content() -> None:
    assert get_action_description({"content": []}) == ""


# ---------------------------------------------------------------------------
# get_inline_reasoning: multi-part joined by newline. Native top-level
# ``reasoning_content`` is deliberately NOT merged (pinned).
# ---------------------------------------------------------------------------

def test_get_inline_reasoning_multipart_newline_join() -> None:
    msg = {
        "content": [
            {"type": "inline_reasoning", "text": "a"},
            {"type": "inline_reasoning", "text": "b"},
        ]
    }
    assert get_inline_reasoning(msg) == "a\nb"


def test_get_inline_reasoning_ignores_native_reasoning_content() -> None:
    # Native reasoning lives on a different channel and is NOT merged.
    assert get_inline_reasoning({"reasoning_content": "NATIVE", "content": []}) == ""


# ---------------------------------------------------------------------------
# peel_system_message EDGE cases.
# ---------------------------------------------------------------------------

def test_peel_system_message_none_is_not_an_empty_message_list() -> None:
    with pytest.raises(TypeError):
        peel_system_message(None)


def test_peel_system_message_empty() -> None:
    assert peel_system_message([]) == (None, [])


def test_peel_system_message_no_system_returns_copy() -> None:
    messages = [
        {"role": "user", "content": []},
        {"role": "assistant", "content": []},
    ]
    system, content = peel_system_message(messages)

    assert system is None
    assert content == messages
    assert content is not messages


def test_peel_system_message_only_first_system_peeled() -> None:
    # Multiple system messages: ONLY the first is peeled; a later system
    # message stays in the content list as-is.
    msgs = [
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u"},
        {"role": "system", "content": "s2"},
    ]
    assert peel_system_message(msgs) == (
        {"role": "system", "content": "s1"},
        [
            {"role": "user", "content": "u"},
            {"role": "system", "content": "s2"},
        ],
    )
