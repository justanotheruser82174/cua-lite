"""Tests for MAIUIHistoryProtocol.

Covers:
  1. Backward-compat: ``summary_history_size=None`` (default) matches the
     existing UITars-style behavior — every old assistant turn survives.
  2. Cap behavior: when set, only the most recent ``summary_history_size``
     old assistants (turns BEFORE the image window) are kept.
  3. Image window unchanged: ``full_history_size`` still controls the
     number of recent turns kept with images.
  4. First user (instruction text) always preserved regardless of cap.
  5. Always splits the first user message (MAI-UI-specific structural
     property — UITars only splits on windowing).

Run:
    uv run pytest tests/agents/models/mai_ui/test_mai_ui_protocol.py -v
"""
from __future__ import annotations

from lite_samples import sample_trajectory_long, sample_trajectory_two_turns

from lite.agents.models.mai_ui.protocol import MAIUIHistoryProtocol
from lite.agents.models.ui_tars.protocol import UITarsHistoryProtocol


def _old_assistant_count(out) -> int:
    """Consecutive assistant messages right after the first user (the "old"
    zone — these were stripped of their preceding images by windowing)."""
    n = 0
    for m in out[1:]:
        if m["role"] == "assistant":
            n += 1
        else:
            break
    return n


class TestMAIUIHistoryProtocolDefaults:
    """Default field values + protocol registration."""

    def test_full_history_size_default(self):
        protocol = MAIUIHistoryProtocol()
        assert protocol.full_history_size == 3

    def test_summary_history_size_default_none(self):
        """Default None = unlimited, matches upstream MAI-UI ``_build_messages``."""
        protocol = MAIUIHistoryProtocol()
        assert protocol.summary_history_size is None


class TestSummaryHistoryUnlimited:
    """``summary_history_size=None`` keeps every old assistant, matching upstream."""

    def test_no_cap_keeps_all_old_assistants(self):
        sample = sample_trajectory_long(num_turns=10)
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=None)
        out = protocol.process_messages(sample.messages)
        # 10 turns - 2 window = 8 old assistants
        assert _old_assistant_count(out) == 8

    def test_no_cap_matches_uitars_windowing(self):
        """With ``summary_history_size=None``, MAI-UI output should have the
        same number of old-zone assistants as UITars would produce for the
        same trajectory + window."""
        sample = sample_trajectory_long(num_turns=8)
        mai_out = MAIUIHistoryProtocol(
            full_history_size=3, summary_history_size=None,
        ).process_messages(sample.messages)
        uitars_out = UITarsHistoryProtocol(
            full_history_size=3,
        ).process_messages(sample.messages)
        assert _old_assistant_count(mai_out) == _old_assistant_count(uitars_out)


class TestSummaryHistoryCap:
    """``summary_history_size=N`` — keep only N most recent old assistants."""

    def test_cap_drops_oldest_assistants(self):
        sample = sample_trajectory_long(num_turns=10)
        # 10 turns - window=2 = 8 old; cap to 3 most recent old assistants
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=3)
        out = protocol.process_messages(sample.messages)
        assert _old_assistant_count(out) == 3

    def test_cap_keeps_most_recent_old(self):
        """The 3 surviving old assistants should be steps 6/7/8 (1-indexed
        from sample_trajectory_long's "Action step N." text), NOT 1/2/3."""
        sample = sample_trajectory_long(num_turns=10)
        # turns 0..7 are "old" (window_start = 10 - 2 = 8); cap to 3
        # → keep turns 5, 6, 7 → text "Action step 6.", "Action step 7.", "Action step 8."
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=3)
        out = protocol.process_messages(sample.messages)
        # Skip first user; next 3 messages are the surviving old assistants.
        kept_texts = [
            (m.get("content") or [{}])[0].get("text", "")
            for m in out[1:4]
        ]
        assert kept_texts == [
            "Action step 6.", "Action step 7.", "Action step 8.",
        ]

    def test_cap_zero_drops_all_old(self):
        """``summary_history_size=0`` keeps zero old assistants; only first
        user + image-window turns survive."""
        sample = sample_trajectory_long(num_turns=10)
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=0)
        out = protocol.process_messages(sample.messages)
        assert _old_assistant_count(out) == 0
        # Expected: first user + (user+assistant) × 2 = 5 messages.
        assert len(out) == 5

    def test_cap_larger_than_old_zone_is_noop(self):
        """When ``summary_history_size`` >= number of old turns, behavior
        is identical to ``None`` (every old assistant survives)."""
        sample = sample_trajectory_long(num_turns=10)
        # old zone size = 10 - 2 = 8; cap=100 doesn't bite
        capped = MAIUIHistoryProtocol(
            full_history_size=2, summary_history_size=100,
        ).process_messages(sample.messages)
        unlimited = MAIUIHistoryProtocol(
            full_history_size=2, summary_history_size=None,
        ).process_messages(sample.messages)
        assert len(capped) == len(unlimited)
        assert _old_assistant_count(capped) == _old_assistant_count(unlimited)

    def test_cap_does_not_affect_image_window(self):
        """``summary_history_size`` only caps the OLD assistant zone; the
        image window (``full_history_size``) is untouched."""
        sample = sample_trajectory_long(num_turns=10)
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=1)
        out = protocol.process_messages(sample.messages)
        # Image-window user messages (image inside): exactly full_history_size=2
        n_window_users = sum(
            1 for m in out
            if m["role"] == "user"
            and any(item.get("type") == "image" for item in m.get("content", []))
        )
        assert n_window_users == 2


class TestSummaryHistoryFirstUser:
    """The first user message (instruction) is always kept, independent of cap."""

    def test_first_user_kept_with_summary_zero(self):
        sample = sample_trajectory_long(num_turns=10)
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=0)
        out = protocol.process_messages(sample.messages)
        first = out[0]
        assert first["role"] == "user"
        # Has the instruction text from turn 0's user message.
        assert any(item.get("type") == "text" for item in first.get("content", []))

    def test_first_user_image_always_stripped(self):
        """Independent of cap, first user message never carries an image —
        MAI-UI always splits instruction text from image (the structural
        difference vs UITars)."""
        sample = sample_trajectory_long(num_turns=2)  # short → MAI-UI still splits
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=None)
        out = protocol.process_messages(sample.messages)
        assert out[0]["role"] == "user"
        assert not any(
            item.get("type") == "image" for item in out[0].get("content", [])
        )


class TestSummaryHistoryEdgeCases:
    """Boundary inputs."""

    def test_empty_input(self):
        assert MAIUIHistoryProtocol(summary_history_size=2).process_messages([]) == []

    def test_short_trajectory_no_old_zone(self):
        """When num_turns == full_history_size, there is no old zone — cap
        has nothing to act on and output is identical to unlimited."""
        sample = sample_trajectory_two_turns()
        capped = MAIUIHistoryProtocol(
            full_history_size=2, summary_history_size=0,
        ).process_messages(sample.messages)
        unlimited = MAIUIHistoryProtocol(
            full_history_size=2, summary_history_size=None,
        ).process_messages(sample.messages)
        assert len(capped) == len(unlimited)

    def test_system_message_preserved_with_cap(self):
        """System message is preserved regardless of cap setting."""
        sample = sample_trajectory_long(num_turns=10)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a GUI agent."}]},
            *sample.messages,
        ]
        protocol = MAIUIHistoryProtocol(full_history_size=2, summary_history_size=1)
        out = protocol.process_messages(messages)
        assert out[0]["role"] == "system"
        assert out[0]["content"][0]["text"] == "You are a GUI agent."
