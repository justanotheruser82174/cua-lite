"""DAgger teacher and student must truncate to the SAME step (turn-rule bug).

The student's step index comes from ``count_sample_turns`` and its
prompt from the trajectory as it stood when it acted; the DAgger teacher
re-derives turn ``k``'s pre-action context with
``dagger.teacher._truncate_to_observation``. That helper used to count ONE turn
per ``role:"user"`` message, while the core rule
(``lite.core.messages.group_into_turns``) folds a consecutive ``{user, tool}`` run into
ONE observation block. So on the realistic post-tool I/O shape
``U A [T U] A [T U] A`` the teacher relabelled an EARLIER step than the student
took, silently — no exception, no assertion, wrong DAgger targets:

    k=2   teacher: U A T        student: U A T U
    k=3   teacher: U A T U      student: U A T U A T U

This file pins the agreement. The rule now has ONE home
(``lite.core.messages.turns``) which the teacher derives from.

Hermetic: ``dagger.teacher`` imports ``slime.utils`` at module load (unavailable
outside the Slime container), so tiny stubs are installed before the import —
same pattern as ``tests/train/rollout/dagger/test_dagger_truncate_role_tool.py``.

Run:
    uv run pytest tests/train/rollout/dagger/test_dagger_turn_rule_agreement.py -q
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_slime_stubs() -> list[str]:
    if "slime.utils" in sys.modules:
        return []
    inserted: list[str] = []
    for name in ("slime", "slime.utils"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            inserted.append(name)
    http = types.ModuleType("slime.utils.http_utils")
    http.post = lambda *a, **k: None
    proc = types.ModuleType("slime.utils.processing_utils")
    proc.encode_image_for_rollout_engine = lambda *a, **k: None
    sys.modules["slime.utils.http_utils"] = http
    sys.modules["slime.utils.processing_utils"] = proc
    inserted.extend(["slime.utils.http_utils", "slime.utils.processing_utils"])
    return inserted


_SLIME_STUBS = _install_slime_stubs()

import lite.train.rollout.dagger.teacher as dt  # noqa: E402
from lite.core import LiteCUAMetadata, LiteSample  # noqa: E402
from lite.core.messages import group_into_turns  # noqa: E402
from lite.core.messages.turns import count_sample_turns, truncate_sample_to_turn  # noqa: E402

for _name in reversed(_SLIME_STUBS):
    sys.modules.pop(_name, None)

_truncate_to_observation = dt._truncate_to_observation

_ROLE_OF = {"S": "system", "U": "user", "A": "assistant", "T": "tool"}


def _messages(shape: str) -> list[dict]:
    out: list[dict] = []
    for i, tag in enumerate(shape):
        msg: dict = {
            "role": _ROLE_OF[tag],
            "content": [{"type": "text", "text": f"{tag}{i}"}],
        }
        if tag == "T":
            msg["tool_call_id"] = f"call_{i}"
        out.append(msg)
    return out


def _render(messages: list[dict]) -> str:
    inv = {v: k for k, v in _ROLE_OF.items()}
    return "".join(inv[m["role"]] for m in messages)


def _sample(messages: list[dict]) -> LiteSample:
    return LiteSample(metadata=LiteCUAMetadata(), images=[], messages=list(messages))


#: The realistic post-tool I/O trajectory: task query, then per-turn tool result +
#: screenshot observation blocks.
MIXED_SHAPE = "SUATUATUA"


def _student_prefixes(messages: list[dict]) -> list[list[dict]]:
    """The trajectory as the student saw it at each step it acted on — every
    prefix that ends just before an assistant message."""
    return [
        messages[:i]
        for i, m in enumerate(messages)
        if m["role"] == "assistant"
    ]


# -----------------------------------------------------------------------------
# 1. teacher == student, step for step
# -----------------------------------------------------------------------------
def test_teacher_truncation_matches_student_context_on_mixed_block() -> None:
    messages = _messages(MIXED_SHAPE)
    full = _sample(messages)
    prefixes = _student_prefixes(messages)
    assert [_render(p) for p in prefixes] == ["SU", "SUATU", "SUATUATU"]

    for k, prefix in enumerate(prefixes, start=1):
        # the student's own step index for that context
        assert count_sample_turns(_sample(prefix)) == k
        # ... and the teacher's reconstruction of it
        assert _truncate_to_observation(full, k).messages == prefix, (
            f"k={k}: teacher {_render(_truncate_to_observation(full, k).messages)!r} "
            f"!= student {_render(prefix)!r}"
        )


def test_teacher_render_input_matches_student_render_input() -> None:
    """render_step re-truncates with ``truncate_sample_to_turn``; feeding it the
    teacher slice must land on the same messages as feeding it the student's live
    trajectory — i.e. all three turn rules now agree end to end."""
    messages = _messages(MIXED_SHAPE)
    full = _sample(messages)
    for k, prefix in enumerate(_student_prefixes(messages), start=1):
        teacher_view = truncate_sample_to_turn(_truncate_to_observation(full, k), k)
        student_view = truncate_sample_to_turn(_sample(prefix), k)
        assert teacher_view.messages == student_view.messages


@pytest.mark.parametrize(
    "shape, expected",
    [
        # legacy all-user observations: unchanged by the fix
        ("SUAUAUA", ["SU", "SUAU", "SUAUAU"]),
        # one tool-result observation per turn (no trailing screenshot user msg)
        ("SUATATA", ["SU", "SUAT", "SUATAT"]),
        # the mixed block that used to diverge
        ("SUATUATUA", ["SU", "SUATU", "SUATUATU"]),
        # several per-call tool results in one observation block
        ("SUATTUATTUA", ["SU", "SUATTU", "SUATTUATTU"]),
    ],
)
def test_teacher_truncation_across_observation_shapes(shape, expected) -> None:
    full = _sample(_messages(shape))
    got = [_render(_truncate_to_observation(full, k).messages)
           for k in range(1, len(expected) + 1)]
    assert got == expected


# -----------------------------------------------------------------------------
# 2. the teacher slice is the core grouping, not a private walk
# -----------------------------------------------------------------------------
def test_teacher_truncation_is_the_core_grouping() -> None:
    """For every k, the teacher slice == turns 1..k-1 in full + turn k's whole
    observation block, as computed by ``lite.core.messages.group_into_turns``."""
    messages = _messages(MIXED_SHAPE)
    full = _sample(messages)
    system, content = messages[0], messages[1:]
    turns = group_into_turns(content)
    for k in range(1, len(turns) + 1):
        expected = [system]
        for turn in turns[: k - 1]:
            expected.extend(turn["observations"])
            expected.append(turn["assistant"])
        expected.extend(turns[k - 1]["observations"])
        assert _truncate_to_observation(full, k).messages == expected


def test_teacher_truncation_preserves_system_message_and_is_a_copy() -> None:
    full = _sample(_messages(MIXED_SHAPE))
    out = _truncate_to_observation(full, 2)
    assert out.messages[0]["role"] == "system"
    assert out is not full and out.messages is not full.messages
    assert len(full.messages) == len(MIXED_SHAPE)  # untouched
