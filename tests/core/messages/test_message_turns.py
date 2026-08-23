"""``lite.core.messages.turns`` — THE observation/assistant turn rule.

The rule used to be hand-rolled at every consumer and the copies drifted (see
``tests/train/rollout/dagger/test_dagger_turn_rule_agreement.py`` for the
training bug that caused). This file pins the new core module:

  1. ``OBSERVATION_ROLES`` is built from the role vocabulary, not literals.
  2. ``group_into_turns`` produces a PINNED grouping on a spread of trajectory
     shapes: the mixed ``[tool, user]`` observation block, a leaked
     mid-trajectory system message, a trailing observation with no assistant,
     tool-first, empty. (This used to be an equivalence assertion against
     ``BaseProtocol.group_into_turns``; that copy has been retired into this
     module, so the semantics are written down directly.)
  3. ``observation_cut_index`` / ``truncate_to_observation`` express turn ``k``'s
     PRE-ACTION prefix in terms of that same grouping, and ``turn_cut_index`` /
     ``truncate_to_turn`` express the action-inclusive one.
  4. The walk TERMINATES on every input, including a malformed one -- the
     property the deleted ``unknown role: skip, never spin`` branch used to
     carry by hand.

Run:
    uv run pytest tests/core/messages/test_message_turns.py -q
"""

from __future__ import annotations

import itertools
import signal
from contextlib import contextmanager

import pytest

from lite.core.messages import (
    ASSISTANT_ROLE,
    MESSAGE_ROLES,
    SYSTEM_ROLE,
    TOOL_ROLE,
    USER_ROLE,
)
from lite.core.messages.turns import (
    OBSERVATION_ROLES,
    count_sample_turns,
    group_into_turns,
    observation_cut_index,
    truncate_sample_to_turn,
    truncate_to_observation,
    truncate_to_turn,
    turn_cut_index,
    turn_spans,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.samples import LiteSample

_ROLE_OF = {"U": USER_ROLE, "A": ASSISTANT_ROLE, "T": TOOL_ROLE, "S": SYSTEM_ROLE}


def _messages(shape: str) -> list[dict]:
    """Build a message list from a compact shape string, e.g. ``"UATUA"``."""
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


class _HangTimeout(Exception):
    pass


@contextmanager
def _hard_timeout(seconds: int):
    def _handler(signum, frame):
        raise _HangTimeout(f"helper did not return within {seconds}s (hang)")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _user_obs(index: int, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image", "index": index},
            {"type": "text", "text": text},
        ],
    }


def _tool_obs(index: int, text: str, call_id: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [
            {"type": "image", "index": index},
            {"type": "text", "text": text},
        ],
    }


def _assistant(step: int) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": f"step {step}"}],
        "tool_calls": [
            {
                "id": f"call_{step}",
                "type": "function",
                "function": {
                    "name": "click",
                    "arguments": {"coordinate": [100 + step, 200]},
                },
            }
        ],
    }


def _sample(messages: list[dict]) -> LiteSample:
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
        images=[],
        messages=messages,
    )


def _legacy_traj(n_turns: int) -> list[dict]:
    msgs: list[dict] = []
    for k in range(n_turns):
        msgs.append(_user_obs(k, "instruction" if k == 0 else "obs"))
        msgs.append(_assistant(k))
    return msgs


def _tool_traj(n_turns: int) -> list[dict]:
    msgs: list[dict] = []
    for k in range(n_turns):
        obs = _user_obs(k, "instruction") if k == 0 else _tool_obs(k, "obs", f"call_{k - 1}")
        msgs.append(obs)
        msgs.append(_assistant(k))
    return msgs


def _assistant_texts(sample: LiteSample) -> list[str]:
    out: list[str] = []
    for message in sample.messages:
        if message.get("role") == "assistant":
            for part in message.get("content") or []:
                if part.get("type") == "action_description":
                    out.append(part["text"])
    return out


#: The spread of shapes every equivalence claim is checked against.
SHAPES = [
    "",  # empty
    "U",  # predict-time partial turn
    "UA",
    "UAUA",  # legacy all-user observations
    "UATUATUA",  # post-tool I/O mixed [tool, user] observation blocks
    "UATTUA",  # multi tool-result block
    "UAU",  # trailing observation, no assistant
    "TATA",  # tool-first
    "AUA",  # assistant-first (assistant-only leading turn)
    "USA",  # leaked system message inside an observation block
    "UASUA",  # leaked system message between two turns
]


# -----------------------------------------------------------------------------
# 1. role set provenance
# -----------------------------------------------------------------------------
def test_observation_roles_come_from_the_role_vocabulary() -> None:
    assert OBSERVATION_ROLES == frozenset({USER_ROLE, TOOL_ROLE})
    # The constant IS the predicate (no ``is_observation_role`` wrapper), so the
    # membership test has to answer for every role AND for non-role objects.
    assert USER_ROLE in OBSERVATION_ROLES and TOOL_ROLE in OBSERVATION_ROLES
    assert ASSISTANT_ROLE not in OBSERVATION_ROLES
    assert SYSTEM_ROLE not in OBSERVATION_ROLES and None not in OBSERVATION_ROLES


# -----------------------------------------------------------------------------
# 2. the grouping itself, shape by shape
#
# This USED to assert equality against ``BaseProtocol.group_into_turns``, the
# hand-rolled copy that has since been retired in favour of this module. A
# self-comparison would prove nothing, so the semantics that equivalence test
# was really defending are written down instead: one row per shape, giving the
# observation block and the assistant of every turn. `[measured before the
# retirement: 19531 role sequences (5 roles incl. ``system`` and an unknown
# role, length 0..6), 0 disagreements between the two implementations]`
# -----------------------------------------------------------------------------

#: shape -> [(observation block, assistant role or None), ...] per turn.
_GROUPING_GOLDEN: dict[str, list[tuple[str, str | None]]] = {
    "": [],
    "U": [("U", None)],
    "UA": [("U", "A")],
    "UAUA": [("U", "A"), ("U", "A")],
    "UATUATUA": [("U", "A"), ("TU", "A"), ("TU", "A")],
    "UATTUA": [("U", "A"), ("TTU", "A")],
    "UAU": [("U", "A"), ("U", None)],
    "TATA": [("T", "A"), ("T", "A")],
    # assistant-first: the leading turn has an EMPTY observation block.
    "AUA": [("", "A"), ("U", "A")],
    # a leaked system message ENDS turn 1's observation block, so turn 2 is
    # assistant-only — the quirk the old equivalence test called out by name.
    "USA": [("U", None), ("", "A")],
    "UASUA": [("U", "A"), ("U", "A")],
}


@pytest.mark.parametrize("shape", SHAPES)
def test_group_into_turns_shape_by_shape(shape: str) -> None:
    turns = group_into_turns(_messages(shape))
    assert [
        (_render(turn["observations"]),
         None if turn["assistant"] is None else _render([turn["assistant"]]))
        for turn in turns
    ] == _GROUPING_GOLDEN[shape]


def test_group_into_turns_mixed_tool_user_block_is_one_observation() -> None:
    messages = _messages("UATUATUA")
    turns = group_into_turns(messages)
    assert len(turns) == 3
    assert [_render(t["observations"]) for t in turns] == ["U", "TU", "TU"]
    assert all(t["assistant"]["role"] == ASSISTANT_ROLE for t in turns)


def test_group_into_turns_groups_multiple_tool_results() -> None:
    first_assistant = _assistant(0)
    first_assistant["tool_calls"].append(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "pwd"}},
        }
    )
    tool_a = _tool_obs(1, "obs", "call_0")
    tool_b = {
        "role": TOOL_ROLE,
        "tool_call_id": "call_1",
        "content": [{"type": "text", "text": "bash output"}],
    }
    messages = [
        _user_obs(0, "Do the task."),
        first_assistant,
        tool_a,
        tool_b,
        _assistant(1),
    ]

    turns = group_into_turns(messages)

    assert len(turns) == 2
    assert turns[0]["observations"] == [messages[0]]
    assert turns[1]["observations"] == [tool_a, tool_b]
    assert turns[1]["assistant"] == messages[-1]


def test_group_into_turns_empty_and_trailing_observation() -> None:
    assert group_into_turns([]) == []
    turns = group_into_turns(_messages("UAU"))
    assert len(turns) == 2
    assert turns[1]["assistant"] is None  # predict-time partial turn


def test_group_into_turns_tool_role_terminates() -> None:
    messages = [
        _user_obs(0, "instruction"),
        _assistant(0),
        _tool_obs(1, "obs", "call_0"),
        _assistant(1),
    ]

    with _hard_timeout(5):
        turns = group_into_turns(messages)

    assert isinstance(turns, list)
    grouped_assistants = [turn["assistant"] for turn in turns if turn.get("assistant")]
    assert grouped_assistants == [_assistant(0), _assistant(1)]


def test_turn_spans_indices_are_positional() -> None:
    spans = turn_spans(_messages("UATUATUA"))
    assert [(s.observations, s.assistant) for s in spans] == [
        ((0,), 1),
        ((2, 3), 4),
        ((5, 6), 7),
    ]


# -----------------------------------------------------------------------------
# 3. the pre-action prefix, derived from the SAME grouping
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, expected",
    [
        ("UATUATUA", ["U", "UATU", "UATUATU", "UATUATUA"]),
        ("UAUA", ["U", "UAU", "UAUA", "UAUA"]),
        ("UATTUA", ["U", "UATTU", "UATTUA", "UATTUA"]),
        ("TATA", ["T", "TAT", "TATA", "TATA"]),
        ("UAU", ["U", "UAU", "UAU", "UAU"]),
        ("USA", ["U", "US", "USA", "USA"]),
    ],
)
def test_truncate_to_observation_prefixes(shape: str, expected: list[str]) -> None:
    messages = _messages(shape)
    got = [_render(truncate_to_observation(messages, k)) for k in (1, 2, 3, 4)]
    assert got == expected


@pytest.mark.parametrize("shape", SHAPES)
def test_truncate_to_observation_is_a_prefix_of_turns(shape: str) -> None:
    """For every k the slice equals turns 1..k-1 in full plus turn k's whole
    observation block — i.e. it is DERIVED from ``group_into_turns``, not a
    parallel walk. (Checked modulo leaked system messages, which belong to no
    turn but stay in the positional slice.)"""
    messages = _messages(shape)
    turns = group_into_turns(messages)
    for k in range(1, len(turns) + 1):
        kept = truncate_to_observation(messages, k)
        expected: list[dict] = []
        for turn in turns[: k - 1]:
            expected.extend(turn["observations"])
            if turn["assistant"] is not None:
                expected.append(turn["assistant"])
        expected.extend(turns[k - 1]["observations"])
        assert [m for m in kept if m["role"] != SYSTEM_ROLE] == expected
    # k past the last turn keeps everything.
    assert truncate_to_observation(messages, len(turns) + 1) == messages
    assert observation_cut_index(messages, len(turns) + 1) == len(messages)


@pytest.mark.parametrize(
    "cut_func",
    [observation_cut_index, turn_cut_index, truncate_to_observation, truncate_to_turn],
)
@pytest.mark.parametrize("k", [0, -1])
def test_turn_cut_helpers_reject_non_positive_k(cut_func, k: int) -> None:
    with pytest.raises(ValueError, match="turn index k must be >= 1"):
        cut_func(_messages("UA"), k)


# -----------------------------------------------------------------------------
# 4. the action-inclusive prefix, and the three walks that are now ONE walk
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, expected",
    [
        ("UATUATUA", ["UA", "UATUA", "UATUATUA", "UATUATUA"]),
        ("UAUA", ["UA", "UAUA", "UAUA", "UAUA"]),
        ("UAU", ["UA", "UAU", "UAU", "UAU"]),
        ("AUA", ["A", "AUA", "AUA", "AUA"]),
        ("UASUA", ["UAS", "UASUA", "UASUA", "UASUA"]),
    ],
)
def test_truncate_to_turn_prefixes(shape: str, expected: list[str]) -> None:
    messages = _messages(shape)
    got = [_render(truncate_to_turn(messages, k)) for k in (1, 2, 3, 4)]
    assert got == expected


@pytest.mark.parametrize("shape", SHAPES)
def test_sample_turn_count_and_truncation_derive_from_the_same_spans(
    shape: str,
) -> None:
    """Sample turn helpers derive from the same walk as ``group_into_turns``.

    The old adapter-owned copies disagreed with this walk on role:"tool"
    observation blocks. Moving the sample helpers beside the message turn owner
    makes that disagreement unrepresentable.
    """
    messages = _messages(shape)
    sample = LiteSample(metadata=LiteCUAMetadata(), messages=messages)
    assert count_sample_turns(sample) == len(group_into_turns(messages))
    for k in range(1, len(messages) + 2):
        assert truncate_sample_to_turn(sample, k).messages == truncate_to_turn(
            messages, k
        )


def test_count_sample_turns_legacy_by_user_characterization() -> None:
    sample = _sample(_legacy_traj(n_turns=3))
    assert count_sample_turns(sample) == 3


def test_count_sample_turns_by_assistant() -> None:
    sample = _sample(_tool_traj(n_turns=3))
    n_assistant = sum(1 for message in sample.messages if message.get("role") == "assistant")

    assert n_assistant == 3
    assert count_sample_turns(sample) == n_assistant


def test_truncate_legacy_by_user_characterization() -> None:
    legacy = _sample(_legacy_traj(n_turns=3))
    truncated = truncate_sample_to_turn(legacy, 2)

    assert _assistant_texts(truncated) == ["step 0", "step 1"]


def test_truncate_role_tool_matches_legacy() -> None:
    legacy = _sample(_legacy_traj(n_turns=3))
    tool = _sample(_tool_traj(n_turns=3))

    legacy_cov = _assistant_texts(truncate_sample_to_turn(legacy, 2))
    tool_cov = _assistant_texts(truncate_sample_to_turn(tool, 2))

    assert legacy_cov == ["step 0", "step 1"]
    assert tool_cov == legacy_cov


def test_truncate_keeps_multiple_tool_results_for_one_assistant_turn() -> None:
    first_assistant = _assistant(0)
    first_assistant["tool_calls"].append(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "pwd"}},
        }
    )
    sample = _sample([
        _user_obs(0, "instruction"),
        first_assistant,
        _tool_obs(1, "screen", "call_0"),
        _tool_obs(2, "stdout", "call_1"),
        _assistant(1),
    ])

    truncated = truncate_sample_to_turn(sample, 2)
    tool_call_ids = [
        message["tool_call_id"]
        for message in truncated.messages
        if message.get("role") == "tool"
    ]

    assert tool_call_ids == ["call_0", "call_1"]
    assert _assistant_texts(truncated) == ["step 0", "step 1"]


# -----------------------------------------------------------------------------
# 5. TERMINATION — the property the deleted "never spin" branch carried by hand
# -----------------------------------------------------------------------------
def test_the_three_branches_exhaust_the_closed_role_vocabulary() -> None:
    """``turn_spans`` branches on observation / assistant / system. Because
    ``role`` is closed and validated at the boundary, that is every role there
    is — so there is no fourth case whose semantics anyone has to invent."""
    assert OBSERVATION_ROLES | {ASSISTANT_ROLE, SYSTEM_ROLE} == MESSAGE_ROLES


# ``turn_spans`` used to be guarded here by a SHAPE pin as well
# (``test_turn_spans_has_no_cursor_that_could_fail_to_advance``: no ``ast.While``
# and exactly one ``ast.For`` in its source). It pinned the implementation, not
# the property: a behaviour-preserving ``_walk()`` extraction reddens it, while
# the exhaustive 144-case termination test below stays green and is what actually
# proves no branch can fail to advance. Deleted rather than repaired.

_MALFORMED_ROLES = ["unknown", "", "developer", "human", None, 0, True, ("tool",)]


@pytest.mark.parametrize(
    "roles",
    list(itertools.product(_MALFORMED_ROLES + sorted(MESSAGE_ROLES), repeat=2)),
)
def test_turn_spans_terminates_on_any_role_value(roles: tuple) -> None:
    """Runs to completion for EVERY role value, valid or malformed, and returns a
    well-formed span set: indices in range, each message claimed at most once.
    Malformed roles are rejected upstream by ``validate_message_roles``; this
    pins that the walk cannot hang even when they are not."""
    messages = [{"role": role, "content": []} for role in roles]
    spans = turn_spans(messages)
    claimed = [
        index
        for span in spans
        for index in (*span.observations, *(() if span.assistant is None else (span.assistant,)))
    ]
    assert len(claimed) == len(set(claimed))
    assert all(0 <= index < len(messages) for index in claimed)
