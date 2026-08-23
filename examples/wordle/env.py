"""Wordle environment: a text-only env whose multi-turn structure is load-bearing.

The model guesses a hidden four-letter word. Each guess returns one mark per
letter — ``correct`` / ``present`` / ``absent`` — and nothing else, so turn N is
unanswerable without turn N-1's feedback and the target can never leak while the
episode can still continue.

Default configs expose no tools: the model answers in natural language and the
runtime delivers a transient ``response(text=...)`` action carrying that text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, NamedTuple

from examples.wordle.words import ANSWERS
from lite.core.metadata import LiteBaseMetadata, LiteGenericMetadata
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.results import make_tool_result
from lite.core.tools.schemas import tool_schema_name
from lite.gym.base import LiteBaseEnv
from lite.gym.registry import _clear_env_registration, register, registry
from lite.gym.services import BackendFamily, register_family
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY, LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.ingress import (
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
)
from lite.gym.utils.feedback.results import canonical_result_call_id

ENV_ID = "wordle"
WORD_LENGTH = 4
DEFAULT_MAX_TURNS = 6
DEFAULT_TASK_ID = "word_000000"

CORRECT = "correct"
PRESENT = "present"
ABSENT = "absent"

# Reward weights. GRPO consumes one scalar per trajectory, so these describe the
# EPISODE return, not per-step credit; the env pays the whole thing on the step
# that ends the episode.
_SOLVE_BASE = 1.0
_SOLVE_SPEED = 0.5
# Weights favour the skill being trained. Measured with a 2B policy: once a
# per-turn format reminder cut the unparseable rate to 22%, the consistency rate
# was only 11% against 100% for a perfect filter — the model emits guesses but
# does not narrow the candidate set. Consistency therefore carries the bulk of
# the unsolved score, and format is worth little once it is easy to earn.
_W_GREEN = 0.15
_W_CONSISTENCY = 0.40
_W_FORMAT = 0.05

# The model states its guess on a "Guess: <word>" line. The greedy prefix takes
# the LAST "guess:" on a line, and non-letters act as separators so markdown and
# trailing punctuation fall away.
_GUESS_LINE = re.compile(r".*guess\s*:\s*(.*)", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z]+")


class _Scored(NamedTuple):
    guess: str
    marks: tuple[str, ...]


def _task_metadata(
    *, extra_tool_schemas: list[dict[str, Any]] | None = None
) -> LiteGenericMetadata:
    return LiteGenericMetadata(dims=(), extra_tool_schemas=extra_tool_schemas or [])


def _mark_guess(guess: str, target: str) -> tuple[str, ...]:
    marks = [ABSENT] * WORD_LENGTH
    pool = Counter(target)
    for i in range(WORD_LENGTH):
        if guess[i] == target[i]:
            marks[i] = CORRECT
            pool[guess[i]] -= 1
    for i in range(WORD_LENGTH):
        if marks[i] == CORRECT:
            continue
        if pool[guess[i]] > 0:
            marks[i] = PRESENT
            pool[guess[i]] -= 1
    return tuple(marks)


def _is_consistent(candidate: str, history: list[tuple[str, tuple[str, ...]]]) -> bool:
    # Consistency is re-marking, never accumulated letter constraints: `absent`
    # means "no FURTHER copies", so naive accumulation rejects the true target
    # whenever a guess repeated a letter.
    return all(
        _mark_guess(past_guess, candidate) == past_marks for past_guess, past_marks in history
    )


def _parse_guess(visible_text: str) -> str | None:
    text = unicodedata.normalize("NFKC", visible_text).replace("\r\n", "\n").replace("\r", "\n")
    for line in reversed(text.split("\n")):
        match = _GUESS_LINE.match(line)
        if match is None:
            continue
        for token in _TOKEN.findall(match.group(1)):
            if len(token) == WORD_LENGTH:
                return token.lower()
        # Fall through to an earlier "guess:" line. Measured on 55 unparseable
        # 2B replies: 17 named a wrong-length word on the last such line while an
        # earlier one carried a valid guess.
    return None


def _catalog_block() -> str:
    """The answer pool, wrapped for the reset observation.

    The pool is shown because it is otherwise an unobservable rule: measured, a
    2B policy guessed ordinary English words (twig, face, hole) that were not in
    the catalog, so it could not satisfy the consistency term no matter how well
    it reasoned. Showing the pool turns "enumerate the catalog from memory" into
    "filter a visible list", which is the skill the reward scores.
    """
    return "\n".join(" ".join(ANSWERS[i : i + 16]) for i in range(0, len(ANSWERS), 16))


def _render_marks(guess: str, marks: tuple[str, ...]) -> str:
    body = " | ".join(
        f"{i + 1} {letter} {mark}" for i, (letter, mark) in enumerate(zip(guess, marks))
    )
    return f"{guess} -> {body}"


class WordleEnv(LiteBaseEnv):
    """Guess a hidden four-letter word from per-letter marks."""

    @classmethod
    def extra_tool_schemas(
        cls,
        selection: list[str] | None = None,
        **ctor_args: Any,
    ) -> list[dict[str, Any]]:
        schemas = super().extra_tool_schemas(selection, **ctor_args)
        unsupported = sorted(
            tool_schema_name(schema) for schema in schemas if tool_schema_name(schema) != "response"
        )
        if unsupported:
            raise ValueError(
                f"{cls.__name__} cannot execute extra_tools {unsupported}; "
                "only ['response'] is wired"
            )
        return schemas

    def __init__(
        self,
        task_id: str = DEFAULT_TASK_ID,
        *,
        target: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        extra_tools: list[str] | None = None,
    ) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise TypeError(f"WordleEnv max_turns must be an integer, got {max_turns!r}")
        if max_turns < 1:
            raise ValueError(f"WordleEnv max_turns must be >= 1, got {max_turns}")
        self._task_id = task_id
        self.max_turns = max_turns
        self._extra_tools = list(extra_tools or [])
        self._extra_tool_schemas = type(self).extra_tool_schemas(self._extra_tools)
        if target is not None:
            if not re.fullmatch(rf"[A-Za-z]{{{WORD_LENGTH}}}", target):
                raise ValueError(
                    f"WordleEnv target must be {WORD_LENGTH} ASCII letters, got {target!r}"
                )
            self._target = target.lower()
        else:
            index = _task_index(task_id)
            if index is None or index >= len(ANSWERS):
                raise KeyError(f"unknown Wordle task_id: {task_id!r}")
            self._target = ANSWERS[index]
        self._attempts = 0
        self._solved = False
        self._history: list[tuple[str, tuple[str, ...]]] = []
        self._best_green = 0
        self._consistent_count = 0
        self._format_count = 0

    def _runtime_metadata(self) -> LiteBaseMetadata:
        return _task_metadata(extra_tool_schemas=self._extra_tool_schemas)

    def _episode_return(self) -> float:
        if self._solved:
            speed = (self.max_turns - self._attempts) / max(self.max_turns - 1, 1)
            return _SOLVE_BASE + _SOLVE_SPEED * speed
        return (
            _W_GREEN * (self._best_green / WORD_LENGTH)
            + _W_CONSISTENCY * (self._consistent_count / self.max_turns)
            + _W_FORMAT * (self._format_count / self.max_turns)
        )

    async def reset(self) -> LiteEnvObservation:
        self._attempts = 0
        self._solved = False
        self._history = []
        self._best_green = 0
        self._consistent_count = 0
        self._format_count = 0
        return LiteEnvObservation(
            image=None,
            text=(
                f"Wordle. Guess the hidden {WORD_LENGTH}-letter word.\n"
                "After each guess you get one mark per letter: "
                f"{CORRECT} (right letter, right position), "
                f"{PRESENT} (right letter, wrong position), "
                f"{ABSENT} (letter not in the word).\n"
                f"You have {self.max_turns} attempts.\n"
                "The answer is one of these words:\n"
                f"{_catalog_block()}"
            ),
            metadata={"env_id": ENV_ID, "task_id": self._task_id},
        )

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        # Already finished. This runs BEFORE ingress, so the incoming call id has
        # to be read straight off the action to pair the rejection back to it.
        if self._solved or self._attempts >= self.max_turns:
            call_id = canonical_result_call_id(actions[0]) if actions else None
            return LiteEnvStepResult(
                results=[make_tool_result(call_id, error="task already finished")],
                reward=0.0,
                terminated=True,
                info={
                    EXECUTED_ACTIONS_INFO_KEY: [],
                    "attempt": self._attempts,
                    "max_turns": self.max_turns,
                    "solved": self._solved,
                },
            )

        metadata = self.metadata
        prepared, feedback = prepare_env_tool_calls(actions, metadata)
        results = [
            make_tool_result(call_id, error=item.message) for call_id, item in feedback.items()
        ]
        executed: list[dict[str, Any]] = []
        self._attempts += 1
        response_seen = False
        scored: _Scored | None = None
        scored_call_id: str | None = None

        for action, call_id in prepared:
            name = action["name"]
            args = action["arguments"]
            executed.append({"call": name, "args": dict(args)})
            if tool_feedback := standalone_tool_call_feedback(
                action,
                type(self).known_standalone_tool_names(),
                metadata.extra_tool_schemas,
            ):
                results.append(make_tool_result(call_id, error=tool_feedback.message))
                continue
            if name != "response":
                results.append(make_tool_result(call_id, error=f"unknown tool: {name}"))
                continue
            if response_seen:
                results.append(
                    make_tool_result(call_id, error="only one response is accepted per attempt")
                )
                continue
            response_seen = True
            scored_call_id = call_id
            guess = _parse_guess(str(args.get("text", "")))
            if guess is not None:
                scored = self._score_guess(guess)

        # The episode scalar is paid once, on the step that ends the episode.
        # GRPO sums step rewards, so paying it on any earlier step multiplies it.
        terminated = self._solved or self._attempts >= self.max_turns
        reward = self._episode_return() if terminated else 0.0
        if response_seen:
            results.append(make_tool_result(scored_call_id, text=self._render_feedback(scored)))
        return LiteEnvStepResult(
            results=results,
            reward=reward,
            terminated=terminated,
            info={
                EXECUTED_ACTIONS_INFO_KEY: executed,
                "attempt": self._attempts,
                "max_turns": self.max_turns,
                "solved": self._solved,
            },
        )

    async def close(self) -> None:
        return None

    def _score_guess(self, guess: str) -> _Scored:
        # Any well-formed guess counts for format_frac. There is no dictionary
        # gate: a word list the model cannot see is a rule it cannot learn, and
        # a curated one is unfair (a 1721-word list rejected 21% of common
        # English words in measurement). Invented words earn nothing anyway —
        # they score no greens and break consistency.
        if _is_consistent(guess, self._history):
            self._consistent_count += 1
        self._format_count += 1
        marks = _mark_guess(guess, self._target)
        self._best_green = max(self._best_green, marks.count(CORRECT))
        self._history.append((guess, marks))
        self._solved = guess == self._target
        return _Scored(guess=guess, marks=marks)

    def _render_feedback(self, scored: _Scored | None) -> str:
        remaining = self.max_turns - self._attempts
        if scored is None:
            if remaining <= 0:
                return f"no guess found; no attempts remain; the word was {self._target}"
            return (
                "no guess found; end your reply with a line of exactly: "
                f"Guess: <{WORD_LENGTH}-letter word>; attempts remaining: {remaining}"
            )
        line = _render_marks(scored.guess, scored.marks)
        if self._solved:
            return f"{line}; solved in {self._attempts} attempts"
        if remaining <= 0:
            return f"{line}; no attempts remain; the word was {self._target}"
        # The format reminder rides on every non-terminal turn, not just the
        # unparseable branch: measured 48.4% of turns produced no usable
        # "Guess:" line even though most replies mentioned a guess.
        return (
            f"{line}; attempts remaining: {remaining}; "
            f"reply with a line of exactly: Guess: <{WORD_LENGTH}-letter word>"
        )


def _task_index(task_id: str) -> int | None:
    match = re.fullmatch(r"word_(\d{6})", task_id)
    return int(match.group(1)) if match else None


def register_wordle(*, split: str = "train", limit: int | None = None) -> None:
    """Register one Wordle task per catalog word.

    ``limit`` exists only for tests and short doc snippets. With no arguments
    the WHOLE catalog registers, identically in the driver, the exporter, and
    every Ray worker — the registration module is imported with no arguments in
    each of them, so a default subset would leave exported task ids unresolvable
    at rollout time.
    """
    words = ANSWERS if limit is None else ANSWERS[:limit]
    _clear_env_registration(ENV_ID)
    register_family(ENV_ID, BackendFamily.PURE)
    registry.set_env_supported_kwargs(ENV_ID, {"extra_tools", "max_turns"})
    for index, word in enumerate(words):
        task_id = f"word_{index:06d}"
        register(
            f"{ENV_ID}@{task_id}",
            WordleEnv,
            split=split,
            metadata=_task_metadata(),
            task_id=task_id,
            target=word,
        )


__all__ = ["ENV_ID", "WORD_LENGTH", "WordleEnv", "register_wordle"]
