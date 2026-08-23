"""Tests for the Wordle example env."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from examples.wordle.env import (
    ENV_ID,
    WORD_LENGTH,
    WordleEnv,
    _is_consistent,
    _mark_guess,
    _parse_guess,
    register_wordle,
)
from examples.wordle.words import ANSWERS
from lite.agents.core.agent.base import AdapterBasedAgent
from lite.core import LiteGenericMetadata
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.gym import registry
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY

_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_MODULE = importlib.import_module("lite.gym.registry")


class _WordleFakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False) -> str:
        del add_generation_prompt, tokenize
        return f"<prompt {len(messages)} messages>"


class _WordleScriptedTextAdapter:
    _registry_key = "wordle.fake"
    metadata = LiteGenericMetadata()

    def __init__(self, turns: list[str | list[dict[str, object]]]) -> None:
        self._turns = turns
        self._i = 0

    @classmethod
    def get_registry_key(cls) -> str:
        return cls._registry_key

    def render_step(self, lite_sample, k: int, processed_images) -> list[dict[str, object]]:
        del k, processed_images
        return list(lite_sample.messages)

    def process_image(self, img):
        return img

    def parse_raw_assistant_response(self, response: str) -> dict[str, object]:
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    def convert_message_from_agent(self, agent_message) -> dict[str, object]:
        del agent_message
        turn = self._turns[self._i] if self._i < len(self._turns) else ""
        self._i += 1
        if isinstance(turn, list):
            return {"role": "assistant", "content": [], "tool_calls": turn}
        return {"role": "assistant", "content": [{"type": "text", "text": turn}]}


@pytest.fixture(autouse=True)
def _local_registry_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _clear_wordle_registry():
    _REGISTRY_MODULE._clear_env_registration(ENV_ID)
    yield
    _REGISTRY_MODULE._clear_env_registration(ENV_ID)


async def _guess(env: WordleEnv, word: str):
    # The no-tool path: runtime turns visible text into a transient response
    # action carrying the content_only_final sidecar, which bypasses the
    # extra-tool gate. A bare make_tool_call would be an EXPLICIT tool call.
    return await env.step(make_no_tool_call_final_actions(f"reasoning\nGuess: {word}"))


async def _say(env: WordleEnv, text: str):
    return await env.step(make_no_tool_call_final_actions(text))


def _agent(turns: list[str | list[dict[str, object]]]) -> AdapterBasedAgent:
    async def _generate(**kwargs):
        del kwargs
        return {"response": "ignored by scripted adapter"}

    return AdapterBasedAgent(
        generate_fn=_generate,
        processor=_WordleFakeProcessor(),
        adapter=_WordleScriptedTextAdapter(turns),
    )


# --------------------------------------------------------------------------- #
# Marking and consistency
# --------------------------------------------------------------------------- #


def test_wordle_marks_all_correct_for_exact_guess() -> None:
    assert _mark_guess("care", "care") == ("correct",) * WORD_LENGTH


@pytest.mark.parametrize(
    ("target", "guess", "expected"),
    [
        ("alee", "eela", ("present", "present", "present", "present")),
        ("abbe", "babe", ("present", "present", "correct", "correct")),
        ("abac", "aaaa", ("correct", "absent", "correct", "absent")),
    ],
)
def test_wordle_marking_golden_cases(target, guess, expected) -> None:
    assert _mark_guess(guess, target) == expected


def test_wordle_consistency_accepts_target_after_duplicate_letter_guess() -> None:
    # Naive constraint accumulation records "l correct at 2" AND "l absent",
    # which rejects the true target and silently zeroes all later shaping.
    history = [("eell", _mark_guess("eell", "alee"))]
    assert _is_consistent("alee", history) is True


def test_wordle_repeated_non_solving_guess_is_inconsistent() -> None:
    history = [("dawn", _mark_guess("dawn", "care"))]
    assert _is_consistent("dawn", history) is False


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Guess: dawn", "dawn"),
        ("Guess: dawn.", "dawn"),
        ("**Guess: DAWN**", "dawn"),
        ("Guess: `dawn`", "dawn"),
        ("Best guess: dawn", "dawn"),
        ("Guess: dawn or milk", "dawn"),
        ("Guess: dawn\nGuess: milk", "milk"),
        ("Guess: d a w n", None),
        ("Guess: dawŷn", None),
        ("", None),
    ],
)
def test_wordle_parse_guess_cases(text, expected) -> None:
    assert _parse_guess(text) == expected


def test_wordle_ignores_same_length_words_in_reasoning_prose() -> None:
    # No positional fallback: "about"/"words"/"think" must not become guesses.
    assert _parse_guess("I think about which words might fit") is None


# --------------------------------------------------------------------------- #
# Env semantics
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_wordle_non_terminal_feedback_never_contains_target() -> None:
    env = WordleEnv(target="care", max_turns=6)
    await env.reset()
    for word in ("dawn", "milk", "fish", "tree", "road"):
        result = await _guess(env, word)
        assert result.terminated is False
        text = result.results[-1].text or ""
        assert "care" not in text
        assert "target" not in result.info


@pytest.mark.asyncio
async def test_wordle_invented_word_is_marked_with_no_dictionary_gate() -> None:
    # There is no allowed-guess list: a word list the model cannot see is a rule
    # it cannot learn. Invented words are marked like any other well-formed guess.
    env = WordleEnv(target="care", max_turns=6)
    await env.reset()
    result = await _guess(env, "zzzz")
    text = result.results[-1].text or ""
    assert "zzzz ->" in text
    assert "word list" not in text
    assert result.info["attempt"] == 1
    assert result.reward == 0.0


@pytest.mark.asyncio
async def test_wordle_unparseable_response_consumes_attempt() -> None:
    env = WordleEnv(target="care", max_turns=6)
    await env.reset()
    result = await _say(env, "I am not sure")
    assert result.info["attempt"] == 1
    assert result.terminated is False
    assert result.reward == 0.0
    assert "no guess found" in (result.results[-1].text or "")


@pytest.mark.asyncio
async def test_wordle_dropping_the_guess_line_never_pays_more_than_formatting() -> None:
    # The runtime sums step rewards into episode_return, so the episode scalar
    # may only be paid on the terminating step. Paying it on non-terminal turns
    # would make an episode worth MORE for failing to format.
    async def _episode(replies: list[str]) -> float:
        env = WordleEnv(target="able", max_turns=6)
        await env.reset()
        total = 0.0
        for reply in replies:
            total += (await _say(env, reply)).reward
        return total

    guesses = ["acid", "bore", "dune", "glim", "hunk", "jolt"]
    formatted = await _episode([f"Guess: {w}" for w in guesses])
    dropped = await _episode(
        ["Guess: acid", "I think bore", "Guess: dune", "maybe", "Guess: glim", "hmm"]
    )
    assert dropped < formatted


@pytest.mark.asyncio
async def test_wordle_solve_terminates_before_attempts_are_exhausted() -> None:
    env = WordleEnv(target="care", max_turns=6)
    await env.reset()
    await _guess(env, "dawn")
    result = await _guess(env, "care")
    assert result.terminated is True
    assert result.info["solved"] is True
    assert result.info["attempt"] == 2
    assert "solved in 2 attempts" in (result.results[-1].text or "")


@pytest.mark.asyncio
async def test_wordle_terminates_when_attempts_are_exhausted() -> None:
    env = WordleEnv(target="care", max_turns=2)
    await env.reset()
    first = await _guess(env, "dawn")
    second = await _guess(env, "milk")
    assert first.terminated is False
    assert second.terminated is True
    assert second.info["solved"] is False
    assert "the word was care" in (second.results[-1].text or "")


@pytest.mark.asyncio
async def test_wordle_post_terminal_step_is_inert_after_solve() -> None:
    env = WordleEnv(target="care", max_turns=6)
    await env.reset()
    await _guess(env, "care")
    after = await _guess(env, "dawn")
    assert after.results[0].error == "task already finished"
    assert after.info[EXECUTED_ACTIONS_INFO_KEY] == []
    assert after.reward == 0.0
    assert after.info["attempt"] == 1


@pytest.mark.asyncio
async def test_wordle_max_turns_one_terminates_on_first_guess() -> None:
    env = WordleEnv(target="care", max_turns=1)
    await env.reset()
    result = await _guess(env, "dawn")
    assert result.terminated is True
    assert "no attempts remain" in (result.results[-1].text or "")


@pytest.mark.parametrize("max_turns", [True, 1.5, "3"])
def test_wordle_rejects_non_integer_max_turns(max_turns) -> None:
    with pytest.raises(TypeError, match="max_turns must be an integer"):
        WordleEnv(target="care", max_turns=max_turns)


def test_wordle_rejects_nonpositive_max_turns() -> None:
    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        WordleEnv(target="care", max_turns=0)


@pytest.mark.asyncio
async def test_wordle_rejects_second_response_in_one_step() -> None:
    env = WordleEnv(target="care", max_turns=6, extra_tools=["response"])
    await env.reset()
    result = await env.step(
        [
            make_tool_call("response", {"text": "Guess: dawn"}, call_id="call_0"),
            make_tool_call("response", {"text": "Guess: care"}, call_id="call_1"),
        ]
    )
    errors = [r.error for r in result.results if r.error]
    assert "only one response is accepted per attempt" in errors
    assert result.info["attempt"] == 1
    assert result.info["solved"] is False


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #


async def _solve_on(attempt: int, max_turns: int = 6) -> float:
    env = WordleEnv(target="care", max_turns=max_turns)
    await env.reset()
    total = 0.0
    fillers = ["dawn", "milk", "fish", "tree", "road"]
    for i in range(attempt - 1):
        total += (await _guess(env, fillers[i])).reward
    total += (await _guess(env, "care")).reward
    return total


@pytest.mark.asyncio
async def test_wordle_episode_return_is_strictly_decreasing_in_attempts_to_solve() -> None:
    returns = [await _solve_on(k) for k in range(1, 7)]
    assert returns == sorted(returns, reverse=True)
    assert len(set(round(r, 6) for r in returns)) == 6
    assert [round(r, 2) for r in returns] == [1.50, 1.40, 1.30, 1.20, 1.10, 1.00]


@pytest.mark.asyncio
async def test_wordle_solved_return_is_at_least_one_and_unsolved_is_below_one() -> None:
    # R >= 1.0 <=> solved is what makes solve rate recoverable from logged rewards.
    solved = [await _solve_on(k) for k in range(1, 7)]
    assert min(solved) >= 1.0

    env = WordleEnv(target="care", max_turns=6)
    await env.reset()
    unsolved = 0.0
    for word in ("dawn", "milk", "fish", "tree", "road", "gold"):
        unsolved += (await _guess(env, word)).reward
    assert unsolved < 1.0


@pytest.mark.asyncio
async def test_wordle_max_turns_one_solve_does_not_divide_by_zero() -> None:
    env = WordleEnv(target="care", max_turns=1)
    await env.reset()
    result = await _guess(env, "care")
    assert result.reward == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Loop integration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_wordle_agent_loop_continues_natural_language_guesses() -> None:
    env = WordleEnv(target="care", max_turns=6)
    agent = _agent(["Guess: dawn", "Guess: milk", "Guess: care"])

    sample = await agent.sample(env, max_steps=10)
    messages = sample.lite_sample.messages

    assert [m["role"] for m in messages] == ["user", "assistant"] * 3
    assert not any(m.get("role") == "tool" for m in messages)
    assert len(sample.steps) == 3
    assert sample.terminated is True
    blob = json.dumps(messages)
    assert "dawn ->" in blob
    assert "care" not in blob.split("Guess: care")[0]


@pytest.mark.asyncio
async def test_wordle_agent_loop_handles_explicit_response_tool() -> None:
    env = WordleEnv(target="care", max_turns=3, extra_tools=["response"])
    agent = _agent(
        [
            [make_tool_call("response", {"text": "Guess: dawn"}, call_id="call_0")],
            [make_tool_call("response", {"text": "Guess: care"}, call_id="call_1")],
        ]
    )
    sample = await agent.sample(env, max_steps=10)
    roles = [m["role"] for m in sample.lite_sample.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]


def test_wordle_extra_tools_only_exposes_response() -> None:
    with pytest.raises(ValueError, match=r"only \['response'\] is wired"):
        WordleEnv(target="care", extra_tools=["terminate"])


# --------------------------------------------------------------------------- #
# Catalog and registration
# --------------------------------------------------------------------------- #


def test_wordle_answers_are_lowercase_four_letters_sorted_and_unique() -> None:
    assert len(ANSWERS) == 256
    assert len(set(ANSWERS)) == len(ANSWERS)
    assert list(ANSWERS) == sorted(ANSWERS)
    assert all(re.fullmatch(r"[a-z]{4}", w) for w in ANSWERS)


def test_wordle_parser_falls_back_to_an_earlier_guess_line() -> None:
    # Measured: 17 of 55 unparseable 2B replies named a wrong-length word on the
    # last "guess:" line while an earlier line carried a valid four-letter word.
    assert _parse_guess("Guess: care\nrambling\nGuess: Cagey") == "care"
    assert _parse_guess("Guess: cagey\nGuess: appoint") is None


def test_wordle_registers_one_task_per_answer() -> None:
    register_wordle()
    ids = registry.task_ids(ENV_ID, split="train")
    assert len(ids) == len(ANSWERS)
    assert ids[0] == "word_000000"
    assert ids[-1] == f"word_{len(ANSWERS) - 1:06d}"


def test_wordle_registry_declares_runtime_env_kwargs() -> None:
    register_wordle()
    assert set(registry.env_supported_kwargs(ENV_ID)) >= {"extra_tools", "max_turns"}
    assert registry.env_supports_kwarg(ENV_ID, "max_turns")


def test_wordle_metadata_is_generic_with_empty_dims() -> None:
    env = WordleEnv(target="care")
    assert isinstance(env.metadata, LiteGenericMetadata)
    assert env.metadata.dims == ()


def test_wordle_unknown_task_id_raises() -> None:
    with pytest.raises(KeyError, match="unknown Wordle task_id"):
        WordleEnv(task_id="word_999999")


def test_wordle_registration_module_registers_full_catalog() -> None:
    # The driver, the exporter, and every Ray worker import this with no
    # arguments; a subset here would leave exported task ids unresolvable.
    code = (
        "import examples.wordle.registration;"
        "from lite.gym import registry;"
        "print(len(registry.task_ids('wordle', split='train')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == str(len(ANSWERS))


def test_wordle_export_tasks_help_does_not_touch_env_server() -> None:
    env = os.environ.copy()
    env["CUA_LITE_ENV_SERVER_URL"] = "http://127.0.0.1:39999"
    env["CUA_LITE_ENV_SERVER_TOKEN"] = "stale-token"

    result = subprocess.run(
        [sys.executable, "-m", "examples.wordle.export_tasks", "--help"],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "Create a parquet task list" in result.stdout


def test_wordle_grpo_wrapper_execs_canonical_with_example_contract(tmp_path) -> None:
    root = tmp_path / "cua-lite"
    wrapper = root / "examples" / "wordle" / "scripts" / "run_grpo.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text((_ROOT / "examples" / "wordle" / "scripts" / "run_grpo.sh").read_text())
    wrapper.chmod(0o755)

    canonical = root / "scripts" / "train" / "run_grpo.sh"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$MODEL_ID\" \"$ROLLOUT_MODULE\" \"$CONFIG_PATH\" \"$PROMPT_DATA\" "
        "\"$CUA_LITE_REGISTRATION_MODULES\" \"$NUM_TRAIN_GPUS\" "
        "\"$ROLLOUT_BATCH_SIZE\" \"$N_SAMPLES_PER_PROMPT\" "
        "\"$NUM_STEPS_PER_ROLLOUT\" \"$ENV_CONCURRENCY\" "
        "\"$NUM_ROLLOUT_GPUS\" \"${CUA_LITE_ENV_SERVER_URL:-unset}\" "
        "\"${CUA_LITE_ENV_SERVER_TOKEN:-unset}\" > \"$FAKE_CAPTURE\"\n"
    )
    canonical.chmod(0o755)

    models = root / "scripts" / "train" / "utils" / "models.sh"
    models.parent.mkdir(parents=True)
    models.write_text(
        'case "$MODEL_ID" in\n'
        '  Qwen/Qwen3.5*) MODEL_FAMILY=qwen3_5 ;;\n'
        '  *) MODEL_FAMILY=qwen3_vl ;;\n'
        "esac\n"
    )

    config = root / "examples" / "wordle" / "configs" / "qwen3_vl" / "wordle.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("agent_id: qwen3_vl.base\nenv_id: wordle\n")
    prompt = root / "prompt.parquet"
    prompt.write_text("fake")

    env = os.environ.copy()
    env.update({
        "CUA_LITE_ROOT": str(tmp_path / "stale-root"),
        "CUA_LITE_REGISTRATION_MODULES": "extra.registration",
        "PROMPT_DATA": str(prompt),
        "FAKE_CAPTURE": str(root / "capture.txt"),
        "NUM_TRAIN_GPUS": "2",
        "NUM_ROLLOUT_GPUS": "2",
        "CUA_LITE_ENV_SERVER_URL": "http://127.0.0.1:39999",
        "CUA_LITE_ENV_SERVER_TOKEN": "stale-token",
    })

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert (root / "capture.txt").read_text().splitlines() == [
        "Qwen/Qwen3-VL-2B-Instruct",
        "lite.train.rollout.grpo",
        str(config),
        str(prompt),
        "extra.registration,examples.wordle.registration",
        "2",
        "32",
        "8",
        "1",
        "256",
        "2",
        "unset",
        "unset",
    ]


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("family", "agent_id"),
    [("qwen3_5", "qwen3_5.base"), ("qwen3_vl", "qwen3_vl.base")],
)
def test_wordle_real_configs_are_no_tool_natural_language(family, agent_id) -> None:
    import yaml

    path = _ROOT / "examples" / "wordle" / "configs" / family / "wordle.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["agent_id"] == agent_id
    assert cfg["agent_kwargs"]["render_tools_section"] is False
    assert "extra_tools" not in cfg["agent_kwargs"]
    assert "Guess:" in cfg["agent_kwargs"]["system_prompt"]
    assert cfg["env_id"] == ENV_ID
    assert cfg["env_kwargs"]["max_turns"] == 6
