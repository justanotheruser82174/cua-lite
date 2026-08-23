"""ScaleCUA final-action reward behavior tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import verify as scalecua_verify


@pytest.mark.parametrize(
    "arguments",
    [
        {"status": "failure"},
        # An OpenAI-style provider passthrough hands ``arguments`` over as a
        # JSON STRING. ``env.step`` rejects that shape at ingress, but
        # ``evaluate_final_fn`` is ALSO called directly (no env) by the
        # oracle/replay harnesses under devs/envs/lite.scalecua/validate/.
        '{"status": "failure"}',
    ],
    ids=["dict_arguments", "json_string_arguments"],
)
@pytest.mark.asyncio
async def test_scalecua_terminal_failure_scores_zero_instead_of_erroring(arguments):
    """A terminal-failure final action must SCORE 0.0, never raise.

    ``evaluate_final_fn`` calls the final-action reader unguarded, so a raise
    there does not degrade the score -- it errors the episode out of the eval
    denominator entirely, which is strictly worse than scoring the explicit
    terminal failure as 0.0.
    """
    task = SimpleNamespace(metadata={"evaluator": {"func": "rule"}})
    actions = [{"name": "terminate", "arguments": arguments}]

    result = await scalecua_verify.evaluate_final_fn(
        task, None, actions=actions, debug=True,
    )

    assert result == (0.0, {"terminal_failure": True, "reason": "explicit_failure"})


@pytest.mark.parametrize(
    "malformed",
    [None, "FAIL", 42, {}, {"name": "terminate"}, {"action_type": "FAIL"}, {"name": ""}],
)
@pytest.mark.asyncio
async def test_scalecua_unreadable_final_action_scores_zero_not_one(malformed, monkeypatch):
    """An UNREADABLE final action must reach the reward AS a failure, not as the
    ABSENCE of one.

    Merely declining to raise is not enough. Making the reader total while
    asking it "is this a failure?" answers *no* for junk, which falls THROUGH to
    metric evaluation -- so a malformed terminal action can score 1.0, which is
    strictly worse than the raise it replaced (a raise is at least visible; being
    counted as a success is not). The metric layer is stubbed to 1.0 here
    precisely so a fallthrough is unmissable.
    """
    task = SimpleNamespace(metadata={"evaluator": {"func": "rule"}})

    async def _perfect_score(*args, debug=False, **kwargs):
        return (1.0, {"reached": "metric evaluation"}) if debug else 1.0

    monkeypatch.setattr(
        scalecua_verify.base_runner, "evaluate_osworld_task", _perfect_score
    )
    monkeypatch.setattr(scalecua_verify, "evaluate_scalecua_task", _perfect_score)

    assert await scalecua_verify.evaluate_final_fn(task, None, actions=[malformed]) == 0.0
    assert await scalecua_verify.evaluate_final_fn(
        task, None, actions=[malformed], debug=True,
    ) == (0.0, {"terminal_failure": True, "reason": "unreadable_final_action"})

    # ...and the opposite polarity is NOT flipped with it: on an infeasible task
    # a match PAYS 1.0, so junk must not be read as a correct infeasible report.
    infeasible = SimpleNamespace(metadata={"evaluator": {"func": "infeasible"}})
    assert await scalecua_verify.evaluate_final_fn(
        infeasible, None, actions=[malformed],
    ) == 0.0


def test_scalecua_final_action_reader_is_total_and_drops_osworld_sentinels():
    """The reader never raises, and OSWorld's wire sentinels are NOT accepted.

    ``"FAIL"`` / ``{"action_type": "FAIL"}`` are the ``osworld``/``osworld_2``
    envs' container encoding (they POST ``{"cmd": "FAIL"}``), unrepresentable as
    a canonical Lite tool call and never emitted on this env's path. They
    are dropped, but dropping is not silent: an unread name is ``None``, which
    ``_final_action_forfeit_reason`` turns into a 0.0.
    """
    for unreadable in ("FAIL", {"action_type": "FAIL"}, 42, {}, {"name": "terminate"}):
        assert scalecua_verify._action_call(unreadable) == (None, {})
        # POSITIVE predicate: unreadable is not a declared failure...
        assert scalecua_verify._action_is_failure(unreadable) is False
        # ...but it still forfeits credit before metric evaluation.
        assert (
            scalecua_verify._final_action_forfeit_reason([unreadable])
            == "unreadable_final_action"
        )

    assert scalecua_verify._action_is_failure({"name": "report_infeasible", "arguments": {}})
    assert scalecua_verify._final_action_forfeit_reason(
        [{"name": "terminate", "arguments": {"status": "failure"}}]
    ) == "explicit_failure"
    assert scalecua_verify._final_action_forfeit_reason([]) is None
    assert scalecua_verify._final_action_forfeit_reason(
        [{"name": "terminate", "arguments": {"status": "success"}}]
    ) is None
    assert scalecua_verify._reported_infeasible(
        [{"name": "response", "arguments": '{"text": "[infeasible] no such field"}'}]
    )


@pytest.mark.asyncio
async def test_scalecua_reward_reader_descends_canonical_final_action():
    """ScaleCUA's OSWorld reward boundary descends nested canonical calls."""
    from lite.core.tools import make_tool_call

    task = SimpleNamespace(metadata={"evaluator": {"func": "infeasible"}})
    actions = [
        make_tool_call(
            "report_infeasible",
            {"reason": "no app"},
            call_id="call_report",
        )
    ]

    assert await scalecua_verify.evaluate_final_fn(
        task,
        None,
        actions,
        debug=True,
    ) == (1.0, {"infeasible": True})


class _FakeInterface:
    def __init__(
        self,
        stdout: str | list[str] = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.commands: list[str] = []
        self.command_calls: list[dict[str, object]] = []
        self.hotkeys: list[tuple[str, ...]] = []
        self.typed_text: list[str] = []
        self.stdout = stdout
        self.files = files or {}

    async def read_bytes(self, path: str) -> bytes:
        if path in self.files:
            return self.files[path]
        return f"bytes:{path}".encode()

    async def screenshot(self) -> bytes:
        return b"png"

    async def get_screen_size(self):
        return {"width": 800, "height": 600}

    async def run_command(self, command: str, timeout=None):
        self.commands.append(command)
        self.command_calls.append({"command": command, "timeout": timeout})
        if isinstance(self.stdout, list):
            stdout = self.stdout.pop(0) if self.stdout else ""
        else:
            stdout = self.stdout
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    async def hotkey(self, *keys: str):
        self.hotkeys.append(tuple(keys))

    async def type_text(self, text: str):
        self.typed_text.append(text)


class _FakeComputer:
    def __init__(
        self,
        stdout: str = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.interface = _FakeInterface(stdout=stdout, files=files)


@pytest.mark.asyncio
async def test_scalecua_non_infeasible_last_fail_short_circuits(monkeypatch):
    async def should_not_evaluate(*args, **kwargs):
        raise AssertionError("non-infeasible FAIL should short-circuit before eval")

    monkeypatch.setattr(scalecua_verify, "evaluate_scalecua_task", should_not_evaluate)
    task = SimpleNamespace(
        metadata={
            "evaluator": {"func": "exact_match"},
            "scalecua": {"runtime_split": "train"},
        }
    )

    reward = await scalecua_verify.evaluate_final_fn(
        task,
        _FakeComputer(),
        actions=[{"name": "terminate", "arguments": {"status": "failure"}}],
    )

    assert reward == 0.0


@pytest.mark.asyncio
async def test_scalecua_postconfig_runs_before_terminal_failure(monkeypatch):
    calls = []

    async def fake_run_postconfig(computer, evaluator, cache_dir):
        calls.append((computer, evaluator, cache_dir))

    async def should_not_evaluate(*args, **kwargs):
        raise AssertionError("terminal failure should not run metrics")

    monkeypatch.setattr(scalecua_verify, "_run_postconfig", fake_run_postconfig)
    monkeypatch.setattr(scalecua_verify, "evaluate_scalecua_task", should_not_evaluate)
    task = SimpleNamespace(
        metadata={
            "evaluator": {
                "func": "exact_match",
                "postconfig": [{"type": "sleep", "parameters": {"seconds": 0}}],
            },
            "scalecua": {"runtime_split": "train"},
        }
    )

    reward = await scalecua_verify.evaluate_final_fn(
        task,
        _FakeComputer(),
        actions=[{"name": "terminate", "arguments": {"status": "failure"}}],
    )

    assert reward == 0.0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_scalecua_eval_split_runs_postconfig_once_before_base_eval(monkeypatch):
    calls = []

    async def fake_run_postconfig(computer, evaluator, cache_dir):
        calls.append((computer, evaluator, cache_dir))
        evaluator["_postconfig_done"] = True

    async def fake_base_eval(computer, evaluator, cache_dir=None, debug=False):
        assert evaluator["_postconfig_done"] is True
        assert cache_dir
        calls.append(("base", cache_dir))
        return 1.0

    monkeypatch.setattr(scalecua_verify, "_run_postconfig", fake_run_postconfig)
    monkeypatch.setattr(
        scalecua_verify.base_runner,
        "evaluate_osworld_task",
        fake_base_eval,
    )
    task = SimpleNamespace(
        metadata={
            "evaluator": {
                "func": "exact_match",
                "postconfig": [{"type": "sleep", "parameters": {"seconds": 0}}],
            },
            "scalecua": {"runtime_split": "eval"},
        }
    )

    reward = await scalecua_verify.evaluate_final_fn(task, _FakeComputer())

    assert reward == 1.0
    assert len(calls) == 2
    assert calls[0][2] == calls[1][1]


@pytest.mark.asyncio
async def test_scalecua_eval_respects_postconfig_done_marker(monkeypatch):
    calls = []

    async def should_not_run_postconfig(*args, **kwargs):
        raise AssertionError("postconfig is already done")

    async def fake_eval(computer, evaluator, **kwargs):
        calls.append(kwargs)
        assert evaluator["_postconfig_done"] is True
        assert kwargs["run_postconfig"] is False
        assert kwargs["cache_dir"]
        return 1.0

    monkeypatch.setattr(scalecua_verify, "_run_postconfig", should_not_run_postconfig)
    monkeypatch.setattr(scalecua_verify, "evaluate_scalecua_task", fake_eval)
    task = SimpleNamespace(
        metadata={
            "evaluator": {
                "func": "exact_match",
                "postconfig": [{"type": "sleep", "parameters": {"seconds": 0}}],
                "_postconfig_done": True,
            },
            "scalecua": {"runtime_split": "rl"},
        }
    )

    reward = await scalecua_verify.evaluate_final_fn(task, _FakeComputer())

    assert reward == 1.0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_scalecua_infeasible_accepts_flat_dict_arguments():
    task = SimpleNamespace(metadata={"evaluator": {"func": "infeasible"}})

    reward = await scalecua_verify.evaluate_final_fn(
        task,
        _FakeComputer(),
        actions=[
            {
                "name": "terminate",
                "arguments": {"status": "failure"},
            }
        ],
    )
    response_reward = await scalecua_verify.evaluate_final_fn(
        task,
        _FakeComputer(),
        actions=[
            {
                "name": "response",
                "arguments": {"text": "[infeasible] blocked"},
            }
        ],
    )

    assert reward == 1.0
    assert response_reward == 1.0


@pytest.mark.asyncio
async def test_scalecua_infeasible_scores_malformed_actions_instead_of_erroring():
    """A malformed final action is NOT a terminal failure — and never raises.

    This helper is called unguarded from ``evaluate_final_fn``, so raising does
    not degrade the score, it errors the episode out of the eval denominator.
    ``0.0`` (the task was not reported infeasible) is the answer; the malformed
    envelope is already caught, with model feedback, at ``env.step`` ingress.
    """
    task = SimpleNamespace(metadata={"evaluator": {"func": "infeasible"}})

    reward = await scalecua_verify.evaluate_final_fn(
        task,
        _FakeComputer(),
        actions=[{"name": "terminate"}],
    )

    assert reward == 0.0


@pytest.mark.asyncio
async def test_scalecua_infeasible_ignores_historical_failures():
    task = SimpleNamespace(metadata={"evaluator": {"func": "infeasible"}})

    reward = await scalecua_verify.evaluate_final_fn(
        task,
        _FakeComputer(),
        actions=[
            {
                "name": "terminate",
                "arguments": {"status": "failure"},
            },
            {"name": "response", "arguments": {"text": "done"}},
        ],
    )

    assert reward == 0.0
