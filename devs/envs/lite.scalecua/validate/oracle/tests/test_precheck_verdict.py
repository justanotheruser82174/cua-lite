"""The lite.scalecua oracle precheck must read the CHECKER, not the episode.

``env.step``'s ``reward`` is ``None`` whenever the step was not terminal -- a
non-verdict, kept distinct on purpose so nothing coalesces it into ``0.0``. The
precheck used to route a ``response`` call through ``env.step`` and float that
value; because finish tools are gated by ``env_kwargs.extra_tools`` (opt-in,
default ``[]``) the call was swallowed as an inactive extra tool, the step never
terminated, and every fixture of a sweep died in
``float() argument must be ... not 'NoneType'`` with no verdict recorded
anywhere. ``evaluate_final_fn`` answers the question the precheck actually asks
and returns a float on every path, so "no verdict" is unrepresentable once it is
the only channel.

Run:
    uv run pytest devs/envs/lite.scalecua/validate/oracle/tests/test_precheck_verdict.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

VALIDATE = Path(__file__).resolve().parents[1] / "validate.py"
VERIFY_MODULE = "lite.gym.envs.lite.scalecua.src.osworld.verify"


def _load() -> ModuleType:
    """Import the validator by path -- ``devs/envs/lite.scalecua`` is not a
    package (its name is not a legal Python identifier)."""
    spec = importlib.util.spec_from_file_location(
        "_scalecua_oracle_validate_precheck", VALIDATE,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubEnv:
    """An env that WOULD hand back a non-verdict if anyone asked it to step."""

    def __init__(self) -> None:
        self.step_calls: list[Any] = []
        self.closed = False
        self._task = SimpleNamespace(metadata={"evaluator": {"func": "exact_match"}})
        self._computer = object()

    async def reset(self):
        return SimpleNamespace(image=b"reset-png")

    async def step(self, actions):
        self.step_calls.append(actions)
        return SimpleNamespace(reward=None, terminated=False, truncated=False, info={})

    async def close(self) -> None:
        self.closed = True


def _run_precheck(module, monkeypatch, tmp_path, *, evaluator_score, fixture=None):
    """Drive ``_run_noop_precheck`` with no container and no catalog."""
    import lite.gym as gym

    env = _StubEnv()
    monkeypatch.setattr(gym, "make", lambda *a, **k: env)
    monkeypatch.setitem(
        sys.modules,
        VERIFY_MODULE,
        SimpleNamespace(
            evaluate_final_fn=_recording_eval(evaluator_score),
        ),
    )
    monkeypatch.setattr(
        module, "_find_task", lambda split, task_id: {"instruction": "do the thing"},
    )
    monkeypatch.setattr(module, "_write_png", lambda path, data: None)

    async def _no_screenshot(_env, _path):
        return None

    monkeypatch.setattr(module, "_capture_png", _no_screenshot)

    result = asyncio.run(
        module._run_noop_precheck(
            fixture if fixture is not None else {"split": "rl", "task_id": "t1"},
            artifacts=tmp_path,
            reset_timeout=1.0,
        )
    )
    return result, env


def _recording_eval(score):
    calls: list[dict[str, Any]] = []

    async def _evaluate_final_fn(task, computer, actions=None, debug=False):
        calls.append({"task": task, "actions": actions, "debug": debug})
        return score

    _evaluate_final_fn.calls = calls
    return _evaluate_final_fn


def test_precheck_verdict_comes_from_the_evaluator_not_from_env_step(
    monkeypatch, tmp_path,
):
    module = _load()
    result, env = _run_precheck(module, monkeypatch, tmp_path, evaluator_score=0.0)

    assert result["reward"] == 0.0
    assert result["passed"] is True
    # THE regression: one call to ``env.step`` reintroduces the ``None`` channel.
    assert env.step_calls == [], "precheck stepped the env; reward can be None again"
    assert env.closed is True


def test_precheck_reports_a_trivial_pass_rather_than_raising(monkeypatch, tmp_path):
    """A checker that already scores is a FAILED precheck -- never an exception."""
    module = _load()
    result, _env = _run_precheck(module, monkeypatch, tmp_path, evaluator_score=1.0)

    assert result["reward"] == 1.0
    assert result["passed"] is False


def test_precheck_honours_expected_pre_reward(monkeypatch, tmp_path):
    module = _load()
    result, _env = _run_precheck(
        module, monkeypatch, tmp_path, evaluator_score=0.5,
        fixture={"split": "rl", "task_id": "t1", "expected_pre_reward": 0.5},
    )

    assert result["passed"] is True


def test_response_action_is_the_env_internal_projection():
    """The direct oracle evaluator call receives the env-internal action shape."""
    module = _load()

    assert module._response_action("hello") == [
        {"name": "response", "arguments": {"text": "hello"}},
    ]


def test_reward_matches_is_only_ever_fed_a_real_number(monkeypatch, tmp_path):
    """Pins the crash this file exists for: ``None`` must never reach ``float()``."""
    module = _load()

    with pytest.raises(TypeError):
        module._reward_matches(None, 0.0)

    result, _env = _run_precheck(module, monkeypatch, tmp_path, evaluator_score=0.0)
    assert isinstance(result["reward"], float)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
