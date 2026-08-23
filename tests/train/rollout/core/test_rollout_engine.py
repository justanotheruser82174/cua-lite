"""Stress tests for CUA-Lite RL rollout pipeline.

Tests the rollout pipeline components: generate_fn → on_step → Sample building,
convert_samples_to_train_data, and env fault isolation.

Usage:
    pytest tests/train/rollout/core/test_rollout_engine.py -v
    # Inside Slime container:
    pytest tests/train/rollout/core/test_rollout_engine.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("slime.utils", reason="slime not installed")
from slime.utils.types import Sample  # noqa: E402  (must follow importorskip)

from lite.train.rollout.core.engine import (  # noqa: E402
    AbortError,
    _active_envs,
    _empty_sample,
    _register_env,
    _unregister_env,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockArgs:
    """Minimal args for convert_samples_to_train_data."""
    n_samples_per_prompt: int = 4
    # v0.3.0: ``flatten_and_align`` pads the trajectory list up to
    # ``n_expected = rollout_batch_size * n_samples_per_prompt`` with
    # zero-gradient dummy GROUPS (the old segment-level GBS trim is gone).
    # Each test therefore sets ``rollout_batch_size`` to its real prompt count
    # so ``n_expected`` equals the post-filter trajectory count → no padding →
    # the exact sample-count assertions hold. ``calculate_per_token_loss`` is
    # left unset (getattr default False) as GRPO's converter asserts it False.
    rollout_batch_size: int = 1
    advantage_estimator: str = "grpo"
    rewards_normalization: bool = True
    grpo_std_normalization: bool = True

def make_samples(
    n_prompts: int,
    n_samples: int,
    n_turns: int,
    terminal_reward_fn=None,
) -> list[Sample]:
    """Create flat list of turn samples for testing.

    terminal_reward_fn(prompt_idx, sample_idx) -> float
    """
    if terminal_reward_fn is None:
        def _default_reward(pi, si):
            return float(pi * n_samples + si + 1)
        terminal_reward_fn = _default_reward

    samples = []
    for pi in range(n_prompts):
        for si in range(n_samples):
            idx = pi * n_samples + si
            episode_return = terminal_reward_fn(pi, si)
            for turn in range(n_turns):
                is_terminal = turn == n_turns - 1
                reward = episode_return if is_terminal else 0.0
                s = Sample(
                    group_index=pi,
                    index=idx,
                    tokens=list(range(10 + turn * 5)) + list(range(5)),
                    loss_mask=[1] * 5,
                    rollout_log_probs=[-0.1] * 5,
                    response_length=5,
                    response=f"resp_{pi}_{si}_{turn}",
                    reward=reward,
                    status=Sample.Status.COMPLETED,
                    metadata={
                        "turn_range": (turn, turn),
                        "env_key": f"test@task_{pi}",
                        "others": {"episode_return": episode_return},
                    },
                )
                samples.append(s)
    return samples


def make_single_turn_sample(
    *,
    response: str,
    episode_return: float = 0.0,
    tool_error: str | None = None,
) -> Sample:
    metadata = {
        "turn_range": (0, 0),
        "env_key": "test@task",
        "others": {"episode_return": episode_return},
    }
    if tool_error is not None:
        metadata["tool_result_error"] = tool_error
    return Sample(
        group_index=0,
        index=0,
        group_id=0,
        prompt="orig",
        tokens=[10, 20, 30],
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        response_length=1,
        response=response,
        reward=episode_return,
        status=Sample.Status.COMPLETED,
        metadata=metadata,
    )

# ---------------------------------------------------------------------------
# Tests: _emit_rl_sample (covered by tests/train/rollout/core/test_length1_equivalence.py
# and tests/train/rollout/core/test_radix_segmenter.py — exercising the new
# segment-centric API directly).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tests: _empty_sample
# ---------------------------------------------------------------------------

class TestEmptySample:
    def test_basic(self):
        orig = Sample(group_index=1, index=5, prompt="p", label="l",
                      metadata={"env_key": "lite.demo@create_file"})
        es = _empty_sample(orig)
        assert es.group_index == 1
        assert es.index == 5
        assert es.tokens == []
        assert es.reward == 0.0
        assert es.status == Sample.Status.FAILED
        assert es.response_length == 0

# ---------------------------------------------------------------------------
# Tests: env registry
# ---------------------------------------------------------------------------

class TestEnvRegistry:
    def setup_method(self):
        _active_envs.clear()

    def test_register_unregister(self):
        env = object()
        _register_env(env)
        assert env in _active_envs
        _unregister_env(env)
        assert env not in _active_envs

    def test_double_unregister(self):
        env = object()
        _register_env(env)
        _unregister_env(env)
        _unregister_env(env)  # should not raise
        assert env not in _active_envs

    def test_multiple_envs(self):
        envs = [object() for _ in range(10)]
        for e in envs:
            _register_env(e)
        assert len(_active_envs) == 10
        for e in envs:
            _unregister_env(e)
        assert len(_active_envs) == 0

# ---------------------------------------------------------------------------
# Tests: convert_samples_to_train_data
# ---------------------------------------------------------------------------

class TestAbortError:
    def test_is_exception(self):
        assert issubclass(AbortError, Exception)

    def test_can_raise_catch(self):
        with pytest.raises(AbortError):
            raise AbortError()
