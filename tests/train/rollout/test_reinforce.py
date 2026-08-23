"""Tests for lite.train.rollout.reinforce.convert_samples_to_train_data.

Tests the REINFORCE / filtered-BC specific conversion logic: trajectory
filtering by reward threshold, uniform reward assignment, and edge cases.

Usage:
    pytest tests/train/rollout/test_reinforce.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("slime.utils", reason="slime not installed")
from slime.utils.types import Sample  # noqa: E402  (must follow importorskip)

from lite.train.rollout.reinforce import convert_samples_to_train_data  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockArgs:
    """Minimal args for convert_samples_to_train_data."""
    min_episode_return: float = 0.0
    # v0.3.0: ``flatten_and_align`` pads the trajectory list up to
    # ``n_expected = rollout_batch_size * n_samples_per_prompt`` with
    # zero-gradient dummy groups. Each test sets ``rollout_batch_size`` to its
    # real prompt count so ``n_expected`` equals the post-filter trajectory
    # count → no padding → the exact sample-count assertions hold.
    rollout_batch_size: int = 1
    n_samples_per_prompt: int = 1

def make_samples(
    n_rollouts: int,
    n_turns: int,
    terminal_reward_fn=None,
) -> list[Sample]:
    """Create flat list of turn samples for testing.

    Each rollout has n_turns turns. terminal_reward_fn(rollout_idx) -> float.
    """
    if terminal_reward_fn is None:
        def _default_reward(ri):
            return 1.0
        terminal_reward_fn = _default_reward

    samples = []
    for ri in range(n_rollouts):
        for turn in range(n_turns):
            s = Sample(
                group_index=ri,
                index=ri,
                tokens=list(range(10 + turn * 5)) + list(range(5)),
                loss_mask=[1] * 5,
                rollout_log_probs=[-0.1] * 5,
                response_length=5,
                response=f"resp_{ri}_{turn}",
                reward=0.0,
                status=Sample.Status.COMPLETED,
                metadata={
                    "turn_range": (turn, turn),
                    "env_key": f"test@task_{ri}",
                    "others": {"episode_return": terminal_reward_fn(ri)},
                },
            )
            samples.append(s)
    return samples

def make_empty_sample(group_index: int = 0, index: int = 0) -> Sample:
    """Create a failed sample with empty tokens (env crash)."""
    return Sample(
        group_index=group_index,
        index=index,
        tokens=[],
        loss_mask=[],
        rollout_log_probs=[],
        response_length=0,
        response="",
        reward=0.0,
        status=Sample.Status.FAILED,
        metadata={
            "turn_range": (0, 0),
            "env_key": "test@task",
            "others": {"episode_return": 0.0},
        },
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReinforceConvertSamplesToTrainData:
    def test_empty(self):
        args = MockArgs()
        result = convert_samples_to_train_data(args, [])
        assert len(result["tokens"]) == args.rollout_batch_size * args.n_samples_per_prompt
        assert all(sum(mask) == 0 for mask in result["loss_masks"])
        assert all(reward == 0.0 for reward in result["rewards"])

    def test_all_successful(self):
        """All rollouts have reward > threshold → all kept."""
        args = MockArgs()
        args.rollout_batch_size = 4  # 4 rollouts all kept → n_expected = 4 → no padding
        samples = make_samples(4, 3, terminal_reward_fn=lambda ri: 1.0)
        result = convert_samples_to_train_data(args, samples)

        assert len(result["tokens"]) == 12  # 4 rollouts x 3 turns
        # All rewards should be 1.0 (uniform for SFT loss)
        for r in result["rewards"]:
            assert r == 1.0

    def test_filtering_by_threshold(self):
        """Only rollouts with episode_return >= min_episode_return are kept."""
        args = MockArgs()
        args.min_episode_return = 1.0
        # 2 of 4 rollouts pass the filter → n_expected must equal the kept
        # count (2) so flatten_and_align does no padding.
        args.rollout_batch_size = 2
        # Rollouts 0,1 fail (reward=0.0), rollouts 2,3 succeed (reward=1.0)
        samples = make_samples(4, 2, terminal_reward_fn=lambda ri: 1.0 if ri >= 2 else 0.0)
        result = convert_samples_to_train_data(args, samples)

        # Only 2 successful rollouts x 2 turns = 4 samples
        assert len(result["tokens"]) == 4
        for r in result["rewards"]:
            assert r == 1.0

    def test_custom_threshold(self):
        """Custom min_episode_return filters at a different level."""
        args = MockArgs()
        args.min_episode_return = 0.5
        args.rollout_batch_size = 2  # 2 of 3 kept → n_expected = 2 → no padding
        # Rewards: [0.3, 0.6, 0.9]
        samples = make_samples(3, 1, terminal_reward_fn=lambda ri: 0.3 * (ri + 1))
        result = convert_samples_to_train_data(args, samples)

        # Only rollouts with episode_return >= 0.5 kept: 0.6 and 0.9
        assert len(result["tokens"]) == 2

    def test_all_returns_zero(self):
        """0% nonzero-return rate → dummy samples with zero gradient."""
        args = MockArgs()
        args.min_episode_return = 1.0
        args.rollout_batch_size = 4  # all 4 fail → 4 dummy groups, n_expected = 4
        # All rollouts fail (reward=0.0, which is NOT >= 1.0)
        samples = make_samples(4, 2, terminal_reward_fn=lambda ri: 0.0)
        result = convert_samples_to_train_data(args, samples)

        # Should have dummy samples (loss_mask=[0])
        assert len(result["tokens"]) > 0
        for mask in result["loss_masks"]:
            assert mask == [0], "Dummy samples should have loss_mask=[0]"
        # Dummy samples should have reward=0.0 (not 1.0)
        for r in result["rewards"]:
            assert r == 0.0, "Dummy samples should have reward=0.0"

    def test_env_failed_rollouts_filtered(self):
        """Rollouts with empty tokens (env crashes) are filtered before reward filtering."""
        args = MockArgs()
        args.rollout_batch_size = 2  # 2 valid rollouts kept → n_expected = 2 → no padding
        samples = make_samples(2, 2, terminal_reward_fn=lambda ri: 1.0)
        # Add env-failed rollouts
        samples.append(make_empty_sample(group_index=10, index=10))
        samples.append(make_empty_sample(group_index=11, index=11))
        result = convert_samples_to_train_data(args, samples)

        # 2 successful rollouts x 2 turns = 4 samples (env-failed ones dropped)
        assert len(result["tokens"]) == 4

    def test_mixed_env_failure_and_reward_filter(self):
        """Both env failures and reward filtering apply."""
        args = MockArgs()
        # Need an explicit threshold > 0.0 so the reward filter actually drops
        # the reward=0.0 trajectory; default 0.0 keeps it.
        args.min_episode_return = 1.0
        args.rollout_batch_size = 2  # 2 rollouts kept → n_expected = 2 → no padding
        # 2 rollouts succeed (reward=1.0), 1 fails reward filter (reward=0.0)
        samples = make_samples(3, 1, terminal_reward_fn=lambda ri: 1.0 if ri < 2 else 0.0)
        # Plus 1 env-failed rollout
        samples.append(make_empty_sample(group_index=99, index=99))
        result = convert_samples_to_train_data(args, samples)

        # Only 2 successful rollouts x 1 turn = 2 samples
        assert len(result["tokens"]) == 2

    def test_reward_set_to_uniform(self):
        """All kept samples get reward=1.0 regardless of original reward."""
        args = MockArgs()
        args.rollout_batch_size = 3  # all 3 kept → n_expected = 3 → no padding
        # Different rewards: 0.5, 0.8, 1.0 — all > 0.0 threshold
        samples = make_samples(3, 1, terminal_reward_fn=lambda ri: 0.5 + ri * 0.25)
        result = convert_samples_to_train_data(args, samples)

        assert len(result["tokens"]) == 3
        for r in result["rewards"]:
            assert r == 1.0, f"Expected uniform reward 1.0, got {r}"

    def test_raw_reward_preserved(self):
        """raw_reward should contain original episode_return, not the uniform 1.0."""
        args = MockArgs()
        args.rollout_batch_size = 2  # 2 kept → n_expected = 2 → no padding
        samples = make_samples(2, 1, terminal_reward_fn=lambda ri: float(ri + 1))
        result = convert_samples_to_train_data(args, samples)

        assert result["raw_reward"][0] == 1.0
        assert result["raw_reward"][1] == 2.0

    def test_group_padding_to_n_expected(self):
        """v0.3.0: flatten_and_align pads the trajectory list up to
        ``n_expected = rollout_batch_size * n_samples_per_prompt`` with
        zero-gradient dummy groups (the old segment-level global_batch_size
        trim/pad is gone). 3 kept rollouts + rollout_batch_size=4 → 1 dummy
        group is appended → 4 samples total."""
        args = MockArgs()
        args.rollout_batch_size = 4  # n_expected = 4; 3 kept → pad 1 dummy group
        # 3 rollouts x 1 turn = 3 kept samples, padded up to 4.
        samples = make_samples(3, 1, terminal_reward_fn=lambda ri: 1.0)
        result = convert_samples_to_train_data(args, samples)

        assert len(result["tokens"]) == 4
        # The 3 real samples keep reward=1.0; the padded dummy is zero-gradient.
        assert result["rewards"][:3] == [1.0, 1.0, 1.0]
        assert result["loss_masks"][3] == [0]
        assert result["rewards"][3] == 0.0

    def test_rollout_log_probs_included(self):
        """rollout_log_probs should be in output when present."""
        args = MockArgs()
        samples = make_samples(1, 1, terminal_reward_fn=lambda ri: 1.0)
        result = convert_samples_to_train_data(args, samples)
        assert "rollout_log_probs" in result

    def test_turn_ordering_preserved(self):
        """Turns within a trajectory should be ordered by turn_range[0]."""
        args = MockArgs()
        # Create samples in reverse turn order
        samples = []
        for turn in reversed(range(3)):
            s = Sample(
                group_index=0, index=0,
                tokens=[1, 2, 3 + turn], loss_mask=[1], rollout_log_probs=[-0.1],
                response_length=1, response=f"turn_{turn}",
                reward=0.0,
                status=Sample.Status.COMPLETED,
                metadata={
                    "turn_range": (turn, turn),
                    "env_key": "test@task",
                    "others": {"episode_return": 1.0},
                },
            )
            samples.append(s)
        result = convert_samples_to_train_data(args, samples)
        assert result["tokens"][0] == [1, 2, 3]  # turn 0
        assert result["tokens"][2] == [1, 2, 5]  # turn 2

    def test_single_rollout(self):
        """Single rollout should work without errors."""
        args = MockArgs()
        samples = make_samples(1, 1, terminal_reward_fn=lambda ri: 1.0)
        result = convert_samples_to_train_data(args, samples)
        assert len(result["tokens"]) == 1
        assert result["rewards"][0] == 1.0
