"""Stress tests for CUA-Lite RL rollout pipeline.

Tests the rollout pipeline components: generate_fn → on_step → Sample building,
convert_samples_to_train_data, and env fault isolation.

Usage:
    pytest tests/train/rollout/test_rollout.py -v
    # Inside Slime container:
    pytest tests/train/rollout/test_rollout.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("slime.utils", reason="slime not installed")
from slime.utils.types import Sample  # noqa: E402  (must follow importorskip)

from lite.train.rollout.grpo import convert_samples_to_train_data  # noqa: E402

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

class TestConvertSamplesToTrainData:
    def test_empty(self):
        args = MockArgs()
        result = convert_samples_to_train_data(args, [])
        assert len(result["tokens"]) == args.rollout_batch_size * args.n_samples_per_prompt
        assert all(sum(mask) == 0 for mask in result["loss_masks"])
        assert all(reward == 0.0 for reward in result["rewards"])

    def test_basic_grpo(self):
        """2 prompts x 4 samples x 3 turns = 24 training samples."""
        args = MockArgs()
        # rollout_batch_size == prompt count so n_expected (= rollout_batch_size
        # * n_samples_per_prompt) equals the real trajectory count (2*4=8) and
        # flatten_and_align performs NO dummy-group padding.
        args.rollout_batch_size = 2
        samples = make_samples(2, 4, 3)
        result = convert_samples_to_train_data(args, samples)

        assert len(result["tokens"]) == 24
        assert len(result["rewards"]) == 24
        assert len(result["loss_masks"]) == 24
        assert len(result["response_lengths"]) == 24
        # v0.3.0: per-trajectory aggregation fields are emitted.
        assert len(result["group_ids"]) == 24
        assert len(result["group_mask_sums"]) == 24

        # All turns in same trajectory get same advantage
        rewards = result["rewards"]
        for i in range(8):  # 8 trajectories
            traj_rewards = rewards[i * 3:(i + 1) * 3]
            assert len(set(traj_rewards)) == 1, f"Trajectory {i}: {traj_rewards}"
        # Each trajectory has 3 turns x 5 mask=1 tokens = 15 total loss mask,
        # broadcast to every segment of that trajectory.
        for ms in result["group_mask_sums"]:
            assert ms == 15

    def test_advantages_symmetric_within_group(self):
        """Within each prompt group, advantages should sum to ~0."""
        args = MockArgs()
        args.n_samples_per_prompt = 4
        args.rollout_batch_size = 1  # 1 prompt → n_expected = 4 = real trajs
        samples = make_samples(1, 4, 2)
        result = convert_samples_to_train_data(args, samples)

        # Extract one advantage per trajectory (each has 2 turns)
        traj_advs = [result["rewards"][i * 2] for i in range(4)]
        assert abs(sum(traj_advs)) < 1e-5, f"Advantages don't sum to 0: {traj_advs}"

    def test_all_same_reward(self):
        """All trajectories get same terminal reward → all advantages ~0."""
        args = MockArgs()
        args.rollout_batch_size = 1  # 1 prompt x 4 samples → n_expected = 4 = real trajs
        samples = make_samples(1, 4, 2, terminal_reward_fn=lambda pi, si: 5.0)
        result = convert_samples_to_train_data(args, samples)

        for r in result["rewards"]:
            assert abs(r) < 1e-5, f"Expected ~0 advantage, got {r}"

    def test_single_trajectory(self):
        """Single trajectory should not produce NaN (regression test)."""
        args = MockArgs()
        args.n_samples_per_prompt = 1
        args.rollout_batch_size = 1  # 1 prompt x 1 sample → n_expected = 1 = real trajs
        samples = make_samples(1, 1, 3, terminal_reward_fn=lambda pi, si: 5.0)
        result = convert_samples_to_train_data(args, samples)

        # With correction=0 std, single sample → advantage 0 (not NaN)
        for r in result["rewards"]:
            assert not torch.isnan(torch.tensor(r)), f"NaN reward: {r}"
            assert abs(r) < 1e-5

    def test_failed_samples_filtered(self):
        """Samples with empty tokens (env failures) should be filtered out."""
        samples = make_samples(1, 2, 2)
        # Add a failed sample with empty tokens
        failed = Sample(
            group_index=0, index=99, tokens=[], response_length=0,
            reward=0.0, status=Sample.Status.FAILED,
            metadata={"turn_range": (0, 0), "others": {"episode_return": 0.0}},
        )
        samples.append(failed)

        args = MockArgs()
        args.n_samples_per_prompt = 2
        # 2 valid trajectories survive the errored filter (the empty-token
        # one is dropped). rollout_batch_size=1 → n_expected = 1*2 = 2 == the
        # post-filter trajectory count → no dummy-group padding.
        args.rollout_batch_size = 1
        result = convert_samples_to_train_data(args, samples)
        # 2 trajectories x 2 turns = 4 training samples (failed one filtered)
        assert len(result["tokens"]) == 4

    def test_nonempty_failed_sample_survives_convert(self):
        """A FAILED turn that still carries tokens is a real trajectory, not an
        infra-empty drop. GRPO keeps its zero-gradient sample in the batch."""
        args = MockArgs()
        args.n_samples_per_prompt = 1
        args.rollout_batch_size = 1
        sample = Sample(
            group_index=0,
            index=0,
            group_id=0,
            prompt="orig",
            tokens=[10, 20],
            loss_mask=[0],
            rollout_log_probs=[0.0],
            response_length=1,
            response="",
            reward=0.0,
            status=Sample.Status.FAILED,
            metadata={
                "turn_range": (0, 0),
                "env_key": "test@task",
                "others": {"episode_return": 0.0},
                "lite_error_retryable": False,
            },
        )

        result = convert_samples_to_train_data(args, [sample])

        assert result["tokens"] == [[10, 20]]
        assert result["loss_masks"] == [[0]]
        assert result["raw_reward"] == [0.0]

    @pytest.mark.parametrize(
        ("response", "tool_error"),
        [
            (
                "computer key failed",
                "invalid arguments for key: unknown key token 'not_a_real_key'",
            ),
            (
                "computer click failed",
                "click failed: execution failed",
            ),
            (
                "computer wait failed",
                "invalid arguments for wait: wait.duration must be a finite number",
            ),
        ],
    )
    def test_pairable_xdotool_action_failures_survive_grpo_convert(
        self,
        response: str,
        tool_error: str,
    ):
        """Pairable action feedback stays a model-action trajectory, not infra.

        Regression evidence for the prerefactor xdotool class:
        bad key tokens are paired model feedback, and true xdotool/interface
        execution failures are sanitized tool feedback. Both carry tokens and
        must survive the GRPO empty-infra filter.
        """
        args = MockArgs()
        args.n_samples_per_prompt = 1
        args.rollout_batch_size = 1
        sample = make_single_turn_sample(response=response, tool_error=tool_error)

        result = convert_samples_to_train_data(args, [sample])

        assert "lite_failure_reason" not in sample.metadata
        assert result["tokens"] == [[10, 20, 30]]
        assert result["loss_masks"] == [[1]]
        assert result["raw_reward"] == [0.0]

    def test_rollout_log_probs_included(self):
        """rollout_log_probs should be in output when present."""
        args = MockArgs()
        args.n_samples_per_prompt = 1
        args.rollout_batch_size = 1
        samples = make_samples(1, 1, 1)
        result = convert_samples_to_train_data(args, samples)
        assert "rollout_log_probs" in result

    def test_multimodal_train_inputs(self):
        """multimodal_train_inputs should be in output when present."""
        args = MockArgs()
        args.n_samples_per_prompt = 1
        args.rollout_batch_size = 1
        samples = make_samples(1, 1, 1)
        samples[0].multimodal_train_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
        result = convert_samples_to_train_data(args, samples)
        assert "multimodal_train_inputs" in result

    def test_turn_ordering(self):
        """Turns within a trajectory should be ordered by turn_range[0]."""
        args = MockArgs()
        args.n_samples_per_prompt = 1
        args.rollout_batch_size = 1
        # Create samples in reverse turn order
        samples = []
        for turn in reversed(range(3)):
            s = Sample(
                group_index=0, index=0,
                tokens=[1, 2, 3 + turn], loss_mask=[1], rollout_log_probs=[-0.1],
                response_length=1, response=f"turn_{turn}",
                reward=1.0 if turn == 2 else 0.0,
                status=Sample.Status.COMPLETED,
                metadata={"turn_range": (turn, turn)},
            )
            samples.append(s)
        result = convert_samples_to_train_data(args, samples)
        # First token sequence should be turn 0's
        assert result["tokens"][0] == [1, 2, 3]
        assert result["tokens"][2] == [1, 2, 5]

    def test_no_std_normalization(self):
        """When grpo_std_normalization=False, advantages are just mean-centered."""
        args = MockArgs()
        args.grpo_std_normalization = False
        args.n_samples_per_prompt = 2
        args.rollout_batch_size = 1  # 1 prompt x 2 samples → n_expected = 2 = real trajs
        # Terminal rewards: [1.0, 3.0] → mean=2.0 → advantages [-1.0, 1.0]
        samples = make_samples(1, 2, 1, terminal_reward_fn=lambda pi, si: float(si * 2 + 1))
        result = convert_samples_to_train_data(args, samples)
        assert abs(result["rewards"][0] - (-1.0)) < 1e-5
        assert abs(result["rewards"][1] - 1.0) < 1e-5

    def test_large_scale(self):
        """Stress test: 64 prompts x 8 samples x 5 turns = 2560 samples."""
        args = MockArgs()
        args.n_samples_per_prompt = 8
        # rollout_batch_size == 64 prompts so n_expected = 64*8 = 512 trajs ==
        # the real trajectory count → no dummy-group padding.
        args.rollout_batch_size = 64
        samples = make_samples(64, 8, 5)
        result = convert_samples_to_train_data(args, samples)
        assert len(result["tokens"]) == 64 * 8 * 5
        # Verify no NaN rewards
        for r in result["rewards"]:
            assert not torch.isnan(torch.tensor(r)), "NaN reward in large-scale test"

# ---------------------------------------------------------------------------
# Tests: AbortError
# ---------------------------------------------------------------------------

