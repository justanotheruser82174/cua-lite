"""End-to-end GRPO advantage math through ``rollout_grpo.convert_samples_to_train_data``.

This is a Slime-required test that wires synthetic Samples through the
**actual** ``rollout_grpo.convert_samples_to_train_data`` function (not a
re-implementation), proving that:

  1. Group-level (R - μ) / σ advantages are computed correctly.
  2. Each Sample's reward is set to the RAW advantage (v0.3.0: the old
     ``advantage / n_turns`` hand-normalization is gone; per-trajectory
     weighting is delegated to slime's loss reducer via ``group_mask_sums``).
  3. ``group_ids`` + ``group_mask_sums`` (per-trajectory total mask) are emitted.
  4. Constant-return groups (σ=0) yield zero advantages → zero gradient. With
     ``DROP_ZERO_STD_GROUP`` (default on) such a group is replaced by zero-reward
     2-token dummies; with it off the real zero-reward samples are emitted. Either
     way the group contributes no gradient.

Slime-required (uses Sample type + actual rollout_grpo module).
"""

from __future__ import annotations

import math
from argparse import Namespace

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.types import Sample

from lite.train.rollout.grpo import convert_samples_to_train_data


def _make_sample(
    *, group_index, index, episode_return, n_turns,
    rollout_length, prompt="p", reward=0.0,
):
    """Build a slime.Sample with all the metadata fields rollout_grpo reads.

    ``rollout_length`` is the number of segment Samples per trajectory
    (== len(rollout) under the GRPO group-by). ``n_turns`` is the original
    trajectory turn count (in metadata) — these can differ under packing.
    """
    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt,
        tokens=[1, 2, 3, 4],
        loss_mask=[1, 1],
        rollout_log_probs=[-0.1, -0.2],
        response_length=2,
        response="r",
        label=None,
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata={
            "env_key": "lite.demo@test",
            "n_turns": n_turns,
            "turn_range": (0, n_turns - 1),
            "step_statuses": ("completed",) * rollout_length,
            "others": {"episode_return": episode_return},
        },
    )


def _make_args(**kw):
    """Build a minimal Namespace expected by convert_samples_to_train_data."""
    defaults = dict(
        rollout_batch_size=1,
        n_samples_per_prompt=4,
        num_rollout=1,
        rollout_global_dataset=False,
        rollout_max_response_len=8192,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        calculate_per_token_loss=False,  # GRPO guard: keeps the group_mask_sums path
    )
    defaults.update(kw)
    return Namespace(**defaults)


def test_e2e_grpo_reward_is_raw_advantage_with_group_mask_sums():
    """v0.3.0 contract: reward = RAW group-normalized advantage (no adv/n_turns),
    and per-trajectory normalization is delegated to slime via ``group_mask_sums``
    (= each trajectory's total loss mask), NOT hand-applied. n_turns in metadata
    must be IGNORED for the reward value (it was the old divisor)."""
    samples_in: list[Sample] = []
    for traj_idx in range(4):
        episode_return = traj_idx / 3.0  # 0.0, 0.333, 0.667, 1.0
        s = _make_sample(
            group_index=0,
            index=traj_idx,
            episode_return=episode_return,
            n_turns=10,          # must NOT affect the reward (the old /n_turns is gone)
            rollout_length=1,    # packed → 1 segment per trajectory
        )
        samples_in.append(s)

    args = _make_args()
    out = convert_samples_to_train_data(args, samples_in)
    rewards = out["rewards"]

    # Expected RAW advantage (R - μ)/σ, Bessel-corrected std — NO /n_turns.
    returns = [s.metadata["others"]["episode_return"] for s in samples_in]
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    sigma = math.sqrt(var) if var > 0 else 1.0
    expected = [(r - mu) / sigma for r in returns]

    assert len(rewards) == len(expected), (
        f"reward count mismatch: {len(rewards)} vs {len(expected)}"
    )
    for i, (got, want) in enumerate(zip(rewards, expected)):
        assert math.isclose(got, want, abs_tol=1e-5), (
            f"sample {i}: reward must be RAW advantage (no /n_turns); got {got}, want {want}"
        )

    # The v0.3.0 normalization plumbing must be present.
    assert "group_ids" in out and out["group_ids"] == [0, 1, 2, 3], out.get("group_ids")
    # Each trajectory is 1 segment with loss_mask=[1,1] → per-trajectory total mask = 2.
    assert "group_mask_sums" in out, out.keys()
    assert out["group_mask_sums"] == [2, 2, 2, 2], out["group_mask_sums"]


def test_e2e_grpo_multi_segment_trajectory_broadcasts_one_denominator():
    """Radix/unrolled trajectories with multiple Samples must still count once.

    Slime flattens compact rollout output before custom convert; cua-lite
    reconstructs trajectories by ``(group_index, index)`` and emits one shared
    ``group_mask_sums`` denominator per trajectory. This is the exact shape
    radix packing relies on when one trajectory becomes several segments.
    """
    samples_in: list[Sample] = []
    segment_counts = [3, 1, 2, 4]
    loss_masks_by_traj = [
        [[1], [1, 1], [1, 1, 1]],
        [[1, 1]],
        [[1], [1, 1, 1]],
        [[1], [1], [1], [1]],
    ]
    returns = [0.0, 1.0, 2.0, 3.0]

    for traj_idx, (episode_return, masks) in enumerate(zip(returns, loss_masks_by_traj)):
        for seg_idx, loss_mask in enumerate(masks):
            s = _make_sample(
                group_index=0,
                index=traj_idx,
                episode_return=episode_return,
                n_turns=10,
                rollout_length=segment_counts[traj_idx],
            )
            s.loss_mask = loss_mask
            s.response_length = len(loss_mask)
            s.rollout_log_probs = [-0.1] * len(loss_mask)
            s.tokens = [0] * (len(loss_mask) + 1)
            s.metadata["turn_range"] = (seg_idx, seg_idx)
            samples_in.append(s)

    out = convert_samples_to_train_data(_make_args(), samples_in)

    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    sigma = math.sqrt(var)
    expected_adv = [(r - mu) / sigma for r in returns]

    expected_rewards: list[float] = []
    expected_group_ids: list[int] = []
    expected_denoms: list[int] = []
    for traj_idx, masks in enumerate(loss_masks_by_traj):
        total_mask = sum(sum(mask) for mask in masks)
        expected_rewards.extend([expected_adv[traj_idx]] * len(masks))
        expected_group_ids.extend([traj_idx] * len(masks))
        expected_denoms.extend([total_mask] * len(masks))

    assert out["group_ids"] == expected_group_ids
    assert out["group_mask_sums"] == expected_denoms
    for got, want in zip(out["rewards"], expected_rewards, strict=True):
        assert math.isclose(got, want, abs_tol=1e-5)


def test_e2e_grpo_zero_std_group_yields_zero_advantage():
    """Constant-return group → σ=0 → all-zero advantages → zero gradient.

    Robust to DROP_ZERO_STD_GROUP either way: on (default) → group becomes
    zero-reward dummies; off → real zero-reward samples. Both ⇒ rewards all 0.
    """
    samples_in: list[Sample] = []
    for traj_idx in range(4):
        s = _make_sample(
            group_index=0,
            index=traj_idx,
            episode_return=0.5,  # all same
            n_turns=10,
            rollout_length=10,
        )
        samples_in.append(s)

    args = _make_args()
    out = convert_samples_to_train_data(args, samples_in)
    # DROP_ZERO_STD_GROUP (default on) replaces the zero-std group with zero-reward
    # dummies; with it off the real samples are emitted with zero reward. Either way
    # the rewards are all 0 (zero gradient) — accept both.
    if out["rewards"]:
        assert all(r == 0.0 for r in out["rewards"]), (
            f"zero-std group must yield zero rewards (or be filtered); got {out['rewards']}"
        )


def test_empty_batch_convert_returns_valid_noop_not_empty_dict():
    """convert([]) must return a valid no-op train_data, not {} — an empty
    dict KeyErrors in slime's _split_train_data_by_dp (data["group_ids"]/["tokens"]).
    Normally unreachable because the all-empty guard keeps data non-empty; hardened defensively."""
    out = convert_samples_to_train_data(_make_args(), [])
    assert out, "convert([]) returned an empty dict (BUG 4 regression)"
    assert "group_ids" in out and "tokens" in out, (
        "no-op train_data must carry the keys _split_train_data_by_dp accesses"
    )
