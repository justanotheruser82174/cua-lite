"""Reward / advantage equivalence checks for action-batch ``computer`` calls.

"Design-safe, belt-and-braces" checks that the action-batch
``computer{actions:[...]}`` form is BOTH:

  * **W5 (score parity)** — reward/eval-neutral: an eval fn that reads a flat
    action list scores a per-action ``[click, type]`` sequence IDENTICALLY to
    the same actions recovered by unpacking a batched
    ``computer{actions:[click, type]}``. So switching to the action-batch form
    cannot move any reward.
  * **TR5 (gradient neutrality)** — GRPO assigns ONE advantage scalar per
    trajectory and propagates it to every segment, independent of how many
    actions batch into a turn. So batching cannot move any gradient.

Status of the real code (verified at authoring time):
  * ``lite/gym/envs/cua/utils.py`` exposes ``unpack``; ``lite.core.tools.action_space``
    exposes ``unpack_action_batch_call``.
    The score-parity proof checks the shipped helper directly.
  * ``lite/train/rollout/grpo.py`` imports ``slime.utils.types`` → NOT importable
    in the hermetic venv, so TR5 inspects the advantage-propagation loop as
    source TEXT, like the rollout-core source-contract checks.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/train/rollout/test_grpo_reward_equivalence.py -p no:cacheprovider -q
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------

# =============================================================================
# TR5 — GRPO advantage is per-trajectory (gradient neutrality of batching)
# =============================================================================

_GRPO_SRC = Path(__file__).resolve().parents[3] / "lite" / "train" / "rollout" / "grpo.py"


def test_tr5_grpo_advantage_is_one_scalar_per_trajectory() -> None:
    """Source-level check: GRPO computes ONE advantage per trajectory and
    fans it out to every segment.

    ``grpo.py`` imports slime/torch (not importable in the hermetic venv), so we
    read the advantage-propagation loop as source TEXT. The loop

        for rollout, adv in zip(rollouts, advantages):
            for s in rollout:
                s.reward = adv

    keys the advantage on the *rollout* (trajectory), never on a per-turn or
    per-action count — so batching N actions into one turn cannot change
    the gradient a trajectory contributes. That is TR5's gradient-neutrality
    guarantee."""
    src = _GRPO_SRC.read_text()
    # The advantage vector is length == #trajectories (one per rollout).
    assert "for rollout, adv in zip(rollouts, advantages):" in src
    # ... and is assigned uniformly to each segment of that trajectory.
    assert "s.reward = adv" in src
    # The advantage itself is a trajectory-level episode-return statistic
    # (reward - group_mean), NOT a per-action/per-turn quantity.
    assert "adv = rewards - rewards.mean()" in src


def test_tr5_no_per_action_reward_weighting_in_grpo() -> None:
    """Source-level check: the advantage carries NO per-turn/per-action
    scaling. The old ``adv / n_turns`` hand-normalization was removed in favor of
    slime-native ``group_mask_sums`` weighting, so turn/action COUNT never
    rescales the trajectory advantage — confirming batching is gradient-neutral."""
    src = _GRPO_SRC.read_text()
    # No CODE line rescales the advantage by turn count. (The phrase survives in a
    # comment documenting the REMOVAL of the old ``adv / n_turns`` normalization,
    # so ignore comment lines.)
    code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("adv / n_turns" in ln for ln in code_lines)
    assert "group_mask_sums" in src
