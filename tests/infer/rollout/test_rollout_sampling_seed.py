"""Rollout sampling seed derivation tests.

Sampling seeds are per trajectory, not per GRPO group. That keeps sibling
rollouts reproducible while allowing them to sample different completions.

Coverage:
  - training rollout sampling: ``(rollout_id, sample.index)``;
  - local collection sampling: ``(seed, task_id, sample_idx)``;
  - sibling trajectories are distinct while identical inputs are reproducible.

Run::

    pytest tests/infer/rollout/test_rollout_sampling_seed.py -v
"""

from __future__ import annotations

import random

# =============================================================================
# Deterministic sampling seed derivation
# =============================================================================


class TestSamplingSeedDerivation:
    """The SAMPLING seed is the per-MEMBER-DISTINCT twin of the per-group-
    SHARED env seed: same ``random.Random(f"{base}:{identity}").randint`` recipe
    and same base, but keyed on the globally-unique ``sample.index`` (training,
    engine.py) / ``(task_id, sample_idx)`` (collect, rollout.py) instead of the
    shared ``group_index`` / ``task_id``. So a group's siblings sample
    DIFFERENTLY — the variance GRPO's group baseline measures — while staying
    reproducible from the master seed. The identity is the ONLY thing that flips
    relative to the env seed; this suite pins that the flip really produces
    sibling-DISTINCT (not shared) seeds."""

    @staticmethod
    def _train(rollout_id: int, sample_index: int) -> int:
        return random.Random(f"{rollout_id}:{sample_index}").randint(0, 2**31 - 1)

    @staticmethod
    def _collect(seed: int, task_id: str, sample_idx: int) -> int:
        return random.Random(f"{seed}:{task_id}:{sample_idx}").randint(0, 2**31 - 1)

    def test_reproducible(self):
        """Same inputs → same seed (the reproducibility baseline)."""
        assert self._train(7, 3) == self._train(7, 3)
        assert self._collect(42, "task1", 1) == self._collect(42, "task1", 1)

    def test_train_siblings_distinct(self):
        """Same rollout_id, distinct ``sample.index`` → distinct seeds. This is
        the OPPOSITE of the env seed (shared within a group): if siblings shared
        a sampling seed they'd sample identically → zero GRPO variance → the
        whole group gets stubbed by DROP_ZERO_STD with no gradient."""
        seeds = {self._train(rollout_id=0, sample_index=i) for i in range(32)}
        assert len(seeds) >= 30, "sibling sampling seeds collide → zero variance"

    def test_collect_siblings_distinct(self):
        """Same task, distinct ``sample_idx`` (a task's repeats) → distinct seeds."""
        seeds = {self._collect(42, "task1", i) for i in range(32)}
        assert len(seeds) >= 30

    def test_mirror_env_seed_sharing(self):
        """The crux of the design: for siblings of ONE group, the env seed is
        SHARED while the sampling seed is DISTINCT — same recipe, opposite
        identity (constant ``group_index`` vs varying ``sample.index``)."""
        # env seed: keyed by the within-group-constant group_index → shared
        env_a = random.Random(f"{0}:{5}").randint(0, 2**31 - 1)
        env_b = random.Random(f"{0}:{5}").randint(0, 2**31 - 1)
        assert env_a == env_b
        # sampling seed: keyed by the per-trajectory sample.index → distinct
        assert self._train(0, 100) != self._train(0, 101)

    def test_cross_axis_independence(self):
        """The ``:`` separator keeps ``(0, 1)`` and ``(1, 0)`` from colliding."""
        assert self._train(0, 1) != self._train(1, 0)
