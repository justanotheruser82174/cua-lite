"""GRPO-side seed-injection tests.

Server warm-pool claim forwarding is retired; this file verifies the
client-side contract: ``lite/train/rollout/core/engine.py::generate`` picks
the right seed per ``(rollout_id, group_index)`` and only injects it
on the train path when the env declares ``seed`` support.

Coverage:
  - Determinism of ``_group_seeds[group_index]`` derivation from
    ``(rollout_id, group_index)`` — same inputs → same seed, different
    inputs → different seeds.
  - Train split injects ``env_kwargs["seed"]`` only for seed-capable envs;
    eval split does not.
  - ``group_shared_seed=False`` suppresses injection entirely.
  - Sibling rollouts in the same group share the seed (the GRPO
    advantage-baseline invariant); cross-group rollouts get
    independent seeds.

The ``generate()``-driven half is slime-required (it imports ``engine.py``)
and skips on the host venv; the pure derivation half runs anywhere.

Run::

    pytest tests/train/rollout/core/test_seed_injection.py -v
"""
from __future__ import annotations

import random
from unittest.mock import patch

import pytest

#: Slime is only installed inside the training Docker image. The pure
#: ``_group_seeds`` formula tests run unconditionally on the host venv; tests
#: that drive ``generate()`` are gated below and skip cleanly on the host.
try:
    from slime.utils.types import Sample  # type: ignore[import-not-found]
    _slime_available = True
except ImportError:
    Sample = None  # type: ignore[assignment]
    _slime_available = False


# =============================================================================
# Deterministic _group_seeds derivation
# =============================================================================


class TestGroupSeedDerivation:
    """The seed for a group is derived as
    ``random.Random(f"{rollout_id}:{group_index}").randint(0, 2**31-1)``
    (see ``lite/train/rollout/core/engine.py`` lines 679-682). It MUST be a
    pure function of those two inputs so concurrent rollout coroutines
    that race on first-seen of the same group converge to the same
    value (GRPO's baseline correctness depends on group-mates seeing
    identical initial env state)."""

    @staticmethod
    def _derive(rollout_id: int, group_index: int) -> int:
        return random.Random(f"{rollout_id}:{group_index}").randint(0, 2**31 - 1)

    def test_same_inputs_same_seed(self):
        for rollout_id in (0, 1, 7, 12345):
            for group_index in (0, 1, 31):
                a = self._derive(rollout_id, group_index)
                b = self._derive(rollout_id, group_index)
                assert a == b, (
                    f"derive({rollout_id}, {group_index}) not deterministic"
                )

    def test_different_rollout_ids_distinct_seeds(self):
        seeds = {self._derive(rid, group_index=0) for rid in range(32)}
        assert len(seeds) >= 30, (
            "rollout_id derivation collision rate too high; "
            "GRPO would re-use task params across iterations"
        )

    def test_different_group_indices_distinct_seeds(self):
        seeds = {self._derive(rollout_id=0, group_index=g) for g in range(32)}
        assert len(seeds) >= 30, (
            "group_index derivation collision rate too high; "
            "cross-group rollouts would share initial state"
        )

    def test_cross_axis_independence(self):
        """``(0, 1)`` and ``(1, 0)`` must NOT collide (the formatter
        is ``f"{rollout_id}:{group_index}"`` so the colon separator
        guarantees this — verify regression here in case the format
        ever flattens)."""
        assert self._derive(0, 1) != self._derive(1, 0)

# =============================================================================
# generate() seed injection — slime-gated (needs the training image)
# =============================================================================

pytestmark_grpo = pytest.mark.skipif(
    not _slime_available, reason="slime not installed (host venv)",
)


def _run_generate_capture_env_kwargs(
    *, group_shared_seed: bool, split: str, group_index: int,
    rollout_id: int, env_supports_seed: bool = True,
) -> dict:
    """Run ``generate(args, sample, sampling_params)`` with all I/O mocked,
    intercepting the ``env_kwargs`` dict at gym.make time. Returns the
    captured kwargs.

    ``env_supports_seed`` drives ``gym.registry.env_supports_kwarg`` — the
    capability gate that keeps generate() from handing a ``seed`` to an env
    that never declared it."""
    from lite.train.rollout.core.engine import generate  # noqa: PLC2701
    from tests.train.rollout.core.test_engine_integration import (  # noqa: PLC2701
        FakeAgent,
        FakeEnv,
        FakeGenerateState,
        MockArgs,
        fake_post,
    )
    args = MockArgs()
    args.group_shared_seed = group_shared_seed
    sample = Sample(
        group_index=group_index, index=0, prompt="test",
        metadata={"env_key": "androidworld@task1", "split": split},
    )
    captured: dict = {}
    fake_env = FakeEnv(n_steps=1, terminal_reward=0.0)
    with (
        patch("lite.train.rollout.core.engine.GenerateState", FakeGenerateState),
        patch("lite.train.rollout.core.engine.post", fake_post),
        patch("lite.train.rollout.core.engine.gym") as mock_gym,
        patch("lite.train.rollout.core.engine.AgentRegistry") as mock_registry,
        patch("lite.train.rollout.core.engine._current_rollout_id", rollout_id),
    ):
        def _make(env_key, **kw):
            captured.update(kw)
            return fake_env
        mock_gym.make.side_effect = _make
        mock_gym.registry.env_supports_kwarg.return_value = env_supports_seed
        mock_registry.get.side_effect = lambda key, **kw: FakeAgent(
            kw.get("generate_fn"), n_steps=1, terminal_reward=0.0,
        )
        import asyncio
        asyncio.run(generate(
            args, sample, {"temperature": 1.0, "max_new_tokens": 4096},
        ))
        # Caller clears _group_seeds between tests if needed.
        return captured


@pytest.fixture(autouse=True)
def _reset_grpo_globals():
    """``_group_seeds`` is module-level state that survives across
    tests in a process. Clear before each test so tests are
    independent. No-op when slime is absent (the dict doesn't import)."""
    if _slime_available:
        from lite.train.rollout.core.engine import _group_seeds  # noqa: PLC2701
        _group_seeds.clear()
        yield
        _group_seeds.clear()
    else:
        yield


@pytestmark_grpo
class TestGenerateSeedInjection:
    """``generate()`` is where the ``env_kwargs["seed"]`` actually
    gets stamped. The matrix:

      split   | group_shared_seed | env declares seed | inject seed?
      --------|-------------------|-------------------|--------------
      train   | True              | yes               | YES
      train   | True              | no                | no (capability gate)
      train   | False             | yes               | no
      eval    | True              | yes               | no (registered seed authoritative)
      eval    | False             | yes               | no (warns)
    """

    def test_train_split_with_shared_seed_injects(self):
        kw = _run_generate_capture_env_kwargs(
            group_shared_seed=True, split="train",
            group_index=3, rollout_id=42,
        )
        assert "seed" in kw, "train + shared → seed must be in env_kwargs"
        expected = TestGroupSeedDerivation._derive(
            rollout_id=42, group_index=3,
        )
        assert kw["seed"] == expected, (
            f"injected seed {kw['seed']} doesn't match derivation"
        )

    def test_train_split_without_shared_seed_no_inject(self):
        kw = _run_generate_capture_env_kwargs(
            group_shared_seed=False, split="train",
            group_index=0, rollout_id=42,
        )
        assert "seed" not in kw, (
            f"group_shared_seed=False must not inject any seed; got {kw.get('seed')!r}"
        )

    def test_env_without_seed_capability_no_inject(self):
        """Registry capability gate: an env that never declared ``seed`` as an
        accepted soft kwarg must not be handed one (it would be silently
        dropped, or worse, rejected at gym.make)."""
        kw = _run_generate_capture_env_kwargs(
            group_shared_seed=True, split="train",
            group_index=3, rollout_id=42, env_supports_seed=False,
        )
        assert "seed" not in kw, (
            "env_supports_kwarg(env_id, 'seed') is False → no injection; "
            f"got {kw.get('seed')!r}"
        )

    def test_eval_split_never_injects_regardless_of_flag(self):
        """The registered ``seed=42`` on each eval-split spec is
        authoritative — generate() must not overwrite it from the
        group-derived seed pool."""
        for gss in (True, False):
            kw = _run_generate_capture_env_kwargs(
                group_shared_seed=gss, split="eval",
                group_index=0, rollout_id=42,
            )
            assert "seed" not in kw, (
                f"eval + group_shared_seed={gss}: caller injected seed "
                f"but eval should defer to spec default; got {kw.get('seed')}"
            )

    def test_siblings_in_same_group_share_seed(self):
        """Two rollouts with the same ``group_index`` in the same
        ``rollout_id`` MUST get the same seed (GRPO baseline
        correctness)."""
        kw_a = _run_generate_capture_env_kwargs(
            group_shared_seed=True, split="train",
            group_index=5, rollout_id=100,
        )
        # Don't clear _group_seeds between calls — second call should
        # hit the cache populated by the first.
        kw_b = _run_generate_capture_env_kwargs(
            group_shared_seed=True, split="train",
            group_index=5, rollout_id=100,
        )
        assert kw_a["seed"] == kw_b["seed"], (
            "group-mates got different seeds; GRPO advantage baseline "
            "would conflate initial-state variance with policy variance"
        )

    def test_cross_groups_get_independent_seeds(self):
        kw_a = _run_generate_capture_env_kwargs(
            group_shared_seed=True, split="train",
            group_index=0, rollout_id=100,
        )
        kw_b = _run_generate_capture_env_kwargs(
            group_shared_seed=True, split="train",
            group_index=1, rollout_id=100,
        )
        assert kw_a["seed"] != kw_b["seed"], (
            "different groups got identical seeds; cross-group "
            "exploration is broken"
        )
