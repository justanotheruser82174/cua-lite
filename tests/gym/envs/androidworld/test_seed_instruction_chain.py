"""AndroidWorld instance-seed derivation tests.

AndroidWorld derives an instance seed from ``self._seed`` before generating
task parameters. This pure test pins that derivation without requiring an
emulator.

Coverage:
  - androidworld's ``_seed`` → ``instance_seed`` md5 derivation is a
    pure function: distinct ``self._seed`` values produce distinct
    ``instance_seed`` values → distinct task params → distinct
    instruction strings. We test the derivation in isolation (no
    emulator required) since the function is deterministic and pure.

Run::

    pytest tests/gym/envs/androidworld/test_seed_instruction_chain.py -v
"""
from __future__ import annotations

import hashlib

# =============================================================================
# androidworld: self._seed → instance_seed → task params → instruction
# (pure derivation test — no emulator)
# =============================================================================

class TestAndroidWorldSeedInstructionChain:
    """``androidworld/main.py`` line 1037-1041 derives an instance
    seed from ``self._seed`` via md5:

        instance_seed = int(md5(f"{task_name}:{self._seed}").hexdigest(), 16) % (2**31)

    This instance_seed is what's passed to ``task_class.generate_random_params(seed=...)``
    which determines the task params, which in turn determines
    ``self._task.goal`` (the instruction the agent sees).

    Verifying the WHOLE chain requires the emulator, but the md5
    derivation is a pure function we CAN exercise: distinct
    ``self._seed`` values must produce distinct ``instance_seed``
    values within a single task. If this holds, the downstream
    instructions WILL differ (modulo the upstream
    ``generate_random_params`` being non-degenerate, which is
    upstream androidworld's responsibility)."""

    @staticmethod
    def _instance_seed(task_name: str, self_seed: int | None) -> int | None:
        # Mirror the exact derivation in
        # ``lite/gym/envs/androidworld/main.py`` line 1037-1041.
        if self_seed is None:
            return None
        return int(hashlib.md5(
            f"{task_name}:{self_seed}".encode()
        ).hexdigest(), 16) % (2**31)

    def test_different_self_seed_produces_different_instance_seed(self):
        """The GRPO ``group_shared_seed`` contract relies on this: when
        ``self._seed`` differs across groups, the downstream task
        params (and thus instruction text) must also differ."""
        task = "OpenAppTask"
        seeds = {self._instance_seed(task, s) for s in range(50)}
        assert len(seeds) >= 48, (
            f"only {len(seeds)} unique instance_seeds for 50 inputs — "
            "md5 truncation is leaking too much. Different self._seed "
            "values would produce the same task params, breaking "
            "cross-group exploration."
        )

    def test_same_self_seed_same_instance_seed(self):
        """Idempotence: repeated reset/retry must hit the same task params."""
        assert (
            self._instance_seed("OpenAppTask", 12345)
            == self._instance_seed("OpenAppTask", 12345)
        )

    def test_different_task_names_isolate_seeds(self):
        """``task_name`` is in the md5 input — same ``self._seed``
        with different task_names produces different instance_seeds.
        Without this, two warm-eligible tasks would step on each
        other's RNG sequences."""
        assert (
            self._instance_seed("OpenAppTask", 7)
            != self._instance_seed("ContactsAddContact", 7)
        )

    def test_none_seed_passes_through_as_none(self):
        """``self._seed = None`` ⇒ ``instance_seed = None`` ⇒ task
        params drawn from the in-container python's global RNG. This
        is the eval-default for tasks where the spec didn't bake
        a seed."""
        assert self._instance_seed("OpenAppTask", None) is None
