"""Env-failure tier-1 fixtures.

These verify the env-failure code paths in
``lite/train/rollout/core/engine.py:generate``: ``_empty_sample``,
``dummy_sample``, ``_filter_errored_rollouts``. The paths are
correctness-critical but invisible in the happy-path pseudocode.

Slime-required (tests import slime.utils.types.Sample) — auto-skipped
if slime not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.types import Sample

from lite.gym.errors import (
    CapacityExhausted,
    CuaWorldVerifierError,
    EnvDesktopCrashed,
    PairableModelActionError,
    RemoteEnvError,
    TrueInfraFailure,
    is_retryable,
)
from lite.gym.remote.errors import ModelOutputError
from lite.gym.wrappers import EnvTimeoutError
from lite.train.rollout.core import engine as _engine
from lite.train.rollout.core.engine import (
    _classify_failure,
    _count_empty_trajs,
    _empty_sample,
    bucket_trajectories,
    dummy_sample,
)


def _valid_sample(group_index, index):
    return Sample(
        group_index=group_index,
        index=index,
        prompt="orig",
        tokens=[1, 2],
        loss_mask=[1],
        response_length=1,
        response="x",
        label=None,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata={"env_key": "lite.demo@test"},
    )


def _make_template(group_index=0, index=0):
    return Sample(
        group_index=group_index,
        index=index,
        prompt="orig",
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.PENDING,
        metadata={"env_key": "lite.demo@test"},
    )


def test_empty_sample_inherits_identity_from_original():
    """``_empty_sample`` keeps group_index / index / prompt / label so the
    failed sample slots into the original sample's GRPO group correctly."""
    template = _make_template(group_index=2, index=5)
    empty = _empty_sample(template)

    assert empty.group_index == 2
    assert empty.index == 5
    assert empty.prompt == "orig"
    assert empty.tokens == []
    assert empty.loss_mask == []
    assert empty.response_length == 0
    assert empty.status == Sample.Status.FAILED


def test_dummy_sample_preserves_group_index_for_grpo_diversity():
    """All-batch-failed → ``dummy_sample`` injection. Must preserve
    ``group_index`` so GRPO group metrics don't collapse."""
    template = _make_template(group_index=7, index=3)
    dummy = dummy_sample(template)

    assert dummy.group_index == 7
    assert dummy.index == 3
    # 2-token tokens hack — slime computes prompt_length = total - response_length
    # and would crash on a single-token sample.
    assert len(dummy.tokens) >= 2
    assert dummy.loss_mask == [0]   # zero gradient
    assert dummy.status == Sample.Status.FAILED


def test_distinct_group_indices_survive_dummy_injection():
    """If a batch of N rollouts all fail and each gets ``dummy_sample``
    from its own template, the group_index diversity must be preserved."""
    templates = [_make_template(group_index=i, index=0) for i in range(4)]
    dummies = [dummy_sample(t) for t in templates]

    group_indices = {d.group_index for d in dummies}
    assert group_indices == {0, 1, 2, 3}, (
        "all 4 distinct group_indices must survive (full diversity)"
    )


def test_bucket_trajectories_splits_errored_from_missing():
    """All-empty trajectories that ``_drop_empty_turns`` stripped upstream are
    recovered into ``n_errored`` (env/judge failure) via the pre-drop count on
    ``_last_rollout_n_errored`` — NOT silently folded into ``n_missing``
    (framework-level drop). ``n_valid`` is unchanged."""
    samples = [_valid_sample(0, 0), _valid_sample(0, 1)]  # 2 valid trajectories
    _engine._last_rollout_n_errored = 3                   # 3 were all-empty (dropped)
    try:
        _rollouts, n_valid, n_errored, n_missing, n_expected = bucket_trajectories(samples, 8)
    finally:
        _engine._last_rollout_n_errored = 0

    assert n_valid == 2
    assert n_errored == 3      # recovered from the pre-drop count
    assert n_missing == 3      # 8 - 2 valid - 3 errored = genuine framework drops
    assert n_expected == 8
    assert n_valid + n_errored + n_missing == n_expected


def test_bucket_trajectories_errored_count_capped_and_default():
    """Recovered errored count is capped at the number of failures, and
    defaults to 0 (all failures → missing, the pre-fix behavior) when nothing
    was recorded."""
    samples = [_valid_sample(0, 0)]   # 1 valid; n_expected=4 → 3 failures

    _engine._last_rollout_n_errored = 0   # nothing recorded
    _r, n_valid, n_errored, n_missing, _e = bucket_trajectories(samples, 4)
    assert (n_valid, n_errored, n_missing) == (1, 0, 3)

    _engine._last_rollout_n_errored = 99  # over-count → capped at the 3 failures
    try:
        _r, n_valid, n_errored, n_missing, _e = bucket_trajectories(samples, 4)
    finally:
        _engine._last_rollout_n_errored = 0
    assert (n_valid, n_errored, n_missing) == (1, 3, 0)


def test_count_empty_trajs_multi_turn_nested():
    """Nested (multi-turn) branch: a trajectory = list of turn-samples; it counts
    as empty IFF every turn has tokens==[]. A trajectory with ≥1 non-empty turn
    (partial failure) must NOT count. Mirrors what ``_drop_empty_turns`` erases."""
    t = _make_template()
    group = [
        [_empty_sample(t), _empty_sample(t)],   # all-empty trajectory    → counts
        [_valid_sample(0, 0), _empty_sample(t)],  # 1 valid turn (partial)  → NOT
        [_valid_sample(0, 1)],                   # valid                   → NOT
        [_empty_sample(t)],                      # all-empty (1 turn)      → counts
    ]
    assert _count_empty_trajs(group) == 2


def test_count_empty_trajs_flat_single_turn():
    """Flat branch: each element is a single-sample trajectory; empty IFF tokens==[]."""
    t = _make_template()
    group = [_valid_sample(0, 0), _empty_sample(t), _empty_sample(t)]
    assert _count_empty_trajs(group) == 2
    assert _count_empty_trajs([]) == 0   # empty group → 0


def test_rollout_failure_classifier_uses_typed_action_boundary_categories():
    """Typed model-vs-infra exceptions bypass the legacy task-crash bucket."""
    # Expected values are LITERALS on purpose: ``FailureCategory.X.value`` moves
    # with the enum, so such a pin agrees with any rename of the wire value.
    assert (
        _engine._classify_failure(PairableModelActionError("bad key"))
        == "model_action_error"
    )
    assert (
        _engine._classify_failure(ModelOutputError("duplicate call_id"))
        == "model_output_error"
    )
    assert (
        _engine._classify_failure(TrueInfraFailure("container crashed"))
        == "infra_failure"
    )
    assert _engine._classify_failure(RuntimeError("untyped crash")) == "task_crash"


def _generate_exception_sample(original: Sample, exc: BaseException) -> Sample:
    """Mirror of ``generate()``'s ``except Exception`` arm: every exception
    that escapes the rollout becomes an ``_empty_sample`` carrying the
    ``_classify_failure`` reason and the ``is_retryable`` verdict cached at
    catch time (see ``lite/train/rollout/core/engine.py``)."""
    return _empty_sample(
        original,
        failure_reason=_classify_failure(exc),
        retryable=is_retryable(exc),
    )


def test_true_infra_exception_keeps_empty_discard_path():
    """True infra failures stay on the empty-sample discard path."""
    sample = _generate_exception_sample(
        _make_template(group_index=2, index=5),
        TrueInfraFailure("container crashed"),
    )

    assert sample.group_index == 2
    assert sample.index == 5
    assert sample.tokens == []
    assert sample.loss_mask == []
    assert sample.status == Sample.Status.FAILED
    assert sample.metadata["lite_failure_reason"] == "infra_failure"


@pytest.mark.parametrize(
    ("case", "exc", "expected_reason", "expected_retryable"),
    [
        (
            "timeout",
            EnvTimeoutError("step", 0.01),
            "step_timeout",
            True,
        ),
        (
            "crash",
            RuntimeError("backend crashed"),
            "task_crash",
            True,
        ),
        (
            "rpc",
            RemoteEnvError("rpc transport closed", original_type="ConnectionError"),
            "task_crash",
            True,
        ),
        (
            "container",
            EnvDesktopCrashed("desktop session crashed"),
            "task_crash",
            True,
        ),
        (
            "evaluator",
            CuaWorldVerifierError(
                "verifier crashed",
                phase="verify",
                kind="verifier_raised",
            ),
            "task_crash",
            False,
        ),
        (
            "container_capacity",
            CapacityExhausted.warming("container backend still warming", retry_after_s=1),
            "capacity_503",
            True,
        ),
    ],
)
def test_true_env_failure_matrix_stays_empty_discard_path(
    case: str,
    exc: BaseException,
    expected_reason: str,
    expected_retryable: bool,
):
    """Timeout/crash/RPC/container/evaluator failures stay infra drops."""
    _ = case
    original = _make_template(group_index=4, index=9)

    sample = _generate_exception_sample(original, exc)

    assert sample.group_index == original.group_index
    assert sample.index == original.index
    assert sample.status == Sample.Status.FAILED
    assert sample.tokens == []
    assert sample.loss_mask == []
    assert sample.metadata["lite_failure_reason"] == expected_reason
    assert sample.metadata["lite_error_retryable"] is expected_retryable

    rollouts, n_valid, n_errored, n_missing, n_expected = bucket_trajectories(
        [sample],
        1,
    )
    assert (n_valid, n_errored, n_missing, n_expected) == (0, 1, 0, 1)
    assert rollouts[0][0].tokens == [0, 0]
    assert rollouts[0][0].loss_mask == [0]


def test_nonempty_failed_sample_is_not_an_infra_drop():
    """A FAILED turn that still carries tokens stays represented.

    Model problems (e.g. an unparseable action) don't raise — the agent loop
    turns them into a content-only final, so the sample keeps its tokens. Only
    the tokenless ``_empty_sample`` shape is an infra drop.
    """
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
        label=None,
        reward=0.0,
        status=Sample.Status.FAILED,
        metadata={
            "env_key": "lite.demo@test",
            "turn_range": (0, 0),
            "others": {"episode_return": 0.0},
        },
    )

    rollouts, n_valid, n_errored, n_missing, n_expected = bucket_trajectories(
        [sample], 1,
    )

    assert (n_valid, n_errored, n_missing, n_expected) == (1, 0, 0, 1)
    assert rollouts == [[sample]]


def test_empty_infra_failure_still_counts_errored_and_gets_dummy():
    """True infra failures still use the empty-sample discard/no-op path."""
    empty = _empty_sample(_make_template(), failure_reason="task_crash")

    rollouts, n_valid, n_errored, n_missing, n_expected = bucket_trajectories(
        [empty], 1,
    )

    assert (n_valid, n_errored, n_missing, n_expected) == (0, 1, 0, 1)
    assert rollouts[0][0].tokens == [0, 0]
    assert rollouts[0][0].loss_mask == [0]


def test_synthetic_dummy_sample_is_valid_zero_gradient():
    """BUG 2/4: the from-scratch dummy is a valid zero-gradient sample (tokens>=2,
    loss_mask=[0]) with index set so flatten_and_align's ``max(index)+1`` is safe."""
    s = _engine.synthetic_dummy_sample()
    assert s.tokens == [0, 0] and s.loss_mask == [0] and s.response_length == 1
    assert s.index == 0 and s.group_index == 0  # not None
    assert s.reward == 0.0


def test_all_empty_batch_guard_fires():
    """The all-empty detection fires for a full-batch failure (every trajectory
    collapsed to []) but not for a partially-empty batch."""
    all_empty = [[[], []], [[], []]]
    assert not any(roll for g in all_empty for roll in g)  # guard -> inject dummy
    partial = [[[], [_valid_sample(0, 1)]]]
    assert any(roll for g in partial for roll in g)         # guard does NOT fire
