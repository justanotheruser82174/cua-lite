"""Tier-1 packing-arithmetic invariants.

These tests target the silent-correctness corners of packing-specific
arithmetic and propagation — fixtures that exercise ``_emit_rl_sample``
directly with synthetic ``LiteRLSample`` / ``LiteRLStep`` data and
assert the load-bearing metadata propagates correctly.

They do NOT need a model forward; they test:
- ``metadata["n_turns"]`` is the ORIGINAL turn count (not segment count)
- ``metadata["turn_range"]`` replaces the legacy ``turn_idx`` scalar
- ``metadata["step_statuses"]`` preserves per-step status detail
- ``Sample.status`` uses max-severity policy (ABORTED > TRUNCATED > COMPLETED)

These run in CI per PR (pure-function, slime-required for
``slime.utils.types.Sample`` import — skipped if slime not installed).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.types import Sample

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.samples import (
    STATUS_ABORTED,
    STATUS_COMPLETED,
    STATUS_TRUNCATED,
    LiteRLSample,
    LiteRLStep,
)
from lite.train.rollout.core.segmenter import _emit_rl_sample, build_segment_samples


def _make_step(prompt="p", response="r", reward=0.0, status=STATUS_COMPLETED):
    return LiteRLStep(
        prompt=prompt,
        image_indices=(),
        response=response,
        response_tokens=[1, 2, 3],
        response_log_probs=[-0.1, -0.2, -0.3],
        reward=reward,
        status=status,
    )


def _make_result(steps, episode_return=0.0):
    return LiteRLSample(
        processed_images=[],
        steps=list(steps),
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=episode_return,
        terminated=False,
        truncated=False,
    )


def _make_original():
    return Sample(
        group_index=0,
        index=0,
        prompt="orig prompt",
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.PENDING,
        metadata={"env_key": "lite.demo@test"},
    )


class _TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]


# -- n_turns propagation --------------------------------------------------

def test_n_turns_metadata_is_original_turn_count_not_segment_count():
    """🔴 Critical invariant: ``sample.metadata['n_turns']`` MUST equal the
    original trajectory turn count, independent of segmentation. v0.3.0 no
    longer divides the advantage by it (group_mask_sums does the per-trajectory
    weighting), but the field still feeds metrics and must not silently become
    the segment count."""
    steps = [_make_step(response=f"r{i}") for i in range(10)]
    result = _make_result(steps, episode_return=1.0)
    original = _make_original()

    # Emit length-1 segment for step 4 (mid-trajectory).
    sample = _emit_rl_sample(
        segment=[steps[4]],
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=[[0, 1, 2]],
        multimodal_train_inputs=None,
    )

    assert sample.metadata["n_turns"] == 10, (
        "n_turns must equal original turn count (10), not segment count (1)"
    )


def test_n_turns_identical_across_all_segments_of_same_trajectory():
    """All segments of one trajectory must report the same n_turns."""
    steps = [_make_step(response=f"r{i}") for i in range(7)]
    result = _make_result(steps)
    original = _make_original()

    samples = []
    for k in range(7):
        s = _emit_rl_sample(
            segment=[steps[k]],
            original_sample=original,
            rl_sample=result,
            per_step_prompt_ids=[[0]],
            multimodal_train_inputs=None,
        )
        samples.append(s)

    n_turns_values = {s.metadata["n_turns"] for s in samples}
    assert n_turns_values == {7}, f"n_turns drifted across segments: {n_turns_values}"


# -- turn_range replaces turn_idx ----------------------------------------

def test_turn_range_replaces_legacy_turn_idx():
    """``metadata['turn_range']`` is (first, last) inclusive. ``turn_idx``
    is gone — never present in emitted Sample.metadata."""
    steps = [_make_step(response=f"r{i}") for i in range(3)]
    result = _make_result(steps)
    original = _make_original()

    sample = _emit_rl_sample(
        segment=[steps[1]],
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=[[0]],
        multimodal_train_inputs=None,
    )

    assert sample.metadata["turn_range"] == (1, 1)
    assert "turn_idx" not in sample.metadata, "legacy turn_idx must not appear"


def test_turn_range_uses_step_identity_not_dataclass_equality():
    """Equal-valued turns are common enough in rollout data: repeated wait/no-op
    or repeated text-only actions can produce identical ``LiteRLStep`` values.
    The emitted range must track the concrete turn object, not the first equal
    value in ``rl_sample.steps``.
    """
    steps = [_make_step(), _make_step()]
    assert steps[0] == steps[1]
    assert steps[0] is not steps[1]
    result = _make_result(steps)
    original = _make_original()

    sample = _emit_rl_sample(
        segment=[steps[1]],
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=[[0]],
        multimodal_train_inputs=None,
    )

    assert sample.metadata["turn_range"] == (1, 1)


def test_explicit_turn_range_takes_precedence_for_equal_valued_segment():
    steps = [_make_step(), _make_step()]
    result = _make_result(steps)
    original = _make_original()

    sample = _emit_rl_sample(
        segment=[steps[0]],
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=[[0]],
        multimodal_train_inputs=None,
        turn_range=(1, 1),
    )

    assert sample.metadata["turn_range"] == (1, 1)


def test_build_segment_samples_preserves_turn_range_for_equal_steps(monkeypatch):
    """The public rollout path must thread segment indices into the emitter.

    This pins GRPO/SFT consistency: both consume ``build_segment_samples`` and
    later regroup by ``metadata["turn_range"]``. Equal-valued steps must not
    collapse to the first equal turn when radix is disabled for per-turn GSPO or
    when radix naturally emits separate length-1 segments.
    """
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    steps = [_make_step(), _make_step()]
    assert steps[0] == steps[1]
    result = _make_result(steps)

    samples = build_segment_samples(
        rl_sample=result,
        original_sample=_make_original(),
        processor=None,
        tokenizer=_TinyTokenizer(),
        state=SimpleNamespace(aborted=False),
    )

    assert [s.metadata["turn_range"] for s in samples] == [(0, 0), (1, 1)]


def _prefix_extensional_prompt_ids(segment, initial_prompt_len: int = 1) -> list[list[int]]:
    """Build per-step prompt_ids satisfying the radix prefix invariant.

    Each step's prompt = previous prompt + previous response + a small
    delta (one synthetic user-obs token). Mirrors what a real
    prefix-extensional protocol produces.
    """
    out: list[list[int]] = []
    cumulative = list(range(initial_prompt_len))      # initial prompt
    for i, step in enumerate(segment):
        if i == 0:
            out.append(list(cumulative))
        else:
            prev = segment[i - 1]
            # delta = 1 user-obs token
            cumulative = cumulative + list(prev.response_tokens) + [9000 + i]
            out.append(list(cumulative))
    return out


def test_turn_range_for_packed_segment_spans_first_to_last():
    """Length-k>1 segment turn_range is (first, last). Exercises the
    packing path — emitter must handle multi-step segments correctly
    with prefix-extensional prompt_ids."""
    steps = [_make_step(response=f"r{i}") for i in range(5)]
    result = _make_result(steps)
    original = _make_original()

    seg = [steps[1], steps[2], steps[3]]
    sample = _emit_rl_sample(
        segment=seg,
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=_prefix_extensional_prompt_ids(seg),
        multimodal_train_inputs=None,
    )

    assert sample.metadata["turn_range"] == (1, 3)


# -- status propagation policy --------------------------------------------

def test_status_aborted_wins_over_truncated_and_completed():
    """ABORTED > TRUNCATED > COMPLETED per max-severity policy."""
    steps = [
        _make_step(status=STATUS_COMPLETED),
        _make_step(status=STATUS_TRUNCATED),
        _make_step(status=STATUS_ABORTED),
    ]
    result = _make_result(steps)
    original = _make_original()

    sample = _emit_rl_sample(
        segment=steps,
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=_prefix_extensional_prompt_ids(steps),
        multimodal_train_inputs=None,
    )

    assert sample.status == Sample.Status.ABORTED


def test_status_truncated_wins_over_completed():
    steps = [
        _make_step(status=STATUS_COMPLETED),
        _make_step(status=STATUS_COMPLETED),
        _make_step(status=STATUS_TRUNCATED),
    ]
    result = _make_result(steps)
    original = _make_original()

    sample = _emit_rl_sample(
        segment=steps,
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=_prefix_extensional_prompt_ids(steps),
        multimodal_train_inputs=None,
    )

    assert sample.status == Sample.Status.TRUNCATED


def test_status_out_of_vocabulary_mixed_with_a_legal_status_raises():
    """An unknown status must not be outranked by a legal sibling status.

    The rank lookup is a plain subscript of the severity map built from
    ``STEP_STATUSES_BY_SEVERITY``, so an unknown string raises instead of
    scoring below ``completed`` and letting a corrupted turn train as complete.
    """
    steps = [
        _make_step(status=STATUS_COMPLETED),
        _make_step(status="not-a-status"),
    ]
    result = _make_result(steps)
    original = _make_original()

    with pytest.raises(KeyError, match="not-a-status"):
        _emit_rl_sample(
            segment=steps,
            original_sample=original,
            rl_sample=result,
            per_step_prompt_ids=_prefix_extensional_prompt_ids(steps),
            multimodal_train_inputs=None,
        )


def test_step_statuses_metadata_preserves_per_step_detail():
    """Even when the segment's status collapses to a single max-severity
    value, the per-step detail must survive in metadata for downstream
    filters that want finer granularity."""
    steps = [
        _make_step(status=STATUS_COMPLETED),
        _make_step(status=STATUS_TRUNCATED),
        _make_step(status=STATUS_COMPLETED),
    ]
    result = _make_result(steps)
    original = _make_original()

    sample = _emit_rl_sample(
        segment=steps,
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=_prefix_extensional_prompt_ids(steps),
        multimodal_train_inputs=None,
    )

    assert sample.metadata["step_statuses"] == (
        STATUS_COMPLETED, STATUS_TRUNCATED, STATUS_COMPLETED,
    )


# -- episode_return propagation ------------------------------------------

def test_episode_return_propagates_to_sample_metadata():
    steps = [_make_step(response=f"r{i}") for i in range(3)]
    result = _make_result(steps, episode_return=2.5)
    original = _make_original()

    sample = _emit_rl_sample(
        segment=[steps[0]],
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=[[0]],
        multimodal_train_inputs=None,
    )

    assert sample.metadata["others"]["episode_return"] == 2.5
    assert "episode_return" not in sample.metadata


# -- token aggregation across segments -----------------------------------

def test_response_tokens_aggregate_via_prefix_extension():
    """Length-k segment with prefix-extensional prompt_ids.

    The final ``Sample.tokens`` is ``segment[-1].prompt_tokens +
    segment[-1].response_tokens`` — every earlier step's response is
    already embedded inside the last step's prefix via the radix
    invariant. Multi-span loss_mask marks each step's response
    positions as 1.
    """
    steps = [
        LiteRLStep(prompt="p", image_indices=(), response="a",
                   response_tokens=[10, 11], response_log_probs=[-0.1, -0.2],
                   reward=0.0, status=STATUS_COMPLETED),
        LiteRLStep(prompt="p", image_indices=(), response="b",
                   response_tokens=[20, 21, 22], response_log_probs=[-0.3, -0.4, -0.5],
                   reward=0.0, status=STATUS_COMPLETED),
    ]
    result = _make_result(steps)
    original = _make_original()

    # Prefix-extensional: step_0 prompt=[1,2,3]; step_1 prompt=[1,2,3,10,11,9001].
    per_step = _prefix_extensional_prompt_ids(steps, initial_prompt_len=3)

    sample = _emit_rl_sample(
        segment=steps,
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=per_step,
        multimodal_train_inputs=None,
    )

    # Final tokens = step_1.prompt + step_1.response.
    # step_1.prompt = step_0.prompt ([0,1,2]) + step_0.response ([10,11])
    #                 + delta ([9001]) = [0,1,2,10,11,9001]
    # step_1.response = [20,21,22]
    # Final tokens = [0,1,2,10,11,9001,20,21,22], length 9.
    assert sample.tokens == [0, 1, 2, 10, 11, 9001, 20, 21, 22]
    # response_length = total - initial_prompt_len = 9 - 3 = 6.
    assert sample.response_length == 6
    # Multi-span mask: step_0 response at mask[0:2], delta at mask[2],
    # step_1 response at mask[3:6].
    assert sample.loss_mask == [1, 1, 0, 1, 1, 1]
    # log_probs follow the same layout (delta position is filler 0.0).
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, -0.3, -0.4, -0.5]
