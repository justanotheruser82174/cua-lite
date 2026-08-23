"""Length-1 ``_emit_rl_sample`` contract tests.

For RL configs with `history_n=1` / `full_history_size=1` (the current
common case), the radix segmenter produces only length-1 segments.
The emitted ``slime.Sample`` must keep the one-turn token/loss shape while
carrying the current trajectory metadata:
  - ``n_turns``, ``turn_range``, ``step_statuses``
  - ``others["episode_return"]``
  - no ``turn_idx``

Slime-required.
"""

from __future__ import annotations

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
from lite.train.rollout.core.segmenter import _emit_rl_sample


def _make_orig():
    return Sample(
        group_index=2,
        index=5,
        prompt="ORIG PROMPT",
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.PENDING,
        metadata={"env_key": "lite.demo@smoke", "split": "train"},
    )


@pytest.mark.parametrize("status_str,expected_slime_status", [
    (STATUS_COMPLETED, Sample.Status.COMPLETED),
    (STATUS_TRUNCATED, Sample.Status.TRUNCATED),
    (STATUS_ABORTED, Sample.Status.ABORTED),
])
@pytest.mark.parametrize("response_len", [1, 5, 32, 128])
def test_length1_segment_emits_current_sample_fields(
    status_str, expected_slime_status, response_len,
):
    """Length-1 segments emit one prompt plus one response and current metadata."""
    prompt_ids = list(range(10))
    response_ids = list(range(100, 100 + response_len))
    response_log_probs = [-(i + 1) * 0.01 for i in range(response_len)]
    response_text = f"resp_{response_len}"
    reward = 0.7 if status_str == STATUS_COMPLETED else 0.0
    multimodal = {"pixel_values": "MOCKED"}

    original = _make_orig()

    step = LiteRLStep(
        prompt="rendered prompt text",
        image_indices=(0,),
        response=response_text,
        response_tokens=response_ids,
        response_log_probs=response_log_probs,
        reward=reward,
        status=status_str,
    )
    result = LiteRLSample(
        processed_images=[],
        steps=[step],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=reward,
        terminated=(status_str == STATUS_COMPLETED),
    )
    post = _emit_rl_sample(
        segment=[step],
        original_sample=original,
        rl_sample=result,
        per_step_prompt_ids=[prompt_ids],
        multimodal_train_inputs=multimodal,
    )

    assert post.tokens == prompt_ids + response_ids
    assert post.loss_mask == [1] * response_len
    assert post.response_length == response_len
    assert post.rollout_log_probs == response_log_probs
    assert post.response == response_text
    assert post.reward == reward
    assert post.status == expected_slime_status
    assert post.multimodal_train_inputs == multimodal
    # Lazy payloads default to None unless the caller passes the shared payload.
    assert post.multimodal_lazy_payloads is None
    assert post.group_index == original.group_index
    assert post.index == original.index
    assert post.group_id == original.index
    assert post.prompt == original.prompt
    assert post.label == original.label

    # Metadata carries current trajectory fields; turn_idx is no longer emitted.
    assert "turn_idx" not in post.metadata, (
        "turn_idx must not appear in current Sample.metadata"
    )
    assert post.metadata["n_turns"] == 1, "single-step trajectory → n_turns=1"
    assert post.metadata["turn_range"] == (0, 0), "length-1 segment at turn 0"
    assert post.metadata["step_statuses"] == (status_str,)
    assert post.metadata["others"]["episode_return"] == reward
    assert "episode_return" not in post.metadata

    # Common metadata from the original sample is preserved.
    for k in ("env_key", "split"):
        assert post.metadata[k] == original.metadata[k], f"{k} mismatch"


def test_length1_metadata_n_turns_equals_segment_count_when_T_eq_1():
    """When T=1, the emitter sees 1 segment of 1 step → n_turns=1.
    This is the only case where len(rollout) == n_turns (so the
    metadata['n_turns'] read in rollout_grpo doesn't drift)."""
    step = LiteRLStep(
        prompt="p", image_indices=(),
        response="r", response_tokens=[1, 2, 3], response_log_probs=[-0.1] * 3,
        reward=1.0, status=STATUS_COMPLETED,
    )
    result = LiteRLSample(
        processed_images=[],
        steps=[step],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )
    sample = _emit_rl_sample(
        segment=[step],
        original_sample=_make_orig(),
        rl_sample=result,
        per_step_prompt_ids=[[1, 2]],
        multimodal_train_inputs=None,
    )
    assert sample.metadata["n_turns"] == 1
    assert sample.metadata["turn_range"] == (0, 0)


def test_lazy_payload_passes_through_emit_rl_sample():
    """``_emit_rl_sample`` faithfully threads ``multimodal_lazy_payloads``
    onto the emitted Sample without touching it. Asserts the field flows
    through unmodified — the slime get_batch hook is the only consumer
    that interprets the dict contents."""
    step = LiteRLStep(
        prompt="p", image_indices=(0,),
        response="r", response_tokens=[1, 2, 3], response_log_probs=[-0.1] * 3,
        reward=1.0, status=STATUS_COMPLETED,
    )
    result = LiteRLSample(
        processed_images=[object()],
        steps=[step],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )
    shared_images = {0: {"pixel_values": "MOCKED"}}
    payload = {"images": shared_images, "indices": (0,)}
    sample = _emit_rl_sample(
        segment=[step],
        original_sample=_make_orig(),
        rl_sample=result,
        per_step_prompt_ids=[[1, 2]],
        multimodal_train_inputs=None,
        multimodal_lazy_payloads=payload,
    )
    assert sample.multimodal_train_inputs is None
    # Identity, not just equality — the rollout-side dedup relies on
    # the dict object being the same Python instance.
    assert sample.multimodal_lazy_payloads is payload
    assert sample.multimodal_lazy_payloads["images"] is shared_images
    assert sample.multimodal_lazy_payloads["indices"] == (0,)


def test_long_trajectory_n_turns_propagates_correctly():
    """For T=10 trajectory, every emitted Sample must carry n_turns=10.
    v0.3.0: n_turns is metadata-only (the old adv/n_turns hand-normalization is
    gone; per-trajectory weighting flows via group_mask_sums) but must stay
    accurate for metrics/debugging."""
    steps = [
        LiteRLStep(
            prompt=f"p{k}", image_indices=(k,),
            response=f"r{k}", response_tokens=[k, k + 1],
            response_log_probs=[-0.1, -0.2],
            reward=0.0, status=STATUS_COMPLETED,
        )
        for k in range(10)
    ]
    result = LiteRLSample(
        processed_images=[],
        steps=steps,
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )
    # Emit each step as a length-1 segment (current RL config behavior).
    for k in range(10):
        sample = _emit_rl_sample(
            segment=[steps[k]],
            original_sample=_make_orig(),
            rl_sample=result,
            per_step_prompt_ids=[[1, 2]],
            multimodal_train_inputs=None,
        )
        assert sample.metadata["n_turns"] == 10, (
            f"step {k}: n_turns must be ORIGINAL turn count (10), got {sample.metadata['n_turns']}"
        )
        assert sample.metadata["turn_range"] == (k, k)
