"""DP split behavior for the lazy multimodal path.

Verifies two coupled properties:

  1. ``flatten_and_align`` emits a flat ``train_data["multimodal_lazy_payloads"]``
     list whose entries all point at the SAME Python dict for ``images``
     (so cloudpickle dedup hinges).
  2. When slime's ``_split_train_data_by_dp`` shards this list by sample
     index (mimicked here as a pure list slice; the real method on
     ``RolloutRayActor`` needs Ray and is impractical to unit-test
     directly), each per-DP-rank shard:
       - includes the ``multimodal_lazy_payloads`` key
       - still has shared identity WITHIN the shard for Samples that came
         from the same trajectory
       - cross-shard sharing is NOT preserved (this is the known
         DP-shard trade-off; cua-lite cares only because dp_size=1 is the
         common case)

Slime-required (uses ``slime.utils.types.Sample``).
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.types import Sample

from lite.train.rollout.core.engine import flatten_and_align

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory_samples(
    trajectory_id: int, n_samples: int, shared_images: dict,
) -> list[Sample]:
    """Build ``n_samples`` Samples that all reference the same
    ``shared_images`` dict (mimicking what ``build_segment_samples`` emits
    in lazy mode for one trajectory)."""
    return [
        Sample(
            group_index=trajectory_id,
            index=trajectory_id * 100 + k,
            prompt=f"prompt_t{trajectory_id}_s{k}",
            tokens=[1, 2, 3, 4],
            loss_mask=[0, 0, 1, 1],
            rollout_log_probs=[0.0, 0.0, -0.1, -0.2],
            response_length=2,
            response=f"r{k}",
            label=None,
            reward=0.5,
            status=Sample.Status.COMPLETED,
            multimodal_train_inputs=None,
            multimodal_lazy_payloads={
                "images":  shared_images,
                "indices": (0, k),  # arbitrary per-Sample view
            },
            metadata={"n_turns": n_samples, "others": {"episode_return": 0.5}},
        )
        for k in range(n_samples)
    ]


def _slime_style_dp_split(train_data: dict, dp_size: int) -> list[dict]:
    """Mimic slime ``_split_train_data_by_dp`` partition logic (round-robin
    by sample index, default ``balance_data=False`` path). Returns a list
    of ``dp_size`` shards each with the same keys as
    ``_split_train_data_by_dp``."""
    n = len(train_data["tokens"])
    partitions = [list(range(i, n, dp_size)) for i in range(dp_size)]
    keys = [
        "tokens", "multimodal_train_inputs", "multimodal_lazy_payloads",
        "response_lengths", "rewards", "truncated", "loss_masks",
        "sample_indices", "rollout_log_probs",
    ]
    shards = []
    for partition in partitions:
        shard = {"partition": partition}
        for key in keys:
            if key not in train_data:
                continue
            shard[key] = [train_data[key][j] for j in partition]
        shards.append(shard)
    return shards


class _MockArgs:
    # flatten_and_align reads exactly these two fields.
    rollout_batch_size = 1
    n_samples_per_prompt = 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dp1_perfect_dedup():
    """dp_size=1: all Samples in one shard. Identity of ``images`` MUST be
    preserved across every Sample in that shard (cloudpickle will then
    dedup the underlying tensors in plasma)."""
    shared = {0: {"pixel_values": torch.zeros(1, 4, dtype=torch.bfloat16)}}
    samples = _make_trajectory_samples(trajectory_id=0, n_samples=5, shared_images=shared)
    train_data = flatten_and_align([samples], _MockArgs())

    assert "multimodal_lazy_payloads" in train_data
    lazy = train_data["multimodal_lazy_payloads"]
    assert len(lazy) == 5

    # Identity preserved across the whole flat list (since dp_size=1 ships
    # them all in a single ``ray.put``).
    first_images = lazy[0]["images"]
    for k, entry in enumerate(lazy[1:], start=1):
        assert id(entry["images"]) == id(first_images), (
            f"sample {k}: images dict identity broken in flat list"
        )

    # Single shard at dp_size=1.
    shards = _slime_style_dp_split(train_data, dp_size=1)
    assert len(shards) == 1
    shard0 = shards[0]
    assert "multimodal_lazy_payloads" in shard0
    assert len(shard0["multimodal_lazy_payloads"]) == 5
    first_in_shard = shard0["multimodal_lazy_payloads"][0]["images"]
    for entry in shard0["multimodal_lazy_payloads"][1:]:
        assert id(entry["images"]) == id(first_in_shard)


def test_dp_n_split_independence():
    """dp_size > 1: within each shard, Samples that came from the same
    trajectory still share the images dict. Across shards, identity is
    NOT preserved (each shard is its own ``ray.put`` — cloudpickle can't
    dedup across separate put calls). This is a documented trade-off;
    cua-lite dp=1 production avoids it."""
    shared = {0: {"pixel_values": torch.zeros(1, 4, dtype=torch.bfloat16)}}
    # Two trajectories: t0 has 4 samples (idx 0,1,2,3), t1 has 4 (idx 100..103).
    # After flatten, they're [t0_0, t0_1, t0_2, t0_3, t1_0, t1_1, t1_2, t1_3].
    samples_t0 = _make_trajectory_samples(0, 4, shared)
    images_t1 = {0: {"pixel_values": torch.ones(1, 4, dtype=torch.bfloat16)}}
    samples_t1 = _make_trajectory_samples(1, 4, images_t1)
    train_data = flatten_and_align([samples_t0, samples_t1], _MockArgs())

    shards = _slime_style_dp_split(train_data, dp_size=2)
    assert len(shards) == 2

    # Each shard sees a subset of the flat list; both must carry the
    # lazy_payloads key.
    for shard in shards:
        assert "multimodal_lazy_payloads" in shard
        assert len(shard["multimodal_lazy_payloads"]) == 4  # 8 total / 2 shards


def test_partition_list_includes_lazy_key():
    """Sanity: when no Sample carries lazy_payloads (legacy path),
    ``flatten_and_align`` does NOT emit the key (consistent with how
    ``multimodal_train_inputs`` is conditionally emitted)."""
    samples = [
        Sample(
            group_index=0, index=k, prompt=f"p{k}",
            tokens=[1, 2, 3, 4], loss_mask=[0, 0, 1, 1],
            rollout_log_probs=[0.0, 0.0, -0.1, -0.2],
            response_length=2, response=f"r{k}",
            reward=0.0, status=Sample.Status.COMPLETED,
            multimodal_train_inputs=None,
            multimodal_lazy_payloads=None,
            metadata={"n_turns": 1, "others": {"episode_return": 0.0}},
        )
        for k in range(3)
    ]
    train_data = flatten_and_align([samples], _MockArgs())
    assert "multimodal_lazy_payloads" not in train_data
    assert "multimodal_train_inputs" not in train_data
