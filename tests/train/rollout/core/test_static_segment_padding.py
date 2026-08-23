"""Integration: cua-lite convert output → slime static-path padding → schedule.

The bshd STATIC dp_schedule path (qwen3.5 / GDN) requires each step's
micro-batch count to divide ``dp_size * micro_batch_size``; slime pads every
group with zero-loss dummy rows at the convert boundary
(``pad_static_groups``, called from ``_split_train_data_by_dp``) — hit in
production on qwen3.5 mobilegym dp=4 (415-segment step). cua-lite itself does
NO segment padding: these tests pin the full chain
``flatten_and_align → pad_static_groups → build_dp_schedule`` so a regression
on either side fails here.

Slime-required.

Usage:
    pytest tests/train/rollout/core/test_static_segment_padding.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.dp_schedule import build_dp_schedule, pad_static_groups  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

from lite.train.rollout.core.engine import flatten_and_align  # noqa: E402


@dataclass
class Args:
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 1
    use_dynamic_batch_size: bool = False
    micro_batch_size: int = 1
    max_tokens_per_gpu: int | None = None
    balance_data: bool = False
    global_batch_size: int = 2


def make_tp(dp_size=4):
    return {
        "dp_size": dp_size,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }


def make_rollouts(seg_counts: list[int]) -> list[list[Sample]]:
    rollouts = []
    for ti, n_segs in enumerate(seg_counts):
        rollouts.append([
            Sample(
                group_index=ti,
                index=ti,
                group_id=ti,
                prompt=f"p{ti}",
                tokens=[1, 2, 3, 4],
                loss_mask=[1, 1],
                rollout_log_probs=[-0.1, -0.1],
                response_length=2,
                response="x",
                reward=1.0,
                status=Sample.Status.COMPLETED,
                metadata={"others": {"episode_return": float(ti)}},
            )
            for _ in range(n_segs)
        ])
    return rollouts


def test_full_chain_static_dp4():
    """Ragged trajectories (3+5 segments) flow through cua-lite's convert
    UNPADDED, then slime pads at the boundary and the static schedule builds."""
    args = Args()
    td = flatten_and_align(make_rollouts([3, 5]), args)

    # cua-lite emits exactly the real segments — no padding on our side.
    assert len(td["tokens"]) == 8

    tp = make_tp(dp_size=4)
    pad_static_groups(args, tp, td)

    # slime padded each group to a multiple of 4 (3→4, 5→8) with zero-loss rows
    # whose group_mask_sums reuse the group's (real) total.
    assert len(td["tokens"]) == 12
    for i in range(8, 12):
        assert sum(td["loss_masks"][i]) == 0
        gid = td["group_ids"][i]
        assert td["group_mask_sums"][i] == (6 if gid == 0 else 10)
    # raw_reward keeps cua-lite's per-trajectory shape.
    assert td["raw_reward"] == [0.0, 1.0]

    total_lengths = [len(t) for t in td["tokens"]]
    partitions, mbi, nmb, gbs = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, group_indices=td["group_ids"]
    )
    assert nmb == [3]  # 12 mbs / 4 ranks
    assert sorted(i for p in partitions for i in p) == list(range(12))


def test_full_chain_dynamic_unpadded():
    """VL/THD path: slime's pad is a no-op; alignment is the bin splitter's job."""
    args = Args(use_dynamic_batch_size=True, max_tokens_per_gpu=64)
    td = flatten_and_align(make_rollouts([3, 5]), args)
    pad_static_groups(args, make_tp(dp_size=4), td)
    assert len(td["tokens"]) == 8  # untouched


def test_full_chain_group_dummy_padded_by_slime():
    """A dropped trajectory becomes a 1-segment dummy group in cua-lite's
    group-level padding; slime's boundary pad must grow it to the unit too."""
    args = Args(rollout_batch_size=3)  # 2 real + 1 cua-lite dummy group
    td = flatten_and_align(make_rollouts([4, 4]), args)
    assert len(td["tokens"]) == 9  # 4 + 4 + 1 (group dummy)

    pad_static_groups(args, make_tp(dp_size=4), td)
    counts: dict[int, int] = {}
    for gid in td["group_ids"]:
        counts[gid] = counts.get(gid, 0) + 1
    assert sorted(counts.values()) == [4, 4, 4]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
