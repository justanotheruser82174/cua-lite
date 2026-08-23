"""Pin slime v0.3.0's bshd/static micro-batch divisibility constraint (doc section 6 "M2").

``build_dp_schedule``'s STATIC path (qwen3.5 / GDN: ``--qkv-format bshd`` +
fixed ``--micro-batch-size``) requires each step's micro-batch count to be a
multiple of ``dp_size * mb_group``; with mbs=1 that is **per-step segment
count % dp_size == 0**. Segment count is data-dependent (turns × packing) and
cua-lite's group-level padding cannot guarantee it, so qwen3.5 at dp>1 can
crash mid-run with a loud AssertionError. All single-GPU runs (dp=1) are
trivially safe. The DYNAMIC path (qwen3-vl / qwen2.5-vl THD) aligns by
splitting bins; when splitting saturates (max_tokens_per_gpu < 2x sample
length → 1 sample/bin) it falls back to MERGING the smallest bins down to the
previous aligned multiple (cua-lite slime patch — the stock assert crashed
androidworld dp=4 at rollout 0).

These tests document the constraint. If a slime upgrade ever makes the static
path auto-split (test 2 stops raising), revisit doc section 6 M2 and drop the dp=1
recommendation for qwen3.5.

Slime-required.

Usage:
    pytest tests/train/utils/test_dp_schedule_bshd_divisibility.py -v
"""

from __future__ import annotations

from argparse import Namespace

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.dp_schedule import build_dp_schedule  # noqa: E402


def _schedule(group_indices, dp_size, *, dynamic=False):
    n = len(group_indices)
    args = Namespace(
        micro_batch_size=1,
        use_dynamic_batch_size=dynamic,
        max_tokens_per_gpu=64 if dynamic else None,
        balance_data=False,
    )
    tpc = {
        "dp_size": dp_size,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    return build_dp_schedule(
        args, tpc, [16] * n, global_batch_size=2, group_indices=group_indices
    )


# one step of 2 groups: group0 has 3 segments, group1 has 2 → 5 segments (ODD)
ODD = [0, 0, 0, 1, 1]
# 2 + 2 → 4 segments (EVEN)
EVEN = [0, 0, 1, 1]


def test_static_odd_segments_dp1_ok():
    """dp=1: align_to=1, any segment count schedules (all our verified runs)."""
    partitions, mb_indices, num_mbs, gbs = _schedule(ODD, dp_size=1)
    assert num_mbs == [5] and gbs == [2]


def test_static_odd_segments_dp2_raises():
    """dp=2 + odd per-step segment count → the documented loud crash (M2)."""
    with pytest.raises(AssertionError, match="static path"):
        _schedule(ODD, dp_size=2)


def test_static_even_segments_dp2_ok():
    """dp=2 + even segment count satisfies the invariant."""
    partitions, mb_indices, num_mbs, gbs = _schedule(EVEN, dp_size=2)
    assert num_mbs == [2]  # 4 mbs / 2 ranks


def test_dynamic_path_immune_at_dp2():
    """The THD/dynamic path (VL family) auto-splits bins to the alignment
    target — same odd-segment input must NOT raise."""
    partitions, mb_indices, num_mbs, gbs = _schedule(ODD, dp_size=2, dynamic=True)
    assert num_mbs[0] >= 1


def test_dynamic_singleton_bins_merge_down_not_raise():
    """Splitting-saturated case (cap 64 < 2x length 40 → 1 sample/bin): 5 odd
    singleton bins at dp=2 cannot split up to 6 — the merge-down fallback must
    align to 4 instead of raising (androidworld dp=4 regression)."""
    n = len(ODD)
    args = Namespace(
        micro_batch_size=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=64,
        balance_data=False,
    )
    tpc = {"dp_size": 2, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
    partitions, mb_indices, num_mbs, gbs = build_dp_schedule(
        args, tpc, [40] * n, global_batch_size=2, group_indices=ODD
    )
    assert num_mbs == [2]  # 5 bins -> merge down to 4 -> 2 per rank
    assert sorted(i for p in partitions for i in p) == list(range(n))  # no sample dropped
