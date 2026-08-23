"""Radix segmentation unit tests — pure-function, no model forward.

Exercise ``_segment_steps`` against synthetic ``per_step_tokens`` data.
Verifies the greedy prefix-equality check at the heart of the segmenter.

Slime-required for the import side (``_segment_steps`` lives next to
``_emit_rl_sample``); auto-skipped without slime.
"""

from __future__ import annotations

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")

from lite.train.rollout.core.segmenter import _segment_steps


class _MockStep:
    """Minimal LiteRLStep stand-in carrying image_indices for the segmenter."""
    def __init__(self, image_indices=()):
        self.image_indices = tuple(image_indices)


def _step(prompt_tokens, response_tokens, image_indices=()):
    return {
        "prompt_tokens": list(prompt_tokens),
        "response_tokens": list(response_tokens),
        "step": _MockStep(image_indices=image_indices),
    }


# -- happy path: full extensional trajectory packs to length-1 segment --

def test_all_pack_full_extensional():
    """Trajectory of T=4 where each step's prompt is exactly the prior
    step's prompt + response. Segmenter must produce one length-T segment.
    """
    p1, r1 = [1, 2, 3], [10, 11]
    p2, r2 = p1 + r1 + [99, 88], [12, 13]
    p3, r3 = p2 + r2 + [77, 66], [14]
    p4, r4 = p3 + r3 + [55], [15, 16, 17]

    per_step = [_step(p1, r1), _step(p2, r2), _step(p3, r3), _step(p4, r4)]
    segments = _segment_steps(per_step)

    assert segments == [[0, 1, 2, 3]]


def test_markov_protocol_yields_t_length_1_segments():
    """Markov protocol (k=1, every step starts fresh) — segmenter
    naturally produces T length-1 segments. Packing doesn't regress these."""
    per_step = [
        _step([1, 2], [10]),
        _step([3, 4], [11]),     # NOT a prefix-extension of step 0
        _step([5, 6], [12]),
        _step([7, 8], [13]),
    ]
    segments = _segment_steps(per_step)

    assert segments == [[0], [1], [2], [3]]


def test_partial_break_mid_trajectory():
    """Trajectory where steps 0-2 are extensional, then step 3 breaks
    (e.g. fold/window slide), then steps 4-5 form a new extensional run."""
    p0, r0 = [1, 2], [10]
    p1, r1 = p0 + r0 + [3], [11]
    p2, r2 = p1 + r1 + [4], [12]
    # Step 3 breaks: completely fresh prompt (e.g. window folded).
    p3, r3 = [50, 51], [20]
    # Steps 4-5 extensional from step 3.
    p4, r4 = p3 + r3 + [52], [21, 22]
    p5, r5 = p4 + r4 + [53], [23]

    per_step = [
        _step(p0, r0), _step(p1, r1), _step(p2, r2),
        _step(p3, r3), _step(p4, r4), _step(p5, r5),
    ]
    segments = _segment_steps(per_step)

    assert segments == [[0, 1, 2], [3, 4, 5]]


def test_single_step_trajectory():
    """T=1 trajectory: one segment of one step."""
    per_step = [_step([1, 2, 3], [10, 11])]
    segments = _segment_steps(per_step)

    assert segments == [[0]]


def test_empty_input():
    """No steps → no segments."""
    assert _segment_steps([]) == []


def test_strict_prefix_required_not_just_overlap():
    """The check is strict prefix on prompt_tokens, not partial overlap.
    A step where ``curr.prompt_tokens`` overlaps but doesn't fully
    contain ``prev.prompt + prev.response`` must break the segment."""
    # Prev: prompt=[1,2,3], response=[10,11] → expected prefix = [1,2,3,10,11]
    # Curr: prompt=[1,2,3,99,88]  — contains [1,2,3] but NOT [1,2,3,10,11]
    per_step = [
        _step([1, 2, 3], [10, 11]),
        _step([1, 2, 3, 99, 88], [12]),
    ]
    segments = _segment_steps(per_step)

    assert segments == [[0], [1]]


def test_prefix_extending_trajectory_with_zero_response():
    """Edge case: a step with empty response_tokens. Next step's prompt
    must start with prev's prompt (no response to add) for packing."""
    p0, r0 = [1, 2, 3], []
    p1 = p0 + [4]   # prefix extension of p0 (since r0 is empty)
    r1 = [10]
    per_step = [_step(p0, r0), _step(p1, r1)]
    segments = _segment_steps(per_step)

    assert segments == [[0, 1]]


# -- 🔴 image_indices prefix check (defensive correctness) ---

def test_image_indices_must_also_be_prefix_to_pack():
    """Token-prefix-equal steps with mismatched image_indices must NOT pack.

    For VLMs, the multimodal processor expands ``<|image_pad|>`` into N
    copies of a constant ``image_token_id`` where N depends only on
    image dimensions. Two different images of the same size produce
    byte-identical input_ids at IMG_PAD positions; the actual visual
    content lives in pixel_values. Without the image-index check, a
    pathological non-extensional protocol that swaps images while
    preserving text prefix would incorrectly pack.
    """
    # Token-level prefix-extensional pair, but step_1 references a
    # different image set (image_indices: [0,1] → [2,3] instead of
    # the extensional [0,1] → [0,1,2]).
    p0, r0 = [1, 2, 3], [10, 11]
    p1 = p0 + r0 + [99, 88]  # token prefix-extends p0+r0
    r1 = [12]
    per_step = [
        _step(p0, r0, image_indices=(0, 1)),
        _step(p1, r1, image_indices=(2, 3)),  # NOT a prefix of (0,1)
    ]
    segments = _segment_steps(per_step)

    assert segments == [[0], [1]], (
        f"image_indices mismatch must force length-1 fallback; got {segments}"
    )


def test_image_indices_extension_is_packed():
    """Token-prefix-equal AND image-index-prefix steps SHOULD pack.

    The extensional case: step_1 references step_0's images plus new
    ones — image_indices=(0,1) → (0,1,2). Both checks pass.
    """
    p0, r0 = [1, 2, 3], [10, 11]
    p1 = p0 + r0 + [99]
    r1 = [12]
    per_step = [
        _step(p0, r0, image_indices=(0, 1)),
        _step(p1, r1, image_indices=(0, 1, 2)),  # extends (0,1)
    ]
    segments = _segment_steps(per_step)
    assert segments == [[0, 1]]


def test_nonmonotonic_goal_image_prefix_packs():
    """Goal-image prepends can make image_indices non-monotonic.

    What matters for binding is prompt image-pad order, not numeric sort order:
    a goal image appended at physical index 1 can be shown before page image 0,
    yielding (1, 0) -> (1, 0, 2) -> (1, 0, 2, 3).
    """
    p0, r0 = [1, 2], [10]
    p1, r1 = p0 + r0 + [3], [11]
    p2, r2 = p1 + r1 + [4], [12]
    per_step = [
        _step(p0, r0, image_indices=(1, 0)),
        _step(p1, r1, image_indices=(1, 0, 2)),
        _step(p2, r2, image_indices=(1, 0, 2, 3)),
    ]
    assert _segment_steps(per_step) == [[0, 1, 2]]


def test_goal_image_window_drop_breaks_image_prefix():
    """Keeping the goal image while dropping an earlier page image must not pack."""
    p0, r0 = [1, 2], [10]
    p1, r1 = p0 + r0 + [3], [11]
    per_step = [
        _step(p0, r0, image_indices=(1, 0)),
        _step(p1, r1, image_indices=(1, 2)),
    ]
    assert _segment_steps(per_step) == [[0], [1]]


def test_duplicate_image_indices_prefix_pack():
    """Repeated placeholders are positional and can still be prefix-extensional."""
    p0, r0 = [1, 2], [10]
    p1, r1 = p0 + r0 + [3], [11]
    per_step = [
        _step(p0, r0, image_indices=(0, 0)),
        _step(p1, r1, image_indices=(0, 0, 1)),
    ]
    assert _segment_steps(per_step) == [[0, 1]]


def test_image_indices_identical_packs_when_no_new_images():
    """Step_1 keeps step_0's image set verbatim (no new images) — packs."""
    p0, r0 = [1, 2], [10]
    p1 = p0 + r0 + [3]
    r1 = [11]
    per_step = [
        _step(p0, r0, image_indices=(5,)),
        _step(p1, r1, image_indices=(5,)),  # exactly equal → trivially a prefix
    ]
    segments = _segment_steps(per_step)
    assert segments == [[0, 1]]


def test_image_indices_subset_in_middle_breaks_segmentation():
    """A protocol that drops the first image while keeping the second
    breaks the image-index prefix even if text tokens align — must
    fall back to length-1 segments at the boundary."""
    p0, r0 = [1, 2, 3], [10]
    # Token-extensional: p1 extends p0+r0
    p1 = p0 + r0 + [4]
    r1 = [11]
    per_step = [
        _step(p0, r0, image_indices=(0, 1)),
        _step(p1, r1, image_indices=(1,)),  # dropped image 0 → not a prefix of (0,1)
    ]
    segments = _segment_steps(per_step)
    assert segments == [[0], [1]]


def test_image_indices_empty_throughout_packs_under_text_extension():
    """Trajectories without images (text-only) — image_indices=() for
    every step. Empty tuple is trivially a prefix of empty tuple, so
    packing decisions reduce to the token check (back-compat)."""
    p0, r0 = [1, 2, 3], [10, 11]
    p1 = p0 + r0 + [99]
    r1 = [12]
    per_step = [
        _step(p0, r0, image_indices=()),
        _step(p1, r1, image_indices=()),
    ]
    segments = _segment_steps(per_step)
    assert segments == [[0, 1]]


def test_trajectory_with_initial_extensional_then_break_then_extensional():
    """Three segments: [0,1] pack, [2] alone (prefix break), [3,4] pack."""
    p0, r0 = [1, 2], [10]
    p1, r1 = p0 + r0 + [3], [11]
    p2, r2 = [99], [20]               # break — not a prefix of step 1's tail
    p3, r3 = [99, 20, 4], [21]
    p4, r4 = p3 + r3 + [5], [22]

    per_step = [_step(p0, r0), _step(p1, r1), _step(p2, r2), _step(p3, r3), _step(p4, r4)]
    segments = _segment_steps(per_step)

    assert segments == [[0, 1], [2, 3, 4]]
