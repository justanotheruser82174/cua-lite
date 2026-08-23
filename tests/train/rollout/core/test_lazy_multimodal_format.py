"""Lazy multimodal format unit tests.

Verifies that ``build_segment_samples`` correctly produces the lazy
``Sample.multimodal_lazy_payloads`` shape under the opt-in
``CUA_LITE_MULTIMODAL_LAZY_EXPAND=1`` mode and leaves the default legacy
``Sample.multimodal_train_inputs`` path untouched under ``=0`` or unset.

After the bytes-only simplification, there are exactly two modes:

  * ``LAZY_EXPAND=1`` (opt-in) — each image dict carries
    ``{"image_data": uint8_tensor}`` (PNG bytes).
  * ``LAZY_EXPAND=0`` or unset — legacy pre-computed pixel_values per Sample.

Test surface:

  - Shape: lazy mode emits ``{images, indices}`` and clears
    ``multimodal_train_inputs``; legacy mode untouched.
  - Identity: all Samples emitted from one trajectory share the SAME
    Python dict reference for ``images`` (cloudpickle dedup hinge).
  - Index correctness: each Sample's ``indices`` tuple matches the
    segment-tail step's ``image_indices``.
  - PNG fidelity: decoded uint8 bytes round-trip to byte-identical PIL.

Slime-required (uses ``slime.utils.types.Sample``).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("slime.utils", reason="slime not installed")

from slime.utils.types import Sample

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.samples import STATUS_COMPLETED, LiteRLSample, LiteRLStep
from lite.train.rollout.core.segmenter import build_segment_samples

# ---------------------------------------------------------------------------
# Fixtures: minimal processor + state mocks
# ---------------------------------------------------------------------------

class _MockImageProcessor:
    """Mock for ``processor.image_processor(images=[img])``."""

    def __call__(self, images=None, return_tensors=None):
        n = len(images) if images else 0
        fingerprints = torch.tensor(
            [[float(id(img) % 997)] for img in (images or [])],
            dtype=torch.float32,
        )
        return {
            "pixel_values": torch.cat([fingerprints] * 4, dim=1) if n else torch.empty(0, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * n, dtype=torch.long),
        }


class _MockProcessor:
    """Mock for the full ``processor(text=..., images=...)`` call —
    cua-lite uses this to compute prompt_ids whose length depends on
    image_grid_thw vision-token expansion."""

    image_token = "<|image_pad|>"

    def __init__(self):
        self.image_processor = _MockImageProcessor()

    def __call__(self, text=None, images=None, text_kwargs=None, **kwargs):
        n = len(images) if images else 0
        prompt_ids = list(range(5)) + [42] * (n * 4)
        return {
            "input_ids": prompt_ids,
            "pixel_values": torch.randn(n, 1, 4, dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * n, dtype=torch.long),
        }


class _MockTokenizer:
    """Used only for text-only (no images) paths — irrelevant here."""

    def encode(self, text, add_special_tokens=False):
        return [99] * len(text.split())


def _state(hook_path: str | None = "lite.train.utils.multimodal_expand.expand"):
    """State stub: ``build_segment_samples`` reads ``aborted`` and
    ``args.multimodal_lazy_expand_fn_path``. Set ``hook_path=None`` to
    exercise the fail-fast assert."""
    return SimpleNamespace(
        args=SimpleNamespace(multimodal_lazy_expand_fn_path=hook_path),
        aborted=False,
    )


def _prompt_with_image_pads(text: str, image_indices: tuple[int, ...]) -> str:
    pads = " ".join("<|image_pad|>" for _ in image_indices)
    return f"{text} {pads}".strip()


def _build_trajectory(n_steps: int, window: int) -> LiteRLSample:
    """Construct a synthetic trajectory with real PIL images (required
    since the producer always PNG-encodes in lazy mode). Each image has
    distinct dimensions so a "decoded recovery matches" test can verify
    per-step identity is preserved."""
    from PIL import Image
    images = [
        Image.new("RGB", (10 + i * 5, 20 + i * 5), color=(50 + 10 * i, 100, 200))
        for i in range(n_steps)
    ]
    steps = []
    for k in range(n_steps):
        lo = max(0, k - window + 1)
        idx = tuple(range(lo, k + 1))
        steps.append(LiteRLStep(
            prompt=_prompt_with_image_pads(f"prompt step {k}", idx),
            image_indices=idx,
            response=f"r{k}",
            response_tokens=[k, k + 1],
            response_log_probs=[-0.1, -0.2],
            reward=0.0,
            status=STATUS_COMPLETED,
        ))
    return LiteRLSample(
        processed_images=images,
        steps=steps,
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )


def _original_sample():
    return Sample(
        group_index=0, index=0, prompt="x",
        metadata={"env_key": "lite.demo@smoke", "split": "train"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lazy_sample_shape(monkeypatch):
    """LAZY=1 (opt-in): every emitted Sample has multimodal_train_inputs=None
    and a well-formed multimodal_lazy_payloads dict whose per-image entries
    carry ``image_data`` (uint8 PNG bytes), not pixel_values."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")  # force length-1 segments

    rl_sample = _build_trajectory(n_steps=5, window=3)
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=_MockProcessor(),
        tokenizer=_MockTokenizer(),
        state=_state(),
    )

    assert len(samples) == 5, "length-1 segments → one Sample per step"
    for s in samples:
        assert s.multimodal_train_inputs is None, (
            f"lazy mode must clear multimodal_train_inputs, got {s.multimodal_train_inputs}"
        )
        assert s.multimodal_lazy_payloads is not None
        assert "images" in s.multimodal_lazy_payloads
        assert "indices" in s.multimodal_lazy_payloads
        assert isinstance(s.multimodal_lazy_payloads["indices"], tuple)

        for img_idx, img_dict in s.multimodal_lazy_payloads["images"].items():
            assert "image_data" in img_dict, f"image {img_idx}: missing image_data"
            assert "pixel_values" not in img_dict, (
                f"image {img_idx}: lazy mode must not carry pre-computed pixel_values"
            )
            assert img_dict["image_data"].dtype == torch.uint8
            assert img_dict["image_data"].ndim == 1, "image_data must be a 1-D byte buffer"


def test_shared_reference_per_trajectory(monkeypatch):
    """Cloudpickle dedup hinges on ALL Samples from one trajectory pointing
    at the SAME Python dict object for ``images``. Verify identity (not
    just equality) holds."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")

    rl_sample = _build_trajectory(n_steps=7, window=3)
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=_MockProcessor(),
        tokenizer=_MockTokenizer(),
        state=_state(),
    )

    first_images = samples[0].multimodal_lazy_payloads["images"]
    for k, s in enumerate(samples[1:], start=1):
        assert s.multimodal_lazy_payloads["images"] is first_images, (
            f"sample {k}: images dict identity broken — cloudpickle dedup will not fire"
        )

    assert set(first_images.keys()) == set(range(7))


def test_indices_match_segment_view(monkeypatch):
    """Each Sample's ``indices`` tuple equals the step's ``image_indices``."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")

    rl_sample = _build_trajectory(n_steps=5, window=3)
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=_MockProcessor(),
        tokenizer=_MockTokenizer(),
        state=_state(),
    )

    for k, s in enumerate(samples):
        expected = rl_sample.steps[k].image_indices
        assert s.multimodal_lazy_payloads["indices"] == tuple(expected), (
            f"step {k}: indices mismatch — got {s.multimodal_lazy_payloads['indices']}, "
            f"expected {tuple(expected)}"
        )


def test_png_encoding_magic_and_roundtrip(monkeypatch):
    """The producer always PNG-encodes. Verify (a) magic bytes are PNG,
    (b) decoded PIL matches the original byte-for-byte (lossless)."""
    from io import BytesIO

    from PIL import Image

    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")

    rl_sample = _build_trajectory(n_steps=3, window=1)
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=_MockProcessor(),
        tokenizer=_MockTokenizer(),
        state=_state(),
    )

    images_dict = samples[0].multimodal_lazy_payloads["images"]
    for img_idx, orig_pil in enumerate(rl_sample.processed_images):
        encoded = images_dict[img_idx]["image_data"].numpy().tobytes()
        assert encoded.startswith(b"\x89PNG"), (
            f"image {img_idx}: expected PNG magic, got {encoded[:4]!r}"
        )
        recovered = Image.open(BytesIO(encoded))
        recovered.load()
        assert recovered.size == orig_pil.size, (
            f"image {img_idx}: size mismatch after PNG round-trip"
        )
        assert recovered.tobytes() == orig_pil.tobytes(), (
            f"image {img_idx}: PNG round-trip changed pixel buffer"
        )


def test_legacy_when_lazy_disabled(monkeypatch):
    """``LAZY_EXPAND=0``: lazy field None, legacy field carries the standard
    pixel_values dict (the pre-default-on behavior)."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")

    rl_sample = _build_trajectory(n_steps=3, window=2)
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=_MockProcessor(),
        tokenizer=_MockTokenizer(),
        state=_state(),  # legacy mode → hook not required
    )

    for s in samples:
        assert s.multimodal_lazy_payloads is None, "legacy mode must not emit lazy payload"
        assert s.multimodal_train_inputs is not None, "legacy mode must emit train_inputs"
        assert "pixel_values" in s.multimodal_train_inputs


def test_fail_fast_missing_slime_hook(monkeypatch):
    """LAZY=1 but state.args.multimodal_lazy_expand_fn_path is None →
    AssertionError surfaces at build_segment_samples entry, before any
    rollout work is wasted (materialize_lazy_payloads would otherwise crash
    at the start of the trainer's first iter)."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")

    rl_sample = _build_trajectory(n_steps=2, window=1)
    with pytest.raises(RuntimeError, match="multimodal-lazy-expand-fn-path"):
        build_segment_samples(
            rl_sample=rl_sample,
            original_sample=_original_sample(),
            processor=_MockProcessor(),
            tokenizer=_MockTokenizer(),
            state=_state(hook_path=None),
        )


def test_text_only_emits_no_payloads(monkeypatch):
    """Text-only rollouts with ``processor=None`` skip the lazy path
    entirely. Both Sample fields stay None."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")

    images: list = []
    steps = [
        LiteRLStep(
            prompt="text only step",
            image_indices=(),
            response="r",
            response_tokens=[1, 2, 3],
            response_log_probs=[-0.1, -0.2, -0.3],
            reward=0.0,
            status=STATUS_COMPLETED,
        )
        for _ in range(2)
    ]
    rl_sample = LiteRLSample(
        processed_images=images,
        steps=steps,
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )

    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=None,                                  # ← gate must fire
        tokenizer=_MockTokenizer(),
        state=_state(),
    )

    assert len(samples) == 2
    for s in samples:
        assert s.multimodal_train_inputs is None
        assert s.multimodal_lazy_payloads is None, (
            "lazy mode + processor=None must not emit lazy_payloads"
        )


def test_lazy_indices_match_segment_tail_under_default_radix(monkeypatch):
    """Regression for the segment-tail invariant under DEFAULT radix.

    ``test_indices_match_segment_view`` covers the same invariant but
    forces length-1 segments via ``CUA_LITE_DISABLE_RADIX=1``. This test
    runs WITHOUT that override — exercising the radix interaction with
    the lazy code path. The invariant: for every emitted Sample,
    ``multimodal_lazy_payloads["indices"]`` must equal the SEGMENT-TAIL
    step's ``image_indices`` (recovered from ``metadata["turn_range"][1]``)."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)

    from PIL import Image
    images = [
        Image.new("RGB", (10 + i * 5, 20 + i * 5), color=(50 + 10 * i, 100, 200))
        for i in range(4)
    ]
    steps = []
    for k in range(4):
        idx = tuple(range(k + 1))   # monotonic prefix-extension
        steps.append(LiteRLStep(
            prompt=_prompt_with_image_pads("P" * (k + 1), idx),
            image_indices=idx,
            response="r" + str(k),
            response_tokens=[100 + k, 200 + k],
            response_log_probs=[-0.1, -0.2],
            reward=0.0,
            status=STATUS_COMPLETED,
        ))
    rl_sample = LiteRLSample(
        processed_images=images,
        steps=steps,
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )

    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=_MockProcessor(),
        tokenizer=_MockTokenizer(),
        state=_state(),
    )

    for s in samples:
        first, last = s.metadata["turn_range"]
        expected_tail_indices = tuple(rl_sample.steps[last].image_indices)
        assert s.multimodal_lazy_payloads is not None
        assert s.multimodal_lazy_payloads["indices"] == expected_tail_indices, (
            f"segment ({first}, {last}): got indices "
            f"{s.multimodal_lazy_payloads['indices']}, expected tail-step "
            f"indices {expected_tail_indices}"
        )


@pytest.mark.parametrize("lazy_expand", [False, True])
def test_sparse_processed_images_ignore_unreferenced_none(monkeypatch, lazy_expand):
    """Segmenter only dereferences sparse slots named by ``image_indices``."""
    from PIL import Image

    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1" if lazy_expand else "0")

    images = [
        Image.new("RGB", (12, 12), color=(10, 20, 30)),
        None,
        Image.new("RGB", (14, 14), color=(40, 50, 60)),
    ]
    step = LiteRLStep(
        prompt=_prompt_with_image_pads("sparse prompt", (0, 2)),
        image_indices=(0, 2),
        response="response",
        response_tokens=[1, 2],
        response_log_probs=[-0.1, -0.2],
        reward=0.0,
        status=STATUS_COMPLETED,
    )
    rl_sample = LiteRLSample(
        processed_images=images,
        steps=[step],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )

    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=_original_sample(),
        processor=_MockProcessor(),
        tokenizer=_MockTokenizer(),
        state=_state(),
    )

    assert len(samples) == 1
    sample = samples[0]
    if lazy_expand:
        assert sample.multimodal_train_inputs is None
        assert sample.multimodal_lazy_payloads["indices"] == (0, 2)
        assert set(sample.multimodal_lazy_payloads["images"]) == {0, 2}
    else:
        assert sample.multimodal_lazy_payloads is None
        assert sample.multimodal_train_inputs["pixel_values"].shape[0] == 2
