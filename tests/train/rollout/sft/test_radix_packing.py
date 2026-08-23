"""SFT pipeline-unification.

Verifies that ``rollout_sft.generate_rollout`` runs each parquet v2 row
through the **same** ``build_segment_samples`` radix path that
``rollout_grpo`` uses. Acceptance:

  - Under a prefix-extensional render (qwen3_5 / qwen3_vl with
    ``history_n=∞`` / ``full_history_size=∞``), a T-step trajectory
    packs into ONE ``slime.Sample`` with multi-span ``loss_mask`` —
    matching the unification invariant.
  - Under a non-extensional render (Markov / sliding-window), the same
    pipeline falls back to T length-1 Samples — byte-identical to the
    pre-packing SFT shape.

Slime-required (uses Sample type + actual rollout_sft module).

Note: this test mocks the ``args`` / ``data_buffer`` interface that
slime supplies at runtime — there's no slime trainer running in pytest.
"""

from __future__ import annotations

import io
from argparse import Namespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from PIL import Image
from slime.utils.types import Sample

import lite.train.rollout.sft as rollout_sft_module
from lite.train.rollout.sft import generate_rollout


def _png_bytes(size=(64, 64), color=(128, 128, 128)) -> bytes:
    """Build a tiny valid PNG so ``decode_hf_images`` can decode it."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_extensional_messages(T: int, n_images: int) -> list[list[dict]]:
    """Build T AgentSteps where each step's prompt is a token-prefix of
    the next — i.e. an extensional render that should pack to length-T."""
    steps_messages: list[list[dict]] = []
    history_text = "You are a UI agent. Task: do something useful."
    for k in range(T):
        # Append every prior turn as user/assistant pairs.
        msgs: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": history_text}]},
        ]
        for j in range(k):
            # Prior user (with image at idx j if j < n_images)
            user_content: list[dict] = []
            if j < n_images:
                user_content.append({"type": "image", "index": j})
            user_content.append({"type": "text", "text": f"observation at step {j+1}"})
            msgs.append({"role": "user", "content": user_content})
            msgs.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"action at step {j+1}"}],
            })
        # Current user turn (with image at idx k if available)
        cur_user: list[dict] = []
        if k < n_images:
            cur_user.append({"type": "image", "index": k})
        cur_user.append({"type": "text", "text": f"observation at step {k+1}"})
        msgs.append({"role": "user", "content": cur_user})
        # Target assistant turn
        msgs.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"action at step {k+1}"}],
        })
        steps_messages.append(msgs)
    return steps_messages


def _build_markov_messages(T: int, n_images: int) -> list[list[dict]]:
    """Build T AgentSteps where each step is independent (Markov / no
    history). The token-level prefix check should fail on every adjacent
    pair → segmenter falls back to T length-1 segments."""
    steps_messages: list[list[dict]] = []
    for k in range(T):
        # Use a different system prompt at each step so the prefix breaks.
        msgs = [
            {"role": "system",
             "content": [{"type": "text", "text": f"Markov step {k+1}: do thing {k+1}."}]},
            {"role": "user", "content": [
                {"type": "image", "index": min(k, n_images - 1)},
                {"type": "text", "text": f"unique markov observation {k+1}"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": f"action {k+1}"}]},
        ]
        steps_messages.append(msgs)
    return steps_messages


def _serialized_steps(steps_messages, processor) -> list[dict]:
    """Tokenize each AgentStep into a serialized LiteRLStep struct — exactly
    what ``export_sft`` writes into the parquet ``steps`` column (the new schema
    ``rollout_sft`` consumes)."""
    from lite.train.export.sft_tokenize import (
        agent_step_to_rl_step,
        serialize_rl_step,
    )
    out: list[dict] = []
    for step in steps_messages:
        rl_step = agent_step_to_rl_step(step, processor)
        assert rl_step is not None
        out.append(serialize_rl_step(rl_step))
    return out


def _make_sample_template(steps, processed_images_bytes) -> Sample:
    """Slime feeds rows as Sample with prompt=steps (serialized LiteRLStep
    structs), metadata=processed_images."""
    return Sample(
        group_index=0,
        index=0,
        prompt=steps,
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.PENDING,
        metadata=processed_images_bytes,  # decode_hf_images takes the metadata directly
    )


@pytest.fixture(scope="module")
def real_processor():
    """Load a small real Qwen3-VL processor to exercise apply_chat_template +
    text tokenizer end-to-end."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True
    )


def _patch_processor(monkeypatch, processor):
    """Inject the real processor as the module-level PROCESSOR so
    generate_rollout doesn't try to download a checkpoint via load_processor."""
    monkeypatch.setattr(rollout_sft_module, "PROCESSOR", processor)


def _build_args(*, hook_set: bool = True) -> Namespace:
    """Build the slime-shaped args Namespace. Pass ``hook_set=False`` to
    exercise the fail-fast assert (lazy on but slime arg missing)."""
    return Namespace(
        rollout_batch_size=1,
        rollout_global_dataset=True,
        hf_checkpoint="Qwen/Qwen3-VL-2B-Instruct",
        multimodal_lazy_expand_fn_path=(
            "lite.train.utils.multimodal_expand.expand" if hook_set else None
        ),
    )


def _build_buffer_with_one_row(sample: Sample) -> MagicMock:
    buf = MagicMock()
    # Slime calls get_samples(batch_size) → list of (Sample,) tuples
    buf.get_samples.return_value = [(sample,)]
    return buf


def test_extensional_trajectory_packs_into_one_segment(real_processor, monkeypatch):
    """Prefix-extensional T=5 trajectory → one length-5 packed Sample."""
    T = 5
    n_images = T  # one new image per step
    steps_messages = _build_extensional_messages(T, n_images)
    pngs = [_png_bytes() for _ in range(n_images)]
    steps = _serialized_steps(steps_messages, real_processor)
    sample = _make_sample_template(steps, pngs)

    _patch_processor(monkeypatch, real_processor)
    args = _build_args()
    buf = _build_buffer_with_one_row(sample)
    out = generate_rollout(args, rollout_id=0, data_buffer=buf, evaluation=False)

    assert len(out) == 1, (
        f"prefix-extensional T={T} must pack into 1 segment; got {len(out)} segments"
    )
    s = out[0]
    # n_turns metadata should equal T (original turn count).
    assert s.metadata["n_turns"] == T
    assert s.metadata["turn_range"] == (0, T - 1)
    # Multi-span loss_mask: should have T contiguous 1-spans separated by 0-spans.
    n_one_runs = 0
    in_run = False
    for v in s.loss_mask:
        if v == 1 and not in_run:
            n_one_runs += 1
            in_run = True
        elif v == 0:
            in_run = False
    assert n_one_runs == T, (
        f"multi-span loss_mask must have T={T} 1-runs, got {n_one_runs}"
    )


def test_markov_trajectory_falls_back_to_length1_segments(real_processor, monkeypatch):
    """Non-extensional Markov T=4 → 4 length-1 Samples (byte-identical
    to pre-Beat-2 SFT shape)."""
    T = 4
    n_images = T
    steps_messages = _build_markov_messages(T, n_images)
    pngs = [_png_bytes() for _ in range(n_images)]
    steps = _serialized_steps(steps_messages, real_processor)
    sample = _make_sample_template(steps, pngs)

    _patch_processor(monkeypatch, real_processor)
    args = _build_args()
    buf = _build_buffer_with_one_row(sample)
    out = generate_rollout(args, rollout_id=0, data_buffer=buf, evaluation=False)

    assert len(out) == T, (
        f"Markov T={T} must fall back to {T} length-1 segments; got {len(out)}"
    )
    # Every Sample's n_turns must report ORIGINAL T, not 1.
    for s in out:
        assert s.metadata["n_turns"] == T, (
            f"every length-1 Sample must report n_turns={T} (original turn count); "
            f"got {s.metadata['n_turns']}"
        )
    # Each Sample's loss_mask should have exactly one contiguous 1-run.
    for i, s in enumerate(out):
        n_one_runs = 0
        in_run = False
        for v in s.loss_mask:
            if v == 1 and not in_run:
                n_one_runs += 1
                in_run = True
            elif v == 0:
                in_run = False
        assert n_one_runs == 1, (
            f"length-1 Sample {i} must have a single 1-run; got {n_one_runs}"
        )


def test_sft_lazy_mode_produces_lazy_payloads(real_processor, monkeypatch):
    """Under ``CUA_LITE_MULTIMODAL_LAZY_EXPAND=1``, every Sample produced by
    SFT carries ``multimodal_lazy_payloads`` (with shared ``images`` ref)
    instead of ``multimodal_train_inputs``. Verifies the lazy path is
    wired through the SFT rollout, not just RL."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")  # force length-1 segments

    T = 3
    steps_messages = _build_markov_messages(T, n_images=T)
    pngs = [_png_bytes() for _ in range(T)]
    steps = _serialized_steps(steps_messages, real_processor)
    sample = _make_sample_template(steps, pngs)

    _patch_processor(monkeypatch, real_processor)
    args = _build_args()
    buf = _build_buffer_with_one_row(sample)
    out = generate_rollout(args, rollout_id=0, data_buffer=buf, evaluation=False)

    assert len(out) == T
    for s in out:
        assert s.multimodal_train_inputs is None, (
            "lazy mode must clear multimodal_train_inputs in SFT path too"
        )
        assert s.multimodal_lazy_payloads is not None
        assert "images" in s.multimodal_lazy_payloads
        assert "indices" in s.multimodal_lazy_payloads

    # Shared identity across all T Samples — the SFT path constructs ONE
    # synthetic LiteRLSample per row, so all emitted Samples share the
    # trajectory_images dict via Python id.
    first_images = out[0].multimodal_lazy_payloads["images"]
    for k, s in enumerate(out[1:], start=1):
        assert id(s.multimodal_lazy_payloads["images"]) == id(first_images), (
            f"SFT lazy mode: Sample {k} broke shared-reference identity"
        )


def test_sft_lazy_mode_fails_fast_without_slime_arg(real_processor, monkeypatch):
    """SFT under LAZY=1 but with ``multimodal_lazy_expand_fn_path=None`` →
    fail-fast AssertionError. Same convention as the RL path."""
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")

    steps_messages = _build_markov_messages(T=2, n_images=2)
    pngs = [_png_bytes() for _ in range(2)]
    steps = _serialized_steps(steps_messages, real_processor)
    sample = _make_sample_template(steps, pngs)

    _patch_processor(monkeypatch, real_processor)
    args = _build_args(hook_set=False)
    buf = _build_buffer_with_one_row(sample)

    with pytest.raises(RuntimeError, match="multimodal-lazy-expand-fn-path"):
        generate_rollout(args, rollout_id=0, data_buffer=buf, evaluation=False)


def test_segment_count_eq_original_step_count_when_unrolled(real_processor, monkeypatch):
    """Smoke: under any render, sum of (turns covered by each segment) == T."""
    T = 6
    n_images = T
    steps_messages = _build_markov_messages(T, n_images)
    pngs = [_png_bytes() for _ in range(n_images)]
    steps = _serialized_steps(steps_messages, real_processor)
    sample = _make_sample_template(steps, pngs)

    _patch_processor(monkeypatch, real_processor)
    args = _build_args()
    buf = _build_buffer_with_one_row(sample)
    out = generate_rollout(args, rollout_id=0, data_buffer=buf, evaluation=False)

    covered = 0
    for s in out:
        first, last = s.metadata["turn_range"]
        covered += (last - first + 1)
    assert covered == T, (
        f"sum of turn_range coverage must equal T={T}; got {covered} from {len(out)} segments"
    )
