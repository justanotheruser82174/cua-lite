"""Qwen3.5 adapter + Qwen3_5HistoryProtocol radix packing.

Exercises the full qwen3_5 adapter pipeline:
  LiteSample → Qwen3_5DesktopUseAdapter.unroll → AgentSample
  → per-step apply_chat_template + tokenize → _segment_steps

Scenarios:

1. **Production trajectory shape packs**: T=5 of ``role:"tool"`` observations
   answering structured ``tool_calls`` — what the preprocessors emit today.
   Every step's prompt prefix-extends the previous, so the segmenter packs the
   whole trajectory into ONE segment. This is the shape that matters: packing is
   the whole point of radix segmentation, and a regression to length-1 segments
   here is a silent throughput cliff, not a correctness failure, so nothing else
   would catch it.

2. **Free-text assistant turns are NOT extensional**: the same T=5 trajectory
   written the legacy way (``role:"user"`` observations, the tool call as free
   TEXT). Qwen3.5's chat template inserts a ``<think>\\n\\n</think>`` scaffold
   into the assistant *generation prompt*, but the same completed assistant turn
   renders without it as history, so the token-prefix invariant breaks and the
   segmenter must fall back to length-1 segments rather than pack wrongly.
   Scenario 1 is unaffected because a structured tool-call turn renders
   identically in both positions.

3. **Fold-mid-trajectory (non-extensional after fold)**: T=12 with
   ``image_max=5, fold_size=5`` → at step 6+ the oldest images get
   replaced by ``collapse_text``. The text-prefix invariant breaks at
   the fold boundary; the segmenter should close the segment there
   and start a new one.

Runs WITHOUT slime: ``segmenter.py`` needs only ``Sample.Status`` and
``build_processor_kwargs`` at import, so a tiny stub surface stands in when slime
is absent (the package import is preferred whenever it works). Previously this
module ``importorskip``-ed ``sglang_router`` and so silently skipped everywhere
off-container -- which is exactly how scenario 1's assertion came to be inverted
without anything going red.

Run: uv run pytest tests/agents/models/qwen3_5/test_qwen3_5_packing.py -q
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

pytest.importorskip("transformers", reason="transformers not installed")

from PIL import Image
from transformers import AutoTokenizer

from lite.agents.models.qwen3_5.adapter import (
    Qwen3_5DesktopUseAdapter,
)
from lite.agents.models.qwen3_5.protocol import Qwen3_5HistoryProtocol
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools.action_space import LiteDesktopActionSet, merge_adjacent_lite_action_batches
from lite.data.utils.messages import finalize_use_messages, structural_final_message
from lite.utils.path import project_root


def _load_segment_steps():
    """``_segment_steps``, via the real package when slime is installed.

    Off-container, load ``segmenter.py`` by path behind a minimal slime stub: going
    through ``lite.train.rollout.core.__init__`` would drag in the sglang rollout
    engine, which is the only reason this pure, CPU-only function was unreachable
    (and this module skipped) without a full slime install.
    """
    try:
        from lite.train.rollout.core.segmenter import _segment_steps
        return _segment_steps
    except ImportError:
        pass

    class _Sample:
        class Status:
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            ABORTED = "aborted"
            FAILED = "failed"

    stub_names = (
        "slime",
        "slime.utils",
        "slime.utils.processing_utils",
        "slime.utils.types",
    )
    original_modules = {name: sys.modules.get(name) for name in stub_names}

    slime_mod = types.ModuleType("slime")
    slime_mod.__path__ = []
    utils_mod = types.ModuleType("slime.utils")
    utils_mod.__path__ = []
    processing_mod = types.ModuleType("slime.utils.processing_utils")
    processing_mod.build_processor_kwargs = lambda payload: {}
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = _Sample
    slime_mod.utils = utils_mod
    utils_mod.processing_utils = processing_mod
    utils_mod.types = types_mod

    try:
        sys.modules.update({
            "slime": slime_mod,
            "slime.utils": utils_mod,
            "slime.utils.processing_utils": processing_mod,
            "slime.utils.types": types_mod,
        })
        path = project_root() / "lite/train/rollout/core/segmenter.py"
        spec = importlib.util.spec_from_file_location("cua_lite_test_segmenter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module._segment_steps


_segment_steps = _load_segment_steps()


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (64, 64), (200, 200, 200))


def _build_lite_sample(T: int, n_images: int) -> LiteSample:
    """Build a LiteSample with T user/assistant pairs, plus a final user
    turn so ``count_sample_turns`` returns T (predicting turn T)."""
    # Note: the qwen3_5 adapter prepends its own system prompt (with tool
    # descriptions). Don't include one here or the chat template fails with
    # "System message must be at the beginning" (two system messages).
    images: list[Image.Image] = [_tiny_image() for _ in range(n_images)]
    messages: list[dict] = []
    for k in range(T - 1):
        # User message k (with image at index k if available)
        cur_user_content: list[dict] = []
        if k < n_images:
            cur_user_content.append({"type": "image", "index": k})
        cur_user_content.append({
            "type": "text",
            "text": f"Task: do step {k+1}. Observation: screen looks fresh."
        })
        messages.append({"role": "user", "content": cur_user_content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": (
                f"Action: do_thing\n"
                f"<tool_call>\n"
                f'{{"name": "computer_use", "arguments": '
                f'{{"action": "click", "coordinate": [100, {200 + k}]}}}}\n'
                f"</tool_call>"
            )}],
        })
    # Final user turn (the one the model is predicting at turn T).
    final_user_content: list[dict] = []
    if (T - 1) < n_images:
        final_user_content.append({"type": "image", "index": T - 1})
    final_user_content.append({
        "type": "text",
        "text": f"Task: do step {T}. Observation: continuing."
    })
    messages.append({"role": "user", "content": final_user_content})
    # And its target assistant (so unroll has a complete trajectory of T pairs).
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": (
            f"Action: do_thing\n"
            f"<tool_call>\n"
            f'{{"name": "computer_use", "arguments": '
            f'{{"action": "click", "coordinate": [100, {200 + T - 1}]}}}}\n'
            f"</tool_call>"
        )}],
    })

    return LiteSample(
        images=images,
        messages=messages,
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
    )


def _build_production_lite_sample(T: int) -> LiteSample:
    """The trajectory shape the preprocessors emit TODAY.

    Post-action screenshots are ``role:"tool"`` messages carrying the
    ``tool_call_id`` of the batched ``computer`` call they answer; that assistant
    call carries the matching ``id``. ``finalize_use_messages`` rewrites the
    screenshot-only ``user`` turns, actions are structured ``tool_calls`` rather
    than free text, and the episode ends on the content-only ``Done.`` final.
    Contrast :func:`_build_lite_sample`, which is the legacy ``role:"user"``
    shape.
    """
    messages: list[dict] = [{"role": "user", "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Task: do the thing."},
    ]}]
    for k in range(T):
        if k > 0:
            messages.append({"role": "user", "content": [{"type": "image", "index": k}]})
        messages.append({
            "role": "assistant",
            "tool_calls": merge_adjacent_lite_action_batches(
                [LiteDesktopActionSet.click(coordinate=[100, 200 + k])]
            ),
            "content": [{"type": "action_description", "text": f"do step {k + 1}"}],
        })
    messages.append(structural_final_message())
    return LiteSample(
        images=[_tiny_image() for _ in range(T)],
        messages=finalize_use_messages(messages),
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
    )


def _per_step_tokens_from_agent_sample(agent_sample, processor) -> list[dict]:
    """For each AgentStep, run apply_chat_template (tokenize=True) on the
    full step messages → derive prompt_tokens (everything up to but not
    including the last assistant) and response_tokens (the last assistant
    target). Returns the list `_segment_steps` expects."""
    out: list[dict] = []
    for step_idx, step_messages in enumerate(agent_sample.steps):
        if not step_messages or step_messages[-1].get("role") != "assistant":
            continue
        prompt_msgs = step_messages[:-1]
        full_msgs = step_messages
        prompt_text = processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        full_text = processor.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
        )
        # Tokenize text — pure text tokenizer (Qwen3.5 tokenizer is text-only;
        # the segmenter's prefix check only cares about *text* tokens.)
        prompt_tokens = processor.encode(prompt_text, add_special_tokens=False)
        full_tokens = processor.encode(full_text, add_special_tokens=False)
        if not full_tokens or len(full_tokens) <= len(prompt_tokens):
            response_tokens: list[int] = []
        else:
            response_tokens = full_tokens[len(prompt_tokens):]
        # Derive image_indices from the rendered messages.
        image_indices: list[int] = []
        seen: set[int] = set()
        for m in step_messages:
            for part in m.get("content") or []:
                if part.get("type") == "image" and "index" in part:
                    idx = int(part["index"])
                    if idx not in seen:
                        seen.add(idx)
                        image_indices.append(idx)

        class _MockStep:
            pass
        s = _MockStep()
        s.image_indices = tuple(image_indices)
        out.append({
            "prompt_tokens": list(prompt_tokens),
            "response_tokens": list(response_tokens),
            "step": s,
        })
    return out


@pytest.fixture(scope="module")
def qwen3_5_tokenizer():
    """Use Qwen2.5-tokenizer as a stand-in (Qwen3.5's tokenizer follows the
    Qwen2 chat template). Cached locally."""
    # Try Qwen3.5 first; fall back to Qwen2.5 if not cached.
    for model_id in ("Qwen/Qwen3.5-2B", "Qwen/Qwen2.5-VL-3B-Instruct",
                     "Qwen/Qwen2.5-1.5B-Instruct"):
        try:
            return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            continue
    pytest.skip("No Qwen tokenizer available locally")


def test_qwen3_5_production_tool_role_trajectory_packs_into_one_segment(qwen3_5_tokenizer):
    """The shape production emits today must PACK — one segment, not one per step.

    ``role:"tool"`` observations answering structured ``tool_calls`` render
    identically as target and as history, so every step's prompt prefix-extends
    the previous one and the whole trajectory packs. Packing is the entire point
    of radix segmentation; a regression to length-1 segments here costs ~6x the
    training compute while still producing *correct* data, so no correctness test
    would ever catch it.

    Regression guard: this assertion was once flipped to ``[[0], [1], ...]`` and
    re-documented as "a model template fact". That is true only of the LEGACY
    free-text shape (see the test below) — which production no longer emits.
    """
    sample = _build_production_lite_sample(T=5)
    roles = [m["role"] for m in sample.messages]
    # T actions are answered by T-1 screenshots; the last is followed by the Done. final.
    assert roles.count("tool") == 4, "fixture must be production role:'tool', not role:'user'"
    assert roles.count("user") == 1, "only the goal turn stays a user message"
    adapter = Qwen3_5DesktopUseAdapter(
        protocol=Qwen3_5HistoryProtocol(history_n=100, image_max=20, fold_size=10),
    )
    agent_sample = adapter.unroll(sample)
    per_step = _per_step_tokens_from_agent_sample(agent_sample, qwen3_5_tokenizer)
    assert len(per_step) == 6, f"5 actions + the Done. final; got {len(per_step)}"

    segments = _segment_steps(per_step)
    assert segments == [[0, 1, 2, 3, 4, 5]], (
        f"production tool-role trajectory must pack into ONE segment; got {segments}"
    )


def test_qwen3_5_free_text_generation_prompt_scaffold_falls_back_to_length1(
    qwen3_5_tokenizer,
):
    """Free-text role:user fixture only: the thinking scaffold defeats packing.

    With the action written as free TEXT (and ``role:"user"`` observations), the
    target assistant turn is rendered behind the generation prompt's
    ``<think>\\n\\n</think>`` scaffold, while the same completed turn renders
    without it as history — so the token-prefix invariant genuinely fails and the
    segmenter must fall back to length-1 segments rather than pack wrongly.

    This is a real model-template fact, kept because it pins the segmenter's
    safe fallback. It is NOT a statement about production, which emits structured
    tool_calls and packs — see the test above.
    """
    sample = _build_lite_sample(T=5, n_images=5)
    assert "tool" not in [m["role"] for m in sample.messages], "role:user fixture by construction"
    adapter = Qwen3_5DesktopUseAdapter(
        protocol=Qwen3_5HistoryProtocol(history_n=100, image_max=20, fold_size=10),
    )
    agent_sample = adapter.unroll(sample)

    assert len(agent_sample.steps) == 5, f"expected 5 steps, got {len(agent_sample.steps)}"
    per_step = _per_step_tokens_from_agent_sample(agent_sample, qwen3_5_tokenizer)
    assert len(per_step) == 5

    segments = _segment_steps(per_step)
    assert segments == [[0], [1], [2], [3], [4]], (
        "qwen3_5 generation-prompt scaffold must prevent unsafe packing; "
        f"got {segments}"
    )


def test_qwen3_5_fold_mid_trajectory_breaks_segment(qwen3_5_tokenizer):
    """T=12 with image_max=5, fold_size=5, history_n=100 — at step 6+ the
    fold logic replaces the oldest 5 images with ``collapse_text``,
    breaking the text-prefix invariant. Segmenter must close the segment
    at the fold boundary."""
    sample = _build_lite_sample(T=12, n_images=12)
    folded_protocol = Qwen3_5HistoryProtocol(
        history_n=100, image_max=5, fold_size=5,
    )
    adapter = Qwen3_5DesktopUseAdapter(protocol=folded_protocol)
    agent_sample = adapter.unroll(sample)
    assert len(agent_sample.steps) == 12

    per_step = _per_step_tokens_from_agent_sample(agent_sample, qwen3_5_tokenizer)
    assert len(per_step) == 12

    segments = _segment_steps(per_step)
    # Fold triggers when total_turns > image_max (=5), so steps 6-12 see folding.
    # The segmenter shouldn't pack ALL 12 into one segment when the fold
    # text changes mid-trajectory. We expect at least one boundary break.
    assert len(segments) >= 2, (
        f"fold-mid-trajectory must break the segment at least once; got {segments}"
    )
    # Total step coverage must equal T.
    total = sum(len(seg) for seg in segments)
    assert total == 12, f"step coverage must equal T=12; got {total}"


def test_qwen3_5_window_slide_breaks_segment(qwen3_5_tokenizer):
    """T=10 with history_n=3 — the rolling window slides past the first
    turn, and the kept-window content changes (older turns get
    summarized into a Step-N text). Token-level prefix breaks; segmenter
    falls back to length-1 segments at the boundary."""
    sample = _build_lite_sample(T=10, n_images=10)
    windowed_protocol = Qwen3_5HistoryProtocol(
        history_n=3, image_max=20, fold_size=10,
    )
    adapter = Qwen3_5DesktopUseAdapter(protocol=windowed_protocol)
    agent_sample = adapter.unroll(sample)
    assert len(agent_sample.steps) == 10

    per_step = _per_step_tokens_from_agent_sample(agent_sample, qwen3_5_tokenizer)
    segments = _segment_steps(per_step)
    # With history_n=3, after step 4 the summary changes each step → no
    # contiguous extensional run can span the slide boundary.
    assert len(segments) >= 2
    total = sum(len(seg) for seg in segments)
    assert total == 10


def test_qwen3_5_t1_single_segment(qwen3_5_tokenizer):
    """T=1 sanity: one step → one length-1 segment."""
    sample = _build_lite_sample(T=1, n_images=1)
    adapter = Qwen3_5DesktopUseAdapter()
    agent_sample = adapter.unroll(sample)
    assert len(agent_sample.steps) == 1

    per_step = _per_step_tokens_from_agent_sample(agent_sample, qwen3_5_tokenizer)
    segments = _segment_steps(per_step)
    assert segments == [[0]]
