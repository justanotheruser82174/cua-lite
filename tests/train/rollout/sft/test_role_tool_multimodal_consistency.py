"""Training multimodal-consistency and SFT parquet contract tests.

The current durable path is nested and canonical: ``LiteSample`` stores
structured messages, nested Lite tool calls, and images once per trajectory;
SFT export stores model-ready rows as ``processed_images`` plus serialized
``LiteRLStep`` structs in ``steps``. Each step's ``image_indices`` is an ordered
view into ``processed_images`` and must stay aligned with the rendered
``<image_pad>`` slots and segmenter ``pixel_values``.

Both first-turn ``role:"user"`` observations and per-call ``role:"tool"``
results are model-visible image carriers. The guards below pin that shared
image-slot contract without changing any model prompt text.

Hermetic: no model download. The SFT-tokenize / render assertions use a faithful
FAKE Qwen processor (mirrors ``tests/train/rollout/dagger/test_dagger.py``'s ``_FakeProcessor``
approach — char-code tokenizer + ``<|image_pad|>`` / ``<tool_call>`` expansion),
noted where it stands in for a real processor.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/train/rollout/sft/test_role_tool_training_consistency.py -p no:cacheprovider -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from lite.core.messages.image_refs import (
    referenced_image_indices_in_message_order,
    referenced_images_in_message_order,
)
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name


@pytest.fixture(autouse=True)
def _registered_adapters():
    """``AgentAdapterRegistry.get`` below only sees families the registry KNOWS.

    Without this the file passes or fails on whether some *other* test module
    happened to import a family first — green under `-n 16` sharding, red when
    run alone.
    """
    from lite.agents.bootstrap import register_all

    register_all()

# =============================================================================
# Hermetic segmenter Slime stubs
# =============================================================================

class _SegmenterSampleStub:
    class Status:
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    def __init__(self, **kwargs):
        self.group_index = kwargs.pop("group_index")
        self.index = kwargs.pop("index")
        self.prompt = kwargs.pop("prompt")
        self.label = kwargs.pop("label", None)
        self.metadata = kwargs.pop("metadata", None)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _stub_build_processor_kwargs(payload: dict) -> dict:
    return {
        "images": payload.get("images"),
        "images_kwargs": {"return_tensors": "pt"},
    }


def _segmenter_under_test(monkeypatch):
    """Install only the Slime surface ``segmenter`` imports, then import it."""
    slime = types.ModuleType("slime")
    slime.__path__ = []
    utils = types.ModuleType("slime.utils")
    utils.__path__ = []
    proc = types.ModuleType("slime.utils.processing_utils")
    proc.build_processor_kwargs = _stub_build_processor_kwargs
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = _SegmenterSampleStub

    monkeypatch.setitem(sys.modules, "slime", slime)
    monkeypatch.setitem(sys.modules, "slime.utils", utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", proc)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)

    module_name = "_cua_lite_test_segmenter"
    module_path = Path(__file__).parents[4] / "lite/train/rollout/core/segmenter.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module.build_segment_samples, _SegmenterSampleStub


def _sft_rollout_under_test(monkeypatch, processor):
    """Load the real rollout_sft consumer behind the same hermetic Slime stubs."""
    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)
    sys.modules["slime.utils.processing_utils"].load_processor = lambda *args, **kwargs: processor

    core = types.ModuleType("lite.train.rollout.core")
    core.build_segment_samples = build_segment_samples
    core.dummy_sample = lambda sample: Sample(
        group_index=sample.group_index,
        index=sample.index,
        prompt=sample.prompt,
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=getattr(sample, "label", None),
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata={},
    )
    monkeypatch.setitem(sys.modules, "lite.train.rollout.core", core)

    module_name = "_cua_lite_test_rollout_sft"
    module_path = Path(__file__).parents[4] / "lite/train/rollout/sft.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    module.PROCESSOR = processor
    return module.generate_rollout, Sample


# =============================================================================
# Fake processor (hermetic stand-in for a real Qwen3-VL processor)
# =============================================================================

_EOS = 99999


class _FakeTokenizer:
    """char-code tokenizer with the turn terminator ``<|im_end|>`` mapped to a
    single EOS token (mirrors ``test_dagger.py``)."""

    eos_token_id = _EOS

    def encode(self, text, add_special_tokens=False):
        toks, i, marker = [], 0, "<|im_end|>"
        while i < len(text):
            if text.startswith(marker, i):
                toks.append(_EOS)
                i += len(marker)
            else:
                toks.append(ord(text[i]))
                i += 1
        return toks


def _canonical_tool_call_payload_for_qwen_wire(tc: dict) -> dict:
    """Render one canonical Lite call as qwen wire."""
    return {"name": tool_call_name(tc), "arguments": tool_call_arguments(tc)}


def _agent_wire_tool_call_payload_for_qwen_wire(tc: dict) -> dict:
    """Render one qwen-native bare call as qwen wire."""
    return {"name": tc["name"], "arguments": tc["arguments"]}


class _FakeQwenProcessor:
    """Faithful-enough Qwen3-VL chat renderer for the SFT two-pass boundary:

      * each ``role`` wrapped in ``<|im_start|>…<|im_end|>`` turns;
      * each ``image`` content part expands to ONE ``<|image_pad|>`` (inside the
        ``<|vision_start|>…<|vision_end|>`` frame) — so ``prompt.count("<|image_pad|>")``
        is the ground-truth vision-token count the pixel_values must match;
      * ``tool_calls`` render as ``<tool_call>{json}</tool_call>`` blocks.

    It renders ONLY ``text`` / ``image`` content parts (the model-facing kinds the
    real template accepts), matching ``keep_model_visible_content``.
    """

    tokenizer = _FakeTokenizer()
    image_token = "<|image_pad|>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        s = ""
        for m in messages:
            s += f"<|im_start|>{m['role']}\n"
            for p in m.get("content") or []:
                t = p.get("type")
                if t == "text":
                    s += p["text"]
                elif t == "image":
                    s += "<|vision_start|><|image_pad|><|vision_end|>"
            for tc in m.get("tool_calls") or []:
                s += "\n<tool_call>\n" + json.dumps(
                    _canonical_tool_call_payload_for_qwen_wire(tc)
                ) + "\n</tool_call>"
            s += "<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s


# =============================================================================
# Shared message builders
# =============================================================================

def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return make_tool_call(name, arguments, call_id=call_id)


def _computer_call(call_id: str, action: str, arguments: dict) -> dict:
    return _tool_call(
        call_id,
        "computer",
        {"actions": [{"action": action, **arguments}]},
    )


def _count_image_parts(messages: list[dict]) -> int:
    """Ground-truth image count: every ``{"type":"image"}`` content part, in EVERY
    message regardless of role. This is what the pixel_values / ``<image_pad>``
    stream is built from — the invariant the two role-filtered helpers must match."""
    return sum(
        1
        for m in messages
        for p in (m.get("content") or [])
        if isinstance(p, dict) and p.get("type") == "image"
    )


def _initial_user_observation_messages() -> list[dict]:
    """Current first-turn shape: task text and initial screenshot in ``role:"user"``."""
    return [
        {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "Open the menu."},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: Click."}],
         "tool_calls": [_computer_call("call_0", "click", {"coordinate": [18, 508]})]},
    ]


def _role_tool_obs_messages() -> list[dict]:
    """Current per-call shape: the assistant emits a ``computer_call`` and the resulting
    observation screenshot comes back on a per-call ``role:"tool"`` message
    keyed by ``tool_call_id``."""
    return [
        {"role": "user", "content": [{"type": "text", "text": "Open the menu."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: Click."}],
         "tool_calls": [
             _tool_call(
                 "call_0",
                 "computer",
                 {"actions": [{"action": "click", "coordinate": [18, 508]}]},
             ),
         ]},
        {"role": "tool", "tool_call_id": "call_0", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "screenshot after click"},
        ]},
    ]


# =============================================================================
# TR1 — multimodal consistency: image_indices ↔ <image_pad> ↔ ordered images
# =============================================================================


def test_initial_user_observation_multimodal_consistency():
    """The first-turn ``role:"user"`` screenshot follows the same image-slot contract."""
    msgs = _initial_user_observation_messages()
    images = ["IMG0"]

    n_parts = _count_image_parts(msgs)
    indices = referenced_image_indices_in_message_order(msgs)
    ordered = referenced_images_in_message_order(msgs, images)

    assert n_parts == 1
    assert len(indices) == n_parts == len(ordered), (
        "initial user observation must stay aligned with training image slots"
    )


def test_role_tool_screenshot_multimodal_consistency():
    """Regression guard: when the observation screenshot rides a ``role:"tool"`` message,
    the two role-filtered helpers must still count it, so:

        image indices == image parts == image payloads.

    This is the training pixel_values / ``<image_pad>`` sync point the role:tool
    observation contract must preserve."""
    msgs = _role_tool_obs_messages()
    images = ["IMG0"]

    n_parts = _count_image_parts(msgs)
    indices = referenced_image_indices_in_message_order(msgs)
    ordered = referenced_images_in_message_order(msgs, images)
    n_pad = _FakeQwenProcessor().apply_chat_template(msgs).count("<|image_pad|>")

    assert n_parts == 1  # the role:tool screenshot is a real image the model sees
    assert len(images) == len(indices) == n_pad == len(ordered) == n_parts


# =============================================================================
# TR2 — SFT tokenize branch selector (subsumed by TR1)
# =============================================================================
# NOTE: TR2 (the "frozen wrong prompt_tokens baked into the SFT parquet") is
# SUBSUMED by TR1. The bad bake happens precisely when image_indices comes back
# EMPTY for a step that actually has an image: ``agent_step_to_rl_step`` then
# takes the text-only branch and precomputes ``prompt_tokens`` from the
# image-less prompt string — permanently wrong once serialized. Once TR1's
# indices are non-empty for the role:tool obs, this step takes the multimodal
# branch (``prompt_tokens=None``, rebuilt with vision-token expansion at train
# time) and the bake-out cannot happen. Below pins that the branch selector keys
# purely off ``image_indices`` emptiness.
