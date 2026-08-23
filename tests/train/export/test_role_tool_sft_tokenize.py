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

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance
from lite.core import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.train.export.sft_tokenize import (
    agent_step_to_rl_step,
    deserialize_rl_step,
    serialize_rl_step,
)


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
    module_path = Path(__file__).parents[3] / "lite/train/rollout/core/segmenter.py"
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
    module_path = Path(__file__).parents[3] / "lite/train/rollout/sft.py"
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

def test_sft_tokenize_branch_selector_keys_off_image_indices():
    """The SFT-tokenize branch is selected by ``image_indices``:
      * non-empty → multimodal branch → ``prompt_tokens is None`` (rebuilt at
        train time with per-image vision-token expansion);
      * empty     → text-only branch → ``prompt_tokens`` precomputed & frozen.

    This is the mechanism TR1 protects: an image step whose indices are WRONGLY
    empty would silently fall into the text-only branch and bake bad tokens."""
    proc = _FakeQwenProcessor()

    # Image step: initial user image → non-empty indices → multimodal branch.
    img_step = [
        {"role": "user", "content": [
            {"type": "image", "index": 0}, {"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: Click."}],
         "tool_calls": [_computer_call("call_img", "click", {"coordinate": [1, 2]})]},
    ]
    img_rl = agent_step_to_rl_step(img_step, proc)
    assert img_rl.image_indices == (0,)
    assert img_rl.prompt_tokens is None, "image step must defer prompt_tokens (multimodal branch)"

    # Text-only step: no image parts → empty indices → text-only branch.
    text_step = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: wait."}],
         "tool_calls": [_computer_call("call_text", "wait", {"duration": 1})]},
    ]
    text_rl = agent_step_to_rl_step(text_step, proc)
    assert text_rl.image_indices == ()
    assert text_rl.prompt_tokens is not None, "text-only step must precompute prompt_tokens"


# TR3 — rollout-log → SFT export round-trip (byte-identical + images==image_pad)
# =============================================================================
# Frozen golden for the current first-turn SFT-tokenize output: initial user
# screenshot + task text → assistant target. Provenance coverage is
# metadata-only; this is a byte-level guard on prompt/response, the serialized
# LiteRLStep struct shape, and the images==<image_pad> invariant.
_SFT_PROMPT_GOLDEN = (
    "<|im_start|>system\nYou are an agent.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Open the menu.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_SFT_RESPONSE_GOLDEN = (
    "Action: Click the menu.\n"
    '<tool_call>\n{"name": "computer", "arguments": {"actions": [{"action": '
    '"click", "coordinate": [18, 508]}]}}\n</tool_call>'
    "<|im_end|>\n"
)


def test_rollout_log_to_sft_roundtrip():
    """Current guard: a rendered rollout step → ``agent_step_to_rl_step`` →
    parquet-struct serialize/deserialize round-trips BYTE-IDENTICAL to a frozen
    golden, AND ``#images (image_indices) == #<image_pad>`` in the prompt.

    Uses the faithful fake Qwen processor (no model download) — the real
    processor is unavailable hermetically; the golden pins the fake render's
    byte layout as the regression anchor."""
    proc = _FakeQwenProcessor()
    step = [
        {"role": "system", "content": [{"type": "text", "text": "You are an agent."}]},
        {"role": "user", "content": [
            {"type": "image", "index": 0}, {"type": "text", "text": "Open the menu."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: Click the menu."}],
         "tool_calls": [_computer_call("call_0", "click", {"coordinate": [18, 508]})]},
    ]

    rl = agent_step_to_rl_step(step, proc)
    # Parquet-struct round-trip (what export_sft writes / rollout_sft reads back).
    serialized = serialize_rl_step(rl)
    assert set(serialized) == {
        "prompt",
        "image_indices",
        "response",
        "response_tokens",
        "reward",
        "status",
        "prompt_tokens",
    }
    assert isinstance(serialized["image_indices"], list)
    rt = deserialize_rl_step(serialized)

    # Byte-identical prompt / response through the round-trip.
    assert rt.prompt == _SFT_PROMPT_GOLDEN
    assert rt.response == _SFT_RESPONSE_GOLDEN

    # Multimodal balance: one referenced image ⇔ one <image_pad> in the prompt.
    n_pad = rt.prompt.count("<|image_pad|>")
    assert len(rt.image_indices) == n_pad == 1, (
        f"images ({len(rt.image_indices)}) must equal <image_pad> ({n_pad})"
    )


# =============================================================================
# TR4 — cross-model: GPT-teacher multi-action computer_call → qwen student target
# =============================================================================
# Frozen NEW-contract target: a GPT multi-action ``computer_call.actions`` parses to
# ONE canonical ``computer`` action-batch tool call whose ``arguments.actions`` holds the GUI
# actions. The qwen student wire then expands that canonical batch back into
# adjacent ``computer_use`` calls.
_QWEN_BATCHED_TARGET_GOLDEN = (
    '<tool_call>\n{"name": "computer_use", "arguments": {"action": '
    '"left_click", "coordinate": [500, 500]}}\n</tool_call>\n'
    '<tool_call>\n{"name": "computer_use", "arguments": {"action": "type", '
    '"text": "hello"}}\n</tool_call>'
)


def _reconstruct_qwen_target(agent_message: dict) -> str:
    """Rebuild the qwen student render target from a rendered AgentMessage:
    the ``Action:`` body text lines followed by one ``<tool_call>`` block per
    tool_call (the qwen wire layout, minus the chat template's turn frame)."""
    parts: list[str] = []
    for p in agent_message.get("content") or []:
        if p.get("type") == "text" and p.get("text"):
            parts.append(p["text"])
    for tc in agent_message.get("tool_calls") or []:
        parts.append(
            "<tool_call>\n"
            + json.dumps(_agent_wire_tool_call_payload_for_qwen_wire(tc))
            + "\n</tool_call>"
        )
    return "\n".join(parts)


def test_crossmodel_gpt_multi_action_to_qwen_student_golden():
    """NEW contract: a GPT-teacher multi-action ``computer_call`` →
    one canonical ``computer`` action-batch tool_call → qwen student render expands
    to adjacent native ``computer_use`` calls, byte-equal to the frozen golden.

    Hermetic: drives the real GPT parse + qwen adapter conversion (no model /
    processor)."""
    # GPT teacher emits ONE computer_call batching two actions.
    gpt_output_items = [
        {"type": "computer_call", "call_id": "call_0000", "actions": [
            {"type": "click", "x": 512, "y": 384},
            {"type": "type", "text": "hello"},
        ]},
    ]
    canonical = parse_output_items_with_provenance(
        gpt_output_items, GPTDesktopActionSpace(), resolution=(1024, 768),
    ).message
    assert canonical["tool_calls"] == [
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [500, 500]},
                    {"action": "type", "text": "hello"},
                ],
            },
            call_id="call_0000",
        ),
    ]

    adapter = AgentAdapterRegistry.get(
        "qwen3_vl@desktop@use",
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
    )
    agent_message = adapter._convert_message_to_agent(canonical)

    assert [tc["name"] for tc in agent_message["tool_calls"]] == [
        "computer_use",
        "computer_use",
    ]

    # And the unrolled render target is byte-equal to the frozen batched golden.
    assert _reconstruct_qwen_target(agent_message) == _QWEN_BATCHED_TARGET_GOLDEN
