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
from types import SimpleNamespace

import pytest

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.image_refs import (
    referenced_images_in_message_order,
)
from lite.core.samples import STATUS_COMPLETED, STATUS_TRUNCATED, LiteRLSample, LiteRLStep
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.train.export.sft_tokenize import (
    agent_step_to_rl_step,
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

def test_role_tool_image_indices_reach_sft_and_segmenter_multimodal_branch(monkeypatch):
    """A role:"tool" screenshot must be non-empty ``image_indices`` through SFT
    tokenization and then enter the segmenter's multimodal path."""
    proc = _FakeQwenProcessor()
    step = [
        {"role": "system", "content": [{"type": "text", "text": "You are an agent."}]},
        {"role": "user", "content": [{"type": "text", "text": "Open the menu."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: Click."}],
         "tool_calls": [_computer_call("call_0", "click", {"coordinate": [18, 508]})]},
        {"role": "tool", "tool_call_id": "call_0", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "screenshot after click"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: wait."}],
         "tool_calls": [_computer_call("call_1", "wait", {"duration": 1})]},
    ]
    rl = agent_step_to_rl_step(step, proc)

    assert rl is not None
    assert rl.image_indices == (0,)
    assert rl.prompt_tokens is None, "role:tool image step must use SFT multimodal branch"
    assert (
        len(referenced_images_in_message_order(step, ["IMG0"]))
        == len(rl.image_indices)
        == rl.prompt.count("<|image_pad|>")
        == 1
    )

    import torch
    from PIL import Image

    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)

    class _TinyImageProcessor:
        def __call__(self, images=None, return_tensors=None):
            n = len(images or [])
            return {
                "pixel_values": torch.ones((n, 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
            }

    class _TinyProcessor:
        image_token = "<|image_pad|>"

        def __init__(self):
            self.image_processor = _TinyImageProcessor()
            self.calls = []

        def __call__(self, text=None, images=None, text_kwargs=None, **kwargs):
            n = len(images or [])
            self.calls.append((text, n))
            return {
                "input_ids": [1, 2, 3] + [42] * n,
                "pixel_values": torch.ones((n, 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
            }

    class _TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")

    rl_sample = LiteRLSample(
        processed_images=[Image.new("RGB", (4, 4), color=(10, 20, 30))],
        steps=[LiteRLStep(
            prompt=rl.prompt,
            image_indices=rl.image_indices,
            response=rl.response,
            response_tokens=rl.response_tokens,
            response_log_probs=rl.response_log_probs,
            reward=0.0,
            status=STATUS_COMPLETED,
            prompt_tokens=rl.prompt_tokens,
        )],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
    )
    processor = _TinyProcessor()
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=Sample(group_index=0, index=0, prompt="x"),
        processor=processor,
        tokenizer=_TinyTokenizer(),
        state=SimpleNamespace(
            args=SimpleNamespace(multimodal_lazy_expand_fn_path=None),
            aborted=False,
        ),
    )

    assert len(samples) == 1
    assert processor.calls == [(rl.prompt, 1)]
    assert samples[0].multimodal_train_inputs is not None
    assert samples[0].multimodal_train_inputs["pixel_values"].shape[0] == len(rl.image_indices)
    assert samples[0].multimodal_lazy_payloads is None


def test_role_tool_image_indices_reach_segmenter_lazy_payloads(monkeypatch):
    """The lazy multimodal path must carry the role:tool screenshot index too."""
    proc = _FakeQwenProcessor()
    step = [
        {"role": "user", "content": [{"type": "text", "text": "Open the menu."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: Click."}],
         "tool_calls": [_computer_call("call_0", "click", {"coordinate": [18, 508]})]},
        {"role": "tool", "tool_call_id": "call_0", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "screenshot after click"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Action: wait."}],
         "tool_calls": [_computer_call("call_1", "wait", {"duration": 1})]},
    ]
    rl = agent_step_to_rl_step(step, proc)
    assert rl is not None

    import torch
    from PIL import Image

    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)

    class _TinyImageProcessor:
        def __call__(self, images=None, return_tensors=None):
            n = len(images or [])
            return {
                "pixel_values": torch.ones((n, 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
            }

    class _TinyProcessor:
        image_token = "<|image_pad|>"

        def __init__(self):
            self.image_processor = _TinyImageProcessor()

        def __call__(self, text=None, images=None, text_kwargs=None, **kwargs):
            n = len(images or [])
            return {
                "input_ids": [1, 2, 3] + [42] * n,
                "pixel_values": torch.ones((n, 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
            }

    class _TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1")

    rl_sample = LiteRLSample(
        processed_images=[Image.new("RGB", (4, 4), color=(10, 20, 30))],
        steps=[LiteRLStep(
            prompt=rl.prompt,
            image_indices=rl.image_indices,
            response=rl.response,
            response_tokens=rl.response_tokens,
            response_log_probs=rl.response_log_probs,
            reward=0.0,
            status=STATUS_COMPLETED,
            prompt_tokens=rl.prompt_tokens,
        )],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
    )
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=Sample(group_index=0, index=0, prompt="x"),
        processor=_TinyProcessor(),
        tokenizer=_TinyTokenizer(),
        state=SimpleNamespace(
            args=SimpleNamespace(multimodal_lazy_expand_fn_path="lite.train.utils.multimodal_expand.expand"),
            aborted=False,
        ),
    )

    assert len(samples) == 1
    assert samples[0].multimodal_train_inputs is None
    payload = samples[0].multimodal_lazy_payloads
    assert payload["indices"] == rl.image_indices == (0,)
    assert set(payload["images"]) == {0}
    assert payload["images"][0]["image_data"].dtype == torch.uint8


@pytest.mark.parametrize("lazy_expand", [False, True])
def test_segmenter_ignores_unreferenced_sparse_processed_image_slots(
    monkeypatch,
    lazy_expand,
):
    """Sparse ``processed_images`` holes are legal unless a step references them."""
    import torch
    from PIL import Image

    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)

    class _TinyImageProcessor:
        def __call__(self, images=None, return_tensors=None):
            n = len(images or [])
            return {
                "pixel_values": torch.ones((n, 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
            }

    class _TinyProcessor:
        image_token = "<|image_pad|>"

        def __init__(self):
            self.image_processor = _TinyImageProcessor()

        def __call__(self, text=None, images=None, text_kwargs=None, **kwargs):
            n = len(images or [])
            return {
                "input_ids": [1, 2, 3] + [42] * n,
                "pixel_values": torch.ones((n, 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((n, 3), dtype=torch.long),
            }

    class _TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1" if lazy_expand else "0")

    rl_sample = LiteRLSample(
        processed_images=[
            Image.new("RGB", (4, 4), color=(10, 20, 30)),
            None,
            Image.new("RGB", (4, 4), color=(40, 50, 60)),
        ],
        steps=[LiteRLStep(
            prompt="<|image_pad|><|image_pad|>",
            image_indices=(0, 2),
            response="target",
            response_tokens=[1, 2],
            response_log_probs=[-0.1, -0.2],
            reward=0.0,
            status=STATUS_COMPLETED,
        )],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
    )
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=Sample(group_index=0, index=0, prompt="x"),
        processor=_TinyProcessor(),
        tokenizer=_TinyTokenizer(),
        state=SimpleNamespace(
            args=SimpleNamespace(
                multimodal_lazy_expand_fn_path=(
                    "lite.train.utils.multimodal_expand.expand" if lazy_expand else None
                )
            ),
            aborted=False,
        ),
    )

    assert len(samples) == 1
    if lazy_expand:
        payload = samples[0].multimodal_lazy_payloads
        assert samples[0].multimodal_train_inputs is None
        assert payload["indices"] == (0, 2)
        assert set(payload["images"]) == {0, 2}
    else:
        assert samples[0].multimodal_lazy_payloads is None
        assert samples[0].multimodal_train_inputs["pixel_values"].shape[0] == 2


@pytest.mark.parametrize("lazy_expand", [False, True])
def test_segmenter_does_not_silently_accept_referenced_sparse_processed_image_none(
    monkeypatch,
    lazy_expand,
):
    """A referenced sparse slot must fail before producing a train sample."""
    from PIL import Image

    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)

    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "1" if lazy_expand else "0")

    rl_sample = LiteRLSample(
        processed_images=[
            Image.new("RGB", (4, 4), color=(10, 20, 30)),
            None,
        ],
        steps=[LiteRLStep(
            prompt="<|image_pad|><|image_pad|>",
            image_indices=(0, 1),
            response="target",
            response_tokens=[1],
            response_log_probs=[-0.1],
            reward=0.0,
            status=STATUS_COMPLETED,
        )],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
    )

    with pytest.raises((AttributeError, TypeError)):
        build_segment_samples(
            rl_sample=rl_sample,
            original_sample=Sample(group_index=0, index=0, prompt="x"),
            processor=object(),
            tokenizer=object(),
            state=SimpleNamespace(
                args=SimpleNamespace(
                    multimodal_lazy_expand_fn_path=(
                        "lite.train.utils.multimodal_expand.expand"
                        if lazy_expand else None
                    )
                ),
                aborted=False,
            ),
        )


def test_segmenter_keeps_prompt_image_order_and_tool_error_delta(monkeypatch):
    """No-slime B6 guard for radix packing under tool-calling interaction.

    The processor binds images positionally: the first image object passed to
    ``processor(text=..., images=[...])`` fills the first ``<|image_pad|>`` in
    the prompt. This test uses non-monotonic indices to pin that order, then
    verifies that role:tool error text between assistant turns is only an
    unmasked prompt delta, not a segmentation break.
    """
    import torch
    from PIL import Image

    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)

    class _RecordingImageProcessor:
        def __call__(self, images=None, return_tensors=None):
            return {
                "pixel_values": torch.ones((len(images or []), 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((len(images or []), 3), dtype=torch.long),
            }

    class _RecordingProcessor:
        image_token = "<|image_pad|>"

        def __init__(self):
            self.image_processor = _RecordingImageProcessor()
            self.image_call_ids: list[tuple[int, ...]] = []

        def __call__(self, text=None, images=None, text_kwargs=None, **kwargs):
            imgs = list(images or [])
            self.image_call_ids.append(tuple(getattr(img, "_cua_test_id") for img in imgs))
            return {
                "input_ids": [ord(ch) for ch in (text or "")],
                "pixel_values": torch.ones((len(imgs), 1), dtype=torch.float32),
                "image_grid_thw": torch.ones((len(imgs), 3), dtype=torch.long),
            }

    class _Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

    def _img(idx: int):
        img = Image.new("RGB", (4, 4), color=(idx, idx, idx))
        img._cua_test_id = idx
        return img

    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)

    prompt0 = "user <|image_pad|><|image_pad|>\n"
    response0 = "Action: click\n"
    tool_error_delta = "role:tool click failed: invalid arguments\n<|image_pad|>\n"
    prompt1 = prompt0 + response0 + tool_error_delta
    response1 = "Action: wait\n"

    rl_sample = LiteRLSample(
        processed_images=[_img(0), _img(1), _img(2)],
        steps=[
            LiteRLStep(
                prompt=prompt0,
                image_indices=(1, 0),
                response=response0,
                response_tokens=[ord(ch) for ch in response0],
                response_log_probs=[-0.1] * len(response0),
                reward=0.0,
                status=STATUS_COMPLETED,
            ),
            LiteRLStep(
                prompt=prompt1,
                image_indices=(1, 0, 2),
                response=response1,
                response_tokens=[ord(ch) for ch in response1],
                response_log_probs=[-0.2] * len(response1),
                reward=1.0,
                status=STATUS_TRUNCATED,
            ),
        ],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
        episode_return=1.0,
    )

    processor = _RecordingProcessor()
    samples = build_segment_samples(
        rl_sample=rl_sample,
        original_sample=Sample(group_index=0, index=7, prompt="orig"),
        processor=processor,
        tokenizer=_Tokenizer(),
        state=SimpleNamespace(
            args=SimpleNamespace(multimodal_lazy_expand_fn_path=None),
            aborted=False,
        ),
    )

    assert processor.image_call_ids == [(1, 0), (1, 0, 2)]
    assert len(samples) == 1
    sample = samples[0]
    assert sample.metadata["turn_range"] == (0, 1)
    assert sample.metadata["step_statuses"] == (STATUS_COMPLETED, STATUS_TRUNCATED)
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.response == response0 + response1
    assert sample.loss_mask == (
        [1] * len(response0)
        + [0] * len(tool_error_delta)
        + [1] * len(response1)
    )
    assert sample.multimodal_train_inputs["pixel_values"].shape[0] == 3


def test_segmenter_rejects_image_step_without_multimodal_processor(monkeypatch):
    """Non-empty ``image_indices`` must not fall through to tokenizer-only
    processing when the training processor is missing.

    The guard now lives in ``_process_step`` (``traj_proc is None`` on an
    image-bearing step) rather than in a pre-pass slot-count validator.
    """
    build_segment_samples, Sample = _segmenter_under_test(monkeypatch)
    from PIL import Image

    class _TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")

    rl_sample = LiteRLSample(
        processed_images=[Image.new("RGB", (4, 4), color=(10, 20, 30))],
        steps=[LiteRLStep(
            prompt="<|image_pad|>",
            image_indices=(0,),
            response="",
            response_tokens=[],
            response_log_probs=[],
            reward=0.0,
            status=STATUS_COMPLETED,
        )],
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("desktop", "use"))),
    )

    with pytest.raises(RuntimeError, match="no multimodal processor"):
        build_segment_samples(
            rl_sample=rl_sample,
            original_sample=Sample(group_index=0, index=0, prompt="x"),
            processor=None,
            tokenizer=_TinyTokenizer(),
            state=SimpleNamespace(
                args=SimpleNamespace(multimodal_lazy_expand_fn_path=None),
                aborted=False,
            ),
        )


def test_rollout_sft_text_steps_reach_radix_segmenter_without_slime(monkeypatch):
    """No-slime SFT consumer guard for the successful radix path.

    This uses the real ``rollout_sft.generate_rollout`` source behind Slime
    stubs, but keeps the row text-only so no real HF processor or Slime runtime is
    implied. The contract under test is just: serialized ``LiteRLStep`` rows are
    deserialized and handed to the shared segmenter, which packs prefix-
    extensional turns and builds the multi-span mask.
    """

    class _UnusedImageProcessor:
        def __call__(self, images=None, return_tensors=None):
            raise AssertionError("text-only SFT row should not process images")

    class _TextProcessor:
        tokenizer = _FakeTokenizer()
        image_processor = _UnusedImageProcessor()

    processor = _TextProcessor()
    generate_rollout, Sample = _sft_rollout_under_test(monkeypatch, processor=processor)
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)

    prompt0 = "user: observe\nassistant: "
    response0 = "Action: click\n"
    delta = "tool: click succeeded\nassistant: "
    prompt1 = prompt0 + response0 + delta
    response1 = "Action: done\n"
    steps = [
        serialize_rl_step(LiteRLStep(
            prompt=prompt0,
            image_indices=(),
            response=response0,
            response_tokens=processor.tokenizer.encode(response0),
            response_log_probs=[-0.1] * len(response0),
            reward=0.0,
            status=STATUS_COMPLETED,
        )),
        serialize_rl_step(LiteRLStep(
            prompt=prompt1,
            image_indices=(),
            response=response1,
            response_tokens=processor.tokenizer.encode(response1),
            response_log_probs=[-0.2] * len(response1),
            reward=1.0,
            status=STATUS_TRUNCATED,
        )),
    ]
    sample = Sample(
        group_index=2,
        index=9,
        prompt=steps,
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata=[],
    )

    class _Buffer:
        def get_samples(self, batch_size):
            return [(sample,)]

    args = SimpleNamespace(
        rollout_batch_size=1,
        rollout_global_dataset=True,
        hf_checkpoint="fake-qwen",
        multimodal_lazy_expand_fn_path=None,
    )
    out = generate_rollout(args, rollout_id=0, data_buffer=_Buffer(), evaluation=False)

    assert len(out) == 1
    packed = out[0]
    assert packed.group_id == 9
    assert packed.metadata["turn_range"] == (0, 1)
    assert packed.metadata["n_turns"] == 2
    assert packed.status == Sample.Status.TRUNCATED
    assert packed.response == response0 + response1
    assert packed.loss_mask == (
        [1] * len(response0)
        + [0] * len(delta)
        + [1] * len(response1)
    )
    assert packed.multimodal_train_inputs is None
    assert packed.multimodal_lazy_payloads is None


def test_rollout_sft_uses_exported_steps_and_keeps_lite_metadata_provenance_only(
    monkeypatch,
):
    """SFT training consumes exported LiteRLStep structs, not Lite task metadata.

    ``export_sft`` keeps Lite metadata as a provenance column in the parquet, while
    ``run_sft.sh`` routes ``processed_images`` through Slime's ``Sample.metadata``.
    The harness must therefore train from already-rendered prompts/tokens and not
    spread Lite metadata into emitted train-sample metadata.
    """

    class _NoTrainTokenizer:
        def encode(self, *_args, **_kwargs):
            raise AssertionError("SFT training must use exported prompt_tokens")

    class _UnusedImageProcessor:
        def __call__(self, images=None, return_tensors=None):
            raise AssertionError("text-only SFT row should not process images")

    class _NoRenderProcessor:
        tokenizer = _NoTrainTokenizer()
        image_processor = _UnusedImageProcessor()

        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("SFT training must not render chat templates")

    processor = _NoRenderProcessor()
    generate_rollout, Sample = _sft_rollout_under_test(monkeypatch, processor=processor)
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")
    monkeypatch.delenv("CUA_LITE_DISABLE_RADIX", raising=False)

    lite_metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        others={"env_id": "lite.demo", "task_id": "task_0"},
    ).to_dict()
    prompt_tokens = [101, 102, 103]
    response_tokens = [201, 202]
    step = serialize_rl_step(LiteRLStep(
        prompt="<rendered before export>",
        image_indices=(),
        response="<target>",
        response_tokens=response_tokens,
        reward=1.0,
        status=STATUS_COMPLETED,
        prompt_tokens=prompt_tokens,
    ))
    sample = Sample(
        group_index=0,
        index=7,
        prompt=[step],
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata=[],
    )

    class _Buffer:
        def get_samples(self, batch_size):
            return [(sample,)]

    args = SimpleNamespace(
        rollout_batch_size=1,
        rollout_global_dataset=True,
        hf_checkpoint="fake-qwen",
        multimodal_lazy_expand_fn_path=None,
    )
    out = generate_rollout(args, rollout_id=0, data_buffer=_Buffer(), evaluation=False)

    assert len(out) == 1
    packed = out[0]
    assert packed.tokens == prompt_tokens + response_tokens
    assert packed.response == "<target>"
    assert "metadata_kind" in lite_metadata
    assert "metadata_kind" not in packed.metadata
    assert "dims" not in packed.metadata
    assert "extra_tool_schemas" not in packed.metadata
    assert "processed_images" not in packed.metadata
    assert packed.metadata["turn_range"] == (0, 0)
    assert packed.metadata["n_turns"] == 1


def test_rollout_sft_multi_row_batch_preserves_flat_order_and_indices(monkeypatch):
    """SFT rollout fetches ``rollout_batch_size`` rows and extends outputs in
    row order while preserving each row's group/index identity."""

    class _UnusedImageProcessor:
        def __call__(self, images=None, return_tensors=None):
            raise AssertionError("text-only SFT row should not process images")

    class _TextProcessor:
        tokenizer = _FakeTokenizer()
        image_processor = _UnusedImageProcessor()

    processor = _TextProcessor()
    generate_rollout, Sample = _sft_rollout_under_test(monkeypatch, processor=processor)
    monkeypatch.setenv("CUA_LITE_DISABLE_RADIX", "1")
    monkeypatch.setenv("CUA_LITE_MULTIMODAL_LAZY_EXPAND", "0")

    def _step(prompt: str, response: str, reward: float = 0.0) -> dict:
        return serialize_rl_step(LiteRLStep(
            prompt=prompt,
            image_indices=(),
            response=response,
            response_tokens=processor.tokenizer.encode(response),
            reward=reward,
            status=STATUS_COMPLETED,
        ))

    row0_steps = [
        _step("row0 turn0 assistant: ", "a0\n", reward=0.1),
        _step("row0 turn1 assistant: ", "a1\n", reward=0.2),
    ]
    row1_steps = [_step("row1 turn0 assistant: ", "b0\n", reward=0.3)]
    row0 = Sample(
        group_index=7,
        index=101,
        prompt=row0_steps,
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata=[],
    )
    row1 = Sample(
        group_index=8,
        index=202,
        prompt=row1_steps,
        tokens=[],
        loss_mask=[],
        response_length=0,
        response="",
        label=None,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata=[],
    )

    class _Buffer:
        def __init__(self):
            self.batch_sizes: list[int] = []

        def get_samples(self, batch_size):
            self.batch_sizes.append(batch_size)
            return [(row0,), (row1,)]

    args = SimpleNamespace(
        rollout_batch_size=2,
        rollout_global_dataset=True,
        hf_checkpoint="fake-text",
        multimodal_lazy_expand_fn_path=None,
    )
    buf = _Buffer()
    out = generate_rollout(args, rollout_id=0, data_buffer=buf, evaluation=False)

    assert buf.batch_sizes == [2]
    assert [(s.group_index, s.index, s.group_id) for s in out] == [
        (7, 101, 101),
        (7, 101, 101),
        (8, 202, 202),
    ]
    assert [s.metadata["turn_range"] for s in out] == [(0, 0), (1, 1), (0, 0)]
    assert [s.metadata["n_turns"] for s in out] == [2, 2, 1]
    assert [s.response for s in out] == ["a0\n", "a1\n", "b0\n"]


# =============================================================================
