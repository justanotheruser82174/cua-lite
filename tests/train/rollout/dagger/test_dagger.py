"""Unit tests for DAgger online teacher-forcing relabel.

Validates the core invariant of ``lite.train.rollout.dagger.teacher.relabel_with_teacher``
against a FAKE ``/generate`` (no real model / server):

  * each step's SFT *target* (``response``/``response_tokens``) is swapped to the
    teacher's generated action, re-encoded in the student's tokenizer + EOS;
  * the step's ``prompt`` (the student's on-policy context) is UNCHANGED;
  * a teacher response with no parseable action clears the step (dropped downstream);
  * ``<think>`` is stripped from the target when configured.

Run: python -m pytest tests/train/rollout/dagger/test_dagger.py -q
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.samples import LiteRLSample, LiteRLStep


# dagger imports slime modules at import time. The tests below exercise CUA-Lite
# logic only, so provide a tiny Slime surface off-container instead of skipping.
def _install_slime_stubs() -> list[str]:
    """Install fake ``slime.*`` modules, but ONLY when slime is genuinely absent.

    The guard must probe by IMPORT, not by ``sys.modules`` membership: ``import
    slime`` does not pull in ``slime.utils``, so a membership test reads False
    inside the Slime container and the stubs get installed over the real
    package. Every test module imported after this one in the same process then
    sees the stub ``Sample`` while product code holds the real one --
    ``isinstance`` fails and 78 tests go red, order-dependently, with no
    product defect. Invisible on a slime-free host because those modules skip.
    """
    try:
        importlib.import_module("slime.utils.types")
    except ImportError:
        pass
    else:
        return []

    inserted: list[str] = []

    def put(name: str, *, package: bool = False) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = types.ModuleType(name)
        if package:
            mod.__path__ = []
        mod.__dict__["_cua_lite_test_stub"] = True
        sys.modules[name] = mod
        inserted.append(name)
        return mod

    put("slime", package=True)
    put("slime.rollout", package=True)
    put("slime.rollout.filter_hub", package=True)
    put("slime.utils", package=True)

    class _Sample:
        class Status:
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            ABORTED = "aborted"
            FAILED = "failed"

        def __init__(self, **kwargs):
            self.group_index = kwargs.pop("group_index", None)
            self.index = kwargs.pop("index", None)
            self.group_id = kwargs.pop("group_id", None)
            self.prompt = kwargs.pop("prompt", None)
            self.tokens = kwargs.pop("tokens", None)
            self.loss_mask = kwargs.pop("loss_mask", None)
            self.rollout_log_probs = kwargs.pop("rollout_log_probs", None)
            self.response_length = kwargs.pop("response_length", None)
            self.response = kwargs.pop("response", None)
            self.label = kwargs.pop("label", None)
            self.reward = kwargs.pop("reward", None)
            self.status = kwargs.pop("status", None)
            self.metadata = kwargs.pop("metadata", None)
            self.multimodal_train_inputs = kwargs.pop("multimodal_train_inputs", None)
            self.multimodal_lazy_payloads = kwargs.pop("multimodal_lazy_payloads", None)
            for key, value in kwargs.items():
                setattr(self, key, value)

    base_types = put("slime.rollout.base_types")
    base_types.RolloutFnEvalOutput = types.SimpleNamespace
    base_types.RolloutFnTrainOutput = types.SimpleNamespace

    filter_base_types = put("slime.rollout.filter_hub.base_types")

    class _MetricGatherer:
        def collect(self):
            return {}

    filter_base_types.MetricGatherer = _MetricGatherer
    filter_base_types.call_dynamic_filter = lambda *args, **kwargs: None

    # ``engine.py`` imports these at module scope; this stub set has to mirror
    # its slime surface or the module import above aborts mid-way and leaks the
    # half-installed stubs into every later module in the same worker.
    sglang_rollout = put("slime.rollout.sglang_rollout")

    class _GenerateState:
        def __init__(self, args=None):
            self.args = args

    sglang_rollout.GenerateState = _GenerateState
    sglang_rollout.generate_and_rm = lambda *args, **kwargs: None

    async_utils = put("slime.utils.async_utils")
    async_utils.run = lambda coro: None

    http_utils = put("slime.utils.http_utils")
    http_utils.get = lambda *args, **kwargs: None
    http_utils.post = lambda *args, **kwargs: None

    misc = put("slime.utils.misc")
    misc.load_function = lambda *args, **kwargs: None

    proc = put("slime.utils.processing_utils")
    proc.build_processor_kwargs = lambda payload: {}
    proc.encode_image_for_rollout_engine = lambda *args, **kwargs: None

    types_mod = put("slime.utils.types")
    types_mod.Sample = _Sample
    return inserted


_SLIME_STUBS = _install_slime_stubs()
from slime.utils.types import Sample  # noqa: E402

import lite.train.rollout.dagger.teacher as dt  # noqa: E402  (must follow importorskip)
from lite.core.tools.calls import make_tool_call, tool_call_arguments  # noqa: E402
from lite.train.rollout.dagger import convert_samples_to_train_data  # noqa: E402

# Sibling shims, imported inside the stub window for the empty-batch parity test
# at the bottom of this file (the stubs are torn down immediately below).
from lite.train.rollout.grpo import (  # noqa: E402
    convert_samples_to_train_data as grpo_convert,
)
from lite.train.rollout.reinforce import (  # noqa: E402
    convert_samples_to_train_data as reinforce_convert,
)

for _name in reversed(_SLIME_STUBS):
    sys.modules.pop(_name, None)

_EOS = 99999


class _FakeTokenizer:
    """char-code tokenizer: encode(s) = [ord(c) ...], with the literal turn terminator
    ``<|im_end|>`` mapped to the single EOS token (mirrors a real tokenizer's special token)
    so the round_trip test can detect a double-EOS."""
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

    def decode(self, tokens, skip_special_tokens=False):
        """Inverse of ``encode``; ``skip_special_tokens`` drops the turn terminator.

        Mirrors a real tokenizer, where the template's terminator is a special token:
        decoding an empty target with ``skip_special_tokens=True`` yields ``''``.
        Measured on the real ones -- Qwen2.5-VL/Qwen3.5 render an empty assistant turn
        to ``'<|im_end|>\\n'`` (2 tokens) and Llama-3 to ``'<|eot_id|>'`` (1 token).
        """
        return "".join(
            ("" if skip_special_tokens else "<|im_end|>") if t == _EOS else chr(t)
            for t in tokens
        )


class _FakeAdapter:
    """Parses ``<tool_call>{...}</tool_call>`` as a single click action; else no action.

    For round_trip, ``convert_message_to_agent`` re-renders the parsed action into a fixed
    STUDENT-grammar string (``STUDENT_ACTION``) — deliberately unlike the teacher's raw text
    so the test can prove round_trip used the re-render, not the raw bytes.
    """
    metadata = LiteCUAMetadata(dims=("mobile", "use"))

    def parse_raw_assistant_response(self, response: str):
        return {"role": "assistant", "_raw": response}

    def convert_message_from_agent(self, agent_message):
        raw = agent_message.get("_raw", "")
        if "<tool_call>" in raw:
            return {"role": "assistant", "tool_calls": [
                make_tool_call(
                    "mobile_use",
                    {"action": "click", "coordinate": [1, 2]},
                    call_id="call_0000",
                )
            ]}
        if raw == "NO_ACTION":
            return {"role": "assistant"}  # no tool_calls and no text -> unparseable
        if raw in ("UNRENDERABLE_ACTION", "RENDER_RAISES"):
            # A well-formed teacher action the STUDENT's grammar cannot express.
            return {"role": "assistant", "tool_calls": [
                make_tool_call(
                    "mobile_use",
                    {"action": raw},
                    call_id="call_0000",
                )
            ]}
        if raw == "PARSE_ERROR_RAISES":
            # Families that RAISE malformed model output instead of marking it.
            raise ModelToolCallParseError("teacher emitted an unparseable action")
        if raw == "CONVERTER_BUG":
            # Not model output: a defect in the adapter's own conversion code.
            raise KeyError("adapter bug")
        if raw == "BAD_ACTION":
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": "Action: click bad"}],
                "_lite_model_output_error": "malformed tool call",
            }
        return {"role": "assistant", "content": [{"type": "text", "text": raw}]}

    def convert_message_to_agent(self, lite_msg):
        if not lite_msg.get("tool_calls"):
            return {"role": "assistant", "content": lite_msg.get("content") or []}
        action = tool_call_arguments(lite_msg["tool_calls"][0]).get("action")
        if action == "RENDER_RAISES":            # student grammar rejects it outright
            raise ValueError("student adapter cannot express this action")
        if action == "UNRENDERABLE_ACTION":      # renders, but to nothing at all
            return {"role": "assistant", "content": []}
        return {"role": "assistant", "content": [{"type": "text", "text": "STUDENT_ACTION"}]}


class _FakeProcessor:
    """Renders messages as ``[role]text`` so the SFT two-pass diff yields the response body."""
    tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **_kwargs
    ):
        s = ""
        for m in messages:
            text = "".join(p.get("text", "") for p in (m.get("content") or [])
                           if p.get("type") == "text")
            s += f"[{m['role']}]{text}"
        # add_generation_prompt → open assistant turn (no terminator); else terminate the
        # final assistant turn with <|im_end|> (the real template's EOS) — this is exactly
        # why round_trip must NOT append another eos_id.
        s += "[assistant]" if add_generation_prompt else "<|im_end|>"
        return s


def _make_agent():
    return types.SimpleNamespace(adapter=_FakeAdapter(), processor=_FakeProcessor())


def _make_rl_sample(prompts_and_resps):
    """Build a LiteRLSample whose steps carry the STUDENT's prompt + (nonempty) response."""
    steps = [
        LiteRLStep(prompt=p, image_indices=(), response="student_resp",
                   response_tokens=[1, 2, 3], response_log_probs=[0.0, 0.0, 0.0])
        for (p, _r) in prompts_and_resps
    ]
    return LiteRLSample(
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("mobile", "use"))),
        processed_images=[], steps=steps,
    )


def _args(dagger=None, **over):
    # All DAgger config (teacher agent + runtime knobs) lives under the ``dagger:``
    # namespace (args.dagger dict). teacher_agent_id=None ⇒ same-obs path (no teacher adapter).
    d = dict(teacher_agent_id=None, teacher_agent_kwargs=None,
             teacher_url="http://fake/generate", strip_teacher_think=True,
             teacher_temperature=0.0, teacher_max_new_tokens=64)
    if dagger is not None:
        d.update(dagger)
    base = dict(rollout_max_response_len=512, dagger=d)
    base.update(over)
    return types.SimpleNamespace(**base)


async def test_relabel_swaps_target_keeps_prompt(monkeypatch):
    teacher_text = "Action: tap.\n<tool_call>{\"name\":\"mobile_use\",\"arguments\":{}}</tool_call>"

    async def fake_post(url, payload):
        # same-obs path: teacher sees the student's exact prompt
        assert payload["text"] == "PROMPT_0"
        assert payload["sampling_params"]["max_new_tokens"] == 64
        return {"text": teacher_text}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    step = rl.steps[0]
    student_prompt, student_tokens = step.prompt, list(step.response_tokens)

    out = await dt.relabel_with_teacher(_args(), rl, types.SimpleNamespace(metadata={}),
                                        _make_agent(), "qwen3_vl@mobile@use")

    assert out is rl
    # target swapped to teacher's action, re-encoded + EOS
    assert step.response == teacher_text
    assert step.response_tokens == [ord(c) for c in teacher_text] + [_EOS]
    assert step.response_log_probs == []
    # INVARIANT: the student's context (prompt) is untouched; not the student's tokens
    assert step.prompt == student_prompt == "PROMPT_0"
    assert step.response_tokens != student_tokens


async def test_unparseable_teacher_clears_step(monkeypatch):
    async def fake_post(url, payload):
        return {"text": "NO_ACTION"}  # parser yields no tool_calls and no final text

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(_args(), rl, types.SimpleNamespace(metadata={}),
                                  _make_agent(), "qwen3_vl@mobile@use")
    # no parseable action ⇒ target cleared so the empty sample is dropped downstream
    assert rl.steps[0].response_tokens == []
    assert rl.steps[0].response == ""


async def test_content_only_done_teacher_keeps_final_text(monkeypatch):
    async def fake_post(url, payload):
        return {"text": "Done."}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(_args(), rl, types.SimpleNamespace(metadata={}),
                                  _make_agent(), "qwen3_vl@mobile@use")
    assert rl.steps[0].response == "Done."
    assert rl.steps[0].response_tokens == [ord(c) for c in "Done."] + [_EOS]


async def test_parse_error_teacher_visible_text_is_dropped(monkeypatch):
    async def fake_post(url, payload):
        return {"text": "BAD_ACTION"}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(_args(), rl, types.SimpleNamespace(metadata={}),
                                  _make_agent(), "qwen3_vl@mobile@use")
    assert rl.steps[0].response_tokens == []
    assert rl.steps[0].response == ""


async def test_raised_parse_error_drops_the_step_without_counting_it_errored(monkeypatch):
    """A raised ``ModelToolCallParseError`` is malformed TEACHER OUTPUT, not a failure.

    It is the raising twin of the ``_lite_model_output_error`` marker the qwen
    families set instead, so it belongs in the same bucket: the step is dropped
    and counted unusable, while ``dagger_n_errored`` — the teacher/network
    outage signal — stays clean.
    """
    async def fake_post(url, payload):
        return {"text": "PARSE_ERROR_RAISES"}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    sample = types.SimpleNamespace(metadata={})
    await dt.relabel_with_teacher(_args(), rl, sample,
                                  _make_agent(), "qwen3_vl@mobile@use")
    assert rl.steps[0].response == ""
    assert rl.steps[0].response_tokens == []
    assert sample.metadata["dagger_n_dropped"] == 1
    assert sample.metadata["dagger_n_errored"] == 0


async def test_converter_bug_is_counted_not_swallowed_as_an_empty_teacher_message(monkeypatch):
    """An adapter defect must stay visible instead of reading as an empty teacher turn.

    A bare ``except Exception`` around parse+convert used to substitute ``{}``,
    which is indistinguishable downstream from a teacher that emitted nothing:
    the step was dropped with ``dagger_n_errored == 0`` and the bug never
    surfaced. Only the owned parse error is absorbed now; everything else
    reaches the gather handler that clears the step AND counts it.
    """
    async def fake_post(url, payload):
        return {"text": "CONVERTER_BUG"}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    sample = types.SimpleNamespace(metadata={})
    await dt.relabel_with_teacher(_args(), rl, sample,
                                  _make_agent(), "qwen3_vl@mobile@use")
    # The trajectory survives and the step is never left holding the student's target...
    assert rl.steps[0].response == ""
    assert rl.steps[0].response_tokens == []
    assert sample.metadata["dagger_n_dropped"] == 1
    # ...but unlike an unparseable teacher, the defect is counted.
    assert sample.metadata["dagger_n_errored"] == 1


async def test_round_trip_content_only_teacher_keeps_final_text(monkeypatch):
    async def fake_post(url, payload):
        return {"text": "Final answer from teacher."}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(_args(dagger={"relabel_mode": "round_trip"}), rl,
                                  types.SimpleNamespace(metadata={}), _make_agent(),
                                  "qwen3_vl@mobile@use")

    assert rl.steps[0].response == "Final answer from teacher."
    assert rl.steps[0].response_tokens == [
        ord(c) for c in "Final answer from teacher."
    ] + [_EOS]
    assert rl.steps[0].response_tokens.count(_EOS) == 1


async def test_strip_think(monkeypatch):
    teacher_text = "<think>plan the tap</think>Action: tap.\n<tool_call>{}</tool_call>"

    async def fake_post(url, payload):
        return {"text": teacher_text}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(_args(dagger={"strip_teacher_think": True}), rl,
                                  types.SimpleNamespace(metadata={}), _make_agent(),
                                  "qwen3_vl@mobile@use")
    expected = "Action: tap.\n<tool_call>{}</tool_call>"
    assert rl.steps[0].response == expected
    assert rl.steps[0].response_tokens == [ord(c) for c in expected] + [_EOS]


async def test_think_kept_by_default(monkeypatch):
    """Default (no strip_teacher_think) KEEPS the teacher's <think> — learnable CoT."""
    teacher_text = "<think>plan the tap</think>Action: tap.\n<tool_call>{}</tool_call>"

    async def fake_post(url, payload):
        return {"text": teacher_text}

    monkeypatch.setattr(dt, "post", fake_post)
    # base _args sets strip=True; drop it so the cfg falls to the False default.
    args = _args()
    del args.dagger["strip_teacher_think"]
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(args, rl, types.SimpleNamespace(metadata={}),
                                  _make_agent(), "qwen3_vl@mobile@use")
    assert rl.steps[0].response == teacher_text  # <think> preserved verbatim
    assert rl.steps[0].response_tokens == [ord(c) for c in teacher_text] + [_EOS]


async def test_max_new_tokens_defaults_to_student(monkeypatch):
    """With no teacher_max_new_tokens, the teacher decode budget = the student's."""
    seen = {}

    async def fake_post(url, payload):
        seen["mnt"] = payload["sampling_params"]["max_new_tokens"]
        return {"text": "<tool_call>{}</tool_call>"}

    monkeypatch.setattr(dt, "post", fake_post)
    args = _args(rollout_max_response_len=777)
    del args.dagger["teacher_max_new_tokens"]
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(args, rl, types.SimpleNamespace(metadata={}),
                                  _make_agent(), "qwen3_vl@mobile@use")
    assert seen["mnt"] == 777


async def test_same_context_relabel_ignores_unreferenced_sparse_processed_image_slot(
    monkeypatch,
):
    from PIL import Image

    payloads = []

    def fake_encode(image):
        return f"encoded:{image.getpixel((0, 0))[0]}"

    async def fake_post(url, payload):
        payloads.append(payload)
        return {"text": "Done."}

    monkeypatch.setattr(dt, "encode_image_for_rollout_engine", fake_encode)
    monkeypatch.setattr(dt, "post", fake_post)

    rl = LiteRLSample(
        lite_sample=LiteSample(metadata=LiteCUAMetadata(dims=("mobile", "use"))),
        processed_images=[
            Image.new("RGB", (2, 2), color=(0, 0, 0)),
            None,
            Image.new("RGB", (2, 2), color=(2, 0, 0)),
        ],
        steps=[
            LiteRLStep(
                prompt="PROMPT_0",
                image_indices=(0, 2),
                response="student_resp",
                response_tokens=[1, 2, 3],
                response_log_probs=[0.0, 0.0, 0.0],
            )
        ],
    )

    await dt.relabel_with_teacher(
        _args(),
        rl,
        types.SimpleNamespace(metadata={}),
        _make_agent(),
        "qwen3_vl@mobile@use",
    )

    assert payloads[0]["image_data"] == ["encoded:0", "encoded:2"]
    assert rl.steps[0].response == "Done."


async def test_invalid_relabel_mode_raises(monkeypatch):
    """A typo'd dagger.relabel_mode fails fast (instead of silently falling back to raw)."""
    async def fake_post(url, payload):
        return {"text": "<tool_call>{}</tool_call>"}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("P0", "r")])
    with pytest.raises(ValueError, match="relabel_mode"):
        await dt.relabel_with_teacher(_args(dagger={"relabel_mode": "typo"}), rl,
                                      types.SimpleNamespace(metadata={}), _make_agent(),
                                      "qwen3_vl@mobile@use")


async def test_raw_collapses_preexisting_eos(monkeypatch):
    """raw mode: if sglang did NOT trim and the teacher text already ends in the terminator,
    strip-then-append yields EXACTLY ONE trailing eos (no double-EOS)."""
    teacher_text = "Action: tap.\n<tool_call>{}</tool_call><|im_end|>"  # un-trimmed terminator

    async def fake_post(url, payload):
        return {"text": teacher_text}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(_args(), rl, types.SimpleNamespace(metadata={}),
                                  _make_agent(), "qwen3_vl@mobile@use")
    toks = rl.steps[0].response_tokens
    assert toks[-1] == _EOS            # ends in the terminator
    assert toks.count(_EOS) == 1       # collapsed, not doubled


@pytest.mark.parametrize("teacher_text", ["<tool_call>{}</tool_call>", "Done."])
async def test_missing_eos_token_id_raises_at_load(monkeypatch, teacher_text):
    """A tokenizer with no ``eos_token_id`` fails at load, BEFORE any teacher query.

    The old code guarded both EOS appends with ``if eos_id is not None``, so a
    terminator-less tokenizer silently produced SFT targets with NO terminator -- for both
    the tool-call and the content-only-final branch -- and trained a student that never
    stops. The misconfiguration must be loud where the tokenizer is read, not absorbed
    into the data.
    """
    called = False

    async def fake_post(url, payload):
        nonlocal called
        called = True
        return {"text": teacher_text}

    monkeypatch.setattr(dt, "post", fake_post)

    class _NoEosTokenizer(_FakeTokenizer):
        eos_token_id = None

    class _NoEosProcessor(_FakeProcessor):
        tokenizer = _NoEosTokenizer()

    agent = types.SimpleNamespace(adapter=_FakeAdapter(), processor=_NoEosProcessor())
    rl = _make_rl_sample([("PROMPT_0", "r")])
    with pytest.raises(ValueError, match="eos_token_id"):
        await dt.relabel_with_teacher(_args(), rl, types.SimpleNamespace(metadata={}),
                                      agent, "qwen3_vl@mobile@use")
    assert not called                            # raised at load, not per step
    assert rl.steps[0].response_tokens == [1, 2, 3]   # student's target left untouched


async def test_round_trip_rerenders_in_student_grammar(monkeypatch):
    """round_trip: target is the STUDENT-grammar re-render of the parsed action (via the
    shared SFT action), NOT the teacher's raw bytes."""
    teacher_text = (
        'TEACHER raw bytes <tool_call>{"name":"mobile_use","arguments":{}}</tool_call>'
    )

    async def fake_post(url, payload):
        return {"text": teacher_text}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    await dt.relabel_with_teacher(_args(dagger={"relabel_mode": "round_trip"}), rl,
                                  types.SimpleNamespace(metadata={}), _make_agent(),
                                  "qwen3_vl@mobile@use")
    step = rl.steps[0]
    # target = student re-render, not the teacher's raw text
    assert "STUDENT_ACTION" in step.response
    assert "TEACHER raw bytes" not in step.response
    # template-terminated tokens used VERBATIM: exactly ONE EOS. Regression guard for the
    # round_trip double-EOS bug (old code appended a second eos_id after the <|im_end|>).
    assert step.response_tokens == [ord(c) for c in "STUDENT_ACTION"] + [_EOS]
    assert step.response_tokens.count(_EOS) == 1


async def test_relabel_stashes_unparseable_counts(monkeypatch):
    """relabel stashes per-trajectory scored/dropped counts on sample.metadata (for wandb)."""
    async def fake_post(url, payload):
        # first step parses, second does not
        return {"text": "<tool_call>{}</tool_call>"} if "P0" in payload["text"] \
            else {"text": "NO_ACTION"}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("P0", "r"), ("P1", "r")])
    sample = types.SimpleNamespace(metadata={})
    await dt.relabel_with_teacher(_args(), rl, sample,
                                  _make_agent(), "qwen3_vl@mobile@use")
    assert sample.metadata["dagger_n_scored"] == 2
    assert sample.metadata["dagger_n_dropped"] == 1


async def test_round_trip_empty_render_is_dropped_and_counted(monkeypatch):
    """An action that renders to '' must be DROPPED, not trained on.

    The chat template terminates even an empty assistant turn, so the target is the
    terminator alone -- a 1-token (Llama ``<|eot_id|>``) / 2-token (Qwen ``<|im_end|>\\n``)
    list that is TRUTHY. The old ``not rl.response_tokens`` guard therefore passed it
    through: the step trained on an empty target and ``dagger_n_dropped`` read 0, so the
    very metric built to catch this reported clean. The predicate must be on the decoded
    body, not the token count.
    """
    async def fake_post(url, payload):
        return {"text": "UNRENDERABLE_ACTION"}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("PROMPT_0", "r")])
    sample = types.SimpleNamespace(metadata={})
    await dt.relabel_with_teacher(_args(dagger={"relabel_mode": "round_trip"}), rl,
                                  sample, _make_agent(), "qwen3_vl@mobile@use")
    step = rl.steps[0]
    # The empty target is gone -- and specifically is NOT the bare terminator.
    assert step.response == ""
    assert step.response_tokens == []
    assert step.response_tokens != [_EOS]
    # ...and the drop is COUNTED, not silent.
    assert sample.metadata["dagger_n_scored"] == 1
    assert sample.metadata["dagger_n_dropped"] == 1
    assert sample.metadata["dagger_n_errored"] == 0


async def test_one_failing_step_drops_that_step_not_the_trajectory(monkeypatch):
    """A per-step render failure costs one STEP, and is counted apart from a drop.

    Previously ``_render_student_step`` had no ``try/except`` and the gather had no
    ``return_exceptions``, so one unrenderable action propagated out of
    ``relabel_with_teacher`` -- the whole trajectory became ``n_trajs_errored``
    (indistinguishable from a network error) and the counter stash never ran.
    """
    async def fake_post(url, payload):
        return {"text": "RENDER_RAISES"} if "P1" in payload["text"] \
            else {"text": '<tool_call>{"name":"mobile_use","arguments":{}}</tool_call>'}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("P0", "r"), ("P1", "r"), ("P2", "r")])
    sample = types.SimpleNamespace(metadata={})

    # Does not raise: the trajectory survives.
    out = await dt.relabel_with_teacher(_args(dagger={"relabel_mode": "round_trip"}), rl,
                                        sample, _make_agent(), "qwen3_vl@mobile@use")
    assert out is rl

    # The two good steps still got the teacher's action...
    assert "STUDENT_ACTION" in rl.steps[0].response
    assert "STUDENT_ACTION" in rl.steps[2].response
    # ...and only the failing one was cleared.
    assert rl.steps[1].response == ""
    assert rl.steps[1].response_tokens == []
    assert sample.metadata["dagger_n_scored"] == 3
    assert sample.metadata["dagger_n_dropped"] == 1


async def test_teacher_request_failure_is_counted_as_errored_not_unparseable(monkeypatch):
    """A raising teacher request is counted, and distinguishable from an unparseable one.

    ``dagger_n_errored`` is the subset of ``dagger_n_dropped`` that RAISED, which is what
    separates a teacher/network outage (errored == scored) from a teacher that merely
    emitted junk. The step's target is cleared either way -- a step left holding the
    STUDENT's own response would silently train self-imitation instead of the teacher.
    """
    async def fake_post(url, payload):
        if "P1" in payload["text"]:
            raise ConnectionError("teacher server unreachable")
        return {"text": "NO_ACTION"}          # unparseable, but did not raise

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("P0", "r"), ("P1", "r")])
    student_tokens = list(rl.steps[1].response_tokens)
    sample = types.SimpleNamespace(metadata={})
    await dt.relabel_with_teacher(_args(), rl, sample,
                                  _make_agent(), "qwen3_vl@mobile@use")

    assert rl.steps[1].response_tokens == []
    assert rl.steps[1].response_tokens != student_tokens   # never keeps the student's target
    assert sample.metadata["dagger_n_scored"] == 2
    assert sample.metadata["dagger_n_dropped"] == 2        # both unusable
    assert sample.metadata["dagger_n_errored"] == 1        # but only one RAISED


async def test_only_steps_with_response_relabeled(monkeypatch):
    calls = {"n": 0}

    async def fake_post(url, payload):
        calls["n"] += 1
        return {"text": "<tool_call>{}</tool_call>"}

    monkeypatch.setattr(dt, "post", fake_post)
    rl = _make_rl_sample([("P0", "r"), ("P1", "r")])
    rl.steps[1].response_tokens = []  # an empty student step is not relabeled
    await dt.relabel_with_teacher(_args(), rl, types.SimpleNamespace(metadata={}),
                                  _make_agent(), "qwen3_vl@mobile@use")
    assert calls["n"] == 1  # only the one step with a non-empty student response
    assert rl.steps[0].response_tokens == [ord(c) for c in "<tool_call>{}</tool_call>"] + [_EOS]
    assert rl.steps[1].response_tokens == []


async def test_missing_teacher_url_fails_instead_of_self_distilling(monkeypatch):
    """No teacher URL ⇒ raise, never fall through to the SERVED STUDENT. A
    fall-through turns a misconfigured DAgger run into a silent self-distillation
    run, and no downstream artifact records which one happened.
    """
    monkeypatch.delenv("DAGGER_TEACHER_URL", raising=False)
    posted: list[str] = []

    async def fake_post(url, payload):
        posted.append(url)
        return {"text": "<tool_call>{}</tool_call>"}

    monkeypatch.setattr(dt, "post", fake_post)

    args = _args(dagger={"teacher_url": None})
    args.sglang_router_ip, args.sglang_router_port = "10.0.0.1", 30000
    with pytest.raises(ValueError, match="no teacher URL"):
        await dt.relabel_with_teacher(args, _make_rl_sample([("P0", "r")]),
                                      types.SimpleNamespace(metadata={}), _make_agent(),
                                      "qwen3_vl@mobile@use")

    # A mis-nested `dagger:` block: ``args.dagger`` is absent entirely.
    mis_nested = types.SimpleNamespace(rollout_max_response_len=512,
                                       sglang_router_ip="10.0.0.1", sglang_router_port=30000)
    with pytest.raises(ValueError, match="no teacher URL"):
        await dt.relabel_with_teacher(mis_nested, _make_rl_sample([("P0", "r")]),
                                      types.SimpleNamespace(metadata={}), _make_agent(),
                                      "qwen3_vl@mobile@use")
    assert posted == [], "the student was queried as its own teacher"


async def test_self_distill_is_reachable_only_by_name(monkeypatch):
    """The null case (teacher == served student) stays available — asked for by name."""
    monkeypatch.delenv("DAGGER_TEACHER_URL", raising=False)
    posted: list[str] = []

    async def fake_post(url, payload):
        posted.append(url)
        return {"text": "<tool_call>{}</tool_call>"}

    monkeypatch.setattr(dt, "post", fake_post)
    args = _args(dagger={"teacher_url": dt.SELF_TEACHER_URL})
    args.sglang_router_ip, args.sglang_router_port = "10.0.0.1", 30000
    await dt.relabel_with_teacher(args, _make_rl_sample([("P0", "r")]),
                                  types.SimpleNamespace(metadata={}), _make_agent(),
                                  "qwen3_vl@mobile@use")
    assert posted == ["http://10.0.0.1:30000/generate"]


async def test_env_teacher_url_is_used_when_yaml_has_none(monkeypatch):
    """run_dagger.sh's normal path: dynamic port forwarded through the env var."""
    monkeypatch.setenv("DAGGER_TEACHER_URL", "http://127.0.0.1:41234/generate")
    posted: list[str] = []

    async def fake_post(url, payload):
        posted.append(url)
        return {"text": "<tool_call>{}</tool_call>"}

    monkeypatch.setattr(dt, "post", fake_post)
    await dt.relabel_with_teacher(_args(dagger={"teacher_url": None}),
                                  _make_rl_sample([("P0", "r")]),
                                  types.SimpleNamespace(metadata={}), _make_agent(),
                                  "qwen3_vl@mobile@use")
    assert posted == ["http://127.0.0.1:41234/generate"]


def _dagger_args(**kwargs):
    data = {
        "rollout_batch_size": 1,
        "n_samples_per_prompt": 1,
        "dagger": {"step_filter": "keep_all"},
    }
    data.update(kwargs)
    return types.SimpleNamespace(**data)


def _dagger_sample(
    *,
    turn: int,
    loss_mask: list[int],
    response_length: int | None = None,
    response: str = "target",
) -> Sample:
    response_length = len(loss_mask) if response_length is None else response_length
    return Sample(
        group_index=0,
        index=0,
        group_id=0,
        tokens=[10, 11, 12] + list(range(response_length)),
        loss_mask=loss_mask,
        rollout_log_probs=[0.0] * response_length,
        response_length=response_length,
        response=response,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        metadata={
            "turn_range": (turn, turn),
            "others": {"episode_return": 1.0},
            "dagger_n_scored": 2,
            "dagger_n_dropped": 1,
        },
    )


def test_convert_samples_drops_dagger_zero_loss_segments():
    samples = [
        _dagger_sample(turn=0, loss_mask=[1, 1], response="good"),
        _dagger_sample(turn=1, loss_mask=[], response_length=0, response=""),
    ]

    out = convert_samples_to_train_data(_dagger_args(), samples)

    assert out["tokens"] == [[10, 11, 12, 0, 1]]
    assert out["loss_masks"] == [[1, 1]]
    assert out["response_lengths"] == [2]


def test_convert_samples_uses_dummy_when_all_dagger_segments_are_zero_loss():
    samples = [_dagger_sample(turn=0, loss_mask=[], response_length=0, response="")]

    out = convert_samples_to_train_data(_dagger_args(), samples)

    assert len(out["tokens"]) == 1
    assert out["loss_masks"] == [[0]]
    assert out["response_lengths"] == [1]


# ---------------------------------------------------------------------------
# Empty-convert guard (the {} -> KeyError bug)
# ---------------------------------------------------------------------------
#
# slime hands whatever ``convert_samples_to_train_data`` returns straight to
# ``RolloutManager._split_train_data_by_dp`` (slime/ray/rollout.py:502). That
# function subscripts the dict unconditionally: ``pad_static_groups`` reads
# ``data["group_ids"]`` (slime/utils/dp_schedule.py:275) on the static path, and
# ``data["tokens"]`` is read immediately after on every path. A ``{}`` return
# therefore raises KeyError and kills the run. GRPO and REINFORCE already return
# a synthetic 1-group zero-gradient batch here; DAgger used to return ``{}``.


def _split_train_data_by_dp_key_reads(data: dict) -> None:
    """The unconditional subscripts slime performs on a convert result.

    Mirrors ``pad_static_groups`` (``data["group_ids"]``) and
    ``_split_train_data_by_dp`` (``data["tokens"]``). Raises the same KeyError
    slime would raise, without needing slime installed.
    """
    data["group_ids"]
    data["tokens"]


def test_split_train_data_by_dp_key_reads_reject_an_empty_dict():
    """Forced-failure repro: the old ``return {}`` is what slime chokes on."""
    with pytest.raises(KeyError):
        _split_train_data_by_dp_key_reads({})


def test_convert_samples_on_empty_batch_returns_a_zero_gradient_batch_not_empty_dict():
    args = _dagger_args(rollout_batch_size=2, n_samples_per_prompt=2)

    out = convert_samples_to_train_data(args, [])

    assert out != {}
    _split_train_data_by_dp_key_reads(out)  # would KeyError on the old return
    # Padded out to the launched count, every row zero-gradient.
    assert len(out["tokens"]) == 4
    assert all(sum(mask) == 0 for mask in out["loss_masks"])
    assert all(reward == 0.0 for reward in out["rewards"])
    # Distinct group ids so slime's group-based scheduler sees 4 groups.
    assert len(set(out["group_ids"])) == 4


def test_empty_batch_guard_is_identical_across_grpo_reinforce_and_dagger():
    """The three converters must agree on the empty-batch shape."""
    def shape(fn):
        out = fn(_dagger_args(rollout_batch_size=2, n_samples_per_prompt=2), [])
        return (
            sorted(out),
            len(out["tokens"]),
            out["loss_masks"],
            out["rewards"],
        )

    assert shape(convert_samples_to_train_data) == shape(grpo_convert)
    assert shape(convert_samples_to_train_data) == shape(reinforce_convert)
