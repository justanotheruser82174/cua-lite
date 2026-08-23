"""Cross-model SFT distillation guard (teacher rollout → student SFT target).

The core functional-equivalence bar for cross-model SFT export: a TEACHER
rollout (GPT ``computer_call.actions`` batches; Claude parallel actions arrive
as separate ``tool_use`` items) must distill to the CORRECT STUDENT (qwen3_vl /
qwen3_5) SFT target after canonicalization. This is the "cross-model
rollout→SFT" guard for the nested Lite tool-call and ``role:"tool"`` result
contract.

The canonical bridge exercised here is:

    teacher raw wire  ──teacher.parse──▶  canonical nested LiteMessage tool_calls
                      ──[trajectory]───▶  LiteSample messages + images
                      ──student.unroll─▶  AgentSample.steps  (the SFT render target)

Two families of tests:

* Characterization / file-golden: the current per-action
  cross-model round-trip. A single teacher action (GPT ``computer_call`` /
  Claude ``computer`` ``tool_use``) parses to the SAME canonical LiteMessage and
  renders to a frozen student ``steps`` golden. This baseline must stay
  byte-identical through adapter/export machinery changes.
  ``test_browser_crossagent.py`` stops at the pivot-name / single-action; this
  file goes deeper — the full rendered SFT target across the teacher×student
  matrix.

* Current-contract guards: GPT batched ``computer_call`` and Claude parallel
  ``computer`` both normalize into one canonical ``computer{actions:[...]}``
  call; post-action screenshots are represented as paired ``role:"tool"``
  messages keyed by ``tool_call_id``.

Hermetic by default: ``parse`` + ``unroll`` are pure Python (no model download,
no network); ``steps`` reference images by index, so ``pformat`` is
deterministic. A model-gated ``apply_chat_template`` layer runs only when a
Qwen processor is cached locally (otherwise skips).

Regenerate goldens (ONLY after an intentional render change, review the diff):
    UPDATE_CROSSMODEL_GOLDENS=1 env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/train/export/test_crossmodel_sft.py -p no:cacheprovider -q

Run (verify byte-identity; run twice to confirm golden determinism):
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/train/export/test_crossmodel_sft.py -p no:cacheprovider -q
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.claude.action_space import ClaudeDesktopActionSpace
from lite.agents.models.claude.utils.parse import parse_response_with_provenance
from lite.agents.models.gpt.action_space import GPTDesktopActionSpace
from lite.agents.models.gpt.utils.parse import parse_output_items_with_provenance
from lite.core import LiteCUAMetadata, LiteMessage, LiteSample
from lite.core.messages.image_refs import referenced_image_indices_in_message_order
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.train.export.sft_tokenize import agent_step_to_rl_step

register_all()

_GOLDEN_DIR = Path(__file__).parent / "_crossmodel_goldens"
_UPDATE = os.environ.get("UPDATE_CROSSMODEL_GOLDENS") == "1"

# Both students in scope for the tool-result contract. lite (canonical identity) is
# already covered by test_browser_crossagent; here we pin the render TARGET for the
# two Qwen SFT students the teacher trajectories distill into.
STUDENTS = ["qwen3_vl", "qwen3_5"]

# The teacher rollout resolution the parse normalizes FROM (canonical coords are
# resolution-independent after conversion, so both teachers land identically).
_RESOLUTION = (1024, 768)


def _img(k: int) -> Image.Image:
    """Deterministic tiny RGB image (distinct per index, no randomness)."""
    return Image.new("RGB", (32, 32), color=(k * 30 % 256, 0, 0))


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] * len(text.split())


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    image_token = "<|image_pad|>"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        del tokenize
        del enable_thinking
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"<{message.get('role')}>")
            for part in message.get("content") or []:
                if part.get("type") == "image":
                    chunks.append(self.image_token)
                elif part.get("type") == "text" and part.get("text"):
                    chunks.append(part["text"])
            for tool_call in message.get("tool_calls") or []:
                chunks.append(f"<tool_call>{tool_call_name(tool_call)}</tool_call>")
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Teacher raw → canonical LiteMessage (REAL parse paths, real wire grammars)
# ---------------------------------------------------------------------------

def _gpt_teacher_message(actions: list[dict[str, Any]]) -> LiteMessage:
    """A GPT teacher assistant turn: a native ``computer_call`` carrying a LIST
    of ``actions`` (the Responses-API batched shape), parsed through the REAL
    ``parse_output_items_with_provenance(...).message`` → canonical
    LiteMessage. A single-element list is the per-action baseline; a
    multi-element list becomes one canonical
    ``computer{actions:[...]}`` call."""
    items = [{"type": "computer_call", "call_id": "c1", "actions": actions}]
    return parse_output_items_with_provenance(
        items, GPTDesktopActionSpace(), _RESOLUTION,
    ).message


def _claude_teacher_message(tool_uses: list[dict[str, Any]]) -> LiteMessage:
    """A Claude teacher assistant turn: one ``computer`` ``tool_use`` block per
    action (Claude emits parallel actions as multiple content blocks), parsed
    through the REAL ``parse_response_with_provenance(...).message`` →
    canonical LiteMessage. Claude is not an adapter-side batch source, so
    parallel ``tool_use`` blocks project to flat canonical calls."""
    content = [
        {"type": "tool_use", "id": f"toolu_{i}", "name": "computer", "input": inp}
        for i, inp in enumerate(tool_uses)
    ]
    msg = SimpleNamespace(content=content, tool_calls=[], role="assistant")
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])
    # scale 1.0/1.0 — the teacher already reports env-native coords; canonical
    # conversion (down to the 1000-grid) happens in convert_tool_calls_from_agent.
    return parse_response_with_provenance(
        resp,
        1.0,
        1.0,
        ClaudeDesktopActionSpace(),
        _RESOLUTION,
    ).message


# A single ``click`` action expressed in each teacher's native wire grammar.
# GPT keys the native computer action by ``type`` + x/y; Claude by ``action`` +
# ``coordinate`` — the two dialects the bridge must canonicalize identically.
_GPT_SINGLE_CLICK = [{"type": "click", "x": 512, "y": 384}]
_CLAUDE_SINGLE_CLICK = [{"action": "left_click", "coordinate": [512, 384]}]

# A LIST of parallel actions (click THEN type) — the batched-``computer`` case.
_GPT_PARALLEL = [{"type": "click", "x": 512, "y": 384}, {"type": "type", "text": "hello"}]
_CLAUDE_PARALLEL = [
    {"action": "left_click", "coordinate": [512, 384]},
    {"action": "type", "text": "hello"},
]

TEACHERS = {
    "gpt": lambda: _gpt_teacher_message(_GPT_SINGLE_CLICK),
    "claude": lambda: _claude_teacher_message(_CLAUDE_SINGLE_CLICK),
}


# ---------------------------------------------------------------------------
# The canonical bridge: teacher LiteMessage(s) → student unroll → steps
# ---------------------------------------------------------------------------

def _tool_observation_message(previous_assistant: LiteMessage, image_index: int) -> LiteMessage:
    """Current post-action observation shape: a paired ``role:"tool"`` result."""
    tool_calls = previous_assistant.get("tool_calls") or []
    assert tool_calls, "post-action observation needs a previous tool call"
    return {
        "role": "tool",
        "tool_call_id": tool_call_id(tool_calls[-1]),
        "content": [{"type": "image", "index": image_index}],
    }


def _bridge_sample(
    teacher_msgs: list[LiteMessage],
    *,
    task_text: str = "Open GIMP and apply a filter.",
) -> LiteSample:
    """Weave teacher assistant turns into a desktop trajectory the student
    ``unroll`` consumes.

    Turn 0's user message carries the task and initial screenshot. Later
    screenshots are canonical tool-result messages paired to the previous
    assistant call via ``tool_call_id``. Each assistant turn is the
    already-canonical teacher message.
    """
    messages: list[dict[str, Any]] = []
    for i, tmsg in enumerate(teacher_msgs):
        if i == 0:
            content: list[dict[str, Any]] = [{"type": "image", "index": i}]
            content.append({"type": "text", "text": task_text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append(_tool_observation_message(teacher_msgs[i - 1], i))
        messages.append(tmsg)
    return LiteSample(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
            others={"resolution": list(_RESOLUTION)},
        ),
        images=[_img(i) for i in range(len(teacher_msgs))],
        messages=messages,
    )


def _render_steps(student: str, sample: LiteSample) -> str:
    """Unroll ``sample`` through the student adapter → ``pformat`` of the SFT
    render target. Deterministic (steps reference images by index)."""
    steps = AgentAdapterRegistry.get(f"{student}@desktop@use").unroll(sample).steps
    rendered = pformat(steps, sort_dicts=False, width=100)
    # Non-determinism guard: a raw PIL object would print a memory address.
    # steps must reference images by index only. (``0x`` also appears inside the
    # benign ``1000x1000`` resolution prose — the ``Image`` clause admits that.)
    assert "0x" not in rendered or "Image" not in rendered, (
        f"{student}: render leaks a non-deterministic object (PIL image or "
        f"address) — steps must reference images by index"
    )
    return rendered


def _golden_path(name: str) -> Path:
    return _GOLDEN_DIR / f"{name}.txt"


def _assert_golden(name: str, rendered: str) -> None:
    """File-golden compare / regenerate (mirrors test_render_characterization_goldens)."""
    path = _golden_path(name)
    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"regenerated golden {path.name}")
    assert path.exists(), (
        f"missing golden {path} — regenerate with UPDATE_CROSSMODEL_GOLDENS=1"
    )
    expected = path.read_text()
    assert rendered == expected, (
        f"CROSS-MODEL SFT DRIFT for {name}:\nthe teacher→student render target "
        f"changed vs the frozen golden. If this is an INTENTIONAL format change "
        f"(nested tool-call / role:tool contract), regenerate with "
        f"UPDATE_CROSSMODEL_GOLDENS=1 and review the diff; otherwise the "
        f"adapter/export machinery introduced cross-model distillation drift "
        f"(must be zero)."
    )


def _last_assistant(step: list[dict[str, Any]]) -> dict[str, Any]:
    return [m for m in step if m.get("role") == "assistant"][-1]


def _semantic_tool_calls(msg: LiteMessage) -> list[dict[str, Any]]:
    """Call comparison for the semantic action payload.

    Lite ids are asserted separately where ordering/cardinality matters.
    """
    return [
        {"name": tool_call_name(tc), "arguments": tool_call_arguments(tc)}
        for tc in (msg.get("tool_calls") or [])
    ]


# ===========================================================================
# Single-action cross-model round-trip baseline
# ===========================================================================

def test_gpt_and_claude_single_action_canonicalize_identically():
    """PIVOT: a GPT ``computer_call`` and a Claude ``computer`` ``tool_use`` for
    the SAME click land on the byte-identical canonical LiteMessage. This is the
    root of cross-model distillation — both teachers feed one canonical bridge."""
    gpt = _gpt_teacher_message(_GPT_SINGLE_CLICK)
    claude = _claude_teacher_message(_CLAUDE_SINGLE_CLICK)
    assert _semantic_tool_calls(gpt) == _semantic_tool_calls(claude)
    for msg in (gpt, claude):
        for tc in msg["tool_calls"]:
            assert set(tc) <= {"id", "type", "function"}
    # Sanity: a single canonical screen action is a length-1 ``computer`` wrapper.
    assert gpt["tool_calls"] == [
        make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [500, 500]}]},
            call_id="call_0000",
        )
    ]


@pytest.mark.parametrize("student", STUDENTS)
def test_gpt_teacher_to_qwen_student_roundtrip_single_action(student: str):
    """A GPT teacher response with a SINGLE action → canonical LiteSample →
    student ``unroll`` → frozen ``steps`` golden."""
    sample = _bridge_sample([_gpt_teacher_message(_GPT_SINGLE_CLICK)])
    rendered = _render_steps(student, sample)
    _assert_golden(f"gpt_to_{student}__single", rendered)


@pytest.mark.parametrize("student", STUDENTS)
def test_claude_teacher_to_qwen_student(student: str):
    """Claude teacher (per-action ``computer`` ``tool_use`` projected to the
    provider call shape) → canonical LiteSample → student ``unroll`` → frozen ``steps`` golden.
    Because the single-action canonical is teacher-identical (pinned above),
    this golden is byte-equal to the GPT one — proving one canonical SFT target
    regardless of teacher."""
    sample = _bridge_sample([_claude_teacher_message(_CLAUDE_SINGLE_CLICK)])
    rendered = _render_steps(student, sample)
    _assert_golden(f"claude_to_{student}__single", rendered)

    # Cross-teacher equivalence: the Claude render target IS the GPT render target
    # (same canonical bridge → same student SFT bytes).
    gpt_sample = _bridge_sample([_gpt_teacher_message(_GPT_SINGLE_CLICK)])
    assert rendered == _render_steps(student, gpt_sample)


# ===========================================================================
# Parallel teacher actions: GPT batches; Claude emits separate tool_use items
# ===========================================================================

@pytest.mark.parametrize("student", STUDENTS)
def test_gpt_teacher_parallel_actions_to_qwen_student(student: str):
    """A GPT ``computer_call`` with a LIST of parallel actions (click+type)
    canonicalizes to ONE batched ``computer`` action, then distills to the
    student SFT target carrying both actions in order."""
    teacher = _gpt_teacher_message(_GPT_PARALLEL)
    tcs = teacher["tool_calls"]
    # Current expectation: the parallel list batches into ONE ``computer``
    # action (not two flattened per-action tool_calls).
    assert len(tcs) == 1 and tool_call_name(tcs[0]) == "computer", (
        f"expected a single batched `computer` canonical call, got "
        f"{[tool_call_name(tc) for tc in tcs]}"
    )
    assert tool_call_id(tcs[0]) == "call_0000"
    # …and it distills to a student step carrying BOTH actions in order.
    sample = _bridge_sample([teacher])
    asst = _last_assistant(
        AgentAdapterRegistry.get(f"{student}@desktop@use").unroll(sample).steps[-1]
    )
    blob = pformat(asst)
    assert "left_click" in blob and "type" in blob


def test_examples_lite_gpt_batch_sparse_row_to_qwen3_5_sft_keeps_image_indices():
    """A GPT-batch-collected row may store internal action-batch frames that
    the student never saw.

    The source row keeps all env screenshots in ``LiteSample.images``. Its
    messages reference only the images used for decisions: the initial frame and
    the final action-batch result frame. Qwen3.5 SFT unroll/export must preserve
    that sparse index view instead of compacting image 2 down to image 1.
    """
    teacher = _gpt_teacher_message(_GPT_PARALLEL)
    [tool_call] = teacher["tool_calls"]
    sample = LiteSample(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
            others={"resolution": list(_RESOLUTION)},
        ),
        images=[_img(0), _img(1), _img(2)],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open GIMP and apply a filter."},
                ],
            },
            teacher,
            {
                "role": "tool",
                "tool_call_id": tool_call_id(tool_call),
                "content": [{"type": "image", "index": 2}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
    )

    agent_sample = AgentAdapterRegistry.get("qwen3_5@desktop@use").unroll(sample)

    assert len(agent_sample.processed_images) == 3
    assert agent_sample.processed_images[0] is not None
    assert agent_sample.processed_images[1] is None
    assert agent_sample.processed_images[2] is not None

    final_step = agent_sample.steps[-1]
    assert referenced_image_indices_in_message_order(final_step[:-1]) == (0, 2)

    rl_step = agent_step_to_rl_step(final_step, _FakeProcessor())
    assert rl_step is not None
    assert rl_step.image_indices == (0, 2)
    assert rl_step.prompt.count(_FakeProcessor.image_token) == 2


@pytest.mark.parametrize("student", STUDENTS)
def test_claude_teacher_parallel_actions_merge_to_one_batch_for_qwen_student(student: str):
    """Claude's native computer tool carries ONE action per ``tool_use`` block.

    A turn's parallel blocks are therefore a run of consecutive GUI actions —
    the same shape qwen emits as several ``computer_use`` calls — so they merge
    into ONE canonical ``computer{actions:[...]}`` batch. The student target
    still carries both actions, in order.
    """
    teacher = _claude_teacher_message(_CLAUDE_PARALLEL)
    tcs = teacher["tool_calls"]
    assert [tool_call_name(tc) for tc in tcs] == ["computer"]
    assert [tool_call_id(tc) for tc in tcs] == ["call_0000"]
    assert [a["action"] for a in tool_call_arguments(tcs[0])["actions"]] == [
        "click", "type",
    ]
    sample = _bridge_sample([teacher])
    asst = _last_assistant(
        AgentAdapterRegistry.get(f"{student}@desktop@use").unroll(sample).steps[-1]
    )
    blob = pformat(asst)
    assert "left_click" in blob and "type" in blob


# ===========================================================================
# Teacher observation screenshot → student `role:"tool"` message
# ===========================================================================

@pytest.mark.parametrize("student", STUDENTS)
def test_cross_model_screenshot_becomes_role_tool(student: str):
    """A multi-turn teacher trajectory keeps post-action screenshots as tool results.

    The source ``LiteSample`` uses the current nested contract: image index 1 is
    the env response to turn 1's tool call, represented as ``role:"tool"`` with
    the previous call's ``tool_call_id``. Student unroll must preserve that role
    so SFT stays on-distribution with the tool-calling protocol.
    """
    # 2-turn GPT teacher trajectory: click, observe, click. Image index 1 is the
    # observation resulting from turn 1's action.
    teacher_msgs = [
        _gpt_teacher_message([{"type": "click", "x": 100, "y": 100}]),
        _gpt_teacher_message([{"type": "click", "x": 200, "y": 200}]),
    ]
    sample = _bridge_sample(teacher_msgs)
    steps = AgentAdapterRegistry.get(f"{student}@desktop@use").unroll(sample).steps

    # Find the message carrying image index 1 (the post-action observation) in the
    # final step and assert it is a tool message.
    obs_roles: list[str] = []
    for m in steps[-1]:
        for part in (m.get("content") or []):
            if isinstance(part, dict) and part.get("type") == "image" and part.get("index") == 1:
                obs_roles.append(m.get("role"))
    assert obs_roles, "expected the post-action observation image in the render"
    assert all(r == "tool" for r in obs_roles), (
        f"post-action observation screenshot must be a role:'tool' message, got "
        f"roles={obs_roles}"
    )


# ===========================================================================
# Model-gated: apply_chat_template layer (runs only if a Qwen processor cached)
# ===========================================================================

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Representative student model ids (Instruct variants — no <think> stripping).
_STUDENT_MODEL_IDS = {
    "qwen3_vl": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_5": "Qwen/Qwen3.5-4B",
}


def _model_cached(model_id: str) -> bool:
    repo_dir = _HF_CACHE / f"models--{model_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    return snapshots.is_dir() and any(d.is_dir() for d in snapshots.iterdir())


@functools.lru_cache
def _load_processor(model_id: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


@pytest.mark.parametrize("student", STUDENTS)
def test_gpt_teacher_single_action_apply_chat_template(student: str):
    """Model-gated. When the student's HF processor is cached locally,
    render the single-action GPT-teacher SFT step through the REAL
    ``apply_chat_template`` and assert the native student action surfaces
    (``left_click`` in the wire form). Skips cleanly when the model isn't cached
    (CI / this hermetic env) — the ``pformat`` goldens are the primary guard."""
    model_id = _STUDENT_MODEL_IDS[student]
    if not _model_cached(model_id):
        pytest.skip(f"{model_id} not cached locally")

    sample = _bridge_sample([_gpt_teacher_message(_GPT_SINGLE_CLICK)])
    steps = AgentAdapterRegistry.get(f"{student}@desktop@use").unroll(sample).steps
    # Strip image content parts so apply_chat_template renders text-only (no image
    # processor path) — the SFT assistant target is what we assert on.
    msgs: list[dict[str, Any]] = []
    for m in steps[-1]:
        content = [
            p for p in (m.get("content") or [])
            if not (isinstance(p, dict) and p.get("type") == "image")
        ]
        msgs.append({**m, "content": content})

    processor = _load_processor(model_id)
    rendered = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert "left_click" in rendered, rendered
