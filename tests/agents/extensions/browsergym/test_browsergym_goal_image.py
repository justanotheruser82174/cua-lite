"""Unit tests for the browsergym goal-image agent (VisualWebArena).

    uv run pytest tests/agents/extensions/browsergym/test_browsergym_goal_image.py

Model-free: exercises the helpers (`_goal_image_indices`, `_persist_goal_images`)
and :meth:`VisualWebArenaGoalImageAgent._ingest_goal_images` directly with a fake
adapter (identity ``process_image``), so we test the goal-image logic — decode
into the trajectory arrays (turn-0, idempotent) and re-surface them labeled,
before the per-turn screenshot — without loading a real model adapter.

The split this file pins: goal IMAGE parts and the ``goal_image_indices``
metadata carrier are persisted; the ``Task reference image(s)`` / ``Current
screenshot:`` labels are RENDER-only. A persisted label becomes the turn-0
message's first text part, which is exactly where ``BrowserGymGenericProtocol``
reads the task instruction for its ``## Goal:`` section.
"""

from __future__ import annotations

import base64
import copy
import importlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from lite.agents.extensions.browsergym.goal_image import (
    VisualWebArenaGoalImageAgent,
    _goal_image_indices,
    _persist_goal_images,
    splice_goal_images,
    wrap_goal_image_protocol,
)
from lite.agents.extensions.browsergym.protocol import (
    BrowserGymGenericProtocol,
    BrowserGymGoalImageQwen3_5HistoryProtocol,
    BrowserGymGoalImageQwen3VLHistoryProtocol,
)
from lite.core import (
    LiteCUAMetadata,
    LiteSample,
)
from lite.core.tools import make_tool_call

_REF_LABEL = "Task reference image(s) for the instruction below:"
_OBS_LABEL = "Current screenshot:"


def _register_qwen3_vl_test_modules() -> None:
    for module_name in (
        "lite.agents.models.qwen3_vl.action_space",
        "lite.agents.models.qwen3_vl.protocol",
        "lite.agents.models.qwen3_vl.adapter",
    ):
        importlib.import_module(module_name)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


def _image_indices(message: dict) -> list[int]:
    return [
        part["index"]
        for part in message.get("content", [])
        if part.get("type") == "image"
    ]


# --- helpers: index extraction --------------------------------------------

def _goal_metadata(indices: list[int]) -> dict:
    """The turn-0 metadata part `_ingest_goal_images` records indices on."""
    return {"type": "metadata", "data": {"goal_image_indices": list(indices)}}


def test_indices_read_turn0_only():
    msgs = [
        {"role": "user", "content": [
            {"type": "image", "index": 1},
            {"type": "image", "index": 2},
            {"type": "image", "index": 0},
            {"type": "text", "text": "Find this product."},
            _goal_metadata([1, 2]),
        ]},
        {"role": "user", "content": [{"type": "image", "index": 9}, _goal_metadata([9])]},
    ]
    assert _goal_image_indices(msgs) == [1, 2]


def test_indices_keep_metadata_order_not_numeric_order():
    msgs = [
        {"role": "user", "content": [
            {"type": "image", "index": 2},
            {"type": "image", "index": 1},
            {"type": "image", "index": 0},
            {"type": "text", "text": "Find this product."},
            _goal_metadata([2, 1]),
        ]},
    ]
    assert _goal_image_indices(msgs) == [2, 1]


def test_indices_empty_when_absent():
    assert _goal_image_indices([{"role": "user", "content": [{"type": "text", "text": "x"}]}]) == []


def test_indices_empty_for_plain_first_turn_page_screenshot():
    """A turn-0 page screenshot is not a goal image. The metadata carrier is the
    only marker, so no image-prefix shape can be mistaken for a goal block."""
    msgs = [
        {"role": "user", "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "Find the cheapest hat."},
        ]},
    ]
    assert _goal_image_indices(msgs) == []


# --- helpers: persisted goal-image ordering --------------------------------

def test_persist_goal_images_orders_goal_before_page():
    msg = {
        "role": "user",
        "content": [{"type": "image", "index": 0}, {"type": "text", "text": "go"}],
    }
    _persist_goal_images(msg, [1])
    types = [(c["type"], c.get("index")) for c in msg["content"]]
    assert types == [("image", 1), ("image", 0), ("text", None)]


def test_persist_goal_images_never_writes_render_labels():
    """Closure guard. The labels are a rendering artifact; persisting one makes
    it the turn-0 message's FIRST text part, which is where
    ``BrowserGymGenericProtocol`` reads the task instruction for ``## Goal:``."""
    msg = {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "Find this product."},
            _goal_metadata([1, 2]),
        ],
    }
    _persist_goal_images(msg, [1, 2])
    texts = [c["text"] for c in msg["content"] if c["type"] == "text"]
    assert texts == ["Find this product."]
    assert _REF_LABEL not in texts
    assert _OBS_LABEL not in texts


def test_persist_goal_images_is_idempotent():
    msg = {
        "role": "user",
        "content": [{"type": "image", "index": 0}, {"type": "text", "text": "t"}],
    }
    _persist_goal_images(msg, [1, 2])
    once = copy.deepcopy(msg["content"])
    _persist_goal_images(msg, [1, 2])
    assert msg["content"] == once
    assert [c.get("index") for c in msg["content"] if c["type"] == "image"] == [1, 2, 0]


def test_persist_goal_images_reorders_existing_goal_parts_before_page():
    msg = {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "image", "index": 2},
            {"type": "text", "text": "t"},
            {"type": "image", "index": 1},
            _goal_metadata([1, 2]),
        ],
    }
    _persist_goal_images(msg, [1, 2])
    assert [c.get("index") for c in msg["content"] if c["type"] == "image"] == [1, 2, 0]
    assert _goal_image_indices([msg]) == [1, 2]


# --- protocols: extension-owned splice after protocol windowing -----------

def _goal_image_messages_for_windowing() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "image", "index": 2},
                {"type": "image", "index": 0},
                {"type": "text", "text": "Find this product."},
                _goal_metadata([1, 2]),
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click"}],
            "tool_calls": [
                make_tool_call("computer", {"actions": []}, call_id="call_0"),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [
                {"type": "image", "index": 3},
                {"type": "text", "text": "Current page."},
            ],
        },
    ]


def _first_user(messages: list[dict]) -> dict:
    return next(message for message in messages if message["role"] == "user")


def _image_sequence(messages: list[dict]) -> list[int]:
    return [
        part["index"]
        for message in messages
        for part in message.get("content", [])
        if part.get("type") == "image"
    ]


def _previous_image_for_text(messages: list[dict], text: str) -> int | None:
    previous: int | None = None
    for message in messages:
        for part in message.get("content", []):
            if part.get("type") == "image":
                previous = part["index"]
            elif part.get("type") == "text" and part.get("text") == text:
                return previous
    return None


def _next_image_for_text(messages: list[dict], text: str) -> int | None:
    seen = False
    for message in messages:
        for part in message.get("content", []):
            if part.get("type") == "text" and part.get("text") == text:
                seen = True
                continue
            if seen and part.get("type") == "image":
                return part["index"]
    return None


def test_browsergym_generic_protocol_splices_goal_images():
    out = BrowserGymGenericProtocol().process_messages(_goal_image_messages_for_windowing())
    user = _first_user(out)

    assert _image_indices(user) == [1, 2, 3]
    texts = [part["text"] for part in user["content"] if part.get("type") == "text"]
    assert texts[0] == "Task reference image(s) for the instruction below:"
    assert "Current screenshot:" in texts


def test_browsergym_generic_goal_section_renders_the_task_instruction():
    """Regression: the render labels must not displace the instruction.

    ``BrowserGymGenericProtocol`` builds ``## Goal:`` from the first text part of
    the first user message. When the goal-image label was persisted there, the
    ``## Goal:`` section rendered the label and the instruction never reached the
    model at all."""
    out = BrowserGymGenericProtocol().process_messages(_goal_image_messages_for_windowing())
    user_text = next(
        part["text"]
        for part in _first_user(out)["content"]
        if part.get("type") == "text" and "## Goal:" in part["text"]
    )
    goal_section = user_text.split("## Goal:\n", 1)[1].split("\n#", 1)[0]
    assert goal_section.strip() == "Find this product."


def test_goal_image_splice_is_idempotent_after_generic_protocol_splice():
    source = _goal_image_messages_for_windowing()
    once = BrowserGymGenericProtocol().process_messages(copy.deepcopy(source))
    twice = splice_goal_images(source, copy.deepcopy(once))

    assert twice == once
    user = _first_user(twice)
    texts = [part["text"] for part in user["content"] if part.get("type") == "text"]
    assert texts.count("Task reference image(s) for the instruction below:") == 1
    assert sum(
        1
        for message in twice
        for part in message.get("content", [])
        if part.get("type") == "text" and part.get("text") == "Current screenshot:"
    ) == 1
    assert _image_indices(user) == [1, 2, 3]


@pytest.mark.parametrize(
    ("proto", "kwargs"),
    [
        (BrowserGymGoalImageQwen3VLHistoryProtocol, {"full_history_size": 1}),
        (BrowserGymGoalImageQwen3_5HistoryProtocol, {"history_n": 1}),
    ],
)
def test_goal_image_history_protocol_splices_after_windowing(proto, kwargs):
    out = proto(**kwargs).process_messages(_goal_image_messages_for_windowing())
    user = _first_user(out)

    assert _image_indices(user) == [1, 2, 3]
    texts = [part["text"] for part in user["content"] if part.get("type") == "text"]
    assert texts[0] == "Task reference image(s) for the instruction below:"
    assert "Current screenshot:" in texts
    # The history family renders message content verbatim, so the instruction
    # must survive as its own text part rather than be replaced by the label.
    assert any("Find this product." in text for text in texts)


def test_goal_image_qwen_history_labels_latest_screenshot_under_multi_image_window():
    messages = _goal_image_messages_for_windowing()
    for turn, image_index in enumerate((4, 5), start=1):
        call_id = f"call_{turn}"
        messages.extend([
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"click {turn}"}],
                "tool_calls": [make_tool_call("computer", {"actions": []}, call_id=call_id)],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": [{"type": "image", "index": image_index}],
            },
        ])

    out = BrowserGymGoalImageQwen3VLHistoryProtocol(
        full_history_size=2,
    ).process_messages(messages)

    assert _image_sequence(out) == [1, 2, 4, 5]
    assert _next_image_for_text(out, "Current screenshot:") == 5
    assert _previous_image_for_text(out, "Current screenshot:") == 4


# --- agent: turn-0 ingest --------------------------------------------------

def _fake_agent():
    # Bypass __post_init__ (which resolves a real model adapter) — we only test
    # _ingest_goal_images, which needs self.adapter.process_image (identity here).
    agent = object.__new__(VisualWebArenaGoalImageAgent)
    agent.adapter = SimpleNamespace(process_image=lambda img: img)
    return agent


class _ImagePadProcessor:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        out: list[str] = []
        for msg in messages:
            for part in msg.get("content") or []:
                if part.get("type") == "text":
                    out.append(part.get("text", ""))
                elif part.get("type") == "image":
                    out.append("<image_pad>")
        if add_generation_prompt:
            out.append("<assistant>")
        return "\n".join(out)


class _RenderMessagesAdapter:
    _registry_key = "test"

    @classmethod
    def get_registry_key(cls) -> str:
        return "test"

    def process_image(self, img):
        return img

    def _count_turns(self, lite_sample):
        return 0

    def render_step(self, lite_sample, k, processed_images):
        return splice_goal_images(lite_sample.messages, copy.deepcopy(lite_sample.messages))

    def parse_raw_assistant_response(self, response):
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    def convert_message_from_agent(self, message):
        return message


def _turn0_sample(goal_images: list[bytes], *, with_page: bool) -> LiteSample:
    s = LiteSample(metadata=None)
    content: list[dict] = []
    if with_page:
        s.images.append(Image.new("RGB", (16, 16)))  # page screenshot at index 0
        content.append({"type": "image", "index": 0})
    content.append({"type": "text", "text": "goal"})
    content.append(
        {
            "type": "metadata",
            "data": {
                "goal_images_b64": [
                    base64.b64encode(png).decode() for png in goal_images
                ]
            },
        }
    )
    s.messages.append({"role": "user", "content": content})
    return s


@pytest.mark.asyncio
async def test_ingest_mixed_multi_image_no_page():
    agent = _fake_agent()
    s = _turn0_sample(
        [_png_bytes((255, 0, 0)), _png_bytes((0, 255, 0))],
        with_page=False,
    )
    processed: list = []
    await agent._ingest_goal_images(s, processed)
    data = s.messages[0]["content"][-1]["data"]
    assert len(s.images) == 2 and len(processed) == 2  # kept in lockstep
    assert data["goal_image_indices"] == [0, 1]
    assert _image_indices(s.messages[0]) == [0, 1]
    texts = [part["text"] for part in s.messages[0]["content"] if part.get("type") == "text"]
    assert texts == ["goal"]  # labels are render-only, never persisted
    assert "goal_images_b64" not in data  # raw base64 consumed, not persisted


@pytest.mark.asyncio
async def test_ingest_vision_after_page():
    agent = _fake_agent()
    s = _turn0_sample(
        [_png_bytes((0, 0, 255))],
        with_page=True,
    )  # page already at index 0
    processed = [s.images[0]]
    await agent._ingest_goal_images(s, processed)
    data = s.messages[0]["content"][-1]["data"]
    assert len(s.images) == 2 and len(processed) == 2
    assert data["goal_image_indices"] == [1]
    assert _image_indices(s.messages[0]) == [1, 0]  # prompt order: goal, then page
    texts = [part["text"] for part in s.messages[0]["content"] if part.get("type") == "text"]
    assert texts == ["goal"]
    assert "goal_images_b64" not in data


@pytest.mark.asyncio
async def test_predict_rl_step_image_indices_match_image_pad_prompt_order():
    captured: dict = {}

    async def generate_fn(**kwargs):
        captured.update(kwargs)
        return {"response": "done"}

    agent = object.__new__(VisualWebArenaGoalImageAgent)
    agent.adapter = _RenderMessagesAdapter()
    agent.processor = _ImagePadProcessor()
    agent.generate_fn = generate_fn
    agent.preserve_raw_response = False

    s = _turn0_sample([_png_bytes((0, 0, 255))], with_page=True)
    processed = [s.images[0]]  # page screenshot at index 0; goal appends at index 1
    result = await agent._predict_with_details(s, processed_images=processed)

    assert _image_indices(s.messages[0]) == [1, 0]
    persisted_texts = [
        part["text"]
        for part in s.messages[0]["content"]
        if part.get("type") == "text"
    ]
    assert persisted_texts == ["goal"]
    assert result.step.image_indices == (1, 0)
    assert [id(img) for img in captured["images"]] == [id(processed[1]), id(processed[0])]

    prompt = result.step.prompt
    assert prompt == captured["prompt"]
    assert prompt.count("<image_pad>") == 2
    before_goal, between_goal_and_page, after_page = prompt.split("<image_pad>")
    assert "Task reference image" in before_goal
    assert "Current screenshot:" in between_goal_and_page
    assert "goal" in after_page


@pytest.mark.asyncio
async def test_ingest_idempotent():
    agent = _fake_agent()
    s = _turn0_sample([_png_bytes((1, 2, 3))], with_page=False)
    processed: list = []
    await agent._ingest_goal_images(s, processed)
    await agent._ingest_goal_images(s, processed)  # second call: no-op
    assert len(s.images) == 1 and len(processed) == 1
    assert _image_indices(s.messages[0]) == [0]


@pytest.mark.asyncio
async def test_ingest_keeps_an_already_ingested_turn0_untouched():
    """A resumed/replayed sample already carries the indices and the image
    parts; the raw b64 is gone, so ingest is a no-op and must not re-decode."""
    agent = _fake_agent()
    s = LiteSample(metadata=None)
    s.images.append(Image.new("RGB", (8, 8)))
    s.messages.append({"role": "user", "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "goal"},
        _goal_metadata([0]),
    ]})
    processed = [s.images[0]]
    await agent._ingest_goal_images(s, processed)
    assert len(s.images) == 1 and len(processed) == 1
    assert _image_indices(s.messages[0]) == [0]
    assert _goal_image_indices(s.messages) == [0]
    texts = [part["text"] for part in s.messages[0]["content"] if part.get("type") == "text"]
    assert texts == ["goal"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_page", [True, False])  # som.yaml / mixed.yaml
async def test_ingested_turn0_renders_instruction_and_goal_images_every_turn(with_page):
    """End-to-end over the REAL ingest: ``## Goal:`` carries the instruction and
    the goal images stay the rendered image prefix on a later turn.

    ``with_page=False`` is the ``mixed.yaml`` shape (no per-step screenshot), so
    the goal images land at indices 0/1 — the case an image-prefix heuristic
    cannot distinguish from an ordinary first screenshot."""
    agent = _fake_agent()
    s = _turn0_sample([_png_bytes((255, 0, 0)), _png_bytes((0, 255, 0))], with_page=with_page)
    s.messages[0]["content"][-2]["text"] = "Find me an adapter.\n## AXTree:\nbody"
    processed = list(s.images)
    await agent._ingest_goal_images(s, processed)
    goal_indices = _goal_image_indices(s.messages)

    # ...then two more turns, so the splice is exercised past turn 0.
    for turn in range(2):
        call_id = f"call_{turn}"
        s.messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click"}],
            "tool_calls": [make_tool_call("click", {"bid": "a1"}, call_id=call_id)],
        })
        tool_content: list[dict] = []
        if with_page:
            s.images.append(Image.new("RGB", (16, 16)))
            tool_content.append({"type": "image", "index": len(s.images) - 1})
        tool_content.append({"type": "text", "text": "## AXTree:\nnewer_body"})
        s.messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": tool_content}
        )

    adapter = SimpleNamespace(protocol=BrowserGymGenericProtocol(tool_call_format="xml"))
    wrap_goal_image_protocol(adapter)
    out = adapter.protocol.process_messages(s.messages)
    user = _first_user(out)

    user_text = next(
        part["text"] for part in user["content"]
        if part.get("type") == "text" and "## Goal:" in part["text"]
    )
    goal_section = user_text.split("## Goal:\n", 1)[1].split("\n#", 1)[0]
    assert goal_section.strip() == "Find me an adapter."
    assert _image_indices(user)[:2] == goal_indices


@pytest.mark.asyncio
async def test_ingest_noop_without_goal_metadata():
    agent = _fake_agent()
    s = LiteSample(metadata=None)
    s.messages.append({"role": "user", "content": [{"type": "text", "text": "no goal image"}]})
    processed: list = []
    await agent._ingest_goal_images(s, processed)
    assert len(s.images) == 0 and len(processed) == 0


# --- adapter_key resolution (__post_init__) --------------------------------

class TestAdapterKeyResolution:
    """``adapter_key`` resolution in ``__post_init__``: a bare model slug
    auto-completes ``@{platform}@{task_type}`` from the env metadata (mirroring
    ``make``); a fully-qualified key (containing ``@``) is used verbatim;
    a ``.base`` slug selects the text+bid base family. We assert on the resolved
    adapter's action space — the behavioral contract (coord vs bid)."""

    def _action_space_name(self, adapter_key: str) -> str:
        _register_qwen3_vl_test_modules()
        meta = LiteCUAMetadata(dims=("browser", "use"))
        agent = VisualWebArenaGoalImageAgent(
            generate_fn=lambda **kw: {"response": ""},
            processor=None,
            kwargs={"adapter_key": adapter_key, "metadata": meta},
        )
        return type(agent.adapter.action_space).__name__

    def test_bare_slug_autocompletes_to_vision_coord(self):
        # "qwen3_vl" + env(browser/use) -> the Qwen3-VL desktop-coordinate
        # provider action space; nav is an env extra_tool.
        assert self._action_space_name("qwen3_vl") == "Qwen3VLDesktopActionSpace"

    def test_base_slug_selects_text_bid_family(self):
        # ".base" still selects the base adapter with the env-provided
        # browser/use suffix.
        assert self._action_space_name("qwen3_vl.base") == "BaseActionSpace"

    def test_fully_qualified_key_used_verbatim(self):
        # An explicit current @platform@task_type key resolves verbatim.
        assert self._action_space_name("qwen3_vl@browser@use") == "Qwen3VLDesktopActionSpace"

    def test_stale_web_key_is_not_supported(self):
        with pytest.raises(KeyError, match="not found"):
            self._action_space_name("qwen3_vl@web@use")

    def test_missing_adapter_key_raises(self):
        with pytest.raises(ValueError, match="adapter_key"):
            VisualWebArenaGoalImageAgent(
                generate_fn=lambda **kw: {"response": ""}, processor=None, kwargs={},
            )
