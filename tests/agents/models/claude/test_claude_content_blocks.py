"""Claude provider-boundary normalization: content blocks and visible images.

Anthropic content blocks reach CUA-Lite either as plain JSON dicts (proxy
responses) or as LiteLLM/SDK objects exposing ``model_dump()``. Both shapes go
through one reader, :func:`claude_content_blocks`, so ``text``, ``thinking``
(with its signature) and ``tool_use`` blocks reach the canonical Lite message
and the next provider request identically. A block that is neither becomes a
recorded model output error instead of a silent drop.

The provider-feedback appenders keep every env frame in the durable trajectory
while showing Claude only the final action-batch frame, and their no-tool-call
fallback branches must not smuggle an extra latest frame into a turn that
already answered provider tool calls.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel

from lite.agents.core.agent.utils.provenance import ProviderCallProvenance
from lite.agents.models.claude.action_space import (
    ClaudeDesktopActionSpace,
    ClaudeMobileActionSpace,
)
from lite.agents.models.claude.utils.history import (
    append_desktop_provider_feedback,
    append_mobile_provider_feedback,
    append_provider_assistant_message,
)
from lite.agents.models.claude.utils.parse import (
    ClaudeProviderToolUse,
    _claude_active_provider_tool_names,
    claude_content_blocks,
    parse_mobile_response_with_provenance,
    parse_response_with_provenance,
)
from lite.core import LiteSample
from lite.core.messages.final import pop_model_output_error
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvStepResult

PIXEL_6 = (1080, 2400)


# -----------------------------------------------------------------------------
# Provider fakes: LiteLLM/Anthropic SDK content blocks are pydantic objects.
# -----------------------------------------------------------------------------


class _ObjectTextBlock(BaseModel):
    type: str = "text"
    text: str


class _ObjectThinkingBlock(BaseModel):
    type: str = "thinking"
    thinking: str
    signature: str | None = None


class _ObjectToolUseBlock(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


def _response(content: Any) -> Any:
    """A minimal completion response carrying only content blocks."""
    message = SimpleNamespace(content=content, tool_calls=[], role="assistant")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model_dump=lambda: {"choices": []},
    )


def _parse_desktop(content: Any):
    return parse_response_with_provenance(
        _response(content),
        scale_x=1.0,
        scale_y=1.0,
        action_space=ClaudeDesktopActionSpace(),
        resolution=(1024, 768),
    )


# -----------------------------------------------------------------------------
# claude_content_blocks — the one place block shape is resolved
# -----------------------------------------------------------------------------


class TestClaudeContentBlocks:
    def test_dict_blocks_pass_through_in_order(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

        read = claude_content_blocks(blocks)

        assert read.blocks == tuple(blocks)
        assert read.errors == ()

    def test_object_blocks_are_read_through_model_dump(self):
        read = claude_content_blocks([_ObjectThinkingBlock(thinking="plan", signature="sig_abc")])

        assert read.errors == ()
        assert read.blocks[0]["type"] == "thinking"
        assert read.blocks[0]["thinking"] == "plan"
        assert read.blocks[0]["signature"] == "sig_abc"

    @pytest.mark.parametrize(
        "block",
        ["a bare string block", 42, SimpleNamespace(type="text", text="attrs only")],
    )
    def test_unreadable_blocks_are_reported_not_dropped(self, block):
        read = claude_content_blocks([block])

        assert read.blocks == ()
        assert len(read.errors) == 1
        assert "unreadable Claude content block" in read.errors[0]


# -----------------------------------------------------------------------------
# Parse: object-shaped blocks reach the canonical Lite message
# -----------------------------------------------------------------------------


class TestObjectContentBlockParse:
    def test_object_text_and_thinking_reach_the_lite_message(self):
        parsed = _parse_desktop(
            [
                _ObjectThinkingBlock(thinking="I will click", signature="sig_abc"),
                _ObjectTextBlock(text="Clicking the button."),
            ]
        )

        assert parsed.message["reasoning_content"] == "I will click"
        assert parsed.message["content"] == [{"type": "text", "text": "Clicking the button."}]
        assert parsed.message["tool_calls"] == []

    def test_object_tool_use_block_parses_into_a_canonical_action_batch(self):
        parsed = _parse_desktop(
            [
                _ObjectTextBlock(text="Clicking."),
                _ObjectToolUseBlock(
                    id="toolu_obj_1",
                    name="computer",
                    input={"action": "left_click", "coordinate": [512, 384]},
                ),
            ]
        )

        calls = parsed.message["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["computer"]
        assert tool_call_arguments(calls[0])["actions"][0]["action"] == "click"
        assert [tc.provider_id for tc in parsed.provider_tool_uses] == ["toolu_obj_1"]
        # The text block stays an action description because a tool call ran.
        assert parsed.message["content"] == [{"type": "action_description", "text": "Clicking."}]

    def test_mobile_object_tool_use_block_parses_into_a_mobile_batch(self):
        parsed = parse_mobile_response_with_provenance(
            _response(
                [
                    _ObjectToolUseBlock(
                        id="toolu_obj_tap",
                        name="tap",
                        input={"coordinate": [108, 240]},
                    )
                ]
            ),
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeMobileActionSpace(),
            resolution=PIXEL_6,
        )

        calls = parsed.message["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["mobile"]
        assert tool_call_arguments(calls[0])["actions"][0]["action"] == "tap"
        assert [tc.provider_id for tc in parsed.provider_tool_uses] == ["toolu_obj_tap"]

    def test_unreadable_block_becomes_a_model_output_error(self):
        parsed = _parse_desktop([SimpleNamespace(type="text", text="attrs only")])

        assert parsed.message["tool_calls"] == []
        assert "unreadable Claude content block" in pop_model_output_error(parsed.message)

    def test_unreadable_block_does_not_hide_a_valid_sibling_tool_use(self):
        parsed = _parse_desktop(
            [
                SimpleNamespace(type="text", text="attrs only"),
                _ObjectToolUseBlock(
                    id="toolu_obj_2",
                    name="computer",
                    input={"action": "screenshot"},
                ),
            ]
        )

        assert [tool_call_name(c) for c in parsed.message["tool_calls"]] == ["computer"]
        assert pop_model_output_error(parsed.message) is None


# -----------------------------------------------------------------------------
# Provider replay: the same blocks go back out on the next request
# -----------------------------------------------------------------------------


class TestObjectContentBlockReplay:
    def test_parse_keeps_object_thinking_signature_for_replay(self):
        parsed = _parse_desktop(
            [
                _ObjectThinkingBlock(thinking="I will click", signature="sig_abc"),
                _ObjectTextBlock(text="Clicking button."),
            ]
        )

        assert parsed.replay_content == [
            {"type": "thinking", "thinking": "I will click", "signature": "sig_abc"},
            {"type": "text", "text": "Clicking button."},
        ]

    def test_parse_drops_tool_use_blocks_from_replay_content(self):
        parsed = _parse_desktop(
            [
                _ObjectToolUseBlock(id="toolu_obj_3", name="computer", input={}),
                _ObjectTextBlock(text="after"),
            ]
        )

        assert parsed.replay_content == [{"type": "text", "text": "after"}]

    def test_replayed_assistant_turn_carries_object_blocks_and_tool_calls(self):
        content = [
            _ObjectThinkingBlock(thinking="I will click", signature="sig_abc"),
            _ObjectToolUseBlock(
                id="toolu_obj_4",
                name="computer",
                input={"action": "left_click", "coordinate": [512, 384]},
            ),
        ]
        parsed = _parse_desktop(content)
        completion_messages: list[dict[str, Any]] = []

        append_provider_assistant_message(
            completion_messages,
            replay_content=parsed.replay_content,
            provider_tool_uses=parsed.provider_tool_uses,
        )

        assert completion_messages == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I will click",
                        "signature": "sig_abc",
                    }
                ],
                "tool_calls": [
                    {
                        "id": "toolu_obj_4",
                        "type": "function",
                        "function": {
                            "name": "computer",
                            "arguments": ('{"action": "left_click", "coordinate": [512, 384]}'),
                        },
                    }
                ],
            }
        ]


# -----------------------------------------------------------------------------
# Model/tool-version boundary: the utils side must admit the declared version
# -----------------------------------------------------------------------------


class TestProviderVisibleResultImages:
    """Durable storage keeps every env frame; provider history shows only the
    final action-batch frame, and the no-tool-call fallback branches must not
    append a second image message on a turn that answered provider tool calls."""

    @staticmethod
    def _frames() -> list[bytes]:
        frames = []
        for color in ((10, 20, 30), (80, 120, 160)):
            buf = io.BytesIO()
            Image.new("RGB", (320, 240), color=color).save(buf, format="PNG")
            frames.append(buf.getvalue())
        return frames

    @staticmethod
    def _image_urls(messages: list[dict[str, Any]]) -> list[str]:
        return [
            block["image_url"]["url"]
            for message in messages
            for block in (message.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "image_url"
        ]

    @staticmethod
    def _sent_colors(urls: list[str]) -> list[tuple[int, int, int]]:
        return [
            Image.open(io.BytesIO(base64.b64decode(url.split("base64,", 1)[1]))).getpixel((0, 0))
            for url in urls
        ]

    def _action_batch_step(self, tool_name: str) -> tuple[list[dict[str, Any]], LiteEnvStepResult]:
        call = make_tool_call(tool_name, {"actions": [{"action": "screenshot"}]})
        call["id"] = "call_0000"
        step_result = LiteEnvStepResult(
            results=[
                LiteToolResult(
                    tool_call_id="call_0000",
                    images=self._frames(),
                    text="batched",
                )
            ]
        )
        return [call], step_result

    async def test_desktop_action_batch_stores_all_frames_and_shows_the_last(self):
        lite_tool_calls, step_result = self._action_batch_step("computer")
        trajectory = LiteSample(metadata=LiteCUAMetadata())
        completion_messages: list[dict[str, Any]] = []

        await append_desktop_provider_feedback(
            completion_messages=completion_messages,
            response=_response("done"),
            trajectory=trajectory,
            step_result=step_result,
            lite_tool_calls=lite_tool_calls,
            model_id="claude-opus-4-6",
            resize_target=None,
            many_image=True,
            provider_tool_uses=(
                ClaudeProviderToolUse(
                    provider_id="toolu_batch_1",
                    name="computer",
                    arguments={"action": "screenshot"},
                    replay_arguments='{"action": "screenshot"}',
                    source={},
                    source_type="content_tool_use",
                ),
            ),
            provider_call_provenance=(
                ProviderCallProvenance(canonical_call_id="call_0000", is_final_for_canonical=True),
            ),
            provider_errors={},
        )

        # Every env frame is stored durably...
        assert len(trajectory.images) == 2
        # ...but only the final action-batch frame is provider-visible, and the
        # ``next_sent_image_b64`` fallback added no second image message.
        urls = self._image_urls(completion_messages)
        assert len(urls) == 1
        assert self._sent_colors(urls) == [(80, 120, 160)]

    async def test_desktop_content_only_feedback_sends_text_and_image(self):
        trajectory = LiteSample(metadata=LiteCUAMetadata())
        completion_messages: list[dict[str, Any]] = []
        frame = self._frames()[0]
        step_result = LiteEnvStepResult(
            terminated=False,
            results=[
                LiteToolResult(
                    tool_call_id=None,
                    images=[frame],
                    text="attempt 1: revise the answer",
                )
            ],
        )

        image_bytes, image_index = await append_desktop_provider_feedback(
            completion_messages=completion_messages,
            response=_response("first answer"),
            trajectory=trajectory,
            step_result=step_result,
            lite_tool_calls=[],
            model_id="claude-opus-4-6",
            resize_target=None,
            many_image=True,
            provider_tool_uses=(),
            provider_call_provenance=(),
            provider_errors={},
        )

        assert image_bytes == frame
        assert image_index == 0
        assert len(trajectory.images) == 1
        assert len(completion_messages) == 1
        user_message = completion_messages[0]
        assert user_message["role"] == "user"
        assert user_message["content"][0] == {
            "type": "text",
            "text": "attempt 1: revise the answer",
        }
        assert user_message["content"][1]["type"] == "image_url"
        assert self._sent_colors([user_message["content"][1]["image_url"]["url"]]) == [
            (10, 20, 30)
        ]

    async def test_desktop_content_only_feedback_omits_empty_provider_text(self):
        trajectory = LiteSample(metadata=LiteCUAMetadata())
        completion_messages: list[dict[str, Any]] = []
        step_result = LiteEnvStepResult(
            terminated=False,
            results=[LiteToolResult(tool_call_id=None, text="")],
        )

        image_bytes, image_index = await append_desktop_provider_feedback(
            completion_messages=completion_messages,
            response=_response("first answer"),
            trajectory=trajectory,
            step_result=step_result,
            lite_tool_calls=[],
            model_id="claude-opus-4-6",
            resize_target=None,
            many_image=True,
            provider_tool_uses=(),
            provider_call_provenance=(),
            provider_errors={},
        )

        assert image_bytes is None
        assert image_index is None
        assert completion_messages == []
        assert trajectory.messages == [
            {"role": "user", "content": [{"type": "text", "text": ""}]}
        ]

    async def test_desktop_content_only_feedback_omits_empty_text_with_image(self):
        trajectory = LiteSample(metadata=LiteCUAMetadata())
        completion_messages: list[dict[str, Any]] = []
        frame = self._frames()[0]
        step_result = LiteEnvStepResult(
            terminated=False,
            results=[LiteToolResult(tool_call_id=None, images=[frame], text="")],
        )

        image_bytes, image_index = await append_desktop_provider_feedback(
            completion_messages=completion_messages,
            response=_response("first answer"),
            trajectory=trajectory,
            step_result=step_result,
            lite_tool_calls=[],
            model_id="claude-opus-4-6",
            resize_target=None,
            many_image=True,
            provider_tool_uses=(),
            provider_call_provenance=(),
            provider_errors={},
        )

        assert image_bytes == frame
        assert image_index == 0
        user_message = completion_messages[0]
        assert user_message["role"] == "user"
        assert [block["type"] for block in user_message["content"]] == ["image_url"]
        assert trajectory.messages == [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": ""},
                ],
            }
        ]

    async def test_mobile_action_batch_stores_all_frames_and_shows_the_last(self):
        lite_tool_calls, step_result = self._action_batch_step("mobile")
        trajectory = LiteSample(metadata=LiteCUAMetadata())
        completion_messages: list[dict[str, Any]] = []

        await append_mobile_provider_feedback(
            completion_messages=completion_messages,
            response=_response("done"),
            trajectory=trajectory,
            step_result=step_result,
            lite_tool_calls=lite_tool_calls,
            model_id="claude-opus-4-6",
            resolution=None,
            many_image=True,
            provider_tool_uses=(
                ClaudeProviderToolUse(
                    provider_id="toolu_batch_1",
                    name="tap",
                    arguments={"coordinate": [10, 20]},
                    replay_arguments='{"coordinate": [10, 20]}',
                    source={},
                    source_type="content_tool_use",
                ),
            ),
            provider_call_provenance=(
                ProviderCallProvenance(canonical_call_id="call_0000", is_final_for_canonical=True),
            ),
            provider_errors={},
        )

        assert len(trajectory.images) == 2
        # ``result_content`` (the no-tool-call fallback) must not also be sent.
        urls = self._image_urls(completion_messages)
        assert len(urls) == 1
        assert self._sent_colors(urls) == [(80, 120, 160)]

    async def test_mobile_content_only_feedback_omits_empty_text_with_image(self):
        trajectory = LiteSample(metadata=LiteCUAMetadata())
        completion_messages: list[dict[str, Any]] = []
        frame = self._frames()[0]
        step_result = LiteEnvStepResult(
            terminated=False,
            results=[LiteToolResult(tool_call_id=None, images=[frame], text="")],
        )

        image_bytes, image_index = await append_mobile_provider_feedback(
            completion_messages=completion_messages,
            response=_response("first answer"),
            trajectory=trajectory,
            step_result=step_result,
            lite_tool_calls=[],
            model_id="claude-opus-4-6",
            resolution=None,
            many_image=True,
            provider_tool_uses=(),
            provider_call_provenance=(),
            provider_errors={},
        )

        assert image_bytes == frame
        assert image_index == 0
        user_message = completion_messages[0]
        assert user_message["role"] == "user"
        assert [block["type"] for block in user_message["content"]] == ["image_url"]
        assert trajectory.messages == [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": ""},
                ],
            }
        ]


class TestActiveNativeToolNames:
    """``_get_tool_config_for_model`` picks a computer tool version; the parser's
    admission reader must recognize whatever version reached the request, or an
    explicitly overridden version would make ``computer`` an undeclared tool."""

    @pytest.mark.parametrize(
        "tool_version",
        ["computer_20241022", "computer_20250124", "computer_20251124", "computer_20990101"],
    )
    def test_any_declared_computer_tool_version_admits_the_computer_tool(self, tool_version):
        tools = [
            {
                "type": tool_version,
                "function": {
                    "name": "computer",
                    "parameters": {"display_width_px": 1024, "display_height_px": 768},
                },
            }
        ]

        assert _claude_active_provider_tool_names(tools) == frozenset({"computer"})

    def test_request_without_the_computer_tool_rejects_a_computer_call(self):
        parsed = parse_response_with_provenance(
            _response(
                [
                    _ObjectToolUseBlock(
                        id="toolu_obj_5",
                        name="computer",
                        input={"action": "screenshot"},
                    )
                ]
            ),
            scale_x=1.0,
            scale_y=1.0,
            action_space=ClaudeDesktopActionSpace(),
            resolution=(1024, 768),
            active_provider_tool_names=frozenset(),
        )

        assert parsed.message["tool_calls"] == []
        assert pop_model_output_error(parsed.message) == "undeclared tool_call computer"
        assert "toolu_obj_5" in parsed.provider_errors
