"""Characterization tests for GPTMobileUseAgent.

Parallel to test_claude_mobile_agent.py but asserting GPT Responses-API
specifics:
  - `tools[i].type == "function"` on every tool (no `computer` type)
  - the native GUI tools are provider-flat functions such as `tap` and `swipe`
  - Input images carry `detail: "original"`; the mobile base prompt lives in
    `system_prompt` with mobile pixel-coordinate guidance

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_mobile_agent.py -v
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents.models._support.provider_fakes import png_bytes
from PIL import Image

from lite.agents.core.agent import AgentRegistry
from lite.agents.models.gpt.agent import (
    GPT_MOBILE_API_KWARGS_DEFAULTS,
    GPTMobileUseAgent,
)
from lite.agents.models.gpt.utils.responses import ResponseAPIError
from lite.core import LiteCUAMetadata
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteFinishToolSet, LiteShellToolSet
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

# The production bash schema, so env-side admission in these tests is the real one:
# it requires ``command``.
_BASH_SCHEMA = LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)

PIXEL_6 = (1080, 2400)
NATIVE_MOBILE_TOOLS = {
    "tap",
    "long_press",
    "swipe",
    "drag",
    "pinch",
    "type",
    "system_button",
    "wait",
    "screenshot",
}


@pytest.fixture(autouse=True)
def _stub_gpt_mobile_echoed_dim_fetch(monkeypatch):
    """Keep GPT mobile sample tests hermetic after the mocked provider response."""
    monkeypatch.setattr(
        "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
        AsyncMock(return_value=[PIXEL_6]),
    )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _fake_response(output: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if output is None:
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ]
    return {"output": output, "id": "resp_test", "usage": {}}


async def _tools_sent(agent, monkeypatch, env: Any = None) -> list[dict[str, Any]]:
    """The provider tool list ``agent.sample()`` actually puts on the wire.

    The mobile sample loop hands its assembled tool list to ``litellm.aresponses``
    unmodified, so the mocked request payload is the public read of the agent's
    advertised tool surface — no private tool-assembly helper needed. An empty
    tool list is sent as ``tools=None``, normalized back to ``[]`` here.
    """
    mock = AsyncMock(return_value=_fake_response())
    monkeypatch.setattr("litellm.aresponses", mock)
    await agent.sample(
        env if env is not None else _FakeMobileEnv(terminate_after=1), max_steps=2
    )
    return mock.call_args.kwargs["tools"] or []


def _fake_env_action_name(action: dict[str, Any]) -> str | None:
    return tool_call_name(action)


def _fake_env_action_id(action: dict[str, Any]) -> str | None:
    return tool_call_id(action)


def _colored_png_bytes(
    color: tuple[int, int, int],
    w: int = 1080,
    h: int = 2400,
) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _image_rgb(image: Image.Image) -> tuple[int, int, int]:
    return image.convert("RGB").getpixel((0, 0))


class _FakeMobileEnv:
    def __init__(self, terminate_after: int = 1):
        self.metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
            others={"resolution": [1080, 2400]},
        )
        self._shot = png_bytes(1080, 2400)
        self._step_count = 0
        self._terminate_after = terminate_after

    async def reset(self):
        return LiteEnvObservation(image=self._shot, text="open settings")

    async def step(self, actions):
        self._step_count += 1
        finish = any(
            _fake_env_action_name(action) in {"response", "terminate"} for action in actions
        )
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=_fake_env_action_id(action),
                    images=[self._shot],
                    text="open settings",
                )
                for action in actions
            ],
        )

    async def close(self):
        pass


class _RecordingFakeMobileEnv(_FakeMobileEnv):
    def __init__(self, terminate_after: int = 1):
        super().__init__(terminate_after=terminate_after)
        self.actions_seen: list[list[dict[str, Any]]] = []

    async def step(self, actions):
        self.actions_seen.append(actions)
        return await super().step(actions)


class _RejectEmptyActionsMobileEnv(_RecordingFakeMobileEnv):
    async def step(self, actions):
        if actions == []:
            raise AssertionError("content-only final must not call env.step([])")
        result = await super().step(actions)
        return result


class _PerCallTextMobileEnv(_RecordingFakeMobileEnv):
    async def step(self, actions):
        self.actions_seen.append(actions)
        self._step_count += 1
        finish = any(
            _fake_env_action_name(action) in {"response", "terminate"} for action in actions
        )
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=_fake_env_action_id(action),
                    images=[self._shot],
                    text=f"result {_fake_env_action_id(action)}",
                )
                for action in actions
            ],
        )


class _MultiImageMobileEnv(_FakeMobileEnv):
    def __init__(self, terminate_after: int = 1):
        super().__init__(terminate_after=terminate_after)
        self._result_shots = [
            _colored_png_bytes((170, 30, 30)),
            _colored_png_bytes((30, 170, 30)),
            _colored_png_bytes((30, 30, 170)),
        ]

    async def step(self, actions):
        self._step_count += 1
        finish = any(
            _fake_env_action_name(action) in {"response", "terminate"} for action in actions
        )
        done = finish or self._step_count >= self._terminate_after
        return LiteEnvStepResult(
            reward=1.0 if done else 0.0,
            terminated=done,
            results=[
                LiteToolResult(
                    tool_call_id=_fake_env_action_id(action),
                    images=list(self._result_shots),
                    text="multi mobile obs",
                )
                for action in actions
            ],
        )


# -----------------------------------------------------------------------------
# Tool inventory & strict compliance
# -----------------------------------------------------------------------------


class TestMobileToolInventory:
    async def test_every_tool_is_function_type(self, monkeypatch):
        tools = await _tools_sent(GPTMobileUseAgent(), monkeypatch)
        for t in tools:
            assert t["type"] == "function", t

    async def test_no_computer_tool_present(self, monkeypatch):
        tools = await _tools_sent(GPTMobileUseAgent(), monkeypatch)
        assert not any(t["type"] == "computer" for t in tools)

    async def test_mobile_tool_schema_contains_provider_flat_mobile_tools(self, monkeypatch):
        tools = await _tools_sent(GPTMobileUseAgent(), monkeypatch)
        assert {t["name"] for t in tools} == NATIVE_MOBILE_TOOLS
        tap = next(t for t in tools if t["name"] == "tap")
        swipe = next(t for t in tools if t["name"] == "swipe")
        assert tap["parameters"]["required"] == ["x", "y"]
        assert swipe["parameters"]["required"] == ["start_x", "start_y", "end_x", "end_y"]

    async def test_default_strict_applied_only_to_compliant_tools(self, monkeypatch):
        """Provider-flat tools with only required fields can be strict; optional-field
        tools still get additionalProperties disabled."""
        tools = await _tools_sent(GPTMobileUseAgent(), monkeypatch)
        by_name = {t["name"]: t for t in tools}
        assert "strict" not in by_name["tap"]
        assert by_name["type"]["strict"] is True
        assert "strict" not in by_name["long_press"]
        for tool in tools:
            assert tool["parameters"].get("additionalProperties") is False

    async def test_kwarg_strict_false_omits_strict_on_all_tools(self, monkeypatch):
        tools = await _tools_sent(GPTMobileUseAgent(function_tool_strict=False), monkeypatch)
        for t in tools:
            assert "strict" not in t, t["name"]

    def test_action_space_key_is_not_a_gpt_mobile_runtime_knob(self):
        with pytest.raises(TypeError, match="action_space_key"):
            GPTMobileUseAgent(action_space_key="nonsense")

    async def test_valid_actions_filters_provider_flat_mobile_tools(self, monkeypatch):
        """``valid_actions`` filters the provider-flat native mobile tool list."""
        all_tools = await _tools_sent(GPTMobileUseAgent(), monkeypatch)
        all_names = {t["name"] for t in all_tools}
        assert all_names == NATIVE_MOBILE_TOOLS
        assert "terminate" not in all_names  # finish tools are schema-backed extras
        a = GPTMobileUseAgent(
            metadata=LiteCUAMetadata(valid_actions=["tap", "swipe"]),
        )
        tools = await _tools_sent(a, monkeypatch)
        assert {t["name"] for t in tools} == {"tap", "swipe"}

    async def test_empty_valid_actions_drops_mobile_but_preserves_extra_tools(self, monkeypatch):
        a = GPTMobileUseAgent(
            metadata=LiteCUAMetadata(
                valid_actions=[],
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
            )
        )

        tools = await _tools_sent(a, monkeypatch)

        assert {t["name"] for t in tools} == {"response"}

    async def test_open_app_is_extra_tool_not_native(self, monkeypatch):
        """open_app is surfaced via env extra_tools (make_open_app_tool → app-name
        enum), NOT a native mobile action (mirrors qwen). Removing it from the native
        set keeps it out of the declared action schemas, so extra_tools=["open_app"] doesn't
        trip the collision guard."""
        from lite.core.tools.extra_tools import make_open_app_tool

        # not a native tool
        native_tools = await _tools_sent(GPTMobileUseAgent(), monkeypatch)
        assert "open_app" not in {t["name"] for t in native_tools}
        # via extra_tools: appears with the env catalog as the app_name enum, no collision
        apps = ["Chrome", "Settings", "SMS"]
        a = GPTMobileUseAgent(
            metadata=LiteCUAMetadata(
                others={"apps": apps},
                extra_tool_schemas=[make_open_app_tool(apps)],
            )
        )
        oa = next(t for t in await _tools_sent(a, monkeypatch) if t["name"] == "open_app")
        assert "function" not in oa
        assert oa["parameters"]["properties"]["app_name"]["enum"] == apps


# -----------------------------------------------------------------------------
# System prompt default
# -----------------------------------------------------------------------------


class TestMobileSystemPromptDefault:
    def test_default_base_prompt_is_lean_policy(self):
        """The base prompt lives in ``system_prompt`` and carries only
        cross-cutting POLICY (coordinate frame, verify/anti-loop, completion,
        delegated-task scope) — NOT per-tool mechanics, which live in the tool
        descriptions. It is benchmark-AGNOSTIC: no mobilegym "simulated" override."""
        a = GPTMobileUseAgent()
        sp = a._effective_system_prompt()
        assert "mobile phone" in sp
        assert "terminate" not in sp and "response" not in sp
        # {w}/{h} str.format fields present (filled with sent dims at sample-time).
        assert "{w}" in sp and "{h}" in sp
        # benchmark-agnostic: the mobilegym research-only override is NOT baked in.
        assert "simulated" not in sp.lower()
        # tool mechanics must NOT be restated in the prompt (they live in the tool
        # descriptions) — e.g. no NAVIGATION how-to / "aim for center".
        assert "open_app" not in sp and "aim for the center" not in sp.lower()

    def test_user_can_override_system_prompt(self):
        a = GPTMobileUseAgent(system_prompt="CUSTOM BASE")
        assert a._effective_system_prompt() == "CUSTOM BASE"


# -----------------------------------------------------------------------------
# __post_init__ merges
# -----------------------------------------------------------------------------


class TestPostInitMerge:
    def test_partial_config_merged(self):
        a = GPTMobileUseAgent(api_kwargs={"max_output_tokens": 8192})
        assert a.api_kwargs["max_output_tokens"] == 8192
        assert a.function_tool_strict is True
        assert "mobile phone" in a._effective_system_prompt()

    def test_empty_api_kwargs_all_defaults(self):
        a = GPTMobileUseAgent(api_kwargs={})
        for k, v in GPT_MOBILE_API_KWARGS_DEFAULTS.items():
            assert a.api_kwargs[k] == v


# -----------------------------------------------------------------------------
# Sample loop end-to-end properties
# -----------------------------------------------------------------------------


class TestSampleLoopMobilePath:
    async def test_max_steps_exhaustion_marks_truncated_with_paired_feedback(self, monkeypatch):
        tap_item = {
            "type": "function_call",
            "call_id": "c1",
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        mock = AsyncMock(return_value=_fake_response([tap_item]))
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent()
        result = await agent.sample(_RecordingFakeMobileEnv(terminate_after=99), max_steps=1)

        assert result.terminated is False
        assert result.truncated is True
        assert result.steps[-1].status == "truncated"
        assert mock.call_count == 1
        assert [m["role"] for m in result.lite_sample.messages] == ["user", "assistant", "tool"]
        tool_msg = result.lite_sample.messages[-1]
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 1},
            {"type": "text", "text": "open settings"},
        ]

    async def test_fail_fast_raises_on_failed_status(self, monkeypatch):
        failed = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "not a final answer"}],
                }
            ],
            "status": "failed",
            "error": {"message": "oops"},
        }
        mock = AsyncMock(return_value=failed)
        monkeypatch.setattr("litellm.aresponses", mock)

        with pytest.raises(ResponseAPIError):
            await GPTMobileUseAgent().sample(_FakeMobileEnv(), max_steps=3)

    def test_fail_fast_on_api_error_config_is_rejected(self):
        with pytest.raises(TypeError, match="fail_fast_on_api_error"):
            GPTMobileUseAgent(fail_fast_on_api_error=False)

    def test_preserve_raw_response_is_not_a_gpt_mobile_runtime_knob(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            GPTMobileUseAgent(preserve_raw_response=True)

    def test_registry_rejects_stale_preserve_raw_response(self):
        with pytest.raises(TypeError, match="preserve_raw_response"):
            AgentRegistry.get("gpt@mobile@use", preserve_raw_response=True)

    def test_registry_rejects_stale_action_space_key(self):
        with pytest.raises(TypeError, match="action_space_key"):
            AgentRegistry.get("gpt@mobile@use", action_space_key="gpt@mobile")

    def test_registry_rejects_unknown_gpt_mobile_config(self):
        with pytest.raises(TypeError, match="unknown_mobile_config"):
            AgentRegistry.get("gpt@mobile@use", unknown_mobile_config=True)

    async def test_request_tools_all_function(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent()
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        tools = mock.call_args.kwargs["tools"]
        assert tools
        for t in tools:
            assert t["type"] == "function"
        by_name = {t["name"]: t for t in tools}
        assert set(by_name) == NATIVE_MOBILE_TOOLS
        assert "strict" not in by_name["tap"]
        assert "strict" not in by_name["long_press"]
        assert by_name["tap"]["parameters"].get("additionalProperties") is False
        assert by_name["long_press"]["parameters"].get("additionalProperties") is False

    async def test_request_images_carry_detail_original(self, monkeypatch):
        """Mobile screenshots use detail=original (gpt-5.5) for best accuracy."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent()
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        input_items = mock.call_args.kwargs["input"]
        # Find input_image blocks in user content lists
        details: list[str] = []
        for item in input_items:
            c = item.get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "input_image":
                        details.append(b.get("detail", ""))
        assert details, "expected at least one input_image in first request"
        assert all(d == "original" for d in details)

    async def test_system_prompt_sent_by_default(self, monkeypatch):
        """By default (``system_prompt_in_input``) the system prompt rides as a
        persistent ``developer`` message in ``input`` — NOT the one-shot
        ``instructions`` field — so it survives ``previous_response_id`` chaining."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent()
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        input_items = mock.call_args.kwargs["input"]
        dev = [it for it in input_items if it.get("role") == "developer"]
        assert dev, "expected a developer system-prompt message in input"
        content = dev[0]["content"]
        assert "mobile phone" in content
        assert "terminate" not in content
        # {w}/{h} must be RESOLVED to the sent frame dims (FakeMobileEnv is
        # 1080×2400, agent.resolution=None → no resize). A regression dropping
        # the .format() would ship a literal placeholder.
        assert "{w}" not in content and "{h}" not in content
        assert "1080×2400" in content
        # the one-shot top-level field is unused in this mode
        assert not mock.call_args.kwargs.get("instructions")
        user_items = [it for it in input_items if it.get("role") == "user"]
        assert len(user_items) == 1
        assert user_items[0]["content"][0] == {"type": "input_text", "text": "open settings"}
        assert user_items[0]["content"][1]["type"] == "input_image"

    async def test_finish_extra_tools_restore_mobile_completion_guidance(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        metadata = LiteCUAMetadata(extra_tool_schemas=[
                LiteFinishToolSet.get_tool_schema("response"),
                LiteFinishToolSet.get_tool_schema("terminate"),
            ])
        agent = GPTMobileUseAgent(metadata=metadata)
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        dev = [it for it in mock.call_args.kwargs["input"] if it.get("role") == "developer"]
        assert dev
        content = dev[0]["content"]
        assert (
            "When the task is done, call `terminate` "
            "(use `response` to return an answer the task asks for)."
        ) in content

    async def test_custom_system_prompt_is_augmented_by_finish_tools(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        metadata = LiteCUAMetadata(extra_tool_schemas=[
                LiteFinishToolSet.get_tool_schema("response"),
                LiteFinishToolSet.get_tool_schema("terminate"),
            ])
        agent = GPTMobileUseAgent(metadata=metadata, system_prompt="CUSTOM BASE")
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        dev = [it for it in mock.call_args.kwargs["input"] if it.get("role") == "developer"]
        assert dev
        assert dev[0]["content"].startswith("CUSTOM BASE\n\nWhen the task is done")
        assert "use `response` to return an answer" in dev[0]["content"]

    async def test_mobile_completion_guidance_matches_active_finish_surface(self, monkeypatch):
        async def first_prompt_for(*schema_names: str) -> str:
            mock = AsyncMock(return_value=_fake_response())
            monkeypatch.setattr("litellm.aresponses", mock)
            metadata = LiteCUAMetadata(extra_tool_schemas=[
                    LiteFinishToolSet.get_tool_schema(name) for name in schema_names
                ])
            await GPTMobileUseAgent(metadata=metadata).sample(
                _FakeMobileEnv(terminate_after=1), max_steps=3
            )
            dev = [it for it in mock.call_args.kwargs["input"] if it.get("role") == "developer"]
            return dev[0]["content"]

        response_only = await first_prompt_for("response")
        assert "call `response`" in response_only
        assert "call `terminate`" not in response_only

        terminate_only = await first_prompt_for("terminate")
        assert "When the task is done, call `terminate`." in terminate_only
        assert "call `response`" not in terminate_only

    async def test_system_prompt_in_instructions_when_opted_out(self, monkeypatch):
        """``system_prompt_in_input: false`` restores the one-shot ``instructions``
        field on turn 0 (no developer message in input)."""
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(api_kwargs={"system_prompt_in_input": False})
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        instructions = mock.call_args.kwargs.get("instructions", "")
        assert "mobile phone" in instructions
        assert "terminate" not in instructions
        # placeholders resolved on the instructions path too
        assert "{w}" not in instructions and "{h}" not in instructions
        assert "1080×2400" in instructions
        assert not [it for it in mock.call_args.kwargs["input"] if it.get("role") == "developer"]

    async def test_custom_system_prompt_allows_literal_json_braces(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(
            system_prompt='Return JSON like {"ok": true} on the {w}×{h} screen.'
        )
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        dev = [it for it in mock.call_args.kwargs["input"] if it.get("role") == "developer"]
        assert dev
        assert '{"ok": true}' in dev[0]["content"]
        assert "1080×2400" in dev[0]["content"]

    async def test_custom_system_prompt_literal_json_braces_in_top_level_instructions(
        self,
        monkeypatch,
    ):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(
            system_prompt='Return JSON like {"ok": true} on the {w}×{h} screen.',
            api_kwargs={"system_prompt_in_input": False},
        )
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        instructions = mock.call_args.kwargs.get("instructions", "")
        assert '{"ok": true}' in instructions
        assert "{w}" not in instructions and "{h}" not in instructions
        assert "1080×2400" in instructions
        assert not [it for it in mock.call_args.kwargs["input"] if it.get("role") == "developer"]

    async def test_top_level_instructions_resend_when_unchained(self, monkeypatch):
        tap_item = {
            "type": "function_call",
            "call_id": "c1",
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        mock = AsyncMock(side_effect=[_fake_response([tap_item]), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(
            api_kwargs={
                "system_prompt_in_input": False,
                "chain_previous_response": False,
            }
        )
        await agent.sample(_FakeMobileEnv(terminate_after=2), max_steps=3)

        assert mock.call_count == 2
        for call in mock.call_args_list:
            assert "mobile phone" in call.kwargs.get("instructions", "")
            assert call.kwargs.get("previous_response_id") is None
            assert not [it for it in call.kwargs["input"] if it.get("role") == "developer"]

        first_input = mock.call_args_list[0].kwargs["input"]
        second_input = mock.call_args_list[1].kwargs["input"]

        first_user_items = [item for item in first_input if item.get("role") == "user"]
        assert len(first_user_items) == 1
        assert first_user_items[0]["content"][0] == {
            "type": "input_text",
            "text": "open settings",
        }
        assert first_user_items[0]["content"][1]["type"] == "input_image"
        assert not [item for item in first_input if item.get("type") == "function_call_output"]

        second_user_items = [item for item in second_input if item.get("role") == "user"]
        assert len(second_user_items) == 1
        assert second_user_items[0]["content"][0] == {
            "type": "input_text",
            "text": "open settings",
        }
        assert second_user_items[0]["content"][1]["type"] == "input_image"
        second_function_call_ids = [
            item.get("call_id") for item in second_input if item.get("type") == "function_call"
        ]
        assert second_function_call_ids == ["c1"]
        assert [
            item.get("call_id")
            for item in second_input
            if item.get("type") == "function_call_output"
        ] == ["c1"]

    async def test_prompt_cache_key_is_stable_across_turns(self, monkeypatch):
        tap_item = {
            "type": "function_call",
            "call_id": "c1",
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        mock = AsyncMock(side_effect=[_fake_response([tap_item]), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        await GPTMobileUseAgent().sample(_FakeMobileEnv(terminate_after=2), max_steps=3)

        cache_keys = [
            call.kwargs.get("extra_body", {}).get("prompt_cache_key")
            for call in mock.call_args_list
        ]
        assert len(cache_keys) == 2
        assert cache_keys[0]
        assert cache_keys[0] == cache_keys[1]
        assert cache_keys[0].startswith("cua-lite-")

    async def test_mobile_function_call_routes_through_action_space(self, monkeypatch):
        """A function_call with name='tap' must be converted through
        GPTMobileActionSpace, yielding normalized coords (~500, 500 center).

        Default agent.resolution=None → no resize, model sees the env-native
        1080×2400 frame. Center coord is (540, 1200).
        """
        tap_item = {
            "type": "function_call",
            "call_id": "c1",
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        r1 = _fake_response([tap_item])
        r2 = _fake_response()
        mock = AsyncMock(side_effect=[r1, r2])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent()
        result = await agent.sample(_FakeMobileEnv(terminate_after=2), max_steps=5)

        asst_messages = [m for m in result.lite_sample.messages if m.get("role") == "assistant"]
        assert asst_messages
        first = asst_messages[0]
        assert "raw_response" not in first
        tool_calls = first.get("tool_calls", [])
        assert tool_calls
        tc = tool_calls[0]
        assert tool_call_id(tc) == "call_0000"
        assert "tool_call_id" not in tc
        assert tool_call_name(tc) == "mobile"
        action = tool_call_arguments(tc)["actions"][0]
        assert action["action"] == "tap"
        coord = action.get("coordinate")
        assert coord is not None
        # 540/1080 = 0.5 → ~500 in CUA-lite normalized [0, 1000]
        assert 499 <= coord[0] <= 501
        assert 499 <= coord[1] <= 501

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "c1"

    async def test_second_turn_provider_history_uses_provider_id_but_canonical_result_lookup(
        self, monkeypatch
    ):
        provider_id = "provider_tap_1"
        tap_item = {
            "type": "function_call",
            "call_id": provider_id,
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        mock = AsyncMock(side_effect=[_fake_response([tap_item]), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent()
        result = await agent.sample(_PerCallTextMobileEnv(terminate_after=2), max_steps=3)

        canonical_id = "call_0000"
        assert provider_id != canonical_id
        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert [tool_call_id(call) for call in assistant["tool_calls"]] == [canonical_id]
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == ["mobile"]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in outputs] == [provider_id]
        output_texts = [
            block["text"] for block in outputs[0]["output"] if block.get("type") == "input_text"
        ]
        assert output_texts == [f"result {canonical_id}"]

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == canonical_id
        assert {"type": "text", "text": f"result {canonical_id}"} in tool_msg["content"]

    async def test_multi_image_tool_result_preserves_all_but_sends_last_to_provider(
        self,
        monkeypatch,
    ):
        provider_id = "provider_tap_1"
        tap_item = {
            "type": "function_call",
            "call_id": provider_id,
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        mock = AsyncMock(side_effect=[_fake_response([tap_item]), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _MultiImageMobileEnv(terminate_after=99)

        class _Hook:
            def __init__(self):
                self.current_image_indices = []

            def on_step(self, data):
                self.current_image_indices.append(data.current_image_index)

            def on_complete(self, result):
                del result

        hook = _Hook()
        result = await GPTMobileUseAgent().sample(env, max_steps=3, hooks=[hook])

        assert [_image_rgb(image) for image in result.lite_sample.images] == [
            (255, 255, 255),
            (170, 30, 30),
            (30, 170, 30),
            (30, 30, 170),
        ]
        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 3)]
        assert hook.current_image_indices == [0, 3]
        assert len(result.processed_images) == len(result.lite_sample.images)
        assert result.processed_images[0] is result.lite_sample.images[0]
        assert result.processed_images[1:3] == [None, None]
        assert result.processed_images[3] is result.lite_sample.images[3]

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 3},
            {"type": "text", "text": "multi mobile obs"},
        ]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in outputs] == [provider_id]
        output_images = [
            block for block in outputs[0]["output"] if block.get("type") == "input_image"
        ]
        assert len(output_images) == 1
        payload = output_images[0]["image_url"].split("base64,", 1)[1]
        assert base64.b64decode(payload) == env._result_shots[-1]
        sent_payloads = [
            json.dumps(call.kwargs["input"], ensure_ascii=False, default=str)
            for call in mock.call_args_list
        ]
        assert all("_cua_lite_image_index" not in payload for payload in sent_payloads)
        assert all("_cua_lite_image_index" not in step.prompt for step in result.steps)

    async def test_multi_image_extra_tool_result_keeps_all_model_visible_images(
        self,
        monkeypatch,
    ):
        extra = make_tool_schema("visual_extra", description="Return visual feedback.")
        provider_id = "provider_visual_1"
        visual_item = {
            "type": "function_call",
            "call_id": provider_id,
            "name": "visual_extra",
            "arguments": "{}",
        }
        mock = AsyncMock(side_effect=[_fake_response([visual_item]), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _MultiImageMobileEnv(terminate_after=99)
        result = await GPTMobileUseAgent(
            metadata=LiteCUAMetadata(extra_tool_schemas=[extra]),
        ).sample(env, max_steps=3)

        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 1, 2, 3)]
        assert result.processed_images == result.lite_sample.images

        tool_msg = next(m for m in result.lite_sample.messages if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_0000"
        assert tool_msg["content"] == [
            {"type": "image", "index": 1},
            {"type": "image", "index": 2},
            {"type": "image", "index": 3},
            {"type": "text", "text": "multi mobile obs"},
        ]

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in outputs] == [provider_id]
        output_images = [
            block for block in outputs[0]["output"] if block.get("type") == "input_image"
        ]
        assert [
            base64.b64decode(block["image_url"].split("base64,", 1)[1]) for block in output_images
        ] == env._result_shots

    async def test_provider_flat_mobile_calls_all_get_function_outputs(self, monkeypatch):
        open_app_tool = make_tool_schema(
            "open_app",
            description="Open an app.",
            parameters={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
        )
        first_output = [
            {
                "type": "function_call",
                "call_id": "provider_tap",
                "name": "tap",
                "arguments": '{"x": 540, "y": 1200}',
            },
            {
                "type": "function_call",
                "call_id": "provider_swipe",
                "name": "swipe",
                "arguments": '{"start_x": 540, "start_y": 1600, "end_x": 540, "end_y": 800}',
            },
            {
                "type": "function_call",
                "call_id": "provider_open",
                "name": "open_app",
                "arguments": '{"app_name": "Settings"}',
            },
            {
                "type": "function_call",
                "call_id": "provider_open_bad",
                "name": "open_app",
                "arguments": '{"app_name": 42}',
            },
        ]
        mock = AsyncMock(side_effect=[_fake_response(first_output), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _PerCallTextMobileEnv(terminate_after=99)
        env.metadata.extra_tool_schemas = [open_app_tool]
        agent = GPTMobileUseAgent(metadata=env.metadata)
        result = await agent.sample(env, max_steps=3)

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        assert [item["call_id"] for item in outputs] == [
            "provider_tap",
            "provider_swipe",
            "provider_open",
            "provider_open_bad",
        ]

        def output_text(item):
            return next(
                block["text"] for block in item["output"] if block.get("type") == "input_text"
            )

        # ``app_name: 42`` violates the extra's schema, but argument admission is
        # the env's: the call reaches it and is answered like any other.
        output_texts = [output_text(item) for item in outputs]
        assert output_texts == [
            "result call_0000",
            "result call_0000",
            "result call_0001",
            "result call_0002",
        ]

        def has_output_image(item):
            return any(block.get("type") == "input_image" for block in item["output"])

        assert has_output_image(outputs[0]) is False
        assert has_output_image(outputs[1]) is True
        assert has_output_image(outputs[3]) is True
        assistant = result.lite_sample.messages[1]
        assert [tool_call_name(call) for call in assistant["tool_calls"]] == [
            "mobile",
            "open_app",
            "open_app",
        ]
        tool_messages = [
            message for message in result.lite_sample.messages if message.get("role") == "tool"
        ]
        assert [message["tool_call_id"] for message in tool_messages] == [
            "call_0000",
            "call_0001",
            "call_0002",
        ]

    def test_mobile_parser_restamps_provider_call_ids_to_canonical_ids(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "id": "response_1",
                    "name": "response",
                    "arguments": '{"text": "done"}',
                },
                {
                    "type": "function_call",
                    "call_id": "tap_1",
                    "name": "tap",
                    "arguments": '{"x": 540, "y": 1200}',
                },
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset({"response"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["response", "mobile"]
        assert tool_call_arguments(calls[1])["actions"][0]["action"] == "tap"
        assert [tool_call_id(call) for call in calls] == ["call_0000", "call_0001"]
        assert all("tool_call_id" not in call for call in calls)

    def test_mobile_parser_drops_malformed_function_call_arguments(self, caplog):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "call_bad_json",
                    "name": "response",
                    "arguments": "{not json",
                },
                {
                    "type": "function_call",
                    "call_id": "call_none_args",
                    "name": "response",
                    "arguments": None,
                },
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset({"response"}),
        ).message

        # Only ``{not json`` is dropped: the provider did not give the parser an
        # argument object at all, so there is nothing to route. ``None`` is a
        # well-formed empty object, and an advertised extra with empty arguments
        # goes to the env, which owns ``required: ["text"]``.
        assert msg["tool_calls"] == [make_tool_call("response", call_id="call_0000")]
        assert "malformed arguments" in caplog.text

    def test_mobile_parser_marks_undeclared_function_call_as_model_output_error(self, caplog):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "call_double_tap",
                    "name": "double_tap",
                    "arguments": '{"x": 540, "y": 1200}',
                }
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared function_call double_tap"
        assert "Ignoring undeclared GPT mobile function_call: double_tap" in caplog.text

    def test_mobile_call_without_provider_id_is_a_parse_error(self):
        """An empty ``call_id`` cannot be echoed back, so it is a parse error."""
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        parsed = parse_gpt_mobile_output_items_with_provenance(
            [{"type": "function_call", "name": "tap", "arguments": '{"x": 540, "y": 1200}'}],
            GPTMobileActionSpace(),
            PIXEL_6,
        )

        [provider_call] = parsed.provider_calls
        assert provider_call.provider_call_id == ""
        assert "missing provider id for tap" in provider_call.error
        assert parsed.message["tool_calls"] == []
        assert pop_model_output_error(parsed.message) == "missing provider id for tap"

    def test_mobile_parser_routes_invalid_active_extra_to_env_feedback(self):
        """A bad enum on an advertised extra is env ingress's answer, not a drop.

        The valid GUI call and the malformed extra both survive parsing; the
        model learns what was wrong from the env-owned feedback keyed to the
        extra's call id.
        """
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )
        from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "call_tap",
                    "name": "tap",
                    "arguments": '{"x": 540, "y": 1200}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_terminate",
                    "name": "terminate",
                    "arguments": '{"status": "bogus"}',
                },
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset({"terminate"}),
        ).message

        assert [tool_call_name(call) for call in msg["tool_calls"]] == ["mobile", "terminate"]
        assert pop_model_output_error(msg) is None

        routed, errors = prepare_env_tool_calls(
            msg["tool_calls"],
            LiteCUAMetadata(extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("terminate")]),
        )
        assert all(call["name"] != "terminate" for call, _ in routed)
        assert set(errors) == {"call_0001"}
        assert "terminate" in errors["call_0001"].message

    def test_mobile_parser_accepts_provider_flat_native_action(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "call_tap",
                    "name": "tap",
                    "arguments": '{"x": 540, "y": 1200}',
                },
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset(),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["mobile"]
        action = tool_call_arguments(calls[0])["actions"][0]
        assert action["action"] == "tap"
        assert 499 <= action["coordinate"][0] <= 501
        assert 499 <= action["coordinate"][1] <= 501

    def test_mobile_to_agent_unwraps_canonical_batch_to_provider_flat_calls(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace

        calls = GPTMobileActionSpace().convert_tool_calls_to_agent(
            [
                make_tool_call(
                    "mobile",
                    {
                        "actions": [
                            {"action": "tap", "coordinate": [500, 500]},
                            {"action": "type", "text": "hello"},
                        ]
                    },
                )
            ],
            resolution=PIXEL_6,
        )

        assert calls == [
            {"name": "tap", "arguments": {"x": 540, "y": 1200, "clicks": 1}},
            {"name": "type", "arguments": {"text": "hello"}},
        ]
        assert all("function" not in call and "type" not in call for call in calls)

    @pytest.mark.parametrize(
        "name,arguments",
        [
            ("tap", {}),
            ("swipe", {"start_x": 10, "start_y": 20, "end_x": 30}),
        ],
    )
    def test_mobile_parser_marks_malformed_native_as_model_output_error(self, name, arguments):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_bad",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert "malformed GPT mobile arguments" in pop_model_output_error(msg)

    def test_mobile_parser_rejects_undeclared_provider_mobile_wrapper(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_bad",
                    "name": "mobile",
                    "arguments": json.dumps(
                        {
                            "actions": [{"action": "tap", "x": 540, "y": 1200}],
                        }
                    ),
                }
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared function_call mobile"

    def test_mobile_parser_rejects_replayed_native_when_request_hid_native(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_tap",
                    "name": "tap",
                    "arguments": json.dumps({"x": 100, "y": 200}),
                }
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            active_provider_tool_names=frozenset(),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared function_call tap"

    def test_mobile_parser_active_provider_tools_do_not_union_agent_tool_names(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_tap",
                    "name": "tap",
                    "arguments": json.dumps({"x": 100, "y": 200}),
                }
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            declared_agent_tool_names=frozenset({"tap"}),
            active_provider_tool_names=frozenset(),
        ).message

        assert msg["tool_calls"] == []
        assert pop_model_output_error(msg) == "undeclared function_call tap"

    def test_mobile_parse_reports_malformed_native_provider_error(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        parsed = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_bad",
                    "name": "tap",
                    "arguments": "{}",
                }
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset(),
            active_provider_tool_names=frozenset({"tap"}),
        )

        [provider_call] = parsed.provider_calls
        assert provider_call.provider_call_id == "provider_bad"
        assert provider_call.canonical_call_id is None
        assert "malformed GPT mobile arguments for tap" in provider_call.error

    def test_mobile_parse_bad_native_errors_only_that_provider(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        parsed = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_tap",
                    "name": "tap",
                    "arguments": '{"x": 540, "y": 1200}',
                },
                {
                    "type": "function_call",
                    "call_id": "provider_bad",
                    "name": "tap",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "provider_wait",
                    "name": "wait",
                    "arguments": '{"duration": 1.0}',
                },
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset(),
            active_provider_tool_names=frozenset({"tap", "wait"}),
        )

        calls_by_provider_id = {call.provider_call_id: call for call in parsed.provider_calls}
        assert {
            call_id: call.canonical_call_id for call_id, call in calls_by_provider_id.items()
        } == {
            "provider_tap": "call_0000",
            "provider_bad": None,
            "provider_wait": "call_0000",
        }
        assert calls_by_provider_id["provider_tap"].error is None
        assert calls_by_provider_id["provider_wait"].error is None
        assert (
            "malformed GPT mobile arguments for tap" in calls_by_provider_id["provider_bad"].error
        )

    def test_mobile_parse_reports_undeclared_function_directly(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        parsed = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_unknown",
                    "name": "mystery",
                    "arguments": "{}",
                }
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset(),
            active_provider_tool_names=frozenset({"tap"}),
        )

        [provider_call] = parsed.provider_calls
        assert provider_call.canonical_call_id is None
        assert "undeclared function_call: mystery" in provider_call.error

    def test_mobile_parse_extra_tool_breaks_native_run(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        parsed = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "provider_tap",
                    "name": "tap",
                    "arguments": '{"x": 540, "y": 1200}',
                },
                {
                    "type": "function_call",
                    "call_id": "provider_swipe",
                    "name": "swipe",
                    "arguments": '{"start_x": 540, "start_y": 1600, "end_x": 540, "end_y": 800}',
                },
                {
                    "type": "function_call",
                    "call_id": "provider_response",
                    "name": "response",
                    "arguments": '{"text": "done"}',
                },
                {
                    "type": "function_call",
                    "call_id": "provider_wait",
                    "name": "wait",
                    "arguments": '{"duration": 1.0}',
                },
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset({"response"}),
            active_provider_tool_names=frozenset({"tap", "swipe", "wait"}),
        )

        assert [call.canonical_call_id for call in parsed.provider_calls] == [
            "call_0000",
            "call_0000",
            "call_0001",
            "call_0002",
        ]
        assert [call.error for call in parsed.provider_calls] == [None, None, None, None]

    def test_mobile_parser_merges_adjacent_provider_flat_native_calls_without_crossing_extras(self):
        from lite.agents.models.gpt.action_space import GPTMobileActionSpace
        from lite.agents.models.gpt.utils.parse import (
            parse_gpt_mobile_output_items_with_provenance,
        )

        msg = parse_gpt_mobile_output_items_with_provenance(
            [
                {
                    "type": "function_call",
                    "call_id": "call_tap",
                    "name": "tap",
                    "arguments": '{"x": 100, "y": 200}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_swipe",
                    "name": "swipe",
                    "arguments": '{"start_x": 540, "start_y": 1600, "end_x": 540, "end_y": 800}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_response",
                    "name": "response",
                    "arguments": '{"text": "done"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_wait",
                    "name": "wait",
                    "arguments": '{"duration": 1.0}',
                },
            ],
            GPTMobileActionSpace(),
            PIXEL_6,
            extra_tool_names=frozenset({"response"}),
        ).message

        calls = msg["tool_calls"]
        assert [tool_call_name(call) for call in calls] == ["mobile", "response", "mobile"]
        assert [tool_call_id(call) for call in calls] == [
            "call_0000",
            "call_0001",
            "call_0002",
        ]
        assert [action["action"] for action in tool_call_arguments(calls[0])["actions"]] == [
            "tap",
            "swipe",
        ]
        assert [action["action"] for action in tool_call_arguments(calls[2])["actions"]] == ["wait"]

    async def test_agent_restamp_uses_canonical_call_ids(self, monkeypatch):
        response_tool = make_tool_schema(
            "response",
            description="Submit the final answer.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        resp = _fake_response(
            [
                {
                    "type": "function_call",
                    "id": "response_1",
                    "name": "response",
                    "arguments": '{"text": "done"}',
                },
                {
                    "type": "function_call",
                    "call_id": "tap_1",
                    "name": "tap",
                    "arguments": '{"x": 540, "y": 1200}',
                },
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(
            metadata=LiteCUAMetadata(extra_tool_schemas=[response_tool]),
        )
        result = await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        calls = assistant["tool_calls"]
        assert "raw_response" not in assistant
        assert [tool_call_name(call) for call in calls] == ["response", "mobile"]
        assert tool_call_arguments(calls[1])["actions"][0]["action"] == "tap"
        assert [tool_call_id(call) for call in calls] == ["call_0000", "call_0001"]
        assert all("tool_call_id" not in call for call in calls)

    async def test_content_only_final_text_is_not_saved_as_response_tool(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "  done  "}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RejectEmptyActionsMobileEnv(terminate_after=99)
        agent = GPTMobileUseAgent()
        result = await agent.sample(env, max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert "raw_response" not in assistant
        assert assistant["content"] == [{"type": "text", "text": "  done  "}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert "call_id" not in env.actions_seen[0][0]
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "done"}
        assert result.terminated is True
        assert result.truncated is False
        assert result.episode_return == 1.0
        assert result.steps[0].reward == 1.0

    async def test_content_only_final_uses_runtime_response_when_enabled(self, monkeypatch):
        resp = _fake_response(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "final text"}],
                }
            ]
        )
        mock = AsyncMock(return_value=resp)
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RejectEmptyActionsMobileEnv(terminate_after=99)
        env.metadata.extra_tool_schemas = [LiteFinishToolSet.get_tool_schema("response")]
        agent = GPTMobileUseAgent()
        result = await agent.sample(env, max_steps=3)

        assistant = next(m for m in result.lite_sample.messages if m.get("role") == "assistant")
        assert assistant["content"] == [{"type": "text", "text": "final text"}]
        assert not assistant.get("tool_calls")
        assert len(env.actions_seen) == 1
        assert tool_call_name(env.actions_seen[0][0]) == "response"
        assert "call_id" not in env.actions_seen[0][0]
        assert tool_call_arguments(env.actions_seen[0][0]) == {"text": "final text"}
        assert result.terminated is True
        assert result.truncated is False


# -----------------------------------------------------------------------------
# Responses request payload surface (mobile)
# -----------------------------------------------------------------------------


class TestMobileRequestPayloadFields:
    """The exact kwarg set ``litellm.aresponses`` receives on the mobile path.

    Sibling of the desktop payload test. Mobile declares no ``truncation`` in
    its api_kwargs defaults, so this also pins that the shared request builder
    still sends ``truncation="auto"``.
    """

    _ALWAYS = {
        "model",
        "input",
        "tools",
        "parallel_tool_calls",
        "stream",
        "reasoning",
        "truncation",
        "extra_body",
    }

    async def test_default_request_sends_only_the_unconditional_fields(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        await GPTMobileUseAgent().sample(_FakeMobileEnv(terminate_after=1), max_steps=2)

        kwargs = mock.call_args.kwargs
        assert set(kwargs) == self._ALWAYS
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["stream"] is False
        assert kwargs["parallel_tool_calls"] is False
        assert kwargs["reasoning"] == {"effort": "medium", "summary": "concise"}
        assert kwargs["truncation"] == "auto"
        assert kwargs["extra_body"]["prompt_cache_key"].startswith("cua-lite-")
        assert {tool["name"] for tool in kwargs["tools"]} == NATIVE_MOBILE_TOOLS
        # Default ``system_prompt_in_input`` puts the prompt in ``input``, so the
        # one-shot top-level field stays off the payload entirely.
        assert [it.get("role") for it in kwargs["input"]] == ["developer", "user"]

    async def test_optional_fields_are_sent_only_when_configured(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(
            api_key="sk-test",
            api_base="https://example.invalid/v1",
            api_kwargs={
                "system_prompt_in_input": False,
                "max_output_tokens": 2048,
                "temperature": 0.5,
                "tool_choice": "required",
                "truncation": None,
            },
        )
        await agent.sample(_FakeMobileEnv(terminate_after=1), max_steps=2)

        kwargs = mock.call_args.kwargs
        assert set(kwargs) == (self._ALWAYS - {"truncation"}) | {
            "max_output_tokens",
            "instructions",
            "temperature",
            "tool_choice",
            "api_key",
            "api_base",
        }
        assert kwargs["max_output_tokens"] == 2048
        assert "mobile phone" in kwargs["instructions"]
        assert kwargs["temperature"] == 0.5
        assert kwargs["tool_choice"] == "required"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["api_base"] == "https://example.invalid/v1"
        assert [it.get("role") for it in kwargs["input"]] == ["user"]

    async def test_chained_turn_swaps_instructions_for_previous_response_id(self, monkeypatch):
        """With chaining ON and ``system_prompt_in_input: false``, mobile sends the
        one-shot ``instructions`` field on turn 0 only; the chained turn carries
        ``previous_response_id`` instead. Contrast with the unchained-resend test."""
        tap_item = {
            "type": "function_call",
            "call_id": "c1",
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        mock = AsyncMock(side_effect=[_fake_response([tap_item]), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(api_kwargs={"system_prompt_in_input": False})
        await agent.sample(_FakeMobileEnv(terminate_after=2), max_steps=3)

        assert mock.call_count == 2
        first, second = mock.call_args_list
        assert "mobile phone" in first.kwargs["instructions"]
        assert "previous_response_id" not in first.kwargs
        assert "instructions" not in second.kwargs
        assert second.kwargs["previous_response_id"] == "resp_test"

    async def test_unchained_multi_turn_image_indices_track_the_replayed_request(
        self,
        monkeypatch,
    ):
        """Unchained mobile replays the whole history, so ``image_indices`` must
        list every frame the request re-sends — read back from the markers on the
        ``function_call_output`` image blocks, which never reach the provider."""
        tap_item = {
            "type": "function_call",
            "call_id": "c1",
            "name": "tap",
            "arguments": '{"x": 540, "y": 1200}',
        }
        mock = AsyncMock(side_effect=[_fake_response([tap_item]), _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTMobileUseAgent(
            api_kwargs={
                "system_prompt_in_input": False,
                "chain_previous_response": False,
            }
        )
        result = await agent.sample(_FakeMobileEnv(terminate_after=2), max_steps=3)

        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 1)]

        second_input = mock.call_args_list[1].kwargs["input"]
        output_images = [
            block
            for item in second_input
            if item.get("type") == "function_call_output" and isinstance(item.get("output"), list)
            for block in item["output"]
            if block.get("type") == "input_image"
        ]
        assert len(output_images) == 1
        sent_payloads = [
            json.dumps(call.kwargs["input"], ensure_ascii=False, default=str)
            for call in mock.call_args_list
        ]
        assert all("_cua_lite_image_index" not in payload for payload in sent_payloads)
        assert all("_cua_lite_image_index" not in step.prompt for step in result.steps)
