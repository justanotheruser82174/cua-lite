"""Split GPT agent characterization tests.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_*.py -v
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agents.models.gpt._support import _colored_png_bytes, _fake_response, _FakeEnv
from PIL import Image

from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.core.tools.calls import tool_call_id
from lite.core.tools.results import LiteToolResult
from lite.gym.types import LiteEnvStepResult

# -----------------------------------------------------------------------------
# computer_call_output image when the env executed nothing (I14)
# -----------------------------------------------------------------------------


class _RejectsGuiActionEnv(_FakeEnv):
    """Mirrors ``webharbor.webvoyager``: ``screenshot`` is not in the env's
    ``valid_actions``, so nothing executes and the tool result is text-only
    (``images=[]``) — there is NO new frame for the next turn.
    """

    _REJECTION = (
        "invalid action: screenshot; valid_actions=['click', 'type', 'key', 'scroll', 'wait']"
    )

    async def step(self, actions):
        self._step_count += 1
        return LiteEnvStepResult(
            results=[
                LiteToolResult(
                    tool_call_id=tool_call_id(action),
                    text=self._REJECTION,
                    metadata={"is_error": True},
                )
                for action in actions
            ]
        )


class _FreshImageOnlyEnv(_FakeEnv):
    async def step(self, actions):
        self._step_count += 1
        return LiteEnvStepResult(
            results=[
                LiteToolResult(tool_call_id=tool_call_id(action), images=[self._shot])
                for action in actions
            ]
        )


class _FreshImageErrorEnv(_FakeEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._fresh_shot = _colored_png_bytes()

    async def step(self, actions):
        self._step_count += 1
        return LiteEnvStepResult(
            results=[
                LiteToolResult(
                    tool_call_id=tool_call_id(action),
                    images=[self._fresh_shot],
                    text="## AXTree:\nbutton Search",
                    error="invalid action: screenshot",
                    metadata={"is_error": True},
                )
                for action in actions
            ]
        )


class TestComputerCallOutputAlwaysCarriesAValidImage:
    """A ``computer_call_output`` has no text-only form: the Responses API
    requires an image. When the env executed nothing (every call rejected by
    the env's ``valid_actions`` filter) the loop must re-send the frame the model
    already saw, never interpolate a ``None`` into the data URI — that yields
    the literal ``"data:image/png;base64,None"`` and the API rejects the whole
    request with HTTP 400 "does not represent a valid image" before the first
    env step can complete.
    """

    async def test_no_new_frame_resends_last_frame_not_none(self, monkeypatch):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RejectsGuiActionEnv(terminate_after=5)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(env, max_steps=2)

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert outputs, f"expected a computer_call_output, got: {second_input}"
        url = outputs[0]["output"]["image_url"]
        payload = url.split("base64,", 1)[1]
        assert payload not in ("", "None"), (
            f"computer_call_output carried a non-image payload {payload!r} — "
            "the API rejects this request with 400 invalid image"
        )
        # The payload must be the frame the model already saw, and it must
        # actually decode as an image.
        assert base64.b64decode(payload) == env._shot
        Image.open(io.BytesIO(base64.b64decode(payload))).verify()

    async def test_no_new_frame_marks_the_reused_frame_index_for_unchained_history(
        self,
        monkeypatch,
    ):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"chain_previous_response": False},
        )
        result = await agent.sample(_RejectsGuiActionEnv(terminate_after=5), max_steps=2)

        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0, 0)]
        assert "_cua_lite_image_index" not in result.steps[-1].prompt

    async def test_no_new_frame_keeps_reused_frame_current_image_index(
        self,
        monkeypatch,
    ):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        class _Hook:
            def __init__(self):
                self.current_image_indices = []

            def on_step(self, data):
                self.current_image_indices.append(data.current_image_index)

            def on_complete(self, result):
                del result

        hook = _Hook()
        result = await GPTDesktopUseAgent(model_id="gpt-5.5").sample(
            _RejectsGuiActionEnv(terminate_after=5),
            max_steps=2,
            hooks=[hook],
        )

        assert [tuple(step.image_indices) for step in result.steps] == [(0,), (0,)]
        assert hook.current_image_indices == [0, 0]
        assert result.processed_images == result.lite_sample.images

    async def test_undeclared_computer_call_does_not_get_normal_screenshot_ack(self, monkeypatch):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_hidden",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FakeEnv(terminate_after=2)
        env.metadata.valid_actions = []
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            metadata=env.metadata,
            api_kwargs={"chain_previous_response": False},
        )
        await agent.sample(env, max_steps=2)

        assert len(mock.call_args_list) == 1

    @pytest.mark.parametrize(
        "chained,expected_image_indices",
        [
            # Chained: each request carries only the new items, and the running
            # provider-visible list already holds the frame's index from turn 0.
            (True, [(0,), (0,), (0,), (0,)]),
            # Unchained: the request replays the whole history, so every resent
            # frame has to stay marked with the index the model actually saw.
            (False, [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]),
        ],
    )
    async def test_consecutive_no_new_frame_turns_keep_resending_the_frame(
        self,
        monkeypatch,
        chained,
        expected_image_indices,
    ):
        """A RUN of rejected steps must not degrade: the loop has to hold the
        last sent frame AND the image index it carries across turns, not just
        recompute them from the step that happens to precede the failure. An
        unmarked resend silently drops out of ``LiteRLStep.image_indices``.
        """
        rejected = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[rejected, rejected, rejected, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _RejectsGuiActionEnv(terminate_after=99)
        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={"chain_previous_response": chained},
        )
        result = await agent.sample(env, max_steps=4)

        assert mock.call_count == 4
        for turn, call in enumerate(mock.call_args_list[1:], start=1):
            outputs = [
                item for item in call.kwargs["input"] if item.get("type") == "computer_call_output"
            ]
            assert outputs, f"turn {turn}: no computer_call_output"
            payload = outputs[-1]["output"]["image_url"].split("base64,", 1)[1]
            assert payload not in ("", "None"), (
                f"turn {turn} carried a non-image payload {payload!r}"
            )
            assert base64.b64decode(payload) == env._shot

        assert [tuple(step.image_indices) for step in result.steps] == expected_image_indices

    async def test_no_new_frame_forwards_the_env_rejection_text(self, monkeypatch):
        """The re-sent frame is identical to the previous one, so without the
        env's error text the model has no signal and re-emits the same call.
        """
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_RejectsGuiActionEnv(terminate_after=5), max_steps=2)

        second_input = mock.call_args_list[1].kwargs["input"]
        texts = [
            block.get("text", "")
            for item in second_input
            if isinstance(item.get("content"), list)
            for block in item["content"]
            if isinstance(block, dict) and block.get("type") == "input_text"
        ]
        assert any("invalid action: screenshot" in t for t in texts), (
            f"env rejection text was dropped from the request: {second_input}"
        )

    async def test_fresh_image_result_forwards_projected_error_text(self, monkeypatch):
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FreshImageErrorEnv(terminate_after=5)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(env, max_steps=2)

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "call_1"
        payload = outputs[0]["output"]["image_url"].split("base64,", 1)[1]
        assert base64.b64decode(payload) == env._fresh_shot

        texts = [
            block.get("text", "")
            for item in second_input
            if isinstance(item.get("content"), list)
            for block in item["content"]
            if isinstance(block, dict) and block.get("type") == "input_text"
        ]
        joined = "\n".join(texts)
        assert joined.count("## Error from previous action:") == 1
        assert joined.count("invalid action: screenshot") == 1
        assert "## AXTree:\nbutton Search" in joined
        assert "## Error from previous action:\ninvalid action: screenshot" in joined

    async def test_normal_turn_still_sends_the_fresh_frame_only(self, monkeypatch):
        """Non-vacuity guard for the two above: on a turn that DID execute, the
        fresh screenshot is the only provider feedback when the env result has
        no text/error payload.
        """
        first_resp = _fake_response(
            [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [{"type": "screenshot"}],
                }
            ]
        )
        mock = AsyncMock(side_effect=[first_resp, _fake_response()])
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FreshImageOnlyEnv(terminate_after=5), max_steps=2)

        second_input = mock.call_args_list[1].kwargs["input"]
        outputs = [item for item in second_input if item.get("type") == "computer_call_output"]
        assert outputs
        payload = outputs[0]["output"]["image_url"].split("base64,", 1)[1]
        Image.open(io.BytesIO(base64.b64decode(payload))).verify()
        assert not [item for item in second_input if item.get("role") == "user"], (
            f"no extra user turn expected on a normal step: {second_input}"
        )
