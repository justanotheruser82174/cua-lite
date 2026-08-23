"""Split GPT agent characterization tests.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_*.py -v
"""

from __future__ import annotations

import io
import logging
from unittest.mock import AsyncMock

import pytest
from agents.models.gpt._support import (
    _collect_input_image_details,
    _fake_response,
    _FakeEnv,
    _first_sent_input_image_png,
    _png_bytes,
)
from PIL import Image

from lite.agents.models.gpt.agent import GPTDesktopUseAgent
from lite.agents.models.gpt.utils.image_io import _call_api_with_actual_dim
from lite.gym.types import LiteEnvObservation

# -----------------------------------------------------------------------------
# detail
# -----------------------------------------------------------------------------


class TestInputImageDetail:
    """`detail` kwarg controls `detail` on all input_image items."""

    async def test_default_detail_is_original(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        captured = mock.call_args.kwargs["input"]
        details = _collect_input_image_details(captured)
        assert details, "expected at least one input_image in request"
        assert all(d == "original" for d in details), details

    async def test_non_default_detail_low(self, monkeypatch):
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(
            model_id="gpt-5.5",
            api_kwargs={
                "max_output_tokens": 4096,
                "reasoning_effort": "high",
                "detail": "low",
            },
        )
        await agent.sample(_FakeEnv(terminate_after=1), max_steps=2)

        captured = mock.call_args.kwargs["input"]
        details = _collect_input_image_details(captured)
        assert details
        assert all(d == "low" for d in details), details


# -----------------------------------------------------------------------------
# Image detail + API-limit safety
# -----------------------------------------------------------------------------


class TestImageDetailAndLimits:
    def test_default_resolution_is_none(self):
        # Default: pass screenshot through at env-native resolution.
        a = GPTDesktopUseAgent(model_id="gpt-5.5")
        assert a.resolution is None

    def test_default_image_detail_gpt5(self):
        # OpenAI computer-use guide: always prefer ``original``, never high/low.
        a = GPTDesktopUseAgent(model_id="gpt-5.5")
        assert a.api_kwargs["detail"] == "original"

    async def test_agent_resolution_stretches_sent_image_to_exact_target(self, monkeypatch):
        # Source 1920×1080 (16:9) with ``resolution=(1024, 768)`` (4:3): the
        # provider payload receives an exact stretch to 1024×768 (AR
        # distortion is the caller's responsibility).
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FakeEnv(terminate_after=1)
        env._shot = _png_bytes(1920, 1080)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5", resolution=(1024, 768))
        await agent.sample(env, max_steps=1)

        sent_png = _first_sent_input_image_png(mock.call_args_list[0].kwargs["input"])
        assert Image.open(io.BytesIO(sent_png)).size == (1024, 768)

    async def test_agent_resolution_none_sends_source_bytes_unmodified(self, monkeypatch):
        # resolution=None → no resize: the provider receives the exact source
        # PNG bytes, unmodified (no re-encode).
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FakeEnv(terminate_after=1)
        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        await agent.sample(env, max_steps=1)

        sent_png = _first_sent_input_image_png(mock.call_args_list[0].kwargs["input"])
        assert sent_png == env._shot

    async def test_agent_resolution_upscales_smaller_source_to_target(self, monkeypatch):
        # Stretch semantics: a source smaller than ``resolution`` is upscaled
        # to fill it exactly, same as the downscale case above.
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        env = _FakeEnv(terminate_after=1)  # default env shot is 800×600
        agent = GPTDesktopUseAgent(model_id="gpt-5.5", resolution=(1024, 768))
        await agent.sample(env, max_steps=1)

        sent_png = _first_sent_input_image_png(mock.call_args_list[0].kwargs["input"])
        assert Image.open(io.BytesIO(sent_png)).size == (1024, 768)

    async def test_no_image_observation_skips_processed_dim_lookup(self, monkeypatch):
        # A text-only observation (env.reset() returns image=None) sends a
        # request with no image block, so the API call must not spend a
        # round-trip resolving provider-processed dimensions — it falls back
        # to the caller's own sent dims silently (no error, no fetch).
        fetch = AsyncMock(side_effect=AssertionError("should not fetch dims"))
        monkeypatch.setattr(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            fetch,
        )
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        class _TextOnlyEnv(_FakeEnv):
            async def reset(self):
                return LiteEnvObservation(image=None, text="instr")

        result = await GPTDesktopUseAgent(model_id="gpt-5.5").sample(
            _TextOnlyEnv(terminate_after=1), max_steps=1
        )

        assert result.terminated is True
        fetch.assert_not_awaited()
        sent_input = mock.call_args_list[0].kwargs["input"]
        assert not any(
            isinstance(block, dict) and block.get("type") == "input_image"
            for item in sent_input
            for block in item.get("content", []) or []
            if isinstance(item.get("content"), list)
        )

    async def test_dim_fetch_failure_degrades_to_sent_dims(self, monkeypatch, caplog):
        # The processed-dimension lookup is best-effort. When it blows up the
        # turn must DEGRADE to the sent dims with a warning, never raise: the
        # sent dims are the processed dims unless the API resized, and the
        # resize case is owned by the separate "auto-downsampled" guard that
        # only fires when processed dims are known and differ. Raising here
        # would convert an intermittent provider gap into total trajectory loss
        # (it killed 11/16 GPT mobile trajectories in Phase 9).
        monkeypatch.setattr(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            AsyncMock(side_effect=RuntimeError("input_items unavailable")),
        )
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        with caplog.at_level(logging.WARNING, logger="lite.agents.models.gpt.utils.image_io"):
            result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=1)

        assert result.terminated is True
        assert "falling back to sent dims 800x600" in caplog.text

    async def test_dim_lookup_returning_no_images_degrades_to_sent_dims(self, monkeypatch, caplog):
        # Same degrade, but the lookup call succeeds and simply reports no image
        # items for an image-bearing request — the exact shape observed live
        # once prompt caching engaged on mobile.
        monkeypatch.setattr(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            AsyncMock(return_value=[]),
        )
        mock = AsyncMock(return_value=_fake_response())
        monkeypatch.setattr("litellm.aresponses", mock)

        agent = GPTDesktopUseAgent(model_id="gpt-5.5")
        with caplog.at_level(logging.WARNING, logger="lite.agents.models.gpt.utils.image_io"):
            result = await agent.sample(_FakeEnv(terminate_after=1), max_steps=1)

        assert result.terminated is True
        assert "returned no image items" in caplog.text
        assert "falling back to sent dims 800x600" in caplog.text

    # Temporary migration guard (CD-TESTS-PRIVATE-HELPER-COUPLING): the test
    # below is the only caller of the module-level
    # ``image_io._call_api_with_actual_dim`` import, and that import pins private
    # module placement. It is kept because the degrade-to-sent-dims contract it
    # pins has no public seam -- ``sample()`` surfaces neither the processed
    # image dimensions nor the returned frame. Deletion criterion: drop the
    # import and rewrite this test through the public surface once the GPT agent
    # exposes an injectable processed-image-dimension seam -- the same criterion
    # already recorded for the ``image_io._fetch_processed_image_dims``
    # monkeypatch targets used throughout this file.
    @pytest.mark.parametrize(
        ("fetch_mock", "expected_warning"),
        [
            (
                AsyncMock(side_effect=RuntimeError("input_items unavailable")),
                "Could not fetch processed image dims",
            ),
            (AsyncMock(return_value=[]), "returned no image items"),
            # A response with no id cannot be looked up at all — same degrade.
            (AsyncMock(side_effect=AssertionError("should not fetch")), "carried no id"),
        ],
    )
    async def test_unanswered_dim_lookup_returns_the_sent_frame(
        self, monkeypatch, caplog, fetch_mock, expected_warning
    ):
        # The positive half of the degrade: the returned frame IS the sent
        # frame, so callers normalize by dims they know rather than failing.
        monkeypatch.setattr(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            fetch_mock,
        )
        response = {"output": [], "usage": {}}
        if expected_warning != "carried no id":
            response["id"] = "resp_test"
        api_kwargs = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,x"}],
                }
            ]
        }

        with caplog.at_level(logging.WARNING, logger="lite.agents.models.gpt.utils.image_io"):
            got, actual_w, actual_h = await _call_api_with_actual_dim(
                AsyncMock(return_value=response),
                api_kwargs,
                sent_w=1080,
                sent_h=2400,
                model_id="gpt-5.5",
                api_base=None,
                api_key=None,
            )

        assert got is response
        assert (actual_w, actual_h) == (1080, 2400)
        assert expected_warning in caplog.text
