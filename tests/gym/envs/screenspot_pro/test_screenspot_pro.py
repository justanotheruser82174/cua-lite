"""Tests for the screenspot_pro CUA-Lite gym environment.

Requires the ScreenSpot-Pro dataset to be downloaded.

Run:
    uv run pytest tests/gym/envs/screenspot_pro/test_screenspot_pro.py -v
"""

from __future__ import annotations

import pytest

import lite.gym as gym
from lite.core.tools import make_tool_call
from lite.gym.envs.screenspot_pro.main import _cached_data_dir

# Skip all tests if data is not available. The snapshot is resolved lazily (no
# module-level _DATA_DIR anymore — that ran a blocking snapshot_download at
# import); probe the LOCAL cache only (never download from a skipif).
pytestmark = pytest.mark.skipif(
    _cached_data_dir() is None, reason="ScreenSpot-Pro data not downloaded"
)

# ---------------------------------------------------------------------------
# Registry tests (sync, no env creation needed)
# ---------------------------------------------------------------------------

def test_tasks_registered():
    """screenspot_pro should have tasks registered."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    assert len(task_ids) > 0, "No screenspot_pro tasks registered"

def test_task_id_format():
    """Task IDs should follow <annotation_filename>_<index> format."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    for tid in task_ids[:5]:
        parts = tid.rsplit("_", 1)
        assert len(parts) == 2, f"Task ID '{tid}' doesn't match expected format"
        assert parts[1].isdigit(), f"Task ID '{tid}' index part is not a digit"

def test_make_env():
    """gym.make should return an env with desktop platform."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    assert env is not None
    assert env.metadata.platform == "desktop"

@pytest.mark.asyncio
async def test_metadata_fields():
    """metadata others should have expected keys; instruction comes from reset() obs.text."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    obs = await env.reset()
    assert isinstance(obs.text, str)
    assert len(obs.text) > 0
    meta = env.metadata
    assert "bbox" in meta.others
    assert "img_size" in meta.others
    assert "application" in meta.others
    assert "ui_type" in meta.others

# ---------------------------------------------------------------------------
# Async tests — full lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_returns_screenshot():
    """reset() should return a valid base64 PNG screenshot."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        obs = await env.reset()
        assert obs.image
        raw = obs.image
        assert raw[:4] == b"\x89PNG", "Screenshot should be valid PNG"
    finally:
        await env.close()

@pytest.mark.asyncio
async def test_correct_click():
    """step() with a click inside the bbox should return reward=1.0."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        await env.reset()
        bbox = env.metadata.others["bbox"]
        img_w, img_h = env.metadata.others["img_size"]

        # Compute center of bbox in [0, 1000] coordinates
        center_x = ((bbox[0] + bbox[2]) / 2.0 / img_w) * 1000
        center_y = ((bbox[1] + bbox[3]) / 2.0 / img_h) * 1000

        result = await env.step([
            make_tool_call("point", {"coordinate": [center_x, center_y]}),
        ])
        assert result.reward == 1.0
        assert result.terminated is True
    finally:
        await env.close()

@pytest.mark.asyncio
async def test_incorrect_click():
    """step() with a point outside the bbox should return reward=0.0."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        await env.reset()
        # Point at (0, 0) — very unlikely to be inside any bbox
        result = await env.step([
            make_tool_call("point", {"coordinate": [0, 0]}),
        ])
        assert result.reward == 0.0
        assert result.terminated is True
    finally:
        await env.close()

@pytest.mark.asyncio
async def test_invalid_action():
    """step() with no click action should return reward=0.0."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        await env.reset()
        result = await env.step([
            make_tool_call("type", {"text": "hello"}),
        ])
        assert result.reward == 0.0
        assert result.terminated is True
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_unknown_non_action_tool_returns_unsupported_result():
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        await env.reset()
        result = await env.step([
            make_tool_call("frobnicate", {}, call_id="call_frob"),
        ])
        assert result.terminated is True
        assert result.results[0].tool_call_id == "call_frob"
        assert result.results[0].images == []
        assert result.results[0].text is None
        assert result.results[0].error == "unknown tool: frobnicate"
        assert result.results[0].metadata == {"is_error": True}
        assert result.info["executed_actions"][0] == {
            "call": "noop",
            "args": {
                "name": "frobnicate",
                "reason": "unknown tool",
            },
        }
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_executed_log_keeps_each_result_call_id_separate():
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        await env.reset()
        bbox = env.metadata.others["bbox"]
        img_w, img_h = env.metadata.others["img_size"]
        center_x = ((bbox[0] + bbox[2]) / 2.0 / img_w) * 1000
        center_y = ((bbox[1] + bbox[3]) / 2.0 / img_h) * 1000

        result = await env.step([
            make_tool_call(
                "point",
                {"coordinate": [center_x, center_y]},
                call_id="call_point",
            ),
            make_tool_call("frobnicate", {}, call_id="call_frob"),
        ])

        assert result.info["executed_actions"][0]["call"] == "point"
        assert result.info["executed_actions"][1] == {
            "call": "noop",
            "args": {
                "name": "frobnicate",
                "reason": "unknown tool",
            },
        }
        by_id = {res.tool_call_id: res for res in result.results}
        assert set(by_id) == {"call_frob"}
        assert by_id["call_frob"].error == "unknown tool: frobnicate"
        assert by_id["call_frob"].metadata == {"is_error": True}
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_unknown_non_action_tool_without_call_id_is_logged_as_noop():
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        await env.reset()
        result = await env.step([
            make_tool_call("frobnicate", {}),
        ])
        assert result.terminated is True
        assert result.results == []
        assert result.info["executed_actions"][0] == {
            "call": "noop",
            "args": {
                "name": "frobnicate",
                "reason": "unknown tool",
            },
        }
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_single_step_terminated():
    """Every step should set terminated=True (single-step benchmark)."""
    task_ids = [tid for ids in gym.registry.task_ids("screenspot_pro").values() for tid in ids]
    env = gym.make(f"screenspot_pro@{task_ids[0]}")
    try:
        await env.reset()
        result = await env.step([
            make_tool_call("point", {"coordinate": [500, 500]}),
        ])
        assert result.terminated is True
        # Single-turn rollout rows must have the SAME shape as lite/data/preproc's
        # single-turn label rows: one assistant decision, no observation. Echoing
        # reset's screenshot here puts a duplicate PNG on the row and answers a call
    finally:
        await env.close()

@pytest.mark.asyncio
async def test_boundary_point_logs_the_float_the_reward_used():
    """A log-only reward audit must agree with the recorded reward.

    ``_evaluate`` scores the UNROUNDED de-normalized pixel, so if
    ``info.executed_actions`` rounds to int, an auditor replaying the log
    disagrees with the reward whenever the click lands within half a pixel of a
    bbox edge. Measured on the campaign corpus: 2 of 60 sampled grounding tasks.

    This one is that case exactly: point [163, 192] on 3456x2160 → x = 163/1000 *
    3456 = 563.328, just OUTSIDE bbox ``[545, 402, 563, 425]`` — so the reward is
    0.0, while the old int log (563) landed on the right edge and replayed as
    INSIDE → 1.0.
    """
    env = gym.make("screenspot_pro@linux_common_linux_0")
    try:
        await env.reset()
        res = await env.step([make_tool_call("point", {"coordinate": [163, 192]})])
        bbox = res.info["annotation"]["bbox"]
        assert bbox == [545, 402, 563, 425], (
            "task fixture changed — pick another exact-boundary task"
        )
        logged = res.info["executed_actions"][0]["args"]
        audited = 1.0 if (
            bbox[0] <= logged["x"] <= bbox[2] and bbox[1] <= logged["y"] <= bbox[3]
        ) else 0.0
        assert audited == res.reward == 0.0
    finally:
        await env.close()
