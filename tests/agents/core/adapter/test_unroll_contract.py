"""Smoke tests for the new ``unroll`` / ``render_step`` adapter contract.

These exercise the fundamental shape invariants of every adapter family:
- ``unroll(sample)`` returns an :class:`AgentSample` with
  ``processed_images`` (list[PIL]) + ``steps`` (list[AgentStep]) +
  ``metadata`` (dict).
- ``len(steps)`` matches the original turn count.
- Each step's image references via ``ImageContent.index`` point into
  ``processed_images``.
- ``count_sample_turns`` correctly counts predict-time partial turns.

No slime imports — these run in any environment with cua-lite installed.
"""

from __future__ import annotations

import pytest
from PIL import Image

from lite.agents.core.adapter import AgentAdapterRegistry, AsIsAdapter
from lite.agents.types import AgentSample
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.turns import count_sample_turns
from lite.core.tools.calls import make_tool_call


class _AdapterReferencesUnprocessedImage(AsIsAdapter):
    def render_step(self, sample, k, processed):
        return [{"role": "user", "content": [{"type": "image", "index": 1}]}]


@pytest.fixture(autouse=True, scope="module")
def _registered_adapters():
    from lite.agents.bootstrap import register_all

    register_all()


def _platform_from_adapter_key(adapter_key: str) -> str:
    return adapter_key.split("@")[1]


def _build_synthetic_traj(
    *,
    n_turns: int,
    with_assistant: bool = True,
    platform: str = "desktop",
) -> LiteSample:
    """Build a synthetic LiteSample with ``n_turns`` user→assistant pairs.

    If ``with_assistant=False``, the trajectory ends in a user message
    (predict-time partial turn).
    """
    messages: list[dict] = []
    wrapper_name = "mobile" if platform == "mobile" else "computer"
    action_name = "tap" if platform == "mobile" else "click"
    for k in range(n_turns):
        if k == 0:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "index": k},
                    {"type": "text", "text": "instruction"},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "image", "index": k}],
            })
        if with_assistant or k < n_turns - 1:
            messages.append({
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"Action {k}: do step {k}"}],
                "tool_calls": [make_tool_call(
                    wrapper_name,
                    {
                        "actions": [{
                            "action": action_name,
                            "coordinate": [100 + k, 200 + k],
                        }],
                    },
                    call_id=f"call_{k:04d}",
                )],
            })
    images = [Image.new("RGB", (32, 32), color=(k * 30, 0, 0)) for k in range(n_turns)]
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=images,
        messages=messages,
    )


# -- adapter coverage ----------------------------------------------------

# All navigation adapters that use ``render_step``.
NAV_ADAPTER_KEYS = [
    "lite@desktop@use",
    "lite@mobile@use",
    "qwen3_vl@desktop@use",
    "qwen3_vl@mobile@use",
    "qwen3_5@desktop@use",
    "qwen3_5@mobile@use",
    "ui_tars@desktop@use",
    "ui_tars@mobile@use",
    "ui_tars_15_v1@desktop@use",
    "ui_tars_15_v1@mobile@use",
    "evocua@desktop@use",
    "mai_ui@mobile@use",
    "step_gui@mobile@use",
]


@pytest.mark.parametrize("adapter_key", NAV_ADAPTER_KEYS)
def test_unroll_returns_agent_sample_shape(adapter_key):
    """``unroll`` returns an :class:`AgentSample` with the expected
    shape: ``processed_images`` / ``steps`` / ``metadata``."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    platform = _platform_from_adapter_key(adapter_key)
    sample = _build_synthetic_traj(n_turns=3, platform=platform)

    agent_sample = adapter.unroll(sample)

    assert isinstance(agent_sample, AgentSample)
    # processed_images is the trajectory's image list after process_image.
    assert isinstance(agent_sample.processed_images, list)
    assert len(agent_sample.processed_images) == 3
    # steps is a list of AgentStep (= list[AgentMessage]).
    assert isinstance(agent_sample.steps, list)
    for step in agent_sample.steps:
        assert isinstance(step, list), f"AgentStep must be list, got {type(step)}"
        for msg in step:
            assert isinstance(msg, dict), f"AgentMessage must be dict, got {type(msg)}"
            assert "role" in msg
    # metadata is a dict from LiteCUAMetadata.to_dict.
    assert isinstance(agent_sample.metadata, dict)
    assert agent_sample.metadata.get("metadata_kind") == "cua"
    assert agent_sample.metadata.get("dims") == [platform, "use"]


@pytest.mark.parametrize("adapter_key", NAV_ADAPTER_KEYS)
def test_unroll_step_count_matches_trajectory_turn_count(adapter_key):
    """``len(agent_sample.steps)`` equals the source LiteSample's turn count.

    For T=3 input, expect ``len(steps) == 3`` (one rendered view per turn).
    """
    adapter = AgentAdapterRegistry.get(adapter_key)
    sample = _build_synthetic_traj(
        n_turns=3,
        platform=_platform_from_adapter_key(adapter_key),
    )

    agent_sample = adapter.unroll(sample)

    assert len(agent_sample.steps) == 3, (
        f"adapter={adapter_key}: expected 3 steps, got {len(agent_sample.steps)}"
    )


def test_sample_turn_count_handles_predict_time_partial_turn():
    """At predict time the trajectory ends in a user message (no assistant
    yet for the current turn). ``count_sample_turns`` must count this as a
    full turn — that's the turn we're about to predict."""
    # Post-turn-2 (full pair): 4 messages, 2 turns.
    s_full = _build_synthetic_traj(n_turns=2, with_assistant=True)
    assert count_sample_turns(s_full) == 2

    # Predict-time at turn 3 (partial — user-only): 5 messages, count = 3.
    s_partial = _build_synthetic_traj(n_turns=3, with_assistant=False)
    assert count_sample_turns(s_partial) == 3


def test_as_is_adapter_t_eq_1():
    """AsIsAdapter is single-turn (T=1)."""
    adapter = AsIsAdapter()
    sample = _build_synthetic_traj(n_turns=1)

    agent_sample = adapter.unroll(sample)

    assert len(agent_sample.steps) == 1
    # Step 1 is the whole conversation pass-through.
    assert agent_sample.steps[0] == sample.messages


def test_unroll_rejects_rendered_image_index_with_unprocessed_slot():
    adapter = _AdapterReferencesUnprocessedImage()
    sample = LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
        images=[
            Image.new("RGB", (32, 32), color=(0, 0, 0)),
            Image.new("RGB", (32, 32), color=(255, 0, 0)),
        ],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "instruction"},
            ],
        }],
    )

    with pytest.raises(
        ValueError,
        match=r"render_step\(k=1\).*image index 1.*processed_images\[1\]",
    ):
        adapter.unroll(sample)


@pytest.mark.parametrize("adapter_key", NAV_ADAPTER_KEYS)
def test_unroll_image_indices_point_into_processed_images(adapter_key):
    """Every ``ImageContent.index`` referenced in ``steps[k]`` must be a
    valid index into ``processed_images``. Catches off-by-one errors in
    protocol windowing or image reordering."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    sample = _build_synthetic_traj(
        n_turns=4,
        platform=_platform_from_adapter_key(adapter_key),
    )

    agent_sample = adapter.unroll(sample)
    n_images = len(agent_sample.processed_images)

    for k, step in enumerate(agent_sample.steps):
        for msg in step:
            for part in msg.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "image":
                    idx = part.get("index")
                    if idx is not None:
                        assert 0 <= int(idx) < n_images, (
                            f"adapter={adapter_key} turn k={k}: "
                            f"ImageContent.index={idx} out of range [0, {n_images})"
                        )
                        assert agent_sample.processed_images[int(idx)] is not None, (
                            f"adapter={adapter_key} turn k={k}: "
                            f"ImageContent.index={idx} references an unprocessed slot"
                        )


@pytest.mark.parametrize("adapter_key", [
    "lite@desktop@grounding.action",
    "lite@desktop@understanding",
    "lite@desktop@grounding.bbox",
    "lite@desktop@grounding.point",
])
def test_single_turn_adapters_emit_one_step(adapter_key):
    """Grounding / understanding adapters are T=1 by design."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    sample = _build_synthetic_traj(n_turns=1)

    agent_sample = adapter.unroll(sample)
    assert len(agent_sample.steps) == 1
