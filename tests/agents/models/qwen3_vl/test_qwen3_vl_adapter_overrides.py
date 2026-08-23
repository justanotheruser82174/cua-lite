"""
Qwen3-VL adapter protocol override and real-data smoke tests.

Run:
    uv run pytest tests/agents/models/qwen3_vl/test_qwen3_vl_adapter_overrides.py -v
"""

from __future__ import annotations

import json

import pytest
from lite_samples import get_real_samples, sample_trajectory_long

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol


class TestQwen3VLAdapterProtocolKwargs:
    def test_window_size_1_vs_2(self):
        sample = sample_trajectory_long(num_turns=6)
        adapter_ws1 = AgentAdapterRegistry.get(
            "qwen3_vl@desktop@use",
            protocol_kwargs={"full_history_size": 1},
        )
        adapter_ws2 = AgentAdapterRegistry.get(
            "qwen3_vl@desktop@use",
            protocol_kwargs={"full_history_size": 2},
        )

        assert isinstance(adapter_ws1, Qwen3VLDesktopUseAdapter)
        assert adapter_ws1.kwargs == {}
        assert adapter_ws2.kwargs == {}

        # Compare last-turn rendered messages: smaller window produces fewer messages.
        out_ws1 = adapter_ws1.unroll(sample)
        out_ws2 = adapter_ws2.unroll(sample)

        assert len(out_ws1.steps[-1]) < len(out_ws2.steps[-1])


class TestQwen3VLRealData:
    def test_real_trajectory_when_available(self):
        real = get_real_samples(platform="desktop")
        sample = real.get("use")
        if sample is None:
            pytest.skip("Real ScaleCUA trajectory data not found")
        messages = sample.get("messages")
        if messages is None:
            messages = []
        else:
            messages = json.loads(
                json.dumps(
                    messages,
                    default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
                )
            )
        if len(messages) < 2:
            pytest.skip("Real trajectory sample has too few messages")
        out = Qwen3VLHistoryProtocol(full_history_size=2).process_messages(messages)
        assert len(out) >= 2
