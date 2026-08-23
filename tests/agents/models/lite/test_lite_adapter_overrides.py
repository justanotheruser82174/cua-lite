"""
Lite adapter override and long-unroll coverage.

Run:
    uv run pytest tests/agents/models/lite/test_lite_adapter_overrides.py -v
"""

from __future__ import annotations

from lite_samples import sample_trajectory_long

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.lite import adapter as _lite_adapter  # noqa: F401


class TestLiteUnrollLongTrajectory:
    def test_unroll_yields_n_steps(self):
        sample = sample_trajectory_long(num_turns=6)
        adapter = AgentAdapterRegistry.get("lite@desktop@use")
        agent_sample = adapter.unroll(sample)
        assert len(agent_sample.steps) == 6
