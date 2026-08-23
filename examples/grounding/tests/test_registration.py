from __future__ import annotations

from lite.agents.core.adapter.base import AgentAdapterRegistry
from lite.agents.core.agent.base import AgentRegistry


def test_regionfocus_external_registry_entries_resolve() -> None:
    """RegionFocus keys stay registered by the Grounding example owner."""
    import examples.grounding.adapter  # noqa: F401

    for key in (
        "qwen3_vl.regionfocus@desktop@grounding.point",
        "qwen3_5.regionfocus@desktop@grounding.point",
    ):
        assert AgentAdapterRegistry.contains(key)
        assert AgentRegistry.contains(key)
        assert AgentAdapterRegistry._find_item(key)[0] is not None
        assert AgentRegistry._find_item(key)[0] is not None
