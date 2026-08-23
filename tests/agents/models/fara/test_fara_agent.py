"""
Tests for the Fara-1.0 core agent + factory wiring.

Covers:
  1. Agent registry resolution (``fara@(desktop|browser)@...`` keys)
  2. Agent auto-resolves the matching Fara adapter
  3. ``LOCAL_AGENTS`` launch entry + ``compose_key`` composition

Run:
    uv run pytest tests/agents/models/fara/test_fara_agent.py -v
"""

from __future__ import annotations

from lite.agents.core.agent import AgentRegistry
from lite.agents.factory import LOCAL_AGENTS
from lite.agents.models.fara.adapter import (
    FaraDesktopGroundingActionAdapter,
    FaraDesktopGroundingPointAdapter,
    FaraDesktopUseAdapter,
)
from lite.agents.models.fara.agent import (
    FaraDesktopGroundingActionAgent,
    FaraDesktopGroundingPointAgent,
    FaraDesktopUseAgent,
)
from lite.utils.registry import compose_key


async def _dummy_generate(**kwargs):
    return {"response": ""}


# =============================================================================
# 1. Agent registry resolution
# =============================================================================

class TestAgentRegistry:
    def test_resolves_desktop_and_browser_use(self):
        for key in ("fara@desktop@use", "fara@browser@use"):
            agent = AgentRegistry.get(key, generate_fn=_dummy_generate)
            assert isinstance(agent, FaraDesktopUseAgent)

    def test_resolves_grounding_action(self):
        agent = AgentRegistry.get("fara@browser@grounding.action", generate_fn=_dummy_generate)
        assert isinstance(agent, FaraDesktopGroundingActionAgent)

    def test_resolves_grounding_point(self):
        agent = AgentRegistry.get("fara@desktop@grounding.point", generate_fn=_dummy_generate)
        assert isinstance(agent, FaraDesktopGroundingPointAgent)
        assert isinstance(agent.adapter, FaraDesktopGroundingPointAdapter)


# =============================================================================
# 2. Agent adapter auto-resolution
# =============================================================================

class TestAgentAdapterResolution:
    def test_use_adapter(self):
        agent = AgentRegistry.get("fara@browser@use", generate_fn=_dummy_generate)
        assert isinstance(agent.adapter, FaraDesktopUseAdapter)
        # Fara has no leading "# Response format" block — tools section carries it.
        assert agent.adapter.system_prompt is None

    def test_grounding_action_adapter(self):
        agent = AgentRegistry.get("fara@browser@grounding.action", generate_fn=_dummy_generate)
        assert isinstance(agent.adapter, FaraDesktopGroundingActionAdapter)


# =============================================================================
# 3. Factory launch entry
# =============================================================================

class TestFactoryEntry:
    def test_local_agents_has_fara(self):
        assert LOCAL_AGENTS["microsoft/Fara-7B"] == {"agent_id": "fara"}

    def test_compose_agent_key_browser_use(self):
        assert compose_key("fara", "browser", "use") == "fara@browser@use"

    def test_compose_agent_key_grounding_action(self):
        assert compose_key("fara", "desktop", "grounding.action") == "fara@desktop@grounding.action"
