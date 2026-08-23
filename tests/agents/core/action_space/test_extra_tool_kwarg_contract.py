"""``active_extra_tool_names`` must not soften GPT/Claude native parsing.

On claude/gpt desktop the keyword must NOT admit unknown native actions. Those
agents own no adapter, so neither ``active_extra_tool_names`` injector
(``BaseAgentAdapter`` and ``Qwen3VLBaseAdapter``) can ever target them, and every
one of their own call sites dispatches env extras by NAME in a sibling branch
BEFORE the action space is consulted. A name can therefore only reach the
converter as the ``action`` field INSIDE the provider's own ``computer`` tool —
a closed provider enum — where it is a hallucination even when the same name is
separately offered as its own tool.

Run:
    uv run pytest tests/agents/core/action_space/test_extra_tool_kwarg_contract.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import ActionSpaceRegistry
from lite.agents.core.adapter.base import AgentAdapterRegistry

RESOLUTION = (1920, 1080)

#: The keys whose ``from_agent`` rejects unknown native names.
#: Kept in sync with ``test_unknown_action_direction_policy.UNKNOWN_NATIVE_KEYS``.
UNKNOWN_NATIVE_KEYS = (
    "claude@desktop",
    "claude@desktop@point",
    "claude@browser",
    "claude@browser@point",
    "gemini@desktop",
    "gemini@mobile",
    "gemini@browser",
    "gpt@desktop",
    "gpt@desktop@point",
    "gpt@browser",
    "gpt@browser@point",
)


@pytest.fixture(autouse=True, scope="module")
def _registered():
    from lite.agents.bootstrap import register_all

    register_all()


class TestClaudeAndGptNativeParsingIsNotRelaxable:
    @pytest.mark.parametrize("key", UNKNOWN_NATIVE_KEYS)
    def test_declaring_the_name_active_does_not_admit_it(self, key):
        space = ActionSpaceRegistry.get(key)
        action = {"action": "goto", "type": "goto", "url": "http://example.com"}
        with pytest.raises(ValueError, match="unknown"):
            space.convert_tool_calls_from_agent(
                [action],
                resolution=RESOLUTION,
                active_extra_tool_names={"goto"},
            )

    @pytest.mark.parametrize("key", UNKNOWN_NATIVE_KEYS)
    def test_the_keyword_does_not_change_the_error(self, key):
        space = ActionSpaceRegistry.get(key)
        action = {"action": "goto", "type": "goto", "url": "http://example.com"}
        for kwargs in ({}, {"active_extra_tool_names": {"goto"}}):
            with pytest.raises(ValueError, match="unknown"):
                space.convert_tool_calls_from_agent([action], resolution=RESOLUTION, **kwargs)


class TestTheInjectorsCannotReachClaudeOrGpt:
    """Why the decision above is safe: there is no path that feeds the keyword in."""

    def test_no_adapter_is_registered_for_claude_or_gpt(self):
        """The two injectors live on adapters; claude and gpt have none."""
        for family in ("claude", "gpt"):
            for platform in ("desktop", "browser", "mobile"):
                for task in ("use", "grounding.point", "grounding.action"):
                    with pytest.raises(KeyError):
                        AgentAdapterRegistry.get(f"{family}@{platform}@{task}")
