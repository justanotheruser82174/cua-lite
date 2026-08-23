"""Registry guard: no two regex patterns ``fullmatch`` the same concrete key.

The registries resolve a key by exact match first, then by regex ``fullmatch``
(``lite/utils/registry.py``). If two *patterns* both match one concrete key, the
winner is registration-order-dependent — a latent footgun. The tools refactor
collapsed each in-scope family to a single ``@(desktop|browser)`` class
(browser nav is an env extra_tool, not a per-platform action space), so this
pins (a) that old separate per-platform keys are absent, (b) that ``@browser``
and ``@desktop`` resolve to the SAME class, and (c) that stale ``@web`` keys do
not survive through wildcard registrations.

Run: uv run pytest tests/agents/test_registry_guard.py -v
"""
from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space import ActionSpaceRegistry
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.core.agent import AgentRegistry
from lite.core.metadata import LiteCUAMetadata

register_all()

_FAMILIES = ["lite", "qwen3_vl", "qwen3_5", "gpt", "claude"]
_PLATFORMS = [platform.value for platform in LiteCUAMetadata.Platform]
_TASKS = [task_type.value for task_type in LiteCUAMetadata.TaskType]


def _concrete_keys() -> list[str]:
    keys = [f"{f}@{p}@{t}" for f in _FAMILIES for p in _PLATFORMS for t in _TASKS]
    keys += [f"{f}@{p}" for f in _FAMILIES for p in _PLATFORMS]  # action-space 2-part keys
    return keys


@pytest.mark.parametrize("registry", [
    ActionSpaceRegistry, AgentAdapterRegistry, AgentRegistry,
], ids=["action_space", "adapter", "agent"])
def test_no_two_patterns_match_same_key(registry):
    """For every concrete family@platform@task key, at most one registered
    regex pattern fullmatches it (exact-key registrations are unambiguous by
    construction)."""
    offenders = {}
    for key in _concrete_keys():
        matches = [pat for pat, compiled, _ in registry._patterns if compiled.fullmatch(key)]
        if len(matches) > 1:
            offenders[key] = matches
    assert not offenders, f"{registry.__name__}: ambiguous pattern matches: {offenders}"


@pytest.mark.parametrize("fam,registry", [
    # Students have adapters (lite is data-only — adapter, no live agent);
    # gpt/claude are teacher-only — live agent, no adapter.
    ("lite", AgentAdapterRegistry),
    ("qwen3_vl", AgentAdapterRegistry),
    ("qwen3_5", AgentAdapterRegistry),
    ("qwen3_vl", AgentRegistry),
    ("qwen3_5", AgentRegistry),
    ("gpt", AgentRegistry),
    ("claude", AgentRegistry),
])
def test_browser_navigation_resolves_to_same_class(fam, registry):
    """Each in-scope family resolves ``@browser@use`` and ``@desktop@use``
    to the SAME concrete class via the collapsed ``@(desktop|browser)`` regex."""
    browser, _ = registry._find_item(f"{fam}@browser@use")
    desk, _ = registry._find_item(f"{fam}@desktop@use")
    assert browser is not None and desk is not None
    assert browser is desk


@pytest.mark.parametrize(
    "registry,key",
    [
        (ActionSpaceRegistry, "lite@web"),
        (ActionSpaceRegistry, "qwen3_vl@web"),
        (ActionSpaceRegistry, "qwen3_vl@web@point"),
        (AgentAdapterRegistry, "lite@web@use"),
        (AgentAdapterRegistry, "lite@web@understanding"),
        (AgentAdapterRegistry, "lite@web@grounding.point"),
        (AgentAdapterRegistry, "lite@web@grounding.bbox"),
        (AgentAdapterRegistry, "qwen3_vl@web@use"),
        (AgentAdapterRegistry, "qwen3_vl@web@understanding"),
        (AgentAdapterRegistry, "qwen3_vl@web@grounding.bbox"),
        (AgentAdapterRegistry, "qwen3_vl.base@web"),
        (AgentAdapterRegistry, "qwen3_vl.base@web@use"),
        (AgentAdapterRegistry, "qwen3_5@web@understanding"),
        (AgentAdapterRegistry, "qwen3_5.base@web"),
        (AgentAdapterRegistry, "qwen3_8@web@understanding"),
        (AgentAdapterRegistry, "qwen3_8.base@web"),
        (AgentAdapterRegistry, "fara.base@web@use"),
        (AgentRegistry, "qwen3_vl@web@use"),
        (AgentRegistry, "qwen3_vl.base@web@use"),
        (AgentRegistry, "qwen3_5.base@web@use"),
        (AgentRegistry, "qwen3_8.base@web@use"),
        (AgentRegistry, "fara.base@web@use"),
        (AgentRegistry, "gpt.teacher@web@use"),
    ],
)
def test_stale_web_keys_do_not_resolve(registry, key):
    assert not registry.contains(key)
    with pytest.raises(KeyError):
        registry.get(key)
