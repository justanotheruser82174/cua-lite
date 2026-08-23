"""The two TOOL-LAYER invariants, over every registered action space.

The contract has three name layers — **tool** (what is emitted to the
provider), **action** (the canonical vocabulary ``valid_actions`` gates) and
**native action** (a family's own spellings) — and exactly two invariants, both
strictly WITHIN the tool layer::

    invariant 1   len(get_tool_names()) == len(get_tool_schemas())
    invariant 2   get_tool_schema(n) is not None  <=>  n in get_tool_names()

**No invariant crosses layers.** ``LiteDesktopActionSet`` carries 14 actions and
emits 1 tool, and that divergence is what "batched" MEANS — so nothing here
compares the tool layer against ``get_action_names()``.

Both hold BY CONSTRUCTION today: ``get_tool_schemas()`` is the single root and
the other two accessors are derived from it. This module keeps that public
accessor contract covered without reaching into private declaration tables.

Hermetic: registry construction and schema generation are pure Python (no
model download, no network).

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/action_space/test_tool_layer_invariants.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import ActionSpaceRegistry
from lite.core.tools.action_space import (
    LiteBBoxActionSet,
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
)
from lite.core.tools.extra_tools import (
    LiteAppLaunchToolSet,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
)
from lite.core.tools.schemas import tool_schema_name

register_all()

REGISTRY_KEYS = sorted(ActionSpaceRegistry.list_expanded())

#: The core sets are not registry keys but answer the same three questions, so
#: they are pinned by the same invariants.
CORE_TOOL_SETS = (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
    LiteBBoxActionSet,
    LiteFinishToolSet,
    LiteBrowserNavToolSet,
    LiteAppLaunchToolSet,
)


def _probe_universe(emitted_names: frozenset[str]) -> frozenset[str]:
    """Every emitted name plus one miss exercises the public lookup contract."""
    return emitted_names | frozenset({"definitely_not_a_tool"})


@pytest.mark.parametrize("key", REGISTRY_KEYS)
def test_invariant1_one_name_per_emitted_schema(key: str):
    """``len(get_tool_names()) == len(get_tool_schemas())`` — within the tool layer."""
    space = ActionSpaceRegistry.get(key)
    schemas = space.get_tool_schemas()
    names = space.get_tool_names()
    assert len(names) == len(schemas), (
        f"{key} ({type(space).__name__}) reports {len(names)} tool names "
        f"{sorted(names)} while emitting {len(schemas)} schemas "
        f"{[tool_schema_name(s) for s in schemas]} — get_tool_names() must answer the "
        f"TOOL question, not the action question"
    )
    assert names == frozenset(tool_schema_name(s) for s in schemas)


@pytest.mark.parametrize("key", REGISTRY_KEYS)
def test_invariant2_a_name_has_a_schema_exactly_when_it_is_emitted(key: str):
    """``get_tool_schema(n)`` non-None iff ``n in get_tool_names()``."""
    space = ActionSpaceRegistry.get(key)
    names = space.get_tool_names()
    for name in sorted(_probe_universe(names)):
        schema = space.get_tool_schema(name)
        assert (schema is not None) == (name in names), (
            f"{key} ({type(space).__name__}): get_tool_schema({name!r}) is "
            f"{'a schema' if schema is not None else 'None'} while the name is "
            f"{'in' if name in names else 'NOT in'} get_tool_names()"
        )
        if schema is not None:
            assert tool_schema_name(schema) == name


@pytest.mark.parametrize("key", REGISTRY_KEYS)
def test_both_receivers_answer_the_same_tool_layer(key: str):
    """The accessors are classmethods: instance and class receivers agree.

    Production reads both ways ``[the adapter pipeline calls
    ``type(action_space).filter_*`` while the registry hands out INSTANCES]``,
    so a space whose class-level call diverged — or raised — would satisfy the
    invariants on one receiver and break them on the other.
    """
    space = ActionSpaceRegistry.get(key)
    cls = type(space)
    assert [tool_schema_name(s) for s in cls.get_tool_schemas()] == [
        tool_schema_name(s) for s in space.get_tool_schemas()
    ]
    assert cls.get_tool_names() == space.get_tool_names()


@pytest.mark.parametrize(
    "tool_set", CORE_TOOL_SETS, ids=lambda t: t.__name__,
)
def test_core_tool_sets_satisfy_both_invariants(tool_set: type):
    """The same two invariants on the core sets, which carry no registry key."""
    schemas = tool_set.get_tool_schemas()
    names = tool_set.get_tool_names()
    assert len(names) == len(schemas)
    assert names == frozenset(tool_schema_name(s) for s in schemas)
    for name in sorted(_probe_universe(names)):
        assert (tool_set.get_tool_schema(name) is not None) == (name in names)


def test_batched_sets_separate_the_two_layers():
    """The divergence the invariants must NOT close: 14 actions, 1 tool.

    If this ever collapses to equality, ``get_tool_names()`` has drifted back
    to answering the action question and the two invariants above become
    vacuous on exactly the classes that need them.
    """
    assert LiteDesktopActionSet.get_tool_names() == frozenset({"computer"})
    assert len(LiteDesktopActionSet.get_action_names()) == 14
    assert LiteMobileActionSet.get_tool_names() == frozenset({"mobile"})
    assert len(LiteMobileActionSet.get_action_names()) == 9
    # LitePointActionSet only LOOKS like an identity: it is N=1, not a wrapper.
    assert LitePointActionSet.get_tool_names() == LitePointActionSet.get_action_names()
