"""Anti-drift: the three legal tool-name categories at every parse boundary.

At every parse boundary the set of legal tool names is exactly three DISJOINT
categories, and each must route to its OWN handler:

1. the native GUI surface — the provider's native computer tool (its actions
   arrive on the wire as ``computer_call`` / ``tool_use{name="computer"}``);
2. the env's ``metadata.extra_tool_schemas`` — passed through VERBATIM, since
   the env owns those semantics;
3. the agent's OWN declared function tools (``_build_tools``) — routed through
   the action space's canonical conversion (the mobile provider tools, and the
   click-shaped tool the grounding agents declare).

Only a name in NONE of the three is an error.

This has now broken twice in opposite directions, so the invariant itself is
pinned here rather than one bug's symptom:

* I3 — the GPT parser accepted only category 2, so the grounding agent's own
  ``click`` tool came back as ``undeclared function_call click`` and
  ``gpt-5.5 x osworld_g`` scored 0.0 on 16/16 tasks.
* C7 — the mirror image on the adapter side: canonical-action names were
  routed into the GUI action space without checking category 2 first, so
  ``webharbor.webvoyager/som.yaml`` raised ``TypeError`` every turn.

The categories are DERIVED from what each agent actually declares
(``_build_tools`` / the action space / ``metadata.extra_tool_schemas``), never
from a hardcoded name list — so a family that grows a new tool is covered by
construction.

RECORDED PRIVATE-HELPER EXCEPTION (``_build_tools``). Elsewhere an agent's
advertised tools are read through ``sample()`` and the mocked provider request
payload. That route is unreachable here: ``FAMILIES`` is built at IMPORT time to
feed ``@pytest.mark.parametrize``, and ``sample()`` is a coroutine that needs an
event loop, a ``monkeypatch`` fixture, and an env — none of which exist at
collection time. There is no public, synchronous read of an agent's declared
tool surface, so the derivation calls ``_build_tools`` directly.

Deletion criterion for this exception: delete it once the agents expose a public
synchronous accessor for their declared tool surface (i.e. ``_build_tools``
becomes public, or an equivalent public projection entrypoint exists), and
switch the table to that accessor. Restructuring the table to run a mocked
``sample()`` per family is NOT the intended fix — it would make a parse-boundary
test depend on the whole rollout loop.

Run:
    uv run pytest tests/agents/core/adapter/test_tool_name_categories.py -v
"""

from __future__ import annotations

import inspect
import json
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from lite.core import LiteCUAMetadata
from lite.core.messages.final import pop_model_output_error
from lite.core.tools import make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

_RESOLUTION = (1024, 768)
_UNKNOWN_NAME = "definitely_not_a_declared_tool"

# The env-supplied extra tool every family in the table is built with. Shape
# matches what ``osworld_g`` / ``osworld`` actually advertise.
_EXTRA_TOOL = make_tool_schema(
    "report_infeasible",
    description="Report that the task cannot be completed.",
    parameters={
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
    },
)
_EXTRA_ARGS = {"reason": "no such element"}


# -----------------------------------------------------------------------------
# Deriving call arguments from a declared JSON schema
# -----------------------------------------------------------------------------

def _dummy_value(prop: dict[str, Any]) -> Any:
    """A schema-valid value for one declared property.

    Keeps the table derived-by-construction: a family that grows a tool gets
    exercised without anyone hand-writing its arguments.
    """
    if prop.get("enum"):
        return prop["enum"][0]
    kind = prop.get("type")
    if kind == "integer":
        return 10
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if kind == "array":
        items = prop.get("items") or {"type": "string"}
        default_n = 2 if items.get("type") in {"integer", "number"} else 1
        n = max(int(prop.get("minItems") or default_n), default_n)
        return [_dummy_value(items) for _ in range(n)]
    if kind == "object":
        return {}
    return "x"


def _dummy_args(parameters: dict[str, Any]) -> dict[str, Any]:
    """Arguments covering every REQUIRED property of a tool schema."""
    props = parameters.get("properties") or {}
    required = parameters.get("required") or list(props)
    return {name: _dummy_value(props.get(name) or {}) for name in required}


def _value_for_hint(hint: Any) -> Any:
    """A value satisfying a python type hint on an action-space method.

    The native GUI surface of the desktop families has no JSON schema — the
    provider owns the native computer tool — so arguments for those names come
    from the action-space method signature instead. Still derived, still no
    hardcoded per-family name lists.
    """
    origin = typing.get_origin(hint)
    if origin is typing.Literal:
        return typing.get_args(hint)[0]
    if origin in (list, tuple):
        (item,) = typing.get_args(hint) or (int,)
        return [_value_for_hint(item), _value_for_hint(item)]
    if origin is not None:  # X | None, Union[...]
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if args:
            return _value_for_hint(args[0])
    if hint is bool:
        return True
    if hint is int:
        return 10
    if hint is float:
        return 1.0
    return "x"


def _args_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Arguments for every REQUIRED parameter of an action-space method."""
    hints = typing.get_type_hints(fn)
    return {
        name: _value_for_hint(hints.get(name, str))
        for name, param in inspect.signature(fn).parameters.items()
        if param.default is inspect.Parameter.empty
    }


def _names(calls: list[dict[str, Any]]) -> list[str]:
    return [tool_call_name(call) for call in calls]


# -----------------------------------------------------------------------------
# Family table
# -----------------------------------------------------------------------------

@dataclass
class _Family:
    """One parse boundary, with its three categories derived from the agent."""

    id: str
    agent: Any
    # wire-payload builders → LiteMessage
    parse_native: Callable[[str, dict[str, Any]], dict[str, Any]] | None
    parse_function: Callable[[str, dict[str, Any]], dict[str, Any]]
    # what the action space alone would produce for that name — the oracle for
    # "routed through the action space" (vs. "passed through verbatim").
    oracle: Callable[[str, dict[str, Any]], list[dict[str, Any]]]
    # declared surface, derived from the agent
    native_tool_declared: bool
    agent_function_names: frozenset[str]
    tool_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def extra_names(self) -> frozenset[str]:
        return frozenset({tool_schema_name(_EXTRA_TOOL)})

    def args_for(self, name: str) -> dict[str, Any]:
        """Call arguments for ``name``, derived from its DECLARED schema when the
        agent declares one, else from the action-space method signature."""
        if name in self.tool_parameters:
            return _dummy_args(self.tool_parameters[name])
        method = getattr(type(self.agent.action_space), name, None)
        return _args_from_signature(method) if callable(method) else {}


def _metadata(platform: str, task_type: str) -> LiteCUAMetadata:
    return LiteCUAMetadata(dims=(platform, task_type), extra_tool_schemas=[dict(_EXTRA_TOOL)])


# -- GPT ----------------------------------------------------------------------

def _gpt_function_items(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "type": "function_call",
        "id": "fc_1",
        "call_id": "fc_1",
        "name": name,
        "arguments": json.dumps(args),
    }]


def _gpt_families() -> list[_Family]:
    from lite.agents.models.gpt.agent import (
        GPTDesktopGroundingPointAgent,
        GPTDesktopUseAgent,
        GPTMobileUseAgent,
    )
    from lite.agents.models.gpt.utils.parse import (
        parse_gpt_mobile_output_items_with_provenance,
    )

    families: list[_Family] = []

    for agent_cls, agent_id, platform, task_type in (
        (GPTDesktopUseAgent, "gpt@desktop@use", "desktop", "use"),
        (GPTDesktopGroundingPointAgent, "gpt@desktop@grounding.point",
         "desktop", "grounding.point"),
    ):
        agent = agent_cls(model_id="gpt-5.5", metadata=_metadata(platform, task_type))
        tools = agent._build_tools()
        space = agent.action_space
        families.append(_Family(
            id=agent_id,
            agent=agent,
            parse_native=lambda name, args, a=agent: a._parse_output_items(
                [{"type": "computer_call", "call_id": "cc_1",
                  "actions": [{"type": name, **args}]}],
                resolution=_RESOLUTION,
            ).message,
            parse_function=lambda name, args, a=agent: a._parse_output_items(
                _gpt_function_items(name, args), resolution=_RESOLUTION,
            ).message,
            oracle=lambda name, args, s=space: s.convert_tool_calls_from_agent(
                [{"type": name, **args}], resolution=_RESOLUTION,
            ),
            native_tool_declared=any(t.get("type") == "computer" for t in tools),
            agent_function_names=agent._declared_agent_tool_names(),
            tool_parameters={
                **{tool_schema_name(s): tool_schema_parameters(s)
                   for s in type(space).get_tool_schemas()},
                **{t["name"]: t.get("parameters", {})
                   for t in tools if t.get("type") == "function"},
            },
        ))

    mobile = GPTMobileUseAgent(model_id="gpt-5.5", metadata=_metadata("mobile", "use"))
    mobile_space = mobile.action_space
    families.append(_Family(
        id="gpt@mobile@use",
        agent=mobile,
        parse_native=None,  # no native computer tool on the mobile path
        parse_function=lambda name, args, a=mobile: (
            parse_gpt_mobile_output_items_with_provenance(
                _gpt_function_items(name, args),
                action_space=a.action_space,
                resolution=_RESOLUTION,
                extra_tool_names=a._extra_tool_names(),
                declared_agent_tool_names=a._declared_agent_tool_names(),
            ).message
        ),
        oracle=lambda name, args, s=mobile_space: s.convert_tool_calls_from_agent(
            [{"name": name, "arguments": args}], resolution=_RESOLUTION,
        ),
        native_tool_declared=False,
        agent_function_names=mobile._declared_agent_tool_names(),
        tool_parameters={
            t["name"]: t.get("parameters", {})
            for t in mobile._build_tools() if t.get("type") == "function"
        },
    ))
    return families


# -- Claude -------------------------------------------------------------------

def _claude_response(name: str, arguments: dict[str, Any]) -> Any:
    """Minimal ``litellm.acompletion`` response carrying one tool_call."""
    tc = SimpleNamespace(
        model_dump=lambda: {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    )
    msg = SimpleNamespace(content="", tool_calls=[tc], role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _claude_families() -> list[_Family]:
    from lite.agents.models.claude.agent import (
        ClaudeDesktopGroundingPointAgent,
        ClaudeDesktopUseAgent,
        ClaudeMobileUseAgent,
        _extra_tool_names,
        _get_tool_config_for_model,
    )
    from lite.agents.models.claude.utils.parse import (
        parse_mobile_response_with_provenance,
        parse_response_with_provenance,
    )

    families: list[_Family] = []

    for agent_cls, agent_id, platform, task_type in (
        (ClaudeDesktopUseAgent, "claude@desktop@use", "desktop", "use"),
        (ClaudeDesktopGroundingPointAgent, "claude@desktop@grounding.point",
         "desktop", "grounding.point"),
    ):
        agent = agent_cls(
            model_id="claude-opus-4-6", metadata=_metadata(platform, task_type),
        )
        tools = agent._build_tools(
            _get_tool_config_for_model(agent.model_id), *_RESOLUTION,
        )
        space = agent.action_space
        extra_names = frozenset({tool_schema_name(_EXTRA_TOOL)})
        function_names = frozenset(
            t["function"]["name"] for t in tools
            if t.get("type") == "function" and "function" in t
        ) | frozenset(
            t["name"] for t in tools if t.get("type") != "function" and "name" in t
        )
        families.append(_Family(
            id=agent_id,
            agent=agent,
            parse_native=lambda name, args, a=agent: parse_response_with_provenance(
                _claude_response("computer", {"action": name, **args}),
                scale_x=1.0, scale_y=1.0, action_space=a.action_space,
                resolution=_RESOLUTION, extra_tool_names=_extra_tool_names(a.metadata),
            ).message,
            parse_function=lambda name, args, a=agent: parse_response_with_provenance(
                _claude_response(name, args),
                scale_x=1.0, scale_y=1.0, action_space=a.action_space,
                resolution=_RESOLUTION, extra_tool_names=_extra_tool_names(a.metadata),
            ).message,
            oracle=lambda name, args, s=space: s.convert_tool_calls_from_agent(
                [{"action": name, **args}], resolution=_RESOLUTION,
            ),
            native_tool_declared=any(
                (t.get("function") or {}).get("name") == "computer" for t in tools
            ),
            agent_function_names=function_names - extra_names,
            tool_parameters={
                t["function"]["name"]: t["function"].get("parameters", {})
                for t in tools if t.get("type") == "function" and "function" in t
            },
        ))

    mobile = ClaudeMobileUseAgent(
        model_id="claude-opus-4-6", metadata=_metadata("mobile", "use"),
    )
    mobile_tools = mobile._build_tools()
    mobile_space = mobile.action_space
    mobile_function_names = frozenset(
        t["function"]["name"]
        for t in mobile_tools
        if t.get("type") == "function" and "function" in t
    )
    families.append(_Family(
        id="claude@mobile@use",
        agent=mobile,
        parse_native=None,  # no native computer tool on the mobile path
        parse_function=lambda name, args, a=mobile: parse_mobile_response_with_provenance(
            _claude_response(name, args),
            scale_x=1.0, scale_y=1.0, action_space=a.action_space,
            resolution=_RESOLUTION, extra_tool_names=_extra_tool_names(a.metadata),
        ).message,
        oracle=lambda name, args, s=mobile_space: s.convert_tool_calls_from_agent(
            [{"name": name, "arguments": args}], resolution=_RESOLUTION,
        ),
        native_tool_declared=False,
        agent_function_names=mobile_function_names - frozenset({tool_schema_name(_EXTRA_TOOL)}),
        tool_parameters={
            t["function"]["name"]: t["function"].get("parameters", {})
            for t in mobile_tools
            if t.get("type") == "function" and "function" in t
        },
    ))
    return families


FAMILIES: list[_Family] = _gpt_families() + _claude_families()
_IDS = [f.id for f in FAMILIES]

# Every family in the table must reach the boundary through the agent it
# declares, not through a hand-rolled name list: fail loudly if a family ends
# up with NO declared surface at all (i.e. the derivation silently broke).
assert all(f.native_tool_declared or f.agent_function_names for f in FAMILIES)


def _pairs() -> list[tuple[_Family, str]]:
    """(family, agent-declared function tool) for every declared function tool."""
    return [(f, name) for f in FAMILIES for name in sorted(f.agent_function_names)]


# -----------------------------------------------------------------------------
# Category 1 — the native GUI surface
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_native_action_name_routes_through_the_action_space(family: _Family) -> None:
    if not family.native_tool_declared:
        pytest.skip(f"{family.id} declares no native computer tool")
    name = sorted(type(family.agent.action_space).get_declared_action_schema_names())[0]
    args = family.args_for(name)

    expected = family.oracle(name, args)
    assert expected, f"{family.id}: action space dropped native {name}({args})"

    msg = family.parse_native(name, args)
    assert pop_model_output_error(msg) is None
    assert _names(msg["tool_calls"]) == _names(expected)


# -----------------------------------------------------------------------------
# Category 2 — the env's extra_tool_schemas
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_env_extra_tool_name_passes_through_verbatim(family: _Family) -> None:
    name = tool_schema_name(_EXTRA_TOOL)
    msg = family.parse_function(name, _EXTRA_ARGS)

    assert pop_model_output_error(msg) is None
    calls = msg["tool_calls"]
    assert len(calls) == 1, calls
    # Verbatim: the env owns the semantics, so NOT coord-converted / rewrapped.
    assert tool_call_name(calls[0]) == name
    assert tool_call_arguments(calls[0]) == _EXTRA_ARGS


# -----------------------------------------------------------------------------
# Category 3 — the agent's OWN declared function tools
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family,name", _pairs(), ids=[f"{f.id}-{n}" for f, n in _pairs()],
)
def test_agent_declared_function_tool_routes_through_the_action_space(
    family: _Family, name: str,
) -> None:
    """The category I3 deleted: the agent's own function tool is NOT an extra
    and NOT a native computer action, and must still be converted, not rejected.
    """
    assert name not in family.extra_names, "categories must stay disjoint"
    args = family.args_for(name)

    expected = family.oracle(name, args)
    assert expected, f"{family.id}: action space dropped declared {name}({args})"

    msg = family.parse_function(name, args)
    assert pop_model_output_error(msg) is None, (
        f"{family.id} rejected its OWN declared tool {name!r}"
    )
    assert _names(msg["tool_calls"]) == _names(expected)


# -----------------------------------------------------------------------------
# A name in NONE of the three — the only legal error
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_name_in_no_category_is_a_model_output_error(family: _Family) -> None:
    assert _UNKNOWN_NAME not in family.extra_names
    assert _UNKNOWN_NAME not in family.agent_function_names
    assert _UNKNOWN_NAME not in (
        type(family.agent.action_space).get_declared_action_schema_names()
    )

    msg = family.parse_function(_UNKNOWN_NAME, {})
    assert msg["tool_calls"] == []
    assert _UNKNOWN_NAME in (pop_model_output_error(msg) or "")


# -----------------------------------------------------------------------------
# The categories are disjoint by construction
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_categories_are_disjoint(family: _Family) -> None:
    assert not (family.extra_names & family.agent_function_names)
