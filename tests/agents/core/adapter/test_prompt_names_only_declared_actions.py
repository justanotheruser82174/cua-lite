"""A system prompt may only order action names its own family declares.

Run:
    uv run pytest tests/agents/core/adapter/test_prompt_names_only_declared_actions.py -q
"""

from __future__ import annotations

import re

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import BaseActionSpace
from lite.agents.core.adapter.base import AgentAdapterRegistry, BaseAgentAdapter
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.core.tools.extra_tools import (
    APP_LAUNCH_TOOL_NAME,
    FINISH_TOOL_ORDER,
    LiteBrowserNavToolSet,
)

register_all()

# =============================================================================
# Adapter enumeration
# =============================================================================
# Every registered adapter, from the registry itself, so a newly registered
# family is covered by being registered — no hand-written family list to drift.
# Genuine regex keys live in ``list_patterns()`` and literal-dot keys such as
# ``qwen3_vl@mobile@grounding.point`` in ``list()``; ``get_class`` resolves both
# forms, so the union is the exhaustive class surface.
ADAPTER_KEYS = sorted(
    set(AgentAdapterRegistry.list()) | set(AgentAdapterRegistry.list_patterns())
)

# Enumeration canary: an empty parametrize passes vacuously.
assert len(ADAPTER_KEYS) > 10, f"adapter enumeration found only {ADAPTER_KEYS}"


def _lite_action_name_table(action_space: object) -> dict:
    """Read a family's Lite-action-name -> provider-value table by its declared name.

    There is one spelling per layer: wrapper families declare
    ``LITE_ACTION_NAME_TO_<FAMILY>_ACTION_VALUES`` and provider-flat families declare
    ``LITE_ACTION_NAME_TO_<FAMILY>_PROVIDER_FLAT_TOOL_NAMES``. Both share the prefix,
    so this census reads the prefix rather than naming every family here.
    """
    for name in dir(action_space):
        if name.startswith("LITE_ACTION_NAME_TO_"):
            return getattr(action_space, name) or {}
    return {}


def _extra_tool_names_table(action_space: object) -> dict:
    """Read a family's provider-value -> Lite extra-tool-name table by its declared name.

    Spelled ``<FAMILY>_ACTION_VALUE_TO_EXTRA_TOOL_NAMES`` for wrapper families and
    ``<FAMILY>_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES`` for provider-flat ones.
    """
    for name in dir(action_space):
        if name.endswith("_TO_EXTRA_TOOL_NAMES"):
            return getattr(action_space, name) or {}
    return {}

def _adapter(adapter_key: str) -> BaseAgentAdapter:
    return AgentAdapterRegistry.get_class(adapter_key)()


#: Adapters that ship prompt prose — the only ones with text that can drift.
#: Derived, so a newly registered family is gated by being registered.
PROMPT_BEARING_KEYS = [
    key for key in ADAPTER_KEYS
    if _adapter(key).system_prompt or getattr(_adapter(key), "user_prompt_template", None)
]

# Enumeration canary: a shrunken parametrize would pass vacuously.
assert len(PROMPT_BEARING_KEYS) > 10, (
    f"only {len(PROMPT_BEARING_KEYS)} of {len(ADAPTER_KEYS)} adapters carry a "
    "system prompt — did adapter prompt wiring change?"
)


# =============================================================================
# What a family declares it can spell
# =============================================================================
#: Standalone canonical tools, owned by core (``lite/core/tools/extra_tools.py``)
#: and stamped onto a sample as ``extra_tool_schemas``.
CANONICAL_STANDALONE_TOOLS = (
    frozenset(FINISH_TOOL_ORDER)
    | LiteBrowserNavToolSet.get_tool_names()
    | {APP_LAUNCH_TOOL_NAME}
)


def _declared_action_names(action_space: type[BaseActionSpace]) -> frozenset[str]:
    """Every name this family can spell on its own wire.

    Three declaration surfaces, all read off the class:
      - ``LITE_ACTION_NAME_TO_*`` values: native GUI enum entries.
      - ``*_TO_EXTRA_TOOL_NAMES`` keys: native entries that spell a
        standalone canonical tool (qwen3_vl ``answer`` -> canonical ``response``).
      - declared one-schema-per-tool names, for families whose actions are top-level
        tools rather than one wrapper with an ``action`` enum.

    Plus the canonical names themselves, but ONLY for a family with an empty
    semantic table: an empty table means the family never renamed the canonical
    tools, so the extra-tool schemas reach the wire under their canonical names.
    A family with a PARTIAL table has answered the question — EvoCUA declares
    ``terminate`` and nothing else, and its renderer raises on canonical
    ``response`` — so a canonical name outside a populated table is NOT
    spellable and must not be admitted here.
    """
    declared: set[str] = set()
    lite_to_enum = _lite_action_name_table(action_space)
    native_semantic = _extra_tool_names_table(action_space)
    for entries in lite_to_enum.values():
        declared.update(entries)
    declared.update(native_semantic)
    declared.update(action_space.get_declared_action_schema_names())
    if not native_semantic:
        declared.update(CANONICAL_STANDALONE_TOOLS)
    return frozenset(declared)


#: The candidate vocabulary: every name SOME registered family declares, plus
#: the canonical names. Matching a prompt against this set instead of parsing
#: English is what makes the check mechanical — and it is exactly the bug class,
#: since a prompt that orders an undeclared action is almost always ordering a
#: sibling family's spelling of the same verb.
CANDIDATE_VALID_ACTION_NAMES = frozenset(
    name
    for adapter_key in ADAPTER_KEYS
    for name in _declared_action_names(type(_adapter(adapter_key).action_space))
) | CANONICAL_STANDALONE_TOOLS


# =============================================================================
# What a prompt ORDERS
# =============================================================================
# Instruction shapes, read off the real prompts. Each is a position where the
# prose tells the model to put a name in the emitted call's action field:
#   - ``action=terminate``        lite / qwen3_vl / qwen3_5 / evocua
#   - ``action:CLICK``            step_gui's tab-separated DSL
#   - ``{"action": "click"}``     mai_ui's JSON action-space listing
#   - ``def click(``              a pyautogui-style action-space listing
#   - ``click(...)``              ui_tars' prompt-string action-space listing
#   - ``computer.terminate(``     pyautogui-style prose
# Deliberately NOT a shape: a name inside a JSON tool schema. Those are rendered
# FROM the action space (Qwen's ``<tools>`` block), so they cannot drift from it
# the way hand-written prose can.
_INSTRUCTION_SHAPES: tuple[tuple[str, str], ...] = (
    ("action={name}", r"action[ \t]*=[ \t]*{name}\b"),
    ("action:{name}", r"action[ \t]*:[ \t]*{name}\b"),
    ('"action": "{name}"', r'"action"[ \t]*:[ \t]*"{name}"'),
    ("def {name}(", r"\bdef[ \t]+{name}[ \t]*\("),
    ("{name}(", r"(?m)^[ \t]*{name}[ \t]*\("),
    ("computer.{name}(", r"\bcomputer\.{name}[ \t]*\("),
)


def _ordered_action_names(prompt: str, candidates: frozenset[str]) -> dict[str, str]:
    """Candidate names the prompt puts in an instruction position -> the shape."""
    ordered: dict[str, str] = {}
    for name in sorted(candidates):
        for label, pattern in _INSTRUCTION_SHAPES:
            if re.search(pattern.format(name=re.escape(name)), prompt):
                ordered[name] = label.format(name=name)
                break
    return ordered


# =============================================================================
# Known violations
# =============================================================================
# Real defects of exactly the class this file gates, in prompts outside the
# change that added the gate. Compared for EQUALITY, so fixing one reddens here
# and the entry gets deleted rather than rotting into a permanent exemption.
# Empty: every registered prompt currently orders only names its own family
# declares.
KNOWN_UNDECLARED: dict[str, frozenset[str]] = {}


# =============================================================================
# The gate
# =============================================================================

def _platform_for_prompt_sample(adapter_key: str, adapter: BaseAgentAdapter) -> str:
    if "@mobile" in adapter_key:
        return "mobile"
    platform = getattr(adapter.metadata, "platform", None)
    return "mobile" if platform == "mobile" else "desktop"


def _prompt_sample(adapter_key: str, adapter: BaseAgentAdapter) -> LiteSample:
    platform = _platform_for_prompt_sample(adapter_key, adapter)
    task_type = getattr(adapter.metadata, "task_type", "use")
    image = Image.new("RGB", (32, 32), color=(30, 0, 0))
    user = {
        "role": "user",
        "content": [
            {"type": "image", "index": 0},
            {"type": "text", "text": "Open Settings and report the result."},
        ],
    }
    if task_type == "grounding.point":
        tool_call = make_tool_call(
            "point",
            {"coordinate": [100, 200]},
            call_id="call_0000",
        )
    elif platform == "mobile":
        tool_call = make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": [100, 200]}]},
            call_id="call_0000",
        )
    else:
        tool_call = make_tool_call(
            "computer",
            {"actions": [{"action": "click", "coordinate": [100, 200]}]},
            call_id="call_0000",
        )
    assistant = {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "Click the target."}],
        "tool_calls": [tool_call],
    }
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, task_type)),
        images=[image],
        messages=[user, assistant],
    )


def _rendered_prompt(adapter_key: str, adapter: BaseAgentAdapter) -> str:
    """Model-visible text from the real first rendered step.

    ``unroll`` includes the target assistant message in SFT steps; that target is
    not part of the prompt, so this reads only the input messages preceding it.
    """
    step = adapter.unroll(_prompt_sample(adapter_key, adapter)).steps[0]
    input_messages = step[:-1] if step and step[-1].get("role") == "assistant" else step
    texts: list[str] = []
    for message in input_messages:
        for part in message.get("content") or []:
            if part.get("type") == "text":
                texts.append(part.get("text") or "")
    return "\n".join(texts)


@pytest.mark.parametrize("adapter_key", PROMPT_BEARING_KEYS)
def test_system_prompt_orders_only_declared_actions(adapter_key: str) -> None:
    """An ordered-but-undeclared action is dropped on parse: the model spends a
    step on a name nothing accepts, and a finish verb it never lands runs the
    episode to ``max_steps``."""
    adapter = _adapter(adapter_key)
    prompt = _rendered_prompt(adapter_key, adapter)
    declared = _declared_action_names(type(adapter.action_space))
    ordered = _ordered_action_names(prompt, CANDIDATE_VALID_ACTION_NAMES)
    undeclared = {
        name: shape for name, shape in ordered.items() if name not in declared
    }

    assert frozenset(undeclared) == KNOWN_UNDECLARED.get(adapter_key, frozenset()), (
        f"{adapter_key}: system prompt orders actions its action space "
        f"({type(adapter.action_space).__name__}) does not declare: {undeclared}. "
        f"Declared: {sorted(declared)}"
    )


def test_extractor_finds_names_the_prompts_do_order() -> None:
    """Guards the gate above against passing vacuously: if the extractor stops
    matching real prompts it reports no ordered names and can catch nothing."""
    expected = {
        r"lite@(desktop|browser)@use": {"response", "terminate"},
        r"evocua@(desktop|browser)@use": {"terminate"},
        "step_gui@mobile@use": {"CLICK", "TYPE", "WAIT", "SLIDE", "LONGPRESS"},
        r"ui_tars@(desktop|browser)@use": {"click", "wait"},
        r"ui_tars_15_v1@(desktop|browser)@use": {"click", "wait"},
    }
    for adapter_key, names in expected.items():
        adapter = _adapter(adapter_key)
        ordered = _ordered_action_names(
            _rendered_prompt(adapter_key, adapter), CANDIDATE_VALID_ACTION_NAMES
        )
        assert names <= set(ordered), f"{adapter_key}: extracted only {sorted(ordered)}"


def test_gate_rejects_a_prompt_that_orders_a_sibling_familys_spelling() -> None:
    """Discrimination canary: the exact regression that motivated the gate —
    EvoCUA's prompt ordering qwen3_vl's ``answer``, which EvoCUA drops on parse."""
    adapter = _adapter(r"evocua@(desktop|browser)@use")
    declared = _declared_action_names(type(adapter.action_space))
    injected = adapter.system_prompt.replace(
        "action=terminate", "action=answer"
    )
    assert injected != adapter.system_prompt

    ordered = _ordered_action_names(injected, CANDIDATE_VALID_ACTION_NAMES)
    assert {name for name in ordered if name not in declared} == {"answer"}
