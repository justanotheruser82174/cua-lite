"""
Cross-family tests for the ``grounding.point`` harness.

Covers all 6 families that ship a ``grounding.point`` adapter (qwen3_vl,
qwen3_5, evocua, ui_tars, ui_tars_15_v1, mai_ui). Each family is tested
on three axes:

  1. Registry resolution — ``AgentAdapterRegistry.get(<family>:<plat>:grounding.point)``
     returns the expected adapter class, with an action_space that is the
     trimmed grounding variant (NOT the full nav action_space).
  2. Schema / wire-format shape — schemas (or text format for UI-TARS / MAI-UI)
     advertise ONLY the single click action; refusal lives on the env's
     ``report_infeasible`` extra tool, NOT on the action enum.
  3. Round-trip — cua-lite ``LitePointActionSpace.point(coord)`` →
     family-native click → cua-lite ``point(coord)``. Coordinates preserved.
  4. Extra-tool pass-through — agent-emitted ``report_infeasible`` round-trips
     unchanged (uniform pattern across all 6 families).

Run:
    uv run pytest tests/agents/core/action_space/test_grounding_point.py -n 32
"""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import (
    BASH_SCHEMA,
    agent_adapter_for,
    tool_names,
)

import lite.agents.models.evocua.adapter  # noqa: F401
import lite.agents.models.mai_ui.adapter  # noqa: F401
import lite.agents.models.qwen3_5.adapter  # noqa: F401

# Trigger registration of every grounding.point adapter.
import lite.agents.models.qwen3_vl.adapter  # noqa: F401
import lite.agents.models.ui_tars.adapter  # noqa: F401
import lite.agents.models.ui_tars_15_v1.adapter  # noqa: F401
from lite.agents.core.action_space.base import LitePointActionSpace
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.claude.action_space import ClaudeDesktopGroundingPointActionSpace
from lite.agents.models.evocua.action_space import (
    EvoCUADesktopGroundingPointActionSpace,
)
from lite.agents.models.gpt.action_space import GPTDesktopGroundingPointActionSpace
from lite.agents.models.mai_ui.action_space import (
    MAIUIGroundingPointActionSpace,
)
from lite.agents.models.qwen3_5.action_space import (
    Qwen3_5DesktopGroundingPointActionSpace,
    Qwen3_5MobileGroundingPointActionSpace,
)
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopGroundingPointActionSpace,
    Qwen3VLMobileGroundingPointActionSpace,
)
from lite.agents.models.ui_tars.action_space import (
    UITarsDesktopGroundingPointActionSpace,
    UITarsMobileGroundingPointActionSpace,
)
from lite.agents.models.ui_tars_15_v1.action_space import (
    UITars15V1DesktopGroundingPointActionSpace,
    UITars15V1MobileGroundingPointActionSpace,
)
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected_adapter,expected_space",
    [
        (
            "qwen3_vl@desktop@grounding.point",
            "Qwen3VLDesktopGroundingPointAdapter",
            Qwen3VLDesktopGroundingPointActionSpace,
        ),
        (
            "qwen3_vl@browser@grounding.point",
            "Qwen3VLDesktopGroundingPointAdapter",
            Qwen3VLDesktopGroundingPointActionSpace,
        ),
        (
            "qwen3_vl@mobile@grounding.point",
            "Qwen3VLMobileGroundingPointAdapter",
            Qwen3VLMobileGroundingPointActionSpace,
        ),
        (
            "qwen3_5@desktop@grounding.point",
            "Qwen3_5DesktopGroundingPointAdapter",
            Qwen3_5DesktopGroundingPointActionSpace,
        ),
        (
            "qwen3_5@browser@grounding.point",
            "Qwen3_5DesktopGroundingPointAdapter",
            Qwen3_5DesktopGroundingPointActionSpace,
        ),
        (
            "qwen3_5@mobile@grounding.point",
            "Qwen3_5MobileGroundingPointAdapter",
            Qwen3_5MobileGroundingPointActionSpace,
        ),
        (
            "evocua@desktop@grounding.point",
            "EvoCUADesktopGroundingPointAdapter",
            EvoCUADesktopGroundingPointActionSpace,
        ),
        (
            "evocua@browser@grounding.point",
            "EvoCUADesktopGroundingPointAdapter",
            EvoCUADesktopGroundingPointActionSpace,
        ),
        (
            "ui_tars@desktop@grounding.point",
            "UITarsDesktopGroundingPointAdapter",
            UITarsDesktopGroundingPointActionSpace,
        ),
        (
            "ui_tars@browser@grounding.point",
            "UITarsDesktopGroundingPointAdapter",
            UITarsDesktopGroundingPointActionSpace,
        ),
        (
            "ui_tars@mobile@grounding.point",
            "UITarsMobileGroundingPointAdapter",
            UITarsMobileGroundingPointActionSpace,
        ),
        (
            "ui_tars_15_v1@desktop@grounding.point",
            "UITars15V1DesktopGroundingPointAdapter",
            UITars15V1DesktopGroundingPointActionSpace,
        ),
        (
            "ui_tars_15_v1@browser@grounding.point",
            "UITars15V1DesktopGroundingPointAdapter",
            UITars15V1DesktopGroundingPointActionSpace,
        ),
        (
            "ui_tars_15_v1@mobile@grounding.point",
            "UITars15V1MobileGroundingPointAdapter",
            UITars15V1MobileGroundingPointActionSpace,
        ),
        (
            "mai_ui@desktop@grounding.point",
            "MAIUIGroundingPointAdapter",
            MAIUIGroundingPointActionSpace,
        ),
        (
            "mai_ui@browser@grounding.point",
            "MAIUIGroundingPointAdapter",
            MAIUIGroundingPointActionSpace,
        ),
        (
            "mai_ui@mobile@grounding.point",
            "MAIUIGroundingPointAdapter",
            MAIUIGroundingPointActionSpace,
        ),
    ],
)
def test_registry_resolves_grounding_point(key, expected_adapter, expected_space):
    """Each family's grounding.point key resolves to the trimmed adapter +
    a grounding-specific action_space (NOT the full navigation
    action_space).
    """
    adapter = AgentAdapterRegistry.get(key)
    assert type(adapter).__name__ == expected_adapter, (
        f"{key} -> {type(adapter).__name__}, expected {expected_adapter}"
    )
    assert isinstance(adapter.action_space, expected_space), (
        f"{key} action_space {type(adapter.action_space).__name__} "
        f"is not a {expected_space.__name__}"
    )


# ---------------------------------------------------------------------------
# Schema shape — Qwen3-VL family (computer_use / mobile_use, single enum)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,expected_tool_name,expected_enum",
    [
        (Qwen3VLDesktopGroundingPointActionSpace, "computer_use", ["left_click"]),
        (Qwen3VLMobileGroundingPointActionSpace, "mobile_use", ["click"]),
        (Qwen3_5DesktopGroundingPointActionSpace, "computer_use", ["left_click"]),
        (Qwen3_5MobileGroundingPointActionSpace, "mobile_use", ["click"]),
        (EvoCUADesktopGroundingPointActionSpace, "computer_use", ["left_click"]),
    ],
)
def test_qwen_family_trimmed_schema(cls, expected_tool_name, expected_enum):
    """Qwen3-VL / Qwen3.5 / EvoCUA grounding.point advertises a single tool
    with a single-action enum — no scroll / type / key / drag / terminate.
    Refusal lives on the env's ``report_infeasible`` extra tool.
    """
    schemas = cls().get_tool_schemas()
    assert len(schemas) == 1, f"expected exactly 1 tool, got {len(schemas)}"
    fn = schemas[0]
    assert tool_schema_name(fn) == expected_tool_name
    enum = tool_schema_parameters(fn)["properties"]["action"]["enum"]
    assert enum == expected_enum, (
        f"{cls.__name__} enum = {enum!r}, expected {expected_enum!r}. "
        f"Trimmed grounding schema must not include scroll / type / key / "
        f"terminate / answer / etc."
    )


def test_mai_ui_no_tool_schemas():
    """MAI-UI grounding format lives entirely in the system prompt's
    ``<grounding_think>`` / ``<answer>`` instructions; no tool schemas
    are advertised.
    """
    schemas = MAIUIGroundingPointActionSpace().get_tool_schemas()
    assert schemas == [], f"MAI-UI grounding action_space should advertise no tools, got {schemas}"


# ---------------------------------------------------------------------------
# Forward conversion: cua-lite point(coord) → family-native click
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,expected_name,expected_action",
    [
        (Qwen3VLDesktopGroundingPointActionSpace, "computer_use", "left_click"),
        (Qwen3VLMobileGroundingPointActionSpace, "mobile_use", "click"),
        (Qwen3_5DesktopGroundingPointActionSpace, "computer_use", "left_click"),
        (Qwen3_5MobileGroundingPointActionSpace, "mobile_use", "click"),
        (EvoCUADesktopGroundingPointActionSpace, "computer_use", "left_click"),
    ],
)
def test_qwen_family_forward_point_to_click(cls, expected_name, expected_action):
    """``point(coord)`` → ``<tool>(action=<click>, coordinate=coord)``."""
    asp = cls()
    lite = LitePointActionSpace.point(coordinate=[123, 456])
    agent = asp.convert_tool_calls_to_agent([lite])
    assert len(agent) == 1
    fn = agent[0]
    assert fn["name"] == expected_name
    assert fn["arguments"]["action"] == expected_action
    assert fn["arguments"]["coordinate"] == [123, 456]


@pytest.mark.parametrize(
    "cls",
    [
        UITarsDesktopGroundingPointActionSpace,
        UITarsMobileGroundingPointActionSpace,
        UITars15V1DesktopGroundingPointActionSpace,
        UITars15V1MobileGroundingPointActionSpace,
    ],
)
def test_ui_tars_family_forward_point_to_click(cls):
    """UI-TARS family: ``point(coord)`` → ``click(start_box=coord)``
    (pyautogui-style native form, NOT a tool_call wire format)."""
    asp = cls()
    lite = LitePointActionSpace.point(coordinate=[123, 456])
    agent = asp.convert_tool_calls_to_agent([lite])
    assert len(agent) == 1
    fn = agent[0]
    assert fn["name"] == "click"
    assert fn["arguments"]["start_box"] == [123, 456]


def test_mai_ui_forward_point_to_answer():
    """MAI-UI: ``point(coord)`` → synthetic ``answer(coordinate=coord)``
    (no ``mobile_use`` wrapper; the adapter renders this as the final
    ``<answer>{"coordinate":[x,y]}</answer>`` block)."""
    asp = MAIUIGroundingPointActionSpace()
    lite = LitePointActionSpace.point(coordinate=[123, 456])
    agent = asp.convert_tool_calls_to_agent([lite])
    assert len(agent) == 1
    fn = agent[0]
    assert fn["name"] == "answer"
    assert fn["arguments"]["coordinate"] == [123, 456]


# ---------------------------------------------------------------------------
# Reverse conversion: family-native click → cua-lite point(coord)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,tool_name,action",
    [
        (Qwen3VLDesktopGroundingPointActionSpace, "computer_use", "left_click"),
        (Qwen3VLMobileGroundingPointActionSpace, "mobile_use", "click"),
        (Qwen3_5DesktopGroundingPointActionSpace, "computer_use", "left_click"),
        (Qwen3_5MobileGroundingPointActionSpace, "mobile_use", "click"),
        (EvoCUADesktopGroundingPointActionSpace, "computer_use", "left_click"),
    ],
)
def test_qwen_family_reverse_click_to_point(cls, tool_name, action):
    """Agent ``<tool>(action=click, coord)`` → ``point(coord)``."""
    asp = cls()
    agent_call = {
        "name": tool_name,
        "arguments": {"action": action, "coordinate": [50, 60]},
    }
    lite = asp.convert_tool_calls_from_agent([agent_call])
    assert len(lite) == 1
    assert tool_call_name(lite[0]) == "point"
    assert tool_call_arguments(lite[0])["coordinate"] == [50, 60]


@pytest.mark.parametrize(
    "cls",
    [
        UITarsDesktopGroundingPointActionSpace,
        UITarsMobileGroundingPointActionSpace,
        UITars15V1DesktopGroundingPointActionSpace,
        UITars15V1MobileGroundingPointActionSpace,
    ],
)
def test_ui_tars_family_reverse_click_to_point(cls):
    """UI-TARS: ``click(start_box=coord)`` → ``point(coord)``."""
    asp = cls()
    agent_call = {
        "name": "click",
        "arguments": {"start_box": [50, 60]},
    }
    lite = asp.convert_tool_calls_from_agent([agent_call])
    assert len(lite) == 1
    assert tool_call_name(lite[0]) == "point"
    assert tool_call_arguments(lite[0])["coordinate"] == [50, 60]


def test_mai_ui_reverse_answer_to_point():
    """MAI-UI: ``answer(coordinate=coord)`` → ``point(coord)``."""
    asp = MAIUIGroundingPointActionSpace()
    agent_call = {
        "name": "answer",
        "arguments": {"coordinate": [50, 60]},
    }
    lite = asp.convert_tool_calls_from_agent([agent_call])
    assert len(lite) == 1
    assert tool_call_name(lite[0]) == "point"
    assert tool_call_arguments(lite[0])["coordinate"] == [50, 60]


# ---------------------------------------------------------------------------
# Round-trip identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        Qwen3VLDesktopGroundingPointActionSpace,
        Qwen3VLMobileGroundingPointActionSpace,
        Qwen3_5DesktopGroundingPointActionSpace,
        Qwen3_5MobileGroundingPointActionSpace,
        EvoCUADesktopGroundingPointActionSpace,
        UITarsDesktopGroundingPointActionSpace,
        UITarsMobileGroundingPointActionSpace,
        UITars15V1DesktopGroundingPointActionSpace,
        UITars15V1MobileGroundingPointActionSpace,
        MAIUIGroundingPointActionSpace,
    ],
)
def test_grounding_point_round_trip(cls):
    """``lite point(coord)`` → agent native → ``lite point(coord)`` is identity."""
    asp = cls()
    lite = LitePointActionSpace.point(coordinate=[321, 654])
    agent = asp.convert_tool_calls_to_agent([lite])
    back = asp.convert_tool_calls_from_agent(agent)
    assert len(back) == 1
    assert tool_call_name(back[0]) == "point"
    assert tool_call_arguments(back[0])["coordinate"] == [321, 654]


# ---------------------------------------------------------------------------
# Extra-tool pass-through — uniform across all families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        Qwen3VLDesktopGroundingPointActionSpace,
        Qwen3VLMobileGroundingPointActionSpace,
        Qwen3_5DesktopGroundingPointActionSpace,
        Qwen3_5MobileGroundingPointActionSpace,
        EvoCUADesktopGroundingPointActionSpace,
        UITarsDesktopGroundingPointActionSpace,
        UITarsMobileGroundingPointActionSpace,
        UITars15V1DesktopGroundingPointActionSpace,
        UITars15V1MobileGroundingPointActionSpace,
        MAIUIGroundingPointActionSpace,
    ],
)
def test_extra_tool_pass_through(cls):
    """Agent emits the env's ``report_infeasible(reason=...)`` extra tool;
    every grounding action_space passes it through verbatim (the env
    consumes it directly for refusal scoring).
    """
    asp = cls()
    extra = {
        "name": "report_infeasible",
        "arguments": {"reason": "no such element"},
    }
    lite = asp.convert_tool_calls_from_agent([extra])
    assert len(lite) == 1
    assert tool_call_name(lite[0]) == "report_infeasible"
    assert tool_call_arguments(lite[0])["reason"] == "no such element"


# ---------------------------------------------------------------------------
# MAI-UI text format — adapter-level parsing of <grounding_think> + <answer>
# ---------------------------------------------------------------------------


class TestMAIUIGroundingTextFormat:
    """MAI-UI grounding parses ``<grounding_think>...</grounding_think>``
    + ``<answer>{"coordinate":[x,y]}</answer>`` text — completely
    different wire format from the navigation ``<thinking>`` /
    ``<tool_call>`` shape.
    """

    def setup_method(self):
        self.adapter = AgentAdapterRegistry.get("mai_ui@desktop@grounding.point")

    def test_parse_full_format(self):
        raw = (
            "<grounding_think>The Settings icon is at the top-right.</grounding_think>\n"
            '<answer>\n{"coordinate": [850, 100]}\n</answer>'
        )
        am = self.adapter.parse_raw_assistant_response(raw)
        lm = self.adapter.convert_message_from_agent(am)
        assert tool_call_name(lm["tool_calls"][0]) == "point"
        assert tool_call_arguments(lm["tool_calls"][0])["coordinate"] == [850, 100]
        # Reasoning surfaces as inline_reasoning (not native <think>)
        reasoning_parts = [p for p in lm["content"] if p["type"] == "inline_reasoning"]
        assert len(reasoning_parts) == 1
        assert "Settings icon" in reasoning_parts[0]["text"]

    def test_parse_no_thinking(self):
        """Empty <grounding_think> still yields a valid coordinate parse."""
        raw = '<grounding_think></grounding_think>\n<answer>\n{"coordinate": [10, 20]}\n</answer>'
        am = self.adapter.parse_raw_assistant_response(raw)
        lm = self.adapter.convert_message_from_agent(am)
        assert tool_call_arguments(lm["tool_calls"][0])["coordinate"] == [10, 20]

    def test_parse_malformed_answer_drops_call(self):
        """Malformed JSON in <answer> drops tool_calls cleanly (no partial state)."""
        raw = "<grounding_think>thinking</grounding_think>\n<answer>\nthis is not json\n</answer>"
        am = self.adapter.parse_raw_assistant_response(raw)
        lm = self.adapter.convert_message_from_agent(am)
        assert "tool_calls" not in lm or not lm.get("tool_calls")

    def test_parse_missing_answer_drops_call(self):
        """Output with reasoning but no <answer> block drops tool_calls."""
        raw = "<grounding_think>I cannot see the element.</grounding_think>"
        am = self.adapter.parse_raw_assistant_response(raw)
        lm = self.adapter.convert_message_from_agent(am)
        assert "tool_calls" not in lm or not lm.get("tool_calls")

    def test_render_round_trip(self):
        """lite point + reasoning → MAI-UI text → lite point round-trips."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "Click Settings."},
            ],
            "tool_calls": [LitePointActionSpace.point(coordinate=[850, 100])],
        }
        agent_msg = self.adapter.convert_message_to_agent(msg)
        text = agent_msg["content"][0]["text"]
        assert "<grounding_think>" in text and "</grounding_think>" in text
        assert "<answer>" in text and "</answer>" in text
        # Re-parse
        am = self.adapter.parse_raw_assistant_response(text)
        lm = self.adapter.convert_message_from_agent(am)
        assert tool_call_arguments(lm["tool_calls"][0])["coordinate"] == [850, 100]


# ---------------------------------------------------------------------------
# Adapter-level cross-cuts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "qwen3_vl@desktop@grounding.point",
        "qwen3_vl@browser@grounding.point",
        "qwen3_vl@mobile@grounding.point",
        "qwen3_5@desktop@grounding.point",
        "qwen3_5@browser@grounding.point",
        "qwen3_5@mobile@grounding.point",
        "evocua@desktop@grounding.point",
        "evocua@browser@grounding.point",
        "ui_tars@desktop@grounding.point",
        "ui_tars@browser@grounding.point",
        "ui_tars@mobile@grounding.point",
        "ui_tars_15_v1@desktop@grounding.point",
        "ui_tars_15_v1@browser@grounding.point",
        "ui_tars_15_v1@mobile@grounding.point",
        "mai_ui@desktop@grounding.point",
        "mai_ui@browser@grounding.point",
        "mai_ui@mobile@grounding.point",
    ],
)
def test_grounding_adapter_uses_full_history(key):
    """Every grounding adapter uses single-turn full-history protocol —
    grounding is single-step click prediction with no multi-turn rollout
    semantics.
    """
    adapter = AgentAdapterRegistry.get(key)
    assert type(adapter.protocol).__name__ == "FullHistoryProtocol", (
        f"{key} protocol = {type(adapter.protocol).__name__}, "
        "expected FullHistoryProtocol (grounding is single-step)."
    )


@pytest.mark.parametrize(
    "key",
    [
        "qwen3_vl@desktop@grounding.point",
        "qwen3_5@desktop@grounding.point",
        "evocua@desktop@grounding.point",
    ],
)
def test_qwen_family_grounding_has_system_prompt(key):
    """Qwen3-VL / Qwen3.5 / EvoCUA grounding adapters ship a non-empty
    system prompt (the trimmed GROUNDING_POINT_SYSTEM_PROMPT) — a
    regression on the previous design where grounding adapters had
    ``system_prompt=None``.
    """
    adapter = AgentAdapterRegistry.get(key)
    assert adapter.system_prompt is not None and adapter.system_prompt.strip()


def test_grounding_point_remains_task_local_schema_free() -> None:
    adapter = agent_adapter_for(
        "qwen3_vl@desktop@grounding.point",
        "desktop",
        task_type="grounding.point",
    )
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [make_tool_call("point", {"coordinate": [1, 2]})],
    }

    rendered = adapter.convert_message_to_agent(message)
    assert rendered["tool_calls"] == [
        {
            "name": "computer_use",
            "arguments": {"action": "left_click", "coordinate": [1, 2]},
        }
    ]


def test_grounding_point_render_does_not_gate_use_action_wrapper() -> None:
    adapter = agent_adapter_for(
        "qwen3_vl@desktop@grounding.point",
        "desktop",
        task_type="grounding.point",
    )
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            )
        ],
    }

    adapter.convert_message_to_agent(message)


_GROUNDING_POINT_WRAPPER_MATRIX = [
    ("qwen3_vl@desktop@grounding.point", "desktop", "computer_use", ["click"]),
    ("qwen3_vl@mobile@grounding.point", "mobile", "mobile_use", ["tap"]),
    ("qwen2_5_vl@desktop@grounding.point", "desktop", "computer_use", ["click"]),
    ("qwen2_5_vl@mobile@grounding.point", "mobile", "mobile_use", ["tap"]),
    ("fara@desktop@grounding.point", "desktop", "computer_use", ["click"]),
]


@pytest.mark.parametrize(
    "adapter_key,platform,wrapper,non_point_actions",
    _GROUNDING_POINT_WRAPPER_MATRIX,
    ids=[row[0] for row in _GROUNDING_POINT_WRAPPER_MATRIX],
)
def test_grounding_point_valid_actions_point_controls_wrapper_schema(
    adapter_key,
    platform,
    wrapper,
    non_point_actions,
) -> None:
    def names(valid_actions):
        return tool_names(
            agent_adapter_for(
                adapter_key,
                platform,
                task_type="grounding.point",
                valid_actions=valid_actions,
            )._assemble_tool_schemas()
        )

    assert wrapper in names(None)
    assert wrapper in names(["point"])
    assert wrapper not in names([])
    assert wrapper not in names(non_point_actions)


@pytest.mark.parametrize(
    "adapter_key,platform,wrapper,non_point_actions",
    _GROUNDING_POINT_WRAPPER_MATRIX,
    ids=[row[0] for row in _GROUNDING_POINT_WRAPPER_MATRIX],
)
def test_grounding_point_valid_actions_empty_keeps_env_extra_tools(
    adapter_key,
    platform,
    wrapper,
    non_point_actions,
) -> None:
    for valid_actions in ([], non_point_actions):
        schemas = agent_adapter_for(
            adapter_key,
            platform,
            task_type="grounding.point",
            valid_actions=valid_actions,
            extra_tool_schemas=[BASH_SCHEMA],
        )._assemble_tool_schemas()
        names = tool_names(schemas)

        assert wrapper not in names
        assert "bash" in names


@pytest.mark.parametrize(
    "space,tool_name,non_point_actions",
    [
        (GPTDesktopGroundingPointActionSpace, "click", ["click"]),
        (ClaudeDesktopGroundingPointActionSpace, "left_click", ["click"]),
    ],
    ids=["gpt_grounding", "claude_grounding"],
)
def test_flat_grounding_point_action_spaces_filter_by_canonical_action(
    space,
    tool_name,
    non_point_actions,
) -> None:
    schemas = space.get_tool_schemas()

    assert tool_name in tool_names(space.filter_tool_schemas_for_valid_actions(schemas, ["point"]))
    assert tool_name not in tool_names(space.filter_tool_schemas_for_valid_actions(schemas, []))
    assert tool_name not in tool_names(
        space.filter_tool_schemas_for_valid_actions(schemas, non_point_actions)
    )
