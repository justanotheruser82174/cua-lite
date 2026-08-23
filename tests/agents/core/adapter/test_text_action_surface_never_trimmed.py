"""A TEXT action surface is never trimmed — not by ``extra_tools``, not by ``valid_actions``.

The rule is about the SURFACE, not the family. A **schema** surface is composed
per request, so narrowing it by the active extra tools and by ``valid_actions``
is what it is for. A **text** surface was part of the model's SFT distribution:
it reaches the model as prompt bytes, so deleting a row moves the model off the
distribution it was fine-tuned on. Reachability is the env's answer, not the
prompt's — a call to an inactive tool comes back as model-visible feedback keyed
to its call id, and the model self-corrects.

Measured on UI-TARS: advertising only the reachable finish form removed 16
inactive-tool rejections but collapsed finish attempts from 19 to 2, left 5 of 8
trajectories burning ``max_steps``, and moved the cell score 0.25 -> 0.0.

Each surface below is listed with WHERE its text reaches the model, because a
method name is not a reliable classifier: ``mai_ui`` defines
``_build_tools_section`` yet never passes ``tools=`` to the chat template, so its
``{"action": ...}`` rows ARE the whole tool surface. A family that pairs SFT prose
with a genuinely dynamic ``<tools>`` schema block does not belong here.

Run:
    uv run pytest tests/agents/core/adapter/test_text_action_surface_never_trimmed.py -q
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
from agents._support.valid_actions_gating import (
    OPEN_APP_SCHEMA,
    RESPONSE_SCHEMA,
    TERMINATE_SCHEMA,
    agent_adapter_for,
    rendered_system_text_for,
)
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter.base import AgentAdapterRegistry, BaseAgentAdapter
from lite.agents.models.mai_ui.adapter import _MAI_MOBILE_SYS_PROMPT_BASE
from lite.agents.models.step_gui.adapter import STEP_GUI_MOBILE_SYS_PROMPT
from lite.agents.models.ui_tars.action_space import (
    GROUNDING_DESKTOP_ACTION_SPACE,
    GROUNDING_MOBILE_ACTION_SPACE,
)
from lite.agents.models.ui_tars.adapter import (
    UITARS_CALL_USR_ACTION_SPACE,
    UITARS_MOBILE_ACTION_SPACE,
)
from lite.agents.models.ui_tars_15_v1.action_space import (
    GROUNDING_DESKTOP_ACTION_SPACE as V1_GROUNDING_DESKTOP_ACTION_SPACE,
)
from lite.agents.models.ui_tars_15_v1.action_space import (
    GROUNDING_MOBILE_ACTION_SPACE as V1_GROUNDING_MOBILE_ACTION_SPACE,
)
from lite.agents.models.ui_tars_15_v1.adapter import (
    UITARS15_V1_MOBILE_ACTION_SPACE,
    UITARS_NORMAL_ACTION_SPACE,
)
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.core.tools.extra_tools import LiteFinishToolSet, make_open_app_tool
from lite.core.tools.schemas import tool_schema_name

register_all()


@dataclasses.dataclass(frozen=True)
class TextSurface:
    """One adapter whose action surface reaches the model as prompt bytes.

    ``sft_block`` is the exact text that must survive both gates. ``reaches_model``
    names the substitution site, which is the classification evidence.
    """

    adapter_key: str
    sft_block: str
    reaches_model: str
    #: True only where the prompt carries a genuinely per-request section
    #: alongside the trained block.
    prompt_varies_with_extras: bool = False


TEXT_SURFACES: tuple[TextSurface, ...] = (
    TextSurface(
        r"ui_tars@(desktop|browser)@use",
        UITARS_CALL_USR_ACTION_SPACE,
        "substituted into USE_USER_PROMPT's {action_space} slot by render_step",
    ),
    TextSurface(
        "ui_tars@mobile@use",
        UITARS_MOBILE_ACTION_SPACE,
        "substituted into MOBILE_USE_USER_PROMPT's {action_space} slot",
    ),
    TextSurface(
        r"ui_tars@(desktop|browser)@grounding\.point",
        GROUNDING_DESKTOP_ACTION_SPACE,
        "substituted into GROUNDING_USER_PROMPT's {action_space} slot",
    ),
    TextSurface(
        "ui_tars@mobile@grounding.point",
        GROUNDING_MOBILE_ACTION_SPACE,
        "substituted into GROUNDING_USER_PROMPT's {action_space} slot",
    ),
    TextSurface(
        r"ui_tars_15_v1@(desktop|browser)@use",
        UITARS_NORMAL_ACTION_SPACE,
        "substituted into USE_USER_PROMPT's {action_space} slot",
    ),
    TextSurface(
        "ui_tars_15_v1@mobile@use",
        UITARS15_V1_MOBILE_ACTION_SPACE,
        "substituted into MOBILE_USE_USER_PROMPT's {action_space} slot",
    ),
    TextSurface(
        r"ui_tars_15_v1@(desktop|browser)@grounding\.point",
        V1_GROUNDING_DESKTOP_ACTION_SPACE,
        "substituted into GROUNDING_USER_PROMPT's {action_space} slot",
    ),
    TextSurface(
        "ui_tars_15_v1@mobile@grounding.point",
        V1_GROUNDING_MOBILE_ACTION_SPACE,
        "substituted into GROUNDING_USER_PROMPT's {action_space} slot",
    ),
    TextSurface(
        "step_gui@mobile@use",
        STEP_GUI_MOBILE_SYS_PROMPT,
        "IS the system message; the model gets no <tools> section at all",
    ),
    TextSurface(
        "mai_ui@mobile@use",
        _MAI_MOBILE_SYS_PROMPT_BASE,
        "the ## Action Space rows of the system message; tools= is never passed "
        "to the chat template",
        # The `Available Apps` pair is env DATA gated on `if apps:`, not on tool
        # activity, so the whole prompt legitimately differs with the open_app enum.
        prompt_varies_with_extras=True,
    ),
)

# Enumeration canary: a shrunken table would pass vacuously.
assert len(TEXT_SURFACES) >= 10, f"text-surface table shrank to {len(TEXT_SURFACES)}"

#: Stamped onto the sample as ``extra_tool_schemas``; every combination is rendered.
EXTRA_TOOL_SCHEMAS = (
    LiteFinishToolSet.get_tool_schema("terminate"),
    LiteFinishToolSet.get_tool_schema("response"),
    make_open_app_tool(["Settings"]),
)

#: ``None`` = no GUI gate; ``[]`` = the strictest gate an env can set; the rest are
#: real per-platform GUI narrowings plus the ``["point"]`` the grounding envs ship.
DESKTOP_VALID_ACTIONS = (None, [], ["click"], ["click", "type"], ["point"])
MOBILE_VALID_ACTIONS = (None, [], ["tap"], ["tap", "type"], ["point"])


def _platform(adapter_key: str) -> str:
    return "mobile" if "@mobile" in adapter_key else "desktop"


def _task_type(adapter_key: str) -> str:
    return adapter_key.rsplit("@", 1)[-1].replace("\\", "")


def _valid_actions_variants(adapter_key: str) -> tuple:
    return MOBILE_VALID_ACTIONS if _platform(adapter_key) == "mobile" else DESKTOP_VALID_ACTIONS


def _sample(adapter_key: str) -> LiteSample:
    platform = _platform(adapter_key)
    task_type = _task_type(adapter_key)
    if task_type == "grounding.point":
        tool_call = make_tool_call("point", {"coordinate": [100, 200]}, call_id="call_0000")
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
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, task_type)),
        images=[Image.new("RGB", (32, 32), color=(30, 0, 0))],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open Settings and report the result."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click the target."}],
                "tool_calls": [tool_call],
            },
        ],
    )


def _rendered_prompt(adapter_key: str, extra_tool_schemas, valid_actions) -> str:
    """Every model-visible text part of the first rendered step, minus the target.

    ``unroll`` appends the target assistant message on an SFT step; that is the
    label, not the prompt, so it is dropped here.
    """
    adapter: BaseAgentAdapter = AgentAdapterRegistry.get_class(adapter_key)(
        metadata=LiteCUAMetadata(
            dims=(_platform(adapter_key), _task_type(adapter_key)),
            extra_tool_schemas=list(extra_tool_schemas),
            valid_actions=valid_actions,
        ),
    )
    step = adapter.unroll(_sample(adapter_key)).steps[0]
    inputs = step[:-1] if step and step[-1].get("role") == "assistant" else step
    return "\n".join(
        part.get("text") or ""
        for message in inputs
        for part in message.get("content") or []
        if part.get("type") == "text"
    )


def _extras_combinations():
    for size in range(len(EXTRA_TOOL_SCHEMAS) + 1):
        yield from itertools.combinations(EXTRA_TOOL_SCHEMAS, size)


@pytest.mark.parametrize("surface", TEXT_SURFACES, ids=lambda s: s.adapter_key)
def test_text_surface_is_byte_identical_across_extras_and_valid_actions(
    surface: TextSurface,
) -> None:
    """IDENTITY, not existence, over the full ``extras x valid_actions`` product.

    The test this replaces asserted ``"finished(" in text``: a containment check
    passes on the full block AND on a block with every other row deleted, so it
    could not see the defect at all.
    """
    renders: dict[tuple, str] = {}
    for combination in _extras_combinations():
        names = tuple(sorted(schema["function"]["name"] for schema in combination))
        for valid_actions in _valid_actions_variants(surface.adapter_key):
            renders[(names, repr(valid_actions))] = _rendered_prompt(
                surface.adapter_key,
                combination,
                valid_actions,
            )

    # The trained bytes survive every combination.
    off_distribution = {state for state, text in renders.items() if surface.sft_block not in text}
    assert not off_distribution, (
        f"{surface.adapter_key}: the SFT block ({surface.reaches_model}) was "
        f"edited in {len(off_distribution)} states, e.g. {sorted(off_distribution)[0]}"
    )

    # ``valid_actions`` never changes a byte, whatever the extras are.
    for names in {state[0] for state in renders}:
        distinct = {text for state, text in renders.items() if state[0] == names}
        assert len(distinct) == 1, (
            f"{surface.adapter_key}: extras={names} renders {len(distinct)} "
            "different prompts across valid_actions variants"
        )

    # The extras gate never changes a byte either, except where per-request DATA
    # is part of the prompt by design.
    if not surface.prompt_varies_with_extras:
        assert len(set(renders.values())) == 1, (
            f"{surface.adapter_key}: renders "
            f"{len(set(renders.values()))} different prompts across extras"
        )


#: The negative half: rows that DID disappear before this rule was applied.
#: Each is checked in the strictest state an env can produce — no extra tool
#: advertised and ``valid_actions=[]`` — which is where both gates bit hardest.
ROWS_THE_GATES_USED_TO_DELETE: dict[str, tuple[str, ...]] = {
    r"ui_tars@(desktop|browser)@use": ("finished()", "call_user()", "wait()", "hotkey(key='')"),
    "ui_tars@mobile@use": ("finished(content='')", "press_home()", "press_back()"),
    r"ui_tars@(desktop|browser)@grounding\.point": (
        "click(start_box='<|box_start|>(x1,y1)<|box_end|>')",
    ),
    "ui_tars@mobile@grounding.point": ("click(start_box='<|box_start|>(x1,y1)<|box_end|>')",),
    r"ui_tars_15_v1@(desktop|browser)@use": (
        "finished(content='xxx')",
        "call_user()",
        "wait()",
    ),
    "ui_tars_15_v1@mobile@use": ("finished(content='')", "press_home()", "press_back()"),
    r"ui_tars_15_v1@(desktop|browser)@grounding\.point": (
        "click(start_box='<|box_start|>(x1,y1)<|box_end|>')",
    ),
    "ui_tars_15_v1@mobile@grounding.point": ("click(start_box='<|box_start|>(x1,y1)<|box_end|>')",),
    "step_gui@mobile@use": (
        "以下9类操作",
        "action:CLICK",
        "action:TYPE",
        "action:WAIT",
        "action:SLIDE",
        "action:LONGPRESS",
        "action:COMPLETE",
        "action:AWAKE",
        "action:INFO",
        "action:ABORT",
    ),
    "mai_ui@mobile@use": (
        '{"action": "click", "coordinate": [x, y]}',
        '{"action": "long_press", "coordinate": [x, y]}',
        '{"action": "type", "text": ""}',
        '{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}',
        '{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter',
        '{"action": "wait"}',
        '{"action": "terminate", "status": "success or fail"}',
    ),
}

assert set(ROWS_THE_GATES_USED_TO_DELETE) == {surface.adapter_key for surface in TEXT_SURFACES}, (
    "every text surface needs its deleted-rows record"
)


@pytest.mark.parametrize("surface", TEXT_SURFACES, ids=lambda s: s.adapter_key)
def test_rows_the_gates_used_to_delete_are_advertised_with_no_surface_active(
    surface: TextSurface,
) -> None:
    """No extras, ``valid_actions=[]`` — and every named row is still on the wire.

    A finish row is the load-bearing case: withhold it and the model stops
    declaring completion, so the episode runs to ``max_steps`` instead.
    """
    prompt = _rendered_prompt(surface.adapter_key, (), [])
    missing = [
        row for row in ROWS_THE_GATES_USED_TO_DELETE[surface.adapter_key] if row not in prompt
    ]
    assert not missing, f"{surface.adapter_key}: rows dropped from the prompt: {missing}"


# =============================================================================
# Legacy valid-actions gating rows split from test_valid_actions_gating.py
# =============================================================================

_TEXT_SURFACE_FAMILY_KEYS = [
    ("mai_ui@mobile@use", "mobile"),
    ("step_gui@mobile@use", "mobile"),
]

_NATIVE_SPELLED_EXTRA_SCHEMAS = (RESPONSE_SCHEMA, TERMINATE_SCHEMA, OPEN_APP_SCHEMA)


@pytest.mark.parametrize(
    "adapter_key,platform",
    _TEXT_SURFACE_FAMILY_KEYS,
    ids=[key for key, _ in _TEXT_SURFACE_FAMILY_KEYS],
)
def test_text_surface_prompt_is_byte_identical_across_active_extras(
    adapter_key,
    platform,
) -> None:
    """Every ``extra_tool_schemas`` combination renders the same prompt bytes."""
    baseline = rendered_system_text_for(adapter_key, platform)
    for size in range(1, len(_NATIVE_SPELLED_EXTRA_SCHEMAS) + 1):
        for extras in itertools.combinations(_NATIVE_SPELLED_EXTRA_SCHEMAS, size):
            rendered = rendered_system_text_for(
                adapter_key,
                platform,
                extra_tool_schemas=list(extras),
            )
            assert rendered == baseline, (
                f"{adapter_key}: extras={[tool_schema_name(s) for s in extras]} "
                "changed the rendered prompt bytes"
            )


@pytest.mark.parametrize(
    "adapter_key,platform",
    _TEXT_SURFACE_FAMILY_KEYS,
    ids=[key for key, _ in _TEXT_SURFACE_FAMILY_KEYS],
)
def test_text_surface_families_declare_no_active_extra_gate(adapter_key, platform) -> None:
    """Text surfaces have no caller for the retired ``..._for_active_extra_tools`` gate."""
    action_space_cls = type(agent_adapter_for(adapter_key, platform).action_space)
    assert [
        name
        for name in dir(action_space_cls)
        if name.endswith("_action_values_for_active_extra_tools")
    ] == []
