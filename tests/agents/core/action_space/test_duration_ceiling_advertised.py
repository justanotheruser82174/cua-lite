"""Every action space that offers the model a ``duration`` states its ceiling.

``lite.gym.utils.backend.model_inputs.coerce_model_duration`` hard-rejects an
over-cap ``duration`` and drops the rest of the batch with it. Every family that
lets the model choose that number must say so on the property the model fills:

* the canonical Lite spaces advertise it on the merged batch item's ``duration``;
* Claude and GPT mobile declare ``long_press``/``wait`` as separate tools, so
  each carries its own ``duration`` property;
* the Qwen-derived families wrap all actions in one ``computer_use`` /
  ``mobile_use`` tool and spell the number ``time``, a single property shared by
  ``wait`` (and on mobile ``long_press``). Their converters feed it straight
  into canonical ``duration``, so the same clause belongs there -- and its
  per-action enumeration is what makes one shared property honest.
* Fara wraps its actions the same way and spells the number ``time`` on
  ``computer_use``, serving ``action=wait``. ``fara@browser`` shares the class.

The assertion is IDENTITY against
``lite.core.tools.action_space.duration.duration_description`` -- the single
owner of both the numbers and their wording. A substring check would pass on a
stale hand-typed copy, which is the drift this file exists to stop.

``qwen3_5`` is covered through its parent: it inherits the Qwen3-VL tool
declarations verbatim, and the case below asserts the rendered schema of the
subclass rather than the parent so an override would be caught.

Desktop Claude/GPT/Gemini and Gemini mobile are deliberately absent: they emit
``[]`` from ``get_tool_schemas()`` because those families drive the provider's
own opaque native computer tool, whose description is not ours to write. The Qwen
grounding-point spaces are absent too: their trimmed schemas declare only
``action``/``coordinate``, so there is no model-supplied duration at all.

Two more groups are absent by decision, on the rule that this file pins what the
MODEL reads -- not every ``get_tool_schemas()`` output that happens to hold a
seconds-shaped property:

* ``ui_tars`` / ``ui_tars_15_v1`` desktop+browser declare ``wait`` with NO
  arguments and convert it at a fixed ``duration=5``. The model supplies no
  number, so there is no ceiling it can trip and none to state. Pinned by the
  second test below.
* ``mai_ui@mobile``, ``step_gui@mobile`` and the ``ui_tars*`` mobile spaces DO
  declare a model-supplied seconds property, but their adapters never render
  ``get_tool_schemas()``: the model-visible action space is trained prose owned
  by ``<family>/adapter.py`` -- MAI-UI's ``## Action Space`` JSON block,
  STEP-GUI's Chinese ``# Action Space`` rows, UI-TARS's
  ``long_press(start_box=..., time='')`` line. The render goldens under
  ``tests/agents/core/adapter/_render_goldens`` show no ``<tools>`` section for any
  of them. Putting the clause on those schemas would advertise it to nobody,
  and reaching the model would mean rewriting SFT prose. STEP-GUI additionally
  spells the number ``value``, shared with TYPE/AWAKE/INFO/ABORT, so duration
  wording would be wrong there even if it were rendered.

Run:
    uv run pytest \
        tests/agents/core/action_space/test_duration_ceiling_advertised.py -q
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import (
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
)
from lite.agents.models.claude.action_space import (
    ClaudeDesktopActionSpace,
    ClaudeMobileActionSpace,
)
from lite.agents.models.evocua.action_space import EvoCUADesktopActionSpace
from lite.agents.models.fara.action_space import FaraDesktopActionSpace
from lite.agents.models.gemini.action_space import (
    GeminiDesktopActionSpace,
    GeminiMobileActionSpace,
)
from lite.agents.models.gpt.action_space import (
    GPTDesktopActionSpace,
    GPTMobileActionSpace,
)
from lite.agents.models.qwen2_5_vl.action_space import (
    Qwen2_5VLDesktopActionSpace,
    Qwen2_5VLMobileActionSpace,
)
from lite.agents.models.qwen3_5.action_space import (
    Qwen3_5DesktopActionSpace,
    Qwen3_5MobileActionSpace,
)
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLMobileActionSpace,
)
from lite.agents.models.ui_tars.action_space import UITarsDesktopActionSpace
from lite.agents.models.ui_tars_15_v1.action_space import UITars15V1DesktopActionSpace
from lite.core.tools.action_space.duration import duration_description
from lite.core.tools.schemas import tool_schema_parameters

_MOBILE_TIME = "The seconds to wait. Required only by `action=long_press` and `action=wait`."


def _duration_description(action_space_cls, tool_name: str, property_name: str) -> str:
    """The description the model actually receives for ``property_name``.

    The canonical action-batch tools nest every child's properties under one
    merged ``actions.items``; the per-action provider tools and the Qwen wrapper
    tools declare theirs flat. Read whichever shape the tool has.
    """
    schema = action_space_cls().get_tool_schema(tool_name)
    assert schema is not None, f"{action_space_cls.__name__} has no tool {tool_name!r}"

    properties = tool_schema_parameters(schema)["properties"]
    if "actions" in properties:
        properties = properties["actions"]["items"]["properties"]
    return properties[property_name]["description"]


@pytest.mark.parametrize(
    ("action_space_cls", "tool_name", "property_name", "prefix"),
    [
        (LiteDesktopActionSpace, "computer", "duration", "Duration in seconds."),
        (
            LiteMobileActionSpace,
            "mobile",
            "duration",
            "Duration of the press in seconds.",
        ),
        (
            ClaudeMobileActionSpace,
            "long_press",
            "duration",
            "Press duration in seconds (default 1.0).",
        ),
        (ClaudeMobileActionSpace, "wait", "duration", "Time in seconds to wait."),
        (
            GPTMobileActionSpace,
            "long_press",
            "duration",
            "Hold duration in seconds (default 1.0).",
        ),
        (GPTMobileActionSpace, "wait", "duration", "Time in seconds to wait."),
        (
            Qwen2_5VLDesktopActionSpace,
            "computer_use",
            "time",
            "The seconds to wait. Required only by `action=wait`.",
        ),
        (Qwen2_5VLMobileActionSpace, "mobile_use", "time", _MOBILE_TIME),
        (
            Qwen3VLDesktopActionSpace,
            "computer_use",
            "time",
            "The seconds to wait.",
        ),
        (Qwen3VLMobileActionSpace, "mobile_use", "time", _MOBILE_TIME),
        (
            Qwen3_5DesktopActionSpace,
            "computer_use",
            "time",
            "The seconds to wait.",
        ),
        (Qwen3_5MobileActionSpace, "mobile_use", "time", _MOBILE_TIME),
        (
            EvoCUADesktopActionSpace,
            "computer_use",
            "time",
            "The seconds to wait.",
        ),
        (
            FaraDesktopActionSpace,
            "computer_use",
            "time",
            "The seconds to wait. Required only by `action=wait`.",
        ),
    ],
)
def test_duration_property_advertises_the_owner_rendered_ceiling(
    action_space_cls,
    tool_name,
    property_name,
    prefix,
):
    assert _duration_description(
        action_space_cls, tool_name, property_name
    ) == duration_description(prefix)


@pytest.mark.parametrize(
    "action_space_cls",
    [
        ClaudeDesktopActionSpace,
        GeminiDesktopActionSpace,
        GeminiMobileActionSpace,
        GPTDesktopActionSpace,
    ],
)
def test_opaque_native_families_advertise_no_schema_of_ours(action_space_cls):
    """Scope guard for the parametrization above.

    These spaces render nothing, so there is no description of ours to carry the
    clause. If either ever starts emitting schemas, it joins the table above
    rather than silently shipping an uncapped ``duration``.
    """
    assert action_space_cls().get_tool_schemas() == []


@pytest.mark.parametrize(
    "action_space_cls",
    [
        UITarsDesktopActionSpace,
        UITars15V1DesktopActionSpace,
    ],
)
def test_ui_tars_desktop_wait_takes_no_model_supplied_duration(action_space_cls):
    """Scope guard for the other reason a family is absent above.

    These two (and their ``@browser`` registrations, which share the class) fix
    ``wait`` at five seconds in the converter and declare it with no parameters,
    so the model cannot supply a duration and cannot trip a cap. If ``wait``
    ever grows an argument, it joins the table above rather than silently
    shipping an uncapped duration.
    """
    schema = action_space_cls().get_tool_schema("wait")
    assert tool_schema_parameters(schema)["properties"] == {}
