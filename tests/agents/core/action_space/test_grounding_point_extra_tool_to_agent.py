"""A ``@point`` space narrows the action layer, never the extra-tool channel.

Every ``*@point`` key trims the model-facing surface to ONE verb. That is an
action-layer decision. Until this test existed the same code also consumed
every OTHER canonical call on ``convert_tool_calls_to_agent`` — including
env-supplied standalone tools the family owns no spelling for, such as
``osworld_g``'s ``report_infeasible``. The observable damage was an EMPTY
assistant turn: ``adapter.unroll`` on a refusal trajectory rendered
``content=[]``, an SFT target indistinguishable from "nothing to do".

The rule pinned here, uniform over every registered ``@point`` key:

* an action-layer name (GUI action or action-batch tool) is dropped — the family
  owns the name and a one-verb surface genuinely cannot spell it;
* anything else passes through untouched — it belongs to the gym-shared /
  single-env / standalone extra-tool tiers, which are admitted from
  ``metadata.extra_tool_schemas`` outside the grounding conversion helper.

The ``from_agent`` direction already answered this way
(``tests/agents/core/action_space/test_grounding_point.py::test_extra_tool_pass_through``);
this pins the other direction to the same answer.

Run:
    uv run pytest tests/agents/core/action_space/test_grounding_point_extra_tool_to_agent.py -v
"""

from __future__ import annotations

import copy

import pytest
from PIL import Image

from lite.agents.core.action_space.base import ActionSpaceRegistry
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.agents.core.adapter.base import AgentAdapterRegistry
from lite.core import (
    LiteSample,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_name

RESOLUTION = (1920, 1080)

#: An env-supplied standalone tool. Its Lite envelope is canonical, but its
#: name is not in the built-in action vocabulary, so no family dialect table may
#: consume it.
ENV_EXTRA_CALL = make_tool_call(
    "report_infeasible",
    {"reason": "no such element"},
)
ENV_EXTRA_AGENT_CALL = {"name": "report_infeasible", "arguments": {"reason": "no such element"}}

#: Action-layer names the one-verb grounding surface owns and cannot render.
ACTION_LAYER_CALLS = (
    make_tool_call("click", {"coordinate": [10, 20]}),
    make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": [10, 20]}]},
    ),
)

#: Standalone extra-tool calls are not action-layer names here.
STANDALONE_EXTRA_CALLS = (
    ENV_EXTRA_CALL,
    make_tool_call("terminate", {"status": "success"}),
    make_tool_call("goto", {"url": "http://example.com"}),
    make_tool_call("open_app", {"app_name": "Chrome"}),
)

#: Grounding adapters that feed the action space directly — they own no
#: extra-tool router, so the action space is the ONLY thing standing between
#: the env's refusal tool and an empty SFT target.
DIRECT_FEED_GROUNDING_ADAPTERS = (
    "ui_tars@desktop@grounding.point",
    "ui_tars@mobile@grounding.point",
    "ui_tars_15_v1@desktop@grounding.point",
    "ui_tars_15_v1@mobile@grounding.point",
)


@pytest.fixture(autouse=True, scope="module")
def _registered():
    from lite.agents.bootstrap import register_all

    register_all()


def _family_point_keys() -> list[str]:
    """Every registered ``<family>@<platform>@point`` key.

    ``lite@point`` is excluded: it is the CORE point set, not a family dialect
    over one, so it has no grounding narrowing to get wrong.
    """
    return sorted(
        key
        for key in ActionSpaceRegistry.list_expanded()
        if key.endswith("@point") and key.count("@") > 1
    )


class TestPointSpacesDoNotConsumeEnvExtras:
    def test_every_point_key_passes_an_env_extra_through(self):
        offenders = []
        for key in _family_point_keys():
            space = ActionSpaceRegistry.get(key)
            emitted = space.convert_tool_calls_to_agent(
                [copy.deepcopy(ENV_EXTRA_CALL)], resolution=RESOLUTION
            )
            if emitted != [ENV_EXTRA_AGENT_CALL]:
                offenders.append(f"{key}: {emitted}")
        assert offenders == []

    def test_the_universe_is_not_empty(self):
        """Guards the filter above from silently selecting nothing."""
        keys = _family_point_keys()
        assert len(keys) >= 25, keys

    @pytest.mark.parametrize("call", ACTION_LAYER_CALLS, ids=tool_call_name)
    def test_every_point_key_still_drops_action_layer_names(self, call):
        """The one-verb narrowing itself is unchanged — no new leakage."""
        offenders = []
        for key in _family_point_keys():
            space = ActionSpaceRegistry.get(key)
            emitted = space.convert_tool_calls_to_agent(
                [copy.deepcopy(call)], resolution=RESOLUTION
            )
            if emitted != []:
                offenders.append(f"{key}: {emitted}")
        assert offenders == []

    @pytest.mark.parametrize("call", STANDALONE_EXTRA_CALLS, ids=tool_call_name)
    def test_grounding_helper_preserves_standalone_extras(self, call):
        emitted = convert_non_point_call_for_grounding_space(
            call,
            surface="test grounding action_space",
        )

        assert emitted == [
            {"name": tool_call_name(call), "arguments": copy.deepcopy(tool_call_arguments(call))}
        ]

    def test_both_directions_now_preserve_the_env_extra(self):
        """``from_agent`` and ``to_agent`` both preserve env extras.

        claude and gpt are excluded on the ``from_agent`` side only: their
        input on that direction is the PROVIDER's own action dict, not a
        ``{name, arguments}`` call, and their agents dispatch env extras by
        name BEFORE the action space is consulted. Their parser is a closed
        set by design — that is the direction asymmetry pinned in
        ``test_unknown_action_direction_policy.py``, not a gap.
        """
        for key in _family_point_keys():
            if key.startswith(("claude@", "gpt@")):
                space = ActionSpaceRegistry.get(key)
                assert space.convert_tool_calls_to_agent(
                    [copy.deepcopy(ENV_EXTRA_CALL)], resolution=RESOLUTION
                ) == [ENV_EXTRA_AGENT_CALL], key
                continue
            space = ActionSpaceRegistry.get(key)
            emitted = space.convert_tool_calls_to_agent(
                [copy.deepcopy(ENV_EXTRA_CALL)], resolution=RESOLUTION
            )
            parsed = space.convert_tool_calls_from_agent(
                [copy.deepcopy(ENV_EXTRA_AGENT_CALL)], resolution=RESOLUTION
            )
            assert parsed == [ENV_EXTRA_CALL], key
            assert emitted == [ENV_EXTRA_AGENT_CALL], key


# TestCanonicalVocabulary held one white-box test asserting the membership of
# ``canonical_lite_tool_names()``, which N6.2 deleted as a compound set with no production
# caller. Its observable content -- report_infeasible / bash pass through, canonical names
# drop -- is pinned behaviourally by the rest of this file, which is why the class went with
# it rather than being kept alive around a `pass`.


class TestRefusalTrajectoryKeepsItsTarget:
    """End-to-end: the SFT render of an ``osworld_g``-style refusal turn."""

    @pytest.mark.parametrize("adapter_key", DIRECT_FEED_GROUNDING_ADAPTERS)
    def test_unroll_does_not_emit_an_empty_assistant_target(self, adapter_key):
        import dataclasses

        platform = (
            LiteCUAMetadata.Platform.MOBILE
            if "@mobile@" in adapter_key
            else LiteCUAMetadata.Platform.DESKTOP
        )
        metadata = LiteCUAMetadata(
            dims=(platform, LiteCUAMetadata.TaskType.GROUNDING_POINT),
            extra_tool_schemas=[
                make_tool_schema(
                    "report_infeasible",
                    description="Report the instruction as infeasible.",
                    parameters={
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "required": ["reason"],
                    },
                )
            ],
        )
        adapter = dataclasses.replace(
            AgentAdapterRegistry.get(adapter_key), metadata=copy.deepcopy(metadata)
        )
        sample = LiteSample(
            images=[Image.new("RGB", RESOLUTION, "white")],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": "Click the nonexistent Foo button"},
                    ],
                },
                {"role": "assistant", "content": [], "tool_calls": [copy.deepcopy(ENV_EXTRA_CALL)]},
            ],
            metadata=copy.deepcopy(metadata),
        )
        rendered = adapter.unroll(sample).steps[-1][-1]
        assert rendered["role"] == "assistant"
        blob = str(rendered.get("content")) + str(rendered.get("tool_calls"))
        assert "report_infeasible" in blob, rendered
