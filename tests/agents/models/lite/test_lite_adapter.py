"""
Tests for lite.agents.models.lite: registry, conversion, unroll, batched,
and adapter construction kwargs.

Samples from lite_samples. Run: uv run pytest tests/agents/models/lite/test_lite_adapter.py -v
"""

from __future__ import annotations

import copy as _copy
import dataclasses

import pytest
from lite_samples import (
    sample_grounding_action_desktop,
    sample_grounding_action_mobile,
    sample_trajectory_two_turns,
    sample_understanding,
)
from PIL import Image

import lite.agents.core.adapter as adapter_pkg
from lite.agents.core.action_space.base import (
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
)
from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    BaseAgentAdapter,
)
from lite.agents.core.adapter import AsIsAdapter as AsIsAdapterCls
from lite.agents.models.lite.adapter import (
    USE_SYSTEM_PROMPT,
    LiteDesktopGroundingActionAdapter,
    LiteDesktopUseAdapter,
    LiteMobileGroundingActionAdapter,
    LiteMobileUseAdapter,
)
from lite.agents.models.qwen3_vl import adapter as _qwen3_vl_adapter  # noqa: F401
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments
from lite.core.tools.extra_tools import (
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
    make_open_app_tool,
)
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls

# -----------------------------------------------------------------------------
# (1) Registry: resolve adapter by key (exact and regex)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected_cls",
    [
        ("as_is", AsIsAdapterCls),
        ("lite@desktop@understanding", AsIsAdapterCls),
        ("lite@browser@understanding", AsIsAdapterCls),
        ("lite@desktop@grounding.point", AsIsAdapterCls),
        ("lite@desktop@grounding.bbox", AsIsAdapterCls),
        ("lite@desktop@grounding.action", LiteDesktopGroundingActionAdapter),
        ("lite@desktop@use", LiteDesktopUseAdapter),
        ("lite@browser@grounding.action", LiteDesktopGroundingActionAdapter),
        ("lite@browser@use", LiteDesktopUseAdapter),
        ("lite@mobile@grounding.action", LiteMobileGroundingActionAdapter),
        ("lite@mobile@use", LiteMobileUseAdapter),
    ],
)
def test_registry_get_returns_correct_adapter_class(key, expected_cls):
    """AgentAdapterRegistry.get(key) returns instance of expected adapter class."""
    adapter = AgentAdapterRegistry.get(key)
    assert isinstance(adapter, BaseAgentAdapter)
    assert type(adapter) is expected_cls


# -----------------------------------------------------------------------------
# (2) AsIsAdapter: pass-through
# -----------------------------------------------------------------------------


def test_asis_unroll_deep_copy():
    """AsIsAdapter.unroll yields a single step that's a deep copy of messages."""
    sample = sample_understanding()
    adapter = AgentAdapterRegistry.get("as_is")
    out = adapter.unroll(sample)
    assert len(out.steps) == 1
    step = out.steps[0]
    assert step == sample.messages
    # Mutate text in the user message (content[1] is the text block)
    step[0]["content"][1]["text"] = "mutated"
    assert sample.messages[0]["content"][1]["text"] != "mutated"


# -----------------------------------------------------------------------------
# (3) LiteBaseAdapter (grounding): full history, tools key
# -----------------------------------------------------------------------------


def test_grounding_action_desktop_unroll_preserves_messages():
    """AsIsAdapterCls: full history, messages preserved."""
    sample = sample_grounding_action_desktop()
    adapter = AgentAdapterRegistry.get("lite@desktop@grounding.action")
    step = adapter.unroll(sample).steps[-1]
    assert len(step) == len(sample.messages)
    assert step[0]["role"] == "user" and step[1]["role"] == "assistant"


def test_grounding_action_mobile_unroll():
    """AsIsAdapterCls: full history."""
    sample = sample_grounding_action_mobile()
    adapter = AgentAdapterRegistry.get("lite@mobile@grounding.action")
    step = adapter.unroll(sample).steps[-1]
    assert len(step) == 2


# -----------------------------------------------------------------------------
# (4) LiteDesktopUseAdapter: system prompt + full history
# -----------------------------------------------------------------------------


def test_trajectory_adapter_adds_system_prompt():
    """LiteDesktopUseAdapter always adds USE_SYSTEM_PROMPT."""
    sample = sample_trajectory_two_turns()
    adapter = AgentAdapterRegistry.get("lite@desktop@use")
    step = adapter.unroll(sample).steps[-1]
    assert len(step) > len(sample.messages)
    assert step[0]["role"] == "system"
    assert isinstance(step[0]["content"], list)
    text_parts = [c.get("text", "") for c in step[0]["content"] if c.get("type") == "text"]
    assert any(USE_SYSTEM_PROMPT[:50] in "".join(text_parts) for _ in [None])


# -----------------------------------------------------------------------------
# (5) Per-step views via unroll(...).steps
# -----------------------------------------------------------------------------


def test_unroll_two_turns_yields_two_steps():
    """Trajectory with 2 turns yields 2 training steps (each step includes
    full history of prior turns since Lite uses FullHistoryProtocol)."""
    sample = sample_trajectory_two_turns()
    adapter = AgentAdapterRegistry.get("lite@desktop@use")
    steps = adapter.unroll(sample).steps
    assert len(steps) == 2
    # First step: first turn only (system + user + assistant)
    assert len(steps[0]) <= 4
    # Second step: full history through second turn (>= first step length)
    assert len(steps[1]) >= len(steps[0])


def test_unroll_empty_messages_returns_empty_steps():
    """unroll with empty messages returns no steps."""
    adapter = AgentAdapterRegistry.get("lite@desktop@use")
    out = adapter.unroll(dataclasses.replace(sample_trajectory_two_turns(), messages=[]))
    assert out.steps == []


# -----------------------------------------------------------------------------
# (6) Per-sample conversion: unroll(sample) shape
# -----------------------------------------------------------------------------


def test_asis_unroll_single_step():
    """AsIsAdapter.unroll: one sample in, one step out."""
    adapter = AgentAdapterRegistry.get("as_is")
    sample = sample_understanding()
    out = adapter.unroll(sample)
    assert out.steps and out.steps[0] is not None
    # processed_images defaults to identity for AsIsAdapter (process_image is no-op).
    assert out.processed_images == sample.images


def test_path_image_carriers_pass_through_adapter_image_pipeline():
    """Path strings are an explicit external image-store carrier, not PIL input."""
    adapter = AgentAdapterRegistry.get("as_is", resolution=[4, 3])
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        images=["images/000001.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "inspect"},
                ],
            }
        ],
    )

    out = adapter.unroll(sample)

    assert out.processed_images == ["images/000001.png"]


def test_pil_image_carriers_process_through_adapter_image_pipeline():
    """In-memory rollout PIL carriers use the same explicit adapter boundary."""
    adapter = AgentAdapterRegistry.get("as_is", resolution=[4, 3])
    image = Image.new("RGB", (8, 6), color=(10, 20, 30))
    sample = LiteSample(
        metadata=LiteCUAMetadata(),
        images=[image],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "inspect"},
                ],
            }
        ],
    )

    out = adapter.unroll(sample)

    assert len(out.processed_images) == 1
    processed = out.processed_images[0]
    assert isinstance(processed, Image.Image)
    assert processed.size == (4, 3)
    assert processed.getpixel((0, 0)) == (10, 20, 30)


def test_trajectory_unroll_n_steps():
    """Trajectory adapter: one sample (2 turns) -> 2 unrolled steps."""
    adapter = AgentAdapterRegistry.get("lite@desktop@use")
    sample = sample_trajectory_two_turns()
    out = adapter.unroll(sample)
    assert len(out.steps) == 2


# -----------------------------------------------------------------------------
# (7) Adapter construction kwargs
# -----------------------------------------------------------------------------


def test_protocol_key_and_kwargs_apply_and_are_consumed():
    adapter = AgentAdapterRegistry.get(
        "lite@desktop@use",
        protocol_key="qwen3_vl.history",
        protocol_kwargs={"full_history_size": 1},
    )

    assert isinstance(adapter, LiteDesktopUseAdapter)
    assert adapter.protocol.get_registry_key() == "qwen3_vl.history"
    assert adapter.protocol.full_history_size == 1
    assert adapter.kwargs == {}


def test_documented_constructor_shape_uses_metadata_and_protocol_kwargs():
    schema = make_tool_schema("report_infeasible")
    metadata = LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=[schema],
    )

    adapter = AgentAdapterRegistry.get(
        "lite@desktop@use",
        metadata=metadata,
        protocol_key="qwen3_vl.history",
        protocol_kwargs={"full_history_size": 4},
    )

    assert adapter.metadata is metadata
    assert adapter.metadata.extra_tool_schemas == [schema]
    assert adapter.protocol.full_history_size == 4
    assert adapter.kwargs == {}


def test_base_metadata_default_does_not_infer_task_type_from_registry_key():
    adapter = AgentAdapterRegistry.get("lite@mobile@grounding.action")

    assert adapter.metadata.platform == LiteCUAMetadata.Platform.MOBILE
    assert adapter.metadata.task_type == LiteCUAMetadata.TaskType.USE


def test_subclass_metadata_default_owns_non_use_task_type():
    adapter = AgentAdapterRegistry.get("qwen3_vl@desktop@grounding.point")

    assert adapter.metadata.task_type == LiteCUAMetadata.TaskType.GROUNDING_POINT


def test_metadata_default_rejects_unknown_action_space_platform():
    """An out-of-enum platform fails construction instead of falling back."""
    action_space = LiteDesktopActionSpace()
    action_space.platform = "tablet"

    with pytest.raises(ValueError, match="tablet"):
        LiteDesktopUseAdapter(action_space=action_space)


def test_adapter_field_kwargs_apply_and_are_consumed_before_replace():
    adapter = AgentAdapterRegistry.get("lite@desktop@use", resolution=[1280, 720])

    assert adapter.resolution == (1280, 720)
    assert adapter.kwargs == {}

    replaced = dataclasses.replace(adapter)
    assert replaced.resolution == (1280, 720)
    assert replaced.kwargs == {}


def test_unknown_protocol_kwargs_fail_loudly():
    with pytest.raises(TypeError, match="unknown protocol_kwargs.*typo"):
        AgentAdapterRegistry.get(
            "lite@desktop@use",
            protocol_kwargs={"typo": 1},
        )


def test_malformed_protocol_kwargs_fail_loudly():
    with pytest.raises(TypeError, match="protocol_kwargs must be a dict"):
        AgentAdapterRegistry.get("lite@desktop@use", protocol_kwargs=["bad"])


def test_malformed_packed_kwargs_fail_loudly():
    with pytest.raises(TypeError, match="kwargs must be a dict"):
        LiteDesktopUseAdapter(kwargs=["bad"])


def test_unknown_adapter_kwargs_fail_loudly():
    with pytest.raises(TypeError, match="unknown adapter kwargs.*typo"):
        AgentAdapterRegistry.get("lite@desktop@use", typo=True)


@pytest.fixture
def restored_agent_registry():
    from lite.agents.core.agent import AgentRegistry

    items = dict(AgentRegistry._items)
    patterns = list(AgentRegistry._patterns)
    instances = dict(AgentRegistry._instances)
    yield AgentRegistry
    AgentRegistry._items = items
    AgentRegistry._patterns = patterns
    AgentRegistry._instances = instances


@pytest.mark.parametrize(
    "runtime_kwargs",
    [
        {"model_id": "Qwen/Qwen3-VL-8B-Instruct"},
        {"sampling_kwargs": {"temperature": 0.6}},
    ],
)
def test_auto_adapter_agent_runtime_kwargs_fail_loudly(
    runtime_kwargs,
    restored_agent_registry,
):
    from lite.agents.core.agent import AutoAdapterAgent

    @dataclasses.dataclass
    class _LiteTestAgent(AutoAdapterAgent, key="lite@desktop@use"):
        pass

    with pytest.raises(TypeError, match="unknown adapter kwargs"):
        restored_agent_registry.get(
            "lite@desktop@use",
            generate_fn=lambda **kwargs: {"response": ""},
            **runtime_kwargs,
        )


def test_factory_forwards_model_id_only_for_api_agents(monkeypatch):
    from lite.agents import bootstrap, factory
    from lite.agents.models import AgentRegistry

    metadata = LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
    )
    env = type("Env", (), {"metadata": metadata})()
    calls = []

    def fake_get(cls, agent_key, **kwargs):
        del cls
        calls.append((agent_key, kwargs))
        return object()

    monkeypatch.setattr(bootstrap, "register_all", lambda: None)
    monkeypatch.setattr(AgentRegistry, "get", classmethod(fake_get))

    factory.make(
        "Qwen/Qwen3-VL-2B-Instruct",
        env=env,
        processor=object(),
        generate_fn=lambda **kwargs: {"response": ""},
    )
    local_key, local_kwargs = calls[-1]

    factory.make("gpt-5.5", env=env)
    api_key, api_kwargs = calls[-1]

    assert local_key == "qwen3_vl@desktop@use"
    assert "model_id" not in local_kwargs
    assert local_kwargs["metadata"] is metadata
    assert api_key == "gpt@desktop@use"
    assert api_kwargs["model_id"] == "gpt-5.5"
    assert api_kwargs["metadata"] is metadata


def test_tool_surface_kwargs_fail_loudly_as_adapter_kwargs():
    with pytest.raises(TypeError, match="tool-surface settings.*metadata=LiteCUAMetadata"):
        AgentAdapterRegistry.get("lite@desktop@use", extra_tools=None)


def test_adapter_facade_does_not_expose_config_override_resolver():
    assert not hasattr(adapter_pkg, "get_adapter_with_overrides")


def test_default_trajectory_uses_full_history():
    """Without overrides, lite navigation adapter uses raw FullHistoryProtocol."""
    from lite.agents.core.protocol.base import FullHistoryProtocol

    adapter = AgentAdapterRegistry.get("lite@desktop@use")
    assert isinstance(adapter.protocol, FullHistoryProtocol)


# =============================================================================
# Characterization tests for the `lite` canonical adapter.
#
# Unlike other adapters, `lite` has no "raw" wire format — it IS the canonical
# mid-format. The identity / pass-through contract (only protocol windowing +
# optional system prompt are applied) is captured below as goldens so a planned
# refactor catches behavioral regressions.
#
# All goldens were captured live via a throwaway ``capture_lite.py`` script.
# =============================================================================

# -----------------------------------------------------------------------------
# Action catalogue — all canonical tool_calls emitted by Lite action spaces.
#
# Goldens capture CURRENT output; note the peculiarities flagged below.
# -----------------------------------------------------------------------------


# OBSERVED: default-only fields are dropped by the core action constructors
# (e.g. ``click`` without ``button`` / ``clicks`` → arguments omit them; the
# ``scroll`` action lists arguments in insertion order ``coordinate, direction,
# amount``). ``mouse_down/mouse_up/screenshot/cursor_position`` with no args
# produce ``"arguments": {}`` (empty dict, not missing). ``long_press`` with
# ``duration=None`` also drops ``duration``. ``terminate`` with ``reason=None``
# drops it. Pinch's ``amount`` has default 25, dropped iff explicitly 25.
def _computer_call(action: str, arguments: dict) -> dict:
    return make_tool_call(
        "computer",
        {"actions": [{"action": action, **arguments}]},
    )


def _mobile_call(action: str, arguments: dict) -> dict:
    return make_tool_call(
        "mobile",
        {"actions": [{"action": action, **arguments}]},
    )


_DESKTOP_ACTIONS: dict[str, dict] = {
    "click": _computer_call("click", {"coordinate": [500, 300]}),
    "click_right": _computer_call("click", {"coordinate": [100, 200], "button": "right"}),
    "click_double": _computer_call("click", {"coordinate": [100, 200], "clicks": 2}),
    "type": _computer_call("type", {"text": "hello"}),
    "key": _computer_call("key", {"keys": ["ctrl", "c"]}),
    "key_down": _computer_call("key_down", {"keys": ["shift"]}),
    "key_up": _computer_call("key_up", {"keys": ["shift"]}),
    "hold_key": _computer_call("hold_key", {"keys": ["a"], "duration": 1.0}),
    "mouse_move": _computer_call("mouse_move", {"coordinate": [300, 400]}),
    "drag": _computer_call("drag", {"coordinate": [800, 600], "start_coordinate": [100, 100]}),
    "mouse_down": _computer_call("mouse_down", {}),
    "mouse_up": _computer_call("mouse_up", {}),
    "scroll": _computer_call(
        "scroll", {"coordinate": [500, 500], "direction": "down", "amount": 3}
    ),
    "wait": _computer_call("wait", {"duration": 2.0}),
    "screenshot": _computer_call("screenshot", {}),
    "cursor_position": _computer_call("cursor_position", {}),
    "response": make_tool_call("response", {"text": "42"}),
    "terminate": make_tool_call("terminate", {"status": "success", "reason": "done"}),
}

_MOBILE_ACTIONS: dict[str, dict] = {
    "tap": _mobile_call("tap", {"coordinate": [500, 300], "clicks": 1}),
    "long_press": _mobile_call("long_press", {"coordinate": [100, 200], "duration": 1.5}),
    "type": _mobile_call("type", {"text": "hello"}),
    "swipe": _mobile_call("swipe", {"start_coordinate": [100, 500], "coordinate": [100, 100]}),
    "pinch": _mobile_call("pinch", {"coordinate": [500, 500], "direction": "in", "amount": 25}),
    "open_app": make_tool_call("open_app", {"app_name": "Settings"}),
    "system_button": _mobile_call("system_button", {"button": "Home"}),
    "wait": _mobile_call("wait", {"duration": 2.0}),
    "screenshot": _mobile_call("screenshot", {}),
    "response": make_tool_call("response", {"text": "42"}),
    "terminate": make_tool_call("terminate", {"status": "failure", "reason": "blocked"}),
}


_EXTRA_SCHEMA_BY_ACTION = {
    "open_app": make_open_app_tool(["Settings"]),
    "response": LiteFinishToolSet.get_tool_schema("response"),
    "terminate": LiteFinishToolSet.get_tool_schema("terminate"),
}


def _extra_schemas_for_action(name: str) -> list[dict]:
    schema = _EXTRA_SCHEMA_BY_ACTION.get(name)
    return [schema] if schema is not None else []


def _desktop_adapter_for_action(name: str) -> LiteDesktopUseAdapter:
    return LiteDesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=_extra_schemas_for_action(name),
        )
    )


def _mobile_adapter_for_action(name: str) -> LiteMobileUseAdapter:
    return LiteMobileUseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=_extra_schemas_for_action(name),
        )
    )


def _make_desktop_action(name: str) -> dict:
    """Emit the canonical Lite tool_call for a named desktop action."""
    return _copy.deepcopy(_DESKTOP_ACTIONS[name])


def _make_mobile_action(name: str) -> dict:
    """Emit the canonical Lite tool_call for a named mobile action."""
    return _copy.deepcopy(_MOBILE_ACTIONS[name])


def _build_desktop_traj(n: int, task: str = "Open GIMP.") -> LiteSample:
    """Build an n-turn desktop trajectory (n user+assistant pairs)."""
    msgs: list[dict] = []
    for i in range(n):
        content = [{"type": "image", "index": i}]
        if i == 0:
            content.append({"type": "text", "text": task})
        msgs.append({"role": "user", "content": content})
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"Action step {i}."}],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [100 + i, 100]}]},
                        call_id=f"call_{i:04d}",
                    ),
                ],
            }
        )
    return LiteSample(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
        ),
        messages=msgs,
        images=[f"img{i}.png" for i in range(n)],
    )


def _build_mobile_traj(n: int, task: str = "Find weather.") -> LiteSample:
    msgs: list[dict] = []
    for i in range(n):
        content = [{"type": "image", "index": i}]
        if i == 0:
            content.append({"type": "text", "text": task})
        msgs.append({"role": "user", "content": content})
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"Mobile step {i}."}],
                "tool_calls": [
                    make_tool_call(
                        "mobile",
                        {"actions": [{"action": "tap", "coordinate": [100 + i, 100]}]},
                        call_id=f"call_{i:04d}",
                    ),
                ],
            }
        )
    return LiteSample(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.MOBILE, LiteCUAMetadata.TaskType.USE)
        ),
        messages=msgs,
        images=[f"img{i}.png" for i in range(n)],
    )


# -----------------------------------------------------------------------------
# (A) TestIdentityConversion
#
# For each canonical action, both message-level and tool-call-level
# conversions should be identity (deep copy).
# -----------------------------------------------------------------------------


class TestActionDescriptionFolding:
    """``convert_message_to_agent`` whitelist-picks ``action_description``
    content parts, renders them as ``"Action: <text>"``, and drops every
    other content kind on action turns. ``convert_message_from_agent`` only
    extracts ``action_description`` when the turn also carries ``tool_calls``;
    no-tool-call prose is terminal text."""

    @pytest.mark.parametrize("name", list(_DESKTOP_ACTIONS.keys()))
    def test_desktop_to_agent_renders_action(self, name):
        adapter = _desktop_adapter_for_action(name)
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Perform {name}."}],
            "tool_calls": [_make_desktop_action(name)],
        }
        to_ag = adapter.convert_message_to_agent(msg)
        assert to_ag is not msg  # deep copy
        assert to_ag["content"] == [
            {"type": "text", "text": f"Action: Perform {name}."},
        ]
        assert to_ag["tool_calls"] == [_make_desktop_action(name)]

    @pytest.mark.parametrize("name", list(_DESKTOP_ACTIONS.keys()))
    def test_desktop_from_agent_extracts_action(self, name):
        adapter = _desktop_adapter_for_action(name)
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Perform {name}."}],
            "tool_calls": [_make_desktop_action(name)],
        }
        result = adapter.convert_message_from_agent(adapter.convert_message_to_agent(msg))
        assert result["content"] == [
            {"type": "action_description", "text": f"Perform {name}."},
        ]
        assert result["tool_calls"] == [_make_desktop_action(name)]

    @pytest.mark.parametrize("name", list(_MOBILE_ACTIONS.keys()))
    def test_mobile_to_agent_renders_action(self, name):
        adapter = _mobile_adapter_for_action(name)
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Perform {name}."}],
            "tool_calls": [_make_mobile_action(name)],
        }
        to_ag = adapter.convert_message_to_agent(msg)
        assert to_ag is not msg  # deep copy
        assert to_ag["content"] == [
            {"type": "text", "text": f"Action: Perform {name}."},
        ]
        assert to_ag["tool_calls"] == [_make_mobile_action(name)]

    @pytest.mark.parametrize("name", list(_MOBILE_ACTIONS.keys()))
    def test_mobile_from_agent_extracts_action(self, name):
        adapter = _mobile_adapter_for_action(name)
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Perform {name}."}],
            "tool_calls": [_make_mobile_action(name)],
        }
        result = adapter.convert_message_from_agent(adapter.convert_message_to_agent(msg))
        assert result["content"] == [
            {"type": "action_description", "text": f"Perform {name}."},
        ]
        assert result["tool_calls"] == [_make_mobile_action(name)]

    def test_convert_message_reasoning_content_preserved(self):
        """Top-level reasoning_content survives through to_agent (preserved
        as native channel); action_description is rendered as Action: text."""
        adapter = LiteDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click."}],
            "reasoning_content": "I need to click the button first.",
            "tool_calls": [_make_desktop_action("click")],
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["reasoning_content"] == "I need to click the button first."
        assert out["content"] == [{"type": "text", "text": "Action: click."}]

    def test_to_agent_drops_non_action_description_content(self):
        """Content parts other than action_description are dropped."""
        adapter = LiteDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "content": [
                {"type": "inline_reasoning", "text": "thinking..."},
                {"type": "history_summary", "text": "summary..."},
                {"type": "text", "text": "extra text"},
            ],
            "tool_calls": [_make_desktop_action("click")],
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["content"] == []

    def test_to_agent_tool_calls_only_yields_empty_content(self):
        """When input has no action_description, output content should be []."""
        adapter = LiteDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "tool_calls": [_make_desktop_action("click")],
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["content"] == []


# -----------------------------------------------------------------------------
# (B) TestActionSpaceRoundTrip
#
# For each canonical action, to_agent → from_agent on the action space is
# identity.
# -----------------------------------------------------------------------------


class TestActionSpaceRoundTrip:
    """LiteDesktopActionSpace / LiteMobileActionSpace ``convert_tool_calls_*``
    must be identity round-trips for every canonical action."""

    @pytest.mark.parametrize("name,expected", list(_DESKTOP_ACTIONS.items()))
    def test_desktop_action_matches_golden(self, name, expected):
        """Live-emitted tool_call matches captured golden byte-exact."""
        assert _make_desktop_action(name) == expected

    @pytest.mark.parametrize("name,expected", list(_MOBILE_ACTIONS.items()))
    def test_mobile_action_matches_golden(self, name, expected):
        assert _make_mobile_action(name) == expected

    @pytest.mark.parametrize("name", list(_DESKTOP_ACTIONS.keys()))
    def test_desktop_action_space_round_trip(self, name):
        space = LiteDesktopActionSpace()
        original = _make_desktop_action(name)
        calls = [_copy.deepcopy(original)]
        agent = space.convert_tool_calls_to_agent(calls)
        back = space.convert_tool_calls_from_agent(agent)
        assert back == calls
        # Deep copy: mutation of output must not affect input.
        tool_call_arguments(back[0])["_injected"] = True
        assert "_injected" not in tool_call_arguments(calls[0])

    @pytest.mark.parametrize("name", list(_MOBILE_ACTIONS.keys()))
    def test_mobile_action_space_round_trip(self, name):
        space = LiteMobileActionSpace()
        original = _make_mobile_action(name)
        calls = [_copy.deepcopy(original)]
        agent = space.convert_tool_calls_to_agent(calls)
        back = space.convert_tool_calls_from_agent(agent)
        assert back == calls
        tool_call_arguments(back[0])["_injected"] = True
        assert "_injected" not in tool_call_arguments(calls[0])

    def test_provider_call_id_on_bare_action_call_is_rejected(self):
        adapter = AgentAdapterRegistry.get("lite@browser@use")

        with pytest.raises(ValueError, match="non-bare-call keys \\['call_id'\\]"):
            adapter._route_agent_tool_calls_to_lite(
                [
                    {
                        **{"name": "click", "arguments": {"coordinate": [10, 20]}},
                        "call_id": "provider_gui_call",
                    }
                ]
            )


# -----------------------------------------------------------------------------
# (C) TestSampleConversion
#
# Captures shape (roles / image counts) produced by convert_sample_to_agent
# for both trajectory (summarized) and grounding-action (full-history) adapters.
# -----------------------------------------------------------------------------


class TestSampleConversion:
    """Golden shapes for the final-turn rendered messages on canonical Lite
    trajectories of varying length (predict-time view = ``unroll(...).steps[-1]``)."""

    def test_trajectory_single_turn_shape(self):
        """n=1: system prompt is prepended; no windowing needed."""
        adapter = LiteDesktopUseAdapter()
        agent_sample = adapter.unroll(_build_desktop_traj(1))
        step = agent_sample.steps[-1]
        assert [m["role"] for m in step] == ["system", "user", "assistant"]
        assert len(agent_sample.processed_images) == 1

    def test_trajectory_two_turn_shape(self):
        """n=2 with FullHistoryProtocol: every turn kept verbatim, task text on
        first user, no summary template injected."""
        adapter = LiteDesktopUseAdapter()
        step = adapter.unroll(_build_desktop_traj(2)).steps[-1]
        assert [m["role"] for m in step] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        # Lite uses pass-through FullHistoryProtocol — first user keeps its
        # original instruction text without the Qwen "Please generate ..." wrapper.
        first_user = step[1]
        text = "".join(c.get("text", "") for c in first_user["content"] if c.get("type") == "text")
        assert text == "Open GIMP."

    def test_trajectory_full_history_shape(self):
        """n=5 with FullHistoryProtocol: all 5 turns kept, no windowing,
        no summary template — every turn's user keeps its image."""
        adapter = LiteDesktopUseAdapter()
        agent_sample = adapter.unroll(_build_desktop_traj(5))
        step = agent_sample.steps[-1]
        # 1 system + 5 (user, assistant) pairs
        assert [m["role"] for m in step] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        # Image counts captured golden — every user has its image, no drops.
        img_counts = []
        for m in step:
            c = m.get("content", [])
            n = (
                sum(1 for it in c if isinstance(it, dict) and it.get("type") == "image")
                if isinstance(c, list)
                else 0
            )
            img_counts.append(n)
        assert img_counts == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        # All 5 distinct image indices reachable from user messages.
        image_indices = {
            it["index"]
            for m in step
            if m.get("role") == "user"
            for it in (m.get("content") or [])
            if it.get("type") == "image"
        }
        assert image_indices == {0, 1, 2, 3, 4}
        # First user message keeps its original task text (no Qwen template).
        first_user_text = "".join(c["text"] for c in step[1]["content"] if c.get("type") == "text")
        assert first_user_text == "Open GIMP."


# -----------------------------------------------------------------------------
# (D) TestUnrollStructure
#
# N-turn trajectory → N samples. Role and image-count shape per sample.
# -----------------------------------------------------------------------------


class TestUnrollStructure:
    """Unrolling an N-turn trajectory produces N steps; shape per step is
    captured as golden (system + windowed user/assistant pairs)."""

    def test_desktop_three_turn_unroll_structure(self):
        adapter = LiteDesktopUseAdapter()
        steps = adapter.unroll(_build_desktop_traj(3)).steps
        assert len(steps) == 3
        # FullHistoryProtocol: step[k] keeps all k+1 turns (no windowing).
        expected_roles = [
            ["system", "user", "assistant"],
            ["system", "user", "assistant", "user", "assistant"],
            ["system", "user", "assistant", "user", "assistant", "user", "assistant"],
        ]
        expected_image_count = [1, 2, 3]
        for i, step in enumerate(steps):
            assert [m["role"] for m in step] == expected_roles[i]
            image_indices = {
                it["index"]
                for m in step
                if m.get("role") == "user"
                for it in (m.get("content") or [])
                if it.get("type") == "image"
            }
            assert len(image_indices) == expected_image_count[i]

    def test_mobile_three_turn_unroll_structure(self):
        adapter = LiteMobileUseAdapter()
        steps = adapter.unroll(_build_mobile_traj(3)).steps
        assert len(steps) == 3
        # FullHistoryProtocol: step[k] keeps all k+1 turns (no windowing).
        expected_roles = [
            ["system", "user", "assistant"],
            ["system", "user", "assistant", "user", "assistant"],
            ["system", "user", "assistant", "user", "assistant", "user", "assistant"],
        ]
        for i, step in enumerate(steps):
            assert [m["role"] for m in step] == expected_roles[i]

    def test_unroll_empty_returns_empty_list(self):
        """Empty trajectory → empty step list."""
        empty = LiteSample(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
            ),
            messages=[],
            images=[],
        )
        assert LiteDesktopUseAdapter().unroll(empty).steps == []


# -----------------------------------------------------------------------------
# (E) TestUnrollByteExactTargetPerAction
#
# Since Lite is canonical, the target assistant message must appear verbatim
# in sample i (never re-rendered / coerced).
# -----------------------------------------------------------------------------


class TestUnrollByteExactTargetPerAction:
    """For an N-turn Lite trajectory, the target assistant message in step i
    must be byte-equal to the original ``messages[2i + 1]`` (identity round
    trip end-to-end)."""

    def test_desktop_target_verbatim_per_turn(self):
        """``unroll`` runs ``convert_message_to_agent`` per step, mapping
        ``ActionDescriptionContent`` to ``"Action: <text>"`` and dropping
        non-action_description content. Compare the rendered target against
        what ``convert_message_to_agent`` produces directly on the canonical
        Lite assistant — they must match byte-for-byte."""
        adapter = LiteDesktopUseAdapter()
        traj = _build_desktop_traj(3)
        steps = adapter.unroll(traj).steps
        for i, step in enumerate(steps):
            tgt = [m for m in step if m["role"] == "assistant"][-1]
            expected = adapter.convert_message_to_agent(traj.messages[2 * i + 1])
            assert tgt == expected

    def test_mobile_target_verbatim_per_turn(self):
        adapter = LiteMobileUseAdapter()
        traj = _build_mobile_traj(3)
        steps = adapter.unroll(traj).steps
        for i, step in enumerate(steps):
            tgt = [m for m in step if m["role"] == "assistant"][-1]
            expected = adapter.convert_message_to_agent(traj.messages[2 * i + 1])
            assert tgt == expected

    @pytest.mark.parametrize("name", list(_DESKTOP_ACTIONS.keys()))
    def test_desktop_single_turn_target_byte_exact(self, name):
        """Per action, single-turn unroll renders the assistant target through
        ``convert_message_to_agent`` (action_description → ``"Action: ..."``;
        non-action_description content dropped)."""
        adapter = _desktop_adapter_for_action(name)
        tc = _make_desktop_action(name)
        asst = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Do {name}."}],
            "tool_calls": [tc],
        }
        sample = LiteSample(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE),
                extra_tool_schemas=_extra_schemas_for_action(name),
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": 0},
                        {"type": "text", "text": "task"},
                    ],
                },
                asst,
            ],
            images=["img0.png"],
        )
        steps = adapter.unroll(sample).steps
        assert len(steps) == 1
        got = [m for m in steps[0] if m["role"] == "assistant"][-1]
        expected = adapter.convert_message_to_agent(asst)
        assert got == expected


# -----------------------------------------------------------------------------
# (F) TestUnrollNoLeakage
#
# In a 3-turn trajectory, the user prompt of sample i must not contain
# target_i's distinctive action / step text (windowing + summary should drop
# or precede the target).
# -----------------------------------------------------------------------------


class TestUnrollNoLeakage:
    """Summary / windowing must not leak target_i content into sample i's
    user prompt."""

    def test_desktop_no_future_leakage(self):
        adapter = LiteDesktopUseAdapter()
        traj = _build_desktop_traj(3, task="UniqueDesktopTask42")
        steps = adapter.unroll(traj).steps

        signatures = ["Action step 0.", "Action step 1.", "Action step 2."]

        for i, step in enumerate(steps):
            full_user_text = ""
            for m in step:
                if m["role"] != "user":
                    continue
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        full_user_text += c.get("text", "")
            assert signatures[i] not in full_user_text, (
                f"step {i} leaks own action {signatures[i]!r} into user prompt: {full_user_text!r}"
            )


# -----------------------------------------------------------------------------
# (G) TestProtocolWindowing
#
# Golden boundary captures for Qwen3VLHistoryProtocol — kept here because the
# helper-built desktop trajectories above are the canonical Lite-shape inputs
# the protocol consumes when an SFT/RL config overrides the default
# FullHistoryProtocol with a Qwen-style summarized window.
# -----------------------------------------------------------------------------


class TestProtocolWindowing:
    """``Qwen3VLHistoryProtocol(full_history_size=2)`` output shape at
    boundaries: n < window, n == window, n > window."""

    @pytest.mark.parametrize(
        "n_turns,expected_len,expected_roles",
        [
            (1, 2, ["user", "assistant"]),
            (2, 4, ["user", "assistant", "user", "assistant"]),
            (3, 4, ["user", "assistant", "user", "assistant"]),
            (5, 4, ["user", "assistant", "user", "assistant"]),
            (10, 4, ["user", "assistant", "user", "assistant"]),
        ],
    )
    def test_protocol_shape_at_boundary(self, n_turns, expected_len, expected_roles):
        proto = Qwen3VLHistoryProtocol(full_history_size=2)
        msgs = _build_desktop_traj(n_turns).messages
        out = proto.process_messages(msgs)
        assert len(out) == expected_len
        assert [m["role"] for m in out] == expected_roles

    def test_protocol_summary_template_injected_into_first_user_over_window(self):
        """For n_turns=5 with full_history_size=2: 3 dropped turns are
        summarized into the first kept user message (Step 1..3)."""
        proto = Qwen3VLHistoryProtocol(full_history_size=2)
        msgs = _build_desktop_traj(5).messages
        out = proto.process_messages(msgs)
        first_user = out[0]
        assert first_user["role"] == "user"
        text = "".join(c["text"] for c in first_user["content"] if c.get("type") == "text")
        # Golden template fragments.
        assert "Please generate the next move" in text
        assert "Instruction: Open GIMP." in text
        assert "Step 1: Action step 0." in text
        assert "Step 2: Action step 1." in text
        assert "Step 3: Action step 2." in text
        # Steps inside the window are NOT in the summary
        assert "Step 4:" not in text
        assert "Step 5:" not in text

    def test_protocol_image_counts_preserved_in_window(self):
        """In windowed output, each user message has exactly 1 image; each
        assistant has 0."""
        proto = Qwen3VLHistoryProtocol(full_history_size=2)
        out = proto.process_messages(_build_desktop_traj(5).messages)
        img_counts = []
        for m in out:
            c = m.get("content", [])
            n = (
                sum(1 for it in c if isinstance(it, dict) and it.get("type") == "image")
                if isinstance(c, list)
                else 0
            )
            img_counts.append(n)
        assert img_counts == [1, 0, 1, 0]


# -----------------------------------------------------------------------------
# (H) TestMutationPurity
#
# convert_message_to/from_agent, convert_sample_to_agent, and
# protocol.process_messages must NOT mutate their inputs.
# -----------------------------------------------------------------------------


class TestMutationPurity:
    """Adapter / protocol methods are required to be pure (no in-place
    mutation of caller's sample or messages)."""

    def test_convert_message_to_agent_does_not_mutate(self):
        adapter = LiteDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "click."}],
            "tool_calls": [_make_desktop_action("click")],
        }
        snapshot = _copy.deepcopy(msg)
        out = adapter.convert_message_to_agent(msg)
        assert msg == snapshot
        # Mutating the output does not affect the input
        tool_call_arguments(out["tool_calls"][0])["actions"][0]["coordinate"][0] = 9999
        assert msg == snapshot

    def test_convert_message_from_agent_does_not_mutate(self):
        adapter = LiteMobileUseAdapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "tap."}],
            "tool_calls": [_make_mobile_action("tap")],
        }
        snapshot = _copy.deepcopy(msg)
        _ = adapter.convert_message_from_agent(msg)
        assert msg == snapshot

    def test_unroll_does_not_mutate(self):
        adapter = LiteDesktopUseAdapter()
        sample = _build_desktop_traj(5)
        snapshot = _copy.deepcopy(sample.messages)
        snapshot_images = list(sample.images)
        _ = adapter.unroll(sample)
        assert sample.messages == snapshot
        assert list(sample.images) == snapshot_images

    def test_protocol_process_messages_does_not_mutate(self):
        proto = Qwen3VLHistoryProtocol(full_history_size=2)
        msgs = _build_desktop_traj(5).messages
        snapshot = _copy.deepcopy(msgs)
        _ = proto.process_messages(msgs)
        assert msgs == snapshot


# -----------------------------------------------------------------------------
# (I) TestSampleIndependence
#
# Unrolling twice gives equal but independent samples; mutating sample[0]
# does not ripple to sample[1].
# -----------------------------------------------------------------------------


class TestSampleIndependence:
    """Steps produced by ``unroll`` must be independent objects
    (mutation of one does not affect others)."""

    def test_unroll_called_twice_yields_equal_but_independent(self):
        adapter = LiteDesktopUseAdapter()
        sample = _build_desktop_traj(2)
        a = adapter.unroll(sample).steps
        b = adapter.unroll(sample).steps
        assert len(a) == len(b) == 2
        for sa, sb in zip(a, b):
            assert sa == sb
            assert sa is not sb

    def test_mutation_of_one_step_does_not_affect_siblings(self):
        adapter = LiteDesktopUseAdapter()
        sample = _build_desktop_traj(2)
        steps = adapter.unroll(sample).steps
        assert len(steps) == 2
        # Mutate the target assistant of the first step.
        tool_call_arguments(steps[0][-1]["tool_calls"][0])["coordinate"] = [-1, -1]
        for m in steps[1]:
            for tc in m.get("tool_calls", []) or []:
                assert tool_call_arguments(tc).get("coordinate") != [-1, -1]


# -----------------------------------------------------------------------------
# (J) TestEdgeCases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Empty / single-turn / long / no-tool_calls edge behavior."""

    def test_single_turn_unroll_adds_system_prompt(self):
        adapter = LiteDesktopUseAdapter()
        steps = adapter.unroll(_build_desktop_traj(1)).steps
        assert len(steps) == 1
        assert [m["role"] for m in steps[0]] == ["system", "user", "assistant"]

    def test_ten_turn_unroll_yields_ten_steps_full_history(self):
        """10-turn trajectory → 10 steps; FullHistoryProtocol keeps every
        prior turn, so step[k] has 1 system + 2*(k+1) turn messages."""
        adapter = LiteDesktopUseAdapter()
        steps = adapter.unroll(_build_desktop_traj(10)).steps
        assert len(steps) == 10
        for i, step in enumerate(steps):
            assert len(step) == 1 + 2 * (i + 1)

    def test_assistant_without_tool_calls_keeps_its_text(self):
        """A no-tool-call turn is the TERMINAL turn — its ``text`` must survive.

        This assertion is the reverse of what it used to be, deliberately. The
        old test pinned ``content == []``, i.e. the adapter discarded the final
        turn's prose, which renders an EMPTY SFT target that no count-based check
        can detect (message counts and image counts both still match).

        A turn with no ``tool_calls`` is the termination signal, and every
        dataset-building path normalizes it to exactly one plain ``text`` part
        (``lite.data.utils.messages.normalize_content_only_final``). The
        ``action_description`` whitelist below still applies to turns that DO carry
        an action — that is the only place narration belongs.
        """
        adapter = LiteDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "The answer is 42."}],
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["role"] == "assistant"
        assert out["content"] == [{"type": "text", "text": "The answer is 42."}]

    def test_assistant_without_tool_calls_still_drops_non_text_kinds(self):
        """Only ``text`` survives the terminal branch — not every content kind.

        The fallback is deliberately narrow: the data layer guarantees a
        no-tool-call final carries one ``text`` part, so anything else on such a
        turn is non-conforming input and keeps being dropped rather than leaking
        a kind this wire format has no slot for.
        """
        adapter = LiteDesktopUseAdapter()
        out = adapter.convert_message_to_agent(
            {
                "role": "assistant",
                "content": [{"type": "inline_reasoning", "text": "thinking out loud"}],
            }
        )
        assert out["content"] == []

    def test_assistant_without_tool_calls_to_agent_with_action_description(self):
        """convert_message_to_agent renders action_description as Action: text."""
        adapter = LiteDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "The answer is 42."}],
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["content"] == [
            {"type": "text", "text": "Action: The answer is 42."},
        ]

    def test_assistant_without_tool_calls_from_agent(self):
        """convert_message_from_agent keeps no-tool-call prose as terminal text."""
        adapter = LiteDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "The answer is 42."}],
        }
        result = adapter.convert_message_from_agent(msg)
        assert result == {
            "role": "assistant",
            "content": [{"type": "text", "text": "The answer is 42."}],
        }

    def test_parse_raw_raises_not_implemented(self):
        """LiteBaseAdapter is a data-processing helper (used by export_sft),
        not a live-inference adapter. parse_raw_assistant_response is never
        called on it and should fail loudly if it ever is."""
        import pytest

        adapter = LiteDesktopUseAdapter()
        with pytest.raises(NotImplementedError, match="SFT data processing"):
            adapter.parse_raw_assistant_response("anything")


# -----------------------------------------------------------------------------
# (K) TestBrowserNavCarryThrough
#
# lite-as-student SFT replay: a lite assistant turn carrying a *browser nav*
# tool_call (goto/back/...) must survive convert_message_to_agent / unroll
# byte-exact. The desktop/mobile golden tables above don't include nav verbs
# (those live only in LiteBrowserNavToolSet/extra_tool_schemas), so this pins the browser
# case directly.
# -----------------------------------------------------------------------------

_BROWSER_NAV_TOOLS: dict[str, dict] = {
    "goto": LiteBrowserNavToolSet.goto(url="https://example.com"),
    "back": LiteBrowserNavToolSet.back(),
    "forward": LiteBrowserNavToolSet.forward(),
    "new_tab": LiteBrowserNavToolSet.new_tab(),
    "switch_tab": LiteBrowserNavToolSet.switch_tab(index=2),
    "close_tab": LiteBrowserNavToolSet.close_tab(),
    # standalone extra tool, not part of the browser GUI/nav action space
    "response": make_tool_call("response", {"text": "42"}),
}


def _browser_metadata_for_extra(name: str) -> LiteCUAMetadata:
    if name in LiteBrowserNavToolSet.get_tool_names():
        schemas = LiteBrowserNavToolSet.get_tool_schemas(include=[name])
    elif name == "response":
        schemas = [LiteFinishToolSet.get_tool_schema("response")]
    else:
        schemas = []
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=schemas,
    )


def test_base_adapter_active_extra_collision_routes_key_match_to_env_feedback() -> None:
    """Same-name env extras route by key shape; env owns type feedback."""
    schema = make_tool_schema(
        "click",
        description="Set-of-marks click",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )
    metadata = LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=[schema],
    )
    adapter = AgentAdapterRegistry.get("lite@browser@use", metadata=metadata)

    # A `click` call keyed by `coordinate` does not have the env extra's key
    # shape (`index`), so routing must fall through to ordinary GUI/action-space
    # conversion rather than being admitted as the standalone env tool. Assert
    # this against the same public `action_space.convert_tool_calls_from_agent`
    # entry point GUI calls normally go through, not a private routing
    # predicate: if routing ever misclassified a coordinate-shaped call as the
    # env extra, this call would come back unconverted (as a bare passthrough)
    # instead of matching the GUI conversion result.
    coordinate_call = {"name": "click", "arguments": {"coordinate": [10, 20]}}
    assert adapter._route_agent_tool_calls_to_lite([coordinate_call]) == (
        adapter.action_space.convert_tool_calls_from_agent(
            [coordinate_call],
            active_extra_tool_names=adapter.active_extra_tool_names(),
            active_extra_tool_schemas=list(adapter.metadata.extra_tool_schemas),
        )
    )

    out = adapter._route_agent_tool_calls_to_lite(
        [
            {
                **{"name": "click", "arguments": {"index": 7}},
                "call_id": "same_name_extra",
            },
        ]
    )

    assert out == [make_tool_call("click", {"index": 7}, call_id="same_name_extra")]
    bad = adapter._route_agent_tool_calls_to_lite(
        [
            {
                **{"name": "click", "arguments": {"index": "7"}},
                "call_id": "bad_same_name_extra",
            },
        ]
    )
    assert bad == [make_tool_call("click", {"index": "7"}, call_id="bad_same_name_extra")]
    routed, feedback = prepare_env_tool_calls(bad, metadata)
    assert routed == []
    assert feedback
    assert "click.arguments.index must be an integer" in (feedback["bad_same_name_extra"].message)


def test_base_adapter_parse_direction_reads_only_the_bare_agent_call_shape() -> None:
    """Agent -> Lite routing accepts the bare ``{name, arguments}`` shape only.

    The canonical nested envelope is the RENDER-direction shape. If one reaches
    ``_route_agent_tool_calls_to_lite`` it must fail loudly through the bare-call
    owner (``action_space.convert_tool_calls_from_agent``) rather than be sniffed
    back into the standalone-extra route, even when its function name is an
    active extra whose key shape it satisfies.
    """
    schema = make_tool_schema(
        "click",
        description="Set-of-marks click",
        parameters={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )
    metadata = LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=[schema],
    )
    adapter = AgentAdapterRegistry.get("lite@browser@use", metadata=metadata)

    # Bare shape: routed standalone, as the sibling test above pins.
    assert adapter._route_agent_tool_calls_to_lite(
        [{"name": "click", "arguments": {"index": 7}}]
    ) == [make_tool_call("click", {"index": 7})]

    with pytest.raises(ValueError, match=r"non-bare-call keys \['function', 'type'\]"):
        adapter._route_agent_tool_calls_to_lite([make_tool_call("click", {"index": 7})])


class TestBrowserNavCarryThrough:
    """Browser nav/answer tool_calls survive the lite navigation adapter unchanged
    (change #0): content folds to ``Action:``; tool_calls pass through identity."""

    @pytest.mark.parametrize("name,tc", list(_BROWSER_NAV_TOOLS.items()))
    def test_browser_nav_tool_call_survives_to_agent(self, name, tc):
        adapter = AgentAdapterRegistry.get(
            "lite@browser@use",
            metadata=_browser_metadata_for_extra(name),
        )
        msg = {
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Perform {name}."}],
            "tool_calls": [tc],
        }
        out = adapter.convert_message_to_agent(msg)
        assert out["content"] == [{"type": "text", "text": f"Action: Perform {name}."}]
        assert out["tool_calls"] == [tc]  # nav tool_call carried through verbatim

    def test_browser_nav_unroll_target_byte_exact(self):
        """A 2-turn browser trajectory (goto then back) unrolls with both nav
        tool_calls intact on each step's assistant target."""
        nav = [LiteBrowserNavToolSet.goto(url="https://example.com"), LiteBrowserNavToolSet.back()]
        msgs: list[dict] = []
        for i in range(2):
            content = [{"type": "image", "index": i}]
            if i == 0:
                content.append({"type": "text", "text": "Find the answer."})
            msgs.append({"role": "user", "content": content})
            msgs.append(
                {
                    "role": "assistant",
                    "content": [{"type": "action_description", "text": f"step {i}"}],
                    "tool_calls": [nav[i]],
                }
            )
        metadata = LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=LiteBrowserNavToolSet.get_tool_schemas(include=["goto", "back"]),
        )
        sample = LiteSample(
            metadata=metadata,
            messages=msgs,
            images=["img0.png", "img1.png"],
        )
        adapter = AgentAdapterRegistry.get("lite@browser@use", metadata=metadata)
        steps = adapter.unroll(sample).steps
        assert len(steps) == 2
        for i, step in enumerate(steps):
            tgt = [m for m in step if m["role"] == "assistant"][-1]
            assert tgt["tool_calls"] == [nav[i]]
