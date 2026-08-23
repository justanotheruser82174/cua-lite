"""Coverage pins for non-release mobile agent-family rows.

The row matrix lives elsewhere; this file deliberately does not modify it.  It
locks the prompt/action/result surfaces those rows rely on for:

* Claude config rows (API loop; adapter-free)
* MAI-UI / UI-TARS / StepGUI mobile adapters

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/test_nonrelease_mobile_family_coverage.py -p no:cacheprovider -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.step_gui.action_space import STEPGUIMobileActionSpace
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.image_refs import referenced_images_in_message_order
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.extra_tools import LiteFinishToolSet, make_open_app_tool

register_all()

ROOT = Path(__file__).resolve().parents[2]
MOBILE_ENVS = ("androidlab", "androidworld", "mobilegym", "mobileworld")


def _ask_user_schema() -> dict[str, Any]:
    return make_tool_schema(
        "ask_user",
        description="Ask the user for clarification.",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    )


def _mobile_surface_schemas() -> list[dict[str, Any]]:
    return [
        make_open_app_tool(["Settings", "Clock"]),
        LiteFinishToolSet.get_tool_schema("response"),
        LiteFinishToolSet.get_tool_schema("terminate"),
        _ask_user_schema(),
    ]


def _metadata(platform: str) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(platform, LiteCUAMetadata.TaskType.USE), extra_tool_schemas=_mobile_surface_schemas()
    )


def _adapter_for(adapter_key: str):
    platform = "mobile" if "@mobile" in adapter_key else "desktop"
    return AgentAdapterRegistry.get(adapter_key, metadata=_metadata(platform))


def _assistant_message(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "do it"}],
        "tool_calls": tool_calls,
    }


def _render_text(adapter_key: str, tool_calls: list[dict[str, Any]]) -> str:
    rendered = _adapter_for(adapter_key).convert_message_to_agent(_assistant_message(tool_calls))
    return "\n".join(
        part.get("text", "")
        for part in rendered.get("content") or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _mobile_batch(action: dict[str, Any]) -> dict[str, Any]:
    return make_tool_call("mobile", {"actions": [action]})


_EXPECTED_CONFIG_EXTRAS: dict[tuple[str, str], tuple[str, ...]] = {}
for _family in ("mai_ui", "step_gui"):
    for _env in MOBILE_ENVS:
        _EXPECTED_CONFIG_EXTRAS[(_family, _env)] = (
            "open_app",
            "response",
            "terminate",
        )
for _env in MOBILE_ENVS:
    _EXPECTED_CONFIG_EXTRAS[("ui_tars", _env)] = ("response", "terminate")
    _EXPECTED_CONFIG_EXTRAS[("claude", _env)] = (
        (
            "open_app",
            "ask_user",
            "response",
            "terminate",
        )
        if _env == "mobileworld"
        else (
            "open_app",
            "response",
            "terminate",
        )
    )


@pytest.mark.parametrize(
    "family,env_id,expected",
    [
        (family, env_id, expected)
        for (family, env_id), expected in sorted(_EXPECTED_CONFIG_EXTRAS.items())
    ],
)
def test_mobile_config_rows_pin_extra_tool_surface(
    family: str,
    env_id: str,
    expected: tuple[str, ...],
) -> None:
    path = ROOT / "scripts" / "configs" / family / "default" / f"{env_id}.yaml"
    data = yaml.safe_load(path.read_text())

    assert data["agent_id"] == family
    assert data["env_id"] == env_id
    assert tuple(data["env_kwargs"]["extra_tools"]) == expected


def test_mai_ui_mobile_native_semantic_surface_projection() -> None:
    assert '"action":"open"' in _render_text(
        "mai_ui@mobile@use",
        [make_tool_call("open_app", {"app_name": "Settings"})],
    )
    assert '"text":"Settings"' in _render_text(
        "mai_ui@mobile@use",
        [make_tool_call("open_app", {"app_name": "Settings"})],
    )
    assert '"action":"answer"' in _render_text(
        "mai_ui@mobile@use",
        [make_tool_call("response", {"text": "42"})],
    )
    assert '"action":"terminate"' in _render_text(
        "mai_ui@mobile@use",
        [make_tool_call("terminate", {"status": "success"})],
    )
    assert '"action":"system_button","button":"home"' in _render_text(
        "mai_ui@mobile@use",
        [_mobile_batch({"action": "system_button", "button": "Home"})],
    )


def test_step_gui_mobile_native_semantic_surface_projection() -> None:
    assert "action:AWAKE\tvalue:Settings" in _render_text(
        "step_gui@mobile@use",
        [make_tool_call("open_app", {"app_name": "Settings"})],
    )
    assert "action:INFO\tvalue:42" in _render_text(
        "step_gui@mobile@use",
        [make_tool_call("response", {"text": "42"})],
    )
    assert "action:COMPLETE\treturn:done" in _render_text(
        "step_gui@mobile@use",
        [make_tool_call("terminate", {"status": "success", "reason": "done"})],
    )
    assert "system_button" not in STEPGUIMobileActionSpace.get_declared_action_schema_names()


def test_ui_tars_mobile_finish_and_system_button_surface_projection() -> None:
    assert "finished(content='42')" in _render_text(
        "ui_tars@mobile@use",
        [make_tool_call("response", {"text": "42"})],
    )
    assert "finished()" in _render_text(
        "ui_tars@mobile@use",
        [make_tool_call("terminate", {"status": "success"})],
    )
    assert "press_home()" in _render_text(
        "ui_tars@mobile@use",
        [_mobile_batch({"action": "system_button", "button": "Home"})],
    )
    assert "press_back()" in _render_text(
        "ui_tars@mobile@use",
        [_mobile_batch({"action": "system_button", "button": "Back"})],
    )

    with pytest.raises(ValueError, match="cannot render system_button 'Enter'"):
        _render_text(
            "ui_tars@mobile@use",
            [_mobile_batch({"action": "system_button", "button": "Enter"})],
        )


def _img(k: int) -> Image.Image:
    return Image.new("RGB", (40 + k, 30 + k), color=(k * 80 % 255, 10, 20))


def _role_tool_image_sample(platform: str) -> LiteSample:
    wrapper = "mobile" if platform == "mobile" else "computer"
    action = "tap" if platform == "mobile" else "click"
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=[_img(0), _img(1)],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open Settings."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Tap."}],
                "tool_calls": [
                    make_tool_call(
                        wrapper,
                        {
                            "actions": [
                                {"action": action, "coordinate": [100, 200]},
                            ],
                        },
                        call_id="call_0000",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [
                    {"type": "image", "index": 1},
                    {"type": "text", "text": "SENTINEL current observation"},
                ],
            },
        ],
    )


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        ("mai_ui@mobile@use", "mobile"),
        ("step_gui@mobile@use", "mobile"),
        ("ui_tars@mobile@use", "mobile"),
    ],
)
def test_role_tool_image_placeholder_binds_to_processed_image_data(
    adapter_key: str,
    platform: str,
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    out = adapter.unroll(_role_tool_image_sample(platform))

    bound_images = referenced_images_in_message_order(out.steps[-1], out.processed_images)

    assert bound_images, f"{adapter_key} rendered no image placeholders"
    assert any(img is out.processed_images[1] for img in bound_images), (
        f"{adapter_key} did not bind the current role:tool screenshot to the processed image list"
    )
