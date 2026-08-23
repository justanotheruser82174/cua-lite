"""Qwen3.5 WebGym wire-format characterizations."""

from __future__ import annotations

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call

register_all()


def _md(platform: str = "browser") -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform(platform), LiteCUAMetadata.TaskType.USE),
        others={"id": "wire-char"},
    )


def _assistant_texts(step: list[dict]) -> list[str]:
    out = []
    for message in step:
        if message.get("role") == "assistant":
            out.append(
                "\n".join(
                    part.get("text", "")
                    for part in (message.get("content") or [])
                    if part.get("type") == "text"
                )
            )
    return out


def _content_parts(lite_msg: dict) -> list[tuple]:
    return [(part.get("type"), part.get("text")) for part in (lite_msg.get("content") or [])]


def _sample_inline_reasoning() -> LiteSample:
    return LiteSample(
        images=["img0.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open settings."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "inline_reasoning",
                        "text": "I should open the settings app first.",
                    },
                    {"type": "action_description", "text": "Click the settings icon."},
                ],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [50, 50]}]},
                        call_id="call_0000",
                    )
                ],
            },
        ],
        metadata=_md("browser"),
    )


_RAW_THOUGHT = (
    "Thought: I should open settings.\nAction: Click the settings icon.\n"
    "<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n</parameter>\n"
    "<parameter=coordinate>\n[50, 50]\n</parameter>\n</function>\n</tool_call>"
)


def _adapter(enabled: bool):
    return AgentAdapterRegistry.get(
        "qwen3_5@browser@use",
        metadata=_md("browser"),
        enable_inline_reasoning=enabled,
    )


def test_to_agent_disabled_drops_thought():
    adapter = _adapter(False)
    rendered = _assistant_texts(adapter.unroll(_sample_inline_reasoning()).steps[-1])[0]
    assert "Thought:" not in rendered
    assert rendered.startswith("Action: Click the settings icon.")


def test_to_agent_enabled_prepends_thought():
    adapter = _adapter(True)
    rendered = _assistant_texts(adapter.unroll(_sample_inline_reasoning()).steps[-1])[0]
    assert rendered.startswith("Thought: I should open the settings app first.")
    assert "Action: Click the settings icon." in rendered


def test_from_agent_disabled_drops_thought():
    adapter = _adapter(False)
    lite_msg = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_RAW_THOUGHT)
    )
    assert _content_parts(lite_msg) == [("action_description", "Click the settings icon.")]


def test_from_agent_enabled_extracts_thought():
    adapter = _adapter(True)
    lite_msg = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_RAW_THOUGHT)
    )
    assert _content_parts(lite_msg) == [
        ("inline_reasoning", "I should open settings."),
        ("action_description", "Click the settings icon."),
    ]
