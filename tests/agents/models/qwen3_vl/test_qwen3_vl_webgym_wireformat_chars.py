"""Qwen3-VL WebGym wire-format characterizations."""

from __future__ import annotations

from lite_samples import sample_trajectory_long

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.schemas import tool_schema_name

register_all()


def _md(platform: str = "desktop", extra_tool_schemas=None) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(LiteCUAMetadata.Platform(platform), LiteCUAMetadata.TaskType.USE),
        extra_tool_schemas=extra_tool_schemas or [],
        others={"id": "wire-char"},
    )


_GOTO_SCHEMA = make_tool_schema(
    "goto",
    description="Navigate to a URL.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
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


def _first_user_text(step: list[dict]) -> str:
    for message in step:
        if message.get("role") == "user":
            return "\n".join(
                part.get("text", "")
                for part in (message.get("content") or [])
                if part.get("type") == "text"
            )
    return ""


def _system_text(step: list[dict]) -> str:
    for message in step:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return "\n".join(
                part.get("text", "")
                for part in (content or [])
                if part.get("type") == "text"
            )
    return ""


def _content_parts(lite_msg: dict) -> list[tuple]:
    return [(part.get("type"), part.get("text")) for part in (lite_msg.get("content") or [])]


def _action_descriptions(lite_msg: dict) -> list[str]:
    return [
        part.get("text")
        for part in (lite_msg.get("content") or [])
        if part.get("type") == "action_description"
    ]


def _round_trip_action(adapter, raw: str) -> list[str]:
    parsed = adapter.parse_raw_assistant_response(raw)
    return _action_descriptions(adapter.convert_message_from_agent(parsed))


_RAW_ACTION_QWEN3VL = (
    "Action: Click the login button.\n"
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": '
    '{"action": "left_click", "coordinate": [100, 200]}}\n'
    "</tool_call>"
)
_RAW_ACTION_NO_PREFIX_QWEN3VL = (
    "Just click it now.\n"
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": '
    '{"action": "left_click", "coordinate": [100, 200]}}\n'
    "</tool_call>"
)
_RAW_THOUGHT = (
    "Thought: I should open settings.\nAction: Click the settings icon.\n"
    "<tool_call>\n"
    '{"name": "computer_use", "arguments": '
    '{"action": "left_click", "coordinate": [50, 50]}}\n'
    "</tool_call>"
)

_OFFICIAL_SP = (
    "You are a helpful assistant.\n\n# Response format\n\n"
    "1) Action: ...\n2) <tool_call>...</tool_call>"
)


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


class TestQwen3VLGeneralNav:
    def _adapter(self):
        return AgentAdapterRegistry.get("qwen3_vl@desktop@use", metadata=_md("desktop"))

    def test_recipe_defaults(self):
        adapter = self._adapter()
        assert type(adapter.protocol).__name__ == "Qwen3VLHistoryProtocol"
        assert adapter.enable_inline_reasoning is False

    def test_to_agent_action_only_prefix_and_windowing(self):
        adapter = self._adapter()
        steps = adapter.unroll(sample_trajectory_long(6)).steps
        assert len(steps) == 6
        assert _assistant_texts(steps[-1]) == [
            "Action: Action step 3.",
            "Action: Action step 4.",
            "Action: Action step 5.",
            "Action: Action step 6.",
        ]
        summary = _first_user_text(steps[-1])
        assert "Instruction: Open GIMP and apply filter." in summary
        assert "Step 1: Action step 1." in summary
        assert "Step 2: Action step 2." in summary

    def test_from_agent_extracts_action_line(self):
        adapter = self._adapter()
        assert _round_trip_action(adapter, _RAW_ACTION_QWEN3VL) == [
            "Click the login button."
        ]

    def test_from_agent_no_prefix_falls_back_to_first_line(self):
        adapter = self._adapter()
        assert _round_trip_action(adapter, _RAW_ACTION_NO_PREFIX_QWEN3VL) == [
            "Just click it now."
        ]


class TestQwen3VLWebGymOfficial:
    def _adapter(self):
        return AgentAdapterRegistry.get(
            "qwen3_vl@browser@use",
            metadata=_md("browser", extra_tool_schemas=[_GOTO_SCHEMA]),
            system_prompt=_OFFICIAL_SP,
        )

    def test_recipe_is_action_only(self):
        adapter = self._adapter()
        assert type(adapter.protocol).__name__ == "Qwen3VLHistoryProtocol"
        assert adapter.enable_inline_reasoning is False

    def test_to_agent_identical_action_only_wire_format(self):
        adapter = self._adapter()
        steps = adapter.unroll(sample_trajectory_long(6)).steps
        assert _assistant_texts(steps[-1]) == [
            "Action: Action step 3.",
            "Action: Action step 4.",
            "Action: Action step 5.",
            "Action: Action step 6.",
        ]

    def test_system_prompt_and_extra_tools_present(self):
        adapter = self._adapter()
        steps = adapter.unroll(sample_trajectory_long(6)).steps
        assert _OFFICIAL_SP in _system_text(steps[-1])
        names = {tool_schema_name(schema) for schema in adapter._assemble_tool_schemas()}
        assert "goto" in names

    def test_from_agent_extracts_action_line(self):
        adapter = self._adapter()
        assert _round_trip_action(adapter, _RAW_ACTION_QWEN3VL) == [
            "Click the login button."
        ]


def _inline_adapter(enabled: bool):
    return AgentAdapterRegistry.get(
        "qwen3_vl@browser@use",
        metadata=_md("browser"),
        enable_inline_reasoning=enabled,
    )


def test_to_agent_disabled_drops_thought():
    adapter = _inline_adapter(False)
    rendered = _assistant_texts(adapter.unroll(_sample_inline_reasoning()).steps[-1])[0]
    assert "Thought:" not in rendered
    assert rendered.startswith("Action: Click the settings icon.")


def test_to_agent_enabled_prepends_thought():
    adapter = _inline_adapter(True)
    rendered = _assistant_texts(adapter.unroll(_sample_inline_reasoning()).steps[-1])[0]
    assert rendered.startswith("Thought: I should open the settings app first.")
    assert "Action: Click the settings icon." in rendered


def test_from_agent_disabled_drops_thought():
    adapter = _inline_adapter(False)
    lite_msg = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_RAW_THOUGHT)
    )
    assert _content_parts(lite_msg) == [("action_description", "Click the settings icon.")]


def test_from_agent_enabled_extracts_thought():
    adapter = _inline_adapter(True)
    lite_msg = adapter.convert_message_from_agent(
        adapter.parse_raw_assistant_response(_RAW_THOUGHT)
    )
    assert _content_parts(lite_msg) == [
        ("inline_reasoning", "I should open settings."),
        ("action_description", "Click the settings icon."),
    ]
