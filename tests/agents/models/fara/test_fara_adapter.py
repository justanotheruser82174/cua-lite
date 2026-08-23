"""
Tests for the Fara-1.0 adapter family.

Covers the Fara-specific deltas over the inherited Qwen2.5-VL machinery:
  1. Registry resolution (``fara.base`` + ``fara@(desktop|browser)@...`` regex keys)
  2. smart_resize factor=28 + Fara's full max_pixels cap
  3. ``{display_width_px}x{display_height_px}`` substitution == resized dims
  4. System message = "You are a helpful assistant." + FN_CALL_TEMPLATE preamble
     (NOT the Qwen "# Tools" header), with NO ``Action:`` / ``Thought:`` format
  4b. ``<tools>`` envelope is the NESTED Hermes form (see TestToolsEnvelope)
  5. Coordinate [0,1000] ↔ pixel-in-resized round-trip
  6. ``parse_raw_assistant_response`` (thoughts prose + ``<tool_call>``)
  7. ``FaraHistoryProtocol`` sliding image window

Run:
    uv run pytest tests/agents/models/fara/test_fara_adapter.py -v
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from PIL import Image

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.fara.adapter import (
    _MAX_PIXELS,
    FaraBaseAdapter,
    FaraDesktopGroundingActionAdapter,
    FaraDesktopGroundingPointAdapter,
    FaraDesktopUseAdapter,
)
from lite.agents.models.fara.protocol import FaraHistoryProtocol
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_arguments, tool_call_name

_TERMINATE_SCHEMA = make_tool_schema(
    "terminate",
    description="Finish the task.",
    parameters={"type": "object", "properties": {"status": {"type": "string"}}, "required": []},
)

_RESPONSE_SCHEMA = make_tool_schema(
    "response",
    description="Submit a final answer to the task.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


def _use_sample(n_turns: int = 2, w: int = 1440, h: int = 900) -> LiteSample:
    """A web-navigation sample with real PIL screenshots (needed for resize)."""
    images = [Image.new("RGB", (w, h), "white") for _ in range(n_turns)]
    messages: list[dict] = []
    for i in range(n_turns):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": i},
                    {"type": "text", "text": "Find the price." if i == 0 else "next screenshot"},
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"I will click item {i}."},
                ],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [500, 300]}]},
                        call_id=f"call_{i:04d}",
                    ),
                ],
            }
        )
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("browser", "use")),
        messages=messages,
        images=images,
    )


# =============================================================================
# 1. Registry resolution
# =============================================================================


class TestRegistry:
    def test_base(self):
        assert isinstance(AgentAdapterRegistry.get("fara.base"), FaraBaseAdapter)

    def test_use_desktop_and_browser(self):
        assert isinstance(AgentAdapterRegistry.get("fara@desktop@use"), FaraDesktopUseAdapter)
        assert isinstance(AgentAdapterRegistry.get("fara@browser@use"), FaraDesktopUseAdapter)

    def test_grounding_action(self):
        assert isinstance(
            AgentAdapterRegistry.get("fara@browser@grounding.action"),
            FaraDesktopGroundingActionAdapter,
        )

    def test_grounding_point(self):
        a = AgentAdapterRegistry.get("fara@desktop@grounding.point")
        assert isinstance(a, FaraDesktopGroundingPointAdapter)
        # Uses the grounding rules block, NOT Fara's web-automation preamble.
        assert a.system_prompt is not None and "grounding" in a.system_prompt.lower()
        assert a.smart_resize_max_pixels == _MAX_PIXELS


# =============================================================================
# 2. smart_resize constants
# =============================================================================


class TestResizeConstants:
    def test_factor_and_max_pixels(self):
        a = FaraDesktopUseAdapter()
        assert a.smart_resize_factor == 28
        assert a.smart_resize_max_pixels == _MAX_PIXELS == 16384 * 28 * 28

    def test_image_snapped_to_28_grid(self):
        a = FaraDesktopUseAdapter()
        out = a.process_image(Image.new("RGB", (1440, 900), "white"))
        w, h = out.size
        assert w % 28 == 0 and h % 28 == 0
        assert (w, h) == (1428, 896)


# =============================================================================
# 3. Resolution substitution + 4. system message shape
# =============================================================================


class TestSystemMessage:
    def _system_text(self, sample: LiteSample) -> tuple[str, tuple[int, int]]:
        a = AgentAdapterRegistry.get("fara@browser@use")
        agent_sample = a.unroll(sample)
        size = agent_sample.processed_images[-1].size
        return agent_sample.steps[-1][0]["content"][0]["text"], size

    def test_resolution_matches_resized_dims(self):
        text, (w, h) = self._system_text(_use_sample())
        assert f"The screen's resolution is {w}x{h}." in text
        assert "{display_width_px}" not in text

    def test_helpful_assistant_prefix_and_fn_call_template(self):
        text, _ = self._system_text(_use_sample())
        assert text.startswith("You are a helpful assistant.\n\n")
        assert "You are a web automation agent" in text
        assert "Critical Point" in text
        # Fara uses FN_CALL_TEMPLATE, not the Qwen "# Tools" header.
        assert "# Tools\n" not in text
        assert "<tools>" in text and "<tool_call>" in text

    def test_tools_section_filters_valid_actions_but_does_not_render_metadata_extras(self):
        adapter = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=("browser", "use"),
                valid_actions=[],
                extra_tool_schemas=[
                    make_tool_schema(
                        "response",
                        description="Submit a final answer.",
                        parameters={"type": "object", "properties": {}, "required": []},
                    )
                ],
            )
        )
        text = adapter._build_tools_section(image_size=(1280, 720))
        assert '"name": "computer_use"' not in text
        assert '"name": "response"' not in text


# =============================================================================
# 4b. <tools> envelope oracle
# =============================================================================

# Fara is a Qwen2.5-VL fine-tune and, like the Qwen family, renders canonical
# nested Lite schemas directly into the ``<tools>`` block. The sibling oracle for
# those keys lives in
# ``tests/agents/models/qwen2_5_vl/test_qwen2_5_vl_tools_envelope_oracle.py``.
#
# NOT an oracle: ``apply_chat_template(msgs, tools=[<dict>])``. The Qwen-family
# template renders ``{{- tool | tojson }}``, a pass-through that echoes whatever
# envelope it is handed, so feeding it one shape and seeing the same shape proves
# nothing about what Fara was trained on.

_FARA_TOOLS_ADAPTER_KEYS = [
    "fara@browser@use",
    "fara@desktop@use",
    "fara@browser@grounding.action",
    "fara@desktop@grounding.action",
]


def _fara_tools_section(adapter_key: str, *, with_extras: bool) -> str:
    metadata = LiteCUAMetadata(
        dims=("browser" if "@browser@" in adapter_key else "desktop", adapter_key.split("@")[-1]),
        extra_tool_schemas=[_TERMINATE_SCHEMA] if with_extras else [],
    )
    return AgentAdapterRegistry.get(adapter_key, metadata=metadata)._build_tools_section()


def _fara_tool_objects(section: str) -> list[dict]:
    """Every JSON object on its own line inside the ``<tools>`` block."""
    body = re.search(r"<tools>\n(.*?)\n</tools>", section, re.DOTALL)
    assert body is not None, f"no <tools> block in section:\n{section[:400]}"
    lines = [ln for ln in body.group(1).splitlines() if ln.strip()]
    assert lines, "empty <tools> block"
    return [json.loads(ln) for ln in lines]


class TestToolsEnvelope:
    @pytest.mark.parametrize("adapter_key", _FARA_TOOLS_ADAPTER_KEYS)
    @pytest.mark.parametrize("with_extras", [False, True], ids=["bare", "extras"])
    def test_envelope_matches_transformers_oracle(self, adapter_key, with_extras):
        """Oracle 1: ``transformers.utils.get_json_schema`` -- the builder
        ``apply_chat_template`` itself uses -- emits ``{"type", "function"}``."""
        from transformers.utils import get_json_schema

        def computer_use(action: str) -> None:
            """Use a mouse and keyboard.

            Args:
                action: the action to perform
            """

        expected_keys = set(get_json_schema(computer_use).keys())
        assert expected_keys == {"type", "function"}, (
            "transformers no longer emits the nested envelope; re-derive the "
            "premise before trusting this test"
        )
        for tool in _fara_tool_objects(_fara_tools_section(adapter_key, with_extras=with_extras)):
            assert set(tool.keys()) == expected_keys, (
                f"{adapter_key}: tool envelope {sorted(tool)} != oracle "
                f"{sorted(expected_keys)} -- the canonical schema was serialized wrongly"
            )
            assert tool["type"] == "function"
            inner = tool["function"]
            assert isinstance(inner, dict)
            assert isinstance(inner["name"], str)
            assert "parameters" in inner
            # The flat form's discriminator must not survive inside the envelope.
            assert inner.get("type") != "function"

    @pytest.mark.parametrize("adapter_key", _FARA_TOOLS_ADAPTER_KEYS)
    def test_no_flat_schema_marker(self, adapter_key):
        section = _fara_tools_section(adapter_key, with_extras=True)
        assert '{"type": "function", "name"' not in section, (
            f"{adapter_key}: rendered <tools> carries a legacy flat envelope"
        )
        assert '{"type": "function", "function":' in section

    @pytest.mark.skipif(
        not (
            os.environ.get("CUA_LITE_REFERENCES_ROOT")
            and (
                Path(os.environ["CUA_LITE_REFERENCES_ROOT"])
                / "fara/src/fara/qwen_helpers/fncall_prompt.py"
            ).is_file()
        ),
        reason="CUA_LITE_REFERENCES_ROOT not mounted; transformers oracle still applies",
    )
    def test_fara_reference_builds_nested_envelope(self):
        """Oracle 2: Fara's own ``NousFnCallPrompt`` wraps every function.

        This is the strongest available statement of what Fara-1.0 saw at SFT
        time -- the same helper that renders ``FN_CALL_TEMPLATE``.
        """
        text = (
            Path(os.environ["CUA_LITE_REFERENCES_ROOT"])
            / "fara/src/fara/qwen_helpers/fncall_prompt.py"
        ).read_text(encoding="utf-8", errors="replace")
        assert '[{"type": "function", "function": f} for f in functions]' in text, (
            "the Fara reference no longer builds the nested envelope; the "
            "premise of this adapter's serialization must be re-derived"
        )
        # ...and dumps it without ASCII-escaping, which the adapter mirrors.
        assert "json.dumps(f, ensure_ascii=False) for f in tool_descs" in text

    def test_non_ascii_is_not_escaped(self):
        """Fara's reference dumps with ``ensure_ascii=False``; so must we."""
        adapter = FaraDesktopUseAdapter(metadata=LiteCUAMetadata(dims=("browser", "use")))
        adapter.action_space.get_tool_schemas = lambda: [  # type: ignore[method-assign]
            make_tool_schema(
                "unicode_tool",
                description="Ré­sumé — ✓",
                parameters={"type": "object", "properties": {}, "required": []},
            )
        ]
        section = adapter._build_tools_section()
        assert "✓" in section and "\\u2713" not in section


# =============================================================================
# 5. Coordinate round-trip  +  6. parse_raw
# =============================================================================


class TestWireFormat:
    def test_to_agent_normalizes_to_pixel(self):
        a = AgentAdapterRegistry.get("fara@browser@use")
        agent_sample = a.unroll(_use_sample())
        w, h = agent_sample.processed_images[0].size
        # First assistant message in the final rendered step.
        assistant = next(
            m["content"][0]["text"]
            for m in agent_sample.steps[-1]
            if m["role"] == "assistant" and m.get("content")
        )
        # thoughts prose precedes the tool_call; no Action:/Thought: labels.
        assert "Action:" not in assistant and "Thought:" not in assistant
        assert "<tool_call>" in assistant
        # click [500,300] normalized → pixel [500/1000*w, 300/1000*h]
        assert f'"coordinate": [{round(500 / 1000 * w)}, {round(300 / 1000 * h)}]' in assistant

    def test_terminate_thoughts_become_the_answer_response(self):
        a = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
                extra_tool_schemas=[_RESPONSE_SCHEMA, _TERMINATE_SCHEMA],
            )
        )
        raw = (
            "The cheapest flight is $199 on United.\n"
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "terminate", "status": "success"}}\n</tool_call>'
        )
        lite = a.convert_message_from_agent(a.parse_raw_assistant_response(raw))
        names = [tool_call_name(tc) for tc in lite["tool_calls"]]
        assert names == ["response", "terminate"], names
        answer = "The cheapest flight is $199 on United."
        assert tool_call_arguments(lite["tool_calls"][0])["text"] == answer
        assert tool_call_arguments(lite["tool_calls"][1])["status"] == "success"
        assert lite["content"][0]["text"] == answer

    def test_terminate_answer_turn_rerenders_from_raw_response(self):
        a = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
                extra_tool_schemas=[_RESPONSE_SCHEMA, _TERMINATE_SCHEMA],
            )
        )
        raw = (
            "The cheapest flight is $199 on United.\n"
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "terminate", "status": "success"}}\n</tool_call>'
        )
        lite = a.convert_message_from_agent(a.parse_raw_assistant_response(raw))
        lite["raw_response"] = {"text": raw, "adapter_key": a.get_registry_key()}
        assert a.convert_message_to_agent(lite)["content"][0]["text"] == raw

    def test_terminate_without_advertised_response_is_not_augmented(self):
        a = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
                extra_tool_schemas=[_TERMINATE_SCHEMA],
            )
        )
        raw = (
            "The cheapest flight is $199 on United.\n"
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "terminate", "status": "success"}}\n</tool_call>'
        )
        lite = a.convert_message_from_agent(a.parse_raw_assistant_response(raw))
        names = [tool_call_name(tc) for tc in lite["tool_calls"]]
        assert names == ["terminate"], names
        assert tool_call_arguments(lite["tool_calls"][0])["status"] == "success"
        assert lite["content"][0]["text"] == "The cheapest flight is $199 on United."

    def test_terminate_without_thoughts_unchanged(self):
        a = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
                extra_tool_schemas=[_RESPONSE_SCHEMA, _TERMINATE_SCHEMA],
            )
        )
        raw = (
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "terminate", "status": "failure"}}\n</tool_call>'
        )
        lite = a.convert_message_from_agent(a.parse_raw_assistant_response(raw))
        names = [tool_call_name(tc) for tc in lite["tool_calls"]]
        assert names == ["terminate"]  # no thoughts -> no answer to submit

    def test_non_terminate_not_augmented(self):
        a = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
                extra_tool_schemas=[_RESPONSE_SCHEMA, _TERMINATE_SCHEMA],
            )
        )
        a._current_image_size = (1428, 896)
        raw = (
            'Click login.\n<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "left_click", "coordinate": [700, 400]}}\n</tool_call>'
        )
        lite = a.convert_message_from_agent(a.parse_raw_assistant_response(raw))
        names = [tool_call_name(tc) for tc in lite["tool_calls"]]
        assert names == ["computer"]
        assert tool_call_arguments(lite["tool_calls"][0])["actions"][0]["action"] == "click"

    def test_wire_level_response_parse_fails_loudly(self):
        """FLIPPED (was ``test_schema_less_top_level_response_is_not_persisted``):
        a wire-level ``response`` used to be swallowed (``return []``). Fara has
        no native answer channel, so it must fail loudly instead of silently
        dropping an answer — same policy as the render side."""
        a = FaraDesktopUseAdapter()
        raw = 'answer\n<tool_call>\n{"name": "response", "arguments": {"text": "42"}}\n</tool_call>'
        with pytest.raises(ValueError, match="Fara cannot parse wire tool 'response'"):
            a.convert_message_from_agent(a.parse_raw_assistant_response(raw))

    def test_schema_less_wrapper_embedded_response_reaches_env_ingress(self):
        """``computer_use(action="response")`` is off-schema for Fara -- but it must
        cost a TURN, not the EPISODE.

        Fara advertises a STANDALONE ``response`` extra in every one of its rollout
        configs, so a model confusing the tool layer with the action layer is
        expected output, not a defect. Dropping the call left the turn with zero
        tool calls, which ``AgentBase.sample`` treats as terminal.
        """
        a = FaraDesktopUseAdapter()
        raw = (
            "answer\n"
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "response", "text": "42"}}\n</tool_call>'
        )
        lite = a.convert_message_from_agent(a.parse_raw_assistant_response(raw))
        calls = lite.get("tool_calls")
        assert calls, "zero tool calls would end the episode"
        assert tool_call_name(calls[0]) == "computer"
        assert tool_call_arguments(calls[0])["actions"][0]["action"] == "response"
        assert lite["content"][0]["text"] == "answer"

    def test_declared_wrapper_embedded_response_reaches_env_ingress(self):
        a = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=("browser", "use"),
                extra_tool_schemas=[
                    make_tool_schema(
                        "response",
                        description="Submit a final answer.",
                        parameters={"type": "object", "properties": {}, "required": []},
                    )
                ],
            )
        )
        raw = (
            "answer\n"
            '<tool_call>\n{"name": "computer_use", "arguments": '
            '{"action": "response", "text": "42"}}\n</tool_call>'
        )
        lite = a.convert_message_from_agent(a.parse_raw_assistant_response(raw))
        calls = lite.get("tool_calls")
        # Declaring a standalone ``response`` schema does not turn the WRAPPER
        # action of the same spelling into that tool: the action layer and the
        # tool layer stay separate. It still reaches ingress rather than vanishing.
        assert calls, "zero tool calls would end the episode"
        assert tool_call_name(calls[0]) == "computer"
        assert tool_call_arguments(calls[0])["actions"][0]["action"] == "response"
        assert lite["content"][0]["text"] == "answer"

    def test_native_terminate_render_does_not_require_matching_schema(self):
        msg = {
            "role": "assistant",
            "tool_calls": [make_tool_call("terminate", {"status": "success"})],
        }

        rendered = FaraDesktopUseAdapter().convert_message_to_agent(msg)
        assert '"action": "terminate"' in rendered["content"][0]["text"]

        rendered = FaraDesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=("browser", "use"), extra_tool_schemas=[_TERMINATE_SCHEMA]
            )
        ).convert_message_to_agent(msg)
        assert '"action": "terminate"' in rendered["content"][0]["text"]

    def test_parse_and_from_agent_round_trip(self):
        a = FaraDesktopUseAdapter()
        w, h = 1428, 896
        a._current_image_size = (w, h)
        raw = (
            "I will click the login button.\n"
            '<tool_call>\n{"name": "computer_use", "arguments": '
            f'{{"action": "left_click", "coordinate": [{w // 2}, {h // 2}]}}}}\n</tool_call>'
        )
        agent_msg = a.parse_raw_assistant_response(raw)
        assert agent_msg["content"][0]["text"] == "I will click the login button."
        assert agent_msg["tool_calls"][0]["name"] == "computer_use"

        lite = a.convert_message_from_agent(agent_msg)
        tc = lite["tool_calls"][0]
        assert tool_call_name(tc) == "computer"
        action = tool_call_arguments(tc)["actions"][0]
        assert action["action"] == "click"
        # pixel center → normalized ~[500, 500]
        assert action["coordinate"] == [500, 500]

    def test_canonical_nested_input_renders(self):
        a = FaraDesktopUseAdapter()
        msg = {
            "role": "assistant",
            "tool_calls": [make_tool_call("click", {"coordinate": [1, 2]})],
        }
        rendered = a.convert_message_to_agent(msg)
        assert rendered == {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<tool_call>\n"
                        '{"name": "computer_use", "arguments": '
                        '{"action": "left_click", "coordinate": [1, 2]}}\n'
                        "</tool_call>"
                    ),
                }
            ],
        }


# =============================================================================
# 7. FaraHistoryProtocol sliding image window
# =============================================================================


class TestHistoryProtocol:
    def _msgs(self, n: int) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]
        for i in range(n):
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "index": i},
                        {"type": "text", "text": f"obs {i}"},
                    ],
                }
            )
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"act {i}"}]})
        return msgs

    def test_keeps_newest_n_images_and_all_text(self):
        out = FaraHistoryProtocol(max_n_images=3).process_messages(self._msgs(5))
        users = [m for m in out if m["role"] == "user"]
        img_counts = [sum(1 for p in m["content"] if p.get("type") == "image") for m in users]
        assert img_counts == [0, 0, 1, 1, 1]
        # all user turns keep their text
        assert all(any(p.get("type") == "text" for p in m["content"]) for m in users)
        # every assistant turn survives (no summarization)
        assert sum(1 for m in out if m["role"] == "assistant") == 5

    def test_zero_disables_window(self):
        msgs = self._msgs(4)
        out = FaraHistoryProtocol(max_n_images=0).process_messages(msgs)
        img_total = sum(1 for m in out for p in m.get("content", []) if p.get("type") == "image")
        assert img_total == 4

    def test_default_max_n_images(self):
        assert FaraHistoryProtocol().max_n_images == 3
