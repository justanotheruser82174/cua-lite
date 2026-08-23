"""Regression tests for ``adapter.unroll`` with mixed ``raw_response``
availability across turns.

Scenario: a trajectory where only SOME assistant turns carry a matching
``raw_response`` sidecar (e.g. backfilled from a resumed rollout, or
partially edited). When the per-turn rendered messages include earlier
assistant turns, each assistant message must respect per-message
short-circuit vs canonical on a turn-by-turn basis — not uniformly opt
in or out.

Run: uv run pytest tests/agents/core/adapter/test_unroll_raw_response.py -v
"""

from __future__ import annotations

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.messages.image_refs import referenced_image_indices_in_message_order
from lite.core.tools import make_tool_call
from lite.core.tools.calls import stamp_message_tool_call_ids
from lite.train.export.sft_tokenize import agent_step_to_rl_step

register_all()


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1] * len(text.split())


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        del tokenize
        del enable_thinking
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"<{message.get('role')}>")
            for part in message.get("content") or []:
                if part.get("type") == "image":
                    chunks.append("<image>")
                elif part.get("type") == "text" and part.get("text"):
                    chunks.append(part["text"])
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "\n".join(chunks)


def _computer_click(call_id: str, coordinate: list[int]) -> dict:
    return make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": coordinate}]},
        call_id=call_id,
    )


def _assistant_with_raw(adapter, raw_text: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "canonical"}],
        "tool_calls": [_computer_click("call_0000", [10, 20])],
        "raw_response": {"text": raw_text, "adapter_key": adapter._registry_key},
    }


def _assistant_without_raw() -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "action_description", "text": "canonical_turn_1"}],
        "tool_calls": [_computer_click("call_0001", [30, 40])],
    }


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _user_with_image(text: str, image_index: int) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image", "index": image_index},
        ],
    }


def _qwen35_xml_tool_call(name: str, **arguments) -> str:
    params = "\n".join(
        f"<parameter={key}>\n{value}\n</parameter>" for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>"


def _assistant_messages(step: list[dict]) -> list[dict]:
    return [msg for msg in step if msg.get("role") == "assistant"]


def test_unroll_mixed_raw_response_per_message_selection():
    """3-turn trajectory:
      turn 0: assistant has raw_response matching (expect short-circuit)
      turn 1: assistant WITHOUT raw_response (expect canonical)
      turn 2: assistant has raw_response matching (expect short-circuit)

    ``adapter.unroll`` produces 3 cumulative steps. For each one, the
    last assistant turn's conversion must follow the raw_response presence
    of that specific message — not inherit behavior from siblings.
    """
    adapter = Qwen3VLDesktopUseAdapter()
    raw_0 = "VERBATIM_TURN_0"
    raw_2 = "VERBATIM_TURN_2"

    trajectory = LiteSample(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.TaskType.USE)
        ),
        messages=[
            _user("Task instruction"),
            _assistant_with_raw(adapter, raw_0),  # turn 0
            _user("obs 1"),
            _assistant_without_raw(),  # turn 1
            _user("obs 2"),
            _assistant_with_raw(adapter, raw_2),  # turn 2
        ],
    )

    agent_sample = adapter.unroll(trajectory)
    steps = agent_sample.steps
    assert len(steps) == 3, f"expected 3 cumulative steps, got {len(steps)}"

    def _last_assistant(msgs):
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                return m
        return None

    # Step 0: last assistant is turn-0 → short-circuit → text == raw_0
    asst_0 = _last_assistant(steps[0])
    assert asst_0["content"] == [{"type": "text", "text": raw_0}], (
        f"turn 0 should short-circuit to raw; got {asst_0}"
    )
    assert "tool_calls" not in asst_0
    assert "reasoning_content" not in asst_0

    # Step 1: last assistant is turn-1 → canonical → content NOT equal
    # to any raw text (it's canonical "Action: ..." form); tool_calls retained.
    asst_1 = _last_assistant(steps[1])
    assert asst_1 is not None
    assert asst_1["content"] != [{"type": "text", "text": raw_0}]
    assert asst_1["content"] != [{"type": "text", "text": raw_2}]
    # qwen3_vl canonical form emits a single text content containing
    # "Action: canonical_turn_1"
    assert any(
        c.get("type") == "text" and "Action" in c.get("text", "") for c in asst_1["content"]
    ), f"canonical turn-1 should contain 'Action:' in content: {asst_1}"

    # Step 2: last assistant is turn-2 → short-circuit → text == raw_2
    asst_2 = _last_assistant(steps[2])
    assert asst_2["content"] == [{"type": "text", "text": raw_2}]

    # Also check: earlier assistant turns inside step 2 still follow
    # their own per-message raw_response state. Turn 0 in step 2 should
    # also be short-circuited to raw_0 (not canonical).
    all_assistants_in_step_2 = [m for m in steps[2] if m.get("role") == "assistant"]
    assert len(all_assistants_in_step_2) == 3
    assert all_assistants_in_step_2[0]["content"] == [{"type": "text", "text": raw_0}]
    # middle assistant (turn 1) canonical:
    assert all_assistants_in_step_2[1]["content"] != [{"type": "text", "text": raw_0}]
    assert all_assistants_in_step_2[1]["content"] != [{"type": "text", "text": raw_2}]
    # last assistant (turn 2) short-circuit:
    assert all_assistants_in_step_2[2]["content"] == [{"type": "text", "text": raw_2}]


@pytest.mark.parametrize(
    "adapter_key,platform,batch_name,tool_name,click_action,coordinate,expected_actions",
    [
        (
            "qwen3_5@desktop@use",
            LiteCUAMetadata.Platform.DESKTOP,
            "computer",
            "computer_use",
            "left_click",
            [321, 654],
            [
                {"action": "click", "coordinate": [321, 654]},
                {"action": "type", "text": "done"},
            ],
        ),
        (
            "qwen3_5@mobile@use",
            LiteCUAMetadata.Platform.MOBILE,
            "mobile",
            "mobile_use",
            "click",
            [123, 456],
            [
                {"action": "tap", "coordinate": [123, 456], "clicks": 1},
                {"action": "type", "text": "done"},
            ],
        ),
    ],
)
def test_qwen35_unroll_prefers_raw_response_and_keeps_selected_images_non_none(
    adapter_key: str,
    platform: LiteCUAMetadata.Platform,
    batch_name: str,
    tool_name: str,
    click_action: str,
    coordinate: list[int],
    expected_actions: list[dict],
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    raw = (
        "Action: use two GUI actions with non-canonical XML bytes.\n"
        + _qwen35_xml_tool_call(
            tool_name,
            action=click_action,
            coordinate=f"[{coordinate[0]},  {coordinate[1]}]",
        )
        + "\n"
        + _qwen35_xml_tool_call(tool_name, action="type", text="done")
    )
    assistant = adapter.convert_message_from_agent(adapter.parse_raw_assistant_response(raw))
    assert assistant["tool_calls"] == [make_tool_call(batch_name, {"actions": expected_actions})]
    stamp_message_tool_call_ids(assistant, preserve=False)
    assistant["raw_response"] = {
        "text": raw,
        "adapter_key": adapter._registry_key,
    }

    sample = LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, LiteCUAMetadata.TaskType.USE)),
        images=[
            Image.new("RGB", (64, 64), "white"),
            Image.new("RGB", (64, 64), "red"),
            Image.new("RGB", (64, 64), "blue"),
        ],
        messages=[
            _user_with_image("Task instruction", 0),
            assistant,
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [
                    {"type": "image", "index": 2},
                    {"type": "text", "text": "screen changed"},
                ],
            },
        ],
    )

    agent_sample = adapter.unroll(sample)

    assert len(agent_sample.steps) == 2
    rl_step = agent_step_to_rl_step(agent_sample.steps[0], _FakeProcessor())
    assert rl_step is not None
    assert rl_step.image_indices == (0,)
    assert rl_step.response == f"\n{raw}"
    assert f"[{coordinate[0]},  {coordinate[1]}]" in rl_step.response
    assert (
        "<function=computer_use>" in rl_step.response or "<function=mobile_use>" in rl_step.response
    )

    for step in agent_sample.steps:
        [rendered_assistant] = _assistant_messages(step)
        assert rendered_assistant["content"] == [{"type": "text", "text": raw}]
        assert "tool_calls" not in rendered_assistant

    assert len(agent_sample.processed_images) == 3
    assert agent_sample.processed_images[0] is not None
    assert agent_sample.processed_images[1] is None
    assert agent_sample.processed_images[2] is not None
    assert referenced_image_indices_in_message_order(agent_sample.steps[0]) == (0,)
    assert referenced_image_indices_in_message_order(agent_sample.steps[1]) == (0, 2)
    selected = {
        idx
        for step in agent_sample.steps
        for idx in referenced_image_indices_in_message_order(step)
    }
    assert selected == {0, 2}
    assert all(agent_sample.processed_images[idx] is not None for idx in selected)
