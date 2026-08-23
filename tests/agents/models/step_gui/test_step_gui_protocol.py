"""Step-GUI history protocol prompt folding."""

from __future__ import annotations

from typing import Any

from lite.agents.models.step_gui.protocol import StepGUIHistoryProtocol
from lite.core.tools import make_tool_call


def _asst_msg_with_summary(summary: str, step_idx: int = 0) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {"type": "action_description", "text": f"step {step_idx}"},
            {"type": "history_summary", "text": summary},
        ],
        "tool_calls": [
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [100 + step_idx, 100]}]},
            )
        ],
    }


def _user_msg_with_image(image_idx: int, task: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "image", "index": image_idx}]
    if task:
        content.append({"type": "text", "text": task})
    return {"role": "user", "content": content}


class TestHistoryProtocol:
    def test_first_turn_reports_no_history(self):
        """With no prior assistant, the `已知已经执行过的历史动作如下：` line
        must show the Chinese fallback `暂无历史操作`."""
        proto = StepGUIHistoryProtocol()
        msgs = [_user_msg_with_image(0, "打开设置")]
        out = proto.process_messages(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        text_blobs = [c["text"] for c in out[0]["content"] if c.get("type") == "text"]
        full = "\n".join(text_blobs)
        assert "已知用户指令为：打开设置指令结束" in full
        assert "暂无历史操作" in full

    def test_rolling_summary_from_last_assistant(self):
        """Only the MOST RECENT assistant summary is injected — not concatenated."""
        proto = StepGUIHistoryProtocol()
        msgs = [
            _user_msg_with_image(0, "查天气"),
            _asst_msg_with_summary("已点击搜索框", step_idx=0),
            _user_msg_with_image(1),
            _asst_msg_with_summary("已输入天气", step_idx=1),
            _user_msg_with_image(2),  # current
        ]
        out = proto.process_messages(msgs)
        text_blobs = [c["text"] for c in out[0]["content"] if c.get("type") == "text"]
        full = "\n".join(text_blobs)
        assert "已知已经执行过的历史动作如下：已输入天气" in full
        assert "已点击搜索框" not in full

    def test_only_current_screenshot_kept(self):
        """All historical images are dropped; only the latest is included."""
        proto = StepGUIHistoryProtocol()
        msgs = [
            _user_msg_with_image(0, "任务"),
            _asst_msg_with_summary("s0", 0),
            _user_msg_with_image(1),
            _asst_msg_with_summary("s1", 1),
            _user_msg_with_image(99),  # current
        ]
        out = proto.process_messages(msgs)
        images = [c for c in out[0]["content"] if c.get("type") == "image"]
        assert len(images) == 1
        assert images[0]["index"] == 99

    def test_output_is_single_user_message(self):
        """History protocol always folds to one user message (plus optional system)."""
        proto = StepGUIHistoryProtocol()
        msgs = [
            _user_msg_with_image(0, "任务"),
            _asst_msg_with_summary("s0", 0),
            _user_msg_with_image(1),
            _asst_msg_with_summary("s1", 1),
            _user_msg_with_image(2),
        ]
        out = proto.process_messages(msgs)
        roles = [m["role"] for m in out]
        assert roles == ["user"]

    def test_system_message_is_preserved(self):
        proto = StepGUIHistoryProtocol()
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            _user_msg_with_image(0, "任务"),
        ]
        out = proto.process_messages(msgs)
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"

    def test_format_instruction_block_present(self):
        """The trailing Chinese format hint (<THINK>...) must be in the user text."""
        proto = StepGUIHistoryProtocol()
        msgs = [_user_msg_with_image(0, "任务")]
        out = proto.process_messages(msgs)
        text_blobs = [c["text"] for c in out[0]["content"] if c.get("type") == "text"]
        full = "\n".join(text_blobs)
        assert "<THINK>" in full
        assert "explain:" in full

    def test_current_tool_result_image_text_and_error_are_in_prompt(self):
        """Current role:tool feedback rides with the current screenshot."""
        proto = StepGUIHistoryProtocol()
        msgs = [
            _user_msg_with_image(0, "task"),
            _asst_msg_with_summary("summary_0"),
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [
                    {"type": "image", "index": 1},
                    {
                        "type": "text",
                        "text": (
                            "CURRENT_SCREEN_TEXT\n\n## Error from previous action:\n"
                            "tap failed"
                        ),
                    },
                    {"type": "metadata", "data": {"is_error": True}},
                ],
            },
            _asst_msg_with_summary("target_summary"),
        ]

        out = proto.process_messages(msgs)
        user = out[0]
        full = "\n".join(
            c.get("text", "") for c in user["content"] if c.get("type") == "text"
        )
        images = [c for c in user["content"] if c.get("type") == "image"]

        assert images == [{"type": "image", "index": 1}]
        assert "summary_0" in full
        assert "CURRENT_SCREEN_TEXT" in full
        assert "tap failed" in full


class TestProtocolWindowing:
    """protocol.process_messages structure / edge cases at boundaries, plus
    byte-exact goldens for the rolled-up user prompt."""

    # Reference output for task="打开设置", summary_history="" (first turn),
    # hints=[], user_comment="". Captured from gelab-zero
    # make_status_prompt() via direct invocation.
    _FIRST_TURN_TEXT_0 = (
        "\n已知用户指令为：打开设置指令结束\n\n\n"
        "已知已经执行过的历史动作如下：暂无历史操作\n"
        "当前手机屏幕截图如下：\n"
    )
    _FIRST_TURN_TEXT_1 = (
        "\n\n在执行操作之前，请务必回顾你的历史操作记录和限定的动作空间，"
        "先进行思考和解释然后输出动作空间和对应的参数：\n"
        "1. 思考（THINK）：在 <THINK> 和 </THINK> 标签之间。\n"
        "2. 解释（explain）：在动作格式中，使用 explain: 开头，"
        "简要说明当前动作的目的和执行方式。\n"
        "在执行完操作后，请输出执行完当前步骤后的新历史总结。\n"
        "输出格式示例：\n"
        "<THINK> 思考的内容 </THINK>\n"
        "explain:解释的内容\taction:动作空间和对应的参数\t"
        "summary:执行完当前步骤后的新历史总结\n"
    )

    def test_first_turn_text_parts_byte_exact(self):
        proto = StepGUIHistoryProtocol()
        msgs = [_user_msg_with_image(0, "打开设置")]
        out = proto.process_messages(msgs)
        texts = [c["text"] for c in out[0]["content"] if c.get("type") == "text"]
        assert len(texts) == 2
        assert texts[0] == self._FIRST_TURN_TEXT_0
        assert texts[1] == self._FIRST_TURN_TEXT_1

    def test_summary_interpolation_byte_exact(self):
        """A trajectory where the rolling summary replaces 暂无历史操作."""
        proto = StepGUIHistoryProtocol()
        msgs = [
            _user_msg_with_image(0, "打开设置"),
            _asst_msg_with_summary("已完成登录"),
            _user_msg_with_image(1),
        ]
        out = proto.process_messages(msgs)
        texts = [c["text"] for c in out[0]["content"] if c.get("type") == "text"]
        expected_0 = (
            "\n已知用户指令为：打开设置指令结束\n\n\n"
            "已知已经执行过的历史动作如下：已完成登录\n"
            "当前手机屏幕截图如下：\n"
        )
        assert texts[0] == expected_0
        assert texts[1] == self._FIRST_TURN_TEXT_1  # instructions block is invariant

    def test_empty_messages_returns_empty(self):
        proto = StepGUIHistoryProtocol()
        assert proto.process_messages([]) == []

    def test_only_system_preserved(self):
        proto = StepGUIHistoryProtocol()
        out = proto.process_messages(
            [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]
        )
        # Only the system survives; there is no turn to fold.
        assert out == [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]

    def test_assistant_without_tool_calls_falls_back_to_no_history(self):
        """If the prior assistant has no tool_calls (e.g. parse failure),
        `_extract_summary_field` cannot find a summary — fallback to '暂无'."""
        proto = StepGUIHistoryProtocol()
        broken_asst = {"role": "assistant", "content": [{"type": "text", "text": "oops"}]}
        msgs = [
            _user_msg_with_image(0, "任务"),
            broken_asst,
            _user_msg_with_image(1),
        ]
        out = proto.process_messages(msgs)
        text = "".join(
            c.get("text", "") for c in out[0]["content"] if c.get("type") == "text"
        )
        assert "已知已经执行过的历史动作如下：暂无历史操作" in text

    def test_assistant_with_empty_summary_falls_back(self):
        """Empty summary field should trigger the '暂无' fallback."""
        proto = StepGUIHistoryProtocol()
        empty_asst = _asst_msg_with_summary("")
        msgs = [
            _user_msg_with_image(0, "任务"),
            empty_asst,
            _user_msg_with_image(1),
        ]
        out = proto.process_messages(msgs)
        text = "".join(
            c.get("text", "") for c in out[0]["content"] if c.get("type") == "text"
        )
        assert "暂无历史操作" in text

    def test_older_assistants_summary_is_ignored_when_latest_has_one(self):
        """Rolling summary reads ONLY the most recent valid summary —
        historical summaries must not leak."""
        proto = StepGUIHistoryProtocol()
        msgs = [
            _user_msg_with_image(0, "任务"),
            _asst_msg_with_summary("步骤零"),
            _user_msg_with_image(1),
            _asst_msg_with_summary("步骤一"),
            _user_msg_with_image(2),
            _asst_msg_with_summary("步骤二"),
            _user_msg_with_image(3),
        ]
        out = proto.process_messages(msgs)
        text = "".join(
            c.get("text", "") for c in out[0]["content"] if c.get("type") == "text"
        )
        assert "步骤二" in text
        assert "步骤零" not in text
        assert "步骤一" not in text

    def test_protocol_does_not_mutate_input_messages(self):
        """Protocol must deep-copy; mutating the input list later must not
        affect the output."""
        proto = StepGUIHistoryProtocol()
        msgs = [
            _user_msg_with_image(0, "原始任务"),
            _asst_msg_with_summary("原始摘要"),
            _user_msg_with_image(1),
        ]
        out = proto.process_messages(msgs)
        msgs[0]["content"][1]["text"] = "被篡改"
        msgs[1]["content"][1]["text"] = "被篡改"
        text = "".join(
            c.get("text", "") for c in out[0]["content"] if c.get("type") == "text"
        )
        assert "原始任务" in text
        assert "原始摘要" in text
        assert "被篡改" not in text
