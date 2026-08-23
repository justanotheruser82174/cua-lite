from __future__ import annotations

from lite.agents.models.gpt.utils.convert import lite_calls_from_gpt_computer_actions
from lite.core.tools.action_space import make_lite_action_batch_call
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name


class _DesktopActionSpace:
    def convert_tool_calls_from_agent(self, agent_tool_calls, **kwargs):
        [action] = agent_tool_calls
        assert kwargs["resolution"] == (1024, 768)

        if action["type"] == "extra":
            return [make_tool_call("response", {"text": "done"})]

        return [
            make_lite_action_batch_call(
                "computer",
                make_tool_call(action["type"], action.get("arguments", {})),
            )
        ]


def test_desktop_calls_from_agent_actions_merges_constructor_batches_without_crossing_extras():
    calls = lite_calls_from_gpt_computer_actions(
        _DesktopActionSpace(),
        [
            {"type": "click", "arguments": {"coordinate": [98, 260]}},
            {"type": "type", "arguments": {"text": "hello"}},
            {"type": "extra"},
            {"type": "wait", "arguments": {"duration": 1}},
        ],
        (1024, 768),
    )

    assert [tool_call_name(call) for call in calls] == ["computer", "response", "computer"]
    assert tool_call_arguments(calls[0])["actions"] == [
        {"action": "click", "coordinate": [98, 260]},
        {"action": "type", "text": "hello"},
    ]
    assert tool_call_arguments(calls[2])["actions"] == [
        {"action": "wait", "duration": 1},
    ]
