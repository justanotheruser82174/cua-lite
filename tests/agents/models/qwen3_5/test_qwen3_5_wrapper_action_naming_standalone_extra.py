"""Qwen3.5 wrapper calls that name standalone extra tools are lifted."""

from __future__ import annotations

import pytest

from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.tools.action_space import LiteDesktopActionSet, make_lite_action_batch_call
from lite.core.tools.calls import make_tool_call
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet
from lite.data.utils.rows import validate_raw_rollout_rows

BROWSER_EXTRA_SCHEMAS = (
    LiteBrowserNavToolSet.get_tool_schemas()
    + LiteFinishToolSet.get_tool_schemas(include=["terminate"])
)


def _qwen3_5() -> Qwen3_5DesktopUseAdapter:
    return Qwen3_5DesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=list(BROWSER_EXTRA_SCHEMAS),
        ),
    )


def _xml_wrapper_call(**params: str) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"Action: do it\n<tool_call>\n<function=computer_use>\n{body}</function>\n</tool_call>"


def _lite_calls(adapter, raw_response: str) -> list[dict]:
    parsed = adapter.parse_raw_assistant_response(raw_response)
    calls = adapter.convert_message_from_agent(parsed).get("tool_calls") or []
    assert calls, f"producer dropped the call entirely: {raw_response!r}"
    return calls


def test_qwen3_5_wrapper_named_goto_becomes_the_goto_tool():
    calls = _lite_calls(
        _qwen3_5(),
        _xml_wrapper_call(action="goto", url="http://localhost:8979/"),
    )
    assert calls == [LiteBrowserNavToolSet.goto(url="http://localhost:8979/")]


def test_qwen3_5_wrapper_named_switch_tab_carries_the_declared_int():
    calls = _lite_calls(_qwen3_5(), _xml_wrapper_call(action="switch_tab", index="0"))
    assert calls == [LiteBrowserNavToolSet.switch_tab(index=0)]
    index = calls[0]["function"]["arguments"]["index"]
    assert isinstance(index, int) and not isinstance(index, bool)


def test_qwen3_5_wrapper_named_argumentless_extra_becomes_that_tool():
    calls = _lite_calls(_qwen3_5(), _xml_wrapper_call(action="back"))
    assert calls == [LiteBrowserNavToolSet.back()]


def test_native_action_value_still_becomes_an_action_batch():
    calls = _lite_calls(_qwen3_5(), _xml_wrapper_call(action="left_click", coordinate="[491, 91]"))
    assert calls == [LiteDesktopActionSet.click(coordinate=[491, 91])]


def test_hallucinated_wrapper_action_stays_an_invalid_batch_child():
    calls = _lite_calls(_qwen3_5(), _xml_wrapper_call(action="frobnicate", url="http://x/"))
    assert calls == [
        make_lite_action_batch_call(
            "computer",
            make_tool_call("frobnicate", {"url": "http://x/"}),
        )
    ]


def test_terminate_action_value_still_resolves_to_the_terminate_tool():
    calls = _lite_calls(_qwen3_5(), _xml_wrapper_call(action="terminate", status="success"))
    assert calls == [LiteFinishToolSet.terminate(status="success")]


def _row(assistant_tool_calls: list[dict]) -> dict:
    return {
        "images": ["images/0.png", "images/1.png"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open the homepage."},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [{**call, "id": "call_0000"} for call in assistant_tool_calls],
                "content": [{"type": "action_description", "text": "Navigate to the homepage"}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [
                    {"type": "image", "index": 1},
                    {"type": "text", "text": "done"},
                ],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=list(BROWSER_EXTRA_SCHEMAS),
            others={"terminated": True},
        ).to_dict(),
    }


def test_row_contract_rejects_the_shape_this_producer_used_to_emit():
    nested = make_lite_action_batch_call(
        "computer",
        make_tool_call("goto", {"url": "http://localhost:8979/"}),
    )
    with pytest.raises(
        ValueError,
        match=r"must not nest standalone extra tool 'goto'",
    ):
        validate_raw_rollout_rows([_row([nested])], "where")


def test_row_contract_accepts_what_the_producer_emits_now():
    calls = _lite_calls(
        _qwen3_5(),
        _xml_wrapper_call(action="goto", url="http://localhost:8979/"),
    )
    validate_raw_rollout_rows([_row(calls)], "where")
