"""Qwen3-VL wrapper calls that name standalone extra tools are lifted."""

from __future__ import annotations

from lite.agents.models.qwen3_vl.adapter import Qwen3VLDesktopUseAdapter
from lite.core import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet

BROWSER_EXTRA_SCHEMAS = (
    LiteBrowserNavToolSet.get_tool_schemas()
    + LiteFinishToolSet.get_tool_schemas(include=["terminate"])
)


def _qwen3_vl() -> Qwen3VLDesktopUseAdapter:
    return Qwen3VLDesktopUseAdapter(
        metadata=LiteCUAMetadata(
            dims=(LiteCUAMetadata.Platform.BROWSER, LiteCUAMetadata.TaskType.USE),
            extra_tool_schemas=list(BROWSER_EXTRA_SCHEMAS),
        ),
    )


def _json_wrapper_call(arguments: str) -> str:
    return (
        "Action: do it\n<tool_call>\n"
        f'{{"name": "computer_use", "arguments": {arguments}}}\n'
        "</tool_call>"
    )


def _lite_calls(adapter, raw_response: str) -> list[dict]:
    parsed = adapter.parse_raw_assistant_response(raw_response)
    calls = adapter.convert_message_from_agent(parsed).get("tool_calls") or []
    assert calls, f"producer dropped the call entirely: {raw_response!r}"
    return calls


def test_qwen3_vl_json_wrapper_named_goto_becomes_the_goto_tool():
    calls = _lite_calls(
        _qwen3_vl(),
        _json_wrapper_call('{"action": "goto", "url": "http://localhost:8979/"}'),
    )
    assert calls == [LiteBrowserNavToolSet.goto(url="http://localhost:8979/")]
