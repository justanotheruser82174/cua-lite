"""Qwen3-VL history protocol render goldens."""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pformat

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.core.tools import make_tool_call

register_all()

_GOLDEN_DIR = Path(__file__).parent / "_history_protocol_goldens"
_UPDATE = os.environ.get("UPDATE_BROWSER_GOLDENS") == "1"


def _sys() -> dict:
    return {"role": "system", "content": [{"type": "text", "text": "You are a web agent."}]}


def _window_traj(n_turns: int) -> list[dict]:
    msgs: list[dict] = [_sys()]
    for k in range(n_turns):
        content: list[dict] = [{"type": "image", "index": k}]
        content.append(
            {
                "type": "text",
                "text": (
                    "Find the cheapest laptop."
                    if k == 0
                    else "After the action above, the webpage changed (this means the last "
                    "action was effective)."
                ),
            }
        )
        content.append({"type": "metadata", "data": {"page_title": f"Page {k}"}})
        msgs.append({"role": "user", "content": content})
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"click item {k}"}],
                "tool_calls": [
                    make_tool_call("click", {"coordinate": [10 + k, 20 + k]}, call_id=f"call_{k}")
                ],
            }
        )
    return msgs


def _render_protocol(messages: list[dict]) -> str:
    return pformat(
        Qwen3VLHistoryProtocol().process_messages(messages),
        sort_dicts=False,
        width=100,
    )


_CASES = {
    "qwen3_vl_history__folded6": lambda: _render_protocol(_window_traj(6)),
    "qwen3_vl_history__inwindow3": lambda: _render_protocol(_window_traj(3)),
}


@pytest.mark.parametrize("case_id", list(_CASES), ids=lambda v: str(v))
def test_qwen3_vl_history_protocol_render_golden(case_id: str) -> None:
    rendered = _CASES[case_id]()
    assert not ("0x" in rendered and "Image" in rendered)

    path = _GOLDEN_DIR / f"{case_id}.txt"
    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"regenerated golden {path.name}")

    assert path.exists(), f"missing golden {path} - regenerate with UPDATE_BROWSER_GOLDENS=1"
    assert rendered == path.read_text()
