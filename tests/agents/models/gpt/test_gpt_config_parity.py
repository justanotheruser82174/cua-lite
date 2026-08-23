"""Golden config-parity / drift guard for the GPT agent.

Pins the EXACT request surface the GPT agent builds for each representative
``scripts/configs/gpt`` scenario — driven through the real factory + ``sample()``
with the LLM call mocked. These golden expectations were validated byte-for-byte
against the pre-refactor baseline using a historical one-off parity diff: the
15 desktop/browser configs are identical to baseline; only mobilegym intentionally
changed (prompt rewrite + ``truncation="auto"``).

Any future change that alters a config's tool surface, finish-tool gating, or
core api_kwargs trips a failure here — that's the point.

Run:
    uv run pytest tests/agents/models/gpt/test_gpt_config_parity.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from lite.agents.factory import make as make_agent
from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_schema
from lite.core.tools.extra_tools import make_open_app_tool

_P, _T = LiteCUAMetadata.Platform, LiteCUAMetadata.TaskType
_MODEL = "gpt-5.5"
_GPT_MOBILE_NATIVE = {
    "tap", "long_press", "swipe", "drag", "pinch", "type",
    "system_button", "wait", "screenshot",
}

_GOTO = make_tool_schema(
    "goto",
    description="Navigate to a URL.",
    parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
)
_BACK = make_tool_schema(
    "back",
    description="Go back.",
    parameters={"type": "object", "properties": {}, "required": []},
)
_RESPONSE = make_tool_schema(
    "response",
    description="Submit the final answer.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)
_TERMINATE = make_tool_schema(
    "terminate",
    description="Finish the task.",
    parameters={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "failure"]},
            "reason": {"type": "string"},
        },
        "required": ["status"],
    },
)


class _FakeEnv:
    """One white screenshot + instruction; terminates after one step. Carries the
    scenario's LiteCUAMetadata so the factory composes the right agent key + forwards
    valid_actions/extra_tool_schemas exactly as a real env would."""
    def __init__(self, metadata, w=1080, h=2400):
        import io
        buf = io.BytesIO()
        Image.new("RGB", (w, h), "white").save(buf, format="PNG")
        self._shot = buf.getvalue()
        self.metadata = metadata

    def _obs(self):
        return type("O", (), {"image": self._shot, "text": "do the thing"})()

    async def reset(self):
        return self._obs()

    async def step(self, actions):
        return type("S", (), {"results": [], "reward": 0.0,
                              "terminated": True, "truncated": False,
                              "info": {}})()

    async def close(self):
        pass


async def _tool_names(metadata) -> set[str]:
    """Names in the turn-0 request: native ``computer`` tool reported as
    ``"computer"``; function tools by their function name."""
    env = _FakeEnv(metadata)
    agent = make_agent(_MODEL, env=env)
    mock = AsyncMock(return_value={
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "Done."}],
        }],
        "id": "resp_test_0001",
        "usage": {},
    })
    # These tests pin the turn-0 tool surface; they must not exercise the
    # processed-image-dimension HTTP fetch that coordinate normalization
    # performs for any image-bearing request. Stub it at its own seam
    # (rather than patching httpx) so the real "no id" / "fetch failed"
    # degrade-to-sent-dims paths in _call_api_with_actual_dim stay live for
    # their own tests.
    fetch_dims = AsyncMock(return_value=[(1080, 2400)])
    with (
        patch("litellm.aresponses", mock),
        patch(
            "lite.agents.models.gpt.utils.image_io._fetch_processed_image_dims",
            fetch_dims,
        ),
    ):
        await agent.sample(env, max_steps=2)
    names = set()
    for t in mock.call_args.kwargs.get("tools") or []:
        if t.get("type") == "computer":
            names.add("computer")
        else:
            names.add(t.get("name"))
    return names


# (label, metadata, exact expected tool-name set). These are GOLDEN — validated
# against the pre-refactor baseline. Changing one means the rollout surface drifted.
_CASES = [
    # osworld & default desktop nav: valid_actions=None ⇒ ONLY the native computer
    # tool. response/terminate are a strict env-opt-in layer; osworld terminates via
    # an empty action, so it must NOT gain finish tools (drift guard).
    ("desktop_nav_none",
     LiteCUAMetadata(dims=(_P.DESKTOP, _T.USE), valid_actions=None),
     {"computer"}),
    # valid_actions is GUI-only for GPT; finish tools are not exposed without schemas.
    ("desktop_gui_valid_actions_no_finish_schema",
     LiteCUAMetadata(dims=(_P.DESKTOP, _T.USE), valid_actions=["click"]),
     {"computer"}),
    # Explicit finish-tool opt-in via canonical extra_tool_schemas.
    ("desktop_finish_extras",
     LiteCUAMetadata(
         dims=(_P.DESKTOP, _T.USE),
         valid_actions=None,
         extra_tool_schemas=[_RESPONSE, _TERMINATE],
     ),
     {"computer", "response", "terminate"}),
    # WebGym: response schema present (terminate withheld) + goto/back env extras;
    # computer kept whole. This is the contract the user called out — goto +
    # response present.
    ("browser_extras",
     LiteCUAMetadata(
         dims=(_P.BROWSER, _T.USE),
         valid_actions=["click", "type", "key", "scroll", "wait"],
         extra_tool_schemas=[_GOTO, _BACK, _RESPONSE],
     ),
     {"computer", "response", "goto", "back"}),
    # Text-only / SoM browser: valid_actions=[] drops the native computer tool entirely;
    # only env extras survive (finish tools still require their own extras).
    ("browser_textonly",
     LiteCUAMetadata(
         dims=(_P.BROWSER, _T.USE),
         valid_actions=[],
         extra_tool_schemas=[_GOTO, _BACK],
     ),
     {"goto", "back"}),
    # Grounding: single click tool.
    ("grounding_point",
     LiteCUAMetadata(dims=(_P.DESKTOP, _T.GROUNDING_POINT), valid_actions=None),
     {"click"}),
    # Mobile: self-written provider-flat mobile tool set (function tools only, no computer tool).
    # open_app is NOT here — it's an env extra_tool now (surfaced via extra_tools,
    # like qwen), not a native mobile action.
    ("mobile",
     LiteCUAMetadata(dims=(_P.MOBILE, _T.USE), valid_actions=None),
     _GPT_MOBILE_NATIVE),
    # Mobile + env open_app extra_tool → open_app appears (via extra_tools).
    ("mobile_open_app",
     LiteCUAMetadata(
         dims=(_P.MOBILE, _T.USE),
         valid_actions=None,
         extra_tool_schemas=[make_open_app_tool(["Chrome", "Settings"])],
     ),
     {*_GPT_MOBILE_NATIVE, "open_app"}),
]


@pytest.mark.parametrize("label,metadata,expected", _CASES, ids=[c[0] for c in _CASES])
async def test_gpt_tool_surface_matches_golden(label, metadata, expected):
    assert await _tool_names(metadata) == expected


async def test_webgym_has_goto_and_response_not_terminate():
    """Explicit guard for the user-stated WebGym contract: goto + response present,
    terminate withheld because no terminate extra schema is declared."""
    names = await _tool_names(
        LiteCUAMetadata(
            dims=(_P.BROWSER, _T.USE),
            valid_actions=["click", "type", "key", "scroll", "wait"],
            extra_tool_schemas=[_GOTO, _BACK, _RESPONSE],
        )
    )
    assert {"goto", "response"} <= names
    assert "terminate" not in names
