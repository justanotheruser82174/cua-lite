"""Characterization goldens for BrowserGym protocol render paths.

The R1 render goldens (``test_render_characterization_goldens.py``) freeze the
FULL unrolled render of every *model adapter* family, but they do NOT cover the
BrowserGym extension protocol paths: ``browsergym.generic`` and the
``visualwebarena.goal_image`` splice. Those paths reshape the conversation on
their own render path (consolidate-per-turn and goal-image prepend), so this
file freezes their rendered output end-to-end.

Phase-0 drift oracle (same contract as the R1 goldens):
- **Phase 1** (role-agnostic machinery: ``group_into_turns`` / image-helpers
  scanning ``user|tool``): re-running this file on the SAME legacy fixtures must
  be **byte-identical** — proves "machinery refactor = zero drift".
- **Phase 2** (format migration: per-call ``role:"tool"`` observations):
  regenerate; the diff is the intended change, reviewed explicitly.

Hermetic: every protocol is instantiated DIRECTLY (``BrowserGymGenericProtocol()``
and the registered ``qwen3_vl.history`` base protocol) and ``process_messages``
is pure Python over message dicts. Observations reference images by index, so
``pformat`` is deterministic (no PIL bytes / addresses).

COVERAGE NOTE — ``visualwebarena.goal_image`` is an AGENT
(``VisualWebArenaGoalImageAgent``), not a protocol, and constructing it requires
a live ``generate_fn`` + a downloaded processor (NOT hermetic). Its distinctive,
reward-load-bearing behavior is the turn-0 goal-image SPLICE its ``__post_init__``
wraps around the base protocol's ``process_messages``:

    result = base_process_messages(messages)
    result = splice_goal_images(messages, result)

We freeze exactly that composition using the production functions over a real
base protocol — so the golden covers the splice the agent performs without
building the non-hermetic agent. The full agent wrapper itself is therefore NOT
golden-frozen (needs generate_fn + processor); the persisted-image splice logic
it delegates to IS.

Regenerate goldens (ONLY after an intentional render change, review the diff):
    UPDATE_BROWSER_GOLDENS=1 env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/extensions/browsergym/test_browsergym_protocol_render_goldens.py \
        -p no:cacheprovider -q

Run (verify byte-identity):
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/extensions/browsergym/test_browsergym_protocol_render_goldens.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pformat

import pytest

# Importing extensions registers the protocols; import the classes directly.
from lite.agents.bootstrap import register_all
from lite.agents.core.agent.utils.messages import build_tool_result_message
from lite.agents.core.protocol import ProtocolRegistry
from lite.agents.extensions.browsergym.goal_image import (
    _goal_image_indices,
    splice_goal_images,
)
from lite.agents.extensions.browsergym.protocol import (
    BrowserGymGenericProtocol,
    render_tool_call_json,
)
from lite.core.tools import make_tool_call
from lite.core.tools.results import LiteToolResult

register_all()

_GOLDEN_DIR = Path(__file__).parent / "_browser_protocol_goldens"
_UPDATE = os.environ.get("UPDATE_BROWSER_GOLDENS") == "1"


# =============================================================================
# Fixtures — chat-style web trajectories (image parts referenced by index)
# =============================================================================


def _sys() -> dict:
    return {"role": "system", "content": [{"type": "text", "text": "You are a web agent."}]}


def _browsergym_traj(n_turns: int) -> list[dict]:
    """BrowserGym text+bid chat trajectory: each user turn carries a flat AXTree /
    focused-element obs-text block; each assistant turn a structured ``click(bid)``
    tool_call. ``browsergym.generic`` consolidates the whole history into ONE user
    message every turn (rebuild-per-turn)."""
    msgs: list[dict] = [_sys()]
    for k in range(n_turns):
        goal_or_obs = "Find the cheapest hat." if k == 0 else "Continue the task."
        txt = (
            f"{goal_or_obs}\n"
            f"## AXTree:\n[{k}] button 'Item {k}'\n[{k + 1}] link 'Next'\n"
            f"## Focused element:\nbid='a{k}'"
        )
        content = [{"type": "image", "index": k}, {"type": "text", "text": txt}]
        if k == 0:
            content.append({"type": "metadata", "data": {"query_id": "browsergym-0"}})
        msgs.append({"role": "user", "content": content})
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"step {k}"}],
                "tool_calls": [
                    make_tool_call("click", {"bid": f"a{k}"}, call_id=f"call_{k}"),
                ],
            }
        )
    return msgs


def _goal_image_traj() -> list[dict]:
    """VWA goal-image fixture: turn-0 carries the goal images as the persisted
    image prefix plus the ``goal_image_indices`` metadata carrier the splice
    reads (both written by ``VisualWebArenaGoalImageAgent._ingest_goal_images``).
    """
    return [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 3},
                {"type": "image", "index": 4},
                {"type": "image", "index": 0},
                {"type": "text", "text": "Buy the product matching the reference image."},
                {"type": "metadata", "data": {"goal_image_indices": [3, 4]}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "step 0"}],
            "tool_calls": [make_tool_call("click", {"coordinate": [1, 2]}, call_id="call_0")],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": [{"type": "image", "index": 1}]},
    ]


# =============================================================================
# Render functions — one per case, each returns the deterministic render string
# =============================================================================


def _render_protocol(proto, messages: list[dict]) -> str:
    out = proto.process_messages(messages)
    return pformat(out, sort_dicts=False, width=100)


def _first_text(message: dict) -> str:
    return next(c["text"] for c in message["content"] if c.get("type") == "text")


def test_browsergym_generic_renders_canonical_tool_call_envelope() -> None:
    assert render_tool_call_json(make_tool_call("click", {"bid": "a47"})) == (
        '<tool_call>\n{"name": "click", "arguments": {"bid": "a47"}}\n</tool_call>'
    )


def test_browsergym_generic_flat_tool_call_and_role_tool_observation() -> None:
    messages = [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Find the cheapest hat.\n## AXTree:\nold_body"},
                {"type": "metadata", "data": {"query_id": "browsergym-0"}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click old"}],
            "tool_calls": [
                make_tool_call("click", {"bid": "a47"}, call_id="call_0"),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [
                {"type": "image", "index": 1},
                {"type": "text", "text": "## AXTree:\nnew_body"},
            ],
        },
    ]

    out = BrowserGymGenericProtocol().process_messages(messages)
    user_msg = out[-1]
    text = _first_text(user_msg)

    assert [c for c in user_msg["content"] if c.get("type") == "image"] == [
        {"type": "image", "index": 1}
    ]
    assert [c for c in user_msg["content"] if c.get("type") == "metadata"] == [
        {"type": "metadata", "data": {"query_id": "browsergym-0"}},
    ]
    assert "## Goal:\nFind the cheapest hat." in text
    assert "new_body" in text
    assert "old_body" not in text
    assert '{"name": "click", "arguments": {"bid": "a47"}}' in text
    assert "function" not in text


def test_browsergym_generic_tool_result_error_carrier_preserves_observation_and_label() -> None:
    result = LiteToolResult(
        tool_call_id="call_0",
        images=[b"png"],
        text="## AXTree:\nnew_body\n\n## HTML:\n<button>Search</button>",
        error="unsupported action: bogus",
    )
    assert result.error not in (result.text or "")

    messages = [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Find the cheapest hat.\n## AXTree:\nold_body"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click old"}],
            "tool_calls": [
                make_tool_call("click", {"bid": "a47"}, call_id="call_0"),
            ],
        },
        build_tool_result_message(
            result.tool_call_id,
            (1,) if result.images else (),
            result.text,
            result.metadata,
            error=result.error,
        ),
    ]

    out = BrowserGymGenericProtocol().process_messages(messages)
    user_msg = out[-1]
    text = _first_text(user_msg)

    assert [c for c in user_msg["content"] if c.get("type") == "image"] == [
        {"type": "image", "index": 1}
    ]
    assert "new_body" in text
    assert "<button>Search</button>" in text
    assert "old_body" not in text
    assert "## Error from previous action:\nunsupported action: bogus" in text


def test_browsergym_generic_trims_projected_error_before_payload_sections() -> None:
    result = LiteToolResult(
        tool_call_id="call_0",
        text="## AXTree:\nnew_body",
        error="unsupported action: bogus\n## HTML:\n<button>Search</button>",
        metadata={"is_error": True},
    )

    messages = [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Find the cheapest hat.\n## AXTree:\nold_body"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click old"}],
            "tool_calls": [
                make_tool_call("click", {"bid": "a47"}, call_id="call_0"),
            ],
        },
        build_tool_result_message(
            result.tool_call_id,
            (),
            result.text,
            result.metadata,
            error=result.error,
        ),
    ]

    text = _first_text(BrowserGymGenericProtocol().process_messages(messages)[-1])

    assert "<button>Search</button>" in text
    assert "## Error from previous action:\nunsupported action: bogus\n\n# History" in text
    assert "unsupported action: bogus\n## HTML:" not in text


def test_browsergym_generic_keeps_current_payload_when_later_tool_result_is_error_only() -> None:
    current = LiteToolResult(
        tool_call_id="call_current",
        images=[b"png"],
        text="## AXTree:\nnew_body",
    )
    error_only = LiteToolResult(
        tool_call_id="call_unknown",
        error="unknown tool: foo",
        metadata={"is_error": True},
    )

    messages = [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Find the cheapest hat.\n## AXTree:\nold_body"},
            ],
        },
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call("click", {"bid": "a47"}, call_id="call_current"),
                make_tool_call("foo", {}, call_id="call_unknown"),
            ],
        },
        build_tool_result_message(
            current.tool_call_id,
            (1,) if current.images else (),
            current.text,
            current.metadata,
        ),
        build_tool_result_message(
            error_only.tool_call_id,
            (),
            error_only.text,
            error_only.metadata,
            error=error_only.error,
        ),
    ]

    out = BrowserGymGenericProtocol().process_messages(messages)
    user_msg = out[-1]
    text = _first_text(user_msg)

    assert [c for c in user_msg["content"] if c.get("type") == "image"] == [
        {"type": "image", "index": 1}
    ]
    assert "new_body" in text
    assert "old_body" not in text
    assert "## Error from previous action:\nunknown tool: foo" in text
    assert text.count("## Error from previous action:") == 1


def test_browsergym_generic_has_no_implicit_tool_surface() -> None:
    messages = [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Find the cheapest hat.\n## AXTree:\nbody"},
            ],
        },
    ]

    out = BrowserGymGenericProtocol().process_messages(
        messages,
        action_subsets=["webarena"],
        extra_tools=["goto", "response"],
    )
    text = _first_text(out[-1])

    assert "# Action space:" not in text
    assert "goto" not in text
    assert "response" not in text


def test_browsergym_generic_xml_history_uses_flat_tool_call() -> None:
    messages = [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Find the cheapest hat.\n## AXTree:\nold_body"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "fill"}],
            "tool_calls": [
                make_tool_call("fill", {"bid": "a47", "value": "hi"}, call_id="call_0"),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [
                {"type": "text", "text": "## AXTree:\nnew_body"},
            ],
        },
    ]

    out = BrowserGymGenericProtocol(tool_call_format="xml").process_messages(messages)
    text = _first_text(out[-1])

    assert "<function=fill>" in text
    assert "<parameter=bid>\na47\n</parameter>" in text
    assert "<parameter=value>\nhi\n</parameter>" in text
    assert '{"name": "fill"' not in text


def _render_goal_image_splice() -> str:
    """Reproduce ``VisualWebArenaGoalImageAgent``'s splice over the real
    ``qwen3_vl.history`` base protocol, using the production splice functions —
    the hermetic stand-in for the (non-hermetic) agent wrapper."""
    base = ProtocolRegistry.get("qwen3_vl.history")
    messages = _goal_image_traj()
    result = base.process_messages(messages)
    assert _goal_image_indices(messages)
    result = splice_goal_images(messages, result)
    return pformat(result, sort_dicts=False, width=100)


# case id → zero-arg renderer producing the frozen string.
_CASES: dict[str, callable] = {
    # browsergym.generic — multi-turn history + single-turn, AXTree obs-text.
    "browsergym_generic__multi3": lambda: _render_protocol(
        BrowserGymGenericProtocol(action_describe_text="click(bid)", use_hints=True),
        _browsergym_traj(3),
    ),
    "browsergym_generic__single1": lambda: _render_protocol(
        BrowserGymGenericProtocol(action_describe_text="click(bid)"),
        _browsergym_traj(1),
    ),
    # visualwebarena.goal_image — the turn-0 goal-image splice (production funcs).
    "goal_image__splice": _render_goal_image_splice,
}


@pytest.mark.parametrize("case_id", list(_CASES), ids=lambda v: str(v))
def test_browsergym_protocol_render_golden(case_id: str) -> None:
    rendered = _CASES[case_id]()
    # No PIL object / memory address must leak into a golden.
    assert not ("0x" in rendered and "Image" in rendered), (
        f"{case_id}: render leaks a non-deterministic object — protocols must "
        f"reference images by index only"
    )
    path = _GOLDEN_DIR / f"{case_id}.txt"

    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"regenerated golden {path.name}")

    assert path.exists(), f"missing golden {path} - regenerate with UPDATE_BROWSER_GOLDENS=1"
    assert rendered == path.read_text(), (
        f"BROWSERGYM-PROTOCOL RENDER DRIFT for {case_id}:\n"
        f"process_messages output changed vs the frozen golden. If this is an "
        f"INTENTIONAL format change (Phase 2), regenerate with UPDATE_BROWSER_GOLDENS=1 "
        f"and review the diff; otherwise the machinery refactor introduced drift "
        f"(must be zero)."
    )
