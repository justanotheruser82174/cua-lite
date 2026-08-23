"""Per-family ``role:"tool"`` render guard.

Regression coverage for a high-value render case: an OBSERVATION delivered as a
``role:"tool"`` message carrying (a) a screenshot image and (b) a text-only
result.

The existing render golden (``test_render_characterization_goldens.py``) froze
only LEGACY ``role:"user"`` fixtures, so the VERIFIED hard-drops on a
``role:"tool"`` observation had NO coverage. This file keeps a green contract
guard per family: legacy ``role:"user"`` observations remain a control, and
new ``role:"tool"`` observations must render the same screenshot/text payload.

  * **ui_tars** projects tool observations onto user messages so screenshots are
    not left on ``role:"tool"`` messages, which are out of distribution for its
    native prompt format.

Hermetic: ``AgentAdapterRegistry.get`` + ``unroll`` is pure Python (no model
download, no network). Images are tiny synthetic PIL fixtures.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/adapter/test_role_tool_render_drop_guard.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.agents.types import AgentSample
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call

register_all()

# Distinctive sentinels so a substring match cannot collide with prompt
# boilerplate (system prompts, action-space text, template scaffolding).
INSTRUCTION = "SENTINEL_INSTRUCTION_open_gimp_and_apply_a_filter"
OBS_TEXT = "SENTINEL_TOOLRESULT_the_webpage_changed_url_amazon_com"
ERROR_TEXT = "## Error from previous action:\nSENTINEL_invalid_action_mouse_move"
RAW_REPLAY_TEXT = (
    "SENTINEL_RAW_REPLAY assistant payload\n"
    '<tool_call>{"name":"computer_use","arguments":{"action":"left_click"}}</tool_call>'
)


def _img(k: int) -> Image.Image:
    """Deterministic tiny RGB image (distinct per index, no randomness)."""
    return Image.new("RGB", (32, 32), color=(k * 30 % 256, 0, 0))


def _action_call(platform: str) -> dict[str, Any]:
    wrapper = "mobile" if platform == "mobile" else "computer"
    action = "tap" if platform == "mobile" else "click"
    return make_tool_call(
        wrapper,
        {"actions": [{"action": action, "coordinate": [100, 200]}]},
        call_id="call_0",
    )


# =============================================================================
# Fixtures: the same observation content, delivered two ways
# =============================================================================

def _traj_role_tool(platform: str = "desktop") -> LiteSample:
    """A first-turn ``role:"user"`` query (text only), then an assistant tool
    call, then the OBSERVATION as a ``role:"tool"`` message carrying the
    screenshot image + a text-only result paired by ``tool_call_id``.

    The screenshot lives ONLY on the tool message, so a family that fails to
    render tool observations loses image index 0 outright — a crisp drop
    signal.
    """
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=[_img(0)],
        messages=[
            {"role": "user", "content": [{"type": "text", "text": INSTRUCTION}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click."}],
                "tool_calls": [_action_call(platform)],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": OBS_TEXT},
                    {"type": "text", "text": ERROR_TEXT},
                    {"type": "metadata", "data": {"is_error": True}},
                ],
            },
        ],
    )


def _traj_role_tool_with_raw_response(
    adapter_key: str,
    platform: str = "browser",
) -> LiteSample:
    """Same role-tool observation shape, with a matching assistant
    ``raw_response`` sidecar on the preceding action turn.

    This pins the raw replay path and the observation projection path together:
    the assistant may short-circuit to the raw bytes, but the paired tool-result
    observation that follows still has to carry its current image, text, and
    labelled error into the rendered prompt.
    """
    sample = _traj_role_tool(platform)
    assistant = sample.messages[1]
    assistant["raw_response"] = {
        "text": RAW_REPLAY_TEXT,
        "adapter_key": adapter_key,
    }
    return sample


def _traj_legacy_user(platform: str = "desktop") -> LiteSample:
    """LEGACY control: the SAME screenshot + instruction delivered the old way
    — a single ``role:"user"`` observation (image + text). This is the shape
    the existing render goldens froze; it renders correctly today, so pairing
    it with :func:`_traj_role_tool` localizes the drop to the ROLE alone."""
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=[_img(0)],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": INSTRUCTION},
                ],
            },
        ],
    )


# =============================================================================
# Rendered-step inspection helpers
# =============================================================================

def _image_indices(steps: list[list[dict[str, Any]]]) -> set[int]:
    """Every ``ImageContent.index`` referenced anywhere in the rendered steps."""
    found: set[int] = set()
    for step in steps:
        for msg in step:
            for part in msg.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "image":
                    found.add(part.get("index"))
    return found


def _all_text(steps: list[list[dict[str, Any]]]) -> str:
    """All ``text`` content across every rendered message, newline-joined.
    Covers both ``content: str`` and ``content: [ {type:text} ]`` shapes."""
    chunks: list[str] = []
    for step in steps:
        for msg in step:
            content = msg.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chunks.append(part.get("text", ""))
    return "\n".join(chunks)


def _image_in_nontool_message(steps: list[list[dict[str, Any]]], idx: int) -> bool:
    """True if image ``idx`` is referenced by a message whose role is NOT
    ``tool`` (UI-TARS carries images on user messages — a tool-borne image is
    misplaced / out-of-distribution)."""
    for step in steps:
        for msg in step:
            if msg.get("role") == "tool":
                continue
            for part in msg.get("content") or []:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "image"
                    and part.get("index") == idx
                ):
                    return True
    return False


def _unroll_bounded(adapter, sample: LiteSample, seconds: float = 5.0) -> AgentSample:
    """Run ``adapter.unroll`` under a wall-clock watchdog.

    This guard was introduced while role-tool grouping/render bugs could hang
    the suite. It remains bounded so a future role-walk regression fails cleanly
    instead of wedging the run. The daemon is left to die with the process; it
    never blocks exit.
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["out"] = adapter.unroll(sample)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the main thread
            box["err"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(
            f"unroll did not complete within {seconds}s — role:'tool' message "
            f"render/grouping regression (screenshot never rendered)"
        )
    if "err" in box:
        raise box["err"]
    return box["out"]


# =============================================================================
# browser-local voyager family: Fara / EvoCUA / Lite
# =============================================================================

@pytest.mark.parametrize(
    "adapter_key",
    [
        "fara@browser@use",
        "evocua@browser@use",
        "lite@browser@use",
    ],
)
def test_browser_local_adapters_render_role_tool_image_text_and_error(
    adapter_key: str,
) -> None:
    sample = _traj_role_tool(platform="browser")
    adapter = AgentAdapterRegistry.get(adapter_key, metadata=sample.metadata)
    out = _unroll_bounded(adapter, sample)
    assert 0 in _image_indices(out.steps), "role:tool browser image dropped"
    all_text = _all_text(out.steps)
    assert OBS_TEXT in all_text, "role:tool browser result text dropped"
    assert ERROR_TEXT in all_text, "role:tool browser labelled error dropped"


@pytest.mark.parametrize(
    "adapter_key,raw_adapter_key",
    [
        ("fara@browser@use", "fara@browser@use"),
        ("evocua@browser@use", "evocua@browser@use"),
        ("lite@browser@use", "lite@browser@use"),
    ],
)
def test_browser_local_raw_replay_does_not_drop_following_tool_result(
    adapter_key: str,
    raw_adapter_key: str,
) -> None:
    sample = _traj_role_tool_with_raw_response(raw_adapter_key, platform="browser")
    adapter = AgentAdapterRegistry.get(adapter_key, metadata=sample.metadata)
    out = _unroll_bounded(adapter, sample)
    all_text = _all_text(out.steps)
    assert RAW_REPLAY_TEXT in all_text, "matching raw_response was not replayed"
    assert 0 in _image_indices(out.steps), "raw replay path dropped tool-result image"
    assert OBS_TEXT in all_text, "raw replay path dropped tool-result text"
    assert ERROR_TEXT in all_text, "raw replay path dropped labelled tool-result error"


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        ("step_gui@mobile@use", "mobile"),
    ],
)
def test_single_current_prompt_adapters_render_role_tool_text_and_error(
    adapter_key: str,
    platform: str,
) -> None:
    adapter = AgentAdapterRegistry.get(adapter_key)
    out = _unroll_bounded(adapter, _traj_role_tool(platform))
    assert _image_in_nontool_message(out.steps, 0), (
        "role:tool screenshot dropped or misplaced (not on a user message)"
    )
    all_text = _all_text(out.steps)
    assert OBS_TEXT in all_text, "role:tool result text dropped"
    assert ERROR_TEXT in all_text, "role:tool error text dropped"


# =============================================================================
# ui_tars family
# =============================================================================

@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        ("ui_tars@desktop@use", "desktop"),
        ("ui_tars@mobile@use", "mobile"),
        ("ui_tars_15_v1@desktop@use", "desktop"),
        ("ui_tars_15_v1@mobile@use", "mobile"),
    ],
)
def test_ui_tars_legacy_user_renders_obs(adapter_key: str, platform: str) -> None:
    """GREEN (today): a LEGACY role:'user' observation's screenshot is placed on
    a proper (non-tool) user message by UI-TARS. Control half."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    out = adapter.unroll(_traj_legacy_user(platform))
    assert _image_in_nontool_message(out.steps, 0), (
        "legacy role:user screenshot not placed on a user message"
    )
    assert INSTRUCTION in _all_text(out.steps), "legacy role:user instruction dropped"


@pytest.mark.parametrize(
    "adapter_key,platform",
    [
        ("ui_tars@desktop@use", "desktop"),
        ("ui_tars@mobile@use", "mobile"),
        ("ui_tars_15_v1@desktop@use", "desktop"),
        ("ui_tars_15_v1@mobile@use", "mobile"),
    ],
)
def test_ui_tars_renders_role_tool_obs(adapter_key: str, platform: str) -> None:
    """UI-TARS projects role-tool observations onto user messages."""
    adapter = AgentAdapterRegistry.get(adapter_key)
    out = _unroll_bounded(adapter, _traj_role_tool(platform))
    assert _image_in_nontool_message(out.steps, 0), (
        "role:tool screenshot dropped or misplaced (not on a user message)"
    )
    assert OBS_TEXT in _all_text(out.steps), "role:tool result text dropped"
