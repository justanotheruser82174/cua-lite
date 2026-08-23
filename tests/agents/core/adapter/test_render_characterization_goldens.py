"""Characterization goldens for the FULL unrolled render of every ``render_step``
adapter family.

This is the only test that freezes ``adapter.unroll(sample).steps`` end-to-end.
The per-family tests check a single action's provider wire format, and the
schema-golden tests freeze the tool *schema*; neither pins the rendered
conversation, so without these goldens render drift is invisible.

What the goldens pin, for every adapter key in :data:`NAV_ADAPTER_KEYS` crossed
with every fixture in :data:`FIXTURES`: the rendered system prompt, the per-turn
message sequence, image placement, reasoning handling, and the history-summary
fold. The fixtures are plain clicks, observation text, reasoning content, and a
long trajectory that trips the history window; all of them carry canonical
action-batch ``computer``/``mobile`` calls, so any change to protocol windowing,
``_format_action``, or step rendering surfaces as a golden diff.

Hermetic: ``AgentAdapterRegistry.get`` + ``unroll`` is pure Python (no model
download, no network). ``steps`` reference images by index, so ``pformat`` is
deterministic (no raw bytes / memory addresses).

Regenerate goldens (ONLY after an intentional render change, review the diff):
    UPDATE_RENDER_GOLDENS=1 env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/adapter/test_render_characterization_goldens.py \
        -p no:cacheprovider -q

Run (verify byte-identity):
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/adapter/test_render_characterization_goldens.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import os
from pathlib import Path
from pprint import pformat

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call

# Explicit bootstrap, per the convention in this directory's other modules.
# Without it this file only passes when some *other* test in the same xdist
# worker happens to import a model package first -- which made regenerating
# these goldens standalone impossible.
register_all()

_GOLDEN_DIR = Path(__file__).parent / "_render_goldens"
_UPDATE = os.environ.get("UPDATE_RENDER_GOLDENS") == "1"

# Every navigation adapter that uses ``render_step`` (mirrors
# ``test_unroll_contract.NAV_ADAPTER_KEYS``). Keep in sync if a family is added.
NAV_ADAPTER_KEYS = [
    "lite@desktop@use",
    "lite@mobile@use",
    "qwen3_vl@desktop@use",
    "qwen3_vl@mobile@use",
    "qwen3_5@desktop@use",
    "qwen3_5@mobile@use",
    "ui_tars@desktop@use",
    "ui_tars@mobile@use",
    "ui_tars_15_v1@desktop@use",
    "ui_tars_15_v1@mobile@use",
    "evocua@desktop@use",
    "mai_ui@mobile@use",
    "step_gui@mobile@use",
]


def _img(k: int) -> Image.Image:
    """Deterministic tiny RGB image (distinct per index, no randomness)."""
    return Image.new("RGB", (32, 32), color=(k * 30 % 256, 0, 0))


def _platform_for_adapter(adapter_key: str) -> str:
    return "mobile" if "@mobile" in adapter_key else "desktop"


def _action_call(k: int, platform: str, coordinate: list[int]) -> dict:
    if platform == "mobile":
        return make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": coordinate}]},
            call_id=f"call_{k:04d}",
        )
    return make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": coordinate}]},
        call_id=f"call_{k:04d}",
    )


def _traj_plain(n_turns: int = 3, platform: str = "desktop") -> LiteSample:
    """Multi-turn desktop trajectory: image + instruction (turn 0), one
    ``click`` per assistant turn. The core single-action baseline."""
    messages: list[dict] = []
    for k in range(n_turns):
        content: list[dict] = [{"type": "image", "index": k}]
        if k == 0:
            content.append({"type": "text", "text": "Open GIMP and apply a filter."})
        messages.append({"role": "user", "content": content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Action {k}: click step {k}."}],
            "tool_calls": [_action_call(k, platform, [100 + k, 200 + k])],
        })
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=[_img(k) for k in range(n_turns)],
        messages=messages,
    )


def _traj_obs_text(n_turns: int = 3, platform: str = "desktop") -> LiteSample:
    """Trajectory whose later user turns carry observation TEXT (WebGym-style
    feedback) — exercises the observation-text render path."""
    obs = (
        "After the action above is executed by the environment, the webpage "
        "changed (this means the last action was effective). Current URL: amazon.com"
    )
    messages: list[dict] = []
    for k in range(n_turns):
        content: list[dict] = [{"type": "image", "index": k}]
        content.append({"type": "text", "text": "Find a laptop price." if k == 0 else obs})
        messages.append({"role": "user", "content": content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Action {k}."}],
            "tool_calls": [_action_call(k, platform, [100, 100 + k])],
        })
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=[_img(k) for k in range(n_turns)],
        messages=messages,
    )


def _traj_reasoning(n_turns: int = 2, platform: str = "desktop") -> LiteSample:
    """Trajectory with ``reasoning_content`` on assistant turns — pins that the
    history protocols preserve/collapse reasoning identically."""
    messages: list[dict] = []
    for k in range(n_turns):
        content: list[dict] = [{"type": "image", "index": k}]
        if k == 0:
            content.append({"type": "text", "text": "Open Settings, go to Privacy."})
        messages.append({"role": "user", "content": content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Click step {k}."}],
            "reasoning_content": f"Reasoning for step {k}: I should click.",
            "tool_calls": [_action_call(k, platform, [50 + k, 50 + k])],
        })
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=[_img(k) for k in range(n_turns)],
        messages=messages,
    )


def _traj_folded(n_turns: int = 6, platform: str = "desktop") -> LiteSample:
    """LONG trajectory (> the qwen ``full_history_size`` default of 4) so the
    history-summary protocols (qwen3_vl/qwen3_5/…) actually FOLD:
    older turns collapse into the ``"Step N: <action>"`` previous-actions summary
    while the newest few keep their image bubbles. A 2-3 turn fixture never trips
    the window (`window_start_idx=max(0, n-4)=0`), leaving the fold path — and the
    collapsed first turn — UNGUARDED. This freezes it so changes to
    `_format_action` / `_compute_summary_and_window` surface as a golden diff."""
    messages: list[dict] = []
    for k in range(n_turns):
        content: list[dict] = [{"type": "image", "index": k}]
        if k == 0:
            content.append({"type": "text", "text": "Open GIMP and export as PNG."})
        messages.append({"role": "user", "content": content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Step {k}: click element {k}."}],
            "tool_calls": [_action_call(k, platform, [40 + k * 5, 60 + k * 5])],
        })
    return LiteSample(
        metadata=LiteCUAMetadata(dims=(platform, "use")),
        images=[_img(k) for k in range(n_turns)],
        messages=messages,
    )


FIXTURES = {
    "plain3": _traj_plain,
    "obs_text3": _traj_obs_text,
    "reasoning2": _traj_reasoning,
    "folded6": _traj_folded,   # > full_history_size → exercises the fold/summary path
}

_CASES = [(k, f) for k in NAV_ADAPTER_KEYS for f in FIXTURES]


def _golden_path(adapter_key: str, fixture: str) -> Path:
    safe = adapter_key.replace("@", "__")
    return _GOLDEN_DIR / f"{safe}__{fixture}.txt"


def _render(adapter_key: str, fixture: str) -> str:
    adapter = AgentAdapterRegistry.get(adapter_key)
    sample = FIXTURES[fixture](platform=_platform_for_adapter(adapter_key))
    agent_sample = adapter.unroll(sample)
    rendered = pformat(agent_sample.steps, sort_dicts=False, width=100)
    # Guard against non-deterministic leakage (a raw PIL object would print a
    # memory address). steps must reference images by index only.
    assert "0x" not in rendered or "Image" not in rendered, (
        f"{adapter_key}/{fixture}: render leaks a non-deterministic object "
        f"(PIL image or address) — steps must reference images by index"
    )
    return rendered


@pytest.mark.parametrize("adapter_key,fixture", _CASES, ids=lambda v: str(v))
def test_render_characterization_golden(adapter_key: str, fixture: str) -> None:
    rendered = _render(adapter_key, fixture)
    path = _golden_path(adapter_key, fixture)

    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"regenerated golden {path.name}")

    assert path.exists(), (
        f"missing golden {path} — regenerate with UPDATE_RENDER_GOLDENS=1"
    )
    expected = path.read_text()
    assert rendered == expected, (
        f"RENDER DRIFT for {adapter_key} / {fixture}:\n"
        f"the unrolled steps changed vs the frozen golden. If this is an "
        f"INTENTIONAL render change, regenerate with UPDATE_RENDER_GOLDENS=1 "
        f"and review the diff; otherwise this is unintended render drift."
    )
