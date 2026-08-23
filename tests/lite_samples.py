"""
Centralized CUA-lite samples for tests: synthetic + real (when available).

Schema follows lite/data/preproc/AGENTS.md. Synthetic samples use meta with
at least one key so they are safe to write to Parquet.

Real data: loaded from .data/huggingface/OpenGVLab/ScaleCUA-Data when present
(platform/understanding|caption, grounding/action|point|bbox, trajectory/use).
Use get_real_sample() or get_real_samples() for tests that can use real data;
fall back to sample_*() when real data is missing (e.g. CI).

Run: uv run pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.utils.path import project_root

# Root for real ScaleCUA data (relative to repo root, which is parent of tests).
_REPO_ROOT = project_root()
REAL_DATA_ROOT = _REPO_ROOT / ".data/huggingface/OpenGVLab/ScaleCUA-Data"
_FINISH_TOOL_SCHEMAS = {
    "response": LiteFinishToolSet.get_tool_schema("response"),
    "terminate": LiteFinishToolSet.get_tool_schema("terminate"),
}

def _meta(
    *,
    task_type: str,
    id: str = "x",
    platform: str = "desktop",
    resolution: list[int] | None = None,
) -> LiteCUAMetadata:
    return LiteCUAMetadata(
        dims=(platform, task_type),
        others={"id": id, "resolution": resolution or [1920, 1080], "source": "test"},
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return make_tool_call(name, arguments, call_id=call_id)


def _desktop_call(call_id: str, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return _tool_call(call_id, "computer", {"actions": [{"action": action, **arguments}]})


def _mobile_call(call_id: str, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return _tool_call(call_id, "mobile", {"actions": [{"action": action, **arguments}]})


# -----------------------------------------------------------------------------
# Understanding
# -----------------------------------------------------------------------------

def sample_understanding() -> LiteSample:
    """Understanding task: no tools, user + assistant text."""
    return LiteSample(
        images=["path/to/screen.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Describe this."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "A desktop."}]},
        ],
        metadata=_meta(id="understanding-1", task_type="understanding", platform="desktop"),
    )

# -----------------------------------------------------------------------------
# Grounding: action (desktop / mobile)
# -----------------------------------------------------------------------------

def sample_grounding_action_desktop() -> LiteSample:
    """Grounding action desktop: tool_calls with click/type/key."""
    return LiteSample(
        images=["path/to/screen.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Search for 'test'."},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    _desktop_call("call_0000", "click", {"coordinate": [500, 100]}),
                    _desktop_call("call_0001", "type", {"text": "test"}),
                ],
            },
        ],
        metadata=_meta(id="ground-action-1", task_type="grounding.action", platform="desktop"),
    )

def sample_grounding_action_mobile() -> LiteSample:
    """Grounding action mobile: tap, type."""
    return LiteSample(
        images=["path/to/screen.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Tap search."},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    _mobile_call("call_0000", "tap", {"coordinate": [500, 300]})
                ],
            },
        ],
        metadata=_meta(
            id="ground-mobile-1",
            task_type="grounding.action",
            platform="mobile",
            resolution=[1080, 1920],
        ),
    )

# -----------------------------------------------------------------------------
# Grounding: point / bbox
# -----------------------------------------------------------------------------

def sample_grounding_point() -> LiteSample:
    """Grounding point: single point tool_call."""
    return LiteSample(
        images=["path/to/screen.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Where is the button?"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("call_0000", "point", {"coordinate": [850, 120]})
                ],
            },
        ],
        metadata=_meta(id="ground-point-1", task_type="grounding.point", platform="desktop"),
    )

def sample_grounding_bbox() -> LiteSample:
    """Grounding bbox: bbox tool_call."""
    return LiteSample(
        images=["path/to/screen.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Locate the login button."},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call(
                        "call_0000",
                        "bbox",
                        {"coordinate": [380, 450, 620, 520]},
                    )
                ],
            },
        ],
        metadata=_meta(id="ground-bbox-1", task_type="grounding.bbox", platform="desktop"),
    )

# -----------------------------------------------------------------------------
# Trajectory: one turn (for parquet / path tests; unroll yields 1 sample)
# -----------------------------------------------------------------------------

def sample_trajectory_one_turn() -> LiteSample:
    """Trajectory with one turn; safe for parquet and adapter unroll tests."""
    return LiteSample(
        images=["img0.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open GIMP."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click."}],
                "tool_calls": [_desktop_call("call_0000", "click", {"coordinate": [18, 508]})],
            },
        ],
        metadata=_meta(id="traj-1", task_type="use", platform="desktop"),
    )

def sample_trajectory_two_turns() -> LiteSample:
    """Trajectory: 2 turns (user+assistant, user+assistant). For adapter unroll tests."""
    return LiteSample(
        images=["img0.png", "img1.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open GIMP."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click GIMP."}],
                "tool_calls": [_desktop_call("call_0000", "click", {"coordinate": [18, 508]})],
            },
            {"role": "user", "content": [{"type": "image", "index": 1}]},
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Press Ctrl+O."}],
                "tool_calls": [_desktop_call("call_0001", "key", {"keys": ["ctrl", "o"]})],
            },
        ],
        metadata=_meta(id="traj-2", task_type="use", platform="desktop"),
    )

def sample_trajectory_long(num_turns: int = 6) -> LiteSample:
    """
    Long trajectory (many turns) for testing full_history_size / summarization.

    Real ScaleCUA trajectories can have 2–350 messages. This builds num_turns
    turns (2*num_turns messages) so that Qwen3VLHistoryProtocol with
    full_history_size=1 vs 2 yields different message counts.
    """
    messages: list[dict[str, Any]] = []
    for i in range(num_turns):
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "index": i},
                {"type": "text", "text": "Open GIMP and apply filter." if i == 0 else ""},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Action step {i + 1}."}],
            "tool_calls": [
                _desktop_call(
                    f"call_{i:04d}",
                    "click",
                    {"coordinate": [100 + i * 10, 100]},
                )
            ],
        })
    return LiteSample(
        images=[f"img{i}.png" for i in range(num_turns)],
        messages=messages,
        metadata=_meta(id="traj-long", task_type="use", platform="desktop"),
    )

def sample_trajectory_with_reasoning() -> LiteSample:
    """
    Trajectory with reasoning_content on assistant messages (as in AGENTS.md).

    Real data may include optional reasoning_content; protocol should preserve it
    when processing/summarizing history.
    """
    return LiteSample(
        images=["img0.png", "img1.png"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open Settings and go to Privacy."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": "Click on the Settings icon."}],
                "reasoning_content": "I need to open Settings first, then navigate to Privacy.",
                "tool_calls": [_desktop_call("call_0000", "click", {"coordinate": [50, 50]})],
            },
            {"role": "user", "content": [{"type": "image", "index": 1}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "action_description",
                        "text": "Click Privacy in the sidebar.",
                    }
                ],
                "reasoning_content": (
                    "Settings is open; I can see the Privacy option in the left panel."
                ),
                "tool_calls": [_desktop_call("call_0001", "click", {"coordinate": [200, 300]})],
            },
        ],
        metadata=_meta(id="traj-reasoning", task_type="use", platform="desktop"),
    )

# -----------------------------------------------------------------------------
# Trajectory with observation text (WebGym-style)
# -----------------------------------------------------------------------------

def sample_trajectory_with_obs_text(num_turns: int = 6) -> LiteSample:
    """
    Trajectory where subsequent user messages carry observation text,
    simulating WebGym-style feedback (page changed / navigation failed).
    """
    obs_texts = [
        "After the action above is executed by the environment, the webpage changed "
        "(this means the last action was effective). The URL of the webpage after "
        "executing the action: google.com",
        "After the action above is executed by the environment, the webpage did not change "
        "(this means the last action is not effective). The URL of the webpage after "
        "executing the action: google.com",
        "Navigation failed: The website 'https://example.com' returned a blank page "
        "and is not accessible. The screenshot shows the previous page before the "
        "failed navigation. Please try navigating to a different website or use a "
        "different approach to complete the task. Current URL: google.com",
        "After the action above is executed by the environment, the webpage changed "
        "(this means the last action was effective). The URL of the webpage after "
        "executing the action: example.com",
    ]
    messages: list[dict[str, Any]] = []
    for i in range(num_turns):
        content: list[dict[str, Any]] = [{"type": "image", "index": i}]
        if i == 0:
            content.append({"type": "text", "text": "Find the price of a laptop on Amazon."})
        elif i > 0:
            content.append({"type": "text", "text": obs_texts[(i - 1) % len(obs_texts)]})
        messages.append({"role": "user", "content": content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "action_description", "text": f"Action step {i + 1}."}],
            "tool_calls": [_desktop_call(f"call_{i:04d}", "click", {"coordinate": [100, 100]})],
        })
    return LiteSample(
        images=[f"img{i}.png" for i in range(num_turns)],
        messages=messages,
        metadata=_meta(id="traj-obs-text", task_type="use", platform="browser"),
    )


def build_lite_trajectory(
    adapter: Any,
    raws: list[str],
    *,
    platform: str = "desktop",
    task_text: str = "task",
    others: dict[str, Any] | None = None,
) -> LiteSample:
    """Build a trajectory LiteSample with one turn per raw, using ``adapter``
    itself to parse each raw into a lite assistant message.

    First user turn carries an image + ``task_text``; subsequent turns carry
    only the per-step image. Used by the per-family adapter characterization
    tests (scalecua / mai_ui / ui_tars_15) to drive a multi-turn unroll from a
    list of raw model responses.

    ``others`` defaults to ``{"resolution": [1920, 1080]}`` (the desktop
    families); pass ``others={}`` for the mobile case that carries none.
    """
    messages: list[dict[str, Any]] = []
    for i, raw in enumerate(raws):
        if i == 0:
            messages.append({"role": "user", "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": task_text},
            ]})
        else:
            messages.append({"role": "user", "content": [
                {"type": "image", "index": i},
            ]})
        parsed = adapter.parse_raw_assistant_response(raw)
        lite_asst = adapter.convert_message_from_agent(parsed)
        messages.append(lite_asst)
    explicit_finish_names = {
        tool_call_name(tc)
        for msg in messages
        for tc in msg.get("tool_calls", []) or []
        if tool_call_name(tc) in _FINISH_TOOL_SCHEMAS
    }
    meta = LiteCUAMetadata(
        dims=(platform, "use"),
        extra_tool_schemas=[
            _FINISH_TOOL_SCHEMAS[name] for name in ("response", "terminate")
            if name in explicit_finish_names
        ],
        others={"resolution": [1920, 1080]} if others is None else others,
    )
    return LiteSample(
        metadata=meta,
        messages=messages,
        images=[f"img{i}.png" for i in range(len(raws))],
    )


# -----------------------------------------------------------------------------
# Minimal parquet-shape samples
# -----------------------------------------------------------------------------

def sample_grounding_action_minimal() -> LiteSample:
    """Minimal grounding.action (empty content/tool_calls). For parquet when only shape matters."""
    return LiteSample(
        images=[],
        messages=[{"role": "user", "content": []}, {"role": "assistant", "tool_calls": []}],
        metadata=_meta(id="x", task_type="grounding.action", platform="desktop"),
    )

# -----------------------------------------------------------------------------
# Real data (ScaleCUA under .data/huggingface/OpenGVLab/ScaleCUA-Data)
# -----------------------------------------------------------------------------

def _load_parquet_row(path: Path, row_index: int = 0) -> dict[str, Any] | None:
    """Load one parquet row, parsing JSON columns when present."""
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        if len(df) <= row_index:
            return None
        row = df.iloc[row_index].to_dict()
        if "messages" in row and isinstance(row["messages"], str):
            row["messages"] = json.loads(row["messages"])
        if "metadata" in row and isinstance(row["metadata"], str):
            row["metadata"] = json.loads(row["metadata"])
        return row
    except Exception:
        return None

def load_real_sample(
    platform: str = "desktop",
    task_path: str = "use/use",
    row_index: int = 0,
) -> dict[str, Any] | None:
    """
    Load one real sample from ScaleCUA parquet.

    task_path: relative to platform, e.g. "understanding/caption", "grounding/action/action",
               "grounding/point/point", "grounding/bbox/bbox", "use/use".
    Returns None if REAL_DATA_ROOT or file is missing.
    """
    path = REAL_DATA_ROOT / platform / f"{task_path}.parquet"
    return _load_parquet_row(path, row_index)

def get_real_samples(platform: str = "desktop") -> dict[str, dict[str, Any]]:
    """
    Load one real sample per task type from ScaleCUA when available.

    Returns dict with keys: understanding, grounding_action, grounding_point,
    grounding_bbox, use. Missing keys when that parquet is absent.
    """
    out: dict[str, dict[str, Any]] = {}
    if not REAL_DATA_ROOT.exists():
        return out

    for key, task_path in [
        ("understanding", "understanding/caption"),
        ("grounding_action", "grounding/action/action"),
        ("grounding_point", "grounding/point/point"),
        ("grounding_bbox", "grounding/bbox/bbox"),
        ("use", "use/use"),
    ]:
        sample = load_real_sample(platform=platform, task_path=task_path, row_index=0)
        if sample is not None:
            out[key] = sample
    return out
