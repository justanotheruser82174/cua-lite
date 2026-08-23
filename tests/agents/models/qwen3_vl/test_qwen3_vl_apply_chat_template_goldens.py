"""Qwen3-VL ``apply_chat_template`` full-render goldens.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/models/qwen3_vl/test_qwen3_vl_apply_chat_template_goldens.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core import LiteCUAMetadata, LiteSample
from lite.core.tools import make_tool_call

register_all()

_GOLDEN_DIR = Path(__file__).parent / "_apply_chat_template_goldens"
_UPDATE = os.environ.get("UPDATE_APPLY_CHAT_TEMPLATE_GOLDENS") == "1"

FAMILIES = {
    "qwen3_vl": ("qwen3_vl@desktop@use", "Qwen/Qwen3-VL-4B-Instruct"),
}


def _img(k: int) -> Image.Image:
    """Deterministic tiny RGB image, distinct per index."""
    return Image.new("RGB", (32, 32), color=(k * 30 % 256, 0, 0))


def _computer_click(call_id: str, coordinate: list[int]) -> dict[str, Any]:
    return make_tool_call(
        "computer",
        {"actions": [{"action": "click", "coordinate": coordinate}]},
        call_id=call_id,
    )


def _traj_plain(n_turns: int = 3) -> LiteSample:
    """``plain3`` single-action desktop baseline."""
    messages: list[dict] = []
    for k in range(n_turns):
        content: list[dict] = [{"type": "image", "index": k}]
        if k == 0:
            content.append({"type": "text", "text": "Open GIMP and apply a filter."})
            messages.append({"role": "user", "content": content})
        else:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{k - 1:04d}",
                    "content": content,
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "action_description", "text": f"Action {k}: click step {k}."}],
                "tool_calls": [_computer_click(f"call_{k:04d}", [100 + k, 200 + k])],
            }
        )
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
        images=[_img(k) for k in range(n_turns)],
        messages=messages,
    )


def _traj_folded(n_turns: int = 6) -> LiteSample:
    """``folded6`` trajectory longer than the Qwen full-history window."""
    messages: list[dict] = []
    for k in range(n_turns):
        content: list[dict] = [{"type": "image", "index": k}]
        if k == 0:
            content.append({"type": "text", "text": "Open GIMP and export as PNG."})
            messages.append({"role": "user", "content": content})
        else:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{k - 1:04d}",
                    "content": content,
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "action_description", "text": f"Step {k}: click element {k}."}
                ],
                "tool_calls": [_computer_click(f"call_{k:04d}", [40 + k * 5, 60 + k * 5])],
            }
        )
    return LiteSample(
        metadata=LiteCUAMetadata(dims=("desktop", "use")),
        images=[_img(k) for k in range(n_turns)],
        messages=messages,
    )


FIXTURES = {"plain3": _traj_plain, "folded6": _traj_folded}

_CASES = [(fam, fix) for fam in FAMILIES for fix in FIXTURES]
_VL_IMAGE_CASES = [("qwen3_vl", fix) for fix in ("user_image", "tool_image")]


@functools.cache
def _load_processor(model_id: str):
    """Load a cached HF processor, or return None when it is not on disk."""
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(
            model_id, local_files_only=True, trust_remote_code=True
        )
    except Exception:
        return None


def _strip_images(step: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop image content parts before text-only chat templating."""
    out: list[dict[str, Any]] = []
    for m in step:
        content = [
            p
            for p in (m.get("content") or [])
            if not (isinstance(p, dict) and p.get("type") == "image")
        ]
        out.append({**m, "content": content})
    return out


def _render(processor, family_key: str, fixture: str) -> str:
    """The model-level byte oracle: unroll, strip images, apply chat template."""
    sample = FIXTURES[fixture]()
    steps = AgentAdapterRegistry.get(family_key).unroll(sample).steps
    msgs = _strip_images(steps[-1])
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def _image_messages(fixture: str) -> list[dict[str, Any]]:
    """Minimal VL messages that keep real image parts through templating."""
    if fixture == "user_image":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _img(1)},
                    {"type": "text", "text": "What is visible?"},
                ],
            }
        ]
    if fixture == "tool_image":
        return [
            {"role": "user", "content": [{"type": "text", "text": "Task"}]},
            {
                "role": "assistant",
                "content": (
                    "<tool_call>\n"
                    "<function=computer>\n"
                    "<parameter=action>\nclick\n</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
            },
            {
                "role": "tool",
                "content": [
                    {"type": "image", "image": _img(2)},
                    {"type": "text", "text": "screen"},
                ],
            },
        ]
    raise KeyError(fixture)


def _golden_path(family: str, fixture: str) -> Path:
    return _GOLDEN_DIR / f"{family}__{fixture}.txt"


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-VL-4B-Instruct"])
def test_qwen_template_wraps_consecutive_role_tool_results_once(model_id: str) -> None:
    processor = _load_processor(model_id)
    if processor is None:
        pytest.skip(f"{model_id} processor not cached locally")

    rendered = processor.apply_chat_template(
        [
            {"role": "user", "content": [{"type": "text", "text": "Task"}]},
            {
                "role": "assistant",
                "content": (
                    "<tool_call>\n"
                    "<function=computer_use>\n"
                    "<parameter=action>\n"
                    "left_click\n"
                    "</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
            },
            {"role": "tool", "content": [{"type": "text", "text": "obs A"}]},
            {"role": "tool", "content": [{"type": "text", "text": "obs B"}]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert rendered.count("<tool_response>") == 2
    assert rendered.count("<|im_start|>user") == 2
    assert "<|im_start|>tool" not in rendered
    assert (
        "<tool_response>\nobs A\n</tool_response>\n<tool_response>\nobs B\n</tool_response>"
    ) in rendered


@pytest.mark.parametrize("family,fixture", _CASES, ids=lambda v: str(v))
def test_apply_chat_template_golden(family: str, fixture: str) -> None:
    family_key, model_id = FAMILIES[family]
    processor = _load_processor(model_id)
    if processor is None:
        pytest.skip(f"{model_id} processor not cached locally")

    rendered = _render(processor, family_key, fixture)
    assert rendered == _render(processor, family_key, fixture), (
        f"{family}/{fixture}: apply_chat_template is non-deterministic within a run"
    )

    path = _golden_path(family, fixture)
    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"regenerated golden {path.name}")

    assert path.exists(), (
        f"missing golden {path} — regenerate with UPDATE_APPLY_CHAT_TEMPLATE_GOLDENS=1"
    )
    expected = path.read_text()
    assert rendered == expected, (
        f"APPLY_CHAT_TEMPLATE DRIFT for {family} / {fixture}:\n"
        f"the model-facing prompt string changed vs the frozen golden. If this is "
        f"an INTENTIONAL format change (Phase 2: approved GUI batch payloads / role:tool), "
        f"regenerate with UPDATE_APPLY_CHAT_TEMPLATE_GOLDENS=1 and review the diff; "
        f"otherwise the machinery refactor drifted the bytes the model sees "
        f"(must be zero)."
    )


@pytest.mark.parametrize("family,fixture", _VL_IMAGE_CASES, ids=lambda v: str(v))
def test_vl_apply_chat_template_image_golden(family: str, fixture: str) -> None:
    _, model_id = FAMILIES[family]
    processor = _load_processor(model_id)
    if processor is None:
        pytest.skip(f"{model_id} processor not cached locally")

    messages = _image_messages(fixture)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert rendered == processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    path = _golden_path(family, fixture)
    if _UPDATE:
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        pytest.skip(f"regenerated golden {path.name}")

    assert path.exists(), (
        f"missing golden {path} — regenerate with UPDATE_APPLY_CHAT_TEMPLATE_GOLDENS=1"
    )
    assert rendered == path.read_text()
