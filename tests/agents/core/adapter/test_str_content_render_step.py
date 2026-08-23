"""``render_step`` must survive a str-content user message.

``LiteUserMessage.content`` is declared ``str | list[...]``
(``lite/core/messages/content.py:62-65``), so a bare string is a legal shape.
``ui_tars`` iterated it as a list of content parts and blew up
one character in:

    ui_tars/adapter.py:377  AttributeError: 'str' object has no attribute 'get'

``instruction_text`` already read the str shape correctly, so the goal was
recoverable — only the part iteration was not. ``qwen3_vl`` was never affected
(it rebuilds the user message instead of filtering it), and it is pinned here
as the reference behaviour.

Run:
    uv run pytest tests/agents/core/adapter/test_str_content_render_step.py -v
"""

from __future__ import annotations

import dataclasses

import pytest
from lite_samples import sample_trajectory_two_turns

from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core.messages import instruction_text

GOAL = "Open GIMP."

#: The family that raised, plus one that never did (drift guard).
ADAPTER_KEYS = (
    "ui_tars@desktop@use",
    "qwen3_vl@desktop@use",
)


@pytest.fixture(autouse=True, scope="module")
def _registered():
    from lite.agents.bootstrap import register_all

    register_all()


def _sample_with_str_content(*indices: int):
    """``sample_trajectory_two_turns`` with the given user messages as str."""
    sample = sample_trajectory_two_turns()
    messages = [dict(m) for m in sample.messages]
    for i in indices:
        text = GOAL if i == 0 else "next observation"
        messages[i] = {"role": "user", "content": text}
    # Images are referenced by index; the str messages carry none.
    return dataclasses.replace(sample, messages=messages, images=[])


def _rendered_text(step) -> str:
    out = []
    for message in step:
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
            continue
        for part in content or ():
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part["text"])
    return "\n".join(out)


@pytest.mark.parametrize("key", ADAPTER_KEYS)
@pytest.mark.parametrize(
    "str_indices",
    [(0,), (2,), (0, 2)],
    ids=["first-user", "later-user", "both-users"],
)
@pytest.mark.parametrize("k", [1, 2])
class TestStrContentRenderStep:
    def test_render_step_does_not_raise(self, key, str_indices, k):
        adapter = AgentAdapterRegistry.get(key)
        adapter.render_step(_sample_with_str_content(*str_indices), k, [])

    def test_goal_reaches_the_prompt(self, key, str_indices, k):
        """A str first user message must not cost the task statement."""
        sample = _sample_with_str_content(*str_indices)
        assert instruction_text(sample.messages) == GOAL
        step = AgentAdapterRegistry.get(key).render_step(sample, k, [])
        assert GOAL in _rendered_text(step)


@pytest.mark.parametrize("key", ADAPTER_KEYS)
def test_list_content_baseline_is_unchanged(key):
    """The fix is additive: the ordinary list shape renders as before."""
    adapter = AgentAdapterRegistry.get(key)
    sample = sample_trajectory_two_turns()
    step = adapter.render_step(sample, 1, [])
    assert GOAL in _rendered_text(step)
