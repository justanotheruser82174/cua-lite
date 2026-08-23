"""The shared sandbox desktop dispatcher must RAISE on an action it has no
branch for — never record a silent ``noop`` and let the step screenshot + eval
an unchanged desktop.

Regression for the unknown-action fallback in
``lite/gym/sandbox/base.py:_dispatch_desktop_action``, which used to log a
warning and append ``{"call": "noop", "reason": "unknown action"}``. That
contradicted the function's own docstring ("a bad arg / KeyError here must
raise, not be silently recorded as a noop and then screenshotted + eval'd as if
the action ran"). The model-caused half of that population (a foreign-platform
action such as ``tap``) is now gated BEFORE dispatch against the catalog-
DERIVED ``_SUPPORTED_ACTIONS`` and answered with model-visible feedback, so
anything still reaching the ladder is a dispatcher bug → INFRA_FAILURE.

    uv run pytest tests/gym/sandbox/test_unknown_action_raises.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite.core.tools.action_space import LiteDesktopActionSet
from lite.core.tools.calls import make_tool_call
from lite.gym.errors import FailureCategory, TrueInfraFailure, failure_category
from lite.gym.sandbox.base import (
    _SUPPORTED_ACTIONS,
    SandboxBaseEnv,
    _dispatch_desktop_action,
)
from lite.gym.sandbox.types import SandboxTaskConfig


class _RecordingInterface:
    """Records every interface call; unknown methods are async no-ops."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_screen_size(self):
        return {"width": 100, "height": 100}

    async def screenshot(self):
        return b"shot"

    async def get_cursor_position(self):
        self.calls.append("get_cursor_position")
        return {"x": 1, "y": 2}

    def __getattr__(self, method: str):
        async def call(*_args, **_kwargs):
            self.calls.append(method)

        return call


def _sandbox_env(interface: _RecordingInterface) -> SandboxBaseEnv:
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=interface)
    env._display_resolution = (100, 100)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_step_fn = None
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t",
        instruction="do it",
        computer={},
        extra_tool_schemas=[],
    )
    return env


def test_supported_actions_is_derived_from_the_catalog():
    """The gate's legal set is DERIVED, so a new catalog action cannot drift
    into the "unsupported" bucket by omission."""
    assert _SUPPORTED_ACTIONS == LiteDesktopActionSet.get_action_names()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["bogus", "tap", "swipe", "point", "computer"])
async def test_dispatch_raises_on_unknown_action_instead_of_recording_a_noop(name):
    interface = _RecordingInterface()

    with pytest.raises(TrueInfraFailure, match=f"no branch for action {name!r}"):
        await _dispatch_desktop_action(
            SimpleNamespace(interface=interface), name, {}, 100, 100,
        )

    # nothing touched the desktop, and no recorded-noop was returned
    assert interface.calls == []


@pytest.mark.asyncio
async def test_unknown_action_raise_is_infra_not_a_model_action_error():
    """Category matters: ``step``'s ``except MODEL_ACTION_ERROR_TYPES`` would
    have turned a ValueError family member into per-call agent feedback and kept
    the episode going. A dispatcher gap is not the model's fault."""
    with pytest.raises(TrueInfraFailure) as exc_info:
        await _dispatch_desktop_action(
            SimpleNamespace(interface=_RecordingInterface()), "bogus", {}, 100, 100,
        )

    assert failure_category(exc_info.value) is FailureCategory.INFRA_FAILURE
    assert not isinstance(exc_info.value, (ValueError, TypeError, IndexError, KeyError))


@pytest.mark.asyncio
async def test_dispatch_covers_every_catalog_desktop_action():
    """The complement of the raise: no legitimate model emission can reach the
    fallthrough, because the ladder has a branch for every derived name.

    Every payload is COMPLETE: each action gets the arguments the canonical
    ``LiteDesktopActionSet`` declares with no default, since the ladder rejects
    an absent required argument rather than inventing one (see
    ``test_dispatch_action_integrity.py``). Only ``screenshot`` and
    ``cursor_position`` take none.
    """
    args_for = {
        "click": {"coordinate": [500, 500]},
        "mouse_move": {"coordinate": [500, 500]},
        "drag": {"start_coordinate": [1, 1], "coordinate": [9, 9]},
        "scroll": {"direction": "down", "amount": 1},
        "type": {"text": "hi"},
        "key": {"keys": ["a"]},
        "key_down": {"keys": ["a"]},
        "key_up": {"keys": ["a"]},
        "hold_key": {"keys": ["a"], "duration": 0},
        "wait": {"duration": 0},
    }
    for name in sorted(LiteDesktopActionSet.get_action_names()):
        calls = await _dispatch_desktop_action(
            SimpleNamespace(interface=_RecordingInterface()),
            name,
            args_for.get(name, {}),
            100,
            100,
        )
        assert not any(c.get("call") == "noop" for c in calls), name


@pytest.mark.asyncio
async def test_step_answers_a_foreign_platform_action_without_raising():
    """The model-caused half stays model-visible: ``tap`` on a desktop backend
    is answered with ``unsupported action: tap`` by the pre-dispatch derived
    gate, so the agent can correct itself — the raise is reserved for bugs.

    R2(a): ``tap`` is a GUI action, so the answer carries the current
    observation as well as the error. The model asked for something on screen;
    even though nothing landed it still needs to see the screen to choose what
    to do next.
    """
    interface = _RecordingInterface()
    env = _sandbox_env(interface)

    r = await env.step([
        make_tool_call("tap", {"coordinate": [500, 500]}, call_id="call_tap"),
    ])

    assert r.terminated is False
    assert r.results[0].error == "unsupported action: tap"
    assert r.results[0].images == [b"shot"]
    assert interface.calls == []
