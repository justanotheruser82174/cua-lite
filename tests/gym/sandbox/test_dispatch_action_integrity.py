"""The "the action actually happened" contracts of the sandbox dispatcher:

* ``click(button=..., clicks>=2)`` reaches the backend WITH its button;
* ``hold_key`` releases every key it pressed on EVERY exit path;
* an argument the canonical action requires is never INVENTED — an absent
  ``keys``/``text``/``direction``/``amount``/``duration`` becomes model-visible
  feedback on the call id instead of a plausible wrong action;
* an interface-boundary failure ABORTS the rest of the batch.

    uv run pytest tests/gym/sandbox/test_dispatch_action_integrity.py
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lite.core.tools import make_tool_call
from lite.gym.errors import PairableModelActionError
from lite.gym.sandbox.base import (
    _MOUSE_BUTTONS,
    SandboxBaseEnv,
    _dispatch_desktop_action,
)
from lite.gym.sandbox.types import SandboxTaskConfig


class _CallLog:
    """Records ``(method, args, kwargs)`` for every interface call; any method
    may be armed to raise, which is what the ``_InterfaceBoundary`` proxy
    downgrades to a recorded noop."""

    def __init__(self, fail: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        #: method name -> 1-based call ordinal that must raise
        self._fail = fail or {}
        self._seen: dict[str, int] = {}

    async def get_screen_size(self):
        return {"width": 1000, "height": 1000}

    async def screenshot(self):
        return b"shot"

    def __getattr__(self, method: str):
        async def call(*args: Any, **kwargs: Any):
            self._seen[method] = self._seen.get(method, 0) + 1
            self.calls.append((method, args, kwargs))
            if self._fail.get(method) == self._seen[method]:
                raise RuntimeError(f"{method}: container died")

        return call

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


# =============================================================================
# click(button=..., clicks>=2)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["Left", "primary", "wheel", "", 1])
@pytest.mark.parametrize("action", ["click", "mouse_down", "mouse_up", "drag"])
async def test_off_enum_button_is_a_pairable_model_error(action, bad):
    """A button outside the canonical enum must raise here, not inside
    ``_InterfaceBoundary`` — a noop there aborts the batch, so one model typo
    would drop every later action in the step. ``PairableModelActionError``
    routes it to paired ``role:"tool"`` feedback instead of killing the episode.
    """
    args = {"button": bad}
    if action == "drag":
        args |= {"start_coordinate": [0, 0], "coordinate": [10, 10]}
    elif action == "click":
        args |= {"coordinate": [1, 2], "clicks": 2}

    with pytest.raises(PairableModelActionError, match="unsupported button"):
        await _dispatch_desktop_action(
            SimpleNamespace(interface=_CallLog()), action, args, 1920, 1080
        )


@pytest.mark.parametrize("spelling", ["literal", "interface"])
def test_the_button_vocabulary_has_one_owner(spelling):
    """The dispatcher, the canonical ``Literal`` and the backend's ``_BTN`` table
    must agree, or a fourth button is silently rejected by whichever copy missed.

    The ``literal`` arm reads ``LiteDesktopActionSet.click``'s type ANNOTATION, not the
    generated schema, for two reasons: ``_MOUSE_BUTTONS`` is itself derived from
    the schema (so comparing them would assert ``X == X``), and the schema's
    ``button`` enum is ``setdefault``-merged from the first child that declares
    one (``click``), so a button added only to ``drag`` would not show up there.
    """
    import typing

    from lite.core.tools.action_space import LiteDesktopActionSet
    from lite.gym.sandbox.exec_stdio.client import _BTN

    if spelling == "literal":
        other = set(typing.get_args(
            typing.get_type_hints(LiteDesktopActionSet.click)["button"]
        ))
    else:
        other = set(_BTN)
    assert other == set(_MOUSE_BUTTONS)


@pytest.mark.asyncio
@pytest.mark.parametrize("button", ["left", "right", "middle"])
@pytest.mark.parametrize("clicks", [2, 3])
async def test_multi_click_carries_the_button_to_the_backend(button, clicks):
    """If ``button`` is dropped on this branch, a right double-click executes as
    a LEFT double-click."""
    iface = _CallLog()

    calls = await _dispatch_desktop_action(
        SimpleNamespace(interface=iface),
        "click",
        {"coordinate": [500, 500], "button": button, "clicks": clicks},
        1000,
        1000,
    )

    assert iface.names() == ["multi_click"], (
        "a multi-click must be ONE repeat call so the presses stay inside the "
        "X double-click interval"
    )
    _method, _args, kwargs = iface.calls[0]
    assert kwargs["button"] == button, "the button never reached the backend"
    assert kwargs["clicks"] == clicks
    assert calls[0]["args"]["button"] == button, "the executed-action log lost the button"


@pytest.mark.asyncio
async def test_single_click_button_ladder_is_unchanged():
    """The clicks == 1 arms keep their existing per-button spelling."""
    for button, expected in (
        ("left", ["left_click"]),
        ("right", ["right_click"]),
        ("middle", ["mouse_down", "mouse_up"]),
    ):
        iface = _CallLog()
        await _dispatch_desktop_action(
            SimpleNamespace(interface=iface),
            "click",
            {"coordinate": [500, 500], "button": button},
            1000,
            1000,
        )
        assert iface.names() == expected, button


# =============================================================================
# hold_key releases what it pressed
# =============================================================================

@pytest.mark.asyncio
async def test_hold_key_releases_pressed_keys_when_a_later_press_fails():
    """A failure mid-press must still release what went down, or ctrl stays
    physically held for the rest of the episode."""
    iface = _CallLog(fail={"key_down": 2})

    calls = await _dispatch_desktop_action(
        SimpleNamespace(interface=iface),
        "hold_key",
        {"keys": ["ctrl", "shift"], "duration": 0},
        1000,
        1000,
    )

    assert iface.names() == ["key_down", "key_down", "key_up"], iface.names()
    # exactly the key that went down comes back up; the one that failed does not
    assert iface.calls[-1][1] == ("ctrl",)
    # the interface failure is still surfaced as the recorded noop
    assert calls[-1]["call"] == "noop"
    assert "container died" in calls[-1]["args"]["reason"]


@pytest.mark.asyncio
async def test_hold_key_releases_pressed_keys_on_cancellation():
    """``CancelledError`` during the hold must not leave modifiers down either —
    that is why the release lives in a ``finally``, not an ``except``."""
    import asyncio

    iface = _CallLog()

    async def _cancel(_duration):
        raise asyncio.CancelledError

    import lite.gym.sandbox.base as sandbox_base

    real_sleep = sandbox_base.asyncio.sleep
    sandbox_base.asyncio.sleep = _cancel
    try:
        with pytest.raises(asyncio.CancelledError):
            await _dispatch_desktop_action(
                SimpleNamespace(interface=iface),
                "hold_key",
                {"keys": ["ctrl", "shift"], "duration": 1},
                1000,
                1000,
            )
    finally:
        sandbox_base.asyncio.sleep = real_sleep

    assert iface.names() == ["key_down", "key_down", "key_up", "key_up"]
    assert [c[1][0] for c in iface.calls[2:]] == ["shift", "ctrl"], "release order"


@pytest.mark.asyncio
async def test_hold_key_success_path_is_unchanged():
    iface = _CallLog()
    calls = await _dispatch_desktop_action(
        SimpleNamespace(interface=iface),
        "hold_key",
        {"keys": ["ctrl", "shift"], "duration": 0},
        1000,
        1000,
    )
    assert iface.names() == ["key_down", "key_down", "key_up", "key_up"]
    assert [c["call"] for c in calls] == [
        "computer.interface.key_down",
        "computer.interface.key_down",
        "computer.interface.sleep",
        "computer.interface.key_up",
        "computer.interface.key_up",
    ]


# =============================================================================
# an empty/absent ``keys`` never counts as a keypress that happened
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("args", [{}, {"keys": []}], ids=["absent", "empty"])
@pytest.mark.parametrize("name", ["key", "key_down", "key_up", "hold_key"])
async def test_empty_keys_is_a_model_action_error_not_a_recorded_keypress(name, args):
    """``keys`` is required with no default on every canonical key action
    (``LiteDesktopActionSet.key/key_down/key_up/hold_key(keys: list[str])``), and env
    ingress checks envelope shape only -- ``prepare_env_tool_calls`` passes
    ``{"action": "key"}`` through with ``arguments == {}``. So this dispatcher is
    where an absent list is caught.

    It used to be caught nowhere: ``project_model_keys(..., allow_empty=True)``
    returned ``[]``, ``interface.hotkey()`` was awaited with no keys, and the step
    RECORDED ``computer.interface.hotkey`` as executed -- the model got a normal
    post-action screenshot for a keypress that never happened and could not tell.
    ``ValueError`` is in ``MODEL_ACTION_ERROR_TYPES``, so ``step`` turns this into
    model-visible feedback instead.
    """
    iface = _CallLog()
    payload = dict(args) | ({"duration": 0} if name == "hold_key" else {})

    with pytest.raises(ValueError, match=f"{name}.keys must not be empty"):
        await _dispatch_desktop_action(
            SimpleNamespace(interface=iface), name, payload, 1000, 1000,
        )

    assert iface.names() == [], "nothing may reach the backend for an empty key list"


# =============================================================================
# an absent REQUIRED argument never counts as an action that happened
# =============================================================================

#: ``(action, field, args-missing-that-field)`` for every canonical desktop
#: argument ``LiteDesktopActionSet`` declares with NO default. Each payload is
#: otherwise complete, so the only reason the dispatcher can reject it is the
#: named field.
_MISSING_REQUIRED_ARG_CASES = [
    ("type", "text", {}),
    ("scroll", "direction", {"amount": 3}),
    ("scroll", "amount", {"direction": "down"}),
    ("wait", "duration", {}),
    ("hold_key", "duration", {"keys": ["ctrl"]}),
]


#: The two no-default arguments the dispatcher does NOT check itself, because a
#: narrower owner already raises on absence -- ``project_model_keys`` for
#: ``keys`` (pinned by the section above) and ``norm_to_pixel(on_malformed=
#: "raise")``, reached through ``_to_pixel``, for ``coordinate``.
_REQUIRED_ARGS_WITH_THEIR_OWN_OWNER = frozenset({"keys", "coordinate"})


def test_the_required_argument_list_is_derived_from_the_canonical_action_set():
    """The cases above must stay the full set of no-default desktop arguments
    this dispatcher owns, or a newly required argument silently keeps whatever
    value the ladder invents for it.

    Derived from the canonical action set rather than hand-listed: adding a
    required argument to ``LiteDesktopActionSet`` fails HERE, at the list, and
    not later in a rollout that scrolled in a direction nobody asked for.
    """
    import inspect

    from lite.core.tools.action_space import LiteDesktopActionSet

    required: set[tuple[str, str]] = set()
    for action in sorted(LiteDesktopActionSet.get_action_names()):
        signature = inspect.signature(getattr(LiteDesktopActionSet, action))
        for field, parameter in signature.parameters.items():
            if (
                parameter.default is inspect.Parameter.empty
                and field not in _REQUIRED_ARGS_WITH_THEIR_OWN_OWNER
            ):
                required.add((action, field))

    assert required == {(action, field) for action, field, _ in _MISSING_REQUIRED_ARG_CASES}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "field", "args"),
    _MISSING_REQUIRED_ARG_CASES,
    ids=[f"{action}.{field}" for action, field, _ in _MISSING_REQUIRED_ARG_CASES],
)
async def test_absent_required_argument_is_a_model_action_error(name, field, args):
    """These five arguments are required with no default on the canonical action,
    and env ingress checks envelope shape only -- ``prepare_env_tool_calls``
    passes ``{"action": "type"}`` through with ``arguments == {}``. So this
    dispatcher is where an absent required argument is caught, the same way the
    section above catches an absent ``keys``.

    It used to be caught nowhere: the ladder read ``args.get("text", "")`` /
    ``args.get("direction", "down")`` / ``args.get("amount", 3)`` /
    ``args.get("duration", ...)``, so a malformed call typed nothing or scrolled
    down three units and the model got a normal post-action screenshot for an
    action it never asked for and could not tell apart from success.
    """
    iface = _CallLog()

    with pytest.raises(ValueError, match=f"{name}.{field} is required"):
        await _dispatch_desktop_action(
            SimpleNamespace(interface=iface), name, dict(args), 1000, 1000,
        )

    assert iface.names() == [], "nothing may reach the backend for a missing argument"


# =============================================================================
# an interface failure aborts the batch (ONE rule, both arms)
# =============================================================================

def _sandbox_env(interface: _CallLog, accepted_sink: list) -> SandboxBaseEnv:
    env = SandboxBaseEnv.__new__(SandboxBaseEnv)
    env._computer = SimpleNamespace(interface=interface)
    env._display_resolution = (1000, 1000)
    env._post_action_delay = 0.0
    env._max_steps = 10
    env._step_count = 0
    env._debug = False
    env._evaluate_final_fn = None
    env._task = SandboxTaskConfig(
        task_id="t", instruction="do it", computer={}, extra_tool_schemas=[],
    )

    async def _capture(_task, _computer, accepted, _debug):
        accepted_sink.extend(accepted)
        return None

    env._evaluate_step_fn = _capture
    return env


@pytest.mark.asyncio
async def test_interface_failure_aborts_the_rest_of_the_batch():
    """The batch is a SEQUENCE: action k+1 was chosen against the state action k
    was supposed to produce, so falling through would run the tail against a
    screen that never happened."""
    iface = _CallLog(fail={"left_click": 1})
    accepted: list = []
    env = _sandbox_env(iface, accepted)

    r = await env.step([
        make_tool_call("click", {"coordinate": [10, 10]}, call_id="c1"),
        make_tool_call("type", {"text": "never"}, call_id="c2"),
    ])

    # the tail did NOT run against the broken screen
    assert "type_text" not in iface.names(), iface.names()
    # the failed action is NOT reported to the reward function as performed
    assert accepted == []
    # both the execution error and the batch-abort notice are model-visible on
    # the failing call (a GUI batch collapses onto ONE call_id, so the abort
    # notice lands there and on any sibling call_id in the dropped tail)
    errors = {res.tool_call_id: res.error for res in r.results}
    assert set(errors) == {"c1", "c2"}
    assert "click failed" in errors["c1"]
    assert "batch aborted" in errors["c1"]
    assert "not executed" in errors["c1"]
    assert "batch aborted" in errors["c2"]
    assert "not executed" in errors["c2"]


@pytest.mark.asyncio
async def test_successful_actions_still_accumulate_and_are_accepted():
    """The abort must not swallow the healthy path: with no failure every action
    runs and every action is accepted."""
    iface = _CallLog()
    accepted: list = []
    env = _sandbox_env(iface, accepted)

    await env.step([
        make_tool_call("click", {"coordinate": [10, 10]}, call_id="c1"),
        make_tool_call("type", {"text": "hi"}, call_id="c2"),
    ])

    assert iface.names() == ["left_click", "type_text"]
    assert [a["name"] for a in accepted] == ["click", "type"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "field", "args"),
    _MISSING_REQUIRED_ARG_CASES,
    ids=[f"{action}.{field}" for action, field, _ in _MISSING_REQUIRED_ARG_CASES],
)
async def test_missing_required_argument_reaches_the_model_on_its_call_id(name, field, args):
    """The dispatcher raise must land inside ``step``'s ``except
    MODEL_ACTION_ERROR_TYPES``, not escape it.

    A ``computer`` call that comes back with NO result is not a soft failure:
    zero parsed tool calls is terminal in ``AgentBase.sample``, so a dropped
    call ends the episode instead of telling the model what was wrong. The
    result must exist, be keyed to the call id, and say which argument was
    missing -- the same channel a bad ``wait.duration`` or an empty ``keys``
    already uses.
    """
    iface = _CallLog()
    accepted: list = []
    env = _sandbox_env(iface, accepted)

    r = await env.step([
        make_tool_call(
            "computer", {"actions": [{"action": name, **args}]}, call_id="c1",
        ),
    ])

    errors = {res.tool_call_id: res.error for res in r.results}
    assert errors == {"c1": f"invalid arguments for {name}: {name}.{field} is required"}
    # the invented action never ran, and is never reported as performed
    assert iface.names() == []
    assert accepted == []
    assert not r.terminated


# =============================================================================
# one frame per executed action
# =============================================================================


class _DistinctFrames(_CallLog):
    """Every ``screenshot()`` returns different bytes, so a per-action frame list
    that repeats one cached frame is distinguishable from one that captured N
    times."""

    def __init__(self, fail: dict[str, int] | None = None) -> None:
        super().__init__(fail)
        self._shots = 0

    async def screenshot(self):
        self._shots += 1
        return f"frame-{self._shots}".encode()


@pytest.mark.asyncio
async def test_action_batch_returns_one_distinct_frame_per_executed_action():
    """N executed actions -> N frames, in action order, none a repeat of another.

    This is the reference implementation every sandbox-family env inherits, so
    the count is pinned here rather than only at the leaves. Distinct bytes are
    the point: repeating one cached frame N times satisfies the count while
    carrying no new information. The third action is ``screenshot`` -- read-only
    actions earn a frame too, so the count never depends on WHAT the actions
    were, only on how many ran.
    """
    iface = _DistinctFrames()
    env = _sandbox_env(iface, [])

    r = await env.step([
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [10, 10]},
                    {"action": "type", "text": "hi"},
                    {"action": "screenshot"},
                ],
            },
            call_id="c1",
        ),
    ])

    assert r.results[0].tool_call_id == "c1"
    assert r.results[0].images == [b"frame-1", b"frame-2", b"frame-3"]
    assert r.results[0].error is None


@pytest.mark.asyncio
async def test_an_aborted_batch_frames_only_the_actions_that_ran():
    """The count follows EXECUTED actions, not requested ones.

    ``click`` fails at the interface boundary and aborts the tail, so no action
    completed -- and the turn still owes the model exactly one observation.
    """
    iface = _DistinctFrames(fail={"left_click": 1})
    env = _sandbox_env(iface, [])

    r = await env.step([
        make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [10, 10]},
                    {"action": "type", "text": "never"},
                ],
            },
            call_id="c1",
        ),
    ])

    assert "type_text" not in iface.names(), iface.names()
    assert len(r.results[0].images) == 1
