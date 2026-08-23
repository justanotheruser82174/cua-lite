"""Unit tests for cua.bench pure helpers + graceful degradation — no cua-bench
install needed. Live end-to-end coverage is in test_cua_bench_live.py.

    uv run pytest tests/gym/envs/cua/bench/test_cua_bench.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

import lite.gym as gym
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.gym.envs.cua.bench.translate import normalize_reward, to_cb_action
from lite.gym.errors import (
    EnvDepsMissingError,
    FailureCategory,
    TrueInfraFailure,
    failure_category,
)

_HAVE_CB = importlib.util.find_spec("cua_bench") is not None

_CUA_BENCH_RELEASE_AGENT_EXTRA_TOOLS = [
    pytest.param("gpt", [], id="gpt-release-extra-tools-empty"),
    pytest.param("qwen3_5", ["terminate"], id="qwen3_5-release-extra-tools-terminate"),
    pytest.param("qwen3_vl", ["terminate"], id="qwen3_vl-release-extra-tools-terminate"),
]


# --- normalize_reward (mirrors runners.py:139-145) --------------------------
@pytest.mark.parametrize(
    "result,expected",
    [
        (0.7, 0.7),
        (1, 1.0),
        (True, 1.0),  # bool ⊂ int → 1.0
        (False, 0.0),
        ([0.9, 0.1], 0.9),  # list → first
        ({"reward": 0.3}, 0.3),
        ({"success": True}, None),  # dict WITHOUT "reward" key → None (reward key, not success)
        ([], None),
        ("nope", None),
        (None, None),
    ],
)
def test_normalize_reward(result, expected):
    assert normalize_reward(result) == expected


# --- to_cb_action against a fake cua_bench Action namespace ------------------
class _Rec:
    def __init__(self, name):
        self.name = name

    def __call__(self, **kw):
        return (self.name, kw)


class _A:
    ClickAction = _Rec("Click")
    DoubleClickAction = _Rec("Double")
    RightClickAction = _Rec("Right")
    MiddleClickAction = _Rec("Middle")
    TypeAction = _Rec("Type")
    KeyAction = _Rec("Key")
    HotkeyAction = _Rec("Hotkey")
    MoveToAction = _Rec("MoveTo")
    DragAction = _Rec("Drag")
    ScrollAction = _Rec("Scroll")
    WaitAction = _Rec("Wait")
    DoneAction = _Rec("Done")


def _act(name, **args):
    return {"name": name, "arguments": args}


def _action_call(*actions, call_id: str | None = None):
    return make_tool_call(
        "computer",
        {
            "actions": [{"action": p["name"], **p["arguments"]} for p in actions],
        },
        call_id=call_id,
    )


def _assert_current_image_result(result, *, call_id: str, image: bytes) -> None:
    assert result.results[0].tool_call_id == call_id
    assert result.results[0].images[-1] == image
    assert result.results[0].text is None


def _assert_error_only_result(result, *, call_id: str, error: str) -> None:
    assert result.results[0].tool_call_id == call_id
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == error
    assert result.results[0].metadata == {"is_error": True}


def _assert_cursor_at(png: bytes, x: int, y: int) -> None:
    import io

    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    box = [
        img.getpixel((px, py)) for px in range(x, min(x + 16, w)) for py in range(y, min(y + 24, h))
    ]
    assert any(pixel != (255, 255, 255) for pixel in box)
    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_to_cb_action_click_maps_to_pixels():
    action, done = to_cb_action(_act("click", coordinate=[500, 250]), (1000, 1000), _A)
    assert action == ("Click", {"x": 500, "y": 250}) and done is False  # ClickAction has no button


def test_to_cb_action_click_variants():
    assert (
        to_cb_action(_act("click", coordinate=[10, 10], clicks=2), (1000, 1000), _A)[0][0]
        == "Double"
    )
    assert (
        to_cb_action(_act("click", coordinate=[10, 10], button="right"), (1000, 1000), _A)[0][0]
        == "Right"
    )
    assert (
        to_cb_action(_act("click", coordinate=[10, 10], button="middle"), (1000, 1000), _A)[0][0]
        == "Middle"
    )


def test_to_cb_action_keys_single_vs_combo():
    assert to_cb_action(_act("key", keys=["enter"]), (1000, 1000), _A)[0] == (
        "Key",
        {"key": "enter"},
    )
    assert to_cb_action(_act("key", keys=["ctrl", "c"]), (1000, 1000), _A)[0] == (
        "Hotkey",
        {"keys": ["ctrl", "c"]},
    )


def test_to_cb_action_keys_projected_per_backend():
    # default backend is pynput (native/local, the shipped datasets): underscores + cmd.
    assert to_cb_action(_act("key", keys=["pageup"]), (1000, 1000), _A)[0] == (
        "Key",
        {"key": "page_up"},
    )
    assert to_cb_action(_act("key", keys=["meta", "-"]), (1000, 1000), _A)[0] == (
        "Hotkey",
        {"keys": ["cmd", "-"]},
    )
    assert to_cb_action(_act("key", keys=["meta", "+"]), (1000, 1000), _A)[0] == (
        "Hotkey",
        {"keys": ["cmd", "+"]},
    )
    assert to_cb_action(_act("key", keys=["meta", "="]), (1000, 1000), _A)[0] == (
        "Hotkey",
        {"keys": ["cmd", "="]},
    )
    with pytest.raises(ValueError, match="unknown key token 'minus'"):
        to_cb_action(_act("key", keys=["meta", "minus"]), (1000, 1000), _A)
    # webtop consumes Playwright spelling.
    assert to_cb_action(_act("key", keys=["pageup"]), (1000, 1000), _A, "playwright")[0] == (
        "Key",
        {"key": "PageUp"},
    )


def test_to_cb_action_move_drag_scroll_type():
    assert to_cb_action(_act("mouse_move", coordinate=[500, 500]), (1000, 1000), _A)[0] == (
        "MoveTo",
        {"x": 500, "y": 500},
    )
    assert to_cb_action(
        _act("drag", start_coordinate=[0, 0], coordinate=[1000, 1000]), (1000, 1000), _A
    )[0] == ("Drag", {"from_x": 0, "from_y": 0, "to_x": 1000, "to_y": 1000})
    assert to_cb_action(
        _act("drag", coordinate=[1000, 1000]), (1000, 1000), _A, cursor_px=(123, 456)
    )[0] == ("Drag", {"from_x": 123, "from_y": 456, "to_x": 1000, "to_y": 1000})
    with pytest.raises(ValueError, match="drag.coordinate is required"):
        to_cb_action(_act("drag"), (1000, 1000), _A)
    # Lite `amount` is in scroll clicks; translate scales clicks→cua px-units (×100).
    assert to_cb_action(_act("scroll", direction="down", amount=3), (1000, 1000), _A)[0] == (
        "Scroll",
        {"direction": "down", "amount": 300},
    )
    # cua-bench ScrollAction is vertical-only → left/right is rejected instead of wrong-axis.
    with pytest.raises(ValueError, match="not supported by cua.bench"):
        to_cb_action(_act("scroll", direction="left", amount=3), (1000, 1000), _A)
    assert to_cb_action(_act("type", text="hi"), (1000, 1000), _A)[0] == ("Type", {"text": "hi"})


@pytest.mark.parametrize("coordinate", [None, [], [1], "abc", [None, None]])
def test_to_cb_action_malformed_coordinate_raises_model_error(coordinate):
    with pytest.raises((TypeError, ValueError)):
        to_cb_action(_act("click", coordinate=coordinate), (1000, 1000), _A)


def test_to_cb_action_terminate_is_done():
    action, done = to_cb_action(_act("terminate", status="success"), (1000, 1000), _A)
    assert action == ("Done", {}) and done is True


def test_to_cb_action_wait_rejects_bad_duration():
    with pytest.raises(ValueError, match="wait.duration must be <= 30"):
        to_cb_action(_act("wait", duration=31), (1000, 1000), _A)


@pytest.mark.parametrize(
    "name", ["screenshot", "cursor_position", "key_down", "mouse_down", "hold_key"]
)
def test_to_cb_action_unsupported_raises_model_visible_error(name):
    with pytest.raises(ValueError, match=f"unsupported action: {name}"):
        to_cb_action(_act(name), (1000, 1000), _A)


# --- _png_wh: coordinate space derived from the screenshot's own pixel size --
def _png(w: int, h: int) -> bytes:
    """Real PNG with a valid signature + IHDR width/height."""
    import io

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_png_wh_reads_ihdr(fake_cb):  # fake_cb lets the module import without cua-bench
    from lite.gym.envs.cua.bench.main import _png_wh

    assert _png_wh(_png(800, 600)) == (800, 600)
    assert _png_wh(b"not a png") is None  # non-PNG → None (caller falls back)


# --- max_steps truncation (against a fake cua_bench) ------------------------
class _FakeCbEnv:
    def __init__(self):
        self.tasks = None
        self.steps = []
        self.create_sandbox_calls = []
        self.evaluate_task_fn = lambda *a: 1.0  # present → wrapper's evaluate() guard passes
        # real cua_bench.Environment screenshots via .session.screenshot(); model it so the
        # wrapper's post-action settle re-grab (bench/main.py step) has a session to call.
        # execute_action is on the same DesktopSession protocol — reset() uses it to
        # PARK the real pointer at screen centre before the first capture.
        self.session_actions = []
        self.session = types.SimpleNamespace(
            screenshot=self._screenshot,
            execute_action=self._execute_action,
        )

    async def _execute_action(self, action):
        self.session_actions.append(action)

    async def _screenshot(self):
        return _png(800, 600)

    def tasks_config_fn(self):
        # a task that self-declares a computer — the webtop base leaves it untouched,
        # while CuaBenchLocalEnv overwrites it with provider="native" + our image.
        return [
            types.SimpleNamespace(
                description="do it", computer={"provider": "native", "setup_config": {}}
            )
        ]

    async def create_sandbox(self, *args, **kwargs):
        self.create_sandbox_calls.append((args, kwargs))

    async def reset(self, task_id):
        if self.tasks is None:  # mimic cua-bench: populate from config on first reset
            self.tasks = self.tasks_config_fn()
        # 800x600 PNG → _wh should be derived from it, not the 1024x768 default
        return (_png(800, 600), self.tasks[task_id])

    async def step(self, action):
        self.steps.append(action)
        return _png(800, 600)

    async def evaluate(self):
        return 1.0

    async def close(self):
        pass


@pytest.fixture
def fake_cb(monkeypatch):
    mod = types.ModuleType("cua_bench")
    for name in (
        "ClickAction",
        "DoubleClickAction",
        "RightClickAction",
        "MiddleClickAction",
        "TypeAction",
        "KeyAction",
        "HotkeyAction",
        "MoveToAction",
        "DragAction",
        "ScrollAction",
        "WaitAction",
        "DoneAction",
    ):
        setattr(mod, name, _Rec(name))
    mod.make = lambda path: _FakeCbEnv()
    monkeypatch.setitem(sys.modules, "cua_bench", mod)
    # force cua.bench.main to (re)import under the fake, and drop it afterwards so
    # the graceful-degradation tests re-import cleanly without cua-bench.
    monkeypatch.delitem(sys.modules, "lite.gym.envs.cua.bench.main", raising=False)
    yield mod
    sys.modules.pop("lite.gym.envs.cua.bench.main", None)


async def test_bench_max_steps_truncates(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=2, post_action_delay=0)  # no settle sleep in tests
    obs = await env.reset()
    assert obs.text == "do it"
    assert env._wh == (800, 600)  # coordinate space derived from the screenshot, not the default
    r1 = await env.step([_action_call(_act("wait", duration=0))])
    assert not r1.truncated and not r1.terminated  # step 1 of 2
    r2 = await env.step([_action_call(_act("wait", duration=0), call_id="wait1")])
    assert r2.truncated and not r2.terminated  # hit max_steps → truncated
    assert r2.results[0].tool_call_id == "wait1"
    assert r2.results[0].images[-1] == env._last_png
    assert r2.results[0].text is None
    assert r2.results[0].error is None


async def test_bench_action_batch_returns_one_distinct_frame_per_action(fake_cb):
    """N executed actions → N frames, in action order, none a repeat of another.

    cua-bench's ``step()`` returns the frame it grabbed right after the action,
    so the per-action record costs no extra round-trip. Distinct bytes are the
    point: repeating one cached frame N times would satisfy the count while
    carrying no new information.
    """
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0, cursor=False)
    await env.reset()
    frames = iter([b"frame-1", b"frame-2"])

    async def fake_step(action):
        env._cb.steps.append(action)
        return next(frames)

    env._cb.step = fake_step

    r = await env.step(
        [
            _action_call(
                _act("click", coordinate=[100, 100]),
                _act("type", text="hi"),
                call_id="call_batch",
            ),
        ]
    )

    assert r.results[0].tool_call_id == "call_batch"
    assert r.results[0].images == [b"frame-1", b"frame-2"]
    assert r.results[0].error is None


async def test_bench_a_rejected_child_earns_a_frame_and_costs_no_sibling(fake_cb):
    """A slot the model got wrong owes a frame, and does not abort the batch.

    ``bogus`` is not a name the batch tool carries. Ingress rejects it and
    forwards it anyway, so this env answers it per slot: one frame repeating the
    screen it did not change, with a model-visible reason. The valid sibling
    still reaches the guest -- a model fault costs the model its own slot, never
    the batch. cua-bench has no ``invalid_action_message`` call of its own, so
    this also pins that it reads the shared ingress decision.
    """
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0, cursor=False)
    await env.reset()
    frames = iter([b"frame-1", b"frame-2"])

    async def fake_step(action):
        env._cb.steps.append(action)
        return next(frames)

    env._cb.step = fake_step

    r = await env.step(
        [
            _action_call(
                _act("click", coordinate=[100, 100]),
                _act("bogus"),
                call_id="call_batch",
            ),
        ]
    )

    # the sibling still ran, and the rejected slot never reached the guest
    assert len(env._cb.steps) == 1
    assert len(r.results[0].images) == 2
    assert "bogus" in r.results[0].error


async def test_bench_accepts_seed_but_rejects_typos(fake_cb):
    # factory/rollout.py forwards `seed` to every env → must accept it (regression:
    # CLI rollout was broken on `seed`). But unlike **kwargs, a typo'd env_kwarg must
    # still raise — matching browsergym's strict-typo behavior (seed accepted explicitly).
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, seed=42)
    assert env.metadata.platform.value == "desktop"
    with pytest.raises(TypeError):
        CuaBenchEnv("some/env", 0, max_steps=3, max_step=3)  # typo → not silently swallowed


async def test_bench_webtop_base_does_not_inject_provider(fake_cb):
    # the base env IS the in-process webtop mode (provider stays None) → it must NOT
    # touch the task's own computer config (only CuaBenchLocalEnv injects; see below).
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3)
    await env.reset()
    assert env._cb.tasks[0].computer == {"provider": "native", "setup_config": {}}  # untouched


async def test_bench_empty_actions_still_returns_screenshot(fake_cb):
    # a step with no actions must still emit a screenshot (fall back to the last one),
    # else the agent's post-step obs guard raises. Mirrors cua.sandbox.
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3)
    await env.reset()
    r = await env.step([])  # no actions
    assert r.results == [] and not r.terminated


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_BENCH_RELEASE_AGENT_EXTRA_TOOLS)
async def test_bench_t2_release_rows_active_known_tool(fake_cb, agent_family, extra_tools):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, extra_tools=extra_tools, post_action_delay=0)
    await env.reset()

    call_id = f"{agent_family}_active_known_tool"
    r = await env.step([_action_call(_act("wait", duration=0), call_id=call_id)])

    assert env._cb.steps == [("WaitAction", {"seconds": 0.0})]
    assert not r.terminated and not r.truncated
    _assert_current_image_result(r, call_id=call_id, image=env._last_png)
    assert r.results[0].error is None
    assert r.results[0].metadata is None


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_BENCH_RELEASE_AGENT_EXTRA_TOOLS)
async def test_bench_t2_release_rows_malformed_known_action(fake_cb, agent_family, extra_tools):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, extra_tools=extra_tools, post_action_delay=0)
    await env.reset()

    call_id = f"{agent_family}_malformed_known_action"
    r = await env.step(
        [
            _action_call(_act("click", coordinate=[None, None]), call_id=call_id),
        ]
    )

    assert env._cb.steps == []
    assert not r.terminated and not r.truncated
    _assert_current_image_result(r, call_id=call_id, image=env._last_png)
    assert r.results[0].error == (
        "invalid arguments for click: coordinate values must be finite numbers"
    )
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_BENCH_RELEASE_AGENT_EXTRA_TOOLS)
async def test_bench_t2_release_rows_literal_unknown_tool(fake_cb, agent_family, extra_tools):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, extra_tools=extra_tools, post_action_delay=0)
    await env.reset()

    call_id = f"{agent_family}_literal_unknown_tool"
    r = await env.step(
        [
            make_tool_call("foo", {}, call_id=call_id),
        ]
    )

    assert env._cb.steps == []
    assert not r.terminated and not r.truncated
    _assert_error_only_result(r, call_id=call_id, error="unknown tool: foo")


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_BENCH_RELEASE_AGENT_EXTRA_TOOLS)
async def test_bench_t2_release_rows_content_only_final_text(fake_cb, agent_family, extra_tools):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, extra_tools=extra_tools, post_action_delay=0)
    await env.reset()

    r = await env.step(make_no_tool_call_final_actions(f"{agent_family} final text"))

    assert env._cb.steps == [("DoneAction", {})]
    assert r.terminated and not r.truncated
    assert r.reward == 1.0
    assert r.results == []


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_BENCH_RELEASE_AGENT_EXTRA_TOOLS)
async def test_bench_t2_release_rows_image_data_binding(fake_cb, agent_family, extra_tools):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, extra_tools=extra_tools, post_action_delay=0)
    reset_obs = await env.reset()
    bound_png = _png(640, 480) + b"post-click-bound-frame"

    async def step_returns_bound_frame(action):
        env._cb.steps.append(action)
        return bound_png

    env._cb.step = step_returns_bound_frame
    call_id = f"{agent_family}_image_data_binding"
    r = await env.step(
        [
            _action_call(_act("click", coordinate=[500, 500]), call_id=call_id),
        ]
    )

    assert env._cb.steps == [("ClickAction", {"x": 400, "y": 300})]
    assert bound_png != reset_obs.image
    assert env._last_png != bound_png
    assert not r.terminated and not r.truncated
    _assert_current_image_result(r, call_id=call_id, image=env._last_png)
    assert r.results[0].error is None
    assert r.results[0].metadata is None


async def test_bench_cursor_is_capture_owned_and_tracks_dispatched_cursor(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    raw_png = _png(800, 600)
    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0)
    reset_obs = await env.reset()
    assert env._cursor is True
    assert reset_obs.image == env._last_png
    assert reset_obs.image != raw_png
    # The reset centre is ESTABLISHED, not assumed: a real MoveToAction is
    # executed on the session before the frame is captured, so the composited
    # coordinate is where the pointer actually is.
    assert env._cb.session_actions == [("MoveToAction", {"x": 400, "y": 300})]
    assert env._cursor_px_known is True

    # PIXEL-level: the reset frame carries the arrow at the CENTRE, and the
    # origin — the "never moved" sentinel — is untouched.
    import io as _io

    reset_img = Image.open(_io.BytesIO(reset_obs.image)).convert("RGB")
    assert any(
        reset_img.getpixel((x, y)) != (255, 255, 255)
        for x in range(400, 416)
        for y in range(300, 324)
    )
    assert all(reset_img.getpixel((x, y)) == (255, 255, 255) for x in range(16) for y in range(24))

    async def step_returns_raw_frame(action):
        env._cb.steps.append(action)
        return raw_png

    env._cb.step = step_returns_raw_frame
    result = await env.step(
        [
            _action_call(_act("click", coordinate=[980, 960]), call_id="click0"),
        ]
    )

    assert env._cursor_px == (784, 576)
    assert result.results[0].images[-1] == env._last_png
    assert result.results[0].images[-1] != raw_png

    import io

    img = Image.open(io.BytesIO(result.results[0].images[-1])).convert("RGB")
    box = [img.getpixel((x, y)) for x in range(784, 800) for y in range(576, 600)]
    assert any(px != (255, 255, 255) for px in box)
    assert img.getpixel((0, 0)) == (255, 255, 255)


async def test_bench_scroll_coordinate_updates_returned_cursor(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    raw_png = _png(800, 600)
    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0)
    await env.reset()

    async def step_returns_raw_frame(action):
        env._cb.steps.append(action)
        return raw_png

    env._cb.step = step_returns_raw_frame
    result = await env.step(
        [
            _action_call(
                _act("scroll", coordinate=[250, 750], direction="up", amount=4),
                call_id="scroll0",
            ),
        ]
    )

    assert env._cb.steps == [("ScrollAction", {"direction": "up", "amount": 400})]
    assert env._cursor_px == (200, 450)
    assert result.results[0].tool_call_id == "scroll0"
    assert result.results[0].images[-1] != raw_png
    assert result.results[0].error is None
    _assert_cursor_at(result.results[0].images[-1], 200, 450)


async def test_bench_drag_without_start_uses_tracked_cursor(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0)
    await env.reset()
    await env.step(
        [
            _action_call(
                _act("mouse_move", coordinate=[250, 500]),
                _act("drag", coordinate=[750, 500]),
            )
        ]
    )
    assert env._cb.steps == [
        ("MoveToAction", {"x": 200, "y": 300}),
        ("DragAction", {"from_x": 200, "from_y": 300, "to_x": 600, "to_y": 300}),
    ]


async def test_bench_step_backend_failure_raises_true_infra_failure(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0)
    await env.reset()

    async def fail_step(_action):
        raise RuntimeError("cb step failed")

    env._cb.step = fail_step
    with pytest.raises(
        TrueInfraFailure,
        match="cua.bench click backend failed: cb step failed",
    ) as exc_info:
        await env.step(
            [
                _action_call(_act("click", coordinate=[500, 500]), call_id="call_click"),
            ]
        )

    assert failure_category(exc_info.value) is FailureCategory.INFRA_FAILURE


async def test_bench_malformed_coordinate_returns_error_with_last_screenshot(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0)
    await env.reset()

    r = await env.step(
        [
            _action_call(_act("click", coordinate=[None, None]), call_id="bad_click"),
        ]
    )

    assert env._cb.steps == []
    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "bad_click"
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == (
        "invalid arguments for click: coordinate values must be finite numbers"
    )
    assert r.results[0].images[-1] == env._last_png
    assert r.results[0].text is None


async def test_bench_bad_wait_duration_returns_error_with_last_screenshot(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0)
    await env.reset()

    r = await env.step(
        [
            _action_call(_act("wait", duration=31), call_id="bad_wait"),
        ]
    )

    assert env._cb.steps == []
    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "bad_wait"
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == ("invalid arguments for wait: wait.duration must be <= 30")
    assert r.results[0].images[-1] == env._last_png
    assert r.results[0].text is None


async def test_bench_unsupported_action_is_not_invalid_arguments(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv(
        "some/env",
        0,
        max_steps=3,
        post_action_delay=0,
        valid_actions=None,
    )
    await env.reset()

    r = await env.step(
        [
            _action_call(_act("screenshot"), call_id="shot0"),
        ]
    )

    assert env._cb.steps == []
    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "shot0"
    assert r.results[0].images[-1] == env._last_png
    assert r.results[0].text is None
    assert r.results[0].error == "unsupported action: screenshot"
    assert r.results[0].metadata == {"is_error": True}


async def test_bench_unknown_tool_returns_error_only_result(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, post_action_delay=0)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("frobnicate", {}, call_id="foo0"),
        ]
    )

    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "foo0"
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].error == "unknown tool: frobnicate"
    assert r.results[0].metadata == {"is_error": True}


async def test_bench_inactive_finish_tool_returns_error_only_feedback(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv("some/env", 0, max_steps=3, extra_tools=[], post_action_delay=0)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("response", {"text": "Done."}, call_id="resp0"),
        ]
    )

    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "resp0"
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].error == "response is not available in this task."
    assert r.results[0].metadata == {"is_error": True}


async def test_bench_terminate_beats_truncation(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv(
        "some/env",
        0,
        max_steps=1,
        extra_tools=["terminate"],
        post_action_delay=0,
    )
    await env.reset()
    r = await env.step(
        [
            make_tool_call("terminate", {"status": "success"}, call_id="term0"),
        ]
    )  # → DoneAction, done=True
    assert r.terminated and not r.truncated and r.reward == 1.0  # evaluate() → 1.0
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    assert r.results == []


async def test_bench_response_with_call_id_returns_no_tool_result(fake_cb):
    from lite.gym.envs.cua.bench.main import CuaBenchEnv

    env = CuaBenchEnv(
        "some/env",
        0,
        max_steps=1,
        extra_tools=["response"],
        post_action_delay=0,
    )
    await env.reset()
    r = await env.step(
        [
            make_tool_call("response", {"text": "Done."}, call_id="resp0"),
        ]
    )

    assert r.terminated and not r.truncated and r.reward == 1.0
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    assert r.results == []


# --- cua.bench.local: DEDICATED container naming + reaper contract ----------
def test_dynamic_registration_declares_webtop_pure_and_local_dedicated(
    fake_cb,
    monkeypatch,
    tmp_path,
):
    import importlib

    registry_mod = importlib.import_module("lite.gym.registry")
    monkeypatch.setattr(registry_mod, "_specs", {})
    monkeypatch.setattr(registry_mod, "_splits", {})
    monkeypatch.setattr(registry_mod, "_env_make_kwargs", {})
    monkeypatch.setattr(registry_mod, "_declared_env_ids", set())
    monkeypatch.setattr(registry_mod, "_families", {})
    monkeypatch.setattr(registry_mod, "_services", {})
    monkeypatch.setattr(registry_mod, "_imported", {})
    monkeypatch.setattr(registry_mod, "_tasks_registered", set())

    dataset = tmp_path / "cua-bench-s02fam"
    webtop_env = dataset / "webtop_env"
    local_env = dataset / "local_env"
    webtop_env.mkdir(parents=True)
    local_env.mkdir()
    (webtop_env / "main.py").write_text("")
    (local_env / "main.py").write_text("")
    monkeypatch.setenv("CUA_BENCH_DATASET_ROOT", str(dataset))

    class _DiscoveryEnv:
        def __init__(self, provider):
            self._provider = provider

        def tasks_config_fn(self):
            computer = (
                {"provider": self._provider, "setup_config": {}}
                if self._provider is not None
                else None
            )
            return [types.SimpleNamespace(description="do it", computer=computer)]

    def make(path):
        name = Path(path).name
        return _DiscoveryEnv("webtop" if name == "webtop_env" else None)

    fake_cb.make = make
    cb_main = importlib.import_module("lite.gym.envs.cua.bench.main")
    from lite.gym.services import BackendFamily, family_of, services_for

    assert cb_main.registry.task_ids("cua.bench.webtop.s02fam") == {
        "eval": ["webtop_env"],
    }
    assert cb_main.registry.task_ids("cua.bench.local.s02fam") == {
        "eval": ["local_env"],
    }
    assert family_of("cua.bench.webtop.s02fam") is BackendFamily.PURE
    assert services_for("cua.bench.webtop.s02fam") is None
    assert cb_main.registry.env_make_kwargs("cua.bench.webtop.s02fam") == {"cursor": True}
    assert family_of("cua.bench.local.s02fam") is BackendFamily.DEDICATED
    assert cb_main.registry.env_make_kwargs("cua.bench.local.s02fam") == {"cursor": True}
    assert isinstance(
        services_for("cua.bench.local.s02fam"),
        cb_main.CuaBenchLocalServices,
    )


async def test_local_env_conformant_name_and_reaper_contract(fake_cb, monkeypatch):
    # CuaBenchLocalEnv must name its container per cua-lite's convention so the
    # STOCK container drift-reaper reclaims it (validated live in test_cua_bench_live);
    # here we assert the naming + EnvServerResource contract with docker mocked.
    from lite.gym.envs.cua.bench import main as cb_main
    from lite.gym.services import EnvServerResource

    calls: list[tuple] = []

    async def fake_docker(*args):
        calls.append(args)
        return 0, ""

    released_ports: list[int] = []

    monkeypatch.setattr(cb_main, "_docker", fake_docker)
    monkeypatch.setattr(
        cb_main,
        "allocate_ports",
        lambda **kwargs: [24001, 24002],
    )
    monkeypatch.setattr(
        cb_main,
        "release_ports",
        lambda *ports: released_ports.extend(ports),
    )
    env = cb_main.CuaBenchLocalEnv("some/env", 0, env_id="cua.bench.local.foo", max_steps=3)
    assert isinstance(env, EnvServerResource)
    assert env.external_resource_id is None  # None before the container exists

    await env.reset()
    erid = env.external_resource_id
    # conformant: lite-env-…-<env_id>-<suffix>_latest → matches the reaper's ^prefix.*-env_id- regex
    assert (
        erid
        and erid.startswith("lite-env-")
        and "cua.bench.local.foo" in erid
        and erid.endswith("_latest")
    )
    assert any(a[0] == "tag" for a in calls)  # base image tagged to conformant name
    assert (
        env._provider_setup_config["image"] == erid[: -len("_latest")] + ":latest"
    )  # image → conformant tag
    assert env._provider_setup_config["api_port"] == 24001
    assert env._provider_setup_config["noVNC_port"] == 24002
    assert env._ports_owned == (24001, 24002)
    assert env._cb.tasks[0].computer["provider"] == "native"  # local fixes the backend to native

    await env.close()
    assert any(a[0] == "rmi" for a in calls)  # close() drops the per-instance image ref alias
    assert released_ports == [24001, 24002]
    assert env.external_resource_id is None


async def test_local_env_releases_ports_when_image_tag_fails(fake_cb, monkeypatch):
    from lite.gym.envs.cua.bench import main as cb_main

    calls: list[tuple] = []
    released_ports: list[int] = []

    async def fake_docker(*args):
        calls.append(args)
        return 1, "tag failed"

    monkeypatch.setattr(cb_main, "_docker", fake_docker)
    monkeypatch.setattr(cb_main, "allocate_ports", lambda **kwargs: [24001, 24002])
    monkeypatch.setattr(cb_main, "release_ports", lambda *ports: released_ports.extend(ports))

    env = cb_main.CuaBenchLocalEnv("some/env", 0, env_id="cua.bench.local.foo", max_steps=3)
    with pytest.raises(RuntimeError, match="docker tag"):
        await env.reset()

    assert calls and calls[0][0] == "tag"
    assert released_ports == [24001, 24002]
    assert env._ports_owned == ()
    assert env.external_resource_id is None


def test_local_env_forces_cua_lite_resolution_and_ports(fake_cb):
    # _before_cb_reset FORCES the cua-lite desktop resolution (configs/default.yaml → 1920×1080)
    # on EVERY native task — overriding upstream's per-task setup_config (basic declares 1024×768,
    # workflows 1920×1080) — and os_type=linux with our image. cua-lite normalizes desktop tasks.
    from lite.gym.envs.cua.bench import main as cb_main

    env = cb_main.CuaBenchLocalEnv("some/env", 0, env_id="cua.bench.local.basic")
    env._provider_setup_config = {
        "image": "img:latest",
        "api_port": 24001,
        "noVNC_port": 24002,
    }

    class _CB:
        tasks = None

        async def create_sandbox(self, *args, **kwargs):
            pass

        def tasks_config_fn(self):
            return [
                types.SimpleNamespace(
                    computer={
                        "provider": "native",
                        "setup_config": {"os_type": "linux", "width": 1024, "height": 768},
                    }
                ),  # basic declares 1024×768
                types.SimpleNamespace(computer=None),  # computer-less (KiCad-style)
            ]

    cb = _CB()
    env._before_cb_reset(cb)
    for t in cb.tasks:  # both forced to the cua-lite resolution, regardless of what they declared
        sc = t.computer["setup_config"]
        assert (sc["width"], sc["height"]) == (cb_main._SETUP_W, cb_main._SETUP_H) == (1920, 1080)
        assert sc["os_type"] == "linux" and sc["image"] == "img:latest"
        assert sc["api_port"] == 24001
        assert sc["noVNC_port"] == 24002


async def test_port_aware_session_passes_reserved_ports_to_computer(fake_cb, monkeypatch):
    from lite.gym.envs.cua.bench import main as cb_main

    remote_mod = types.ModuleType("cua_bench.computers.remote")
    sys.modules["cua_bench"].__path__ = []
    computers_pkg = types.ModuleType("cua_bench.computers")
    computers_pkg.__path__ = []

    class _RemoteDesktopSession:
        def __init__(self, **kwargs):
            self._client_only_mode = False
            self._initialized = False
            self._computer = None
            self._os_type = kwargs.get("os_type", "linux")
            self._provider_type = kwargs.get("provider_type", "docker")
            self._image = kwargs.get("image")
            self._width = kwargs.get("width", 1920)
            self._height = kwargs.get("height", 1080)
            self._memory = kwargs.get("memory")
            self._cpu = kwargs.get("cpu")
            self._name = kwargs.get("name")
            self._storage = kwargs.get("storage")
            self._ephemeral = kwargs.get("ephemeral", True)
            self._vnc_port = 8006
            self._api_port = None

    remote_mod.RemoteDesktopSession = _RemoteDesktopSession
    monkeypatch.setitem(sys.modules, "cua_bench.computers", computers_pkg)
    monkeypatch.setitem(sys.modules, "cua_bench.computers.remote", remote_mod)

    computer_mod = types.ModuleType("computer")
    seen: dict[str, object] = {}

    class _Computer:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.noVNC_port = kwargs["noVNC_port"]

        async def run(self):
            seen["ran"] = True

    computer_mod.Computer = _Computer
    monkeypatch.setitem(sys.modules, "computer", computer_mod)

    touched_ports: list[int] = []
    monkeypatch.setattr(cb_main, "touch_ports", lambda *ports: touched_ports.extend(ports))

    session_cls = cb_main._port_aware_remote_session_class()
    session = session_cls(
        api_port=24001,
        noVNC_port=24002,
        image="img:latest",
        width=1920,
        height=1080,
    )
    await session._ensure_computer()

    assert touched_ports == [24001, 24002]
    assert seen["api_port"] == 24001
    assert seen["noVNC_port"] == 24002
    assert seen["image"] == "img:latest"
    assert seen["display"] == "1920x1080"
    assert seen["ran"] is True


async def test_port_aware_create_sandbox_moves_ports_to_session_config(fake_cb, monkeypatch):
    from lite.gym.envs.cua.bench import main as cb_main

    sys.modules["cua_bench"].__path__ = []
    bot_mod = types.ModuleType("cua_bench.bot")
    bot_mod.Bot = lambda env: ("bot", env)
    computers_pkg = types.ModuleType("cua_bench.computers")
    computers_pkg.__path__ = []

    class DesktopSetupConfig(dict):
        pass

    computers_pkg.DesktopSetupConfig = DesktopSetupConfig
    remote_mod = types.ModuleType("cua_bench.computers.remote")

    class _RemoteDesktopSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._api_port = 8000
            self._vnc_port = 8006
            self._client_only_mode = False
            self.started_with = None
            self.page = "page"

        async def start(self, *, config, headless):
            self.started_with = (config, headless)

    remote_mod.RemoteDesktopSession = _RemoteDesktopSession
    monkeypatch.setitem(sys.modules, "cua_bench.bot", bot_mod)
    monkeypatch.setitem(sys.modules, "cua_bench.computers", computers_pkg)
    monkeypatch.setitem(sys.modules, "cua_bench.computers.remote", remote_mod)

    calls: list[tuple] = []

    class _CB:
        headless = True

        async def create_sandbox(self, **kwargs):
            calls.append(kwargs)

    cb = _CB()
    cb_main._install_port_aware_create_sandbox(cb)
    await cb.create_sandbox(
        provider="native",
        provider_config={"memory": "4GB"},
        setup_config={
            "width": 1920,
            "height": 1080,
            "image": "img:latest",
            "api_port": 24001,
            "noVNC_port": 24002,
        },
    )

    assert calls == []
    assert cb.session_name == "native"
    assert cb.session_config == {"memory": "4GB", "api_port": 24001, "noVNC_port": 24002}
    assert cb.setup_config == {"width": 1920, "height": 1080, "image": "img:latest"}
    assert cb.session._api_port == 24001
    assert cb.session._vnc_port == 24002
    assert cb.session.started_with == (cb.setup_config, True)
    assert cb.page == "page"
    assert cb.bot == ("bot", cb)

    await cb.create_sandbox(provider="webtop", provider_config={"x": 1}, setup_config={"y": 2})
    assert calls == [{"provider": "webtop", "provider_config": {"x": 1}, "setup_config": {"y": 2}}]


def test_iter_datasets_rejects_symlink_envs_outside_declared_root(fake_cb, tmp_path):
    from lite.gym.envs.cua.bench import main as cb_main

    cache_root = tmp_path / "cache"
    dataset = cache_root / "cua-bench-safe"
    safe_env = dataset / "inside"
    outside_env = tmp_path / "outside"
    safe_env.mkdir(parents=True)
    outside_env.mkdir()
    (safe_env / "main.py").write_text("")
    (outside_env / "main.py").write_text("")
    try:
        (dataset / "escaped").symlink_to(outside_env, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    assert list(cb_main._iter_datasets(cache_root)) == [("safe", [safe_env])]


def test_iter_datasets_rejects_symlink_direct_envs_outside_declared_root(fake_cb, tmp_path):
    from lite.gym.envs.cua.bench import main as cb_main

    dataset = tmp_path / "cua-bench-direct"
    safe_env = dataset / "inside"
    outside_env = tmp_path / "outside"
    safe_env.mkdir(parents=True)
    outside_env.mkdir()
    (safe_env / "main.py").write_text("")
    (outside_env / "main.py").write_text("")
    try:
        (dataset / "escaped").symlink_to(outside_env, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")

    assert list(cb_main._iter_datasets(dataset)) == [("direct", [safe_env])]


# --- graceful degradation (cua-bench not installed) -------------------------
@pytest.mark.skipif(_HAVE_CB, reason="tests the deps-MISSING path; cua-bench is installed here")
def test_gym_make_raises_env_deps_missing():
    with pytest.raises(EnvDepsMissingError):
        gym.make("cua.bench.local.kicad@154d0750")  # env_id = dataset, task_id = environment


def test_registered_env_ids_unaffected_by_missing_cua():
    # importing the whole catalog must not crash on cua.bench's guarded import
    assert isinstance(gym.registry.registered_env_ids(), list)
