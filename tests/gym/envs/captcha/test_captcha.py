"""Tests for the captcha CUA-Lite gym environment.

Pure-logic tests run anywhere ``lite`` is installed. Tests that import the
captcha challenge servers or register tasks need the captcha deps
(``scripts/install.sh``: flask/pillow/playwright) and downloaded assets,
and skip otherwise.

Run:
    uv run pytest tests/gym/envs/captcha/test_captcha.py -v
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys

import pytest

from lite.core.tools import make_tool_call
from lite.gym.errors import EnvDepsMissingError

try:
    from lite.gym.envs.captcha.main import _find_free_port
except (
    EnvDepsMissingError
) as e:  # captcha deps (playwright, flask) absent until the env's install.sh runs
    pytest.skip(f"captcha env deps not installed: {e}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Pure helpers — no captcha extra / assets needed
# ---------------------------------------------------------------------------

# Keyboard-key normalization moved to the canonical chokepoint
# (LiteDesktopActionSpace.key) + the shared lite/gym/utils/backend/keys.py; its coverage
# lives in tests/gym/utils/backend/test_keys.py. captcha now calls keys.to_playwright on
# already-canonical keys (exercised by the live captcha rollout below).


def _tc(name: str, arguments: dict | None = None, *, call_id: str | None = None) -> dict:
    return make_tool_call(name, arguments, call_id=call_id)


def test_find_free_port_is_bindable():
    port = _find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535
    # The returned port must actually be bindable.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


# ---------------------------------------------------------------------------
# Text-captcha grading — the case-insensitive fix. Needs flask + PIL.
# ---------------------------------------------------------------------------


def _text_server():
    try:
        from lite.gym.envs.captcha.servers import text_captcha_4 as server
    except Exception as exc:  # missing flask/PIL/assets
        pytest.skip(f"servers/text_captcha_4.py not importable: {exc}")
    return server


def test_text_grading_case_insensitive():
    check = _text_server().check_answer
    assert check("AB12", "AB12") is True
    # The fix: normalization is applied on BOTH sides (was user-side only).
    assert check("ab12", "AB12") is True
    assert check("AB12", "ab12") is True
    assert check("Ab12", "aB12") is True
    # Surrounding whitespace is stripped.
    assert check(" ab12 ", "AB12") is True
    # Genuine mismatches still fail.
    assert check("ab13", "AB12") is False
    assert check("", "AB12") is False


# ---------------------------------------------------------------------------
# Registration — needs captcha deps + assets (skips otherwise). Runs in a
# direct-mode subprocess so it works regardless of CUA_LITE_ENV_SERVER_URL
# in the caller's shell (registration is gated on that env var and the
# registry caches the routing mode per process).
# ---------------------------------------------------------------------------

_CAPTCHA_CATEGORIES = (
    "text_captcha_4",
    "slider",
    "rotation",
    "icon_click",
    "math",
    "image_select",
    "icon_match",
    "paged",
)

_REGISTRATION_PROBE = """
import json
import lite.gym as gym

ids = gym.registry.task_ids("captcha")
modes = {}
per_cat = {}
n_train_base = 0
easy_train = []
for split, tasks in ids.items():
    for t in tasks:
        o = gym.registry.task_metadata("captcha", t).others
        cat = o["category"]
        per_cat.setdefault(cat, {}).setdefault(split, 0)
        per_cat[cat][split] += 1
        m = o.get("mode")
        if m:
            modes.setdefault(m, 0)
            modes[m] += 1
        if split == "train" and not m:
            n_train_base += 1
        if split == "train" and m == "easy":
            easy_train.append(t)
print(json.dumps({
    "totals": {k: len(v) for k, v in ids.items()},
    "per_cat": per_cat,
    "n_train_base": n_train_base,
    "easy_train": easy_train,
    "modes": modes,
}))
"""


def test_tasks_registered_with_filter_metadata():
    env = {k: v for k, v in os.environ.items() if k != "CUA_LITE_ENV_SERVER_URL"}
    r = subprocess.run(
        [sys.executable, "-c", _REGISTRATION_PROBE],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        if "EnvDepsMissingError" in r.stderr:
            pytest.skip(f"captcha deps/assets unavailable: {r.stderr.splitlines()[-1]}")
        raise AssertionError(f"registration probe failed:\n{r.stderr[-2000:]}")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    # Split semantics are distribution semantics: train + eval are
    # in-distribution (default train_eval.json), test is always OOD. These
    # totals reflect the FULL canonical asset set (HF OnAnOrange/captcha-assets,
    # installed by scripts/install.sh) — every category ships a 32-sample OOD
    # test set, image_select adds its 72-sample reCAPTCHA "halligan" set, and
    # rotation keeps its one "easy" train task. (A partial install registers
    # fewer: registration is data-driven, silently skipping categories whose
    # asset dir is absent. The env-server health probe now fails closed on a
    # partial install, so a server that reports captcha available=true has the
    # full set below.) Update these numbers in lockstep when the dataset changes.
    assert out["totals"] == {"train": 10, "eval": 297, "test": 296}
    # others["category"] is the primary selection key (one env_id; pick
    # categories at export time via --filter). Every category has train + eval
    # + a 32-sample test set; image_select carries crop + full (2 train, 66
    # eval) plus its 72-sample halligan test set; rotation has 2 train (base +
    # "easy").
    assert set(out["per_cat"]) == set(_CAPTCHA_CATEGORIES)
    for cat, c in out["per_cat"].items():
        if cat == "image_select":
            assert c == {"train": 2, "eval": 66, "test": 72}, (cat, c)
        elif cat == "rotation":
            assert c == {"train": 2, "eval": 33, "test": 32}, (cat, c)
        else:
            assert c == {"train": 1, "eval": 33, "test": 32}, (cat, c)
    # 7 mode-less standard train tasks; image_select's two standard train
    # tasks carry mode=crop/full (it has no single default mode).
    assert out["n_train_base"] == 7
    # others["mode"] is the fine-grained filter key below category.
    assert out["easy_train"] == ["random_rotation_easy_local"]
    assert out["modes"] == {"crop": 34, "full": 34, "easy": 1, "halligan": 72}


# ---------------------------------------------------------------------------
# live_ids — captcha's liveness oracle for the framework's ghost detection.
# It returns the EXACT set of live ``port:<n>`` ext-ids reconstructed from the
# live marker-tagged Flask procs. The framework reconcile loop diffs tracked
# ids against this set: a tracked ``port:<n>`` absent from live_ids is ghosted.
# (Was test_reap_drift_tracks_env_id; ghost detection moved from the env's
# reap_drift to the framework — see tests/gym/remote/test_reconcile.py. Orphan
# reaping is covered by test_reap_zombies_kills_reported_orphans below.)
# ---------------------------------------------------------------------------


def test_live_ids_reflects_live_flask_ports(monkeypatch):
    try:
        from lite.gym.envs.captcha import main as captcha_main
    except Exception as exc:
        pytest.skip(f"captcha main not importable: {exc}")
    import os

    from lite.gym.remote.scope import ServerScope

    # Two live Flask procs (port 50001 as OUR child, 61234 as an orphan).
    # live_ids reconstructs ``port:<n>`` for every live proc; a tracked
    # ``port:<n>`` NOT in this set (e.g. 50002, whose proc is gone) is what
    # the framework ghosts.
    fake_procs = {
        50001: (111, os.getpid(), 9999.0),
        61234: (222, 1, 9999.0),
    }
    # _flask_server_procs now takes the server scope (so a co-tenant server's
    # procs on a different port don't pollute the liveness set); the stub
    # ignores it but must accept it.
    monkeypatch.setattr(captcha_main, "_flask_server_procs", lambda scope: fake_procs)

    live = captcha_main.CaptchaServices().live_ids("captcha", ServerScope())
    assert live == {"port:50001", "port:61234"}
    # A tracked port whose proc is gone is absent -> framework ghosts it.
    assert "port:50002" not in live


def test_live_ids_raises_on_probe_failure(monkeypatch):
    """A pgrep probe failure (``_live_flask_port_ids`` returns None)
    makes live_ids RAISE ReconcileProbeError so the reconcile loop fails closed (skips the
    cycle) — never returns None (None would mean "no per-instance world", which captcha is not)."""
    try:
        from lite.gym.envs.captcha import main as captcha_main
    except Exception as exc:
        pytest.skip(f"captcha main not importable: {exc}")
    from lite.gym.errors import ReconcileProbeError
    from lite.gym.remote.scope import ServerScope

    monkeypatch.setattr(captcha_main, "_live_flask_port_ids", lambda scope: None)
    with pytest.raises(ReconcileProbeError):
        captcha_main.CaptchaServices().live_ids("captcha", ServerScope())


# ---------------------------------------------------------------------------
# Reward lock — regression for the correct->wrong overwrite bug. The env
# reads /result ONCE at the terminal step, so a wrong re-submit after a
# correct one must not clobber the stored result. Runs each Flask server
# in-process via test_client (no browser needed); subprocess per server so
# module-level challenge generation gets its own env vars and the shared
# RESULT_FILE default never collides across xdist workers.
# ---------------------------------------------------------------------------

_LOCK_PROBE = """
import json, os, sys
os.environ["CAPTCHA_MODE"] = sys.argv[2]
os.environ["CAPTCHA_RESULT_FILE"] = sys.argv[1]
os.environ["CAPTCHA_SEED"] = "42"
from lite.gym.envs.captcha.servers import {module} as srv
c = srv.app.test_client()
r1 = c.post("/verify", json={correct}).get_json()
r2 = c.post("/verify", json={wrong}).get_json()
r3 = json.loads(c.get("/result").get_data())
print(json.dumps({{"first": r1["correct"], "after_wrong": r2["correct"],
                   "result": r3["correct"]}}))
"""


@pytest.mark.parametrize(
    "module, mode, correct, wrong",
    [
        (
            "rotation",
            "train_eval",
            "{'angle': srv.ROTATION_ANGLE}",
            "{'angle': (srv.ROTATION_ANGLE + 180) % 360}",
        ),
        (
            "icon_click",
            "train_eval",
            "{'clicks': [{'x': x, 'y': y} for x, y in srv.TARGET_POSITIONS]}",
            "{'clicks': []}",
        ),
        ("image_select", "crop", "{'selected': sorted(srv._POSITIVE_TILES)}", "{'selected': []}"),
        ("slider", "train_eval", "{'x': srv.TARGET_X}", "{'x': srv.TARGET_X + 200}"),
    ],
)
def test_correct_result_locked_against_wrong_resubmit(module, mode, correct, wrong, tmp_path):
    code = _LOCK_PROBE.format(module=module, correct=correct, wrong=wrong)
    r = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "result.json"), mode],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 and (
        "ModuleNotFoundError" in r.stderr
        or "EnvDepsMissingError" in r.stderr
        or "FileNotFoundError" in r.stderr
    ):
        pytest.skip(f"{module} server deps/assets unavailable")
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"first": True, "after_wrong": True, "result": True}, out


# ---------------------------------------------------------------------------
# Client-mode import gate — importing the module with CUA_LITE_ENV_SERVER_URL
# set must be side-effect free: no registration, no asset download, and no
# network call (the URL below is unreachable; touching it would fail loudly).
# ---------------------------------------------------------------------------


def test_client_mode_import_is_side_effect_free():
    env = {**os.environ, "CUA_LITE_ENV_SERVER_URL": "http://127.0.0.1:1"}
    code = (
        "import lite.gym.envs.captcha.main\n"
        "from lite.gym.registry import _splits\n"
        "print(sum(1 for k in _splits if k.startswith('captcha')))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "0"


# ---------------------------------------------------------------------------
# Seed determinism — CAPTCHA_SEED + PYTHONHASHSEED must yield byte-identical
# challenges (the contract behind the 33 pinned eval seeds).
# ---------------------------------------------------------------------------


def test_seeded_challenge_is_byte_identical():
    code = (
        "import hashlib\n"
        "from lite.gym.envs.captcha.servers import math as srv\n"
        "print(hashlib.sha256(srv.CAPTCHA_IMAGE).hexdigest())\n"
    )
    digests = []
    for _ in range(2):
        env = {**os.environ, "CAPTCHA_SEED": "42", "PYTHONHASHSEED": "42"}
        env.pop("CUA_LITE_ENV_SERVER_URL", None)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        if r.returncode != 0 and (
            "ModuleNotFoundError" in r.stderr or "FileNotFoundError" in r.stderr
        ):
            pytest.skip("math server deps/assets unavailable")
        assert r.returncode == 0, r.stderr[-2000:]
        digests.append(r.stdout.strip())
    assert digests[0] == digests[1], digests


# ---------------------------------------------------------------------------
# reap_zombies — kills exactly what _orphan_pids reports, nothing else.
# ---------------------------------------------------------------------------


def test_reap_zombies_kills_reported_orphans(monkeypatch):
    try:
        from lite.gym.envs.captcha import main as captcha_main
    except Exception as exc:
        pytest.skip(f"captcha main not importable: {exc}")
    from lite.gym.remote.scope import ServerScope

    # The Flask marker is server-scoped: a default ServerScope (server_port
    # None) reaps the "direct" marker, NOT the bare unscoped string. A reap
    # keyed off the wrong (unscoped) marker would match nothing — the exact
    # bug this guards. The chromium marker stays unscoped (Playwright owns it).
    flask_marker = captcha_main._server_marker("direct")
    assert "esdirect_" in flask_marker  # scoped, not the legacy bare marker
    fake = {flask_marker: [101, 102], "playwright_chromiumdev": [303]}
    monkeypatch.setattr(captcha_main, "_orphan_pids", lambda pat: fake.get(pat, []))
    killed: list[int] = []
    monkeypatch.setattr(captcha_main.os, "kill", lambda pid, sig: killed.append(pid))
    assert captcha_main.CaptchaServices().reap("captcha", ServerScope(), set()) == 3
    assert killed == [101, 102, 303]


def test_reap_zombies_is_server_scoped(monkeypatch):
    """A reap on server port P must target ONLY P's marker — a co-tenant
    env-server on a different port leaves its leaked procs untouched."""
    try:
        from lite.gym.envs.captcha import main as captcha_main
    except Exception as exc:
        pytest.skip(f"captcha main not importable: {exc}")
    from lite.gym.remote.scope import ServerScope

    seen_patterns: list[str] = []

    def _fake_orphan_pids(pat):
        seen_patterns.append(pat)
        return []

    monkeypatch.setattr(captcha_main, "_orphan_pids", _fake_orphan_pids)
    captcha_main.CaptchaServices().reap(
        "captcha", ServerScope.from_server(server_port=30180, strict_token=None), set()
    )
    # The Flask pattern is scoped to THIS server's port; chromium stays generic.
    assert captcha_main._server_marker("30180") in seen_patterns
    assert captcha_main._server_marker("direct") not in seen_patterns


# ---------------------------------------------------------------------------
# atexit backstop — terminate live Flask servers a crash left unclosed.
# Mirrors the androidworld/androidlab/sandbox.base _LIVE + atexit registries.
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, alive: bool = True):
        self._alive = alive
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False


class _FakeEnv:
    def __init__(self, proc):
        self._server_proc = proc


def test_atexit_terminates_only_live_registered_servers(monkeypatch):
    from lite.gym.envs.captcha import main as captcha_main

    live, dead = _FakeProc(alive=True), _FakeProc(alive=False)
    env_live, env_dead = _FakeEnv(live), _FakeEnv(dead)
    # The hook snapshots under _LIVE_LOCK and terminates only un-exited procs.
    monkeypatch.setattr(captcha_main, "_LIVE_ENVS", {env_live, env_dead})
    captcha_main._cleanup_live_captcha_servers()
    assert live.terminated is True  # live registered server is reaped
    assert dead.terminated is False  # already-exited proc is left alone


# ---------------------------------------------------------------------------
# L2 dead-backend fail-fast: a dead per-episode Flask
# server must TRUNCATE loudly, not silently grind to max_steps scoring 0.
# step() is unit-tested in isolation via __new__ (no Playwright / no real
# server) — it only touches the attributes set below.
# ---------------------------------------------------------------------------


def _bare_captcha_env(proc):
    from lite.gym.envs.captcha import main as captcha_main

    env = captcha_main.LocalCaptchaEnv.__new__(captcha_main.LocalCaptchaEnv)
    env._step_count = 0
    env._max_steps = 15
    env._display_resolution = (800, 600)
    env._post_action_delay = 0.0
    env._port = 12345
    env._server_proc = proc
    env._category = "text_captcha_4"
    env._mode = None
    env._valid_actions = list(captcha_main._VALID_ACTIONS)
    env._extra_tool_schemas = captcha_main.resolve_extra_tools(
        captcha_main._EXTRA_TOOLS,
        tools=captcha_main.CaptchaTools,
    )

    async def _ss():
        return "SCREENSHOT_B64"

    async def _dispatch(name, args, w, h):
        return []

    env._take_screenshot = _ss
    env._dispatch_action = _dispatch
    env._evaluate_final = lambda: 0.0
    return env


class _NoopKeyboard:
    def __init__(self):
        self.calls: list[tuple] = []

    async def type(self, text):
        self.calls.append(("type", text))

    async def press(self, keys):
        self.calls.append(("press", keys))

    async def down(self, key):
        self.calls.append(("down", key))

    async def up(self, key):
        self.calls.append(("up", key))


class _NoopMouse:
    async def click(self, *args, **kwargs):
        pass

    async def move(self, *args, **kwargs):
        pass

    async def down(self, *args, **kwargs):
        pass

    async def up(self, *args, **kwargs):
        pass

    async def wheel(self, *args, **kwargs):
        pass


class _NoopPage:
    def __init__(self):
        self.keyboard = _NoopKeyboard()
        self.mouse = _NoopMouse()


def _use_real_captcha_dispatch(env):
    from lite.gym.envs.captcha import main as captcha_main

    env._page = _NoopPage()
    env._cursor_x = 0
    env._cursor_y = 0
    env._dispatch_action = captcha_main.LocalCaptchaEnv._dispatch_action.__get__(
        env,
        captcha_main.LocalCaptchaEnv,
    )
    return env


@pytest.mark.asyncio
async def test_dead_flask_server_truncates_loudly():
    """A Flask server that exited mid-episode (poll() != None) → truncate +
    info['pool_unreachable'] True, instead of grinding to max_steps."""

    class _DeadProc:
        returncode = 1

        def poll(self):
            return 1

    env = _bare_captcha_env(_DeadProc())
    env._evaluate_final = lambda: (_ for _ in ()).throw(
        AssertionError("dead backend truncation must not evaluate final reward")
    )
    r = await env.step(
        [
            _tc("click", {"coordinate": [10, 10]}),
        ]
    )
    assert r.truncated is True
    assert r.reward is None
    assert r.info.get("pool_unreachable") is True


@pytest.mark.asyncio
async def test_alive_flask_server_steps_normally():
    """A live server (poll() == None) → normal step, NOT truncated, no
    pool_unreachable flag — the fail-fast must not fire on a healthy backend."""

    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _bare_captcha_env(_AliveProc())
    r = await env.step(
        [
            _tc("click", {"coordinate": [10, 10]}),
        ]
    )
    assert r.truncated is False
    assert r.info.get("pool_unreachable") is False


@pytest.mark.asyncio
async def test_action_batch_returns_one_frame_per_executed_action():
    """N executed actions → N frames, in action order, each its own capture.

    Distinctness is asserted, not just the count: re-emitting one cached frame N
    times would satisfy the count while carrying no information at all.
    """

    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _use_real_captcha_dispatch(_bare_captcha_env(_AliveProc()))
    frames = ["SHOT_0", "SHOT_1", "SHOT_2"]
    pending = iter(frames)

    async def _ss():
        return next(pending)

    env._take_screenshot = _ss

    r = await env.step(
        [
            _tc(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [10, 10]},
                        {"action": "type", "text": "ab"},
                        {"action": "scroll", "direction": "down", "amount": 2},
                    ]
                },
                call_id="gui0",
            ),
        ]
    )

    assert r.results[0].tool_call_id == "gui0"
    assert r.results[0].images == frames
    assert len(set(r.results[0].images)) == 3


@pytest.mark.asyncio
async def test_step_with_no_executed_action_still_returns_one_frame():
    """Zero executed actions still owes the model one current observation."""

    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _use_real_captcha_dispatch(_bare_captcha_env(_AliveProc()))
    r = await env.step([_tc("click", {"coordinate": [None, None]}, call_id="click0")])

    assert r.results[0].images == ["SCREENSHOT_B64"]


@pytest.mark.asyncio
async def test_dispatch_exception_returns_tool_result_error():
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _bare_captcha_env(_AliveProc())

    async def _dispatch(name, args, w, h):
        raise RuntimeError("playwright click failed")

    env._dispatch_action = _dispatch
    r = await env.step(
        [
            _tc("click", {"coordinate": [10, 10]}, call_id="call_click"),
        ]
    )

    assert r.truncated is False
    assert r.results[0].tool_call_id == "call_click"
    assert r.results[0].images[-1] == "SCREENSHOT_B64"
    assert r.results[0].error == "click failed: execution failed"
    assert "playwright click failed" not in r.results[0].error
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "expected_error", "expect_current_image"),
    [
        ("type", {"text": ""}, "unsupported action: type", False),
        (
            "type",
            {"text": 123},
            "invalid arguments for type: type.arguments.text must be a string",
            True,
        ),
        ("key", {"keys": []}, "invalid arguments for key: key.keys must not be empty", True),
        (
            "key",
            {"keys": ["plus"]},
            "invalid arguments for key: unknown key token 'plus'",
            True,
        ),
        (
            "key",
            {"keys": [" "]},
            "invalid arguments for key: unknown key token ' '",
            True,
        ),
        (
            "key",
            {"keys": ["Ctrl"]},
            "invalid arguments for key: keys must be lowercase tokens; "
            "split chords into separate keys; got 'Ctrl'",
            True,
        ),
        (
            "click",
            {"coordinate": [None, None]},
            "invalid arguments for click: coordinate values must be finite numbers",
            True,
        ),
        ("drag", {}, "unsupported action: drag", False),
    ],
)
async def test_route_known_gui_dispatch_errors_return_expected_carrier(
    name,
    arguments,
    expected_error,
    expect_current_image,
):
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _use_real_captcha_dispatch(_bare_captcha_env(_AliveProc()))

    r = await env.step(
        [
            _tc(name, arguments, call_id=f"{name}0"),
        ]
    )

    assert r.truncated is False
    assert r.terminated is False
    assert r.reward is None
    assert env._step_count == 1
    assert r.results[0].tool_call_id == f"{name}0"
    assert (r.results[0].images[-1] if r.results[0].images else None) == (
        "SCREENSHOT_B64" if expect_current_image else None
    )
    assert r.results[0].text is None
    assert expected_error in r.results[0].error
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_key_down_accepts_canonical_glyphs():
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _use_real_captcha_dispatch(_bare_captcha_env(_AliveProc()))

    calls = await env._dispatch_action("key_down", {"keys": ["+", "-", "="]}, 800, 600)

    assert env._page.keyboard.calls == [("down", "+"), ("down", "-"), ("down", "=")]
    assert calls == [{"call": "keyboard.down", "args": {"keys": ["+", "-", "="]}}]


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [-1, "NaN", "Infinity", 31])
async def test_bad_wait_duration_returns_current_screenshot_without_sleep(
    monkeypatch,
    duration,
):
    from lite.gym.envs.captcha import main as captcha_main

    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    async def _sleep(_seconds):
        raise AssertionError("bad wait duration reached asyncio.sleep")

    monkeypatch.setattr(captcha_main.asyncio, "sleep", _sleep)
    env = _use_real_captcha_dispatch(_bare_captcha_env(_AliveProc()))

    r = await env.step(
        [
            _tc("wait", {"duration": duration}, call_id="wait0"),
        ]
    )

    assert r.truncated is False
    assert r.terminated is False
    assert env._step_count == 1
    assert r.results[0].tool_call_id == "wait0"
    assert r.results[0].images[-1] == "SCREENSHOT_B64"
    assert r.results[0].text is None
    assert r.results[0].error.startswith("invalid arguments for wait: wait.duration")
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_route_known_gui_dispatch_error_at_max_steps_evaluates_once():
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _use_real_captcha_dispatch(_bare_captcha_env(_AliveProc()))
    env._max_steps = 1
    eval_calls = []
    env._evaluate_final = lambda: eval_calls.append("eval") or 0.5

    r = await env.step(
        [
            _tc("key", {"keys": ["Ctrl"]}, call_id="key0"),
        ]
    )

    assert eval_calls == ["eval"]
    assert r.truncated is True
    assert r.terminated is False
    assert r.reward == 0.5
    assert env._step_count == 1
    assert r.results[0].tool_call_id == "key0"
    assert r.results[0].images[-1] == "SCREENSHOT_B64"
    assert (
        "invalid arguments for key: keys must be lowercase tokens; "
        "split chords into separate keys; got 'Ctrl'"
    ) in (r.results[0].error)
    assert r.results[0].metadata == {"is_error": True}


def test_to_pixel_rejects_malformed_coordinate():
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _bare_captcha_env(_AliveProc())
    with pytest.raises(ValueError, match="malformed normalized coordinate: None"):
        env._to_pixel(None, 800, 600)


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_only_tool_result():
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _bare_captcha_env(_AliveProc())
    r = await env.step(
        [
            _tc("frobnicate", {}, call_id="foo0"),
        ]
    )

    assert r.truncated is False
    assert r.terminated is False
    assert env._step_count == 1
    assert r.results[0].tool_call_id == "foo0"
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].error == "unknown tool: frobnicate"
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_inactive_finish_tool_returns_error_only_feedback():
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _bare_captcha_env(_AliveProc())
    r = await env.step(
        [
            _tc("response", {"text": "Done."}, call_id="resp0"),
        ]
    )

    assert r.truncated is False
    assert r.terminated is False
    assert env._step_count == 1
    assert r.results[0].tool_call_id == "resp0"
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert r.results[0].error == "response is not available in this task."
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_active_response_returns_no_tool_result():
    from lite.gym.envs.captcha import main as captcha_main

    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _bare_captcha_env(_AliveProc())
    env._extra_tool_schemas = captcha_main.resolve_extra_tools(
        ["response"],
        tools=captcha_main.CaptchaTools,
    )
    r = await env.step(
        [
            _tc("response", {"text": "Done."}, call_id="resp0"),
        ]
    )

    assert r.truncated is False
    assert r.terminated is True
    assert r.reward == 0.0
    assert env._step_count == 1
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # ``test_max_steps_truncation_returns_paired_screenshot`` below is the
    # contrast: a TRUNCATED step's last call is an ordinary action and keeps its
    # frame.
    assert r.results == []


@pytest.mark.asyncio
async def test_max_steps_truncation_returns_paired_screenshot():
    class _AliveProc:
        returncode = None

        def poll(self):
            return None

    env = _bare_captcha_env(_AliveProc())
    env._max_steps = 1
    r = await env.step(
        [
            _tc("wait", {"duration": 0}, call_id="wait0"),
        ]
    )

    assert r.truncated is True
    assert r.terminated is False
    assert env._step_count == 1
    assert r.results[0].tool_call_id == "wait0"
    assert r.results[0].images[-1] == "SCREENSHOT_B64"
    assert r.results[0].text is None
    assert r.results[0].error is None


# ---------------------------------------------------------------------------
# Cursor overlay. captcha drives its OWN headless Chromium, whose capture buffer
# carries no native pointer — so the env composites the tracked cursor itself,
# exactly like the other browser-backed envs. captcha is a click-PRECISION
# benchmark, so a cursor-less frame means the agent cannot see what it is aiming.
# ---------------------------------------------------------------------------


def _solid_png(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _cursor_env(*, cursor: bool = True, resolution=(800, 600)):
    from lite.gym.envs.captcha import main as captcha_main

    env = captcha_main.LocalCaptchaEnv(
        server_file="unused.py",
        instruction="solve it",
        display_resolution=resolution,
        cursor=cursor,
    )
    raw = _solid_png(*resolution)

    class _Mouse:
        def __init__(self):
            self.moves: list[tuple[float, float]] = []

        async def move(self, x, y):
            self.moves.append((x, y))

    class _Page:
        def __init__(self):
            self.mouse = _Mouse()

        async def screenshot(self):
            return raw

    env._page = _Page()
    return env, raw


def test_captcha_declares_cursor_make_kwarg():
    """captcha opts into env-owned cursor repaint via ``make_kwargs.cursor``."""
    from lite.gym.envs.captcha import main as captcha_main

    assert captcha_main.CFG.make_kwargs.get("cursor") is True


@pytest.mark.asyncio
async def test_captcha_cursor_origin_is_viewport_centre():
    """Session-start pointer origin = viewport CENTRE, matching the retired
    ``CursorOverlayWrapper`` (``(500, 500)`` normalized) and waa / cua.*."""
    env, _ = _cursor_env(resolution=(800, 600))

    assert (env._cursor_x, env._cursor_y) == (400, 300)

    env._cursor_x, env._cursor_y = 7, 9
    await env._reset_cursor()

    assert (env._cursor_x, env._cursor_y) == (400, 300)

    wide, _ = _cursor_env(resolution=(1920, 1080))

    assert (wide._cursor_x, wide._cursor_y) == (960, 540)


@pytest.mark.asyncio
async def test_captcha_reset_establishes_the_centre_with_a_real_mouse_move():
    """The centre must be a FACT about Chromium's pointer, not an assumption:
    ``_reset_cursor`` issues a real ``page.mouse.move`` so the composited
    coordinate is where the pointer actually is."""
    env, _ = _cursor_env(resolution=(800, 600))

    assert env._page.mouse.moves == []
    await env._reset_cursor()

    assert env._page.mouse.moves == [(400, 300)]
    assert (env._cursor_x, env._cursor_y) == (400, 300)


@pytest.mark.asyncio
async def test_captcha_capture_composites_cursor_sprite():
    """The capture path actually paints the sprite (config alone is not enough:
    a wired-but-unapplied overlay would still ship a cursor-less frame)."""
    from PIL import Image

    env, raw = _cursor_env(cursor=True)
    painted = await env._take_screenshot()

    assert painted != raw

    image = Image.open(io.BytesIO(painted)).convert("RGB")
    x, y = env._cursor_x, env._cursor_y
    # The sprite's tip sits AT the tracked pixel and extends down-right, so the
    # box anchored there is no longer the blank page's uniform white...
    under_cursor = image.crop((x, y, x + 24, y + 32)).getcolors()

    assert under_cursor is None or len(under_cursor) > 1

    # ...while the rest of the frame is untouched.
    assert image.crop((0, 0, x, y)).getcolors() == [(x * y, (255, 255, 255))]


@pytest.mark.asyncio
async def test_captcha_cursor_false_returns_raw_capture():
    """``cursor=False`` (the env-owned make kwarg) disables the repaint."""
    env, raw = _cursor_env(cursor=False)

    assert await env._take_screenshot() == raw


@pytest.mark.asyncio
async def test_captcha_cursor_false_issues_no_mouse_events_at_reset():
    """``cursor=False`` keeps the harness OFF the pointer: ZERO Playwright mouse
    events, not merely an unpainted sprite.

    ``reset()`` calls ``_reset_cursor`` unconditionally, so the guard has to live
    in ``_reset_cursor`` itself -- the shape ``waa`` already uses (its
    ``_park_cursor`` call is wrapped in ``if self._cursor``). On a CAPTCHA
    benchmark a stray real mousemove is not cosmetic: pointer telemetry is
    exactly what anti-bot heuristics read, so an operator who opted out must not
    have one delivered to Chromium anyway. The tracked value is still reset, so
    nothing downstream sees a stale coordinate.
    """
    env, _ = _cursor_env(cursor=False, resolution=(800, 600))
    env._cursor_x, env._cursor_y = 7, 9

    await env._reset_cursor()

    assert env._page.mouse.moves == []
    assert (env._cursor_x, env._cursor_y) == (400, 300)
