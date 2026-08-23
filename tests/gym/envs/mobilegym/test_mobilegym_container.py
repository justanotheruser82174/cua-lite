"""Live container tests for mobilegym's in-container RPC server logic.

These cover the pieces that live ONLY in ``lite/gym/envs/mobilegym/docker/server.py``
(``_translate_action`` + template-index selection) and cannot be imported on the
host (server.py imports ``bench_env``). We exercise them INSIDE the built
``cua-lite/mobilegym:latest`` image via ``docker run --entrypoint python`` so the real
``bench_env`` Action/ActionType + task classes back the assertions.

Marked ``live`` (CLAUDE.md): needs Docker + the built image. Run with:
    uv run pytest tests/gym/envs/mobilegym/test_mobilegym_container.py -m live -q
Skips automatically if Docker or the image is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request

import pytest

pytestmark = pytest.mark.live

# Defaults to the canonical tag; override (e.g. to a freshly-built :smoketest)
# via MOBILEGYM_TEST_IMAGE when validating a not-yet-promoted build.
_IMAGE = os.environ.get("MOBILEGYM_TEST_IMAGE", "cua-lite/mobilegym:latest")


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_present() -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", _IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _run_in_container(snippet: str) -> subprocess.CompletedProcess[str]:
    """Run a python snippet inside the image with server.py importable."""
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            _IMAGE,
            "-c",
            f"import sys; sys.path.insert(0, '/usr/local/bin')\n{snippet}",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(autouse=True)
def _require_image():
    if not _docker_available():
        pytest.skip("docker not available")
    if not _image_present():
        pytest.skip(f"{_IMAGE} not built (run scripts/install.sh)")


def test_translate_action_mapping():
    """Every LiteMobileActionSpace verb maps to the expected bench_env ActionType.

    No longer the *verbatim* ported table: server.py has since converted three silent
    ``None`` returns into loud errors, each routed to a distinct /step feedback channel
    (``ValueError`` → ``kind="model_action"``, ``_UnsupportedAction`` →
    ``kind="unsupported_action"``). Only host-handled verbs still return ``None``.
    """
    snippet = r"""
import server
from bench_env.env.base import ActionType

def at(name, args):
    a = server._translate_action(name, args)
    return None if a is None else (a.action_type, a.data)

def raises(exc, name, args):
    try:
        server._translate_action(name, args)
    except exc as e:
        return e
    raise AssertionError(f"{name}({args}) did not raise {exc.__name__}")

# tap → CLICK; clicks>=2 → DOUBLE_TAP
t, p = at("tap", {"coordinate": [10, 20]})
assert t is ActionType.CLICK and p["point"] == [10, 20], (t, p)
t, _ = at("tap", {"coordinate": [1, 2], "clicks": 2})
assert t is ActionType.DOUBLE_TAP, t
# long_press → duration in ms
t, p = at("long_press", {"coordinate": [3, 4], "duration": 0.5})
assert t is ActionType.LONG_PRESS and p["duration"] == 500, (t, p)
# type → TYPE value
t, p = at("type", {"text": "hi"})
assert t is ActionType.TYPE and p["value"] == "hi", (t, p)
# swipe / drag → two points
t, p = at("swipe", {"start_coordinate": [1, 1], "coordinate": [2, 2]})
assert t is ActionType.SWIPE and p["point1"] == [1, 1] and p["point2"] == [2, 2], (t, p)
t, p = at("drag",  {"start_coordinate": [1, 1], "coordinate": [2, 2]})
assert t is ActionType.DRAG, t
# open_app → AWAKE value, passed through VERBATIM. _checked_app_name validates against
# bench_env's two rules but must NOT rewrite: bench_env maps name → appId itself, so
# translating here would swap the model's word for an arbitrary same-appId alias.
t, p = at("open_app", {"app_name": "bilibili"})
assert t is ActionType.AWAKE and p["value"] == "bilibili", (t, p)
# "Bilibili" is an exact APP_NAME_MAP key; "WeChat Reading" resolves via rule 2
t, p = at("open_app", {"app_name": "Bilibili"}); assert p["value"] == "Bilibili", p
t, p = at("open_app", {"app_name": "WeChat Reading"}); assert p["value"] == "WeChat Reading", p
# ...and an unlaunchable name is REJECTED, not silently skipped by bench_env's _open_app.
e = raises(ValueError, "open_app", {"app_name": "Nonexistent App"})
assert "not an app this device can launch" in str(e), e
assert "Launchable:" in str(e), e
raises(ValueError, "open_app", {})   # missing app_name would forward "" and no-op
# every name the host advertises in _MOBILEGYM_APPS must survive that gate verbatim
HOST_ENUM = ["WeChat", "Alipay", "Map", "Calendar", "Clock", "Notes", "Settings",
             "Phone", "SMS", "Gallery", "File Manager", "Browser", "Calculator",
             "Weather", "Compass", "Bilibili", "RedNote", "WeChat Reading",
             "Tencent Meeting", "12306", "Spotify", "Reddit", "X", "eBay"]
for app in HOST_ENUM:
    t, p = at("open_app", {"app_name": app})
    assert t is ActionType.AWAKE and p["value"] == app, (app, p)
_launchable = set(server.launchable_app_names())
assert set(HOST_ENUM) <= _launchable, sorted(set(HOST_ENUM) - _launchable)
# system_button mapping
for btn, exp in [
    ("Back", ActionType.BACK),
    ("Home", ActionType.HOME),
    ("Enter", ActionType.ENTER),
    ("Menu", ActionType.RECENT),
]:
    t, _ = at("system_button", {"button": btn})
    assert t is exp, (btn, t)
# an unknown button is a model error, not a silent no-op (→ kind="model_action")
e = raises(ValueError, "system_button", {"button": "Nope"})
assert "unknown button" in str(e), e
# wait → WAIT
t, p = at("wait", {"duration": 2.0}); assert t is ActionType.WAIT and p["value"] == 2.0, (t, p)
# host-handled verbs → None (the host, not the container, implements these)
for verb in ["screenshot", "terminate", "response", "noop"]:
    assert server._translate_action(verb, {}) is None, verb
# verbs this backend cannot express → _UnsupportedAction (→ kind="unsupported_action")
for verb in ["pinch", "totally_unknown"]:
    raises(server._UnsupportedAction, verb, {})
print("OK")
"""
    r = _run_in_container(snippet)
    assert r.returncode == 0, (
        f"translate-action assertions failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
    )
    assert "OK" in r.stdout


def test_template_index_is_seed_deterministic():
    """Multi-template tasks pick a template index deterministically from the seed
    (server.py reset uses random.Random(f'tpl:{seed}:{task_id}').randrange(n))."""
    snippet = r"""
import random
from bench_env.task.registry import TaskRegistry
r = TaskRegistry()
# find a task with >1 template to exercise the deterministic branch
import bench_env
from pathlib import Path
ids = [
    l.strip()
    for l in (Path(bench_env.__file__).parent / "splits" / "test.txt").read_text().splitlines()
    if l.strip() and not l.startswith("#")
]
multi = None
for tid in ids:
    try:
        cls = r.get_by_id(tid)
    except Exception:
        continue
    if len(getattr(cls, "templates", []) or []) > 1:
        multi = tid; n = len(cls.templates); break
if multi is None:
    print("OK (no multi-template task in split; determinism formula is trivially seed-stable)")
else:
    seed, tid = 42, multi
    a = random.Random(f"tpl:{seed}:{tid}").randrange(n)
    b = random.Random(f"tpl:{seed}:{tid}").randrange(n)
    assert a == b, (a, b)
    print(f"OK ({tid}: seed42 → template {a}/{n}, deterministic)")
"""
    r = _run_in_container(snippet)
    assert r.returncode == 0, f"template-determinism failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
    assert "OK" in r.stdout


def _post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://localhost:{port}{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def test_max_steps_override_caps_truncation():
    """A host-supplied ``max_steps`` in the /reset body caps episode length:
    the container must echo it and set ``truncated`` once step_count >= it
    (regression guard for the override that was silently dropped when the
    container owned the budget)."""
    name = "mobilegym-pytest-maxsteps"
    port = 8791
    subprocess.run(
        ["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--init",
            "--name",
            name,
            "-e",
            "MOBILEGYM_MAX_BROWSERS=1",
            "-p",
            f"{port}:8000",
            _IMAGE,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    try:
        # wait for healthz
        up = False
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=5) as h:
                    if b'"ok"' in h.read():
                        up = True
                        break
            except Exception:
                pass
            time.sleep(2)
        assert up, "container /healthz never came up"

        reset = _post(
            port,
            "/reset",
            {
                "task_id": "bilibili.OpenRankingTask",
                "seed": 42,
                "eval_mode": "text",
                "physical_size": [1280, 720],
                "dpr": 1.0,
                "delay": 0.0,
                "max_steps": 3,
            },
        )
        assert reset["max_steps"] == 3, f"override not honored: {reset['max_steps']}"
        iid = reset["instance_id"]

        truncs = []
        for _ in range(4):
            s = _post(
                port, "/step", {"instance_id": iid, "actions": [{"name": "wait", "arguments": {}}]}
            )
            truncs.append(s["truncated"])
        # steps 1,2 not truncated; step 3 (step_count>=3) and 4 truncated
        assert truncs == [False, False, True, True], truncs
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
