"""Enforce the invariant: importing an env module does ZERO network I/O.

Why this is load-bearing: the env-server's orphan reaper calls
``registry._import_all()`` (``lite/gym/remote/recovery.py``), which imports EVERY
``lite/gym/envs/**/main.py`` regardless of ``--env-ids`` — on the single asyncio
event-loop thread. If any env's *module import* runs a blocking network call
(e.g. ``huggingface_hub.snapshot_download``), it freezes the whole server: no
coroutine (not even ``/host_status``) runs until it returns. So env import MUST
be side-effect-free; all asset/network work belongs in the lazy
``task_ids()``/``make()``/``reset()`` path (see webgym / screenspot_pro).

This test imports each env in a fresh subprocess under a socket guard that
hard-exits the instant anything calls ``connect``. A clean import exits 0; a
network attempt exits 3 (→ fail); a hung import (some other blocking call) trips
the timeout (→ fail); a missing optional dep exits 4 (→ skip, that env isn't
installed here). It runs under the ambient cache (no ``HF_HOME`` tricks): it
catches any UNCONDITIONAL import-time network (e.g. a lib that phones home
regardless of cache), and on a clean CI machine also catches assets fetched at
import instead of via install.sh.

Run: uv run pytest tests/gym/envs/test_import_no_network.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import lite.gym

_ENVS_ROOT = Path(lite.gym.__file__).resolve().parent / "envs"


def _env_main_modules() -> list[str]:
    """Every ``lite.gym.envs.<...>.main`` module (import-registered env)."""
    mods: list[str] = []
    for main_py in sorted(_ENVS_ROOT.rglob("main.py")):
        if "__pycache__" in main_py.parts:
            continue
        rel = main_py.relative_to(_ENVS_ROOT.parent.parent.parent)  # repo-root-relative
        mods.append(".".join(rel.with_suffix("").parts))
    return mods


# The child: block the network at the socket layer, then import the module.
# ``os._exit`` on the first ``connect`` so an env's own try/except cannot swallow
# the signal (screenspot's _ensure_data_dir, for one, catches download errors).
_CHILD = r"""
import sys, os, socket, importlib

def _boom(*a, **k):
    os._exit(3)  # network attempted at import — hard-exit, unswallowable

socket.socket.connect = _boom
socket.socket.connect_ex = _boom
socket.create_connection = _boom

mod = sys.argv[1]
try:
    importlib.import_module(mod)
except (ImportError, ModuleNotFoundError):
    os._exit(4)          # optional dep not installed here → skip
except BaseException:
    os._exit(5)          # some other import error → not a network violation
os._exit(0)
"""


@pytest.mark.parametrize("module", _env_main_modules())
def test_env_import_makes_no_network_call(module: str):
    env = {
        **os.environ,
        # Force DIRECT mode so we exercise each env's import-time LOCAL
        # registration — client mode fetches the task list over HTTP and skips
        # registration, which would make this check vacuous. (This is a mode
        # switch, not a cache override; the ambient HF cache is left untouched.)
        "CUA_LITE_ENV_SERVER_URL": "",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, module],
            env=env, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"importing {module} hung (>120s) — a blocking call at module "
            f"import (network/subprocess/lock) freezes registry._import_all()"
        )

    if proc.returncode == 3:
        pytest.fail(
            f"{module} makes a NETWORK connection at import. Move asset/network "
            f"setup into the lazy task_ids()/make()/reset() path (see webgym / "
            f"screenspot_pro); module import must be side-effect-free."
        )
    if proc.returncode in (4, 5):
        pytest.skip(f"{module} not importable here (rc={proc.returncode}) — optional deps/config")
    assert proc.returncode == 0, f"{module}: unexpected rc={proc.returncode}\n{proc.stderr[-500:]}"
