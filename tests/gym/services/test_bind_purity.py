"""Bind contract + purity guard for EnvServerPoolable-style envs.

Server warm-pool taskless claims are retired, but direct construction still
runs ``bind()`` after constructor-state setup and recycle-capable envs rely on bind as
cheap in-memory state application. Every constructible legacy poolable env's
``bind`` runs with socket / subprocess / urllib / requests entry points
booby-trapped, and must neither fire them nor take longer than a few ms.

``test_bind_callable_with_zero_positional_args`` is the signature half of the
same contract (ported here when the warm-pool contract file was deleted in the
baked_subset/warm-pool removal — that assertion never touched warm-pool code).

Run: uv run pytest tests/gym/services/test_bind_purity.py
"""
from __future__ import annotations

import inspect
import socket
import subprocess
import time
import urllib.request

import pytest
import requests  # imported BEFORE the booby-trap (its import chain touches socket)

from lite.gym.errors import EnvDepsMissingError

#: Legacy EnvServerPoolable env_ids. Constructed via direct-mode gym.make
#: (which injects framework baked kwargs and runs bind once); the explicit
#: bind below remains a cheap-state guard. Deps-gated envs skip.
_POOLABLES = ["androidworld", "androidlab", "mobileworld"]


def _booby_trap(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("bind() performed IPC — it must be pure/in-memory")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "get", boom)


@pytest.mark.parametrize("env_id", _POOLABLES)
def test_bind_is_pure_and_fast(monkeypatch, env_id):
    import lite.gym as gym

    try:
        splits = gym.registry.task_ids(env_id)
        task_id = next(tid for ids in splits.values() for tid in ids)
        env = gym.make(f"{env_id}@{task_id}")
    except EnvDepsMissingError as e:
        pytest.skip(f"{env_id} deps missing on this host: {e}")

    _booby_trap(monkeypatch)
    t0 = time.perf_counter()
    env.unwrapped.bind()
    dt = time.perf_counter() - t0
    assert dt < 0.05, f"bind() took {dt*1000:.1f} ms"


def test_mobileworld_bind_is_pure_and_fast(monkeypatch):
    """mobileworld constructs without its image (deps checked at ensure/boot,
    not __init__) — assert its bind purity even on image-less hosts."""
    from lite.gym.envs.mobileworld.main import MobileWorldEnv

    env = MobileWorldEnv(task_id="AcceptMeetingTask")
    _booby_trap(monkeypatch)
    t0 = time.perf_counter()
    env.bind()
    dt = time.perf_counter() - t0
    assert dt < 0.05, f"bind() took {dt*1000:.1f} ms"


def test_sandbox_bind_is_pure_and_fast(monkeypatch):
    """SandboxBaseEnv (lite.osworld / lite.demo family) — same guard."""
    from lite.gym.sandbox.base import SandboxBaseEnv

    env = SandboxBaseEnv()
    _booby_trap(monkeypatch)
    t0 = time.perf_counter()
    env.bind()
    dt = time.perf_counter() - t0
    assert dt < 0.05, f"bind() took {dt*1000:.1f} ms"


# ---------------------------------------------------------------------------
# bind() signature contract
# ---------------------------------------------------------------------------

#: The canonical env modules whose ``bind`` is under the signature contract,
#: plus one representative cuaworld leaf. The parametrize row count is pinned to
#: this list's length below.
_POOLABLE_TARGETS = [
    ("lite.gym.envs.lite.osworld.main", "LiteOsworldEnv"),
    ("lite.gym.envs.lite.cuagym.main", "_CuaGymEnv"),
    ("lite.gym.envs.lite.scalecua.main", "LiteScaleCuaEnv"),
    ("lite.gym.envs.androidworld.main", "AndroidWorldEnv"),
    ("lite.gym.envs.androidlab.main", "AndroidLabEnv"),
    ("lite.gym.envs.mobileworld.main", "MobileWorldEnv"),
]


def _poolable_classes() -> list[type]:
    """Concrete EnvServerPoolable subclasses discoverable by importing the
    canonical env modules, plus one representative cuaworld leaf.

    Tolerant of a MISSING dependency only. The except used to be bare
    ``Exception``, which silently deleted a parametrize row for any import-time
    error at all: a ``TypeError`` in an env module took ``6 passed`` to
    ``5 passed`` with no skip and no warning, and the ``assert subs`` floor was
    satisfied by 1 of 6. The row count is now pinned to the declared list, so a
    vanished env is a failure rather than a smaller run.
    """
    subs: list[type] = []
    for mod_path, cls_name in _POOLABLE_TARGETS:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
        except (ImportError, ModuleNotFoundError):
            continue  # module's deps not installed in this test env
        subs.append(getattr(mod, cls_name))
    assert len(subs) == len(_POOLABLE_TARGETS), (
        f"only {len(subs)} of {len(_POOLABLE_TARGETS)} poolable env modules "
        f"imported: {sorted({m for m, _ in _POOLABLE_TARGETS} - {c.__module__ for c in subs})}"
    )
    return subs


@pytest.mark.parametrize("cls", _poolable_classes(), ids=lambda c: c.__name__)
def test_bind_callable_with_zero_positional_args(cls: type) -> None:
    """Framework's ``__init__`` partition always invokes ``bind`` once, even
    with empty kwargs (so soft-field defaults populate uniformly). Each
    subclass's ``bind`` must therefore accept a zero-arg call (``task=None`` or
    ``task_id=""`` default) — otherwise direct-mode
    ``EnvCls(only_baked_kwargs=...)`` would TypeError."""
    sig = inspect.signature(cls.bind)
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.default is inspect.Parameter.empty and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            pytest.fail(
                f"{cls.__name__}.bind has required parameter "
                f"{name!r} (kind={p.kind.name}). Framework calls "
                f"bind() with no args during construction — "
                f"every param must have a default."
            )
