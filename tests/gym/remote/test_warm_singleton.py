"""Unit tests for the background warm-singleton hook.

Covers the load-bearing logic WITHOUT docker: retry-on-CapacityExhausted,
degrade-don't-die, selecting only served SINGLETON envs, skip blocked / empty
allow-list. The real ensure/health are mocked.

Run: uv run pytest tests/gym/remote/test_warm_singleton.py -q
"""

from __future__ import annotations

import sys
import types

import pytest

import lite.gym.remote.server as srv
from lite.gym.errors import CapacityExhausted, EnvDepsMissingError

registry = sys.modules["lite.gym.registry"]
services = sys.modules["lite.gym.services"]


# --------------------------------------------------------------------------- #
# _warm_one_singleton — retry / degrade
# --------------------------------------------------------------------------- #


async def test_warm_one_success_first_try():
    calls = []

    def ensure(leaf):
        calls.append(("ensure", leaf))

    def health(leaf):
        calls.append(("health", leaf))

    await srv._warm_one_singleton("webgym", ensure, health)
    assert calls == [("ensure", "webgym"), ("health", "webgym")]


async def test_warm_one_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(srv, "_WARM_SINGLETON_RETRY_S", 0.0)
    monkeypatch.setattr(srv, "_WARM_SINGLETON_DEADLINE_S", 1000.0)
    n = {"i": 0}

    def ensure(leaf):
        n["i"] += 1
        if n["i"] < 3:
            raise CapacityExhausted(what="warming", retry_after_s=0)

    def health(leaf):
        pass

    await srv._warm_one_singleton("browsergym.webarena", ensure, health)
    assert n["i"] == 3  # two warming raises, third succeeds


async def test_warm_one_degrades_on_deps_missing():
    def ensure(leaf):
        raise EnvDepsMissingError(what="no image", install="x", see="y")

    def health(leaf):
        raise AssertionError("should not reach health")

    await srv._warm_one_singleton("browsergym.visualwebarena", ensure, health)  # no raise


async def test_warm_one_degrades_at_deadline(monkeypatch):
    monkeypatch.setattr(srv, "_WARM_SINGLETON_DEADLINE_S", 0.0)  # immediate deadline
    monkeypatch.setattr(srv, "_WARM_SINGLETON_RETRY_S", 0.0)

    def ensure(leaf):
        raise CapacityExhausted(what="still warming", retry_after_s=0)

    def health(leaf):
        pass

    await srv._warm_one_singleton("webgym", ensure, health)  # no raise → degrade


async def test_warm_one_degrades_on_unexpected_error():
    def ensure(leaf):
        raise RuntimeError("docker down")

    def health(leaf):
        pass

    await srv._warm_one_singleton("mobilegym", ensure, health)  # no raise


# --------------------------------------------------------------------------- #
# _warm_singletons — selects only served SINGLETON envs
# --------------------------------------------------------------------------- #


async def test_warm_singletons_selects_only_singleton(monkeypatch):
    # Real family declarations: webgym=SINGLETON, captcha=DEDICATED, osworld=REMOTE.
    import importlib

    for e in ("webgym", "captcha", "osworld"):
        try:
            importlib.import_module("lite.gym.envs." + ".".join(e.split(".")) + ".main")
        except EnvDepsMissingError as exc:
            pytest.skip(f"{e} env deps unavailable (run its install.sh): {exc}")

    warmed = []
    monkeypatch.setattr(registry, "ensure_registered", lambda e: None)
    monkeypatch.setattr(registry, "ensure_services", lambda e: warmed.append(e))
    monkeypatch.setattr(services, "health_check", lambda e: None)

    state = types.SimpleNamespace(
        allowed_env_ids={"webgym", "captcha", "osworld"}, blocked_env_ids=frozenset()
    )
    await srv._warm_singletons(state)
    assert warmed == ["webgym"]  # captcha (DEDICATED) + osworld (REMOTE) skipped


async def test_warm_singletons_empty_allowed_warms_nothing(monkeypatch):
    # Empty allow-list = serve nothing → warm nothing (NOT fall through to all).
    warmed = []
    monkeypatch.setattr(registry, "ensure_registered", lambda e: None)
    monkeypatch.setattr(registry, "ensure_services", lambda e: warmed.append(e))
    monkeypatch.setattr(services, "health_check", lambda e: None)
    state = types.SimpleNamespace(allowed_env_ids=set(), blocked_env_ids=frozenset())
    await srv._warm_singletons(state)
    assert warmed == []


async def test_warm_singletons_skips_blocked(monkeypatch):
    # A blocked SINGLETON must not be warmed (POST refuses it anyway).
    import importlib

    importlib.import_module("lite.gym.envs.webgym.main")
    warmed = []
    monkeypatch.setattr(registry, "ensure_registered", lambda e: None)
    monkeypatch.setattr(registry, "ensure_services", lambda e: warmed.append(e))
    monkeypatch.setattr(services, "health_check", lambda e: None)
    state = types.SimpleNamespace(allowed_env_ids={"webgym"}, blocked_env_ids=frozenset({"webgym"}))
    await srv._warm_singletons(state)
    assert warmed == []


async def test_warm_singletons_expands_umbrella_to_leaves(monkeypatch):
    """A bare umbrella SINGLETON (e.g. browsergym) must expand to its REGISTERED LEAVES —
    warming the bare umbrella id boots nothing (_benchmark_of→None) yet would report hot.
    Review round-1 gap A1: the existing tests only hit leaf-less webgym, so the
    umbrella→leaf expansion + bare-umbrella guard had zero coverage."""
    import lite.gym as _g
    from lite.gym import services as _svc

    warmed = []
    # "uenv" is an EAGER (SINGLETON) umbrella with two registered leaves.
    monkeypatch.setattr(
        _svc,
        "warm_strategy",
        lambda e: _svc.WarmStrategy.EAGER if e == "uenv" else _svc.WarmStrategy.NONE,
    )
    # _warm_singletons asks only for the umbrella's own children, so that
    # warming one scoped singleton does not register every other env's catalog.
    monkeypatch.setattr(
        _g.registry, "sub_env_ids", lambda e: ["uenv.alpha", "uenv.beta"] if e == "uenv" else []
    )
    monkeypatch.setattr(registry, "ensure_registered", lambda e: None)
    monkeypatch.setattr(registry, "ensure_services", lambda e: warmed.append(e))
    monkeypatch.setattr(services, "health_check", lambda e: None)
    state = types.SimpleNamespace(allowed_env_ids={"uenv"}, blocked_env_ids=frozenset())
    await srv._warm_singletons(state)
    # Warmed the two leaves, NEVER the bare umbrella "uenv".
    assert warmed == ["uenv.alpha", "uenv.beta"]


async def test_warm_singletons_umbrella_skips_blocked_leaf(monkeypatch):
    """Umbrella expansion still honors blocked_env_ids per-leaf."""
    import lite.gym as _g
    from lite.gym import services as _svc

    warmed = []
    monkeypatch.setattr(
        _svc,
        "warm_strategy",
        lambda e: _svc.WarmStrategy.EAGER if e == "uenv" else _svc.WarmStrategy.NONE,
    )
    # _warm_singletons asks only for the umbrella's own children, so that
    # warming one scoped singleton does not register every other env's catalog.
    monkeypatch.setattr(
        _g.registry, "sub_env_ids", lambda e: ["uenv.alpha", "uenv.beta"] if e == "uenv" else []
    )
    monkeypatch.setattr(registry, "ensure_registered", lambda e: None)
    monkeypatch.setattr(registry, "ensure_services", lambda e: warmed.append(e))
    monkeypatch.setattr(services, "health_check", lambda e: None)
    state = types.SimpleNamespace(
        allowed_env_ids={"uenv"}, blocked_env_ids=frozenset({"uenv.beta"})
    )
    await srv._warm_singletons(state)
    assert warmed == ["uenv.alpha"]  # beta blocked
