"""Unit tests for the lock-free per-thread Playwright routing in browsergym
(``_ThreadLocalPlaywright`` / ``_install_pw_proxy`` / ``_pw_start_lock``).

This is the concurrency-critical change that removed BrowserGym's global op-lock so
concurrent envs' browser steps run in parallel (the ~24x env.step win).
The auditor flagged it had zero coverage; these tests pin the three invariants that
make it correct, with a FAKE ``sync_playwright`` (no real Playwright / docker):

  1. per-thread routing — each thread gets its OWN Playwright, started once;
  2. the narrow ``_pw_start_lock`` serializes the subprocess-spawning ``start()``
     across threads (the global lock's one real job) while leaving per-step ops free;
  3. the proxy reads truthy so BrowserGym's ``if not _PLAYWRIGHT`` never overwrites it,
     and ``_install_pw_proxy`` is idempotent.

Run: uv run pytest tests/gym/envs/browsergym/test_browsergym_playwright.py -q
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")
from lite.gym.envs.browsergym import main as bg


@pytest.fixture(autouse=True)
def _clean_main_thread_pw():
    """Drop any Playwright the main (pytest) thread cached, before and after — the
    module-level ``_thread_local`` outlives a single test."""
    if hasattr(bg._thread_local, "pw"):
        del bg._thread_local.pw
    yield
    if hasattr(bg._thread_local, "pw"):
        del bg._thread_local.pw


class _FakePW:
    def __init__(self, tag: int) -> None:
        self.tag = tag


def _fake_sync_playwright(monkeypatch, on_start=None):
    """Install a fake ``playwright.sync_api.sync_playwright`` whose ``.start()``
    returns a fresh tagged object each call. Returns a counter dict (``{"n": k}``)
    of how many times ``start()`` ran."""
    import playwright.sync_api as psa

    counter = {"n": 0}

    class _Ctx:
        def start(self):
            if on_start is not None:
                on_start()
            counter["n"] += 1
            return _FakePW(counter["n"])

    monkeypatch.setattr(psa, "sync_playwright", lambda: _Ctx())
    return counter


def test_per_thread_routing_starts_once_per_thread(monkeypatch):
    counter = _fake_sync_playwright(monkeypatch)
    N = 6
    tags: dict[int, int] = {}
    barrier = threading.Barrier(N)

    def worker(i):
        barrier.wait()
        tags[i] = bg._pw_proxy.tag          # first access → start() on this thread
        again = bg._pw_proxy.tag            # second access → cached, no new start()
        assert again == tags[i]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["n"] == N, "start() must run exactly once per thread"
    assert len(set(tags.values())) == N, "each thread must get its OWN Playwright"


def test_start_lock_serializes_subprocess_spawn(monkeypatch):
    """The narrow ``_pw_start_lock`` must prevent concurrent ``start()`` (the
    node-driver subprocess spawn) from overlapping — the deleted global lock's one
    load-bearing job (uvloop 'Racing with another loop to spawn a process')."""
    state = {"cur": 0, "max": 0}
    track_lock = threading.Lock()

    def on_start():
        with track_lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.02)                    # hold inside start() to expose overlap
        with track_lock:
            state["cur"] -= 1

    _fake_sync_playwright(monkeypatch, on_start=on_start)
    N = 8
    barrier = threading.Barrier(N)

    def worker():
        barrier.wait()
        _ = bg._pw_proxy.tag

    threads = [threading.Thread(target=worker) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max"] == 1, (
        f"concurrent start() overlapped (max={state['max']}) — _pw_start_lock not "
        "guarding the spawn"
    )


def test_proxy_reads_truthy():
    # dunder lookups bypass __getattr__ → no spurious start(); proxy must be truthy
    # so browsergym's `_get_global_playwright` `if not _PLAYWRIGHT` won't replace it.
    assert bool(bg._pw_proxy) is True
    assert (not bg._pw_proxy) is False


def test_install_pw_proxy_idempotent():
    import browsergym.core as bgym_core

    bg._install_pw_proxy()
    assert bgym_core._PLAYWRIGHT is bg._pw_proxy
    bg._install_pw_proxy()                  # second call must not replace
    assert bgym_core._PLAYWRIGHT is bg._pw_proxy
