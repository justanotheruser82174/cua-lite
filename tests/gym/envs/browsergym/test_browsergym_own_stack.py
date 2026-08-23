"""BrowserGym WebArena own-stack adoption tests."""

from __future__ import annotations

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

class TestWaAdoptOwnStack:
    """section 89af1f42 adopt-vs-auto-pick + the R1 homepage tightening — the
    regression net the WA/VWA audit flagged as missing.

    Contract of ``_default_held_by_own_stack(row, port)``:
      * container rows: adopt IFF ``docker ps --filter publish=<port>`` names
        ``<container_base>-<scope>`` (own-scope evidence); foreign name or
        probe error → False (auto-pick).
      * homepage row (host Flask, no container): adopt ONLY in direct mode
        (scope "default") AND only when the body carries start.sh's
        "WebArena" needle — a bare 200 could be any foreign service, and a
        sibling env-server's homepage must never be adopted (its shutdown
        would pkill ours and vice versa).
    """

    def _fn(self):
        from lite.gym.envs.browsergym.main import _default_held_by_own_stack
        return _default_held_by_own_stack

    def _rediscover(self):
        from lite.gym.envs.browsergym.main import _own_stack_published_port
        return _own_stack_published_port

    def test_rediscovery_adopts_own_stack_on_any_port(self, monkeypatch):
        """I1: `docker port <base>-<scope>` finds the stack wherever a prior
        run's auto-pick landed it — not just on the default port."""
        import lite.gym.envs.browsergym.main as bg
        monkeypatch.setattr(bg, "_resolve_env_server_port", lambda: "default")

        class _R:
            returncode = 0
            stdout = "80/tcp -> 0.0.0.0:8503\n80/tcp -> [::]:8503\n"
        monkeypatch.setattr(bg.subprocess, "run", lambda *a, **k: _R())
        assert self._rediscover()({"container_base": "shopping"}) == 8503

    def test_rediscovery_none_when_not_running(self, monkeypatch):
        import lite.gym.envs.browsergym.main as bg

        class _R:
            returncode = 1   # docker port: no such container
            stdout = ""
        monkeypatch.setattr(bg.subprocess, "run", lambda *a, **k: _R())
        assert self._rediscover()({"container_base": "shopping"}) is None

    def test_rediscovery_fails_closed_on_probe_error(self, monkeypatch):
        import lite.gym.envs.browsergym.main as bg
        def _boom(*a, **k):
            raise OSError("docker unavailable")
        monkeypatch.setattr(bg.subprocess, "run", _boom)
        assert self._rediscover()({"container_base": "shopping"}) is None
        assert self._rediscover()({"probe_adopt": True}) is None, "homepage row has no container"

    def test_homepage_never_adopts_under_env_server_scope(self, monkeypatch):
        import lite.gym.envs.browsergym.main as bg
        monkeypatch.setattr(bg, "_resolve_env_server_port", lambda: "30456")
        # No HTTP probe may even be attempted — a sibling server's homepage
        # is indistinguishable from ours.
        import urllib.request
        def _boom(*a, **k):
            raise AssertionError("must not probe under an env-server scope")
        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        assert self._fn()({"probe_adopt": True}, 4399) is False

    def test_homepage_direct_mode_requires_webarena_needle(self, monkeypatch):
        import urllib.request

        import lite.gym.envs.browsergym.main as bg
        monkeypatch.setattr(bg, "_resolve_env_server_port", lambda: "default")

        class _Resp:
            status = 200
            def __init__(self, body): self._body = body
            def read(self, n=-1): return self._body
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: _Resp(b"<h1>Some foreign dashboard</h1>"))
        assert self._fn()({"probe_adopt": True}, 4399) is False, "bare 200 must not adopt"

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **k: _Resp(b"<title>WebArena homepage</title>"))
        assert self._fn()({"probe_adopt": True}, 4399) is True

    def test_homepage_fails_closed_when_probe_errors(self, monkeypatch):
        import urllib.request

        import lite.gym.envs.browsergym.main as bg
        monkeypatch.setattr(bg, "_resolve_env_server_port", lambda: "default")
        def _boom(*a, **k):
            raise ConnectionRefusedError()
        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        assert self._fn()({"probe_adopt": True}, 4399) is False
