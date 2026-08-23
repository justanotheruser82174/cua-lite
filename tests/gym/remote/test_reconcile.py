"""Unit tests for the framework reconcile loop + ServerScope.

Pure — no docker / live env-server. Run: ``uv run pytest
tests/gym/remote/test_reconcile.py``.
"""

from __future__ import annotations

from lite.gym.remote import reconcile as reconcile_mod
from lite.gym.remote.reconcile import collect_valid_ghosts, reconcile, recover
from lite.gym.remote.scope import ServerScope
from lite.gym.services import EnvServices
from lite.gym.utils.config.identity import EnvIdentity


class FakeView:
    """A reconcile view that returns a fixed live set and records calls.

    ``live`` of ``None`` models the fail-closed path (skip this cycle)."""

    def __init__(self, live: set[str] | None, reap_return: int = 0):
        self._live = live
        self._reap_return = reap_return
        self.live_calls: list[tuple] = []
        self.reap_calls: list[tuple] = []

    def live_ids(self, env_id, scope):
        self.live_calls.append((env_id, scope))
        return self._live

    def reap(self, env_id, scope, in_use, *, boot=False):
        self.reap_calls.append((env_id, scope, frozenset(in_use), boot))
        return self._reap_return


SCOPE = ServerScope(server_port=30100, token_hash="abc123")


# --------------------------------------------------------------------------- #
# ServerScope
# --------------------------------------------------------------------------- #

def test_server_scope_from_server_strict_hashes_token():
    s = ServerScope.from_server(server_port=30100, strict_token="secret")
    assert s.server_port == 30100
    # sha256("secret")[:6]
    import hashlib
    assert s.token_hash == hashlib.sha256(b"secret").hexdigest()[:6]


def test_server_scope_from_server_passthrough_no_token():
    s = ServerScope.from_server(server_port=30100, strict_token=None)
    assert s.server_port == 30100
    assert s.token_hash is None


def test_server_scope_is_frozen_sibling_of_env_identity():
    # Distinct types; passthrough token_hash differs in provenance.
    scope = ServerScope.from_server(server_port=1, strict_token=None)
    ident = EnvIdentity(session_id="s", token_hash="caller", server_port=1)
    assert scope.token_hash is None and ident.token_hash == "caller"
    assert type(scope) is not type(ident)


# --------------------------------------------------------------------------- #
# reconcile: ghost / orphan set algebra
# --------------------------------------------------------------------------- #

def test_ghost_when_tracked_resource_not_live():
    # inst i1 tracks "r-dead" which is not in the live set → ghost (old enough).
    view = FakeView(live={"r-live"})
    tracked = {"i1": ("r-dead", 0.0), "i2": ("r-live", 0.0)}
    rep = reconcile(view, "e", SCOPE, tracked, safety_margin_s=10, now=1000.0)
    assert rep.ghost_ids == ("i1",)
    # i2's resource is live → it is in_use, env told not to reap it.
    assert view.reap_calls[0][2] == frozenset({"r-live"})


def test_no_ghost_for_young_instance():
    # Resource gone but instance younger than safety margin → NOT ghosted.
    view = FakeView(live=set())
    tracked = {"i1": ("r1", 995.0)}  # age = 1000-995 = 5 < margin 10
    rep = reconcile(view, "e", SCOPE, tracked, safety_margin_s=10, now=1000.0)
    assert rep.ghost_ids == ()


def test_orphan_count_is_env_reap_return():
    view = FakeView(live={"r1"}, reap_return=3)
    rep = reconcile(view, "e", SCOPE, {}, now=1000.0)
    assert rep.orphans_reaped == 3


def test_in_use_excludes_dead_and_none_external_ids():
    view = FakeView(live={"r1", "r2"})
    tracked = {
        "i1": ("r1", 0.0),     # live → in_use
        "i2": ("r-dead", 0.0), # gone → ghost, NOT in_use
        "i3": (None, 0.0),     # mid-boot, no ext id → NOT in_use, NOT ghost
    }
    reconcile(view, "e", SCOPE, tracked, safety_margin_s=10, now=1000.0)
    assert view.reap_calls[0][2] == frozenset({"r1"})


def test_none_external_id_is_never_ghosted():
    view = FakeView(live=set())
    tracked = {"i1": (None, 0.0)}  # booting instance
    rep = reconcile(view, "e", SCOPE, tracked, safety_margin_s=10, now=1000.0)
    assert rep.ghost_ids == ()


# --------------------------------------------------------------------------- #
# live_ids None (fail-closed) vs empty set() — the sharpest edge
# --------------------------------------------------------------------------- #

def test_live_ids_none_skips_entirely_and_does_not_reap():
    view = FakeView(live=None)
    tracked = {"i1": ("r1", 0.0)}
    rep = reconcile(view, "e", SCOPE, tracked, safety_margin_s=10, now=1000.0)
    assert rep == reconcile.__globals__["ReapReport"](0)
    assert rep.orphans_reaped == 0 and rep.ghost_ids == ()
    assert view.reap_calls == []          # reap NOT called when live is None


def test_live_ids_probe_error_skips_entirely_and_does_not_reap():
    """A fallible probe raises ReconcileProbeError; reconcile()
    catches it INSIDE the loop → ReapReport(0), byte-identical fail-closed (reap NOT
    called, no false-ghosting), never propagating to recovery.py's blanket except."""
    from lite.gym.errors import ReconcileProbeError

    class _RaisingView:
        def __init__(self):
            self.reap_calls = []
        def live_ids(self, env_id, scope):
            raise ReconcileProbeError("docker ps failed")
        def reap(self, env_id, scope, in_use, *, boot=False):
            self.reap_calls.append((env_id, boot))
            return 0

    view = _RaisingView()
    tracked = {"i1": ("r1", 0.0)}
    rep = reconcile(view, "e", SCOPE, tracked, safety_margin_s=10, now=1000.0)
    assert rep == reconcile.__globals__["ReapReport"](0)
    assert rep.orphans_reaped == 0 and rep.ghost_ids == ()
    assert view.reap_calls == []          # reap NOT called when the probe raised


def test_empty_live_set_ghosts_everything():
    # set() (NOT None) means "world exists, currently empty" → all are ghosts.
    view = FakeView(live=set())
    tracked = {"i1": ("r1", 0.0), "i2": ("r2", 0.0)}
    rep = reconcile(view, "e", SCOPE, tracked, safety_margin_s=10, now=1000.0)
    assert set(rep.ghost_ids) == {"i1", "i2"}
    assert view.reap_calls[0][2] == frozenset()  # nothing in_use
    assert view.reap_calls[0][3] is False        # steady reconcile supplies boot=False


# --------------------------------------------------------------------------- #
# recover (boot) = reconcile(∅)
# --------------------------------------------------------------------------- #

def test_recover_reaps_with_empty_in_use_and_no_ghosts():
    view = FakeView(live={"r-orphan"}, reap_return=1)
    n = recover(view, "e", SCOPE)
    assert n == 1
    assert view.reap_calls[0][2] == frozenset()  # in_use empty at boot
    assert view.reap_calls[0][3] is True         # recover() supplies boot=True


def test_recover_reaps_regardless_of_live_ids_oracle():
    # Boot recovery calls reap DIRECTLY (no live_ids gate). A docker-ps hiccup
    # (live_ids would return None) must NOT skip the boot reap — else a
    # container env's boot latch strands and the first PERIODIC reap runs
    # unguarded wide-net. recover never consults live_ids.
    view = FakeView(live=None, reap_return=2)
    assert recover(view, "e", SCOPE) == 2
    assert view.reap_calls and view.reap_calls[0][2] == frozenset()  # in_use=∅
    assert view.reap_calls[0][3] is True  # boot=True even when live_ids would fail closed
    assert view.live_calls == []  # recover does NOT consult live_ids


# --------------------------------------------------------------------------- #
# cross-env validation + cap
# --------------------------------------------------------------------------- #

def test_collect_valid_ghosts_filters_unknown_ids():
    assert collect_valid_ghosts(["a", "b", "x"], {"a", "b"}) == ["a", "b"]
    assert collect_valid_ghosts(("x",), {"a"}) == []


def test_ghost_cap_applied(monkeypatch):
    monkeypatch.setattr(reconcile_mod, "MAX_GHOSTS_PER_CYCLE", 2)
    view = FakeView(live=set())
    tracked = {f"i{n}": (f"r{n}", 0.0) for n in range(5)}
    rep = reconcile(view, "e", SCOPE, tracked, safety_margin_s=1, now=1000.0)
    assert len(rep.ghost_ids) == 2


# --------------------------------------------------------------------------- #
# EnvServices defaults are inert (no external world, no ghosts)
# --------------------------------------------------------------------------- #

def test_env_services_defaults_are_noop():
    svc = EnvServices()
    assert svc.live_ids("e", SCOPE) is None
    assert svc.reap("e", SCOPE, set()) == 0
    assert svc.ensure("e") is None
    assert svc.health("e") is None
    assert svc.shutdown("e", SCOPE) is None
    # A default services object reconciles to a no-op.
    rep = reconcile(svc, "e", SCOPE, {"i1": ("r1", 0.0)}, now=1000.0)
    assert rep.orphans_reaped == 0 and rep.ghost_ids == ()
