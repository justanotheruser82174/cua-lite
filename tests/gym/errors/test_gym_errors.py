"""``is_retryable`` contract — the ONE retry predicate the trajectory-level retry
loops share.

Run::

    uv run pytest tests/gym/errors/test_gym_errors.py
"""
from __future__ import annotations

from lite.gym.errors import (
    CapacityExhausted,
    CuaGymTaskError,
    CuaWorldVerifierError,
    EnvBlocked,
    EnvDepsMissingError,
    EnvUnavailable,
    FailureCategory,
    PairableModelActionError,
    RemoteWireProtocolError,
    ScaleCuaTaskError,
    TrueInfraFailure,
    failure_category,
    is_retryable,
    lite_error_from_payload,
)
from lite.gym.remote.errors import ModelOutputError


def test_is_retryable_by_exception_type():
    # CapacityExhausted is the one transient (overrides the base default).
    assert is_retryable(CapacityExhausted("pool full", retry_after_s=10)) is True
    # Terminal gym errors inherit retryable=False from LiteGymError — no explicit
    # re-statement on the subclass (the simplification under test).
    assert is_retryable(EnvUnavailable("x", 404)) is False
    assert is_retryable(EnvBlocked(what="site blocked")) is False
    assert is_retryable(RemoteWireProtocolError("version skew")) is False
    # EnvDepsMissingError is an ImportError (NOT a LiteGymError), so it can't
    # inherit — it declares retryable=False itself.
    assert is_retryable(EnvDepsMissingError("x", "install.sh", "README.md")) is False
    # Untyped / unknown → retryable: a fresh re-run might recover, and the caller's
    # attempt cap bounds the waste.
    assert is_retryable(RuntimeError("boom")) is True
    assert is_retryable(TimeoutError()) is True
    assert is_retryable(PairableModelActionError("bad action")) is False
    assert is_retryable(ModelOutputError("bad output")) is False
    assert is_retryable(TrueInfraFailure("container crashed")) is True


def test_action_boundary_failure_categories_are_typed():
    pairable = PairableModelActionError("bad key")
    unpairable = ModelOutputError("duplicate call_id")
    infra = TrueInfraFailure("container crashed")

    assert failure_category(pairable) is FailureCategory.MODEL_ACTION_ERROR
    assert failure_category(unpairable) is FailureCategory.MODEL_OUTPUT_ERROR
    assert failure_category(infra) is FailureCategory.INFRA_FAILURE
    assert pairable.pairable is True
    assert unpairable.pairable is False
    assert infra.pairable is False
    assert failure_category(RuntimeError("untyped")) is None


def test_retryable_survives_rpc_roundtrip():
    # retryable is a class attribute, so the client's typed reconstruction carries
    # it with zero extra serialization — a server-raised terminal error stays
    # terminal client-side (no wire change needed).
    reconstructed = lite_error_from_payload(EnvBlocked(what="b").to_payload())
    assert type(reconstructed).__name__ == "EnvBlocked"
    assert is_retryable(reconstructed) is False


def test_typed_statuses_round_trip():
    """InstanceGone (404, retryable), ProtocolMisuse (409, terminal) and
    MakeFailed (400, terminal) survive the wire registry round-trip."""
    from lite.gym.errors import (
        InstanceGone,
        MakeFailed,
        ProtocolMisuse,
        is_retryable,
        lite_error_from_payload,
    )

    gone = lite_error_from_payload(
        InstanceGone(what="unknown instance id='x'").to_payload())
    assert isinstance(gone, InstanceGone) and gone.http_status == 404
    assert is_retryable(gone) is True   # re-make recovers

    misuse = lite_error_from_payload(ProtocolMisuse(what="step before reset").to_payload())
    assert isinstance(misuse, ProtocolMisuse) and misuse.http_status == 409
    assert is_retryable(misuse) is False

    mk = lite_error_from_payload(MakeFailed(what="gym.make failed: KeyError: 'x'").to_payload())
    assert isinstance(mk, MakeFailed) and mk.http_status == 400
    assert is_retryable(mk) is False


def test_b7_remote_env_error_wraps_unknown_types():
    """The client wraps a server-envelope {error_type, what} of an unknown
    class as RemoteEnvError: original name preserved, retryable (matches the
    pre-envelope unknown→retry default), and NOT wire-registered (it never
    originates server-side)."""
    from lite.gym.errors import RemoteEnvError, is_retryable, lite_error_from_payload

    assert lite_error_from_payload(
        {"error_type": "RuntimeError", "what": "boom"}
    ) is None, "unregistered types must NOT reconstruct typed"
    exc = RemoteEnvError(what="boom", original_type="RuntimeError")
    assert exc.original_type == "RuntimeError"
    assert "RuntimeError: boom" in str(exc)
    assert is_retryable(exc) is True
    assert lite_error_from_payload({"error_type": "RemoteEnvError", "what": "x"}) is None


def test_cua_task_errors_are_core_registered_for_remote_clients():
    """Remote rollout clients must reconstruct CUA 422s without importing envs."""
    cuagym = lite_error_from_payload(CuaGymTaskError(
        "reward missing",
        phase="reward",
        kind="no_reward",
    ).to_payload())
    assert isinstance(cuagym, CuaGymTaskError)
    assert cuagym.phase == "reward" and cuagym.kind == "no_reward"
    assert is_retryable(cuagym) is False

    scalecua = lite_error_from_payload(ScaleCuaTaskError(
        "unsupported action",
        phase="config",
        kind="unsupported_action",
        returncode=7,
    ).to_payload())
    assert isinstance(scalecua, ScaleCuaTaskError)
    assert scalecua.returncode == 7
    assert is_retryable(scalecua) is False

    cuaworld = lite_error_from_payload(CuaWorldVerifierError(
        "bad verifier result",
        phase="verify",
        kind="bad_result",
        task="pymol/align",
    ).to_payload())
    assert isinstance(cuaworld, CuaWorldVerifierError)
    assert cuaworld.task == "pymol/align"
    assert is_retryable(cuaworld) is False


def _all_taxonomy_error_types() -> list[type]:
    """Every error the env-server RPC can be asked to carry: :class:`LiteGymError`
    and its transitive subclasses, plus :class:`EnvDepsMissingError` (in the
    taxonomy but an ``ImportError``, so it cannot inherit).
    """
    import lite.gym.remote.errors  # noqa: F401  (registers ModelOutputError et al.)
    from lite.gym.errors import LiteGymError

    seen: list[type] = [LiteGymError, EnvDepsMissingError]
    frontier = [LiteGymError]
    while frontier:
        for sub in frontier.pop().__subclasses__():
            if sub not in seen:
                seen.append(sub)
                frontier.append(sub)
    return seen


def test_unregistered_members_are_exactly_the_client_side_ones():
    """Any taxonomy member missing from this set reaches remote clients as a
    retryable ``RemoteEnvError`` (via the catch-all 500) and burns their attempt
    budget, so adding one must be a deliberate edit here."""
    from lite.gym.errors import _LITE_ERROR_REGISTRY, LiteGymError

    exempt = {
        cls.__name__ for cls in _all_taxonomy_error_types()
        if _LITE_ERROR_REGISTRY.get(cls.__name__) is not cls
    }
    assert exempt == {
        "LiteGymError",           # abstract base, never raised on its own
        "RemoteEnvError",         # client-side wrapper for unknown server types
        "RemoteWireProtocolError",  # client-side version skew
        "RemoteCallTimeout",      # client-side deadline, no server reply at all
        "EnvDepsMissingError",    # crosses the wire as EnvUnavailable
    }, exempt
    assert LiteGymError in _all_taxonomy_error_types()


def test_deps_missing_crosses_the_wire_as_env_unavailable():
    """``EnvDepsMissingError`` reaches a remote caller as a typed, terminal
    ``EnvUnavailable`` instead of a retryable ``RemoteEnvError``."""
    exc = EnvDepsMissingError(
        what="OSWorld (desktop_env) package not installed",
        install="uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh",
        see="lite/gym/envs/osworld/README.md",
    )
    wire = exc.as_env_unavailable("lite.osworld")

    assert isinstance(wire, EnvUnavailable)
    assert wire.http_status == 501
    # install/see are DIRECT MODE ONLY — they must not cross to a remote caller
    # who cannot install on someone else's server.
    payload = wire.to_payload()
    blob = repr(payload)
    assert "install.sh" not in blob and "README.md" not in blob
    assert "lite.osworld" in payload["what"] and "desktop_env" in payload["what"]

    back = lite_error_from_payload(payload)
    assert isinstance(back, EnvUnavailable)
    assert is_retryable(back) is False
    assert back.status_code == 501


def test_env_access_denied_round_trips_terminal():
    """A wrong-token 403 is a permanent misconfiguration, not a retry condition."""
    from lite.gym.errors import EnvAccessDenied

    denied = lite_error_from_payload(
        EnvAccessDenied(what="instance id='abc' owned by another token").to_payload()
    )
    assert isinstance(denied, EnvAccessDenied)
    assert denied.http_status == 403
    assert is_retryable(denied) is False


def test_transport_classification_is_typed_and_phase_aware():
    """The retry rule, at predicate level: TYPED only, and the phase decides.

    Replaces the single ``is_transient_rpc_error`` classifier, which conflated
    "the transport failed" with "the request never reached the worker" and had to
    call a mid-response cut non-transient to stay safe on ``/step``. The two
    questions are now separate, so a cut is a transport failure AND
    reached-the-worker, and only ``replay_safe`` decides what that licenses.
    """
    import urllib.error

    from lite.gym.utils.backend.rpc import (
        is_transport_error,
        may_reissue,
        reached_worker,
    )

    # Transport, and the raiser PROVED nothing executed → re-issuable either way.
    for exc in (ConnectionRefusedError(), urllib.error.URLError(ConnectionRefusedError())):
        assert is_transport_error(exc) is True
        assert reached_worker(exc) is False
        assert may_reissue(exc, replay_safe=False) is True

    # Transport, but the bytes were on the wire → replay-safe ops only.
    cut = ConnectionResetError("osworld server /step cut mid-response: boom")
    assert is_transport_error(cut) is True
    assert reached_worker(cut) is True
    assert may_reissue(cut, replay_safe=False) is False
    assert may_reissue(cut, replay_safe=True) is True

    # Unknown phase counts as REACHED (the polarity is the safety property).
    bare = ConnectionError("connection aborted")
    assert reached_worker(bare) is True
    assert may_reissue(bare, replay_safe=False) is False

    # NOT transport: an ANSWERED non-2xx, and a timeout whose peer may still be
    # working. ``replay_safe`` cannot license these.
    for exc in (
        RuntimeError("server /step returned 500: boom"),
        RuntimeError("server /step returned 500: [Errno 111] Connection refused"),
        TimeoutError(),
    ):
        assert is_transport_error(exc) is False
        assert may_reissue(exc, replay_safe=True) is False
