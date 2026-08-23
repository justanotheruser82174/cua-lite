"""Client-side response-body robustness (section B7 tail) — an empty/truncated 2xx
body maps to a typed retryable RemoteEnvError, never a raw JSONDecodeError.

Observed live during the section 10 reaper-vs-destroy storm: server reset hit its
600s cap exactly as the client's own timeout expired; the response write
raced the cancellation and the client read an empty body.

Run: uv run pytest tests/gym/remote/test_client_response_parse.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys

import httpx
import pytest

from lite.core.tools import make_tool_call
from lite.gym.errors import (
    CapacityExhausted,
    CuaGymTaskError,
    CuaWorldVerifierError,
    FailureCategory,
    RemoteEnvError,
    ScaleCuaTaskError,
    failure_category,
    is_retryable,
)
from lite.gym.remote.client import _response_json
from lite.gym.remote.errors import ModelOutputError


def _resp(body: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=body,
        request=httpx.Request("POST", "http://server/instances/i/reset"),
    )


def _json_resp(payload: dict, status: int = 422) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "http://server/instances/i/reset"),
    )


def test_valid_json_passes_through():
    assert _response_json(_resp(b'{"id": "abc"}'), "create") == {"id": "abc"}


@pytest.mark.parametrize("body", [b"", b"Internal Server Error", b'{"truncat'])
def test_malformed_body_maps_to_typed_retryable(body):
    with pytest.raises(RemoteEnvError) as ei:
        _response_json(_resp(body), "/reset")
    assert ei.value.retryable is True, "transport corruption must stay retryable"
    assert "MalformedServerResponse" in str(ei.value.original_type)
    assert "/reset" in str(ei.value), "error must name the operation"


def test_typed_error_helper_unwraps_router_502():
    """The fleet router synthesizes {error_type, what} on 502 (owner alive
    but the forward failed) — the client must wrap it as retryable
    RemoteEnvError exactly like the server's 500 catch-all envelope."""
    from lite.gym.remote.client import _raise_typed_error_if_any

    r = _resp(b'{"error_type": "ReadTimeout", "what": "node n0: boom"}', status=502)
    with pytest.raises(RemoteEnvError) as ei:
        _raise_typed_error_if_any(r)
    assert ei.value.retryable is True
    assert "ReadTimeout" in str(ei.value.original_type)


@pytest.mark.parametrize("body", [b'["not", "an", "object"]', b'"oops"'])
def test_typed_error_helper_ignores_non_object_json(body):
    from lite.gym.remote.client import _raise_typed_error_if_any

    _raise_typed_error_if_any(_resp(body, status=500))


def test_capacity_exhausted_from_response_preserves_layer():
    """``retry_after_s`` / ``layer`` come from ``CapacityExhausted.from_payload``;
    the client must not re-derive the field names."""
    from lite.gym.remote.client import _capacity_exhausted_from_response

    payload = CapacityExhausted(
        "emergency stop", retry_after_s=3, layer="emergency",
    ).to_payload()
    exc = _capacity_exhausted_from_response(
        _resp(json.dumps(payload).encode(), status=503),
        fallback_retry_after_s=1,
    )
    assert isinstance(exc, CapacityExhausted)
    assert exc.retry_after_s == 3
    assert exc.layer == "emergency"


def test_untyped_503_still_yields_the_capacity_type():
    """A 503 from a proxy carries no ``error_type``. The wait loop keys on the
    TYPE, so the fallback must still be ``CapacityExhausted``, using the caller's
    header hint."""
    from lite.gym.remote.client import _capacity_exhausted_from_response

    exc = _capacity_exhausted_from_response(
        _resp(b"<html>502 Bad Gateway</html>", status=503),
        fallback_retry_after_s=7,
    )
    assert isinstance(exc, CapacityExhausted)
    assert exc.retry_after_s == 7
    assert is_retryable(exc) is True


def test_remote_client_import_registers_remote_error_types():
    """Fresh client-only imports must register remote-specific LiteGymError types."""
    code = """
from lite.gym.errors import lite_error_from_payload, is_retryable
from lite.gym.remote.client import _raise_typed_error_if_any  # noqa: F401

payload = {
    "error_type": "ModelOutputError",
    "what": "bad /step payload",
    "kind": "malformed_step_request",
    "payload_metadata": {"action_index": 0},
}
exc = lite_error_from_payload(payload)
assert type(exc).__name__ == "ModelOutputError", exc
assert exc.payload_metadata == {"action_index": 0}
assert is_retryable(exc) is False
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_typed_error_helper_reconstructs_remote_model_output_error():
    from lite.gym.remote.client import _raise_typed_error_if_any

    metadata = {"action_index": 0, "action_keys": ["function", "id", "type"]}
    payload = ModelOutputError(
        "malformed /step request body: body.actions.0: Input should be an object",
        payload_metadata=metadata,
    ).to_payload()
    r = _resp(json.dumps(payload).encode(), status=422)
    with pytest.raises(ModelOutputError) as ei:
        _raise_typed_error_if_any(r)

    assert ei.value.kind == "malformed_step_request"
    assert ei.value.payload_metadata == metadata
    assert is_retryable(ei.value) is False
    assert failure_category(ei.value) is FailureCategory.MODEL_OUTPUT_ERROR


def test_remote_client_logs_the_cuagym_422_body_with_its_kind(caplog):
    """The 422 body must still reach logs, `kind` included."""
    from lite.gym.remote import client as remote_client

    payload = CuaGymTaskError(
        "lite.cuagym desktop setup failed with rc=1: E: Unable to locate package "
        "imagemagick",
        phase="setup",
        kind="command_failed",
        returncode=1,
    ).to_payload()
    with caplog.at_level(logging.WARNING, logger=remote_client.__name__):
        remote_client._log_error_response(_json_resp(payload), "/reset")

    logged = caplog.text
    assert "/reset" in logged and "422" in logged
    assert "kind='command_failed'" in logged
    assert "phase='setup'" in logged
    assert "returncode=1" in logged
    assert "Unable to locate package" in logged


@pytest.mark.parametrize(
    "response",
    [
        _json_resp({"error_type": "X", "what": "y" * 5000}, status=500),
        httpx.Response(
            502,
            content=b"<html>" + b"z" * 200_000,
            request=httpx.Request("POST", "http://server/instances/i/step"),
        ),
    ],
)
def test_remote_client_error_body_log_stays_bounded(response):
    """Bounded so large server error pages do not flood rollout logs."""
    from lite.gym.remote import client as remote_client

    assert len(remote_client._error_body_summary(response)) <= (
        remote_client._ERROR_BODY_MAX
    )


def test_remote_client_error_body_summary_survives_a_non_json_body():
    from lite.gym.remote import client as remote_client

    empty = httpx.Response(
        422, content=b"", request=httpx.Request("POST", "http://server/x"),
    )
    assert remote_client._error_body_summary(empty) == "<empty body>"


@pytest.mark.parametrize("status", [400, 403, 409, 422, 429, 500, 503])
def test_typed_reconstruction_ignores_the_status_code(status):
    """``error_type`` owns the failure class and the status is PRESENTATION, so
    ONE body reconstructs to the same class with the same ``retryable`` on every
    code the server can put in front of it — no numeric branch at the consumer."""
    from lite.gym.remote.client import _raise_typed_error_if_any

    payload = ModelOutputError("body.actions.0: Input should be an object").to_payload()
    r = _resp(json.dumps(payload).encode(), status=status)
    with pytest.raises(ModelOutputError) as ei:
        _raise_typed_error_if_any(r)
    assert ei.value.kind == "malformed_step_request"
    assert is_retryable(ei.value) is False


@pytest.mark.asyncio
async def test_post_with_retry_raises_typed_model_output_error_not_http_status_error():
    from lite.gym.remote.client import LiteEnvClient

    payload = ModelOutputError(
        "malformed /step request body: body.actions.0: rejected by env ingress",
        payload_metadata={"action_index": 0, "id": "provider_1"},
    ).to_payload()
    body = json.dumps(payload).encode()

    class _FakeHTTPClient:
        async def post(self, url: str, json: dict | None = None):
            return _resp(body, status=422)

    client = LiteEnvClient.__new__(LiteEnvClient)
    client._id = "inst-1"
    client._url = "http://server"
    client._client = _FakeHTTPClient()

    with pytest.raises(ModelOutputError) as ei:
        await client._post_with_retry(
            "/instances/inst-1/step",
            {
                "actions": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="provider_1",
                    )
                ]
            },
            "/step",
            replay_safe=False,
        )

    assert ei.value.payload_metadata["id"] == "provider_1"


@pytest.mark.asyncio
async def test_client_step_json_preflight_rejects_bytes_without_http_send():
    from lite.gym.remote.client import LiteEnvClient

    async def should_not_send(*args, **kwargs):
        raise AssertionError("non-serializable step body should not reach HTTP")

    client = LiteEnvClient.__new__(LiteEnvClient)
    client._id = "inst-1"
    client._post_frame_with_retry = should_not_send

    raw = b"x" * 10_000
    action = make_tool_call(
        "computer",
        {"actions": [{"action": "type", "text": raw}]},
        call_id="call_bytes",
    )
    action["debug"] = raw
    actions = [action]
    with pytest.raises(ModelOutputError) as ei:
        await client.step(actions)

    exc = ei.value
    assert exc.kind == "malformed_step_request"
    assert exc.payload_metadata["action_index"] == 0
    assert exc.payload_metadata["id"] == "call_bytes"
    assert exc.payload_metadata["name"] == "computer"
    assert "debug" not in exc.payload_metadata
    assert "Object of type bytes" in exc.payload_metadata["json_error"]
    assert raw.decode("ascii") not in str(exc)
    assert len(str(exc)) < 400


@pytest.mark.asyncio
async def test_client_step_json_preflight_rejects_nonfinite_without_raw_payload_leak():
    from lite.gym.remote.client import LiteEnvClient

    async def should_not_send(*args, **kwargs):
        raise AssertionError("non-finite step body should not reach HTTP")

    client = LiteEnvClient.__new__(LiteEnvClient)
    client._id = "inst-1"
    client._post_frame_with_retry = should_not_send

    with pytest.raises(ModelOutputError) as ei:
        await client.step([
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [float("nan"), 1]}]},
                call_id="call_nan",
            )
        ])

    exc = ei.value
    assert exc.payload_metadata["action_index"] == 0
    assert exc.payload_metadata["id"] == "call_nan"
    assert exc.payload_metadata["name"] == "computer"
    assert exc.payload_metadata["json_error_type"] == "ValueError"
    assert "Out of range float values" in exc.payload_metadata["json_error"]


@pytest.mark.parametrize(
    ("payload", "expected_type", "attrs"),
    [
        (
            CuaWorldVerifierError(
                "verifier returned nan",
                phase="verify",
                kind="bad_result",
                task="pymol/align",
            ).to_payload(),
            CuaWorldVerifierError,
            {"phase": "verify", "kind": "bad_result", "task": "pymol/align"},
        ),
        (
            CuaGymTaskError(
                "reward.py failed",
                phase="reward",
                kind="command_failed",
                returncode=2,
            ).to_payload(),
            CuaGymTaskError,
            {"phase": "reward", "kind": "command_failed", "returncode": 2},
        ),
        (
            ScaleCuaTaskError(
                "bad generated action",
                phase="config",
                kind="unsupported_action",
                returncode=7,
            ).to_payload(),
            ScaleCuaTaskError,
            {"phase": "config", "kind": "unsupported_action", "returncode": 7},
        ),
    ],
)
def test_typed_error_helper_reconstructs_core_registered_422(
    payload,
    expected_type,
    attrs,
):
    """Client-only remote mode imports core errors, not env packages."""
    from lite.gym.remote.client import _raise_typed_error_if_any

    r = _resp(json.dumps(payload).encode(), status=422)
    with pytest.raises(expected_type) as ei:
        _raise_typed_error_if_any(r)
    exc = ei.value
    assert is_retryable(exc) is False
    for key, value in attrs.items():
        assert getattr(exc, key) == value


# ---------------------------------------------------------------------------
# Client-side read timeout — the client-vs-server reset-budget asymmetry
# ---------------------------------------------------------------------------


def _client_stub(post_side_effect):
    """A ``LiteEnvClient`` shell wired to a fake ``post``.

    Built with ``__new__`` on purpose: ``__init__`` eagerly POSTs /instances,
    and this exercises ``_post_with_retry`` in isolation.
    """
    from lite.gym.remote.client import LiteEnvClient

    calls: list[str] = []

    class _FakeAsyncClient:
        async def post(self, url, json=None):
            calls.append(url)
            return post_side_effect()

    obj = LiteEnvClient.__new__(LiteEnvClient)
    obj._url = "http://server"
    obj._id = "inst-1"
    obj._timeout = 660.0
    obj._client = _FakeAsyncClient()
    return obj, calls


@pytest.mark.parametrize("op_name,path,replay_safe", [
    ("/reset", "/instances/inst-1/reset", True),
    ("/step", "/instances/inst-1/step", False),
])
def test_read_timeout_is_typed_and_never_replayed(op_name, path, replay_safe):
    """``httpx.ReadTimeout`` is the client's own deadline expiring, not a server
    failure. The client waits ``max(reset_timeout, step_timeout) + 60`` = ONE reset
    plus slack, while the server's ``/reset`` runs up to 4 resets inside the held
    ``reset_sema`` slot — so when this fires the server is very likely STILL
    EXECUTING the call. It must therefore be typed (``retryable`` by RE-MAKING,
    not by ``is_retryable``'s unknown→True fallback) and never re-POSTed, which
    would queue a second heavy op behind the first on the same semaphore.
    """
    import asyncio

    from lite.gym.remote.errors import RemoteCallTimeout

    def _boom():
        raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://server"))

    obj, calls = _client_stub(_boom)
    with pytest.raises(RemoteCallTimeout) as ei:
        asyncio.run(obj._post_with_retry(
            path, None, op_name, replay_safe=replay_safe,
        ))

    assert len(calls) == 1, "a read timeout must not be re-sent"
    assert is_retryable(ei.value) is True   # recoverable, but by RE-MAKING
    assert ei.value.op_name == op_name
    assert "660" in str(ei.value), "the client's own budget must be in the message"
    # NOT registered: it never originates server-side.
    from lite.gym.errors import lite_error_from_payload
    assert lite_error_from_payload(
        {"error_type": "RemoteCallTimeout", "what": "x"}
    ) is None


def test_read_error_is_still_transiently_retried_for_replay_safe_ops(monkeypatch):
    """Guards the distinction the timeout case rests on: a ReadError (the
    transport broke) keeps its retry for /reset; a ReadTimeout (server probably
    still working) is not a transport failure and is exempted.

    This hop no longer owns a classifier of its own — the predicate asserted here
    is the shared ``is_transport_error``."""
    import asyncio

    from lite.gym.remote import client as client_module
    from lite.gym.utils.backend.rpc import is_transport_error

    monkeypatch.setattr(client_module, "_CLIENT_RETRY_BASE_DELAY_S", 0.0)

    assert is_transport_error(httpx.ReadError("dropped")) is True
    assert is_transport_error(
        httpx.ReadTimeout("slow", request=httpx.Request("POST", "http://server"))
    ) is False

    def _boom():
        raise httpx.ReadError("dropped")

    obj, calls = _client_stub(_boom)
    with pytest.raises(httpx.ReadError):
        asyncio.run(obj._post_with_retry(
            "/instances/inst-1/reset", None, "/reset", replay_safe=True,
        ))
    assert len(calls) > 1, "ReadError must still be retried for a replay-safe op"


def test_read_error_is_not_retried_for_a_step(monkeypatch):
    """The other half of the one rule: a ``ReadError`` reached the worker, so
    ``replay_safe=False`` pins the answer to False and the identical drop that
    /reset retries is sent exactly once for /step."""
    import asyncio

    from lite.gym.remote import client as client_module

    monkeypatch.setattr(client_module, "_CLIENT_RETRY_BASE_DELAY_S", 0.0)

    def _boom():
        raise httpx.ReadError("dropped")

    obj, calls = _client_stub(_boom)
    with pytest.raises(httpx.ReadError):
        asyncio.run(obj._post_with_retry(
            "/instances/inst-1/step", {"actions": []}, "/step", replay_safe=False,
        ))
    assert len(calls) == 1, "a non-replay-safe op must never be re-sent"
