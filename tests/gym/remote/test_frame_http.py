"""HTTP step/reset binary-frame contract tests.

Two halves:

  * **Handler wiring** — a real FastAPI ``TestClient`` over ``make_app`` with a
    screenshot-bearing stub env: ``/reset`` and ``/step`` 200s carry split
    ``application/octet-stream`` bodies that the frame decoders reconstruct,
    and the literal ``b"screenshot_b64"`` never appears in either body. This is
    the "helper exists != handler wired" gate.
  * **Client skew gate** — reset/step decode helpers mapped directly over forged
    ``httpx.Response``s (wiring a live version-skewed server is heavy): a
    JSON-200 or bad-MAGIC body → terminal :class:`RemoteWireProtocolError` ("upgrade");
    an empty / truncated 200 → retryable :class:`RemoteEnvError`.

Reuses the ``tests/gym/remote/test_server.py`` harness style (``make_app`` +
``_make_test_admission`` + bearer header from ``gym.remote.conftest``).

Run:  uv run pytest tests/gym/remote/test_frame_http.py -x -q
"""
from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from gym.remote.conftest import bearer_header as _h
from gym.remote.conftest import make_test_admission
from PIL import Image

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.core.tools.results import LiteToolResult
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import CapacityExhausted, RemoteEnvError, RemoteWireProtocolError
from lite.gym.remote.client import (
    LiteEnvClient,
    _decode_reset_frame,
    _decode_step_frame,
)
from lite.gym.remote.frame import (
    FRAME_MAGIC,
    FRAME_VERSION,
    encode_reset_observation,
    encode_step_result,
)
from lite.gym.remote.server import State, make_app
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult


def _png(
    *,
    size: tuple[int, int] = (32, 24),
    color: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


_SHOT = _png(color=(4, 8, 12))
_SHOT_2 = _png(size=(16, 12), color=(90, 20, 40))


class _ShotEnv(LiteBaseEnv):
    """Minimal env returning a real PNG-bytes screenshot — no docker."""

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("browser", "use"))

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(
            image=_SHOT, text="task", metadata={"url": "http://x"}
        )

    async def step(self, actions: list) -> LiteEnvStepResult:
        return LiteEnvStepResult(
            reward=1.0,
            terminated=True,
            results=[
                LiteToolResult(
                    tool_call_id="action_0",
                    images=[_SHOT, _SHOT_2],
                    text="stepped",
                )
            ],
        )

    async def close(self) -> None:
        pass


class _CapacityEnv(_ShotEnv):
    async def reset(self) -> LiteEnvObservation:
        raise CapacityExhausted(
            what="mock reset capacity",
            retry_after_s=7.0,
            layer="env_internal",
        )


TOKEN = "zzh-dev"


def _client(env_factory=lambda: _ShotEnv()) -> tuple[TestClient, Any]:
    state = State(
        admission=make_test_admission(max_live_envs=100), idle_ttl_sec=3600.0
    )
    app = make_app(state, token=TOKEN)
    patcher = patch(
        "lite.gym.remote.server.gym.make", lambda env_key, **kw: env_factory()
    )
    patcher.start()
    return TestClient(app), patcher


def _create(client: TestClient) -> str:
    r = client.post(
        "/instances",
        headers=_h(TOKEN),
        json={"env_key": "lite.demo@t", "env_kwargs": {}, "session_id": "s"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Handler wiring — /reset + /step return an octet-stream frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["reset", "step"])
def test_step_reset_return_octet_stream_frame(op: str):
    client, patcher = _client()
    try:
        iid = _create(client)
        # reset first so /step is allowed
        r0 = client.post(f"/instances/{iid}/reset", headers=_h(TOKEN))
        assert r0.status_code == 200, r0.text
        if op == "reset":
            r = r0
        else:
            r = client.post(
                f"/instances/{iid}/step", headers=_h(TOKEN), json={"actions": []}
            )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/octet-stream"

        body = r.content
        # No base64 screenshot field anywhere in the body.
        assert b"screenshot_b64" not in body

        if op == "reset":
            result = _decode_reset_frame(r, "/reset")
            assert result.image == _SHOT
            assert result.text == "task"
            assert result.metadata == {"url": "http://x"}
            return
        result = _decode_step_frame(r, "/step")
        if op == "step":
            assert result.reward == 1.0
            assert result.terminated is True
            assert [(r.tool_call_id, r.images, r.text) for r in result.results] == [
                ("action_0", [_SHOT, _SHOT_2], "stepped")
            ]
    finally:
        patcher.stop()


async def test_lite_env_client_reset_step_preserve_image_bytes(monkeypatch):
    """LiteEnvClient decodes frames but does not repaint cursor pixels client-side."""

    direct_env = _ShotEnv()
    expected_observation = await direct_env.reset()
    expected_step = await direct_env.step([make_tool_call("noop")])

    def _fake_eager_create(self: LiteEnvClient) -> None:
        self._id = "inst-1"
        self._metadata = LiteCUAMetadata(dims=("browser", "use"))

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/reset"):
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_reset_observation(expected_observation),
                request=request,
            )
        if request.url.path.endswith("/step"):
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_step_result(expected_step),
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    monkeypatch.setattr(LiteEnvClient, "_eager_create_sync", _fake_eager_create)
    client = LiteEnvClient("http://server", "lite.demo@t", token="t")
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    try:
        obs = await client.reset()
        result = await client.step([make_tool_call("noop")])
    finally:
        await client._client.aclose()
        client._id = None

    assert obs.image == _SHOT
    assert obs == expected_observation
    assert [tool_result.images for tool_result in result.results] == [
        tool_result.images for tool_result in expected_step.results
    ]


def test_client_gate_rejects_reset_frame_with_multiple_images():
    """Reset remains a single-observation image, unlike step result images."""
    header = {
        "v": FRAME_VERSION,
        "kind": "reset",
        "metadata": None,
        "text": "task",
        "shot_n": [len(_SHOT), len(_SHOT_2)],
    }
    hb = json.dumps(header).encode("utf-8")
    frame = (
        FRAME_MAGIC.encode("ascii")
        + len(hb).to_bytes(4, "big")
        + hb
        + _SHOT
        + _SHOT_2
    )
    response = _resp(frame, content_type="application/octet-stream")

    with pytest.raises(RemoteWireProtocolError, match="expected <= 1"):
        _decode_reset_frame(response, "/reset")


def test_capacity_exhausted_handler_preserves_wire_payload():
    state = State(
        admission=make_test_admission(max_live_envs=100), idle_ttl_sec=3600.0
    )
    app = make_app(state, token=TOKEN)
    patcher = patch(
        "lite.gym.remote.server.gym.make", lambda env_key, **kw: _CapacityEnv()
    )
    patcher.start()
    try:
        client = TestClient(app)
        iid = _create(client)
        r = client.post(f"/instances/{iid}/reset", headers=_h(TOKEN))
        assert r.status_code == 503, r.text
        assert r.headers["Retry-After"] == "7"
        data = r.json()
        assert data["error_type"] == "CapacityExhausted"
        assert data["what"] == "mock reset capacity"
        assert data["retry_after_s"] == 7.0
        assert data["layer"] == "env_internal"
        assert "mock reset capacity" in data["detail"]
    finally:
        patcher.stop()


# ---------------------------------------------------------------------------
# Client skew gate — _decode_step_frame classification
# ---------------------------------------------------------------------------


def _resp(
    content: bytes,
    *,
    content_type: str,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    response_headers = {"content-type": content_type}
    if headers:
        response_headers.update(headers)
    return httpx.Response(
        status_code=status,
        headers=response_headers,
        content=content,
        request=httpx.Request("POST", "http://server/instances/inst-1/step"),
    )


def test_json_200_is_terminal_wire_protocol_error():
    """An old (pre-frame) server returns a JSON obs on 200 → terminal."""
    r = _resp(b'{"observation": {}}', content_type="application/json")
    with pytest.raises(RemoteWireProtocolError):
        _decode_step_frame(r, "/step")


def test_bad_magic_body_is_terminal_wire_protocol_error():
    """A foreign / newer wire (bumped MAGIC) → terminal, not a retry-storm."""
    r = _resp(b"LEF1" + b"\x00" * 16, content_type="application/octet-stream")
    with pytest.raises(RemoteWireProtocolError):
        _decode_step_frame(r, "/step")


def test_empty_200_body_is_malformed_server_response():
    """The live dropped-response race: an empty 200 body → typed malformed frame."""
    r = _resp(b"", content_type="application/octet-stream")
    with pytest.raises(RemoteEnvError) as exc_info:
        _decode_reset_frame(r, "/reset")
    assert exc_info.value.original_type == "MalformedServerResponse"


def test_truncated_200_body_is_malformed_server_response():
    frame = encode_reset_observation(LiteEnvObservation(image=_SHOT))
    r = _resp(frame[:-8], content_type="application/octet-stream")
    with pytest.raises(RemoteEnvError) as exc_info:
        _decode_reset_frame(r, "/reset")
    assert exc_info.value.original_type == "MalformedServerResponse"


def test_valid_frame_decodes_through_client_gate():
    frame = encode_step_result(
        LiteEnvStepResult(
            reward=1.0,
            results=[LiteToolResult(tool_call_id="bash_0", text="stdout")],
        )
    )
    r = _resp(frame, content_type="application/octet-stream")
    result = _decode_step_frame(r, "/step")
    assert result.reward == 1.0
    assert [(r.tool_call_id, r.text) for r in result.results] == [("bash_0", "stdout")]


@pytest.mark.asyncio
async def test_client_retries_truncated_reset_frame(monkeypatch):
    frame = encode_reset_observation(
        LiteEnvObservation(image=_SHOT, text="reset-ok")
    )
    responses = [
        _resp(frame[:-8], content_type="application/octet-stream"),
        _resp(frame, content_type="application/octet-stream"),
    ]
    calls: list[tuple[str, dict | None, str]] = []
    client = LiteEnvClient.__new__(LiteEnvClient)
    client._id = "inst-1"

    async def fake_post(path, json_body, op_name, *, replay_safe=True):
        calls.append((path, json_body, op_name))
        assert replay_safe is True
        return responses.pop(0)

    monkeypatch.setattr(client, "_post_with_retry", fake_post)
    monkeypatch.setattr(
        "lite.gym.remote.client._client_retry_delay_s", lambda _attempt: 0.0
    )

    result = await client.reset()

    assert calls == [
        ("/instances/inst-1/reset", None, "/reset"),
        ("/instances/inst-1/reset", None, "/reset"),
    ]
    assert result.text == "reset-ok"
    assert responses == []


@pytest.mark.asyncio
async def test_client_does_not_retry_truncated_step_frame(monkeypatch):
    frame = encode_step_result(
        LiteEnvStepResult(
            reward=1.0,
            results=[LiteToolResult(tool_call_id="action_0", images=[_SHOT])],
        )
    )
    calls: list[tuple[str, dict | None, str]] = []
    client = LiteEnvClient.__new__(LiteEnvClient)
    client._id = "inst-1"

    async def fake_post(path, json_body, op_name, *, replay_safe=True):
        calls.append((path, json_body, op_name))
        assert replay_safe is False
        return _resp(frame[:-8], content_type="application/octet-stream")

    monkeypatch.setattr(client, "_post_with_retry", fake_post)
    monkeypatch.setattr(
        "lite.gym.remote.client._client_retry_delay_s", lambda _attempt: 0.0
    )

    actions = [make_tool_call("noop")]
    with pytest.raises(RemoteEnvError):
        await client.step(actions)

    assert calls == [
        ("/instances/inst-1/step", {"actions": actions}, "/step"),
    ]


@pytest.mark.asyncio
async def test_client_does_not_retry_transient_step_network_drop():
    calls = 0

    class _DroppingHTTPClient:
        async def post(self, url, json=None):
            nonlocal calls
            calls += 1
            raise httpx.ReadError("connection dropped after step")

    client = LiteEnvClient.__new__(LiteEnvClient)
    client._id = "inst-1"
    client._url = "http://server"
    client._client = _DroppingHTTPClient()

    actions = [make_tool_call("noop")]
    with pytest.raises(httpx.ReadError):
        await client.step(actions)

    assert calls == 1


@pytest.mark.asyncio
async def test_step_does_not_retry_503_capacity_response():
    responses = [
        _resp(
            b'{"error_type":"CapacityExhausted","what":"busy",'
            b'"retry_after_s":0,"layer":"admission"}',
            content_type="application/json",
            status=503,
            headers={"Retry-After": "0"},
        ),
    ]
    calls: list[tuple[str, dict | None]] = []

    class _FakeHTTPClient:
        async def post(self, url: str, json: dict | None = None):
            calls.append((url, json))
            return responses.pop(0)

    client = LiteEnvClient.__new__(LiteEnvClient)
    client._id = "inst-1"
    client._url = "http://server"
    client._client = _FakeHTTPClient()

    actions = [make_tool_call("noop")]
    with pytest.raises(CapacityExhausted, match="busy"):
        await client.step(actions)

    assert calls == [
        ("http://server/instances/inst-1/step", {"actions": actions}),
    ]
    assert responses == []
