"""Fleet-router binary-frame passthrough tests.

The fleet router is a second /step+/reset consumer, and its transparency is a
load-bearing invariant.

A fake node returns real split-frame octet-streams on /reset and /step; the
router's ``_sticky_forward`` → ``_streamed`` path must forward
it **byte-identical** with ``content-type`` preserved (``aiter_raw``, no
transparent decompression, no re-frame of the blob). If the router ever
``json``-walked the body, or dropped/rewrote the content-type, the binary frame
would corrupt in transit.

Self-contained in-process harness (mirrors ``test_fleet_router.py``): each node
is a small FastAPI app mounted into the router's shared ``httpx.AsyncClient``
via ``ASGITransport`` — NO docker, NO sockets.

Run:  uv run pytest tests/gym/remote/test_frame_fleet_passthrough.py -x -q
"""
from __future__ import annotations

import io
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

from lite.core.tools import make_tool_call
from lite.core.tools.results import LiteToolResult
from lite.gym.errors import FailureCategory, failure_category
from lite.gym.remote.client import _raise_typed_error_if_any
from lite.gym.remote.errors import ModelOutputError
from lite.gym.remote.fleet import FleetState, StaticProvider, make_fleet_app
from lite.gym.remote.frame import (
    decode_reset_observation,
    decode_step_result,
    encode_reset_observation,
    encode_step_result,
)
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

ALICE = {"Authorization": "Bearer alice-token"}


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30)).save(buf, format="PNG")
    return buf.getvalue()


_SHOT = _png()
_RESET_FRAME = encode_reset_observation(
    LiteEnvObservation(image=_SHOT, text="reset-task")
)
_STEP_FRAME = encode_step_result(
    LiteEnvStepResult(
        reward=1.0,
        terminated=True,
        results=[LiteToolResult(tool_call_id="action_0", images=[_SHOT])],
    )
)


class _FrameNode:
    """A node whose /reset and /step reply with a raw octet-stream frame."""

    def __init__(
        self,
        name: str,
        *,
        step_error_payload: dict | None = None,
    ) -> None:
        self.name = name
        self.step_error_payload = step_error_payload
        self.instances: dict[str, str] = {}  # id -> token
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        node = self

        def _token(request: Request) -> str | None:
            auth = request.headers.get("authorization") or ""
            return auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else None

        @app.get("/host_status")
        async def host_status() -> dict:
            return {"cua_lite": {"commit": f"{node.name}-commit"}}

        @app.post("/instances")
        async def create(request: Request) -> JSONResponse:
            id_ = f"{node.name}-{uuid.uuid4().hex[:8]}"
            node.instances[id_] = _token(request)
            return JSONResponse(status_code=201, content={"id": id_, "metadata": None})

        @app.get("/instances")
        async def list_instances(request: Request) -> dict:
            tok = _token(request)
            return {"instances": [
                {"id": i, "env_key": "lite.demo@t"}
                for i, t in node.instances.items() if t == tok
            ]}

        @app.post("/instances/{id_}/reset")
        async def reset(id_: str) -> Response:
            return Response(content=_RESET_FRAME, media_type="application/octet-stream")

        @app.post("/instances/{id_}/step")
        async def step(id_: str) -> Response:
            if node.step_error_payload is not None:
                return JSONResponse(
                    status_code=422,
                    content=node.step_error_payload,
                )
            return Response(content=_STEP_FRAME, media_type="application/octet-stream")

        return app


@asynccontextmanager
async def _fleet(tmp_path, node: _FrameNode):
    nodes_file = tmp_path / "nodes.txt"
    nodes_file.write_text(f"http://{node.name}\n")
    node_client = httpx.AsyncClient(
        mounts={f"http://{node.name}": httpx.ASGITransport(app=node.app)},
    )
    fleet = FleetState(StaticProvider(nodes_file), client=node_client)
    router = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_fleet_app(fleet)),
        base_url="http://router",
    )
    try:
        yield fleet, router
    finally:
        await router.aclose()
        await node_client.aclose()


async def _create(router: httpx.AsyncClient) -> str:
    r = await router.post(
        "/instances",
        json={"env_key": "lite.demo@t", "env_kwargs": {}, "session_id": "s"},
        headers=ALICE,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def test_reset_frame_forwarded_byte_identical(tmp_path):
    node = _FrameNode("node-a")
    async with _fleet(tmp_path, node) as (fleet, router):
        await fleet.poll_once()
        iid = await _create(router)
        r = await router.post(f"/instances/{iid}/reset", headers=ALICE)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/octet-stream"
        assert r.content == _RESET_FRAME  # byte-identical passthrough
        obs = decode_reset_observation(r.content)
        assert obs.image == _SHOT
        assert obs.text == "reset-task"


async def test_step_frame_forwarded_byte_identical(tmp_path):
    node = _FrameNode("node-a")
    async with _fleet(tmp_path, node) as (fleet, router):
        await fleet.poll_once()
        iid = await _create(router)
        r = await router.post(
            f"/instances/{iid}/step", headers=ALICE, json={"actions": []}
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/octet-stream"
        assert r.content == _STEP_FRAME
        result = decode_step_result(r.content)
        assert result.reward == 1.0
        assert result.terminated is True
        assert [(r.tool_call_id, r.images[-1] if r.images else None) for r in result.results] == [
            ("action_0", _SHOT)
        ]


async def test_step_typed_error_envelope_forwarded_and_client_reconstructs(tmp_path):
    payload = ModelOutputError(
        "malformed /step request body: body.actions.0: noncanonical Lite tool call",
        kind="noncanonical_tool_call",
        payload_metadata={
            "action_index": 0,
            "id": "call_bad",
            "debug": {"type": "bytes"},
        },
    ).to_payload()
    node = _FrameNode("node-a", step_error_payload=payload)
    async with _fleet(tmp_path, node) as (fleet, router):
        await fleet.poll_once()
        iid = await _create(router)
        r = await router.post(
            f"/instances/{iid}/step",
            headers=ALICE,
            json={"actions": [make_tool_call("computer", {}, call_id="call_bad")]},
        )

        assert r.status_code == 422, r.text
        assert r.headers["content-type"].startswith("application/json")
        assert r.json() == payload
        with pytest.raises(ModelOutputError) as exc_info:
            _raise_typed_error_if_any(r)
        exc = exc_info.value
        assert exc.kind == "noncanonical_tool_call"
        assert exc.payload_metadata == payload["payload_metadata"]
        assert failure_category(exc) is FailureCategory.MODEL_OUTPUT_ERROR
        assert exc.pairable is False
