"""T5e (metadata contract section 11): the factory's client-side identity attr writes are INERT
on ``LiteEnvClient`` — its ``metadata`` property returns the server-computed
copy verbatim (identity appears once, value == the server copy, no re-merge).

``gym.make`` writes ``env.env_id``/``env.task_id`` on BOTH factory branches
(lite/gym/factory.py); on a direct env the base property merges them into
``metadata.others``, but the client overrides the property to serve the server's
authoritative copy — the attr writes must change nothing.

Real ``LiteEnvClient`` (actual httpx transport) against an in-process uvicorn
mock env-server (shared harness in gym.remote.conftest) — needs NO dev
server or docker, but binds a real TCP port + uvicorn, so it carries the
``live`` mark like its sibling test_capacity_503_retry.py: the repo
quarantines socket-serving tests out of the default -n 16 suite (they flake
under load), not because a server is required.

Run: uv run pytest tests/gym/remote/test_client_identity_inertness.py -m live -q
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from gym.remote.conftest import running_mock_server

from lite.core import LiteCUAMetadata
from lite.gym.remote.client import LiteEnvClient

pytestmark = pytest.mark.live

#: What the real server returns from POST /instances: metadata with the
#: SERVER-injected identity under ``metadata.others`` (server.py create path).
_SERVER_METADATA: dict[str, Any] = {
    **LiteCUAMetadata(
        dims=("desktop", "use"),
        valid_actions=None,
        extra_tool_schemas=[],
        others={"env_id": "demo", "task_id": "task-1", "flavor": "server-truth"},
    ).to_dict(),
}


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.post("/instances", status_code=201)
    async def create(_body: dict[str, Any]) -> dict[str, Any]:
        return {"id": "inst-1", "metadata": dict(_SERVER_METADATA)}

    @app.delete("/instances/{id_}")
    async def delete(id_: str) -> dict[str, Any]:
        return {"ok": True}

    return app


def test_t5e_factory_identity_writes_are_inert_on_client() -> None:
    with running_mock_server(_build_app()) as url:
        async def _go() -> None:
            client = LiteEnvClient(url, "demo@task-1")
            try:
                server_md = client.metadata
                # Eager-create parsed the server copy, identity included.
                assert server_md.others["env_id"] == "demo"
                assert server_md.others["task_id"] == "task-1"
                # The gym.make step (factory.py): client-side attr writes.
                client.env_id = "demo"
                client.task_id = "task-1"
                md = client.metadata
                # Inert: same object, no re-merge/rebuild, value == server copy.
                assert md is server_md, "attr writes must not rebuild client metadata"
                assert md.to_dict() == _SERVER_METADATA, "value must equal the server copy"
                # Stable across reads (identity once — never double-injected).
                assert client.metadata is server_md
            finally:
                await client.close()

        asyncio.run(_go())
