"""Unit tests for the fleet router (:mod:`lite.gym.remote.fleet`).

All tests run fully in-process: each fake env-server node is a small FastAPI
app with programmable behaviors (201 create, 503 + Retry-After, token-scoped
lookup, down/fail-path simulation), mounted into the router's shared
``httpx.AsyncClient`` via per-host ``ASGITransport`` mounts — NO docker, NO
sockets. The router app itself is driven through ``httpx.ASGITransport``,
which does not run lifespan, so tests call ``fleet.poll_once()`` directly
instead of waiting on the background poller.

Run::

    uv run pytest tests/gym/remote/test_fleet_router.py -q
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from lite.gym.errors import InstanceGone, lite_error_from_payload
from lite.gym.remote.fleet import (
    FleetState,
    StaticProvider,
    make_fleet_app,
    make_provider,
)

ALICE = {"Authorization": "Bearer alice-token"}


# ---------------------------------------------------------------------------
# Fake env-server node
# ---------------------------------------------------------------------------


class FakeNode:
    """In-process env-server stand-in with programmable behaviors.

    Env transport shapes mirror :mod:`lite.gym.remote.server` where the router
    depends on them: 201 create body ``{"id": ...}``; 503 + Retry-After + typed
    body;
    typed 404 InstanceGone envelope; token-scoped instance ownership.
    Response bodies additionally carry ``"node": <name>`` so tests can assert
    WHICH node served a routed call.
    """

    def __init__(
        self,
        name: str,
        *,
        catalog: dict[str, dict[str, Any]] | None = None,
        tasks_payload: dict[str, Any] | None = None,
        create_behavior: Any = "ok",   # "ok" | "503" | (status, body)
        retry_after: int = 7,
    ) -> None:
        self.name = name
        self.catalog = catalog if catalog is not None else {
            "lite.demo": {"available": True, "n_tasks": 3, "splits": ["train"]},
        }
        self.tasks_payload = tasks_payload
        self.create_behavior = create_behavior
        self.retry_after = retry_after
        self.down = False                      # transport-level ConnectError on every call
        self.fail_paths: tuple[str, ...] = ()  # ConnectError only for these path prefixes
        self.envs_requires_token: str | None = None  # /envs 401s unless this bearer
        self.host_status_requires_token: str | None = None  # /host_status 401s unless this bearer
        self.instances: dict[str, dict[str, Any]] = {}
        self.create_auth: list[str | None] = []
        self.step_auth: list[str | None] = []
        self.host_auth: list[str | None] = []
        self.reset_calls = 0
        self.tasks_calls = 0
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        fake = self

        def _token(request: Request) -> str | None:
            auth = request.headers.get("authorization") or ""
            return auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else None

        def _own(id_: str, request: Request) -> JSONResponse | None:
            inst = fake.instances.get(id_)
            if inst is None:
                return JSONResponse(
                    status_code=404,
                    content={"error_type": "InstanceGone",
                             "what": f"unknown instance id={id_!r}"},
                )
            if inst["token"] != _token(request):
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"instance id={id_!r} owned by another token"},
                )
            return None

        @app.get("/host_status")
        async def host_status(request: Request) -> Any:
            fake.host_auth.append(request.headers.get("authorization"))
            if (fake.host_status_requires_token is not None
                    and _token(request) != fake.host_status_requires_token):
                return JSONResponse(status_code=401, content={"detail": "bad token"})
            return {"cua_lite": {"commit": f"{fake.name}-commit"}}

        @app.get("/envs")
        async def envs(request: Request, expand: str | None = None) -> Any:
            if (fake.envs_requires_token is not None
                    and _token(request) != fake.envs_requires_token):
                return JSONResponse(status_code=401,
                                    content={"detail": "invalid bearer token"})
            if expand == "metadata":
                return fake.catalog
            return sorted(fake.catalog)

        @app.get("/envs/{env_id}/tasks")
        async def tasks(env_id: str) -> dict[str, Any]:
            fake.tasks_calls += 1
            if fake.tasks_payload is not None:
                return fake.tasks_payload
            return {"splits": {"train": [f"{env_id}-t1"]}, "metadata": {},
                    "kwargs": {}, "env_make_kwargs": {}, "node": fake.name}

        @app.post("/instances")
        async def create(request: Request) -> JSONResponse:
            fake.create_auth.append(request.headers.get("authorization"))
            if fake.create_behavior == "503":
                return JSONResponse(
                    status_code=503,
                    headers={"Retry-After": str(fake.retry_after)},
                    content={"detail": f"{fake.name} full",
                             "what": f"{fake.name} full",
                             "retry_after_s": float(fake.retry_after)},
                )
            if isinstance(fake.create_behavior, tuple):
                status, body = fake.create_behavior
                return JSONResponse(status_code=status, content=body)
            body = await request.json()
            id_ = f"{fake.name}-{uuid.uuid4().hex[:8]}"
            fake.instances[id_] = {"token": _token(request), "env_key": body["env_key"]}
            return JSONResponse(status_code=201,
                                content={"id": id_, "metadata": None, "node": fake.name})

        @app.get("/instances")
        async def list_instances(request: Request) -> dict[str, Any]:
            tok = _token(request)
            return {"instances": [
                {"id": id_, "env_key": inst["env_key"], "node": fake.name}
                for id_, inst in fake.instances.items() if inst["token"] == tok
            ]}

        @app.delete("/instances")
        async def bulk_close(request: Request) -> dict[str, Any]:
            tok = _token(request)
            victims = [i for i, inst in fake.instances.items() if inst["token"] == tok]
            for id_ in victims:
                fake.instances.pop(id_)
            return {"closed": victims, "skipped_in_flight": []}

        @app.get("/instances/{id_}")
        async def get_instance(id_: str, request: Request) -> Any:
            err = _own(id_, request)
            if err is not None:
                return err
            return {"id": id_, "node": fake.name}

        @app.post("/instances/{id_}/reset")
        async def reset(id_: str, request: Request) -> Any:
            err = _own(id_, request)
            if err is not None:
                return err
            fake.reset_calls += 1
            return {"observation": {"text": fake.name}, "node": fake.name, "id": id_}

        @app.post("/instances/{id_}/step")
        async def step(id_: str, request: Request) -> Any:
            fake.step_auth.append(request.headers.get("authorization"))
            err = _own(id_, request)
            if err is not None:
                return err
            return {"observation": {"text": fake.name}, "node": fake.name, "id": id_}

        @app.delete("/instances/{id_}")
        async def close(id_: str, request: Request) -> Any:
            err = _own(id_, request)
            if err is not None:
                return err
            fake.instances.pop(id_)
            return {"ok": True}

        return app


class _FakeNodeTransport(httpx.AsyncBaseTransport):
    """ASGITransport wrapper that simulates transport-level failures via the
    fake's ``down`` / ``fail_paths`` flags."""

    def __init__(self, fake: FakeNode) -> None:
        self.fake = fake
        self.inner = httpx.ASGITransport(app=fake.app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.fake.down:
            raise httpx.ConnectError("connection refused")
        if any(request.url.path.startswith(p) for p in self.fake.fail_paths):
            raise httpx.ConnectError("connection refused")
        return await self.inner.handle_async_request(request)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _write_nodes_file(path, fakes: list[FakeNode]) -> None:
    path.write_text(
        "# fleet roster\n" + "".join(f"http://{f.name}\n" for f in fakes)
    )


@asynccontextmanager
async def fleet_env(
    tmp_path,
    fakes: list[FakeNode],
    *,
    roster: list[FakeNode] | None = None,
    **fleet_kwargs: Any,
):
    """Yield ``(fleet, router_client, nodes_file)`` over in-process fakes.

    ``roster`` limits which fakes appear in the initial nodes file (all are
    always mounted, so a later file rewrite can hot-join the rest).
    """
    nodes_file = tmp_path / "nodes.txt"
    _write_nodes_file(nodes_file, roster if roster is not None else fakes)
    node_client = httpx.AsyncClient(
        mounts={f"http://{f.name}": _FakeNodeTransport(f) for f in fakes},
    )
    fleet = FleetState(
        StaticProvider(nodes_file), client=node_client, **fleet_kwargs,
    )
    app = make_fleet_app(fleet)
    router = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router",
    )
    try:
        yield fleet, router, nodes_file
    finally:
        await router.aclose()
        await node_client.aclose()


async def _create(router: httpx.AsyncClient, env_key: str = "lite.demo@t1"):
    return await router.post(
        "/instances",
        json={"env_key": env_key, "env_kwargs": {}, "session_id": "s"},
        headers=ALICE,
    )


# ---------------------------------------------------------------------------
# Create routing
# ---------------------------------------------------------------------------


async def test_create_routes_to_least_outstanding(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        r1 = await _create(router)
        r2 = await _create(router)
        assert r1.status_code == r2.status_code == 201
        # Tie broken by name → node-a first; second create goes to the
        # now-less-loaded node-b.
        assert r1.json()["node"] == "node-a"
        assert r2.json()["node"] == "node-b"
        assert fleet.nodes["node-a"].outstanding == 1
        assert fleet.nodes["node-b"].outstanding == 1


async def test_create_503_cascades_to_next_node(tmp_path):
    a = FakeNode("node-a", create_behavior="503", retry_after=30)
    b = FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        r = await _create(router)
        assert r.status_code == 201
        assert r.json()["node"] == "node-b"
        # the 503 node is cooling down per its Retry-After
        assert fleet.nodes["node-a"].cooldown_until > time.monotonic()


async def test_all_503_returns_minimum_retry_after_verbatim(tmp_path):
    a = FakeNode("node-a", create_behavior="503", retry_after=30)
    b = FakeNode("node-b", create_behavior="503", retry_after=7)
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        r = await _create(router)
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "7"
        assert r.json()["what"] == "node-b full"   # that node's body, verbatim


async def test_cooldown_excludes_node_from_routing(tmp_path):
    a = FakeNode("node-a", create_behavior="503", retry_after=60)
    b = FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        assert (await _create(router)).json()["node"] == "node-b"
        a.create_behavior = "ok"  # node recovered, but its cooldown holds
        r = await _create(router)
        assert r.json()["node"] == "node-b"  # a is cooling despite 0 outstanding
        assert a.create_auth != []  # first attempt hit a; second must not
        assert len(a.create_auth) == 1


async def test_create_respects_catalog_availability(tmp_path):
    a = FakeNode("node-a", catalog={"env1": {"available": True, "n_tasks": 1,
                                             "splits": ["train"]}})
    b = FakeNode("node-b", catalog={
        "env1": {"available": False, "error": "deps missing"},
        "env2": {"available": True, "n_tasks": 2, "splits": ["train"]},
    })
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        assert (await _create(router, "env2@t")).json()["node"] == "node-b"
        assert (await _create(router, "env1@t")).json()["node"] == "node-a"


async def test_create_empty_catalog_cache_is_optimistically_eligible(tmp_path):
    a = FakeNode("node-a", catalog={})
    async with fleet_env(tmp_path, [a]) as (fleet, router, _):
        await fleet.poll_once()
        assert (await _create(router, "anything@t")).status_code == 201


async def test_create_terminal_error_returned_verbatim_without_cascade(tmp_path):
    a = FakeNode("node-a",
                 create_behavior=(400, {"error_type": "MakeFailed", "what": "boom"}))
    b = FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        r = await _create(router)
        assert r.status_code == 400
        assert r.json() == {"error_type": "MakeFailed", "what": "boom"}
        assert b.create_auth == []  # no cascade on a typed terminal rejection


async def test_create_skips_unhealthy_node(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        a.down = True
        await fleet.poll_once()
        assert not fleet.nodes["node-a"].healthy
        assert (await _create(router)).json()["node"] == "node-b"


# ---------------------------------------------------------------------------
# Sticky routing + recovery
# ---------------------------------------------------------------------------


async def test_sticky_reset_and_step_hit_owner_node(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        id_a = (await _create(router)).json()["id"]     # → node-a
        id_b = (await _create(router)).json()["id"]     # → node-b
        r = await router.post(f"/instances/{id_b}/reset", headers=ALICE)
        assert r.json()["node"] == "node-b"
        r = await router.post(f"/instances/{id_a}/step",
                              json={"actions": []}, headers=ALICE)
        assert r.json()["node"] == "node-a"
        assert a.reset_calls == 0 and b.reset_calls == 1


async def test_fanout_recovery_after_router_restart(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        _ = await _create(router)                       # → node-a
        id_b = (await _create(router)).json()["id"]     # → node-b
    # "Restart": a fresh FleetState/app over the SAME live fake nodes.
    async with fleet_env(tmp_path, [a, b]) as (fleet2, router2, _):
        await fleet2.poll_once()
        assert fleet2.instance_map == {}
        r = await router2.post(f"/instances/{id_b}/reset", headers=ALICE)
        assert r.status_code == 200
        assert r.json()["node"] == "node-b"
        assert fleet2.instance_map[id_b].name == "node-b"
        assert fleet2.counters["fanout_recovery_total"] == 1


async def test_fanout_all_miss_forwards_instance_gone_envelope(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        r = await router.post("/instances/nope/reset", headers=ALICE)
        assert r.status_code == 404
        exc = lite_error_from_payload(r.json())
        assert isinstance(exc, InstanceGone)  # client's re-make path fires


async def test_delete_drops_map_entry_and_outstanding(tmp_path):
    a = FakeNode("node-a")
    async with fleet_env(tmp_path, [a]) as (fleet, router, _):
        await fleet.poll_once()
        id_ = (await _create(router)).json()["id"]
        assert fleet.nodes["node-a"].outstanding == 1
        r = await router.delete(f"/instances/{id_}", headers=ALICE)
        assert r.status_code == 200 and r.json() == {"ok": True}
        assert fleet.instance_map == {}
        assert fleet.nodes["node-a"].outstanding == 0
        assert a.instances == {}


async def test_owner_dead_node_synthesizes_instance_gone(tmp_path):
    a = FakeNode("node-a")
    async with fleet_env(tmp_path, [a]) as (fleet, router, _):
        await fleet.poll_once()
        id_ = (await _create(router)).json()["id"]
        a.down = True  # forward fails AND the liveness probe fails
        r = await router.post(f"/instances/{id_}/step",
                              json={"actions": []}, headers=ALICE)
        assert r.status_code == 404
        payload = r.json()
        assert payload["error_type"] == "InstanceGone"
        assert "unreachable" in payload["what"]
        assert fleet.instance_map == {}
        assert not fleet.nodes["node-a"].healthy


async def test_owner_alive_but_request_failing_returns_502_not_gone(tmp_path):
    a = FakeNode("node-a")
    async with fleet_env(tmp_path, [a]) as (fleet, router, _):
        await fleet.poll_once()
        id_ = (await _create(router)).json()["id"]
        a.fail_paths = ("/instances/",)  # instance calls fail; /host_status is fine
        r = await router.post(f"/instances/{id_}/step",
                              json={"actions": []}, headers=ALICE)
        assert r.status_code == 502  # NOT a 404 — the instance is presumed alive
        assert id_ in fleet.instance_map  # sticky entry retained


# ---------------------------------------------------------------------------
# Catalog merge + tasks cache
# ---------------------------------------------------------------------------


async def test_envs_union_and_merge(tmp_path):
    a = FakeNode("node-a", catalog={"env1": {"available": True, "n_tasks": 2,
                                             "splits": ["train"]}})
    b = FakeNode("node-b", catalog={
        "env1": {"available": False, "error": "broken on b"},
        "env2": {"available": True, "n_tasks": 5, "splits": ["test"]},
    })
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        assert (await router.get("/envs")).json() == ["env1", "env2"]
        merged = (await router.get("/envs", params={"expand": "metadata"})).json()
        assert merged["env1"] == {"available": True, "n_tasks": 2, "splits": ["train"]}
        assert merged["env2"]["available"] is True


async def test_envs_error_concat_when_no_node_available(tmp_path):
    a = FakeNode("node-a", catalog={"env1": {"available": False, "error": "errA"}})
    b = FakeNode("node-b", catalog={"env1": {"available": False, "error": "errB"}})
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        merged = (await router.get("/envs", params={"expand": "metadata"})).json()
        assert merged["env1"]["available"] is False
        assert "node-a: errA" in merged["env1"]["error"]
        assert "node-b: errB" in merged["env1"]["error"]
        # single-env endpoint agrees; unknown env → 404
        single = (await router.get("/envs/env1")).json()
        assert single == merged["env1"]
        assert (await router.get("/envs/ghost")).status_code == 404


async def test_tasks_forwarded_then_cached(tmp_path):
    a = FakeNode("node-a", catalog={"env1": {"available": True, "n_tasks": 1,
                                             "splits": ["train"]}})
    async with fleet_env(tmp_path, [a]) as (fleet, router, _):
        await fleet.poll_once()
        r1 = await router.get("/envs/env1/tasks", headers=ALICE)
        r2 = await router.get("/envs/env1/tasks", headers=ALICE)
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()
        assert a.tasks_calls == 1  # second hit served from the TTL cache


async def test_tasks_cache_preserves_tagged_metadata_payload(tmp_path):
    payload = {
        "splits": {"train": ["t1", "t2"]},
        "metadata": {
            "t1": {
                "metadata_kind": "cua",
                "dims": ["browser", "use"],
                "extra_tool_schemas": [],
                "valid_actions": None,
                "others": {"source": "cua"},
            },
            "t2": {
                "metadata_kind": "generic",
                "dims": [],
                "extra_tool_schemas": [
                    {
                        "type": "function",
                        "function": {
                            "name": "answer",
                            "description": "",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                ],
                "others": {"source": "generic"},
            },
        },
        "kwargs": {},
        "env_make_kwargs": {},
        "node": "node-a",
    }
    a = FakeNode(
        "node-a",
        catalog={"env1": {"available": True, "n_tasks": 2, "splits": ["train"]}},
        tasks_payload=payload,
    )
    async with fleet_env(tmp_path, [a]) as (fleet, router, _):
        await fleet.poll_once()
        r1 = await router.get("/envs/env1/tasks", headers=ALICE)
        r2 = await router.get("/envs/env1/tasks", headers=ALICE)
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == payload
        assert r2.json() == payload
        assert a.tasks_calls == 1


# ---------------------------------------------------------------------------
# Fan-out list / bulk
# ---------------------------------------------------------------------------


async def test_list_instances_fans_out_and_merges(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        ids = {(await _create(router)).json()["id"] for _ in range(2)}
        r = await router.get("/instances", headers=ALICE)
        assert {row["id"] for row in r.json()["instances"]} == ids
        assert {row["node"] for row in r.json()["instances"]} == {"node-a", "node-b"}


async def test_bulk_delete_fans_out_and_merges(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        ids = {(await _create(router)).json()["id"] for _ in range(2)}
        r = await router.delete("/instances", params={"force": "true"}, headers=ALICE)
        assert set(r.json()["closed"]) == ids
        assert r.json()["skipped_in_flight"] == []
        assert a.instances == {} and b.instances == {}


# ---------------------------------------------------------------------------
# Provider + poller
# ---------------------------------------------------------------------------


async def test_nodes_file_mtime_reread_hot_joins_node(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b], roster=[a]) as (fleet, router, nodes_file):
        await fleet.poll_once()
        assert set(fleet.nodes) == {"node-a"}
        _write_nodes_file(nodes_file, [a, b])
        st = nodes_file.stat()
        os.utime(nodes_file, (st.st_atime, st.st_mtime + 10))  # force mtime change
        await fleet.poll_once()
        assert set(fleet.nodes) == {"node-a", "node-b"}
        names = {n["name"] for n in (await router.get("/fleet/nodes")).json()["nodes"]}
        assert names == {"node-a", "node-b"}
        assert fleet.nodes["node-b"].healthy


def test_slurm_provider_exits_loudly(tmp_path):
    with pytest.raises(SystemExit, match="not implemented"):
        make_provider("slurm", None)
    with pytest.raises(SystemExit, match="requires --nodes-file"):
        make_provider("static", None)


async def test_probe_401_excludes_node_from_routing(tmp_path):
    """Auth mismatch surfaces via the regular probe (the join-preflight
    machinery was folded in): 401 → unhealthy + a loud token-hint error."""
    a = FakeNode("node-a")
    a.host_status_requires_token = "sekrit"  # ≠ the router's default node token
    b = FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        assert not fleet.nodes["node-a"].healthy
        assert "401" in fleet.nodes["node-a"].last_error
        assert "node-token" in fleet.nodes["node-a"].last_error
        assert (await _create(router)).json()["node"] == "node-b"


# ---------------------------------------------------------------------------
# Auth forwarding + operator surfaces
# ---------------------------------------------------------------------------


async def test_bearer_forwarded_verbatim_and_node_token_for_polling(tmp_path):
    a = FakeNode("node-a")
    async with fleet_env(tmp_path, [a], node_token="poller-tok") as (fleet, router, _):
        await fleet.poll_once()
        id_ = (await _create(router)).json()["id"]
        await router.post(f"/instances/{id_}/step", json={"actions": []}, headers=ALICE)
        # Client calls carry the CALLER'S bearer, byte-identical.
        assert a.create_auth == ["Bearer alice-token"]
        assert a.step_auth == ["Bearer alice-token"]
        # The poller's own probes carry the router's --node-token.
        assert "Bearer poller-tok" in a.host_auth


async def test_host_status_fleet_key_and_admin_501(tmp_path):
    a, b = FakeNode("node-a"), FakeNode("node-b")
    async with fleet_env(tmp_path, [a, b]) as (fleet, router, _):
        await fleet.poll_once()
        await _create(router)
        hs = (await router.get("/host_status")).json()
        assert hs["fleet"] == {"nodes": 2, "healthy": 2, "instances": 1}
        r = await router.get("/admin/budget")
        assert r.status_code == 501
        assert "http://node-a" in r.json()["detail"]
        m = await router.get("/metrics")
        assert "cua_lite_fleet_nodes_healthy 2" in m.text
        assert "cua_lite_fleet_create_total 1" in m.text


def test_proxied_response_has_single_date_server_headers():
    """Live-drill finding: the node's Date/Server passed through and uvicorn
    appended its own — every proxied response carried duplicates. The forward
    filter must drop the upstream pair (per-hop headers in practice)."""
    import httpx

    from lite.gym.remote.fleet import _forward_response_headers
    upstream = httpx.Response(
        200,
        headers={"date": "Mon, 01 Jan 2029 00:00:00 GMT", "server": "uvicorn",
                 "x-custom": "kept"},
        content=b"{}",
        request=httpx.Request("GET", "http://node/envs"),
    )
    fwd = _forward_response_headers(upstream)
    assert "date" not in {k.lower() for k in fwd}
    assert "server" not in {k.lower() for k in fwd}
    assert fwd.get("x-custom") == "kept"


async def test_read_phase_failure_no_retry_returns_502(tmp_path):
    """A read-phase transport failure (node may have already EXECUTED the
    step) must NOT be resent — one attempt, then the alive-node 502 envelope.
    Only connect-phase failures get the bounded resend."""
    import httpx as _httpx

    a = FakeNode("node-a")
    async with fleet_env(tmp_path, [a]) as (fleet, router, _):
        await fleet.poll_once()
        created = (await _create(router)).json()
        sends = {"n": 0}
        real_send = fleet.client.send

        async def _read_fails_once(req, **kw):
            if "/reset" in str(req.url):
                sends["n"] += 1
                raise _httpx.ReadError("connection broken mid-body")
            return await real_send(req, **kw)

        fleet.client.send = _read_fails_once
        try:
            r = await router.post(
                f"/instances/{created['id']}/reset",
                headers={"Authorization": "Bearer drill"},
            )
        finally:
            fleet.client.send = real_send
        assert sends["n"] == 1, "read-phase failure must not be resent"
        assert r.status_code == 502
        body = r.json()
        assert body["error_type"] == "ReadError"
