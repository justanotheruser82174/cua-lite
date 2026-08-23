"""Hermetic direct/server parity matrix for the env construction boundary.

The fake env here is intentionally local and deterministic: no Docker, no web
services, no cursor implementation. The assertion is that one task/config
produces the same caller-visible metadata, reset observation, and step result
when reached through direct ``gym.make`` or through the in-process env-server.

Run:
    uv run pytest tests/gym/remote/test_direct_server_parity_matrix.py -q
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from fastapi.testclient import TestClient
from gym.remote.conftest import bearer_header as _h
from gym.remote.conftest import make_test_admission

import lite.gym as gym
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools import make_tool_call, make_tool_schema
from lite.core.tools.calls import tool_call_id, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.gym.base import LiteBaseEnv
from lite.gym.registry import (
    _env_make_kwargs,
    _imported,
    _specs,
    _splits,
    register,
    registry,
)
from lite.gym.remote.frame import decode_reset_observation, decode_step_result
from lite.gym.remote.server import State, make_app
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.ingress import prepare_env_tool_calls
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)

_ENV_ID = "_parity_matrix"
_TASK_ID = "task_a"
_KEY = f"{_ENV_ID}@{_TASK_ID}"
_TOKEN = "parity-token"
_SESSION_ID = "parity-session"
_SERVER_PORT = 31337
_RESET_IMAGE = b"reset-image-bytes"
_RESULT_IMAGE = b"result-image-bytes"


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(data))


def _obs_payload(obs: LiteEnvObservation) -> dict[str, Any]:
    return {
        "image": obs.image,
        "text": obs.text,
        "metadata": obs.metadata,
    }


def _step_payload(result: LiteEnvStepResult) -> dict[str, Any]:
    return {
        "reward": result.reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info,
        "results": [
            {
                "tool_call_id": item.tool_call_id,
                "images": item.images,
                "text": item.text,
                "metadata": item.metadata,
                "error": item.error,
            }
            for item in result.results
        ],
    }


class _ParityMatrixEnv(LiteBaseEnv):
    def __init__(
        self,
        *,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = None,
        cursor: bool = True,
        post_action_delay: float = 0.0,
        profile: str = "default",
    ) -> None:
        self._valid_actions = valid_actions
        self._extra_tools = list(extra_tools or [])
        self._cursor = cursor
        self._post_action_delay = post_action_delay
        self._profile = profile
        self.seen_actions: list[list[dict[str, Any]]] = []

    def _runtime_metadata(self) -> LiteCUAMetadata:
        identity = getattr(self, "identity", None)
        return LiteCUAMetadata(
            dims=("browser", "use"),
            valid_actions=self._valid_actions,
            extra_tool_schemas=[
                make_tool_schema(
                    name,
                    description=f"{name} tool",
                    parameters={"type": "object", "properties": {}},
                )
                for name in self._extra_tools
            ],
            others={
                "cursor": self._cursor,
                "post_action_delay": self._post_action_delay,
                "profile": self._profile,
                "session_id": (
                    identity.resolved_session_id() if identity is not None else None
                ),
            },
        )

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(
            image=_RESET_IMAGE,
            text=f"reset:{self._profile}",
            metadata={
                "cursor": self._cursor,
                "post_action_delay": self._post_action_delay,
                "profile": self._profile,
            },
        )

    async def step(self, actions: list[dict[str, Any]]) -> LiteEnvStepResult:
        self.seen_actions.append(actions)
        if self._profile == "ingress-feedback":
            ordered_call_ids = ordered_tool_call_ids(actions)
            _prepared, feedback = prepare_env_tool_calls(actions, self.metadata)
            return build_tool_results_from_decisions(
                LiteEnvStepResult(
                    reward=0.25,
                    terminated=False,
                    truncated=False,
                    info={"profile": self._profile},
                ),
                ordered_call_ids=ordered_call_ids,
                continue_call_ids=ordered_call_ids,
                feedback=feedback,
            )

        call_ids = [
            call_id
            for action in actions
            if isinstance(call_id := tool_call_id(action), str)
        ]
        return LiteEnvStepResult(
            results=[
                LiteToolResult(
                    tool_call_id=call_ids[0] if call_ids else None,
                    images=[_RESULT_IMAGE],
                    text=f"stepped:{self._profile}",
                    metadata={
                        "action_names": [tool_call_name(action) for action in actions],
                        "cursor": self._cursor,
                    },
                ),
                LiteToolResult(
                    tool_call_id=call_ids[1] if len(call_ids) > 1 else None,
                    text="soft warning",
                    error="synthetic-error",
                    metadata={"ordinal": 1},
                ),
            ],
            reward=0.5,
            terminated=True,
            truncated=False,
            info={
                "seen_call_ids": call_ids,
                "post_action_delay": self._post_action_delay,
            },
        )

    async def close(self) -> None:
        pass


@pytest.fixture()
def parity_env(monkeypatch):
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_PORT", raising=False)
    _imported[_ENV_ID] = "local"
    register(
        _KEY,
        entry_point=_ParityMatrixEnv,
        split="eval",
        metadata=LiteCUAMetadata(dims=("browser", "use")),
        step_timeout=180.0,
        reset_timeout=45.0,
    )
    registry.set_env_make_kwargs(_ENV_ID, {"cursor": True})
    try:
        yield
    finally:
        _specs.pop(_KEY, None)
        _splits.pop(_ENV_ID, None)
        _env_make_kwargs.pop(_ENV_ID, None)
        _imported.pop(_ENV_ID, None)


def _make_server() -> tuple[State, TestClient]:
    state = State(
        admission=make_test_admission(max_live_envs=4),
        idle_ttl_sec=3600.0,
    )
    app = make_app(state, token=_TOKEN, port=_SERVER_PORT)
    return state, TestClient(app)


def test_direct_server_metadata_observation_and_result_match(parity_env):
    env_kwargs = {
        "valid_actions": ["click", "type"],
        "extra_tools": ["response"],
        "cursor": False,
        "post_action_delay": 0.125,
        "profile": "matrix",
    }
    actions = [
        make_tool_call("click", {"x": 10, "y": 20}, call_id="call_a"),
        make_tool_call("type", {"text": "ok"}, call_id="call_b"),
    ]

    direct_env = gym.make(_KEY, session_id=_SESSION_ID, **env_kwargs)
    try:
        direct_metadata = _jsonable(direct_env.metadata.to_dict())
        direct_obs = asyncio.run(direct_env.reset())
        direct_step = asyncio.run(direct_env.step(actions))
    finally:
        asyncio.run(direct_env.close())

    state, client = _make_server()
    response = client.post(
        "/instances",
        headers=_h(_TOKEN),
        json={
            "env_key": _KEY,
            "env_kwargs": env_kwargs,
            "session_id": _SESSION_ID,
        },
    )
    assert response.status_code == 201, response.text
    instance_id = response.json()["id"]
    try:
        assert response.json()["metadata"] == direct_metadata

        reset_response = client.post(
            f"/instances/{instance_id}/reset",
            headers=_h(_TOKEN),
        )
        assert reset_response.status_code == 200, reset_response.text
        server_obs = decode_reset_observation(reset_response.content)
        assert _obs_payload(server_obs) == _obs_payload(direct_obs)

        step_response = client.post(
            f"/instances/{instance_id}/step",
            headers=_h(_TOKEN),
            json={"actions": actions},
        )
        assert step_response.status_code == 200, step_response.text
        server_step = decode_step_result(step_response.content)
        assert _step_payload(server_step) == _step_payload(direct_step)
    finally:
        client.delete(
            f"/instances/{instance_id}",
            headers=_h(_TOKEN),
        )

    assert state.envs == {}


def test_env_make_kwargs_defaults_match_direct_and_server(parity_env):
    registry.set_env_make_kwargs(_ENV_ID, {"cursor": False})
    env_kwargs = {"valid_actions": ["click"], "profile": "default-cursor"}

    direct_env = gym.make(_KEY, session_id=_SESSION_ID, **env_kwargs)
    try:
        direct_metadata = _jsonable(direct_env.metadata.to_dict())
        direct_obs = asyncio.run(direct_env.reset())
    finally:
        asyncio.run(direct_env.close())

    assert direct_metadata["others"]["cursor"] is False
    assert direct_obs.metadata["cursor"] is False

    state, client = _make_server()
    response = client.post(
        "/instances",
        headers=_h(_TOKEN),
        json={
            "env_key": _KEY,
            "env_kwargs": env_kwargs,
            "session_id": _SESSION_ID,
        },
    )
    assert response.status_code == 201, response.text
    instance_id = response.json()["id"]
    try:
        assert response.json()["metadata"] == direct_metadata

        reset_response = client.post(
            f"/instances/{instance_id}/reset",
            headers=_h(_TOKEN),
        )
        assert reset_response.status_code == 200, reset_response.text
        server_obs = decode_reset_observation(reset_response.content)
        assert _obs_payload(server_obs) == _obs_payload(direct_obs)
        assert server_obs.metadata["cursor"] is False
    finally:
        client.delete(
            f"/instances/{instance_id}",
            headers=_h(_TOKEN),
        )

    assert state.envs == {}


def test_server_construction_ignores_ambient_client_url(parity_env, monkeypatch):
    """Env-server construction stays local even if it inherits client routing env."""
    monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://127.0.0.1:9")

    import lite.gym.remote as remote_pkg
    import lite.gym.remote.client as remote_client

    def _fail_remote(*_args, **_kwargs):
        raise AssertionError("env-server attempted to self-route through remote client")

    monkeypatch.setattr(remote_pkg, "LiteEnvClient", _fail_remote)
    monkeypatch.setattr(remote_client, "register_from_server", _fail_remote)

    state, client = _make_server()
    response = client.post(
        "/instances",
        headers=_h(_TOKEN),
        json={
            "env_key": _KEY,
            "env_kwargs": {"profile": "ambient-url"},
            "session_id": _SESSION_ID,
        },
    )
    assert response.status_code == 201, response.text
    instance_id = response.json()["id"]
    try:
        assert state.envs[instance_id].env.unwrapped._profile == "ambient-url"
    finally:
        client.delete(
            f"/instances/{instance_id}",
            headers=_h(_TOKEN),
        )

    assert state.envs == {}


def test_direct_server_missing_call_id_actions_match(parity_env):
    env_kwargs = {
        "valid_actions": ["wait"],
        "cursor": True,
        "profile": "missing-call-id",
    }
    actions = [make_tool_call("wait")]

    direct_env = gym.make(_KEY, session_id=_SESSION_ID, **env_kwargs)
    try:
        asyncio.run(direct_env.reset())
        direct_step = asyncio.run(direct_env.step(actions))
    finally:
        asyncio.run(direct_env.close())

    state, client = _make_server()
    response = client.post(
        "/instances",
        headers=_h(_TOKEN),
        json={
            "env_key": _KEY,
            "env_kwargs": env_kwargs,
            "session_id": _SESSION_ID,
        },
    )
    assert response.status_code == 201, response.text
    instance_id = response.json()["id"]
    try:
        reset_response = client.post(
            f"/instances/{instance_id}/reset",
            headers=_h(_TOKEN),
        )
        assert reset_response.status_code == 200, reset_response.text

        step_response = client.post(
            f"/instances/{instance_id}/step",
            headers=_h(_TOKEN),
            json={"actions": actions},
        )
        assert step_response.status_code == 200, step_response.text
        server_step = decode_step_result(step_response.content)
        assert _step_payload(server_step) == _step_payload(direct_step)
    finally:
        client.delete(
            f"/instances/{instance_id}",
            headers=_h(_TOKEN),
        )

    assert state.envs == {}


@pytest.mark.parametrize(
    ("action", "expected_error"),
    [
        (
            {
                "id": "call_bad",
                "type": "function",
                "function": {"name": "computer", "arguments": ["bad"]},
            },
            "tool_call.function.arguments must be an object, got list",
        ),
        (
            {
                "id": "call_missing_name",
                "type": "function",
                "function": {"arguments": {}},
            },
            "tool_call.function.name must be a non-empty string",
        ),
        (
            {
                "id": "call_missing_args",
                "type": "function",
                "function": {"name": "computer"},
            },
            "tool_call.function missing arguments dict",
        ),
    ],
)
def test_server_returns_env_feedback_for_pairable_malformed_actions(
    parity_env,
    action: dict[str, Any],
    expected_error: str,
):
    env_kwargs = {
        "valid_actions": ["click"],
        "cursor": True,
        "profile": "ingress-feedback",
    }
    actions = [action]

    state, client = _make_server()
    response = client.post(
        "/instances",
        headers=_h(_TOKEN),
        json={
            "env_key": _KEY,
            "env_kwargs": env_kwargs,
            "session_id": _SESSION_ID,
        },
    )
    assert response.status_code == 201, response.text
    instance_id = response.json()["id"]
    try:
        reset_response = client.post(
            f"/instances/{instance_id}/reset",
            headers=_h(_TOKEN),
        )
        assert reset_response.status_code == 200, reset_response.text

        step_response = client.post(
            f"/instances/{instance_id}/step",
            headers=_h(_TOKEN),
            json={"actions": actions},
        )
        assert step_response.status_code == 200, step_response.text
        step = decode_step_result(step_response.content)
        assert len(step.results) == 1
        result = step.results[0]
        assert result.tool_call_id == action["id"]
        assert result.error is not None and expected_error in result.error
        assert result.metadata and result.metadata["is_error"] is True
        assert state.envs[instance_id].env.unwrapped.seen_actions == [actions]
    finally:
        client.delete(
            f"/instances/{instance_id}",
            headers=_h(_TOKEN),
        )

    assert state.envs == {}


def test_server_identity_is_separate_from_instance_env_kwargs(parity_env):
    env_kwargs = {
        "valid_actions": ["click"],
        "extra_tools": ["response"],
        "cursor": False,
        "post_action_delay": 0.25,
        "profile": "server-kwargs",
    }
    state, client = _make_server()

    response = client.post(
        "/instances",
        headers=_h(_TOKEN),
        json={
            "env_key": _KEY,
            "env_kwargs": env_kwargs,
            "session_id": _SESSION_ID,
        },
    )
    assert response.status_code == 201, response.text
    instance_id = response.json()["id"]
    try:
        session = state.envs[instance_id]
        assert session.env_kwargs == env_kwargs
        assert "token_hash" not in session.env_kwargs
        assert "server_port" not in session.env_kwargs
        assert "session_id" not in session.env_kwargs

        identity = session.env.unwrapped.identity
        assert identity is not None
        assert identity.session_id == _SESSION_ID
        assert identity.token_hash == hashlib.sha256(_TOKEN.encode()).hexdigest()[:6]
        assert identity.server_port == _SERVER_PORT
    finally:
        client.delete(
            f"/instances/{instance_id}",
            headers=_h(_TOKEN),
        )
