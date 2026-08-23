"""Rollout error-file formatting and failed-sample accounting tests."""

from __future__ import annotations

import httpx

from lite.gym.errors import CuaGymTaskError
from lite.gym.remote import client as remote_client
from lite.infer import rollout as rollout_mod


def test_rollout_error_file_includes_typed_remote_422_payload():
    payload = CuaGymTaskError(
        "lite.cuagym desktop reward failed: REWARD sentinel missing",
        phase="reward",
        kind="no_reward",
    ).to_payload()
    response = httpx.Response(
        422,
        json=payload,
        request=httpx.Request("POST", "http://server/instances/i/step"),
    )

    try:
        remote_client._raise_typed_error_if_any(response)
    except CuaGymTaskError as exc:
        report = rollout_mod._format_exception_for_error_file(exc)
    else:  # pragma: no cover - defensive, typed helper must raise here
        raise AssertionError("expected CuaGymTaskError")

    assert "LiteGymError payload:" in report
    assert "error_type='CuaGymTaskError'" in report
    assert "kind='no_reward'" in report
    assert "phase='reward'" in report
    assert "REWARD sentinel missing" in report


def test_error_result_is_excluded_from_valid_mean():
    scored = rollout_mod._make_result("t", 0, 0, episode_return=1.0)
    refused = rollout_mod._make_result("t", 0, 1, error="CuaGymTaskError: excluded_task")

    assert [r for r in (scored, refused) if r["error"] is None] == [scored]
