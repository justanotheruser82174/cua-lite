"""Gemini ``generateContent`` transport boundary.

Hermetic: every request is served by an ``httpx.MockTransport``; nothing here
touches the network.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from lite.agents.models.gemini.utils.transport import (
    API_KEY_HEADER,
    GENERATE_CONTENT_PATH,
    GeminiTransportError,
    agenerate_content,
)

PAYLOAD: dict[str, Any] = {
    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    "tools": [{"computerUse": {"environment": "ENVIRONMENT_DESKTOP"}}],
}


def _patch_client(monkeypatch, handler) -> list[httpx.Request]:
    """Route every AsyncClient through ``handler``; return the seen requests."""
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    original = httpx.AsyncClient.__init__

    def _init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_record)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _init)
    return seen


async def test_url_and_auth_header(monkeypatch):
    """The one working path spelling, and the header (not a bearer token)."""
    seen = _patch_client(monkeypatch, lambda _r: httpx.Response(200, json={"candidates": []}))

    await agenerate_content(
        model="gemini-3.6-flash",
        api_base="https://proxy.example.com/",
        api_key="sk-test",
        payload=PAYLOAD,
        timeout=30.0,
    )

    request = seen[0]
    assert request.url.path == GENERATE_CONTENT_PATH.format(model="gemini-3.6-flash")
    # Trailing slash on api_base must not double up.
    assert "//v1beta" not in str(request.url)
    assert request.headers[API_KEY_HEADER] == "sk-test"
    assert "authorization" not in {k.lower() for k in request.headers}


async def test_returns_parsed_json_unmodified(monkeypatch):
    """Transport does not normalize -- parse.py owns structural judgement."""
    body = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]}
    _patch_client(monkeypatch, lambda _r: httpx.Response(200, json=body))

    got = await agenerate_content(
        model="m", api_base="https://p", api_key="k", payload=PAYLOAD, timeout=1.0
    )

    assert got == body


async def test_missing_api_base_is_rejected_at_call_time(monkeypatch):
    """No usable default exists; failing here beats POSTing to ``None/v1beta``."""
    _patch_client(monkeypatch, lambda _r: httpx.Response(200, json={}))

    with pytest.raises(GeminiTransportError, match="api_base is required"):
        await agenerate_content(model="m", api_base=None, api_key="k", payload=PAYLOAD, timeout=1.0)


async def test_non_2xx_carries_the_request_shape(monkeypatch):
    """The proxy collapses several server-side rejections into an opaque 500.

    Without the excluded-verb list and the part-key set in the message, a bad
    ``excludedPredefinedFunctions`` entry and an unstripped image marker are
    indistinguishable in a rollout log.
    """
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {"mimeType": "image/png", "data": "x"},
                        "_cua_lite_image_index": 3,
                    }
                ],
            }
        ],
        "tools": [
            {
                "computerUse": {
                    "environment": "ENVIRONMENT_DESKTOP",
                    "excludedPredefinedFunctions": ["navigate", "list_apps"],
                }
            }
        ],
    }
    _patch_client(
        monkeypatch,
        lambda _r: httpx.Response(500, text="Internal Server Error"),
    )

    with pytest.raises(GeminiTransportError) as excinfo:
        await agenerate_content(
            model="m", api_base="https://p", api_key="k", payload=payload, timeout=1.0
        )

    message = str(excinfo.value)
    assert "500" in message
    assert "navigate" in message and "list_apps" in message
    assert "_cua_lite_image_index" in message
    assert "inlineData" in message


async def test_2xx_with_non_json_body_is_a_bug(monkeypatch):
    _patch_client(monkeypatch, lambda _r: httpx.Response(200, text="not json"))

    with pytest.raises(GeminiTransportError, match="non-JSON"):
        await agenerate_content(
            model="m", api_base="https://p", api_key="k", payload=PAYLOAD, timeout=1.0
        )


async def test_shape_summary_names_the_excluded_verbs(monkeypatch):
    """The proxy collapses several distinct rejections into an opaque 500, so
    the exclusion list has to ride in the error message to tell them apart."""
    payload = {
        "contents": [],
        "tools": [{"computerUse": {"excludedPredefinedFunctions": ["hotkey"]}}],
    }
    _patch_client(monkeypatch, lambda _r: httpx.Response(500, text="boom"))

    with pytest.raises(GeminiTransportError, match="hotkey"):
        await agenerate_content(
            model="m", api_base="https://p", api_key="k", payload=payload, timeout=1.0
        )
