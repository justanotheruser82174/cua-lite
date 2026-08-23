"""Per-call ``sampling_seed`` merge + ``finish_reason`` passthrough for the generate fns.

``make_server_generate_fn`` / ``make_engine_generate_fn`` (``lite.infer.serving``)
freeze a ``sampling_kwargs`` dict at construction. The rollout collect path wraps
the returned generate fn to pass a per-sample ``sampling_seed`` (member-distinct,
for reproducibility) — see ``lite/infer/rollout.py::_resolve_agent_kwargs``. The
inner ``generate`` must:

  * ``sampling_seed=None`` (no wrapper)          → payload unchanged (back-compat);
  * per-call ``sampling_seed`` + no pinned seed  → merged into ``sampling_params``;
  * a caller-PINNED ``sampling_kwargs["sampling_seed"]`` → WINS over the per-call
    one (mirrors the training path's ``"sampling_seed" not in sampling_params``
    guard in ``engine.py``).

All three backends must also hand a ``finish_reason`` to the caller:
``LiteBaseAgent`` maps ``"length"`` to ``STATUS_TRUNCATED``, and if the key
never arrives a truncated turn is indistinguishable from a clean stop. The two
sglang backends read the provider's own field; ``make_hf_generate_fn`` has no
provider field to read and derives it from the last generated token id.

These run on the host venv — every backend is mocked (no server / no Engine /
no model weights).

Run:
    uv run pytest tests/infer/serving/test_sampling_seed.py -q
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from lite.infer.serving import (
    make_engine_generate_fn,
    make_hf_generate_fn,
    make_server_generate_fn,
)


def _body(finish_type: str = "stop") -> dict:
    """A sglang /generate response body (both backends return this shape)."""
    return {"text": "ok", "meta_info": {"finish_reason": {"type": finish_type}}}


class _Resp:
    def __init__(self, finish_type: str = "stop") -> None:
        self._finish_type = finish_type

    def raise_for_status(self) -> None:  # noqa: D102
        pass

    def json(self) -> dict:  # noqa: D102
        return _body(self._finish_type)


def _server_capture(sampling_kwargs: dict, finish_type: str = "stop"):
    """Build make_server_generate_fn with httpx mocked; return (generate, captured)."""
    captured: dict = {}

    async def _post(url, json):  # noqa: A002 - matches httpx.post(json=...)
        captured["payload"] = json
        return _Resp(finish_type)

    mock_client = MagicMock()
    mock_client.post = _post
    with patch("lite.infer.serving.httpx.AsyncClient", return_value=mock_client):
        gen = make_server_generate_fn("http://x", sampling_kwargs)
    return gen, captured


def _engine_capture(sampling_kwargs: dict, finish_type: str = "stop"):
    """Build make_engine_generate_fn with a mock Engine; return (generate, captured)."""
    captured: dict = {}

    async def _async_generate(prompt, sampling_params, image_data):
        captured["sampling_params"] = sampling_params
        return _body(finish_type)

    engine = MagicMock()
    engine.async_generate = _async_generate
    return make_engine_generate_fn(engine, sampling_kwargs), captured


# --- server path -----------------------------------------------------------

def test_server_no_seed_is_unchanged():
    gen, cap = _server_capture({"temperature": 0.7})
    asyncio.run(gen(prompt="p", images=[]))
    assert cap["payload"]["sampling_params"] == {"temperature": 0.7}


def test_server_per_call_seed_merged():
    gen, cap = _server_capture({"temperature": 0.7})
    asyncio.run(gen(prompt="p", images=[], sampling_seed=123))
    assert cap["payload"]["sampling_params"] == {"temperature": 0.7, "sampling_seed": 123}


def test_server_pinned_seed_wins():
    gen, cap = _server_capture({"temperature": 0.7, "sampling_seed": 999})
    asyncio.run(gen(prompt="p", images=[], sampling_seed=123))
    assert cap["payload"]["sampling_params"]["sampling_seed"] == 999


# --- engine path -----------------------------------------------------------

def test_engine_no_seed_is_unchanged():
    gen, cap = _engine_capture({"temperature": 0.7})
    asyncio.run(gen(prompt="p", images=[]))
    assert cap["sampling_params"] == {"temperature": 0.7}


def test_engine_per_call_seed_merged():
    gen, cap = _engine_capture({"temperature": 0.7})
    asyncio.run(gen(prompt="p", images=[], sampling_seed=123))
    assert cap["sampling_params"] == {"temperature": 0.7, "sampling_seed": 123}


def test_engine_pinned_seed_wins():
    gen, cap = _engine_capture({"temperature": 0.7, "sampling_seed": 999})
    asyncio.run(gen(prompt="p", images=[], sampling_seed=123))
    assert cap["sampling_params"]["sampling_seed"] == 999


# --- finish_reason passthrough (both sglang backends) ----------------------

@pytest.mark.parametrize("finish_type", ["stop", "length"])
def test_server_finish_reason_survives(finish_type):
    gen, _ = _server_capture({"temperature": 0.7}, finish_type)
    assert asyncio.run(gen(prompt="p", images=[]))["finish_reason"] == finish_type


@pytest.mark.parametrize("finish_type", ["stop", "length"])
def test_engine_finish_reason_survives(finish_type):
    gen, _ = _engine_capture({"temperature": 0.7}, finish_type)
    assert asyncio.run(gen(prompt="p", images=[]))["finish_reason"] == finish_type


# --- hf backend: the signal is derived, not read ---------------------------

_EOS_ID = 7
_PROMPT_LEN = 5
_MAX_NEW = 4


class _Inputs(dict):
    """Stand-in for a ``BatchFeature`` — dict-splattable, ``.to()`` is a no-op."""

    def to(self, _device):
        return self


def _hf_generate(last_token_id: int):
    """Build the hf generate fn over a stubbed model that ends on *last_token_id*."""
    import torch

    out_ids = torch.zeros((1, _PROMPT_LEN + _MAX_NEW), dtype=torch.long)
    out_ids[0, -1] = last_token_id

    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = out_ids

    processor = MagicMock()
    processor.return_value = _Inputs(input_ids=torch.zeros((1, _PROMPT_LEN), dtype=torch.long))
    processor.batch_decode.return_value = ["hi"]
    processor.tokenizer.eos_token_id = _EOS_ID

    with patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=model):
        return make_hf_generate_fn(
            "stub-model", processor, {"temperature": 0.0, "max_new_tokens": _MAX_NEW})


def test_hf_budget_exhausted_is_length():
    gen = _hf_generate(last_token_id=_EOS_ID + 1)
    assert asyncio.run(gen(prompt="p", images=[]))["finish_reason"] == "length"


def test_hf_eos_is_stop():
    gen = _hf_generate(last_token_id=_EOS_ID)
    assert asyncio.run(gen(prompt="p", images=[]))["finish_reason"] == "stop"
