"""Lock the lazy-import contract of ``lite.infer.serving``.

``lite.infer.serving`` wraps an sglang ``Engine`` / server but must NOT import
``sglang`` (nor ``torch``) at module load — the heavy backends are imported
lazily inside the wrapper functions so that merely *referencing* the module
(e.g. a gym-side or CPU-only consumer doing ``from lite.infer.serving import
make_server_generate_fn``) stays cheap and dependency-free. This locks that
contract; it passes today.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/infer/serving/test_sglang_boundary.py -p no:cacheprovider -q
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

from lite.infer.serving import make_server_generate_fn
from lite.utils.path import project_root

_TARGET_MODULE = "lite.infer.serving"

_REPO_ROOT = project_root()

# Import the target in a fresh interpreter, then assert the heavy backends are
# absent from sys.modules. exit 0 = clean; exit 1 = a backend leaked.
_SUBPROCESS_CODE = (
    "import importlib, sys\n"
    f"importlib.import_module({_TARGET_MODULE!r})\n"
    "bad = [m for m in ('sglang', 'torch') if m in sys.modules]\n"
    "print('LEAKED', bad)\n"
    "sys.exit(1 if bad else 0)\n"
)


def _import_in_fresh_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with CUA_LITE_ENV_SERVER_URL stripped."""
    env = dict(os.environ)
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )


def test_sglang_module_does_not_eager_import_backends() -> None:
    """Importing ``lite.infer.serving`` must not pull ``sglang`` or ``torch``."""
    result = _import_in_fresh_subprocess(_SUBPROCESS_CODE)
    assert result.returncode == 0, (
        f"{_TARGET_MODULE} eager-imported a heavy backend: {result.stdout.strip()}\n"
        f"stderr={result.stderr[-2000:]}"
    )


def test_sglang_server_url_is_a_base_url_for_generate_endpoint() -> None:
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            # meta_info.finish_reason is part of sglang's /generate body and the
            # generate fn hands it to the agent (see test_serving_sampling_seed).
            return {"text": "ok", "meta_info": {"finish_reason": {"type": "stop"}}}

    async def _post(url: str, json: dict) -> _Resp:  # noqa: A002 - httpx name
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    mock_client = MagicMock()
    mock_client.post = _post

    with patch("lite.infer.serving.httpx.AsyncClient", return_value=mock_client):
        gen = make_server_generate_fn("http://localhost:30184/", {"temperature": 0.0})

    asyncio.run(gen(prompt="hello", images=[]))

    assert captured["url"] == "http://localhost:30184/generate"
    assert captured["payload"] == {
        "text": "hello",
        "sampling_params": {"temperature": 0.0},
    }
