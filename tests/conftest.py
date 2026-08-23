"""Pytest process setup shared by the default test suite."""

from __future__ import annotations

import os

_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_ROUTING_VARS = ("CUA_LITE_ENV_SERVER_URL", "CUA_LITE_ENV_SERVER_TOKEN")


for _var in _THREAD_CAP_VARS:
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_configure() -> None:
    """Keep default collection in direct mode even on hosts with env-server vars."""

    for var in _ROUTING_VARS:
        os.environ.pop(var, None)
