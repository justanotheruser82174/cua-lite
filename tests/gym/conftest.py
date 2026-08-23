"""Gym-test leak safety net.

WHY: direct-mode env tests do ``env = gym.make(...)`` which boots a docker
container named ``<service>-d<pid>`` (pid = this process). ``env.close()`` is an
atomic ``docker rm -f``, BUT a test leaks the container whenever it either
(a) never calls close, or (b) calls close AFTER an assert that fails (no
try/finally). Across a run that orphans many containers / ports / threads.

This conftest adds:
  * ``import_env_main_or_skip`` / ``importable_env_ids`` — per-env tolerant
    import helpers for suites that iterate all envs (named skip per env /
    FAIL-on-mass-skip floors); imported directly as plain functions.
  * ``_reap_leaked_direct_containers`` — an autouse, session-scoped fixture that
    runs PER pytest process (incl. each xdist worker) and, at teardown,
    ``docker rm -f`` any ``*-d<os.getpid()>`` container this process left behind,
    emitting a warning per leak (attribution). It only ever touches THIS
    process's own ``d<pid>`` scope — never a co-tenant container (those are
    ``<service>-<port>`` named) and never another worker's live containers.

Run: env -u CUA_LITE_ENV_SERVER_URL -u CUA_LITE_ENV_SERVER_TOKEN \
     uv run pytest tests/gym -p no:cacheprovider -q
"""

from __future__ import annotations

import os
import subprocess
import warnings

import pytest


def _direct_scope_suffix() -> str:
    """The container-name suffix this process's direct-mode envs use: ``-d<pid>``."""
    return f"-d{os.getpid()}"


def _list_own_direct_containers() -> list[str]:
    """Names of running containers in THIS process's ``-d<pid>`` direct scope.

    Returns [] if docker is unavailable (env tests that need docker are skipped
    on such hosts, so there is nothing to reap)."""
    suffix = _direct_scope_suffix()
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if out.returncode != 0:
        return []
    return [n for n in out.stdout.split() if n.endswith(suffix)]


@pytest.fixture(scope="session", autouse=True)
def _reap_leaked_direct_containers():
    """Session-teardown sweep of this process's leaked direct-mode containers."""
    yield
    leaked = _list_own_direct_containers()
    for name in leaked:
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                           text=True, timeout=30)
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass
    if leaked:
        warnings.warn(
            "gym-test leak: direct-mode container(s) not closed by their test "
            f"(reaped at session end): {leaked}. Wrap `gym.make` in try/finally "
            "so `env.close()` runs even on failure.",
            stacklevel=2,
        )


def _production_cuaworld_leaves() -> list[str]:
    """The 40 production lite.cuaworld.* env ids, read from main.py's AST
    (register_cuaworld_software calls) — immune to runtime class pollution
    (probe classes created by other tests via ``_make_env_class``). Shared by
    the family suites."""
    import ast
    from pathlib import Path

    main_path = (
        Path(__file__).resolve().parents[2]
        / "lite/gym/envs/lite/cuaworld/main.py"
    )
    tree = ast.parse(main_path.read_text())
    software = [
        node.value.args[0].value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "register_cuaworld_software"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and isinstance(node.value.args[0].value, str)
    ]
    return sorted(f"lite.cuaworld.{name}" for name in software)


def _env_main_module(env_id: str) -> str:
    """Module path of an env's main. ``lite.cuaworld.<software>`` leaves all
    share one module (lite/gym/envs/lite/cuaworld/main.py) — the only
    id→module exception (PR #26's software-catalog envs)."""
    if env_id.startswith("lite.cuaworld."):
        return "lite.gym.envs.lite.cuaworld.main"
    return "lite.gym.envs." + env_id + ".main"


def import_env_main_or_skip(env_id: str) -> None:
    """Import one env's main module (its module-bottom register_family()/
    register() calls run at import) or SKIP the calling test item.

    Per-env isolation for suites that iterate many envs: one env's missing
    deps must never error the other envs' items (the pre-R5 module-scoped
    import loop turned one missing package into 118 setup errors). Import
    directly (plain function, not a fixture): ``from gym.conftest import
    import_env_main_or_skip`` — same idiom as gym.remote.conftest.
    """
    import importlib

    import pytest

    from lite.gym.errors import EnvDepsMissingError

    try:
        importlib.import_module(_env_main_module(env_id))
    except (EnvDepsMissingError, ModuleNotFoundError) as e:
        # ModuleNotFoundError breadth is REQUIRED (several env mains do
        # unguarded top-level third-party imports: numpy/requests/httpx) and
        # accepted at a cost: a broken FIRST-PARTY import in an env main also
        # skips instead of failing loud; the per-file coverage floors only
        # catch wholesale breakage.
        pytest.skip(f"{env_id}: env deps missing ({type(e).__name__}: {e})")


def importable_env_ids(env_ids) -> list[str]:
    """The subset of ``env_ids`` whose main imports on this host (tolerant
    loop, order-independent) — for coverage floors that FAIL on mass-skip."""
    import importlib

    from lite.gym.errors import EnvDepsMissingError

    out = []
    for env_id in env_ids:
        try:
            importlib.import_module(_env_main_module(env_id))
        except (EnvDepsMissingError, ModuleNotFoundError):
            continue
        out.append(env_id)
    return out
