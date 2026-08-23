"""The #40 self-containment invariant: importing any ``lite.gym`` module
must NOT drag in the heavy ML / sibling-tier stack.

A fresh env-server (and any direct-mode ``gym.make`` consumer) imports
``lite.gym`` and the per-env ``main.py`` modules. If importing one of those
transitively pulls ``torch`` / ``transformers`` / ``sglang`` or a sibling tier
(``lite.agents`` / ``lite.train`` / ``lite.data`` / ``lite.infer``), the
import-boundary that keeps the gym deployable on a CPU-only env-server is
broken. This test locks that boundary so the upcoming ``lite/`` refactor cannot
silently reintroduce a forbidden import.

Each target module is imported in a FRESH subprocess interpreter (a clean
``sys.modules``) so leaks attributed to one module can't be masked by a previous
one. The env discovery is dynamic (globs ``lite/gym/envs/*/main.py`` and the
nested ``lite/gym/envs/lite/*/main.py``) so a newly-added env auto-enrolls.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/gym/test_self_containment.py -p no:cacheprovider -q
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from lite.utils.path import project_root

# Tier / heavy-dep modules that a self-contained gym import must never load.
FORBIDDEN = (
    "lite.agents",
    "lite.train",
    "lite.data",
    "lite.infer",
    "transformers",
    "torch",
    "sglang",
)

# Repo root = three levels up from this file (tests/gym/<file> -> repo root).
_REPO_ROOT = project_root()

# Subprocess body: import the target module then report any forbidden leak.
#   exit 0 -> clean
#   exit 1 -> leaked (stdout: "LEAKED <sorted roots>")
#   exit 2 -> env deps missing (stdout: "SKIP_DEPS"); test should skip
#   exit 3 -> unexpected import error (stdout: "IMPORT_ERROR ...")
# NOTE: no str.format / f-string here — the body contains literal ``{}`` set
# comprehensions. FORBIDDEN is injected by prepending an assignment line.
_SUBPROCESS_CODE = "FORBIDDEN = " + repr(FORBIDDEN) + "\n" + r"""
import importlib
import sys

mod = sys.argv[1]
try:
    importlib.import_module(mod)
except Exception as exc:  # noqa: BLE001 - boundary: classify any failure
    if type(exc).__name__ == "EnvDepsMissingError":
        print("SKIP_DEPS")
        sys.exit(2)
    print("IMPORT_ERROR", type(exc).__name__, repr(str(exc))[:200])
    sys.exit(3)

leaked = sorted(
    m for m in sys.modules for f in FORBIDDEN if m == f or m.startswith(f + ".")
)
roots = sorted({f for m in leaked for f in FORBIDDEN if m == f or m.startswith(f + ".")})
print("LEAKED", roots)
sys.exit(1 if leaked else 0)
"""


def _import_in_fresh_subprocess(module: str) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a fresh interpreter with CUA_LITE_ENV_SERVER_URL stripped."""
    env = dict(os.environ)
    env.pop("CUA_LITE_ENV_SERVER_URL", None)
    return subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CODE, module],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )


def _discover_env_mains() -> list[str]:
    """Dotted names of every ``lite.gym.envs.<...>.main`` package module.

    Globs both the flat ``lite/gym/envs/*/main.py`` and the nested
    ``lite/gym/envs/lite/*/main.py`` (handles ``lite.demo`` / ``lite.osworld``).
    Only paths whose every directory up to ``lite`` is a real package
    (has ``__init__.py``) become importable dotted names — this drops the
    non-package ``lite/gym/envs/lite/osworld/docker/server/main.py``.
    """
    envs_dir = _REPO_ROOT / "lite" / "gym" / "envs"
    candidates = sorted(envs_dir.glob("*/main.py")) + sorted(
        envs_dir.glob("lite/*/main.py")
    )
    mains: list[str] = []
    for path in candidates:
        rel = path.relative_to(_REPO_ROOT)
        # Every package dir on the path (from the env dir down to the file's
        # parent) must have __init__.py for the dotted import to resolve.
        parents = list(path.parents)[: len(rel.parts) - 1]
        package_dirs = [p for p in parents if _REPO_ROOT in p.parents or p == _REPO_ROOT]
        is_package = all(
            (p / "__init__.py").exists()
            for p in package_dirs
            if str(p).startswith(str(_REPO_ROOT / "lite")) and p != _REPO_ROOT
        )
        if is_package:
            mains.append(".".join(rel.with_suffix("").parts))
    return mains


# Core gym modules an env-server boots through, plus every discovered env main.
_CORE_MODULES = [
    "lite.gym",
    "lite.gym.registry",
    "lite.gym.remote.server",
    "lite.gym.services",
]
_ENV_MAINS = _discover_env_mains()
# These two env mains each fork a fresh interpreter whose import is slow
# (~27s webgym, ~22s browsergym), dominating the default suite — gate them
# behind ``stress`` so the fast params still give coverage by default.
_SLOW_MAINS = {"lite.gym.envs.webgym.main", "lite.gym.envs.browsergym.main"}
_TARGETS = [
    pytest.param(m, marks=pytest.mark.stress) if m in _SLOW_MAINS else m
    for m in _CORE_MODULES + _ENV_MAINS
]


@pytest.mark.parametrize("module", _TARGETS)
def test_module_is_self_contained(module: str) -> None:
    """Importing ``module`` in a fresh interpreter must not leak FORBIDDEN deps."""
    result = _import_in_fresh_subprocess(module)
    if result.returncode == 2:
        pytest.skip(f"{module}: env deps missing (EnvDepsMissingError on import)")
    assert result.returncode != 3, (
        f"{module}: unexpected import error\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )
    assert result.returncode == 0, (
        f"{module} leaked forbidden modules: {result.stdout.strip()}\n"
        f"stderr={result.stderr[-2000:]}"
    )
