"""Static AST tier-DAG check: ``lite/`` tiers may only import *downward*.

The #40 safety net, generalized from "gym doesn't import torch" to "the tier
import graph stays acyclic and one-directional", enforced WITHOUT importing any
module (pure ``ast`` parse). Static parsing is the whole point: it catches
function-local / lazy ``import lite.foo`` that a runtime import test would miss,
so the upcoming ``lite/`` refactor can't bury a forbidden edge inside a deferred
import.

Tier order (high tier may import low tier, never the reverse). ``lite.core`` is
the shared contract tier: implementation layers may import it, but it must not
import implementation layers.

    train / infer  >  agents  >  gym  >  utils  >  core

The forbidden edges are encoded in ``FORBIDDEN_EDGES`` below — a one-line-per-
rule table so the rename refactor edits data, not logic.

Current-state notes (verified at authoring time, 2026-06):
  * Rule (a) gym -/-> {agents,train,data,infer}: the keystone — PASSES.
  * Rule (b) agents -/-> {train,infer,data}: PASSES. The agents->data edge
    (9 edges via peel_system_message / write_records_to_parquet) was severed by
    the C2 refactor (-> lite.core / lite.utils.parquet); it is now a hard
    rule in ``FORBIDDEN_EDGES``, not a known-broken xfail.
  * Rule (c) utils -/-> any lite.* at all. The former exception for
    same-phase compatibility shims re-exporting ``lite.core`` is gone with
    its last member (``lite/types.py``); utils is now a strict leaf.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
      tests/static/test_tier_dag.py -p no:cacheprovider -q
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from lite.utils.path import project_root

_REPO_ROOT = project_root()

# --- DAG rule table -------------------------------------------------------
# (source-tier path prefix, tuple of forbidden imported-module prefixes).
# Edit THIS when the refactor renames tiers; the logic below is generic.
FORBIDDEN_EDGES: list[tuple[str, tuple[str, ...]]] = [
    # (0) core is the provider-free shared contract tier. It may depend on
    #     stdlib/third-party code and sibling ``lite.core`` modules, never on
    #     implementation layers.
    (
        "lite/core",
        ("lite.agents", "lite.gym", "lite.data", "lite.train", "lite.infer"),
    ),
    # (a) keystone: the deployable gym must not reach into sibling/upper tiers.
    ("lite/gym", ("lite.agents", "lite.train", "lite.data", "lite.infer")),
    # (b) agents must not import upward (train/infer/data). The agents->data edge
    #     was severed by the C2 refactor (peel_system_message -> lite.core,
    #     write_records_to_parquet -> lite.utils.parquet), so it is now a hard rule.
    ("lite/agents", ("lite.train", "lite.infer", "lite.data")),
]

# (c) utils is the leaf tier: it must import NO ``lite.*`` at all beyond its own
# ``lite.utils`` submodules. Enforced by ``test_utils_is_leaf_tier`` below, with
# no shim exemption list — the last exempted module (``lite/types.py``) is gone.

# Paths excluded from the whole walk. ``.venv`` holds installed third-party
# site-packages (not our source tier) — and some (e.g. a joblib test fixture)
# aren't even UTF-8 — so they must not be walked.
_EXCLUDED_DIR_PARTS = {".cache", "__pycache__", ".venv"}
_VENDORED_PREFIX = "lite/gym/envs/lite/osworld/src"


def _iter_py_files(prefix: str) -> list[Path]:
    """Every ``*.py`` under ``prefix`` minus cache / pycache / vendored gen."""
    files: list[Path] = []
    for path in sorted((_REPO_ROOT / prefix).glob("**/*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if any(part in _EXCLUDED_DIR_PARTS for part in path.parts):
            continue
        if rel.startswith(_VENDORED_PREFIX):
            continue
        files.append(path)
    return files


def _lite_imports(path: Path) -> set[str]:
    """All ``lite.*`` module targets imported by ``path`` (incl. lazy/local).

    Uses ``ast`` only — no execution — so function-local imports count.
    Relative (``from . import x``) imports are skipped: ``ast`` can't resolve
    them to a dotted ``lite.*`` target without package context, and tier edges
    are expressed as absolute imports in this codebase.
    """
    tree = ast.parse(path.read_text(), str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("lite."):
                    targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("lite."):
                targets.add(node.module)
    return targets


def _matches(imported: str, forbidden_prefix: str) -> bool:
    return imported == forbidden_prefix or imported.startswith(forbidden_prefix + ".")


def _violations(prefix: str, forbidden: tuple[str, ...]) -> list[tuple[str, str]]:
    """``(relpath, imported)`` pairs where a file under ``prefix`` imports a
    forbidden target."""
    out: list[tuple[str, str]] = []
    for path in _iter_py_files(prefix):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for imported in sorted(_lite_imports(path)):
            if any(_matches(imported, fb) for fb in forbidden):
                out.append((rel, imported))
    return out


@pytest.mark.parametrize("prefix,forbidden", FORBIDDEN_EDGES)
def test_tier_dag_edge(prefix: str, forbidden: tuple[str, ...]) -> None:
    """No file under ``prefix`` may import any of ``forbidden`` (one rule/assert)."""
    violations = _violations(prefix, forbidden)
    assert not violations, (
        f"forbidden tier edge(s) from {prefix}/** -> {forbidden}:\n"
        + "\n".join(f"  {f} imports {imp}" for f, imp in violations)
    )


def test_utils_is_leaf_tier() -> None:
    """``lite/utils/**`` imports NO ``lite.*`` beyond its own ``lite.utils``
    submodules.

    Post Stage U the non-leaf modules (sample/rollout -> infer, agents ->
    agents/factory) are gone, and the last compatibility shim exemption
    (``lite/types.py``) is deleted, so any surviving ``lite/utils ->
    lite.<other>`` edge is a regression.
    """
    violators: dict[str, list[str]] = {}
    for path in _iter_py_files("lite/utils"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        bad = sorted(imp for imp in _lite_imports(path) if not imp.startswith("lite.utils"))
        if bad:
            violators[rel] = bad

    assert not violators, (
        "lite/utils -> lite.* edge(s):\n"
        + "\n".join(f"  {f} imports {violators[f]}" for f in sorted(violators))
    )


def test_gym_imports_no_heavy_deps() -> None:
    """#40 deploy intent: importing gym and the shared core action-space contract
    must NOT pull torch / transformers / sglang. gym stays deployable from the
    light contract alone, without the model stack.

    Run in a subprocess so a heavy dep loaded by another test can't mask a real
    regression.
    """
    import subprocess
    import sys
    code = (
        "import sys, lite.gym\n"
        "import lite.core.tools.action_space\n"
        "heavy = [m for m in ('torch', 'transformers', 'sglang') if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"importing gym + core action_space failed:\n{out.stderr}"
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("HEAVY:")), None)
    assert line == "HEAVY:", (
        "importing gym pulled heavy model deps (breaks #40 import-DAG invariant): "
        f"{line}\nstdout:\n{out.stdout}"
    )
