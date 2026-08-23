"""Removal-discipline gates kept durable as tests.

Two invariants over the WHOLE tree:
  1. Python: no inline ``["docker", "rm", ...]`` argv outside the one helper
     (`lite.gym.utils.backend.docker._rm_argv`) — every runtime rm path must route
     through `docker_rm_f` / `docker_rm_f_async` / `_rm_argv`, so the ``-v``
     (anon-volume clean) flag can never silently diverge again.
  2. Bash: every ``docker rm`` carries ``-v`` (``-fv`` / ``-v``), and the
     invalid ``docker rmi ... -v`` / ``docker image rm ... -v`` never appears
     (image-rm has no -v flag; it would break install/uninstall).

Run: uv run pytest tests/gym/utils/backend/test_rm_discipline.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # tests/gym/utils/backend/<f> -> repo root

# The one place the argv literal is allowed to exist.
_PY_ALLOWED = {
    "lite/gym/utils/backend/docker.py",             # _rm_argv itself
    # standalone build-time script (no lite import; follows the bash -v rule)
    "lite/gym/envs/waa/scripts/utils/prepare_snapshot.py",
}


def _git_files(pattern: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", pattern], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip() and (REPO / ln).is_file()]


def test_no_inline_python_docker_rm_argv():
    offenders = []
    for rel in _git_files("lite/**/*.py"):
        if rel in _PY_ALLOWED:
            continue
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r'"docker",\s*"rm"', text):
            offenders.append(rel)
    assert not offenders, (
        f"inline docker-rm argv outside the A0 helper: {offenders} — "
        "route through lite.gym.utils.backend.docker (docker_rm_f / docker_rm_f_async / _rm_argv)"
    )


def test_bash_docker_rm_always_carries_v():
    offenders = []
    for rel in _git_files("*.sh"):
        path = REPO / rel
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue    # prose; only executable lines are the leak surface
            for m in re.finditer(r"docker rm (-[a-z]+ )*", line):
                flags = "".join(re.findall(r"-([a-z]+) ", m.group(0)))
                if "v" not in flags:
                    offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, f"docker rm without -v in shell scripts: {offenders}"


def test_no_docker_image_rm_with_v_flag():
    offenders = []
    pat = re.compile(r"docker (rmi|image rm)\b[^|;&\n]*\s-(v|fv|vf)\b")
    for rel in _git_files("*.sh") + _git_files("lite/**/*.py"):
        path = REPO / rel
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines, 1):
            if pat.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, f"`docker rmi/image rm` with -v (invalid flag): {offenders}"
