"""Shared helpers for the split CUAWorld test family.

Run: uv run pytest tests/gym/envs/lite/cuaworld/test_cuaworld_*.py -q
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.envs.lite.cuaworld.src.adapter import (
    _write_bytes_before_deadline,
    run_cuaworld_verify,
)
from lite.gym.errors import CuaWorldVerifierError

_GOOD = {
    "success": {"spec": {"program": "verifier.py::verify"}},
    "description": "do the thing",
    "init": {"max_steps": 5},
}


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, format="PNG")
    return stream.getvalue()


def _jpeg_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, format="JPEG")
    return stream.getvalue()


def _partial_request_worker(
    _verifier_path,
    _verifier_target,
    _traj,
    _task_info,
    _workdir,
    requests,
    _responses,
    _results,
):
    os.write(requests.fileno(), struct.pack("!i", 1024) + b"x")
    time.sleep(5)


def _slow_read_regular_text(path):
    from lite.gym.envs.lite.cuaworld.src.adapter import _read_optional_regular_text

    time.sleep(5)
    return _read_optional_regular_text(path)


def _slow_validated_image_mime(image_bytes):
    from lite.gym.envs.lite.cuaworld.src.vlm import validated_image_mime

    time.sleep(5)
    return validated_image_mime(image_bytes)


def _slow_write_bytes(*args, **kwargs):

    time.sleep(1)
    return _write_bytes_before_deadline(*args, **kwargs)


def _delayed_marker(path, delay):
    time.sleep(delay)
    path.write_text("late")


def _stamp_materials(cache, env):
    from lite.gym.envs.lite.cuaworld.src import image_spec

    (cache / ".materials_revision").write_text(
        software._materials_identity(env) + "\n"
    )
    (cache / ".materials_digest").write_text(
        image_spec.cache_materials_digest(cache) + "\n"
    )
    (cache / ".complete").touch()


def _materials(root, sw, env, *, tasks, registered):
    """Write a fake ``.cache/<sw>/<env>`` materials tree under ``root``."""
    cache = root / ".cache" / sw / env
    (cache / "scripts").mkdir(parents=True, exist_ok=True)
    for tid, spec in tasks.items():
        d = cache / "tasks" / tid
        d.mkdir(parents=True)
        (d / "task.json").write_text(
            spec if isinstance(spec, str) else json.dumps(spec))
        (d / "verifier.py").write_text(
            "def verify(traj, env_info, task_info):\n    return {'score': 100.0}\n")
    (cache / "registered.json").write_text(json.dumps(registered))
    (cache / "env.json").write_text(json.dumps({
        "id": f"{env}@official",
        "resources": {"mem_gb": 4},
    }))
    _stamp_materials(cache, env)




class _FakeInterface:
    def __init__(self, *, export_rc=0):
        self.export_rc = export_rc

    async def write_bytes(self, _path, _data):
        return None

    async def run_command(self, command, timeout=None):
        rc = self.export_rc if "cuaworld_export.sh" in command else 0
        return SimpleNamespace(
            returncode=rc,
            stdout="captured",
            stderr="export failed" if rc else "",
        )

    async def read_bytes(self, _path):
        return b"payload"

    async def screenshot(self):
        return _png_bytes()


class _RecordingInterface(_FakeInterface):
    def __init__(self, *, export_rc=0):
        super().__init__(export_rc=export_rc)
        self.writes = {}
        self.commands = []

    async def write_bytes(self, path, data):
        self.writes[path] = data

    async def run_command(self, command, timeout=None):
        self.commands.append(command)
        return await super().run_command(command, timeout=timeout)


async def _verify_error(*args, **kwargs) -> CuaWorldVerifierError:
    """Await ``run_cuaworld_verify`` expecting it to REFUSE to score, and return why.

    Infrastructure failures — the process would not start, it timed out, it raised,
    it returned something that is not a score — now raise ``CuaWorldVerifierError``
    instead of returning ``(0.0, {"error": ...})``. They have to: a rollout cannot
    tell that tuple apart from an agent that genuinely failed, so a broken verifier
    used to become a well-formed, wrong training sample. A real 0 still comes back as
    a value — that is ``{"passed": False}`` with a finite score, and it goes through
    ``run_cuaworld_verify`` normally.
    """
    with pytest.raises(CuaWorldVerifierError) as excinfo:
        await run_cuaworld_verify(*args, **kwargs)
    return excinfo.value


def _verifier_task(root, score_source: str, *, export=False):
    task = root / "tasks" / root.name
    task.mkdir(parents=True)
    (task / "verifier.py").write_text(
        "def verify(traj, env_info, task_info):\n"
        f"{textwrap.indent(score_source, '    ')}\n"
    )
    if export:
        (task / "export_result.sh").write_text("exit 17\n")
    return task




def _cuaworld_root() -> Path:
    return Path(__file__).resolve().parents[4] / "lite/gym/envs/lite/cuaworld"


def _materials_root() -> Path:
    return _cuaworld_root() / ".cache"


def _task_dir_shell_scripts() -> list[Path]:
    """Every `.sh` under the on-disk cuaworld task dirs (the 3216-task basis)."""
    return sorted(_materials_root().glob("*/*/tasks/*/**/*.sh"))


def _require_full_pinned_materials() -> None:
    libraries = sorted(_materials_root().glob("*/*/scripts/task_utils.sh"))
    scripts = _task_dir_shell_scripts()
    if len(libraries) < 35 or len(scripts) < 3000:
        pytest.skip("cuaworld pinned materials not fully fetched on this host")




def _pinned_library(software: str) -> Path:
    library = next(
        _materials_root().glob(f"{software}/*/scripts/task_utils.sh"), None
    )
    if library is None:
        pytest.skip("cuaworld materials not fetched")
    return library


def _fake_env_tree(root: Path, software: str, task: str) -> Path:
    """A `.cache/<software>/<env>/{scripts,tasks/<task>}` skeleton carrying the REAL
    pinned `task_utils.sh`, so `_hook_helpers` resolves its names the production way
    (two levels up from the task dir) instead of being handed a list by the test."""
    env = root / ".cache" / software / f"{software}_env"
    (env / "scripts").mkdir(parents=True)
    (env / "scripts" / "task_utils.sh").write_bytes(
        _pinned_library(software).read_bytes()
    )
    task_dir = env / "tasks" / task
    task_dir.mkdir(parents=True)
    return task_dir




def _registered_non_excluded(software: str, task: str) -> bool:
    root = _materials_root()
    excludes = json.loads(
        (_cuaworld_root() / "data/validation_excludes.json").read_text()
    )
    if (excludes.get(software) or {}).get(task):
        return False
    catalog = json.loads(
        next(root.glob(f"{software}/*/registered.json")).read_text()
    )
    return any(task in ids for ids in catalog.values() if isinstance(ids, list))




_VLM_SCHEMA_JSON = '{"currency_hidden": true, "percent_visible": true, "symbols_visible": false}'


def _stub_completion(monkeypatch, text):
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            completion=lambda **_kwargs: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )
        ),
    )




def _write_local_materials_tree(root: Path, env_text: str = "env") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "env.json").write_text(env_text)
    (root / "registered.json").write_text("{}")
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "setup.sh").write_text("echo setup\n")
    (root / "tasks").mkdir(exist_ok=True)
    (root / "tasks" / "verifier.py").write_text("print('host only')\n")
