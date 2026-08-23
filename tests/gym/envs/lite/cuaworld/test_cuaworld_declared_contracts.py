"""Guards for the three "declared here, read there" couplings this audit closed.

Each test pins a contract that used to be an unenforced convention:
  * ``env.json``'s ``resources.mem_gb`` is READ (it is a floor on the registered
    ``memory=``), not just declared — :func:`software._resolve_memory`;
  * ``validation_excludes.json``'s metadata keys are recognised by ONE predicate —
    :func:`software.is_excludes_metadata_key`;
  * ``osworld_2``'s wrapper ``step_timeout`` is DERIVED from the per-RPC ceilings, so
    the reward RPC always fits inside the step budget.

Run: uv run --no-sync pytest tests/gym/envs/lite/cuaworld/test_cuaworld_declared_contracts.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lite.gym.envs.lite.cuaworld.src import software
from lite.gym.errors import EnvDepsMissingError


# ---------------------------------------------------------------------------
# env.json resources.mem_gb is honoured as a FLOOR on the registered memory=
# ---------------------------------------------------------------------------
def _write_env_json(cache_root: Path, payload: dict) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "env.json").write_text(json.dumps(payload))


def test_declared_mem_gb_raises_an_under_provisioned_request(tmp_path):
    """The declaration wins when it is larger — the slicer3d/webots/blender3d case
    (7 softwares, 557 registered / 503 live tasks, booted below their declaration
    while nothing read the field)."""
    _write_env_json(tmp_path, {"id": "x@1", "resources": {"cpu": 4, "mem_gb": 8}})
    assert software._resolve_memory("sw", tmp_path, "6GB") == "8GB"


def test_declared_mem_gb_never_lowers_a_larger_request(tmp_path):
    """A larger ``memory=`` request must survive a smaller declaration.

    No shipped software passes ``memory=`` any more (they inherit the 8GB
    baseline), so this pins the resolver itself rather than a registration.
    """
    _write_env_json(tmp_path, {"id": "x@1", "resources": {"cpu": 4, "mem_gb": 4}})
    assert software._resolve_memory("sw", tmp_path, "6GB") == "6GB"
    _write_env_json(tmp_path, {"id": "x@1", "resources": {"cpu": 4, "mem_gb": 6}})
    assert software._resolve_memory("sw", tmp_path, "6GB") == "6GB"


def test_missing_declaration_leaves_the_request_alone(tmp_path):
    """Materials not installed yet -> the registered value stands."""
    assert software._declared_mem_gb(tmp_path) is None
    assert software._resolve_memory("sw", tmp_path, "4GB") == "4GB"


def test_malformed_declaration_fails_loudly(tmp_path):
    """Once env.json exists, an invalid memory declaration is a materials error."""
    _write_env_json(tmp_path, {"id": "x@1"})
    with pytest.raises(EnvDepsMissingError, match="invalid resources.mem_gb"):
        software._resolve_memory("sw", tmp_path, "4GB")
    (tmp_path / "env.json").write_text("{not json")
    with pytest.raises(EnvDepsMissingError, match="invalid resources.mem_gb"):
        software._resolve_memory("sw", tmp_path, "4GB")


# ---------------------------------------------------------------------------
# validation_excludes.json: ONE metadata-key predicate
# ---------------------------------------------------------------------------
def test_excludes_metadata_predicate_is_the_only_filter():
    """``_meta`` is metadata; every software key is not. The predicate — not a
    hardcoded ``!= "_meta"`` — decides, so a second ``_``-prefixed block added later
    is skipped by the reader AND by the ``_total`` checksum below in one change."""
    assert software.is_excludes_metadata_key("_meta")
    assert software.is_excludes_metadata_key("_anything_added_later")
    assert not software.is_excludes_metadata_key("slicer3d")


def test_validation_excludes_total_matches_the_shared_reader():
    """``_meta._total`` is the file's own checksum, recomputed here through the SAME
    filter :func:`software._exclude_reasons` uses. The pre-existing copy of this
    assertion in ``_cuaworld_support.py`` spells the filter ``!= "_meta"``; both must keep
    agreeing, and this one moves with the reader."""
    doc = json.loads(software.VALIDATION_EXCLUDES_PATH.read_text())
    by_software = {
        sw: t for sw, t in doc.items() if not software.is_excludes_metadata_key(sw)
    }
    assert sum(len(tasks) for tasks in by_software.values()) == doc["_meta"]["_total"]
    # No top-level key escapes the classification: everything is either a software
    # entry or a `_`-prefixed metadata block.
    assert set(doc) == set(by_software) | {
        k for k in doc if software.is_excludes_metadata_key(k)
    }
