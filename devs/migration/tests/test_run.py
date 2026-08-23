"""Tests for migration run.py path gating and local artifact writing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_migration_module(filename: str):
    path = _PROJECT_ROOT / "devs" / "migration" / filename
    spec = importlib.util.spec_from_file_location(f"cua_lite_migration_{filename}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _old_jsonl_use_row() -> dict:
    return {
        "images": ["screen0.png", "screen1.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "response"],
        },
        "messages": [
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "click",
                            "arguments": {"coordinate": [10, 20]},
                        },
                    }
                ],
            },
            {"role": "user", "content": [{"type": "image", "index": 1}]},
        ],
    }


def _old_jsonl_key_batch_row() -> dict:
    row = _old_jsonl_use_row()
    row["metadata"]["valid_actions"] = ["key"]
    row["messages"][1]["tool_calls"] = [
        _old_call("computer", actions=[{"action": "key", "keys": ["ctrl", "plus"]}]),
    ]
    return row


def _old_grounding_key_row() -> dict:
    return {
        "images": ["screen0.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "grounding.action",
            "extra_tool_schemas": [],
            "valid_actions": None,
            "others": {},
        },
        "messages": [
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [_old_call("key", keys=["ctrl", "plus"])],
            },
        ],
    }


def _old_call(name: str, **arguments) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def _old_noop_mid_trajectory_row() -> dict:
    """4 turns, the MIDDLE one a noop-only ``screenshot``, 4 pictures.

    Migration drops that turn and the observation it answered, so ``screen1``
    ends up referenced by nothing while ``screen2`` / ``screen3`` are still
    referenced -- a mid-sequence orphan, the case where every later index shifts.
    """
    return {
        "images": ["screen0.png", "screen1.png", "screen2.png", "screen3.png"],
        "metadata": {
            "platform": "desktop",
            "task_type": "use",
            "valid_actions": ["click", "screenshot"],
            "others": {"task_id": "noop-mid"},
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "Open settings."},
                ],
            },
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    _old_call("click", coordinate=[10, 20]),
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 1},
                    {"type": "text", "text": "clicked"},
                ],
            },
            {"role": "assistant", "content": [], "tool_calls": [_old_call("screenshot")]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 2},
                    {"type": "text", "text": "same screen"},
                ],
            },
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    _old_call("click", coordinate=[30, 40]),
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 3},
                    {"type": "text", "text": "settings open"},
                ],
            },
        ],
    }


def _lite_input_path(tmp_path: Path, filename: str, dataset: str = "Lite.OSWorld") -> Path:
    path = tmp_path / dataset / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_run_migration_scope_is_exact_five_repo_whitelist():
    """Migration reaches only the five uploaded repos selected for rewrite."""
    run = _load_migration_module("run.py")
    expected = frozenset(
        {
            "Lite.OSWorld",
            "Lite.CUAGym",
            "Lite.CUAWorld",
            "Lite.ScaleCUA",
            "WebGym",
        }
    )

    assert run.ALLOWED_MIGRATION_DATASETS == expected
    assert frozenset(run._DEV_DATASET_COMPONENTS.values()) == expected


def test_run_partition_path_components_use_canonical_owners():
    """The path shape is migration-local, but its literals are not."""
    from lite.core.metadata import LiteCUAMetadata
    from lite.data.staging import CANONICAL_SPLITS

    run = _load_migration_module("run.py")

    assert run._CANONICAL_PLATFORMS == frozenset(
        str(platform) for platform in LiteCUAMetadata.Platform
    )
    assert run._CANONICAL_TASK_TYPES == frozenset(
        str(task_type) for task_type in LiteCUAMetadata.TaskType
    )
    assert run._CANONICAL_SPLITS == frozenset(CANONICAL_SPLITS)


def test_run_writes_local_jsonl_artifact_and_verifies(tmp_path):
    run = _load_migration_module("run.py")
    src = _lite_input_path(tmp_path, "old.jsonl")
    dst = tmp_path / "new.jsonl"
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    summary = run.migrate_path(src, dst, verify=True)

    assert summary.to_dict()["files"] == 1
    assert summary.to_dict()["rows"] == 1
    assert summary.to_dict()["verified"] == 1
    migrated = json.loads(dst.read_text(encoding="utf-8").strip())
    call = migrated["messages"][1]["tool_calls"][0]
    assert tool_call_name(call) == "computer"
    assert set(call) == {"id", "type", "function"}
    assert tool_call_arguments(call) == {"actions": [{"action": "click", "coordinate": [10, 20]}]}
    assert migrated["messages"][2]["role"] == "tool"


def test_run_dry_run_migrates_and_verifies_in_memory_without_writing(tmp_path):
    """``--dry-run`` is a full in-memory migration that writes NOTHING: rows are
    upgraded and verified, the counts are real, and no output file/directory is
    created even though an output path was supplied.
    """
    run = _load_migration_module("run.py")
    root = _lite_input_path(tmp_path, "old.jsonl").parent
    (root / "old.jsonl").write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")
    dst = tmp_path / "out"

    summary = run.migrate_path(root, dst, dry_run=True, verify=True)

    assert summary.to_dict()["files"] == 1
    assert summary.to_dict()["rows"] == 1
    assert summary.to_dict()["verified"] == 1
    assert summary.to_dict()["dry_run"] is True
    # The ignored output path is reported as null, and nothing landed on disk.
    assert summary.to_dict()["output_path"] is None
    assert not dst.exists()


def test_run_migration_parquet_output_uses_canonical_staging_schema(tmp_path):
    run = _load_migration_module("run.py")
    pd = pytest.importorskip("pandas")

    src = _lite_input_path(tmp_path, "old.parquet", dataset="Lite.ScaleCUA")
    dst = tmp_path / "new.parquet"
    pd.DataFrame([_old_jsonl_use_row()]).to_parquet(src, index=False)

    summary = run.migrate_path(src, dst, verify=True)

    assert summary.to_dict()["files"] == 1
    assert summary.to_dict()["rows"] == 1
    assert summary.to_dict()["verified"] == 1

    table = pq.read_table(dst)
    assert pa.types.is_string(table.schema.field("messages").type)
    assert pa.types.is_string(table.schema.field("metadata").type)
    row = table.to_pylist()[0]
    assert isinstance(row["messages"], str)
    assert isinstance(row["metadata"], str)
    messages = json.loads(row["messages"])
    metadata = json.loads(row["metadata"])
    assert tool_call_name(messages[1]["tool_calls"][0]) == "computer"
    assert [tool_schema_name(schema) for schema in metadata["extra_tool_schemas"]] == ["response"]
    assert all("function" in schema for schema in metadata["extra_tool_schemas"])


def test_run_jsonl_verify_smoke_normalizes_legacy_key_aliases(tmp_path):
    run = _load_migration_module("run.py")
    src = _lite_input_path(tmp_path, "old.jsonl")
    dst = tmp_path / "new.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(row) for row in [_old_jsonl_key_batch_row(), _old_grounding_key_row()]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run.migrate_path(src, dst, verify=True)

    assert summary.to_dict()["rows"] == 2
    assert summary.to_dict()["verified"] == 2
    migrated = [
        json.loads(line)
        for line in dst.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    use_call = migrated[0]["messages"][1]["tool_calls"][0]
    grounding_call = migrated[1]["messages"][1]["tool_calls"][0]
    assert tool_call_arguments(use_call)["actions"] == [
        {"action": "key", "keys": ["ctrl", "+"]}
    ]
    assert tool_call_name(grounding_call) == "key"
    assert tool_call_arguments(grounding_call) == {"keys": ["ctrl", "+"]}


def test_run_parquet_verify_smoke_normalizes_legacy_key_aliases(tmp_path):
    run = _load_migration_module("run.py")
    pd = pytest.importorskip("pandas")

    src = _lite_input_path(tmp_path, "old.parquet", dataset="Lite.ScaleCUA")
    dst = tmp_path / "new.parquet"
    rows = [_old_jsonl_key_batch_row(), _old_grounding_key_row()]
    parquet_rows = []
    for row in rows:
        parquet_rows.append(
            {
                **row,
                "messages": json.dumps(row["messages"]),
                "metadata": json.dumps(row["metadata"]),
            }
        )
    pd.DataFrame(parquet_rows).to_parquet(src, index=False)

    summary = run.migrate_path(src, dst, verify=True)

    assert summary.to_dict()["rows"] == 2
    assert summary.to_dict()["verified"] == 2
    migrated = pq.read_table(dst).to_pylist()
    use_messages = json.loads(migrated[0]["messages"])
    grounding_messages = json.loads(migrated[1]["messages"])
    use_call = use_messages[1]["tool_calls"][0]
    grounding_call = grounding_messages[1]["tool_calls"][0]
    assert tool_call_arguments(use_call)["actions"] == [
        {"action": "key", "keys": ["ctrl", "+"]}
    ]
    assert tool_call_name(grounding_call) == "key"
    assert tool_call_arguments(grounding_call) == {"keys": ["ctrl", "+"]}


def test_run_compacts_orphan_images_left_by_a_dropped_noop_turn(tmp_path):
    """End-to-end through the real entry point: no orphan survives migration.

    ``run.migrate_path`` is what the CLI calls, and the parquet path is the one
    published rows take (pandas hands ``images`` back as a numpy array). The
    dropped middle turn orphans ``screen1``; the written row must not publish it,
    must number the survivors ``0..N-1``, and every reference must still resolve
    to the same picture.
    """
    run = _load_migration_module("run.py")
    pd = pytest.importorskip("pandas")

    src = _lite_input_path(tmp_path, "old.parquet")
    dst = tmp_path / "new.parquet"
    row = _old_noop_mid_trajectory_row()
    pd.DataFrame([row]).to_parquet(src, index=False)

    summary = run.migrate_path(src, dst, verify=True)
    assert summary.to_dict()["verified"] == 1

    out = pq.read_table(dst).to_pylist()[0]
    messages = json.loads(out["messages"])
    images = list(out["images"])
    indices = [
        part["index"]
        for message in messages
        for part in message.get("content") or []
        if part.get("type") == "image"
    ]

    assert images == ["screen0.png", "screen2.png", "screen3.png"]
    assert sorted(set(indices)) == list(range(len(images)))
    assert "screen1.png" not in images
    # Same pictures as before, by content: the goal screenshot, the screen the
    # noop turn observed, and the final one.
    assert [images[i] for i in indices] == ["screen0.png", "screen2.png", "screen3.png"]


def test_run_allows_nested_lite_dataset_partition_path(tmp_path):
    run = _load_migration_module("run.py")
    pd = pytest.importorskip("pandas")

    src = tmp_path / "Lite.ScaleCUA" / "desktop" / "use" / "train.parquet"
    dst = tmp_path / "new.parquet"
    src.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_old_jsonl_use_row()]).to_parquet(src, index=False)

    summary = run.migrate_path(src, dst, dry_run=True, verify=True)

    assert summary.to_dict()["files"] == 1
    assert summary.to_dict()["rows"] == 1
    assert summary.to_dict()["verified"] == 1


def test_run_allows_nested_lite_dataset_jsonl_path(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "Lite.OSWorld" / "desktop" / "use" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    summary = run.migrate_path(src, dst, dry_run=True, verify=True)

    assert summary.to_dict()["files"] == 1
    assert summary.to_dict()["rows"] == 1
    assert summary.to_dict()["verified"] == 1


def test_run_allows_devs_data_lite_dataset_component(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "devs" / "data" / "lite.cuagym" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    summary = run.migrate_path(src, dst, dry_run=True, verify=True)

    assert summary.to_dict()["rows"] == 1


def test_run_rejects_non_lite_dataset_path_by_default(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "WebGymRT" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


@pytest.mark.parametrize("component", ["OtherDataset", "random"])
def test_run_rejects_unknown_component_below_lite_dataset_path(tmp_path, component: str):
    run = _load_migration_module("run.py")
    src = tmp_path / "Lite.OSWorld" / component / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


def test_run_allows_devs_data_webgym_path(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "devs" / "data" / "webgym" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    summary = run.migrate_path(src, dst, dry_run=True, verify=True)

    assert summary.to_dict()["rows"] == 1


def test_run_allows_webgym_dataset_path(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "WebGym" / "web" / "use" / "train.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    summary = run.migrate_path(src, dst, dry_run=True, verify=True)

    assert summary.to_dict()["rows"] == 1


def test_run_rewrites_historical_web_partition_to_browser_output(tmp_path):
    run = _load_migration_module("run.py")
    root = tmp_path / "WebGym"
    src = root / "web" / "use" / "train" / "old.jsonl"
    dst = tmp_path / "out"
    src.parent.mkdir(parents=True, exist_ok=True)
    row = _old_jsonl_use_row()
    row["metadata"]["platform"] = "web"
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = run.migrate_path(root, dst, verify=True)

    migrated_path = dst / "browser" / "use" / "train" / "old.jsonl"
    assert summary.to_dict()["rows"] == 1
    assert migrated_path.exists()
    assert not (dst / "web").exists()
    migrated = json.loads(migrated_path.read_text(encoding="utf-8").strip())
    assert migrated["metadata"]["dims"] == ["browser", "use"]


def test_run_rejects_lite_directory_with_webgymrt_child_file(tmp_path):
    run = _load_migration_module("run.py")
    root = tmp_path / "Lite.OSWorld"
    src = root / "WebGymRT" / "old.jsonl"
    dst = tmp_path / "out"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(root, dst, dry_run=True, verify=True)


def test_run_allows_devs_data_webgym_with_lite_descendant(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "devs" / "data" / "webgym" / "Lite.OSWorld" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    summary = run.migrate_path(src, dst, dry_run=True, verify=True)

    assert summary.to_dict()["rows"] == 1


def test_run_rejects_lite_dataset_below_webgymrt_scratch_root(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "WebGymRT" / "devs" / "data" / "lite.cuagym" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


def test_run_rejects_lite_dataset_below_webgymtest_scratch_root(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "WebGymTest" / "Lite.OSWorld" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


def test_run_rejects_lite_dataset_below_webgymrt_scratch_root_by_name(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "WebGymRT" / "Lite.OSWorld" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


def test_run_rejects_nested_lite_dataset_below_webgymrt_scratch_root(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "WebGymRT" / "Lite.OSWorld" / "desktop" / "use" / "old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


def test_run_rejects_webgymrt_with_lite_dataset_filename(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "WebGymRT" / "Lite.OSWorld.old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


def test_run_rejects_lite_dataset_filename_without_lite_directory(tmp_path):
    run = _load_migration_module("run.py")
    src = tmp_path / "Lite.OSWorld.old.jsonl"
    dst = tmp_path / "new.jsonl"
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out-of-scope input path"):
        run.migrate_path(src, dst, dry_run=True, verify=True)


def test_run_refuses_implicit_in_place_migration(tmp_path):
    run = _load_migration_module("run.py")
    src = _lite_input_path(tmp_path, "old.jsonl")
    src.write_text(json.dumps(_old_jsonl_use_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to migrate in place"):
        run.migrate_path(src, dry_run=False)
