"""
Tests for data path patterns used in SFT configs.

Uses a temporary directory to mimic ScaleCUA/AgentNet layout and verifies
expand_path_pattern and discover_files_under_paths with absolute paths.

Run: uv run pytest tests/data/load/test_path_patterns.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lite_samples import (
    sample_grounding_action_minimal,
    sample_grounding_bbox,
    sample_grounding_point,
    sample_trajectory_one_turn,
    sample_understanding,
)

from lite.data.load import (
    discover_files_under_paths,
    expand_path_pattern,
)
from lite.utils.parquet import write_records_to_parquet


@pytest.fixture
def data_root(tmp_path):
    """Create temp dir with ScaleCUA + AgentNet layout and parquet files."""
    root = tmp_path

    scale_cua = root / "OpenGVLab/ScaleCUA-Data"
    for platform in ("desktop", "browser", "mobile"):
        (scale_cua / platform / "understanding").mkdir(parents=True)
        (scale_cua / platform / "grounding/action").mkdir(parents=True)
        (scale_cua / platform / "grounding/point").mkdir(parents=True)
        (scale_cua / platform / "grounding/bbox").mkdir(parents=True)
        (scale_cua / platform / "use").mkdir(parents=True)

    agent_net = root / "xlangai/AgentNet"
    (agent_net / "desktop/use").mkdir(parents=True)

    for platform in ("desktop", "browser", "mobile"):
        write_records_to_parquet(
            [sample_understanding()],
            scale_cua / platform / "understanding/understanding.parquet",
        )
        write_records_to_parquet(
            [sample_grounding_action_minimal()],
            scale_cua / platform / "grounding/action/action.parquet",
        )
        write_records_to_parquet(
            [sample_grounding_point()],
            scale_cua / platform / "grounding/point/point.parquet",
        )
        write_records_to_parquet(
            [sample_grounding_bbox()],
            scale_cua / platform / "grounding/bbox/bbox.parquet",
        )
        write_records_to_parquet(
            [sample_trajectory_one_turn()],
            scale_cua / platform / "use" / f"{platform}.parquet",
        )
    write_records_to_parquet(
        [sample_trajectory_one_turn()],
        agent_net / "desktop/use/agent_net.parquet",
    )

    return root


# --- expand_path_pattern (absolute paths) ---

def test_expand_base_only(data_root):
    got = expand_path_pattern(str(data_root / "OpenGVLab/ScaleCUA-Data"))
    assert len(got) == 1
    assert got[0].name == "ScaleCUA-Data"

def test_expand_trajectory_desktop_or_browser(data_root):
    got = expand_path_pattern(str(data_root / "OpenGVLab/ScaleCUA-Data") + "/(desktop|browser)/use")
    got_str = sorted(str(p) for p in got)
    assert len(got_str) == 2
    assert any("desktop" in s and "use" in s for s in got_str)
    assert any("browser" in s and "use" in s for s in got_str)

def test_expand_trajectory_any_platform(data_root):
    got = expand_path_pattern(str(data_root / "OpenGVLab/ScaleCUA-Data") + "/[^/]+/use")
    assert len(got) == 3

def test_expand_empty_base_raises():
    with pytest.raises(ValueError, match="literal base prefix"):
        expand_path_pattern("[^/]+/use")


# --- discover_files_under_paths (absolute paths) ---

def test_discover_single_path_base(data_root):
    files = discover_files_under_paths([str(data_root / "OpenGVLab/ScaleCUA-Data")])
    assert len(files) >= 7

def test_discover_trajectory_only(data_root):
    files = discover_files_under_paths([str(data_root / "OpenGVLab/ScaleCUA-Data") + "/[^/]+/use"])
    assert len(files) == 3


def test_discover_pattern_keeps_same_stem_from_each_matched_platform(tmp_path):
    ds = tmp_path / "cua-lite" / "GUIAct"
    for platform in ("mobile", "browser"):
        write_records_to_parquet(
            [sample_trajectory_one_turn()],
            ds / platform / "use/train/use.parquet",
        )

    files = discover_files_under_paths([str(ds) + "/(mobile|browser)/use"])

    assert [rel for _, rel in files] == [
        "browser/use/train/use.parquet",
        "mobile/use/train/use.parquet",
    ]

def test_discover_multiple_roots(data_root):
    paths = [
        str(data_root / "OpenGVLab/ScaleCUA-Data") + "/[^/]+/use",
        str(data_root / "xlangai/AgentNet") + "/[^/]+/use",
    ]
    files = discover_files_under_paths(paths)
    assert len(files) == 4


def test_discover_overlapping_inputs_loads_each_physical_file_once(data_root):
    ds = data_root / "OpenGVLab/ScaleCUA-Data"

    files = discover_files_under_paths([
        str(ds),
        str(ds) + "/(desktop|browser)/use",
    ])

    assert len(files) == len({path.resolve() for path, _ in files})
    assert len(files) == len(discover_files_under_paths([str(ds)]))


def test_discover_excludes_validation_partitions_by_default(tmp_path):
    """The documented root-pointing invocation must not sweep the held-out carve.

    A canonical dataset keeps its split in the PATH and nothing in the row, so a
    validation shard swept in here is undetectable downstream.
    """
    ds = tmp_path / "cua-lite" / "Carved"
    for split in ("train", "validation"):
        write_records_to_parquet([sample_trajectory_one_turn()],
                                 ds / "desktop/use" / f"{split}.parquet")
        # multi-variant layout: the split is the parent dir, not the stem
        write_records_to_parquet([sample_trajectory_one_turn()],
                                 ds / "browser/use" / split / "rollout.parquet")

    rels = [rel for _, rel in discover_files_under_paths([str(ds)])]
    assert rels == ["browser/use/train/rollout.parquet", "desktop/use/train.parquet"], rels

    both = [rel for _, rel in discover_files_under_paths(
        [str(ds)], splits=("train", "validation"))]
    assert len(both) == 4, both

    only_val = [rel for _, rel in discover_files_under_paths([str(ds)], splits=("validation",))]
    assert only_val == ["browser/use/validation/rollout.parquet", "desktop/use/validation.parquet"]

    # pointed INSIDE the partition dir, the split is still readable from the stem
    inside = [rel for _, rel in discover_files_under_paths([str(ds / "desktop/use")])]
    assert inside == ["train.parquet"], inside


def test_discover_reports_the_files_its_split_filter_dropped(tmp_path, caplog):
    """The default filter changes the row count, so it must say so out loud."""
    ds = tmp_path / "Carved"
    for split in ("train", "validation"):
        write_records_to_parquet([sample_trajectory_one_turn()],
                                 ds / "desktop/use" / f"{split}.parquet")

    with caplog.at_level("WARNING", logger="lite.data.load"):
        discover_files_under_paths([str(ds)])
    assert "skipped 1 in 'validation'" in caplog.text, caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="lite.data.load"):
        discover_files_under_paths([str(ds)], splits=("train", "validation"))
    assert caplog.text == ""


def test_discover_keeps_raw_rollout_logs_under_any_split_dir(tmp_path):
    """A raw log-root has no canonical partition layout — nothing to filter by.

    Its ``<log_root>/<registry-split>/`` dirs are ENV task splits (train/rl/eval),
    a different vocabulary; the trajectory files must survive the default filter.
    """
    log_root = tmp_path / "logs"
    for split in ("train", "rl", "eval"):
        write_records_to_parquet(
            [sample_trajectory_one_turn()],
            log_root / split / "task_a" / "sample_00" / "trajectory.parquet",
        )
    rels = sorted(rel for _, rel in discover_files_under_paths([str(log_root)]))
    assert len(rels) == 3, rels


def test_discover_multiple_roots_keeps_duplicate_relative_stems(tmp_path):
    """Canonical datasets can be mixed even when their partition names match."""
    left = tmp_path / "left"
    right = tmp_path / "right"
    rel = Path("desktop/use/train.parquet")
    write_records_to_parquet([sample_trajectory_one_turn()], left / rel)
    write_records_to_parquet([sample_trajectory_one_turn()], right / rel)

    files = discover_files_under_paths([str(left), str(right)])

    assert len(files) == 2
    assert {path for path, _ in files} == {left / rel, right / rel}
