from __future__ import annotations

from pathlib import Path

from lite.data.staging import split_of_partition_file


def test_split_of_partition_file_reads_the_parent_before_the_stem() -> None:
    assert split_of_partition_file(Path("desktop/use/validation/train.parquet")) == "validation"
    assert split_of_partition_file(Path("desktop/use/validation.parquet")) == "validation"
    assert split_of_partition_file(Path("desktop/use/train/rollout.parquet")) == "train"
    assert split_of_partition_file(Path("run/sample_00/trajectory.parquet")) is None
