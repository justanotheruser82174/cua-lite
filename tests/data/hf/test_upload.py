from __future__ import annotations

import inspect
import json

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.data.hf.upload import push_dataset
from lite.data.staging import partition_path, write_partition


def _repo_meta() -> dict:
    return {
        "description": "Synthetic test dataset.",
        "original_urls": ["https://example.com/source"],
        "license": "Test license.",
        "citation": "Test citation.",
    }


def test_upload_reuploads_existing_shards_by_default() -> None:
    assert inspect.signature(push_dataset).parameters["skip_existing"].default is False


def test_upload_dry_run_is_transport_not_row_validator(tmp_path) -> None:
    staging_dir = tmp_path / "staged"
    preproc_dir = tmp_path / "preproc"
    preproc_dir.mkdir()
    (preproc_dir / "repo.json").write_text(json.dumps(_repo_meta()), encoding="utf-8")

    invalid_row = {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "click"}]},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": ["bad", 2]}]},
                    )
                ],
            },
        ],
        "metadata": LiteCUAMetadata(dims=("desktop", "use")).to_dict(),
    }
    write_partition(
        [invalid_row],
        partition_path(
            staging_dir,
            platform="desktop",
            task_type="use",
            split="train",
            variant="rollout",
        ),
    )

    push_dataset(
        "TestSet",
        staging_dir=staging_dir,
        preproc_dir=preproc_dir,
        dry_run=True,
    )

    assert (staging_dir / "README.md").is_file()
    assert json.loads((staging_dir / "stats.json").read_text())["rows_out"] == 1
