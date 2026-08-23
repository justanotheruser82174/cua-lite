"""Offline HF upload/download transport round trips."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.data.hf import upload as hf_upload
from lite.data.hf.download import download_dataset
from lite.data.staging import (
    ImageStore,
    coerce_messages,
    coerce_meta,
    image_rel_prefix,
    iter_parquet_rows,
    iter_partitions,
    partition_path,
    write_partition,
)


def test_hf_transport_round_trip_preserves_canonical_row_payload(tmp_path):
    name = "TransportRoundTrip"
    staging_root = tmp_path / "staged"
    staging = staging_root / "cua-lite" / name
    store = ImageStore(staging / "images", rel_prefix=image_rel_prefix(name))

    source_image = tmp_path / "screen.png"
    Image.new("RGB", (4, 4), (12, 34, 56)).save(source_image)
    image_rel = store.put(source_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "Click the target."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "click target"}],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 500]}]},
                    call_id="call_0000",
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0000",
            "content": [{"type": "text", "text": "ok"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=["click", "wait"],
        others={
            "task_id": "task_transport",
            "episode_return": 1.0,
            "terminated": True,
            "truncated": False,
        },
    ).to_dict()
    row = {"images": [image_rel], "messages": messages, "metadata": metadata}
    write_partition(
        [row],
        partition_path(
            staging,
            platform="desktop",
            task_type="use",
            split="train",
            variant="rollout",
        ),
    )

    [(platform, task_type, split, variant, parquet_path)] = list(iter_partitions(staging))
    hf_rows = list(iter_parquet_rows(parquet_path))
    hf_dataset = hf_upload._rows_to_dataset(hf_rows, store)
    snapshot = tmp_path / "snapshot"
    shard_path = partition_path(
        Path(""),
        platform=platform,
        task_type=task_type,
        split=split,
        variant=variant,
        shard_idx=0,
        shard_total=1,
    )
    out_shard = snapshot / shard_path
    out_shard.parent.mkdir(parents=True, exist_ok=True)
    hf_dataset.to_parquet(str(out_shard), batch_size=50, write_page_index=True)

    downloaded = download_dataset(
        name,
        out_dir=tmp_path / "downloaded" / "cua-lite" / name,
        snapshot_dir=snapshot,
    )

    [(_, _, _, _, downloaded_parquet)] = list(iter_partitions(downloaded))
    [downloaded_row] = list(iter_parquet_rows(downloaded_parquet))
    downloaded_image = downloaded.parent.parent / downloaded_row["images"][0]

    assert coerce_messages(downloaded_row["messages"]) == messages
    assert coerce_meta(downloaded_row["metadata"]) == metadata
    assert downloaded_image.is_file()
    assert Image.open(downloaded_image).getpixel((0, 0)) == (12, 34, 56)
