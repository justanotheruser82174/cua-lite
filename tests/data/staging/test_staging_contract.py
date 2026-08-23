"""Contract tests for the shared staging read/write boundary.

Run:
    uv run --extra data pytest tests/data/staging/test_staging_contract.py -q
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lite.core import LiteCUAMetadata
from lite.data.staging import (
    CorruptImageError,
    ImageStore,
    SplitAssigner,
    coerce_messages,
    collect_stats_from_disk,
    content_fingerprint,
    flush_buffers,
    hash_split,
    iter_parquet_rows,
    iter_partitions,
    partition_path,
    resolve_artifact_path,
    write_partition,
)


def test_image_store_names_only_decode_failures_as_corrupt(tmp_path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")

    with pytest.raises(CorruptImageError, match="bad.png"):
        ImageStore(tmp_path / "images").put(bad)


def test_image_store_never_leaves_a_partial_final_file(tmp_path, monkeypatch) -> None:
    src = tmp_path / "source.png"
    src.write_bytes(b"complete source")
    store = ImageStore(tmp_path / "images", verify=False)

    def interrupted_copy(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="interrupted"):
        store.put(src)

    assert not any(path.is_file() for path in (tmp_path / "images").rglob("*"))


def test_partition_writer_never_leaves_a_partial_final_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "train" / "use.parquet"

    def interrupted_write(table, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr("lite.data.staging.pq.write_table", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        write_partition([{"id": "a"}], destination)

    assert not destination.exists()
    assert not any(path.is_file() for path in destination.parent.iterdir())


def test_json_string_messages_preserve_explicit_empty_tool_calls() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
            "tool_calls": [],
        }
    ]

    assert coerce_messages(json.dumps(messages)) == messages


def test_flush_buffers_refuses_a_leftover_collapsed_partition(tmp_path) -> None:
    """A pre-PR ``<split>.parquet`` cannot coexist: both spellings parse.

    ``parse_partition_path`` accepts the 3-part form, so ``iter_partitions``
    would yield the split twice and ``collect_stats_from_disk`` would fold both
    into one key -- doubled row counts in the card and duplicate rows on the
    hub, with nothing raising. The refusal is what makes that unrepresentable.
    """
    (tmp_path / "browser" / "use").mkdir(parents=True)
    (tmp_path / "browser" / "use" / "train.parquet").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="collapsed layout"):
        flush_buffers(tmp_path, {("browser", "use", "train", "v"): [{"id": "a"}]})

    # refused BEFORE writing: no partial tree left behind
    assert not (tmp_path / "browser" / "use" / "train").exists()


def test_flush_buffers_checks_the_whole_tree_not_just_written_splits(tmp_path) -> None:
    """The scan covers cohorts this run does not touch.

    Keying it off the run's own buffers is the tempting shape and the wrong one:
    an operator who deletes the one file the error named, then reruns a job that
    happens to produce only ``train``, would publish the mixed tree the check
    exists to prevent.
    """
    (tmp_path / "desktop" / "use").mkdir(parents=True)
    (tmp_path / "desktop" / "use" / "validation.parquet").write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="validation.parquet"):
        flush_buffers(tmp_path, {("desktop", "use", "train", "v"): [{"id": "a"}]})


def test_flush_buffers_does_not_scan_the_image_store_for_partitions(
    tmp_path, monkeypatch
) -> None:
    """``images/<hash>/<hash>.parquet`` matches the collapsed shape positionally."""
    (tmp_path / "images" / "ab").mkdir(parents=True)
    (tmp_path / "images" / "ab" / "abcd.parquet").write_bytes(b"not-a-partition")

    original_glob = Path.glob

    def reject_root_glob(path, pattern):
        if path == tmp_path:
            raise AssertionError("flush_buffers must not scan from the dataset root")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", reject_root_glob)

    flush_buffers(tmp_path, {("browser", "use", "train", "v"): [{"id": "a"}]})

    assert (tmp_path / "browser" / "use" / "train" / "v.parquet").is_file()


def test_iter_partitions_does_not_recursively_walk_the_image_store(
    tmp_path, monkeypatch
) -> None:
    partition = tmp_path / "browser" / "use" / "train" / "v.parquet"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"partition")
    (tmp_path / "images" / "aa").mkdir(parents=True)
    (tmp_path / "images" / "aa" / "not-a-partition.parquet").write_bytes(b"image")

    def reject_rglob(*_args, **_kwargs):
        raise AssertionError("iter_partitions must not recursively scan images/")

    monkeypatch.setattr(Path, "rglob", reject_rglob)
    assert list(iter_partitions(tmp_path)) == [
        ("browser", "use", "train", "v", partition)
    ]


def test_collect_stats_rejects_legacy_web_platform_roots(tmp_path) -> None:
    write_partition(
        [{"id": "legacy"}],
        tmp_path / "web" / "use" / "train" / "legacy.parquet",
    )

    with pytest.raises(ValueError, match=r"legacy web/ platform partitions"):
        collect_stats_from_disk(tmp_path)


def test_collect_stats_rejects_mixed_web_and_browser_platform_roots(tmp_path) -> None:
    write_partition(
        [{"id": "legacy"}],
        tmp_path / "web" / "use" / "train" / "legacy.parquet",
    )
    write_partition(
        [{"id": "current"}],
        tmp_path / "browser" / "use" / "train" / "current.parquet",
    )

    with pytest.raises(ValueError, match=r"legacy web/ platform partitions"):
        collect_stats_from_disk(tmp_path)


def test_partition_path_keeps_a_dotted_variant_whole(tmp_path) -> None:
    """``stage --config-names`` spells variants like ``cfg.rl``.

    Building the filename with ``with_suffix`` reads ``.rl`` as an extension and
    replaces it, so two configs collapse onto one file.
    """
    assert partition_path(
        tmp_path, platform="browser", task_type="use", split="train", variant="cfg.rl"
    ).name == "cfg.rl.parquet"


def test_flush_buffers_keeps_existing_expanded_layout_on_subset_rerun(tmp_path) -> None:
    key_a = ("browser", "use", "train", "a")
    key_b = ("browser", "use", "train", "b")
    flush_buffers(tmp_path, {key_a: [{"id": "a1"}], key_b: [{"id": "b1"}]})
    flush_buffers(tmp_path, {key_a: [{"id": "a2"}]})

    partitions = list(iter_partitions(tmp_path))
    assert [(variant, list(iter_parquet_rows(path))[0]["id"])
            for _, _, _, variant, path in partitions] == [("a", "a2"), ("b", "b1")]


def test_decoded_canonical_messages_preserve_explicit_empty_tool_calls() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
            "tool_calls": [],
        }
    ]

    assert coerce_messages(messages) == messages


def test_decoded_canonical_messages_preserve_null_evidence_for_validators() -> None:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "index": None, "text": None},
            {"type": "text", "text": "finish", "index": None},
        ],
        "tool_calls": None,
    }]

    assert coerce_messages(messages) == messages


def test_json_string_messages_preserve_null_tool_argument_evidence() -> None:
    messages = [{
        "role": "assistant",
        "tool_calls": [{
            "id": "call_0000",
            "type": "function",
            "function": {"name": "back", "arguments": None},
        }],
    }]

    assert coerce_messages(json.dumps(messages)) == messages


# ---------------------------------------------------------------------------
# SplitAssigner: content-identical rows share a split
# ---------------------------------------------------------------------------

# ``hash_split`` on these ids, at the default val_frac/seed, disagrees:
# uv run python -c "from lite.data.staging import hash_split; \
#   print([i for i in map(str, range(400)) if hash_split(i) == 'validation'])"
_VAL_IDS = ("6", "125", "285", "291", "376")
_TRAIN_IDS = ("0", "1", "2", "3", "4", "5", "7", "8", "9", "10")


def _row(row_id: str, *, images=("a.png",), text: str = "same sample") -> dict:
    return {
        "images": list(images),
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "understanding"),
            others={"id": row_id, "source_id": f"src-{row_id}"},
        ).to_dict(),
    }


def _assigner(**kw) -> SplitAssigner:
    kw.setdefault("val_cap", 2000)
    return SplitAssigner(
        key_fn=lambda r: r["metadata"]["others"]["id"],
        key_desc="metadata.others.id",
        bucket_fn=lambda r: tuple(r["metadata"]["dims"]),
        bucket_desc="metadata.dims",
        **kw,
    )


def test_content_identical_rows_land_in_one_split_despite_disagreeing_keys() -> None:
    """The leak this closes: one upstream sample published under two ids."""
    a = _assigner()
    # On their own keys these two disagree, which is how the same sample ended up
    # in both splits: 5.05% of published validation rows had a byte-identical twin.
    assert hash_split(_VAL_IDS[0]) == "validation"
    assert hash_split(_TRAIN_IDS[0]) == "train"

    first = a.assign(_row(_VAL_IDS[0]))
    second = a.assign(_row(_TRAIN_IDS[0]))
    assert (first, second) == ("validation", "validation")


def test_the_first_member_of_a_group_decides_not_the_validation_side() -> None:
    """Order, not preference: co-location has no opinion about which split wins."""
    a = _assigner()
    assert a.assign(_row(_TRAIN_IDS[0])) == "train"
    assert a.assign(_row(_VAL_IDS[0])) == "train"


def test_a_row_with_no_content_twin_is_never_moved_even_when_the_cap_binds() -> None:
    """The acceptance criterion, stated as a test rather than a measurement.

    ``val_cap`` is deliberately small enough to bind, because that is the case
    where a naive memo *would* move unrelated rows: freeing or consuming cap
    budget shifts the cap boundary for every later row.
    """
    rows = [_row(str(i), text=f"row {i}") for i in range(400)]
    # ...plus one duplicate group whose members' own keys disagree.
    rows += [_row("twin-a", text="twinned"), _row("twin-b", text="twinned")]

    coalesced = _assigner(val_cap=2)
    uncoalesced = _assigner(val_cap=2)
    got = [coalesced.assign(r) for r in rows]
    want = [uncoalesced._assign_unique(r, bucket_extra=()) for r in rows]

    assert "validation" in want and want.count("validation") == 2, "cap must bind"
    movers = [r["metadata"]["others"]["id"] for r, g, w in zip(rows, got, want) if g != w]
    assert movers == [] or set(movers) <= {"twin-a", "twin-b"}
    assert coalesced._val_counts == uncoalesced._val_counts


def test_content_fingerprint_ignores_the_ids_and_reads_images_and_messages() -> None:
    assert content_fingerprint(_row("a")) == content_fingerprint(_row("b"))
    assert content_fingerprint(_row("a")) != content_fingerprint(_row("a", text="other"))
    assert content_fingerprint(_row("a")) != content_fingerprint(_row("a", images=("b.png",)))


def test_split_policy_names_the_key_and_the_cap_it_cannot_reproduce() -> None:
    """``val_frac``/``seed`` are useless without the key, and the cap is the one
    input that does NOT make the carve reproducible — say so, don't imply it."""
    described = _assigner(val_cap=7).describe()
    assert "metadata.others.id" in described
    assert "val_frac=0.02" in described and "seed=42" in described
    assert "val_cap=7" in described
    assert "cap-bound" in described and "iteration order" in described


def test_resolve_artifact_path_uses_single_anchor_ancestor_match(tmp_path) -> None:
    sample_dir = tmp_path / "logs" / "train" / "task_a" / "sample_00"
    artifact = sample_dir / "__stage_contract_artifacts__" / "screen.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"not-a-real-png")
    trajectory = sample_dir / "trajectory.parquet"
    trajectory.write_text("")

    assert (
        resolve_artifact_path(
            "__stage_contract_artifacts__/screen.png",
            anchor_path=trajectory,
        )
        == artifact.resolve()
    )


def test_resolve_artifact_path_rejects_ambiguous_ancestor_matches(tmp_path) -> None:
    sample_dir = tmp_path / "logs" / "train" / "task_a" / "sample_00"
    trajectory = sample_dir / "trajectory.parquet"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text("")

    rel = "__stage_contract_artifacts__/screen.png"
    for base in (sample_dir, tmp_path / "logs"):
        artifact = base / rel
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"not-a-real-png")

    with pytest.raises(ValueError, match="ambiguous relative artifact path"):
        resolve_artifact_path(rel, anchor_path=trajectory)
