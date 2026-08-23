"""End-to-end round-trip for lite.data.hf.stage ↔ lite.data.hf.unstage.

Proves the "continue collecting on top of a published run" loop:
    rollout log → stage → (publish) → unstage → resume skips done → re-stage = OLD.

Run: uv run pytest tests/data/hf/test_unstage.py -q
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from lite.core import LiteCUAMetadata, LiteGenericMetadata
from lite.core.errors import LiteContractError
from lite.core.messages.final import CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments
from lite.core.tools.schemas import make_tool_schema
from lite.data.hf.stage import stage
from lite.data.hf.unstage import unstage
from lite.data.staging import (
    coerce_messages,
    coerce_meta,
    parse_partition_path,
)
from lite.data.utils.rows import validate_canonical_rows
from lite.infer.rollout import TaskSpec, get_pending
from lite.utils.parquet import write_records_to_parquet


def _make_rollout_log(root: Path, img_dir: Path, tasks: list[str], split: str = "train") -> None:
    """Minimal rollout log-root with trajectory parquet and summary JSON."""
    for i, task_id in enumerate(tasks):
        png = img_dir / f"{task_id}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (10 * i, 20, 30)).save(png)

        sample_dir = root / split / task_id / "sample_00"
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_records_to_parquet(
            [{
                "images": [str(png.resolve())],  # absolute → stage resolves directly
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": f"task {task_id}"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                ],
                "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                        "task_id": task_id,
                        "env_id": "webgym",
                        "episode_return": 1.0,
                        "terminated": True,
                        "truncated": False,
                    },
        ).to_dict(),
            }],
            sample_dir / "trajectory.parquet",
            json_fields=("messages", "metadata"),
        )
        (sample_dir / "summary.json").write_text(json.dumps({
            "n_turns": 1, "episode_return": 1.0, "terminated": True,
            "truncated": False, "duration_seconds": 1.5,
        }))


def _write_stage_record(tmp_path: Path, record: dict, task_id: str) -> Path:
    sample_dir = tmp_path / "logs" / "train" / task_id / "sample_00"
    sample_dir.mkdir(parents=True)
    write_records_to_parquet(
        [record],
        sample_dir / "trajectory.parquet",
        json_fields=("messages", "metadata"),
    )
    return tmp_path / "logs"


def test_stage_unstage_round_trip_enables_resume(tmp_path):
    tasks = ["task_a", "task_b"]
    log_root = tmp_path / "logs" / "run1"
    _make_rollout_log(log_root, tmp_path / "imgs", tasks)

    # stage → canonical. out_dir MUST be <root>/cua-lite/<name> so unstage's
    # image-root derivation (dataset.parent.parent) lands on <root>.
    name = "RoundTrip"
    dataset = tmp_path / "ds" / "cua-lite" / name
    stage([log_root], name=name, out_dir=dataset, filter_expr=None)
    assert list(dataset.rglob("*.parquet")), "stage produced no canonical parquet"
    assert (dataset / "images").is_dir(), "stage produced no image store"

    # unstage → a FRESH log-root
    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    # (1) RESUME GATE — every collected (task, sample_00) reads as done.
    specs = [TaskSpec(t, "webgym", "train") for t in tasks]
    assert get_pending(resumed, specs, group_size=1) == [], \
        "resume would wrongly re-run already-collected samples"
    # a NOT-yet-collected task is still pending → resume samples ONLY it.
    new_spec = TaskSpec("task_new", "webgym", "train")
    pending_new = get_pending(resumed, specs + [new_spec], group_size=1)
    assert pending_new == [(new_spec, 0)]

    # (2) trajectory.parquet round-trips — data + resolvable image + recovered outcomes.
    for task_id in tasks:
        sd = resumed / "train" / task_id / "sample_00"
        df = pd.read_parquet(sd / "trajectory.parquet")
        assert len(df) == 1
        row = df.iloc[0]
        imgs = list(row["images"])
        assert imgs and Path(imgs[0]).exists(), f"image path unresolvable: {imgs}"
        messages = coerce_messages(row["messages"])
        assert isinstance(messages, list)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        md = coerce_meta(row["metadata"])
        assert "task_id" not in md and "env_id" not in md
        assert md["others"]["task_id"] == task_id and md["others"]["env_id"] == "webgym"
        assert md["others"]["episode_return"] == 1.0
        assert md["others"]["terminated"] is True
        summary = json.loads((sd / "summary.json").read_text())
        assert summary["episode_return"] == 1.0 and summary["terminated"] is True

    # (3) RE-STAGE the resumed log-root → canonical' equals the original (idempotent):
    # same row count AND identical content-addressed image set.
    dataset2 = tmp_path / "ds2" / "cua-lite" / name
    stage([resumed], name=name, out_dir=dataset2, filter_expr=None)
    n1 = sum(len(pd.read_parquet(p)) for p in dataset.rglob("*.parquet"))
    n2 = sum(len(pd.read_parquet(p)) for p in dataset2.rglob("*.parquet"))
    assert n1 == n2 == len(tasks), f"re-stage changed row count: {n1} vs {n2}"
    hashes1 = {p.name for p in (dataset / "images").rglob("*.*")}
    hashes2 = {p.name for p in (dataset2 / "images").rglob("*.*")}
    assert hashes1 == hashes2, "re-stage produced different image hashes (content drift)"


def _split_counts(dataset: Path) -> dict[str, int]:
    """rows per canonical ``train``/``validation`` partition of *dataset*."""
    counts: dict[str, int] = {}
    for p in dataset.rglob("*.parquet"):
        if "images" in p.parts:
            continue
        parsed = parse_partition_path(p.relative_to(dataset))
        assert parsed is not None, f"non-canonical partition path: {p}"
        counts[parsed[2]] = counts.get(parsed[2], 0) + len(pd.read_parquet(p))
    return counts


def test_round_trip_preserves_the_validation_carve(tmp_path):
    """stage(--val-frac 0.02) → unstage → re-stage(DEFAULTS) keeps the same carve.

    The re-stage runs at stage's default ``val_frac=0``, which sends EVERY row to
    ``train``; the carve survives only because the reconstructed log-root carries
    the source partition per row (``metadata.others.split``). The hint is
    transient — it must not appear on the re-staged rows.
    """
    # task_006 hashes to `validation` at val_frac=0.02/seed=42; the other two to `train`.
    tasks = ["task_000", "task_001", "task_006"]
    log_root = tmp_path / "logs" / "run1"
    _make_rollout_log(log_root, tmp_path / "imgs", tasks)

    name = "CarveRoundTrip"
    dataset = tmp_path / "ds" / "cua-lite" / name
    stage([log_root], name=name, out_dir=dataset, filter_expr=None, val_frac=0.02)
    before = _split_counts(dataset)
    assert before == {"train": 2, "validation": 1}, before

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    dataset2 = tmp_path / "ds2" / "cua-lite" / name
    stage([resumed], name=name, out_dir=dataset2, filter_expr=None)  # default val_frac=0
    assert _split_counts(dataset2) == before

    for p in dataset2.rglob("*.parquet"):
        if "images" in p.parts:
            continue
        for md in (coerce_meta(m) for m in pd.read_parquet(p)["metadata"]):
            assert "split" not in md["others"], "routing hint leaked onto a published row"

    # repo.json records the carve's INPUTS, so the assignment stays reproducible
    # without any row carrying a split marker.
    notes = json.loads((dataset / "repo.json").read_text())["extra_notes"]
    assert "val_frac=0.02" in notes and "seed=42" in notes, notes


def test_stage_rejects_noncanonical_unstage_split_hint(tmp_path):
    """The transient ``others.split`` hint is a stage boundary, not upload policy."""
    log_root = tmp_path / "logs" / "run1"
    _make_rollout_log(log_root, tmp_path / "imgs", ["task_bad"])

    trajectory = next(log_root.rglob("trajectory.parquet"))
    row = pd.read_parquet(trajectory).iloc[0].to_dict()
    row["images"] = list(row["images"])
    row["messages"] = coerce_messages(row["messages"])
    metadata = coerce_meta(row["metadata"])
    metadata.setdefault("others", {})["split"] = "test"
    row["metadata"] = metadata
    write_records_to_parquet(
        [row],
        trajectory,
        json_fields=("messages", "metadata"),
    )

    with pytest.raises(ValueError, match=r"metadata\.others\.split must be one of"):
        stage(
            [log_root],
            name="BadSplit",
            out_dir=tmp_path / "ds" / "cua-lite" / "BadSplit",
            filter_expr=None,
        )


def test_unstage_refuses_stale_split_without_overwrite(tmp_path):
    dataset = tmp_path / "ds" / "cua-lite" / "FreshUnstage"
    log_root = tmp_path / "logs" / "source"
    _make_rollout_log(log_root, tmp_path / "imgs", ["task_a", "task_b"])
    stage([log_root], name="FreshUnstage", out_dir=dataset, filter_expr=None)

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")
    with pytest.raises(FileExistsError, match="--overwrite"):
        unstage(dataset, log_root=resumed, splits="train")

    dataset_one = tmp_path / "ds_one" / "cua-lite" / "FreshUnstage"
    log_root_one = tmp_path / "logs" / "source_one"
    _make_rollout_log(log_root_one, tmp_path / "imgs_one", ["task_a"])
    stage([log_root_one], name="FreshUnstage", out_dir=dataset_one, filter_expr=None)
    unstage(dataset_one, log_root=resumed, splits="train", overwrite=True)
    assert {p.name for p in (resumed / "train").iterdir()} == {"task_a"}


def test_unstage_refuses_output_overlapping_dataset(tmp_path):
    dataset = tmp_path / "ds" / "cua-lite" / "OverlapUnstage"
    log_root = tmp_path / "logs" / "source"
    _make_rollout_log(log_root, tmp_path / "imgs", ["task_a"])
    stage([log_root], name="OverlapUnstage", out_dir=dataset, filter_expr=None)

    with pytest.raises(ValueError, match="must not overlap protected input root"):
        unstage(dataset, log_root=dataset / "logs", splits="train")


def test_multi_config_unstage_routes_each_config_to_its_split(tmp_path):
    """One repo, TWO configs → unstage each into its OWN registry-split dir.

    Mirrors the cross-machine resume for lite.scalecua: `rl` + `train` live in one
    dataset as configs `cfg.rl` / `cfg.train`; --config-names lets unstage land each
    under `<log_root>/rl/` resp. `<log_root>/train/` with NO cross-contamination, so
    per-split resume is correct and a final re-stage keeps the two configs clean.
    """
    rl_tasks, train_tasks = ["rl_a", "rl_b"], ["tr_a", "tr_b", "tr_c"]
    rl_log, train_log = tmp_path / "logs" / "rl", tmp_path / "logs" / "train"
    _make_rollout_log(rl_log, tmp_path / "imgs_rl", rl_tasks)
    _make_rollout_log(train_log, tmp_path / "imgs_tr", train_tasks)

    name = "MultiCfg"
    dataset = tmp_path / "ds" / "cua-lite" / name
    stage([rl_log, train_log], name=name, out_dir=dataset, filter_expr=None,
          config_names=["cfg.rl", "cfg.train"])
    staged = {p.name[: -len(".parquet")] for p in dataset.rglob("*.parquet")}
    assert {"cfg.rl", "cfg.train"} <= staged, f"stage did not emit both configs: {staged}"

    # unstage each config into its own registry split — ONE download, two passes.
    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="rl", config_names=["cfg.rl"])
    unstage(dataset, log_root=resumed, splits="train", config_names=["cfg.train"])

    # (1) NO cross-contamination: each split dir holds exactly its own tasks.
    assert {p.name for p in (resumed / "rl").iterdir()} == set(rl_tasks)
    assert {p.name for p in (resumed / "train").iterdir()} == set(train_tasks)

    # (2) per-split resume gate is correct for BOTH splits.
    assert get_pending(resumed, [TaskSpec(t, "webgym", "rl") for t in rl_tasks],
                       group_size=1) == []
    assert get_pending(resumed, [TaskSpec(t, "webgym", "train") for t in train_tasks],
                       group_size=1) == []

    # (3) a config name matching nothing fails loud (typo guard).
    with pytest.raises(SystemExit):
        unstage(dataset, log_root=tmp_path / "nope", splits="train",
                config_names=["cfg.does_not_exist"])


def test_config_match_ignores_dataset_ancestor_dirs(tmp_path):
    """A dataset stored under a dir NAMED like a config must not match everything.

    Regression: _matches_config scanned all path components including ancestors
    of the dataset dir, so `.../cfg.rl/cua-lite/DS` made every parquet (incl.
    cfg.train) match --config-names ["cfg.rl"] and train rows contaminated the
    rl split.
    """
    rl_tasks, train_tasks = ["rl_a"], ["tr_a", "tr_b"]
    rl_log, train_log = tmp_path / "logs" / "rl", tmp_path / "logs" / "train"
    _make_rollout_log(rl_log, tmp_path / "imgs_rl", rl_tasks)
    _make_rollout_log(train_log, tmp_path / "imgs_tr", train_tasks)

    name = "AncestorTrap"
    dataset = tmp_path / "cfg.rl" / "cua-lite" / name  # ancestor named like a config
    stage([rl_log, train_log], name=name, out_dir=dataset, filter_expr=None,
          config_names=["cfg.rl", "cfg.train"])

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="rl", config_names=["cfg.rl"])
    assert {p.name for p in (resumed / "rl").iterdir()} == set(rl_tasks)


def _canonical_dataset(
    dataset: Path, partitions: dict[tuple[str, str, str], list[dict]]
) -> Path:
    """Hand-build a canonical dataset from metadata fragments."""
    img_rel = f"cua-lite/{dataset.name}/images/ab/abcd.png"
    img_abs = dataset.parent.parent / img_rel
    img_abs.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (7, 8, 9)).save(img_abs)
    for (platform, task_type, split), metadata_rows in partitions.items():
        write_records_to_parquet(
            [
                {
                    "images": [img_rel],
                    "messages": [
                        {"role": "user", "content": [{"type": "image", "index": 0}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                    ],
                    "metadata": LiteCUAMetadata(
            dims=(platform, task_type),
            extra_tool_schemas=[],
            valid_actions=None,
            others=dict(row_meta),
        ).to_dict(),
                }
                for row_meta in metadata_rows
            ],
            dataset / platform / task_type / f"{split}.parquet",
            json_fields=("messages", "metadata"),
        )
    return dataset


def test_unstage_writes_no_summary_for_a_single_turn_cohort(tmp_path):
    """A grounding row has no episode: a summary.json for it invents episode_return."""
    dataset = _canonical_dataset(
        tmp_path / "ds" / "cua-lite" / "GroundingOnly",
        {("desktop", "grounding.point", "train"): [{"task_id": "ground_a", "id": "ground_a"}]},
    )

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    assert list(resumed.rglob("summary.json")) == []
    assert list(resumed.rglob("trajectory.parquet")) == []


def test_unstage_rebuilds_a_grounding_env_rollout_row(tmp_path):
    """A single-step grounding ENV rollout row has a REAL episode_return.

    ``screenspot_pro`` / ``osworld_g`` are ``grounding.point`` **envs**, and
    ``build_trajectory_summary``'s inputs are all present on such a row, so skipping it on
    ``task_type`` alone loses a scored outcome instead of declining to invent one.
    Reproduced end-to-end before this qualifier: staging
    ``.logs/famval/{osworld_g,screenspot_pro}`` kept 64/64 rows and unstage
    rebuilt 0 sample dirs.
    """
    dataset = _canonical_dataset(
        tmp_path / "ds" / "cua-lite" / "GroundingEnv",
        {
            ("desktop", "grounding.point", "train"): [
                {"task_id": "osworld_g_a", "episode_return": 1.0,
                 "terminated": True, "truncated": False},
            ]
        },
    )

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    sample = resumed / "train" / "osworld_g_a" / "sample_00"
    assert json.loads((sample / "summary.json").read_text()) == {
        "n_turns": 1, "episode_return": 1.0, "terminated": True,
        "truncated": False, "duration_seconds": 0.0,
    }
    assert (sample / "trajectory.parquet").is_file()


def test_unstage_carries_use_outcomes_into_the_resume_gate(tmp_path):
    """The resume gate's outcomes come from metadata.others."""
    dataset = _canonical_dataset(
        tmp_path / "ds" / "cua-lite" / "UseOnly",
        {
            ("desktop", "use", "train"): [
                {"task_id": "use_a", "episode_return": 1.0,
                 "terminated": True, "truncated": False},
                {"task_id": "use_a", "episode_return": 0.0,
                 "terminated": False, "truncated": True},
            ]
        },
    )

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    task_dir = resumed / "train" / "use_a"
    assert json.loads((task_dir / "sample_00" / "summary.json").read_text()) == {
        "n_turns": 1, "episode_return": 1.0, "terminated": True,
        "truncated": False, "duration_seconds": 0.0,
    }
    assert json.loads((task_dir / "sample_01" / "summary.json").read_text()) == {
        "n_turns": 1, "episode_return": 0.0, "terminated": False,
        "truncated": True, "duration_seconds": 0.0,
    }


def test_unstage_counts_turns_off_messages_not_images(tmp_path):
    """``n_turns`` is a message fact, so a batch's extra frames must not inflate it.

    One batch of three GUI actions stores three frames but is ONE turn, and the
    two the result does not reference are stored anyway. Counting images would
    report 4 turns for a 2-turn episode -- and ``n_turns`` is read back by
    ``lite/infer/rollout.py`` and the eval aggregators, so the wrong number does
    not stay local. The right source is the one ``adapter.unroll`` uses, which is
    what makes this agree with the logger's ``len(steps)``.
    """
    dataset = tmp_path / "ds" / "cua-lite" / "BatchFrames"
    img_rel = f"cua-lite/{dataset.name}/images/ab/abcd.png"
    img_abs = dataset.parent.parent / img_rel
    img_abs.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (7, 8, 9)).save(img_abs)
    write_records_to_parquet(
        [{
            "images": [img_rel] * 4,          # reset frame + one per batch slot
            "messages": [
                {"role": "user", "content": [{"type": "image", "index": 0}]},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_0", "type": "function",
                    "function": {"name": "computer", "arguments": {"actions": [
                        {"action": "click", "coordinate": [1, 1]},
                        {"action": "click", "coordinate": [2, 2]},
                        {"action": "click", "coordinate": [3, 3]},
                    ]}},
                }]},
                # only the batch's LAST frame is model-visible; 1 and 2 are stored orphans
                {"role": "tool", "tool_call_id": "call_0",
                 "content": [{"type": "image", "index": 3}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "batched", "episode_return": 1.0,
                           "terminated": True, "truncated": False},
        ).to_dict(),
        }],
        dataset / "desktop" / "use" / "train.parquet",
        json_fields=("messages", "metadata"),
    )

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    summary = json.loads(
        (resumed / "train" / "batched" / "sample_00" / "summary.json").read_text()
    )
    assert summary["n_turns"] == 2


def test_unstage_skips_single_turn_rows_without_abandoning_the_use_rows(tmp_path):
    """The skip is per-row: a mixed dataset must still reconstruct its use part."""
    dataset = _canonical_dataset(
        tmp_path / "ds" / "cua-lite" / "Mixed",
        {
            ("desktop", "use", "train"): [
                {"task_id": "use_a", "episode_return": 1.0, "terminated": True},
            ],
            ("desktop", "grounding.point", "train"): [
                {"task_id": "ground_a", "id": "ground_a"},
            ],
            ("desktop", "understanding", "train"): [
                {"task_id": "understand_a", "id": "understand_a"},
            ],
        },
    )

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    assert {p.name for p in (resumed / "train").iterdir()} == {"use_a"}
    assert (resumed / "train" / "use_a" / "sample_00" / "trajectory.parquet").is_file()


def test_unstage_resume_resolves_the_use_task_and_leaves_single_turn_unclaimed(tmp_path):
    """Rollout resume must skip the collected use task and never claim a grounding row."""
    dataset = _canonical_dataset(
        tmp_path / "ds" / "cua-lite" / "Resume",
        {
            ("desktop", "use", "train"): [
                {"task_id": "use_a", "episode_return": 1.0, "terminated": True},
            ],
            ("desktop", "grounding.point", "train"): [
                {"task_id": "ground_a", "id": "ground_a"},
            ],
        },
    )

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")

    use_spec = TaskSpec("use_a", "webgym", "train")
    ground_spec = TaskSpec("ground_a", "webgym", "train")
    assert get_pending(resumed, [use_spec], group_size=1) == []
    assert get_pending(resumed, [use_spec, ground_spec], group_size=1) == [(ground_spec, 0)]


def test_stage_requires_metadata_task_id_instead_of_deriving_from_path(tmp_path):
    task_id = "path_task"
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"env_id": "handmade"},
        ).to_dict(),
        },
        task_id,
    )

    dataset = tmp_path / "ds" / "cua-lite" / "PathFallback"
    with pytest.raises(ValueError, match=r"metadata\.others\.task_id is required"):
        stage([logs], name="PathFallback", out_dir=dataset, filter_expr=None)


def test_stage_preserves_messages_opaquely(tmp_path):
    img = tmp_path / "screen.png"
    Image.new("RGB", (4, 4), (40, 50, 60)).save(img)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "do it"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "action_description", "text": "Click then type."}],
            "tool_calls": [make_tool_call(
                "computer",
                {"actions": [
                    {"action": "click", "coordinate": [1, 2]},
                    {"action": "type", "text": "hello"},
                ]},
                call_id="call_0000",
            )],
            "raw_response": {"text": "verbatim", "adapter_key": "qwen3_vl@desktop@use"},
            CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY: {
                "version": 1,
                "stop_reason": "text",
                "content_types": ["text"],
            },
        },
        {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "text", "text": "ok"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    ]
    schema = make_tool_schema(
        "bash",
        parameters={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    )
    schema2 = make_tool_schema(
        "goto",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )
    record = {
        "images": [str(img.resolve())],
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[schema, schema2],
            valid_actions=None,
            others={"task_id": "task_raw"},
        ).to_dict(),
    }
    sample_dir = tmp_path / "logs" / "train" / "task_raw" / "sample_00"
    sample_dir.mkdir(parents=True)
    write_records_to_parquet(
        [record],
        sample_dir / "trajectory.parquet",
        json_fields=("messages", "metadata"),
    )

    dataset = tmp_path / "ds" / "cua-lite" / "RawPolicy"
    stage([tmp_path / "logs"], name="RawPolicy", out_dir=dataset, filter_expr=None)

    [parquet] = [p for p in dataset.rglob("*.parquet") if "images" not in p.parts]
    row = pd.read_parquet(parquet).iloc[0]
    got = coerce_messages(row["messages"])
    # Everything transports opaquely EXCEPT private runtime sidecars. The input
    # fixture deliberately carries both so this assertion cannot pass vacuously.
    expected = deepcopy(messages)
    expected[1].pop("raw_response")
    expected[1].pop(CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY)

    assert got == expected
    assert "raw_response" not in got[1]
    assert CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY not in got[1]
    assert got[2]["role"] == "tool"
    assert got[-1]["content"] == [{"type": "text", "text": "Done."}]
    assert tool_call_arguments(got[1]["tool_calls"][0])["actions"][1] == {
        "action": "type",
        "text": "hello",
    }
    assert coerce_meta(row["metadata"])["extra_tool_schemas"] == [schema, schema2]

    resumed = tmp_path / "logs" / "resumed"
    unstage(dataset, log_root=resumed, splits="train")
    resumed_row = pd.read_parquet(
        resumed / "train" / "task_raw" / "sample_00" / "trajectory.parquet"
    ).iloc[0]
    assert isinstance(resumed_row["messages"], str)
    assert isinstance(resumed_row["metadata"], str)
    assert coerce_messages(resumed_row["messages"]) == expected
    assert coerce_meta(resumed_row["metadata"])["extra_tool_schemas"] == [schema, schema2]


def test_stage_strips_content_only_final_diagnostic_before_publish(tmp_path):
    diagnostic = {
        "version": 1,
        "stop_reason": "reasoning_only",
        "content_types": ["inline_reasoning"],
        "text_part_count": 0,
        "inline_reasoning_part_count": 1,
        "action_description_part_count": 0,
        "history_summary_part_count": 0,
        "has_native_reasoning": False,
        "has_raw_response": True,
        "has_model_output_error": False,
        "visible_text": False,
        "raw_response": {"must": "not publish"},
        "original_text": "must not publish",
    }
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY: diagnostic,
                },
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "diag_task"},
        ).to_dict(),
        },
        "diag_task",
    )

    dataset = tmp_path / "ds" / "cua-lite" / "Diag"
    stage([logs], name="Diag", out_dir=dataset, filter_expr=None)

    [parquet] = [p for p in dataset.rglob("*.parquet") if "images" not in p.parts]
    row = pd.read_parquet(parquet).iloc[0]
    messages = coerce_messages(row["messages"])
    metadata = coerce_meta(row["metadata"])
    assert CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY not in messages[-1]
    assert "content_only_final" not in metadata["others"]
    assert "raw_response" not in json.dumps(metadata["others"], sort_keys=True)
    assert "original_text" not in json.dumps(metadata["others"], sort_keys=True)
    validate_canonical_rows(
        [{"images": list(row["images"]), "messages": messages, "metadata": metadata}],
        "diag",
    )


def test_stage_strips_raw_content_only_final_sidecars_without_rewriting_text(tmp_path):
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done.\nFinal answer."}],
                    "raw_response": {
                        "adapter_key": "qwen3_vl@desktop@use",
                        "text": "Done.\nFinal answer.",
                    },
                },
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "raw_final_task"},
        ).to_dict(),
        },
        "raw_final_task",
    )

    dataset = tmp_path / "ds" / "cua-lite" / "RawFinal"
    stage([logs], name="RawFinal", out_dir=dataset, filter_expr=None)

    [parquet] = [p for p in dataset.rglob("*.parquet") if "images" not in p.parts]
    row = pd.read_parquet(parquet).iloc[0]
    messages = coerce_messages(row["messages"])
    metadata = coerce_meta(row["metadata"])
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Done.\nFinal answer."}],
    }
    assert "content_only_final" not in metadata["others"]
    validate_canonical_rows(
        [{"images": list(row["images"]), "messages": messages, "metadata": metadata}],
        "raw-final",
    )


def test_stage_preserves_non_use_final_label(tmp_path):
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is shown?"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "A settings dialog."}],
                    "raw_response": {
                        "adapter_key": "qwen3_vl@desktop@understanding",
                        "text": "A settings dialog.",
                    },
                },
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "understanding"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "understanding_task"},
        ).to_dict(),
        },
        "understanding_task",
    )

    dataset = tmp_path / "ds" / "cua-lite" / "Understanding"
    stage([logs], name="Understanding", out_dir=dataset, filter_expr=None)

    [parquet] = [p for p in dataset.rglob("*.parquet") if "images" not in p.parts]
    row = pd.read_parquet(parquet).iloc[0]
    messages = coerce_messages(row["messages"])
    metadata = coerce_meta(row["metadata"])
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "A settings dialog."}],
    }
    assert "content_only_final" not in metadata["others"]
    validate_canonical_rows(
        [{"images": list(row["images"]), "messages": messages, "metadata": metadata}],
        "understanding",
    )


def test_stage_durable_parse_failure_overrides_conflicting_content_final_diagnostic(tmp_path):
    diagnostic = {
        "version": 1,
        "stop_reason": "text",
        "content_types": ["text"],
        "text_part_count": 1,
        "inline_reasoning_part_count": 0,
        "action_description_part_count": 0,
        "history_summary_part_count": 0,
        "has_native_reasoning": False,
        "has_raw_response": True,
        "has_model_output_error": False,
        "visible_text": True,
        "raw_response": {"must": "not publish SECRET"},
        "original_text": "not publish SECRET",
    }
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Done."}],
                    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY: diagnostic,
                },
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                    "task_id": "parse_failure_task",
                    "stop_reason": "parse_failure",
                },
        ).to_dict(),
        },
        "parse_failure_task",
    )

    dataset = tmp_path / "ds" / "cua-lite" / "ParseFailure"
    stage([logs], name="ParseFailure", out_dir=dataset, filter_expr=None)

    [parquet] = [p for p in dataset.rglob("*.parquet") if "images" not in p.parts]
    row = pd.read_parquet(parquet).iloc[0]
    messages = coerce_messages(row["messages"])
    metadata = coerce_meta(row["metadata"])
    others = metadata["others"]
    assert CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY not in messages[-1]
    assert others["stop_reason"] == "parse_failure"
    assert "content_only_final" not in others
    assert "SECRET" not in json.dumps(others, sort_keys=True)
    validate_canonical_rows(
        [{"images": list(row["images"]), "messages": messages, "metadata": metadata}],
        "parse_failure",
    )


def test_stage_drops_arbitrary_stop_reason_without_generating_content_final_diagnostic(tmp_path):
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                    "task_id": "arbitrary_stop_reason_task",
                    "stop_reason": "SECRET arbitrary prose",
                },
        ).to_dict(),
        },
        "arbitrary_stop_reason_task",
    )

    dataset = tmp_path / "ds" / "cua-lite" / "ArbitraryStopReason"
    stage([logs], name="ArbitraryStopReason", out_dir=dataset, filter_expr=None)

    [parquet] = [p for p in dataset.rglob("*.parquet") if "images" not in p.parts]
    row = pd.read_parquet(parquet).iloc[0]
    metadata = coerce_meta(row["metadata"])
    others = metadata["others"]
    assert "stop_reason" not in others
    assert "content_only_final" not in others
    assert "SECRET" not in json.dumps(others, sort_keys=True)


@pytest.mark.parametrize("stop_reason", ["content_only_final", "empty"])
def test_stage_drops_routine_no_tool_final_stop_reason(tmp_path, stop_reason):
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                    "task_id": f"{stop_reason}_task",
                    "stop_reason": stop_reason,
                },
        ).to_dict(),
        },
        f"{stop_reason}_task",
    )

    dataset = tmp_path / "ds" / "cua-lite" / "RoutineNoToolFinal"
    stage([logs], name="RoutineNoToolFinal", out_dir=dataset, filter_expr=None)

    [parquet] = [p for p in dataset.rglob("*.parquet") if "images" not in p.parts]
    row = pd.read_parquet(parquet).iloc[0]
    metadata = coerce_meta(row["metadata"])
    assert "stop_reason" not in metadata["others"]
    assert "content_only_final" not in metadata["others"]
    validate_canonical_rows(
        [{
            "images": list(row["images"]),
            "messages": coerce_messages(row["messages"]),
            "metadata": metadata,
        }],
        stop_reason,
    )


def test_stage_rejects_rows_with_existing_content_only_final_metadata(tmp_path):
    """No producer writes ``metadata.others.content_only_final`` (a private local
    diagnostic, see ``validate_canonical_rows``), so stage does not repair it --
    a row that already carries it is a producer bug and must fail the gate."""
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={
                    "task_id": "existing_diag_task",
                    "content_only_final": {
                        "version": 1,
                        "stop_reason": "text",
                        "content_types": ["text"],
                        "raw_response": {"must": "not republish SECRET"},
                        "original_text": "not republish SECRET",
                    },
                },
        ).to_dict(),
        },
        "existing_diag_task",
    )

    with pytest.raises(ValueError, match="content_only_final.*must not be published"):
        stage(
            [logs],
            name="ExistingDiag",
            out_dir=tmp_path / "ds" / "cua-lite" / "ExistingDiag",
            filter_expr=None,
        )


def test_stage_rejects_noncanonical_tool_calls(tmp_path):
    img = tmp_path / "screen.png"
    Image.new("RGB", (4, 4), (40, 50, 60)).save(img)

    record = {
        "images": [str(img.resolve())],
        "messages": [
            {"role": "user", "content": [{"type": "image", "index": 0}]},
                {
                    "role": "assistant",
                    "content": [],
                    "tool_calls": [make_tool_call(
                        "computer",
                        {"foo": "bar"},
                        call_id="call_0000",
                    )],
                },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": "task_bad"},
        ).to_dict(),
    }
    sample_dir = tmp_path / "logs" / "train" / "task_bad" / "sample_00"
    sample_dir.mkdir(parents=True)
    write_records_to_parquet(
        [record],
        sample_dir / "trajectory.parquet",
        json_fields=("messages", "metadata"),
    )

    with pytest.raises(ValueError, match="computer.*actions"):
        stage(
            [tmp_path / "logs"],
            name="BadPolicy",
            out_dir=tmp_path / "ds" / "cua-lite" / "BadPolicy",
            filter_expr=None,
        )


def test_stage_rejects_standalone_call_missing_extra_schema(tmp_path):
    img = tmp_path / "screen.png"
    Image.new("RGB", (4, 4), (40, 50, 60)).save(img)
    task_id = "task_missing_schema"
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [str(img.resolve())],
            "messages": [
                {"role": "user", "content": [{"type": "image", "index": 0}]},
                {
                    "role": "assistant",
                    "content": [],
                    "tool_calls": [make_tool_call(
                        "goto",
                        {"url": "https://example.com"},
                        call_id="call_goto",
                    )],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_goto",
                    "content": [{"type": "text", "text": "result"}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": task_id},
        ).to_dict(),
        },
        task_id,
    )

    with pytest.raises(ValueError, match="missing from metadata\\.extra_tool_schemas"):
        stage(
            [logs],
            name="MissingSchema",
            out_dir=tmp_path / "ds" / "cua-lite" / "MissingSchema",
            filter_expr=None,
        )


def test_stage_rejects_metadata_split_leak(tmp_path):
    img = tmp_path / "screen.png"
    Image.new("RGB", (4, 4), (40, 50, 60)).save(img)
    task_id = "task_split_leak"
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [str(img.resolve())],
            "messages": [
                {"role": "user", "content": [{"type": "image", "index": 0}]},
                {
                    "role": "assistant",
                    "content": [],
                    "tool_calls": [make_tool_call(
                        "computer",
                        {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                        call_id="call_computer",
                    )],
                },
            ],
            "metadata": {**LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"task_id": task_id},
        ).to_dict(), 'split': "train"},
        },
        task_id,
    )

    with pytest.raises(ValueError, match="metadata\\.split"):
        stage(
            [logs],
            name="SplitLeak",
            out_dir=tmp_path / "ds" / "cua-lite" / "SplitLeak",
            filter_expr=None,
        )


def test_stage_rejects_flat_extra_tool_schema(tmp_path):
    img = tmp_path / "screen.png"
    Image.new("RGB", (4, 4), (40, 50, 60)).save(img)
    task_id = "task_flat_schema"
    logs = _write_stage_record(
        tmp_path,
        {
            "images": [str(img.resolve())],
            "messages": [
                {"role": "user", "content": [{"type": "image", "index": 0}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": {
                "metadata_kind": "cua",
                "dims": ["desktop", "use"],
                "extra_tool_schemas": [{
                    "type": "function",
                    "name": "goto",
                    "parameters": {},
                }],
                "valid_actions": None,
                "others": {"task_id": task_id},
            },
        },
        task_id,
    )

    with pytest.raises(LiteContractError, match="extra_tool_schemas.*noncanonical"):
        stage(
            [logs],
            name="FlatSchema",
            out_dir=tmp_path / "ds" / "cua-lite" / "FlatSchema",
            filter_expr=None,
        )


def test_unstage_rejects_tagged_generic_metadata(tmp_path):
    dataset = tmp_path / "ds" / "cua-lite" / "TaggedGeneric"
    img_rel = f"cua-lite/{dataset.name}/images/ab/generic.png"
    img_abs = dataset.parent.parent / img_rel
    img_abs.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(img_abs)
    task_id = "task_generic_metadata"
    write_records_to_parquet(
        [
            {
                "images": [img_rel],
                "messages": [
                    {"role": "user", "content": [{"type": "image", "index": 0}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                ],
                "metadata": LiteGenericMetadata(
                    dims=("generic", "task"),
                    extra_tool_schemas=[],
                    others={
                        "task_id": task_id,
                        "episode_return": 1.0,
                        "terminated": True,
                        "truncated": False,
                    },
                ).to_dict(),
            }
        ],
        dataset / "generic" / "task" / "train.parquet",
        json_fields=("messages", "metadata"),
    )

    with pytest.raises(ValueError, match="only supports CUA metadata"):
        unstage(dataset, log_root=tmp_path / "logs" / "resumed", splits="train")
