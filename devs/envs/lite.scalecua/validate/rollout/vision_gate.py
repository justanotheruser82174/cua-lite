#!/usr/bin/env python3
"""Build and validate lite.scalecua full-census visual-audit artifacts.

This script is intentionally read-only with respect to rollout roots. It can:

* scan an existing rollout log root into a full queue JSONL; and
* validate one or more visual-audit sidecars against that queue, producing a
  gate report that fails closed on missing coverage, duplicate reviews,
  invalid labels, unresolved labels, and reward/vision disagreement.

It does not start an env-server, run a fresh rollout, or call a model/VLM.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lite.infer.debug.log_layout import (
    ACTIONS_JSON,
    RESPONSE_TXT,
    matching_annotation_path,
    turn_dirs,
    turn_images,
    turn_results_path,
)

SCHEMA_VERSION = 1
SAMPLE_RE = re.compile(r"sample_(\d+)$")
REWARD_EPS = 1e-9

# --- Negative-control corpus -------------------------------------------------------
# The pinned negative controls are correctly-reward=0 trajectories: under the
# loosened checkers (CF/formula/`w:shd`, scope/argv/ext-path/string,
# and the vlc/thunderbird config flushes) they must STAY 0. A loosened checker that
# also passes one of these is a NEW false positive — exactly the regression this
# corpus exists to catch. See `check_negative_controls_non_recovery`.
NEGATIVE_CONTROL_CORPUS_PATH = (
    Path(__file__).resolve().parents[1] / "samples" / "negative_control_corpus.jsonl"
)

# Closed set of human-readable negative-control classes: cases that are not
# false negatives and must stay at reward 0.
NEGATIVE_CONTROL_CLASSES = {
    "wrong_setting",
    "window_state",
    "wrong_content",
    "site_block",
}

# Every loosening bucket that must be represented (N>=3 each) so the gate cannot be
# skipped by simply emptying a bucket.
REQUIRED_NEGATIVE_CONTROL_LOOSENINGS = {
    "g1_7_cf",
    "g1_7_formula",
    "g1_7_wshd",
    "g1_8_scope_argv",
    "g1_8_ext_path",
    "g1_8_string",
    "flush_vlc",
    "flush_thunderbird",
}
MIN_ROWS_PER_LOOSENING = 3

VALID_VISUAL_LABELS = {
    "true_success",
    "true_failure",
    "partial_success",
    "false_success",
    "false_failure",
    "setup_failure",
    "action_parse_failure",
    "transient_failure",
    "blocked_upstream",
    "not_visually_decidable",
    "ambiguous_needs_evaluator_probe",
}

VISUAL_SUCCESS_BY_LABEL = {
    "true_success": True,
    "false_failure": True,
    "true_failure": False,
    "partial_success": False,
    "false_success": False,
    "setup_failure": False,
    "action_parse_failure": False,
    "transient_failure": False,
    "blocked_upstream": False,
    "not_visually_decidable": False,
    "ambiguous_needs_evaluator_probe": False,
}

UNRESOLVED_LABELS = {
    "transient_failure",
    "not_visually_decidable",
    "ambiguous_needs_evaluator_probe",
}
FALSE_LABELS = {"false_success", "false_failure"}

REQUIRED_SIDECAR_FIELDS = {
    "schema_version",
    "run_id",
    "source_queue_path",
    "queue_id",
    "sample_id",
    "task_id",
    "split",
    "domain",
    "log_dir",
    "reported_reward",
    "reported_terminated",
    "reported_truncated",
    "setup_ok",
    "action_ok",
    "eval_ok",
    "visual_label",
    "visual_success",
    "notes",
    "reviewer",
    "checked_at",
    "latest_turn",
    "result_json_path",
    "actions_path",
    "response_path",
    "frame_paths",
    "artifact_paths",
    "probe_paths",
    "vision_done",
    "reward_vision_disagree",
    "probe_status",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}
    return data if isinstance(data, dict) else {"_read_error": "json root is not an object"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_index(sample_dir: Path) -> int:
    match = SAMPLE_RE.search(sample_dir.name)
    return int(match.group(1)) if match else -1


def _load_catalog_metadata() -> dict[str, dict[str, Any]]:
    from lite.gym.envs.lite.scalecua.src.utils import dataset

    out: dict[str, dict[str, Any]] = {}
    for split in dataset.RUNTIME_SPLITS:
        path = dataset.catalog_path(split)
        if not path.exists():
            continue
        for _, row in dataset.iter_jsonl(path):
            metadata = row.get("metadata") or {}
            others = metadata.get("others") or {}
            scalecua = metadata.get("scalecua") or {}
            task_id = row["task_id"]
            out[task_id] = {
                "split": split,
                "instruction": row.get("instruction", ""),
                "domain": str(others.get("domain") or "unknown"),
                "related_app_domain": _related_app_domain(scalecua, others),
            }
    return out


def _normalize_domain(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "base_setup": "os",
        "calc": "libreoffice_calc",
        "libreoffice calc": "libreoffice_calc",
        "multiapps": "multi_apps",
        "vscode": "vs_code",
    }
    return aliases.get(text, text or "unknown")


def _related_app_domain(scalecua: dict[str, Any], others: dict[str, Any]) -> str:
    related = scalecua.get("related_apps") or []
    if len(related) > 1:
        return "multi_apps"
    if len(related) == 1:
        return _normalize_domain(related[0])
    return _normalize_domain(scalecua.get("snapshot") or others.get("domain"))


def _reward_bucket(reward: Any) -> str:
    if reward is None:
        return "reward_missing"
    value = float(reward)
    if value >= 1.0 - REWARD_EPS:
        return "reward_1"
    if value <= REWARD_EPS:
        return "reward_0"
    return "reward_partial"


def _iter_sample_dirs(log_root: Path) -> Iterable[Path]:
    sample_dirs: set[Path] = set()
    for parquet in log_root.rglob("trajectory.parquet"):
        if SAMPLE_RE.fullmatch(parquet.parent.name):
            sample_dirs.add(parquet.parent)
    for summary in log_root.rglob("summary.json"):
        if SAMPLE_RE.fullmatch(summary.parent.name):
            sample_dirs.add(summary.parent)
    yield from sorted(
        sample_dirs,
        key=lambda path: (path.parent.as_posix(), _sample_index(path), path.name),
    )


def _path_identity(log_root: Path, sample_dir: Path) -> tuple[str, str, str]:
    rel_parts = sample_dir.relative_to(log_root).parts
    sample_id = sample_dir.name
    task_id = sample_dir.parent.name
    split = rel_parts[-3] if len(rel_parts) >= 3 else ""
    if len(rel_parts) == 2:
        split = ""
    return split, task_id, sample_id


def _load_trajectory(sample_dir: Path) -> tuple[dict[str, Any], list[str]]:
    parquet_path = sample_dir / "trajectory.parquet"
    if not parquet_path.exists():
        return {}, []
    try:
        import pandas as pd

        from lite.data.staging import coerce_meta

        df = pd.read_parquet(parquet_path)
        if df.empty:
            return {}, []
        row = df.iloc[0]
        metadata = coerce_meta(row.get("metadata"))
        images_raw = row.get("images")
        images = [str(image) for image in images_raw] if images_raw is not None else []
        return metadata, images
    except Exception as exc:
        return {"others": {"_trajectory_read_error": str(exc)}}, []


def _trajectory_durable_metadata_value(
    trajectory_metadata: dict[str, Any],
    trajectory_others: dict[str, Any],
    key: str,
) -> Any:
    if key in trajectory_metadata:
        return trajectory_metadata[key]
    # Historical rollout rows stored these durable fields in metadata.others.
    return trajectory_others.get(key)


def _summary_value(
    summary: dict[str, Any],
    trajectory_metadata: dict[str, Any],
    trajectory_others: dict[str, Any],
    result: dict[str, Any],
    key: str,
) -> Any:
    if key in summary:
        return summary[key]
    value = _trajectory_durable_metadata_value(trajectory_metadata, trajectory_others, key)
    if value is not None:
        return value
    return result.get(key)


def build_queue(
    log_root: Path,
    *,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return one queue row per resolved sample under ``log_root``.

    Completion and episode outcome come from ``sample_*/summary.json`` first
    and current top-level ``trajectory.parquet`` metadata second. Historical
    ``metadata.others`` durable fields are compatibility fallback only. Per-turn
    result JSON is read through the shared debug layout helper and is only a
    detail fallback because intermediate turns commonly have ``reward=null``.
    """

    log_root = log_root.resolve()
    catalog = catalog if catalog is not None else _load_catalog_metadata()
    rows: list[dict[str, Any]] = []
    for sample_dir in _iter_sample_dirs(log_root):
        rel = sample_dir.relative_to(log_root).as_posix()
        path_split, path_task_id, sample_id = _path_identity(log_root, sample_dir)
        trajectory_metadata, trajectory_images = _load_trajectory(sample_dir)
        trajectory_others = trajectory_metadata.get("others") or {}
        env_id = _trajectory_durable_metadata_value(
            trajectory_metadata, trajectory_others, "env_id"
        )
        task_id = str(
            _trajectory_durable_metadata_value(
                trajectory_metadata, trajectory_others, "task_id"
            )
            or path_task_id
        )
        split = str(trajectory_others.get("split") or path_split)
        dirs = turn_dirs(sample_dir)
        result_paths = [
            result_path
            for path in dirs
            if (result_path := turn_results_path(path)) is not None
        ]
        latest_turn_dir = result_paths[-1].parent if result_paths else None
        result = _read_json(result_paths[-1]) if result_paths else {}
        summary_path = sample_dir / "summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        reward = _summary_value(
            summary, trajectory_metadata, trajectory_others, result, "episode_return"
        )
        if reward is None:
            reward = result.get("reward")
        terminated = _summary_value(
            summary, trajectory_metadata, trajectory_others, result, "terminated"
        )
        truncated = _summary_value(
            summary, trajectory_metadata, trajectory_others, result, "truncated"
        )
        meta = catalog.get(task_id, {})
        images = turn_images(sample_dir)
        frame_paths: dict[str, str] = {}
        if trajectory_images:
            frame_paths["initial"] = trajectory_images[0]
            frame_paths["final"] = trajectory_images[-1]
        elif images:
            frame_paths["initial"] = str(images[0][1])
            frame_paths["final"] = str(images[-1][1])
        final_turn_dir, final_image = images[-1] if images else (None, None)
        if final_image is not None and "final" not in frame_paths:
            frame_paths["final"] = str(final_image)
        if final_turn_dir is not None and final_image is not None:
            annotated = matching_annotation_path(final_turn_dir, final_image)
            if annotated is not None:
                frame_paths["final_annotated"] = str(annotated)

        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": log_root.name,
            "queue_id": rel,
            "sample_relpath": rel,
            "split": str(meta.get("split") or split or "unknown"),
            "env_id": str(env_id or "unknown"),
            "task_id": task_id,
            "sample_id": sample_id,
            "domain": str(meta.get("domain") or "unknown"),
            "related_app_domain": str(meta.get("related_app_domain") or "unknown"),
            "instruction": str(meta.get("instruction") or ""),
            "log_dir": str(sample_dir),
            "trajectory_path": str(sample_dir / "trajectory.parquet"),
            "summary_path": str(summary_path),
            "sample_error": summary.get("error"),
            "latest_turn": latest_turn_dir.name if latest_turn_dir else None,
            "reported_reward": reward,
            "reported_terminated": terminated,
            "reported_truncated": truncated,
            "reward_bucket": _reward_bucket(reward),
            "result_json_path": str(result_paths[-1]) if result_paths else "",
            "actions_path": str(latest_turn_dir / ACTIONS_JSON) if latest_turn_dir else "",
            "response_path": str(latest_turn_dir / RESPONSE_TXT) if latest_turn_dir else "",
            "frame_paths": frame_paths,
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["split"], row["domain"], row["queue_id"]))


def write_queue(log_root: Path, queue_path: Path) -> list[dict[str, Any]]:
    rows = build_queue(log_root)
    _write_jsonl(queue_path, rows)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(row)
    return rows


def load_negative_control_corpus(
    path: Path = NEGATIVE_CONTROL_CORPUS_PATH,
) -> list[dict[str, Any]]:
    """Load and structurally validate the pinned negative-control corpus.

    Raises ``ValueError`` on any structural violation so a malformed / silently
    emptied corpus is a hard failure rather than a skipped gate.
    """

    rows = _read_jsonl(Path(path))
    if not rows:
        raise ValueError(f"{path}: negative-control corpus is empty")
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows, start=1):
        loosening = row.get("loosening")
        if loosening not in REQUIRED_NEGATIVE_CONTROL_LOOSENINGS:
            raise ValueError(f"{path}:{index}: unknown loosening {loosening!r}")
        if float(row.get("expected_negative_reward", -1)) != 0.0:
            raise ValueError(f"{path}:{index}: expected_negative_reward must be 0")
        if row.get("expected_recovered") is not False:
            raise ValueError(f"{path}:{index}: expected_recovered must be false")
        if row.get("is_fn_recovery_target") is not False:
            raise ValueError(f"{path}:{index}: is_fn_recovery_target must be false")
        cls = row.get("negative_control_class")
        if cls not in NEGATIVE_CONTROL_CLASSES:
            raise ValueError(f"{path}:{index}: bad negative_control_class {cls!r}")
        key = (
            loosening,
            str(row.get("task_id")),
            str(row.get("fixture_id") or row.get("collected_log_dir")),
        )
        if key in seen:
            raise ValueError(f"{path}:{index}: duplicate negative-control row {key}")
        seen.add(key)
        counts[loosening] += 1
    thin = {
        name: counts.get(name, 0)
        for name in REQUIRED_NEGATIVE_CONTROL_LOOSENINGS
        if counts.get(name, 0) < MIN_ROWS_PER_LOOSENING
    }
    if thin:
        raise ValueError(f"{path}: loosenings below N>={MIN_ROWS_PER_LOOSENING}: {thin}")
    return rows


def check_negative_controls_non_recovery(
    rescored_rewards: dict[str, Any],
    *,
    corpus: list[dict[str, Any]] | None = None,
    path: Path = NEGATIVE_CONTROL_CORPUS_PATH,
) -> dict[str, Any]:
    """Assert that no pinned negative control was WRONGLY recovered.

    ``rescored_rewards`` maps ``task_id`` to the reward produced by re-scoring the
    trajectory through the loosened checkers. The report ``passed`` is True only
    when every pinned negative control was re-scored (no missing coverage) AND
    each one STAYED at reward 0 (``_scalar_success`` is False). A future checker
    change that flips any of these to a pass turns ``passed`` False — the durable
    guard behind the negative-control corpus.
    """

    rows = corpus if corpus is not None else load_negative_control_corpus(path)
    recovered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])
        if task_id not in rescored_rewards:
            missing.append({"task_id": task_id, "loosening": row["loosening"]})
            continue
        if _scalar_success(rescored_rewards[task_id]):
            recovered.append(
                {
                    "task_id": task_id,
                    "loosening": row["loosening"],
                    "negative_control_class": row["negative_control_class"],
                    "rescored_reward": rescored_rewards[task_id],
                }
            )
    return {
        "passed": not recovered and not missing,
        "total": len(rows),
        "recovered": recovered,
        "missing": missing,
    }


def _sidecar_files(sidecar_paths: Iterable[Path] | None, sidecar_root: Path | None) -> list[Path]:
    files: list[Path] = []
    if sidecar_root is not None:
        if sidecar_root.is_file():
            files.append(sidecar_root)
        elif sidecar_root.is_dir():
            files.extend(sorted(sidecar_root.rglob("*.json")))
            files.extend(sorted(sidecar_root.rglob("*.jsonl")))
    if sidecar_paths:
        for path in sidecar_paths:
            if path.is_dir():
                files.extend(sorted(path.rglob("*.json")))
                files.extend(sorted(path.rglob("*.jsonl")))
            else:
                files.append(path)
    seen: set[Path] = set()
    out: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def _read_sidecar_rows(
    sidecar_paths: Iterable[Path] | None,
    sidecar_root: Path | None,
) -> list[tuple[dict[str, Any], Path]]:
    rows: list[tuple[dict[str, Any], Path]] = []
    for path in _sidecar_files(sidecar_paths, sidecar_root):
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number}: sidecar row is not an object")
                    rows.append((row, path))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: sidecar list entry is not an object")
                rows.append((row, path))
        elif isinstance(data, dict):
            rows.append((data, path))
        else:
            raise ValueError(f"{path}: sidecar JSON root must be object or list")
    return rows


def _queue_id(row: dict[str, Any]) -> str:
    value = row.get("queue_id") or row.get("sample_relpath")
    if value:
        return str(value)
    return f"{row.get('split')}/{row.get('task_id')}/{row.get('sample_id', 'sample_00')}"


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _scalar_success(reward: Any) -> bool:
    value = _as_float(reward)
    return bool(value is not None and value > 0.5)


def _path_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(item) for item in value.values() if item]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _frame_count(value: Any) -> int:
    return len(_path_values(value))


def _add_error(
    errors: dict[str, list[dict[str, Any]]],
    category: str,
    example: dict[str, Any],
    *,
    limit: int,
) -> None:
    if len(errors[category]) < limit:
        errors[category].append(example)


def _counter_template() -> dict[str, int]:
    return {
        "missing_sidecar": 0,
        "duplicate_sidecar": 0,
        "unknown_sidecar": 0,
        "missing_required_field": 0,
        "identity_mismatch": 0,
        "reward_mismatch": 0,
        "invalid_label": 0,
        "visual_success_mismatch": 0,
        "invalid_bool_field": 0,
        "path_missing": 0,
        "insufficient_frames": 0,
        "unresolved_labels": 0,
        "false_success": 0,
        "false_failure": 0,
        "transient_failure": 0,
        "reward_vision_disagree": 0,
        "disagreement_flag_mismatch": 0,
        "rate_below_threshold": 0,
    }


def validate_full_census(
    *,
    queue_path: Path,
    sidecar_paths: Iterable[Path] | None = None,
    sidecar_root: Path | None = None,
    report_path: Path | None = None,
    require_paths: bool = True,
    min_success_rate: float | None = None,
    expected_total: int | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Validate visual-audit sidecars against a full queue JSONL."""

    queue_rows = _read_jsonl(queue_path)
    queue_by_id: dict[str, dict[str, Any]] = {}
    errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = _counter_template()
    failed_queue_ids: set[str] = set()

    for row in queue_rows:
        qid = _queue_id(row)
        if qid in queue_by_id:
            counts["duplicate_sidecar"] += 1
            failed_queue_ids.add(qid)
            _add_error(errors, "duplicate_queue_id", {"queue_id": qid}, limit=max_examples)
        queue_by_id[qid] = row

    sidecar_by_id: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for row, path in _read_sidecar_rows(sidecar_paths, sidecar_root):
        qid = _queue_id(row)
        sidecar_by_id[qid].append((row, path))

    for qid in sorted(set(sidecar_by_id) - set(queue_by_id)):
        counts["unknown_sidecar"] += 1
        _add_error(
            errors,
            "unknown_sidecar",
            {"queue_id": qid, "sidecar_paths": [str(path) for _, path in sidecar_by_id[qid]]},
            limit=max_examples,
        )

    split_totals: Counter[str] = Counter()
    split_success: Counter[str] = Counter()
    domain_totals: Counter[str] = Counter()
    domain_success: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    reward_bucket_counts: Counter[str] = Counter()

    for qid, queue_row in queue_by_id.items():
        split = str(queue_row.get("split") or "unknown")
        domain = str(queue_row.get("domain") or "unknown")
        split_totals[split] += 1
        domain_totals[domain] += 1
        reward_bucket_counts[_reward_bucket(queue_row.get("reported_reward"))] += 1

        sidecars = sidecar_by_id.get(qid) or []
        if not sidecars:
            counts["missing_sidecar"] += 1
            failed_queue_ids.add(qid)
            _add_error(errors, "missing_sidecar", {"queue_id": qid}, limit=max_examples)
            continue
        if len(sidecars) > 1:
            counts["duplicate_sidecar"] += 1
            failed_queue_ids.add(qid)
            _add_error(
                errors,
                "duplicate_sidecar",
                {"queue_id": qid, "sidecar_paths": [str(path) for _, path in sidecars]},
                limit=max_examples,
            )

        sidecar, sidecar_path = sidecars[0]
        row_failed = len(sidecars) > 1
        missing = sorted(REQUIRED_SIDECAR_FIELDS - set(sidecar))
        if missing:
            counts["missing_required_field"] += 1
            row_failed = True
            _add_error(
                errors,
                "missing_required_field",
                {"queue_id": qid, "missing": missing},
                limit=max_examples,
            )

        for field in ("task_id", "split", "domain", "sample_id"):
            if field in sidecar and str(sidecar[field]) != str(queue_row.get(field)):
                counts["identity_mismatch"] += 1
                row_failed = True
                _add_error(
                    errors,
                    "identity_mismatch",
                    {
                        "queue_id": qid,
                        "field": field,
                        "queue_value": queue_row.get(field),
                        "sidecar_value": sidecar.get(field),
                    },
                    limit=max_examples,
                )

        queue_reward = _as_float(queue_row.get("reported_reward"))
        sidecar_reward = _as_float(sidecar.get("reported_reward"))
        if (
            queue_reward is None
            or sidecar_reward is None
            or abs(queue_reward - sidecar_reward) > REWARD_EPS
        ):
            counts["reward_mismatch"] += 1
            row_failed = True
            _add_error(
                errors,
                "reward_mismatch",
                {
                    "queue_id": qid,
                    "queue_reward": queue_row.get("reported_reward"),
                    "sidecar_reward": sidecar.get("reported_reward"),
                },
                limit=max_examples,
            )

        label = str(sidecar.get("visual_label"))
        label_counts[label] += 1
        if label not in VALID_VISUAL_LABELS:
            counts["invalid_label"] += 1
            row_failed = True
            _add_error(
                errors,
                "invalid_label",
                {"queue_id": qid, "visual_label": label},
                limit=max_examples,
            )
            visual_success = sidecar.get("visual_success")
        else:
            visual_success = sidecar.get("visual_success")
            expected_visual_success = VISUAL_SUCCESS_BY_LABEL[label]
            if visual_success is not expected_visual_success:
                counts["visual_success_mismatch"] += 1
                row_failed = True
                _add_error(
                    errors,
                    "visual_success_mismatch",
                    {
                        "queue_id": qid,
                        "visual_label": label,
                        "expected_visual_success": expected_visual_success,
                        "sidecar_visual_success": visual_success,
                    },
                    limit=max_examples,
                )
            if label in UNRESOLVED_LABELS:
                counts["unresolved_labels"] += 1
                row_failed = True
                _add_error(
                    errors,
                    "unresolved_labels",
                    {"queue_id": qid, "visual_label": label},
                    limit=max_examples,
                )
            if label == "false_success":
                counts["false_success"] += 1
                row_failed = True
                _add_error(errors, "false_success", {"queue_id": qid}, limit=max_examples)
            if label == "false_failure":
                counts["false_failure"] += 1
                row_failed = True
                _add_error(errors, "false_failure", {"queue_id": qid}, limit=max_examples)
            if label == "transient_failure":
                counts["transient_failure"] += 1

        for field in ("setup_ok", "action_ok", "eval_ok", "vision_done", "reward_vision_disagree"):
            if field in sidecar and not isinstance(sidecar[field], bool):
                counts["invalid_bool_field"] += 1
                row_failed = True
                _add_error(
                    errors,
                    "invalid_bool_field",
                    {"queue_id": qid, "field": field, "value": sidecar[field]},
                    limit=max_examples,
                )

        if isinstance(visual_success, bool):
            if visual_success:
                split_success[split] += 1
                domain_success[domain] += 1
            disagreement = _scalar_success(queue_row.get("reported_reward")) != visual_success
            if disagreement:
                counts["reward_vision_disagree"] += 1
            if (
                sidecar.get("reward_vision_disagree") is not None
                and sidecar.get("reward_vision_disagree") is not disagreement
            ):
                counts["disagreement_flag_mismatch"] += 1
                row_failed = True
                _add_error(
                    errors,
                    "disagreement_flag_mismatch",
                    {
                        "queue_id": qid,
                        "expected_reward_vision_disagree": disagreement,
                        "sidecar_reward_vision_disagree": sidecar.get("reward_vision_disagree"),
                    },
                    limit=max_examples,
                )

        if _frame_count(sidecar.get("frame_paths")) < 2:
            counts["insufficient_frames"] += 1
            row_failed = True
            _add_error(errors, "insufficient_frames", {"queue_id": qid}, limit=max_examples)

        if require_paths:
            path_fields = ["log_dir", "result_json_path", "actions_path", "response_path"]
            missing_paths: list[str] = []
            for field in path_fields:
                value = sidecar.get(field)
                if not value:
                    missing_paths.append(f"{field}=<empty>")
                elif not _resolve_path(str(value)).exists():
                    missing_paths.append(f"{field}={value}")
            for field in ("frame_paths", "artifact_paths", "probe_paths"):
                for value in _path_values(sidecar.get(field)):
                    if not _resolve_path(value).exists():
                        missing_paths.append(f"{field}={value}")
            if missing_paths:
                counts["path_missing"] += 1
                row_failed = True
                _add_error(
                    errors,
                    "path_missing",
                    {"queue_id": qid, "missing_paths": missing_paths[:10]},
                    limit=max_examples,
                )

        if row_failed:
            failed_queue_ids.add(qid)

    total = len(queue_rows)
    if expected_total is not None and total != expected_total:
        counts["missing_sidecar"] += max(0, expected_total - total)
        _add_error(
            errors,
            "expected_total_mismatch",
            {"expected_total": expected_total, "actual_total": total},
            limit=max_examples,
        )

    def rate(success: int, count: int) -> float | None:
        return None if count == 0 else success / count

    overall_success = sum(split_success.values())
    overall_rate = rate(overall_success, total)
    if min_success_rate is not None:
        checks = [("overall", overall_rate)]
        checks.extend(
            (f"split:{name}", rate(split_success[name], count))
            for name, count in split_totals.items()
        )
        checks.extend(
            (f"domain:{name}", rate(domain_success[name], count))
            for name, count in domain_totals.items()
        )
        for name, value in checks:
            if value is not None and value < min_success_rate:
                counts["rate_below_threshold"] += 1
                _add_error(
                    errors,
                    "rate_below_threshold",
                    {"bucket": name, "success_rate": value, "min_success_rate": min_success_rate},
                    limit=max_examples,
                )

    passed = total - len(failed_queue_ids)
    hard_fail_count = sum(counts.values())
    passed_gate = hard_fail_count == 0
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "queue_path": str(queue_path),
        "sidecar_root": str(sidecar_root) if sidecar_root is not None else None,
        "total": total,
        "audited": total - counts["missing_sidecar"],
        "passed": passed,
        "passed_gate": passed_gate,
        **counts,
        "reward_buckets": dict(sorted(reward_bucket_counts.items())),
        "visual_labels": dict(sorted(label_counts.items())),
        "visual_success_rate": overall_rate,
        "by_split": {
            name: {
                "total": count,
                "visual_success": split_success[name],
                "visual_success_rate": rate(split_success[name], count),
            }
            for name, count in sorted(split_totals.items())
        },
        "by_domain": {
            name: {
                "total": count,
                "visual_success": domain_success[name],
                "visual_success_rate": rate(domain_success[name], count),
            }
            for name, count in sorted(domain_totals.items())
        },
        "errors": {key: value for key, value in sorted(errors.items())},
    }
    if report_path is not None:
        _write_json(report_path, report)
    return report


def _print_build_summary(rows: list[dict[str, Any]], queue_path: Path) -> None:
    counts = Counter((row["split"], row["domain"], row["reward_bucket"]) for row in rows)
    print(f"wrote {queue_path}")
    print(f"rows={len(rows)}")
    for (split, domain, bucket), count in sorted(counts.items()):
        print(f"count {split}/{domain}/{bucket}: {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-queue")
    build.add_argument("--log-root", type=Path, required=True)
    build.add_argument("--queue", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--queue", type=Path, required=True)
    validate.add_argument("--sidecar-root", type=Path)
    validate.add_argument("--sidecar", type=Path, action="append", default=[])
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--no-require-paths", action="store_true")
    validate.add_argument("--min-success-rate", type=float)
    validate.add_argument("--expected-total", type=int)

    args = parser.parse_args(argv)
    if args.command == "build-queue":
        rows = write_queue(args.log_root, args.queue)
        _print_build_summary(rows, args.queue)
        return 0

    report = validate_full_census(
        queue_path=args.queue,
        sidecar_paths=args.sidecar,
        sidecar_root=args.sidecar_root,
        report_path=args.report,
        require_paths=not args.no_require_paths,
        min_success_rate=args.min_success_rate,
        expected_total=args.expected_total,
    )
    print(f"wrote {args.report}")
    print(f"passed_gate={report['passed_gate']} total={report['total']} passed={report['passed']}")
    return 0 if report["passed_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
