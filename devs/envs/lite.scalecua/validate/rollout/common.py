"""Shared catalog helpers for the lite.scalecua rollout validation scripts.

Not a standalone script; imported by `check_prompt_data.py` and
`make_batch_prompt_data.py` in this directory.
"""

from __future__ import annotations

from typing import Any

from lite.gym.envs.lite.scalecua.src.utils import dataset


def catalog_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in dataset.RUNTIME_SPLITS:
        for _, row in dataset.iter_jsonl(dataset.catalog_path(split)):
            out[row["task_id"]] = row
    return out


def payload_for_exclusion(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    scalecua = metadata.get("scalecua") or {}
    others = metadata.get("others") or {}
    payload_metadata = dict(others) if isinstance(others, dict) else {}
    if isinstance(scalecua, dict):
        for key in ("source_path", "source_split", "related_apps", "snapshot"):
            if key in scalecua:
                payload_metadata[key] = scalecua[key]
    if metadata.get("source_split") is not None:
        payload_metadata["source_split"] = metadata["source_split"]
    oracle_actions = (
        others.get("oracle_actions", []) if isinstance(others, dict) else []
    )
    return {
        "id": metadata.get("osworld_id"),
        "instruction": row.get("instruction", ""),
        "metadata": payload_metadata,
        "config": metadata.get("config", []),
        "evaluator": metadata.get("evaluator", {}),
        "oracle_actions": oracle_actions,
    }
