"""Thread-safe accumulator and evaluator for RegionFocus judge metrics.

Populated by RegionFocus grounding agents in examples.grounding.adapter, then
drained and logged to wandb by rollout entrypoints after each rollout batch.

Usage:
    # in sample():
    rf_metrics.record({
        "judge_accepted": True,
        "initial_point_correct": True,
        "reward": 1.0,
    })

    # in generate_rollout (after eval batch completes):
    entries = rf_metrics.drain()
    rf_metrics.compute_and_log(entries, prefix="eval/osworld_g")
"""
from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_stats: list[dict[str, Any]] = []


def record(entry: dict[str, Any]) -> None:
    with _lock:
        _stats.append(entry)


def drain() -> list[dict[str, Any]]:
    with _lock:
        entries, _stats[:] = list(_stats), []
    return entries


def _is_point_in_polygon(x: float, y: float, polygon: list[float]) -> bool:
    n = len(polygon) // 2
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[2 * i], polygon[2 * i + 1]
        xj, yj = polygon[2 * j], polygon[2 * j + 1]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def evaluate_point_against_annotation(
    annotation: dict[str, Any] | None,
    point: list[int] | None,
) -> bool | None:
    """Return whether a normalized [0,1000] point is correct for known labels.

    Supported annotation shapes:
      * ScreenSpot-Pro: ``bbox=[x1,y1,x2,y2]`` + ``img_size=[w,h]``.
      * OSWorld-G bbox: ``box_type=bbox`` + ``box_coordinates=[x,y,w,h]``.
      * OSWorld-G polygon: ``box_type=polygon`` + flat polygon coordinates.
    """
    if annotation is None or point is None or len(point) < 2:
        return None

    bbox = annotation.get("bbox")
    img_size = annotation.get("img_size")
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and isinstance(img_size, (list, tuple))
        and len(img_size) == 2
    ):
        img_w, img_h = float(img_size[0]), float(img_size[1])
        px = float(point[0]) / 1000.0 * img_w
        py = float(point[1]) / 1000.0 * img_h
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return x1 <= px <= x2 and y1 <= py <= y2

    box_type = annotation.get("box_type")
    if box_type == "refusal":
        return None
    img_size = annotation.get("image_size")
    if not (
        isinstance(img_size, (list, tuple))
        and len(img_size) == 2
        and img_size[0]
        and img_size[1]
    ):
        return None
    img_w, img_h = float(img_size[0]), float(img_size[1])
    px = float(point[0]) / 1000.0 * img_w
    py = float(point[1]) / 1000.0 * img_h
    coords = annotation.get("box_coordinates")
    if box_type == "bbox":
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            return None
        x, y, w, h = (float(v) for v in coords)
        return x <= px <= x + w and y <= py <= y + h
    if box_type == "polygon":
        if not isinstance(coords, (list, tuple)):
            return None
        return _is_point_in_polygon(px, py, [float(v) for v in coords])
    return None


def point_correct_from_sources(
    point: list[int] | None,
    *,
    metadata: Any | None = None,
    step_info: dict[str, Any] | None = None,
) -> bool | None:
    """Evaluate a normalized point against metadata/step annotations."""
    annotations: list[dict[str, Any]] = []
    others = getattr(metadata, "others", None) or {}
    if isinstance(others, dict):
        annotations.append(others)
    info_annotation = (step_info or {}).get("annotation")
    if isinstance(info_annotation, dict):
        annotations.append(info_annotation)
    for annotation in annotations:
        result = evaluate_point_against_annotation(annotation, point)
        if result is not None:
            return result
    return None


def _metric_prefix(args: Any) -> str:
    try:
        dataset_names = [d.name for d in args.rollout_global_dataset.eval_datasets]
        return f"eval/{dataset_names[0]}" if len(dataset_names) == 1 else "eval"
    except AttributeError:
        return "eval"


def drain_and_log_after_rollout(args: Any, *, evaluation: bool) -> None:
    """Drain accumulated metrics after every rollout and log eval batches."""
    entries = drain()
    if evaluation and entries:
        compute_and_log(entries, prefix=_metric_prefix(args))


def compute_and_log(entries: list[dict[str, Any]], prefix: str) -> None:
    """Compute judge metrics from accumulated entries and log via safe_wandb_log."""
    from lite.train.rollout.core import safe_wandb_log

    if not entries:
        return

    n = len(entries)
    verifiable = [
        e for e in entries
        if isinstance(e.get("initial_point_correct"), bool)
    ]
    n_verifiable = len(verifiable)
    n_unverifiable = n - n_verifiable

    n_yes = sum(1 for e in entries if e["judge_accepted"])

    # True self-judge confusion matrix:
    #   predicted positive = judge accepted the initial point
    #   actual positive = initial point is inside the ground-truth bbox
    tp = sum(1 for e in verifiable if e["judge_accepted"] and e["initial_point_correct"])
    fp = sum(1 for e in verifiable if e["judge_accepted"] and not e["initial_point_correct"])
    tn = sum(1 for e in verifiable if not e["judge_accepted"] and not e["initial_point_correct"])
    fn = sum(1 for e in verifiable if not e["judge_accepted"] and e["initial_point_correct"])

    judge_acc = (tp + tn) / n_verifiable if n_verifiable else 0.0
    judge_yes_rate = n_yes / n
    initial_point_acc = (
        sum(1 for e in verifiable if e["initial_point_correct"]) / n_verifiable
        if n_verifiable else 0.0
    )
    reward_entries = [e for e in entries if e.get("reward") is not None]
    final_success_rate = (
        sum(1 for e in reward_entries if float(e["reward"]) > 0) / len(reward_entries)
        if reward_entries else None
    )
    n_actual_neg = fp + tn  # actual negatives: initial point was wrong
    n_actual_pos = fn + tp  # actual positives: initial point was correct
    fp_rate = fp / n_actual_neg if n_actual_neg else 0.0
    fn_rate = fn / n_actual_pos if n_actual_pos else 0.0

    safe_wandb_log({
        f"{prefix}/judge_yes_rate": judge_yes_rate,
        f"{prefix}/judge_acc": judge_acc,
        f"{prefix}/judge_fp_rate": fp_rate,
        f"{prefix}/judge_fn_rate": fn_rate,
        f"{prefix}/judge_tp": tp,
        f"{prefix}/judge_fp": fp,
        f"{prefix}/judge_tn": tn,
        f"{prefix}/judge_fn": fn,
        f"{prefix}/judge_n_episodes": n,
        f"{prefix}/judge_n_verifiable": n_verifiable,
        f"{prefix}/judge_n_unverifiable": n_unverifiable,
        f"{prefix}/initial_point_acc": initial_point_acc,
        **(
            {f"{prefix}/success_rate_when_judge_ran": final_success_rate}
            if final_success_rate is not None
            else {}
        ),
    })
