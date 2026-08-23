from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from examples.grounding.adapter import _draw_marker
from examples.grounding.utils import rf_metrics
from lite.agents.core.agent.logger import TrajectoryLogger, sanitize_trajectory_json
from lite.core.tools.action_space import MAX_NORM, clamp_norm


def summarize_judge_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [
        r
        for r in records
        if r.get("judge_label") is not None
        and r.get("initial_point_correct") is not None
    ]
    tp = sum(1 for r in judged if r["judge_label"] and r["initial_point_correct"])
    tn = sum(1 for r in judged if (not r["judge_label"]) and (not r["initial_point_correct"]))
    fp = sum(1 for r in judged if r["judge_label"] and (not r["initial_point_correct"]))
    fn = sum(1 for r in judged if (not r["judge_label"]) and r["initial_point_correct"])
    total = len(judged)
    acc = (tp + tn) / total if total else 0.0
    precision_yes = tp / (tp + fp) if (tp + fp) else 0.0
    recall_yes = tp / (tp + fn) if (tp + fn) else 0.0
    wrong_sent = [r for r in judged if (not r["judge_label"]) and (not r["initial_point_correct"])]
    correct_sent = [r for r in judged if (not r["judge_label"]) and r["initial_point_correct"]]
    wrong_and_fixed = sum(1 for r in wrong_sent if r.get("final_reward") == 1.0)
    correct_and_broken = sum(1 for r in correct_sent if r.get("final_reward") == 0.0)
    wrong_fix_rate = wrong_and_fixed / len(wrong_sent) if wrong_sent else 0.0
    correct_break_rate = correct_and_broken / len(correct_sent) if correct_sent else 0.0
    return {
        "num_records": len(records),
        "num_judged": total,
        "judge_accuracy": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision_yes": precision_yes,
        "recall_yes": recall_yes,
        "wrong_and_fixed": wrong_and_fixed,
        "wrong_fix_rate": wrong_fix_rate,
        "correct_and_broken": correct_and_broken,
        "correct_break_rate": correct_break_rate,
        "improve": wrong_and_fixed - correct_and_broken,
    }


class RegionFocusTrajectoryLogger(TrajectoryLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._judge_records: list[dict[str, Any]] = []

    def on_step(self, data) -> None:
        super().on_step(data)
        trace = data.predict_result.lite_message.get("regionfocus_trace")
        if not self.save_data or trace is None:
            return

        turn_dir = self.log_dir / f"turn_{data.step_idx:04d}"
        # RegionFocus artifacts are an extension subdirectory under the current
        # turn layout, not an alternate prompt/result image layout.
        rf_dir = turn_dir / "regionfocus"
        rf_dir.mkdir(parents=True, exist_ok=True)
        (rf_dir / "00_trace.json").write_text(
            json.dumps(sanitize_trajectory_json(trace), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        screenshot = data.image
        if screenshot is not None:
            initial = (trace.get("initial") or {}).get("point")
            if initial is not None:
                self._save_marked_image(rf_dir / "01_initial_point.png", screenshot, initial)
            selected = (trace.get("selected") or {}).get("point")
            if selected is not None:
                self._save_marked_image(rf_dir / "02_selected_point.png", screenshot, selected)
            for item in trace.get("refine_candidates", []) or []:
                crop_box = item.get("crop_norm_box")
                if crop_box:
                    candidate_point = ((item.get("result") or {}).get("point"))
                    self._save_crop_image(
                        rf_dir / f"crop_{item.get('crop_index', 0):02d}.png",
                        screenshot,
                        crop_box,
                        candidate_point,
                    )
            aggregate_points = self._aggregate_candidate_points(
                trace.get("refine_candidates", []) or []
            )
            if len(aggregate_points) > 1:
                self._save_aggregate_candidates_image(
                    rf_dir / "04_aggregate_candidates.png",
                    screenshot,
                    aggregate_points,
                )

        annotation = (data.step_result.info or {}).get("annotation")
        initial_point = (trace.get("initial") or {}).get("point")
        judge_label = (trace.get("judge") or {}).get("accepted")
        initial_correct = rf_metrics.evaluate_point_against_annotation(
            annotation, initial_point
        )
        judge_eval = {
            "task_id": ((data.step_result.info or {}).get("annotation") or {}).get("id"),
            "judge_label": judge_label,
            "initial_point_correct": initial_correct,
            "initial_point": initial_point,
            "final_reward": data.step_result.reward,
        }
        self._judge_records.append(judge_eval)
        (rf_dir / "03_judge_eval.json").write_text(
            json.dumps(judge_eval, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def on_complete(self, lite_rl_sample) -> None:
        super().on_complete(lite_rl_sample)
        if lite_rl_sample is None:
            return
        summary_path = self.log_dir / "summary.json"
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text())
        judge_metrics = summarize_judge_records(self._judge_records)
        summary["regionfocus"] = {
            "judge_records": self._judge_records,
            "judge_summary": judge_metrics,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _save_marked_image(self, path: Path, screenshot: Image.Image, point: list[int]) -> None:
        _draw_marker(screenshot, point).save(path)

    def _save_aggregate_candidates_image(
        self,
        path: Path,
        screenshot: Image.Image,
        points: list[list[int]],
    ) -> None:
        labeled = screenshot.copy().convert("RGB")
        for idx, point in enumerate(points, start=1):
            labeled = _draw_marker(labeled, point, label=str(idx))
        labeled.save(path)

    def _aggregate_candidate_points(
        self,
        refine_candidates: list[dict[str, Any]],
    ) -> list[list[int]]:
        points: list[list[int]] = []
        seen_points: set[tuple[int, int]] = set()
        for item in refine_candidates:
            point = ((item.get("result") or {}).get("point"))
            if not (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(v, (int, float)) for v in point)
            ):
                continue
            point_key = (int(point[0]), int(point[1]))
            if point_key in seen_points:
                continue
            seen_points.add(point_key)
            points.append([int(point[0]), int(point[1])])
        return points

    def _save_crop_image(
        self,
        path: Path,
        screenshot: Image.Image,
        crop_box: list[int],
        point: list[int] | None = None,
    ) -> None:
        left, top, width, height = crop_box
        img_w, img_h = screenshot.size
        left_px = int(round(left / 1000 * img_w))
        top_px = int(round(top / 1000 * img_h))
        right_px = int(round((left + width) / 1000 * img_w))
        bottom_px = int(round((top + height) / 1000 * img_h))
        crop_img = screenshot.crop((left_px, top_px, right_px, bottom_px))
        zoom = min(img_w / max(1, crop_img.width), img_h / max(1, crop_img.height))
        cropped = crop_img.resize(
            (max(1, round(crop_img.width * zoom)), max(1, round(crop_img.height * zoom))),
            Image.Resampling.LANCZOS,
        )
        if point is not None and width and height:
            local_point = [
                clamp_norm(int(round((point[0] - left) / width * MAX_NORM))),
                clamp_norm(int(round((point[1] - top) / height * MAX_NORM))),
            ]
            cropped = _draw_marker(cropped, local_point)
        cropped.save(path)


def patch_trajectory_logger() -> None:
    import lite.agents.core.agent.logger as loggers
    loggers.TrajectoryLogger = RegionFocusTrajectoryLogger


def augment_summary(log_root: Path) -> dict[str, Any]:
    summary_path = log_root / "summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text())
    records: list[dict[str, Any]] = []
    skipped_summaries = 0
    for sample_summary in log_root.rglob("sample_*/summary.json"):
        try:
            sample = json.loads(sample_summary.read_text())
        except (OSError, TypeError, json.JSONDecodeError):
            skipped_summaries += 1
            continue
        rf = sample.get("regionfocus") or {}
        records.extend(rf.get("judge_records") or [])
    judge_summary = summarize_judge_records(records)
    summary["regionfocus"] = {
        "judge_summary": judge_summary,
    }
    if skipped_summaries:
        summary["regionfocus"]["skipped_sample_summaries"] = skipped_summaries
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (log_root / "judge_summary.json").write_text(
        json.dumps(judge_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return judge_summary
