"""Render per-axis breakdown markdown tables from ``breakdown.json``.

``breakdown.json`` is produced post-hoc by ``reaggregate_breakdown.py``
(which picks axes via ``--axes``) — run that first, then this script
to render markdown.

Per-env axis cheat sheet (factual ``LiteBaseMetadata.others`` keys; the
choice of which to pivot on lives in the eval scripts, not the env):

* ``osworld_g``  → ``paper_category,box_type,GUI_types``
* ``screenspot_pro`` → ``group,ui_type,application``

Usage::

    # Single model
    uv run python devs/exps/eval/utils/render_breakdown.py \\
        .exps/eval/<env>/<commit-dir>/<run_id>/<slug>/

    # All models in a run, multi-model rows
    uv run python devs/exps/eval/utils/render_breakdown.py .exps/eval/<env>/<commit-dir>/<run_id>/

    # Paper-canonical cross-product (e.g. screenspot_pro 12-cell layout) —
    # this mode reads metadata.json + summary.json directly, so make sure
    # reaggregate_breakdown.py has been run at least once first to write
    # them.
    uv run python devs/exps/eval/utils/render_breakdown.py --cross "group,ui_type" \\
        .exps/eval/<env>/<commit-dir>/<run_id>/

The multi-model form (run-level, no trailing slug) is the one to paste
into a run_<N>.md ``## Breakdown`` section: rows = models, columns =
axis values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_per_slug(run_dir: Path) -> dict[str, dict]:
    """Return ``{slug: breakdown}`` for every model dir under *run_dir*."""
    out: dict[str, dict] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        bp = child / "breakdown.json"
        if not bp.exists():
            continue
        try:
            out[child.name] = json.loads(bp.read_text())
        except Exception:
            continue
    return out


def _render_single(breakdown: dict) -> str:
    """One slug → one markdown block per axis (n / mer)."""
    parts: list[str] = []
    for axis in breakdown.get("axes", []):
        cells = breakdown["by_axis"].get(axis, {})
        if not cells:
            continue
        parts.append(f"### {axis}")
        parts.append("")
        parts.append(f"| {axis} | n | MER |")
        parts.append("|---|---:|---:|")
        for v_key in sorted(cells, key=lambda k: -cells[k]["mean_episode_return"]):
            cell = cells[v_key]
            parts.append(
                f"| `{v_key}` | {cell['n_tasks']} | {cell['mean_episode_return']:.4f} |"
            )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_multi_model(per_slug: dict[str, dict]) -> str:
    """Multi-model pivot: rows = models, columns = axis values."""
    if not per_slug:
        return "_(no models with breakdown.json under this dir)_\n"

    # Collect unified axis-set (intersection of declared axes across models).
    # Intersection because if some models lack an axis the table can't be
    # rendered cleanly; envs declare a stable set so this is a no-op in
    # practice.
    axis_sets = [set(b.get("axes", [])) for b in per_slug.values()]
    common_axes = sorted(set.intersection(*axis_sets) if axis_sets else set())
    if not common_axes:
        return "_(no common breakdown axes across models)_\n"

    parts: list[str] = []
    for axis in common_axes:
        # Collect all values seen across models (union), preserve sorted
        # order for deterministic rendering.
        values: set[str] = set()
        for b in per_slug.values():
            values.update(b["by_axis"].get(axis, {}).keys())
        values_sorted = sorted(values)
        if not values_sorted:
            continue

        parts.append(f"### Breakdown — by `{axis}`")
        parts.append("")
        header = "| Model | " + " | ".join(f"`{v}`" for v in values_sorted) + " | Avg |"
        sep = "|---" + "|---:" * (len(values_sorted) + 1) + "|"
        parts.append(header)
        parts.append(sep)
        # Sort rows by overall Avg descending.
        rows: list[tuple[float, str]] = []
        for slug, b in per_slug.items():
            cells = b["by_axis"].get(axis, {})
            row_vals: list[str] = []
            sum_return = 0.0
            sum_n = 0
            for v in values_sorted:
                c = cells.get(v)
                if c:
                    row_vals.append(f"{c['mean_episode_return']:.4f}")
                    sum_return += c["mean_episode_return"] * c["n_tasks"]
                    sum_n += c["n_tasks"]
                else:
                    row_vals.append("—")
            avg = (sum_return / sum_n) if sum_n else 0.0
            rows.append((
                avg,
                f"| `{slug}` | " + " | ".join(row_vals) + f" | {avg:.4f} |",
            ))
        rows.sort(key=lambda r: -r[0])
        parts.extend(line for _, line in rows)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _compute_cross(slug_dir: Path, axes: list[str]) -> dict[str, dict[str, float]]:
    """Walk per-task ``metadata.json`` + ``summary.json`` under *slug_dir* and
    aggregate by the cross-product of *axes*.

    Returns ``{cell_label: {"n_tasks": int, "sum_return": float,
    "mean_episode_return": float}}`` where ``cell_label`` is
    ``"v1 × v2[ × v3...]"`` (Unicode ``×`` separator, axes order preserved).
    Multi-tag axes (e.g. ``GUI_types: [a, b]``) explode each task into the
    Cartesian product of its tag lists.
    """
    eval_root = slug_dir / "eval"
    if not eval_root.is_dir():
        return {}
    cells: dict[str, dict[str, float]] = {}
    for task_dir in eval_root.iterdir():
        sd = task_dir / "sample_00"
        meta_p = sd / "metadata.json"
        summ_p = sd / "summary.json"
        if not meta_p.exists() or not summ_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text())
            summ = json.loads(summ_p.read_text())
        except Exception:
            continue
        others = meta.get("others") or {}
        # Per-axis value lists (multi-tag → list, scalar → 1-elem list).
        # Skip the task if ANY axis is missing or has no usable value
        # (None, "", or empty list) — cross-product needs every dim.
        axis_values: list[list[str]] = []
        for ax in axes:
            v = others.get(ax)
            vs = v if isinstance(v, list) else [v]
            vs = [str(x) for x in vs if x not in (None, "")]
            if not vs:
                axis_values = []
                break
            axis_values.append(vs)
        if not axis_values:
            continue
        ep = float(summ.get("episode_return", 0.0))
        # Cartesian product across axes.
        combos = [[]]
        for vs in axis_values:
            combos = [c + [v] for c in combos for v in vs]
        for combo in combos:
            label = " × ".join(combo)
            cell = cells.setdefault(label, {"n_tasks": 0, "sum_return": 0.0})
            cell["n_tasks"] += 1
            cell["sum_return"] += ep
    # Finalize MER.
    for label, cell in cells.items():
        n = cell["n_tasks"]
        cell["mean_episode_return"] = round(cell["sum_return"] / n, 6) if n else 0.0
    return cells


def _render_cross(run_dir: Path, axes: list[str]) -> str:
    """Multi-model cross-product table: rows = models, columns = axis-tuple cells."""
    per_slug_cross: dict[str, dict[str, dict[str, float]]] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        cells = _compute_cross(child, axes)
        if cells:
            per_slug_cross[child.name] = cells
    if not per_slug_cross:
        return f"_(no per-task metadata for cross axes {axes!r} under {run_dir})_\n"

    # Union of all cell labels seen, sorted (lexicographic over the tuple).
    all_cells: set[str] = set()
    for cells in per_slug_cross.values():
        all_cells.update(cells.keys())
    cells_sorted = sorted(all_cells)

    axis_label = " × ".join(f"`{a}`" for a in axes)
    parts: list[str] = [
        f"### Breakdown — by {axis_label}",
        "",
        "| Model | " + " | ".join(f"`{c}`" for c in cells_sorted) + " | Avg |",
        "|---" + "|---:" * (len(cells_sorted) + 1) + "|",
    ]
    rows: list[tuple[float, str]] = []
    for slug, cells in per_slug_cross.items():
        row_vals: list[str] = []
        sum_return = 0.0
        sum_n = 0
        for label in cells_sorted:
            cell = cells.get(label)
            if cell:
                row_vals.append(f"{cell['mean_episode_return']:.4f}")
                sum_return += cell["mean_episode_return"] * cell["n_tasks"]
                sum_n += cell["n_tasks"]
            else:
                row_vals.append("—")
        avg = (sum_return / sum_n) if sum_n else 0.0
        rows.append((
            avg,
            f"| `{slug}` | " + " | ".join(row_vals) + f" | {avg:.4f} |",
        ))
    rows.sort(key=lambda r: -r[0])
    parts.extend(line for _, line in rows)
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path", type=Path, help="Run dir or single-slug dir")
    p.add_argument(
        "--cross", default=None,
        help='Render a cross-product axis instead of per-axis tables, '
             'e.g. ``--cross "group,ui_type"`` for ScreenSpot-Pro\'s '
             '12-cell paper layout. Reads per-task metadata.json + '
             'summary.json directly (independent of breakdown.json).',
    )
    args = p.parse_args()

    if not args.path.is_dir():
        raise SystemExit(f"not a directory: {args.path}")

    if args.cross:
        axes = [a.strip() for a in args.cross.split(",") if a.strip()]
        # cross is run-dir mode only (per-slug cells need slug subdirs).
        print(_render_cross(args.path, axes))
        return

    bp = args.path / "breakdown.json"
    if bp.exists():
        # Single-slug mode
        breakdown = json.loads(bp.read_text())
        print(_render_single(breakdown))
    else:
        # Multi-model mode (run dir with sibling slug subdirs)
        per_slug = _load_per_slug(args.path)
        print(_render_multi_model(per_slug))


if __name__ == "__main__":
    main()
