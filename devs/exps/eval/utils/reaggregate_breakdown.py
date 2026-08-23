"""Aggregate ``breakdown.json`` for completed eval campaigns.

``breakdown.json`` is produced post-hoc by this script — the rollout
only writes per-task ``summary.json``. The script re-imports the env
to (re)materialize each task's ``LiteBaseMetadata`` next to the summary as
``metadata.json``, then aggregates per-axis MERs.

Aggregation contract
--------------------
For each ``<run-dir>/<slug>/eval/<task>/sample_00/{summary,metadata}.json``:

* ``--axes`` lists ``LiteBaseMetadata.others`` keys to pivot on (caller
  picks; envs do **not** declare which axes are interesting — that's
  an analysis choice). ``--axes`` is **typed against the keys the run's
  metadata actually declares**: an axis no task declares is a typo, and
  the script exits rather than quietly pivoting on a smaller axis set.
* For each axis, ``others[axis]`` partitions the task into cells.
  List-valued cells (e.g. multi-tag ``GUI_types``) contribute to every
  tag.
* Per cell we accumulate ``n_tasks`` and ``sum_return``; the final
  ``mean_episode_return`` is the obvious quotient.

The script writes ``metadata.json`` for any task missing it (typical
for fresh campaigns — the rollout doesn't persist it). Pass
``--force-metadata`` to **re-render every** metadata.json from current
env code, e.g. after adding a new field to ``others`` so existing
snapshots pick it up.

Per-env axis cheat sheet (cf. ``lite/gym/envs/<env>/main.py:others``):

* ``osworld_g``  → ``paper_category,box_type,GUI_types``
* ``screenspot_pro`` → ``group,ui_type,application``

Usage::

    # osworld_g (one run dir)
    uv run python devs/exps/eval/utils/reaggregate_breakdown.py \\
        --axes paper_category,box_type,GUI_types \\
        .exps/eval/osworld_g/<commit-dir>/<run_id>/

    # screenspot_pro
    uv run python devs/exps/eval/utils/reaggregate_breakdown.py \\
        --axes group,ui_type,application \\
        .exps/eval/screenspot_pro/<commit-dir>/<run_id>/

    # Refresh metadata.json from current env code (e.g. new field added)
    uv run python devs/exps/eval/utils/reaggregate_breakdown.py --force-metadata \\
        --axes paper_category,box_type,GUI_types \\
        .exps/eval/osworld_g/<commit-dir>/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Trigger env registration so we can reload metadata if a task's
# metadata.json is missing.
import lite.gym.envs  # noqa: F401
from lite.gym.registry import registry as gym_registry


def _aggregate_breakdown(
    slug_dir: Path, axes: list[str]
) -> tuple[dict[str, Any] | None, set[str]]:
    """Walk ``<slug_dir>/eval/<task>/sample_00/{summary,metadata}.json``
    and pivot ``episode_return`` along each axis in *axes* (looked up in
    ``metadata.others``).

    Returns ``(breakdown, declared_keys)`` where *breakdown* is
    ``{"axes": [...], "n_tasks_with_metadata": int,
    "by_axis": {axis: {value: {"n_tasks": int,
    "mean_episode_return": float}}}}`` (or ``None`` if no task carried
    any of the axes), and *declared_keys* is the union of
    ``metadata.others`` keys seen across the slug's tasks — the key set
    ``--axes`` is typed against by the caller.
    """
    eval_root = slug_dir / "eval"
    if not eval_root.is_dir():
        return None, set()

    # axis -> value -> {"n_tasks": int, "sum_return": float}.
    # ``n_finished`` is intentionally absent: a task with both
    # metadata.json and summary.json on disk is by definition finished,
    # so n_finished would always equal n_tasks.
    by_axis_acc: dict[str, dict[str, dict[str, Any]]] = {}
    seen_axes: set[str] = set()
    declared_keys: set[str] = set()
    n_tasks_with_metadata = 0

    for task_dir in sorted(eval_root.iterdir()):
        if not task_dir.is_dir():
            continue
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
        n_tasks_with_metadata += 1
        others = meta.get("others") or {}
        declared_keys.update(others)
        ep_return = float(summ.get("episode_return", 0.0))

        for axis in axes:
            value = others.get(axis)
            # List-valued axis (e.g. GUI_types) — one entry per tag.
            values = value if isinstance(value, list) else [value]
            # Skip if THIS task has no usable value for the axis (missing
            # key, empty string, or empty list) without marking the axis
            # as "seen" on a non-contribution.
            values = [v for v in values if v not in (None, "")]
            if not values:
                continue
            seen_axes.add(axis)
            for v in values:
                v_key = str(v)
                cell = by_axis_acc.setdefault(axis, {}).setdefault(
                    v_key, {"n_tasks": 0, "sum_return": 0.0}
                )
                cell["n_tasks"] += 1
                cell["sum_return"] += ep_return

    if not seen_axes:
        return None, declared_keys

    by_axis: dict[str, dict[str, dict[str, Any]]] = {}
    for axis, cells in by_axis_acc.items():
        out_cells = {}
        for v_key, cell in cells.items():
            n = cell["n_tasks"]
            mer = cell["sum_return"] / n if n else 0.0
            out_cells[v_key] = {
                "n_tasks": n,
                "mean_episode_return": round(mer, 6),
            }
        by_axis[axis] = out_cells

    return {
        "axes": sorted(seen_axes),
        "n_tasks_with_metadata": n_tasks_with_metadata,
        "by_axis": by_axis,
    }, declared_keys


def _ensure_metadata(slug_dir: Path, env_id: str, force: bool = False) -> int:
    """If any task in *slug_dir* lacks metadata.json (or ``force=True``),
    re-instantiate the env to dump its metadata. Returns the number of
    metadata files written.
    """
    eval_root = slug_dir / "eval"
    if not eval_root.is_dir():
        return 0
    # Warm the registry so the breakdown below can read task metadata: fire the
    # env-side ``register(...)`` calls via a catalog probe. (``task_metadata`` now
    # auto-imports + lazily registers too, so this is a belt-and-suspenders warm-up.)
    try:
        gym_registry.task_ids(env_id)
    except Exception as e:
        print(f"  WARN: cannot import env {env_id}: {e}", file=sys.stderr)
        return 0
    written = 0
    for task_dir in sorted(eval_root.iterdir()):
        sample_dir = task_dir / "sample_00"
        if not sample_dir.is_dir():
            continue
        if not force and (sample_dir / "metadata.json").exists():
            continue
        md = gym_registry.task_metadata(env_id, task_dir.name)
        if md is None:
            print(f"  WARN: no registered metadata for {task_dir.name}", file=sys.stderr)
            continue
        try:
            data = md.to_dict() if hasattr(md, "to_dict") else dict(md)
        except Exception as e:
            print(f"  WARN: cannot serialize metadata for {task_dir.name}: {e}", file=sys.stderr)
            continue
        (sample_dir / "metadata.json").write_text(json.dumps(data, indent=2, default=str))
        written += 1
    return written


def _unknown_axes_message(axes: list[str], declared_keys: set[str], where: str) -> str:
    """Message for ``--axes`` entries that *where* declares no key for."""
    unknown = [a for a in axes if a not in declared_keys]
    return (
        f"--axes: {unknown} not declared by any task metadata under {where} "
        f"(declared keys: {sorted(declared_keys)}). "
        "Typo, or wrong env — see the per-env axis cheat sheet in this "
        "script's docstring."
    )


def _reaggregate_run(
    run_dir: Path,
    axes: list[str],
    env_id: str | None = None,
    force_metadata: bool = False,
    strict_axes: bool = True,
) -> tuple[int, set[str]]:
    """Reaggregate one run dir (sibling slug subdirs).

    Returns ``(n_slugs_handled, declared_keys)`` — the union of
    ``metadata.others`` keys across every slug visited.

    With *strict_axes* (single-run mode, one env), an ``--axes`` entry that
    a slug's metadata does not declare aborts. Sweep mode passes
    ``strict_axes=False`` because it spans envs with different ``others``
    keys; ``main`` types ``--axes`` once against the union instead.
    """
    if not run_dir.is_dir():
        return 0, set()
    if env_id is None:
        # Infer env_id: first path segment that matches a registered env.
        registered = set(gym_registry.env_ids())
        for p in run_dir.resolve().parts:
            if p in registered:
                env_id = p
                break
    if env_id is None:
        print(f"  WARN: cannot infer env_id from {run_dir} (pass --env-id)", file=sys.stderr)

    n_slugs = 0
    run_declared_keys: set[str] = set()
    for slug_dir in sorted(run_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        eval_root = slug_dir / "eval"
        if not eval_root.is_dir():
            continue
        # Backfill (or refresh) metadata.json files.
        if env_id is not None:
            written = _ensure_metadata(slug_dir, env_id, force=force_metadata)
            if written:
                tag = "refresh" if force_metadata else "backfill"
                print(f"  {slug_dir.name}: wrote {written} metadata.json ({tag})")
        breakdown, declared_keys = _aggregate_breakdown(slug_dir, axes)
        run_declared_keys |= declared_keys
        # Type --axes against the keys this slug's metadata declares.
        if strict_axes and declared_keys and not declared_keys.issuperset(axes):
            raise SystemExit(_unknown_axes_message(axes, declared_keys, str(slug_dir)))
        if breakdown is None:
            print(f"  {slug_dir.name}: no axes matched (others lacks {axes!r}), skipping",
                  file=sys.stderr)
            continue
        out = slug_dir / "breakdown.json"
        out.write_text(json.dumps(breakdown, indent=2))
        n_axes = len(breakdown["axes"])
        n_meta = breakdown["n_tasks_with_metadata"]
        print(f"  {slug_dir.name}: wrote breakdown.json (axes={n_axes}, n={n_meta})")
        n_slugs += 1
    return n_slugs, run_declared_keys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path", type=Path, help="Run dir or .exps/eval root with --sweep")
    p.add_argument("--axes", required=True,
                   help="Comma-separated list of others keys to pivot on, "
                        "e.g. 'paper_category,box_type,GUI_types' for "
                        "osworld_g; 'group,ui_type,application' for "
                        "screenspot_pro.")
    p.add_argument("--env-id", default=None,
                   help="Override env_id inference. Required if the path "
                        "doesn't include a registered env name.")
    p.add_argument("--sweep", action="store_true",
                   help="Treat path as the .exps/eval root and recurse into "
                        "every <env>/<commit>/<run>/ dir found.")
    p.add_argument("--force-metadata", action="store_true",
                   help="Re-render every metadata.json from the env even if "
                        "one already exists on disk. Use after adding new "
                        "fields to LiteBaseMetadata.others.")
    args = p.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    if not axes:
        raise SystemExit("--axes must be a non-empty comma-separated list")

    if args.sweep:
        # Find all <env>/<commit>/<run>/ dirs (depth 3 below the eval root).
        run_dirs: list[Path] = []
        for env_dir in sorted(args.path.iterdir()):
            if not env_dir.is_dir():
                continue
            for commit_dir in sorted(env_dir.iterdir()):
                if not commit_dir.is_dir():
                    continue
                for run_dir in sorted(commit_dir.iterdir()):
                    if run_dir.is_dir() and run_dir.name.startswith("run_"):
                        run_dirs.append(run_dir)
        print(f"sweep: {len(run_dirs)} run dirs")
        # A sweep spans envs with different ``others`` keys, so an axis
        # missing from one run is expected. Type --axes against the union
        # once, at the end: an axis no run declares is still a typo.
        declared_keys: set[str] = set()
        for rd in run_dirs:
            print(f"\n{rd}")
            _, run_keys = _reaggregate_run(rd, axes, env_id=args.env_id,
                                           force_metadata=args.force_metadata,
                                           strict_axes=False)
            declared_keys |= run_keys
        if declared_keys and not declared_keys.issuperset(axes):
            raise SystemExit(_unknown_axes_message(axes, declared_keys, str(args.path)))
    else:
        _reaggregate_run(args.path, axes, env_id=args.env_id,
                        force_metadata=args.force_metadata)


if __name__ == "__main__":
    main()
