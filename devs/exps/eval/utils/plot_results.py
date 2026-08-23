"""Render headline MER charts for one (env, snapshot) pair.

Reads each model's `summary.json` under the passed snapshot dir; emits a PNG
to ``<env>/logs/<commit-ts>_<commit>/figures/<run_id>/<mode>.png`` (per-run
subdir so different configs — default / textonly — don't clobber). The snapshot dir
is the source of truth — if you want carry-forward of older models, update
the snapshot.md (and its corresponding artifact dir) so the artifact tree
matches what the snapshot reports.

Modes:
    bars     — horizontal bar chart sorted by MER desc, color = family,
               within-family shade gradient (bigger model = darker).
    scaling  — scatter plot, x=size (B, log2 scale), y=MER, same-family
               points connected by dashed line ONLY when sizes differ
               (versions-at-same-size, e.g. UI-TARS-DPO vs 1.5, not
               connected).

Usage:
    uv run python devs/exps/eval/utils/plot_results.py \\
        .exps/eval/osworld_g/2026-04-28T21-31_785df232/run_0/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from adjustText import adjust_text  # auto-nudges overlapping labels

# Per-family colormap — bigger model = darker shade within the family.
# Legend uses the family's mid-shade.
FAMILY_CMAP = {
    "Qwen3-VL":   plt.colormaps["Blues"],
    "Qwen3.5":    plt.colormaps["Greens"],
    "UI-TARS":    plt.colormaps["Oranges"],
    "MAI-UI":     plt.colormaps["Purples"],
    "EvoCUA":     plt.colormaps["Reds"],
    "ScaleCUA":   plt.colormaps["YlOrBr"],
    "UI-Voyager": plt.colormaps["PuRd"],
    "GELab-Zero": plt.colormaps["GnBu"],
    "OpenCUA":    plt.colormaps["Greys"],
}
FAMILY_LEGEND_COLOR = {f: cm(0.65) for f, cm in FAMILY_CMAP.items()}


def family_of(slug: str) -> str:
    if "Qwen3-VL" in slug:
        return "Qwen3-VL"
    if "Qwen3.5" in slug:
        return "Qwen3.5"
    if "UI-TARS" in slug:
        return "UI-TARS"
    if "MAI-UI" in slug:
        return "MAI-UI"
    if "EvoCUA" in slug:
        return "EvoCUA"
    if "ScaleCUA" in slug:
        return "ScaleCUA"
    if "UI-Voyager" in slug:
        return "UI-Voyager"
    if "GELab-Zero" in slug:
        return "GELab-Zero"
    if "OpenCUA" in slug:
        return "OpenCUA"
    return "?"


def display_name(slug: str) -> str:
    """Compact label: drop org prefix + redundant suffixes."""
    s = slug.replace("Qwen_", "").replace("ByteDance-Seed_", "").replace(
        "Tongyi-MAI_", "").replace("meituan_", "").replace(
        "OpenGVLab_", "").replace("MarsXL_", "").replace("stepfun-ai_", "")
    s = s.replace("-Instruct", "").replace("-Thinking", "")
    s = re.sub(r"-\d{8}$", "", s)  # drop date suffix on EvoCUA
    return s


def size_key(slug: str) -> tuple[float, int]:
    """For within-family ordering. (param_size_B, version_marker).

    Most slugs encode size as ``<N>B`` (e.g. ``Qwen3-VL-32B-Instruct``);
    a few don't (UI-Voyager, ...) and need a manual fallback below."""
    m = re.search(r"(\d+(?:\.\d+)?)B", slug)
    size = float(m.group(1)) if m else SIZE_FALLBACK.get(slug, 0.0)
    version = 1 if "1.5" in slug else 0  # UI-TARS-1.5 > UI-TARS DPO
    return (size, version)


SIZE_FALLBACK: dict[str, float] = {
    # MarsXL/UI-Voyager: built on Qwen3-VL-4B (config.json shows
    # hidden_size=2560, num_hidden_layers=36 — identical to Qwen3-VL-4B-Instruct;
    # ~4.2B params on disk in bf16). Slug has no `\d+B` pattern.
    "MarsXL_UI-Voyager": 4.0,
}

# Per-env raw task count (incl. filtered). Mirrors README's "Task counts per env"
# table — used to render `n=eligible of total` in chart titles.
ENV_RAW_TASKS: dict[str, int] = {
    "lite.osworld":   369,
    "osworld":        369,
    "androidlab":    138,
    "androidworld":  116,
    "osworld_g":      564,
    "screenspot_pro": 1581,
}


def assign_colors(rows: list[dict]) -> None:
    """For each row, assign a color = family colormap evaluated at a position
    proportional to the model's size rank within its family. Single-model
    families get the mid shade. Mutates `rows` in place (sets row["color"]).

    Shade is keyed on the *distinct* size (via size_key on base_slug), so a
    model's think_on/off variants share the exact same color and are told
    apart only by hatch in the grouped chart."""
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    for fam, fam_rows in by_family.items():
        cm = FAMILY_CMAP.get(fam, plt.colormaps["Greys"])
        keys = sorted({size_key(r["base_slug"]) for r in fam_rows})
        rank = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        for r in fam_rows:
            i = rank[size_key(r["base_slug"])]
            # spread shades from 0.40 (light) to 0.90 (dark); single → 0.65
            t = 0.65 if n == 1 else 0.40 + 0.50 * i / max(n - 1, 1)
            r["color"] = cm(t)


def load_rows(snapshot_dir: Path) -> list[dict]:
    """Read summary.json files under each model subdir."""
    rows = []
    for sub in sorted(snapshot_dir.iterdir()):
        f = sub / "summary.json"
        if not f.is_file():
            continue
        d = json.loads(f.read_text())
        s = d["stats"]
        m = re.search(r"__think_(on|off)$", sub.name)
        think = m.group(1) if m else None
        base = re.sub(r"__think_(on|off)$", "", sub.name)
        rows.append({
            "slug": sub.name,
            "base_slug": base,
            "think": think,  # "on" / "off" / None
            "label": display_name(base),
            "family": family_of(base),
            "mer": s["mean_episode_return"],
            "finished": s["num_valid"],
            "total": s["num_tasks"],
        })
    return rows


def figures_dir(snapshot_dir: Path) -> tuple[Path, str, str, str]:
    """Resolve output dir + names for a snapshot.

    Figures land in a per-run subdir so different configs of the same
    (env, commit) don't clobber each other:
        <repo-root>/devs/exps/eval/<env>/logs/<commit>/figures/<run_id>/
    where <run_id> is the snapshot dir name (e.g. ``run_1_textonly``). The
    trailing config token (``textonly`` / ``default``) is returned separately
    for the chart title."""
    run_id = snapshot_dir.name  # e.g. run_1_textonly
    commit_id = snapshot_dir.parent.name
    env_name = snapshot_dir.parent.parent.name
    short_sha = commit_id.split("_")[-1]
    m = re.match(r"run_\d+_(.+)$", run_id)
    config = m.group(1) if m else run_id  # e.g. textonly / default
    repo_root = Path(__file__).resolve().parents[4]
    out_dir = repo_root / "devs/exps/eval" / env_name / "logs" / commit_id / "figures" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, env_name, short_sha, config


def render_bars(snapshot_dir: Path) -> Path:
    """Read summary.json files in `snapshot_dir/<slug>/`, render bar chart.

    Auto-detects a think on/off axis (``__think_on`` / ``__think_off`` slug
    suffix): when present, renders the grouped-hatch layout (two bars per
    model, solid = think off, ``///`` = think on) instead of the flat one."""
    rows = load_rows(snapshot_dir)
    assign_colors(rows)
    if any(r["think"] for r in rows):
        return render_bars_think(rows, snapshot_dir)

    rows.sort(key=lambda r: r["mer"])  # asc → top of barh chart = highest MER

    out_dir, env_name, short_sha, config = figures_dir(snapshot_dir)

    n_eligible = rows[0]["total"]
    n_total = ENV_RAW_TASKS.get(env_name, n_eligible)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.5))
    bars = ax.barh(
        [r["label"] for r in rows],
        [r["mer"] for r in rows],
        color=[r["color"] for r in rows],
        edgecolor="white", height=0.78,
    )

    for bar, r in zip(bars, rows):
        partial = "" if r["finished"] == r["total"] else f" ⚠ {r['finished']}/{r['total']}"
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f" {r['mer']:.4f}{partial}",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Success rate")
    ax.set_xlim(0, max(r["mer"] for r in rows) * 1.18)
    n_str = f"n={n_eligible}" if n_total == n_eligible else f"n={n_eligible} of {n_total} eligible"
    ax.set_title(f"{env_name}  ·  {config}  ·  {n_str}  ·  {short_sha}", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # legend ordered by family's best MER (most-relevant family first)
    family_best = {}
    for r in rows:
        family_best[r["family"]] = max(family_best.get(r["family"], 0), r["mer"])
    families_present = sorted(family_best, key=lambda f: -family_best[f])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_LEGEND_COLOR.get(f, "#999"))
        for f in families_present
    ]
    ax.legend(handles, families_present, loc="lower right", frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    out = out_dir / "bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_bars_think(rows: list[dict], snapshot_dir: Path) -> Path:
    """Grouped-hatch bar chart for a think on/off axis.

    Two bars per model, kept adjacent (groups sorted by the better of the two
    MERs). Color = family hue + size shade (shared by both variants); hatch
    tells the variants apart: solid = think off, ``///`` = think on."""
    out_dir, env_name, short_sha, config = figures_dir(snapshot_dir)

    # group by model label; each group holds its think on/off variants
    # (or a single None entry for models with no think axis).
    groups: dict[str, dict[str | None, dict]] = {}
    for r in rows:
        groups.setdefault(r["label"], {})[r["think"]] = r
    # asc by best variant MER → top of barh chart = highest MER.
    ordered = sorted(groups.items(), key=lambda kv: max(r["mer"] for r in kv[1].values()))
    n_groups = len(ordered)

    n_eligible = max(r["total"] for r in rows)
    n_total = ENV_RAW_TASKS.get(env_name, n_eligible)
    max_mer = max(r["mer"] for r in rows)

    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.5))
    bar_h = 0.38
    # think_off sits below the group center, think_on above.
    dy = {"off": -0.20, "on": 0.20}

    yticks, yticklabels = [], []
    for gi, (label, variants) in enumerate(ordered):
        yticks.append(gi)
        yticklabels.append(label)
        # A model with both variants gets two offset bars; a model with no
        # think axis (None — api models, UI-TARS) or a lone variant gets one
        # full-height bar centered on the group.
        paired = "on" in variants and "off" in variants
        for mode, r in variants.items():
            y = gi + (dy[mode] if paired else 0.0)
            ax.barh(
                y, r["mer"], height=bar_h if paired else 0.62,
                color=r["color"], edgecolor="white",
                hatch="///" if mode == "on" else None,
            )
            partial = "" if r["finished"] == r["total"] else f" ⚠ {r['finished']}/{r['total']}"
            ax.text(r["mer"] + 0.005, y, f" {r['mer']:.4f}{partial}", va="center", fontsize=8)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-0.6, n_groups - 1 + 0.6)
    ax.set_xlabel("Success rate")
    ax.set_xlim(0, max_mer * 1.18)
    n_str = f"n={n_eligible}" if n_total == n_eligible else f"n={n_eligible} of {n_total} eligible"
    ax.set_title(
        f"{env_name}  ·  {config}  ·  think on/off  ·  {n_str}  ·  {short_sha}",
        loc="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # two legends: family (hue) + think mode (hatch).
    family_best: dict[str, float] = {}
    for r in rows:
        family_best[r["family"]] = max(family_best.get(r["family"], 0), r["mer"])
    families_present = sorted(family_best, key=lambda f: -family_best[f])
    fam_handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_LEGEND_COLOR.get(f, "#999"))
        for f in families_present
    ]
    fam_leg = ax.legend(fam_handles, families_present, loc="lower right",
                        frameon=False, fontsize=8, ncol=2)
    ax.add_artist(fam_leg)
    think_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#b0b0b0", edgecolor="white"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#b0b0b0", edgecolor="white", hatch="///"),
    ]
    ax.legend(think_handles, ["think off", "think on"], loc="upper right",
              frameon=False, fontsize=8)

    fig.tight_layout()
    out = out_dir / "bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def render_scaling(snapshot_dir: Path) -> Path | None:
    """Scatter chart: x=size (B, log2), y=MER, dashed line per family.

    Skips models with no parseable size (api models like gpt/claude can't sit
    on a parameter axis); returns None if that leaves nothing to plot."""
    rows = load_rows(snapshot_dir)
    rows = [r for r in rows if size_key(r["base_slug"])[0]]
    if not rows:
        print(f"[scaling] no sized models in {snapshot_dir.name}; skipping")
        return None
    assign_colors(rows)
    out_dir, env_name, short_sha, config = figures_dir(snapshot_dir)

    has_think = any(r["think"] for r in rows)

    # parse size with tiny version offset so same-size+different-version points
    # in the same family (UI-TARS-DPO vs UI-TARS-1.5) don't overlap exactly.
    # A think_on point is nudged slightly right of its off counterpart.
    for r in rows:
        size, version = size_key(r["base_slug"])
        r["x"] = size * (1 + 0.04 * version) * (1.03 if r["think"] == "on" else 1.0)

    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    n_eligible = max(r["total"] for r in rows)

    # legend ordered by family's best MER (consistent with bars chart)
    family_best = {f: max(r["mer"] for r in fam_rows)
                   for f, fam_rows in by_family.items()}
    families_ordered = sorted(by_family, key=lambda f: -family_best[f])

    # With a think axis, split each family into per-mode sub-series so dashed
    # lines connect within a mode; open markers = think on, filled = off/none.
    # None is kept so mixed folders (api/UI-TARS models with no think axis)
    # still plot their non-think points.
    modes = [m for m in ("off", "on", None) if any(r["think"] == m for r in rows)] \
        if has_think else [None]

    all_xs, all_ys, all_texts = [], [], []
    for fam in families_ordered:
        legend_color = FAMILY_LEGEND_COLOR.get(fam, "#999")
        for mode in modes:
            series = sorted(
                [r for r in by_family[fam] if mode is None or r["think"] == mode],
                key=lambda r: r["x"],
            )
            if not series:
                continue
            open_marker = mode == "on"
            # Connect consecutive points with dashed line only when sizes differ.
            # Same-size pairs (UI-TARS DPO vs 1.5) get markers but no line.
            for a, b in zip(series, series[1:]):
                if size_key(a["base_slug"])[0] != size_key(b["base_slug"])[0]:
                    ax.plot(
                        [a["x"], b["x"]], [a["mer"], b["mer"]], "--",
                        color=legend_color, alpha=0.55, linewidth=1.5, zorder=2,
                    )
            colors = [r["color"] for r in series]
            ax.scatter(
                [r["x"] for r in series], [r["mer"] for r in series],
                facecolor="none" if open_marker else colors,
                edgecolor=colors if open_marker else "white",
                s=110, zorder=3, linewidth=1.6 if open_marker else 1.2,
            )
            for r in series:
                txt = r["label"] + (" (think)" if mode == "on" else "")
                t = ax.text(r["x"], r["mer"], txt, fontsize=10, color="#222", zorder=4)
                all_texts.append(t)
                all_xs.append(r["x"])
                all_ys.append(r["mer"])

    ax.set_xscale("log", base=2)
    sizes_present = sorted(
        {size_key(r["base_slug"])[0] for r in rows if size_key(r["base_slug"])[0]}
    )
    ticks = [t for t in [2, 4, 8, 16, 32] if min(sizes_present) <= t <= max(sizes_present) * 1.1]
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}B")
    )
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("Success rate")
    ax.set_title(
        f"{env_name}  ·  {config}  ·  scaling  ·  n={n_eligible}  ·  {short_sha}",
        loc="left",
    )
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            color=FAMILY_LEGEND_COLOR.get(f, "#999"),
            marker="o", markersize=8, linestyle="--",
        )
        for f in families_ordered
    ]
    fam_leg = ax.legend(handles, families_ordered, loc="lower right", frameon=False, fontsize=9)
    if has_think:
        ax.add_artist(fam_leg)
        think_handles = [
            plt.Line2D([0], [0], color="#666", marker="o", markersize=8,
                       markerfacecolor="#666", linestyle="none"),
            plt.Line2D([0], [0], color="#666", marker="o", markersize=8,
                       markerfacecolor="none", linestyle="none"),
        ]
        ax.legend(think_handles, ["think off", "think on"], loc="upper left",
                  frameon=False, fontsize=9)

    # Auto-nudge labels to avoid overlap with each other and with markers.
    # Thin gray leader line drawn when a label gets pushed away from its point.
    # Cranked-up forces — screenspot_pro / androidlab have multiple families
    # bunched near 4B–8B at similar success rate, so default repulsion is too
    # gentle to separate them.
    adjust_text(
        all_texts,
        x=all_xs, y=all_ys, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#888", alpha=0.5, lw=0.6),
        expand=(1.4, 1.6),
        force_text=(1.5, 2.0),
        force_static=(0.6, 0.9),
        force_pull=(0.005, 0.005),
        iter_lim=500,
    )

    fig.tight_layout()
    out = out_dir / "scaling.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("snapshot_dir", help="Path to .exps/eval/<env>/<commit-ts>_<commit>/<run_id>/")
    p.add_argument("--mode", choices=["bars", "scaling", "all"], default="all")
    args = p.parse_args()
    snap = Path(args.snapshot_dir).resolve()
    if args.mode in ("bars", "all"):
        print(f"wrote {render_bars(snap)}")
    if args.mode in ("scaling", "all"):
        out = render_scaling(snap)
        if out is not None:
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
