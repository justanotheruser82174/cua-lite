"""Per-turn cross-substrate screenshot diff for the lite.osworld ↔ osworld bridge.

The repo has no image-diff util (`lite/utils/image.py` is load / encode / resize
/ smart_resize only), so this adds what §3 of AGENTS.md needs. **Two complementary
metrics — do not rely on SSIM alone:**

  * **SSIM** (grayscale/luminance) — structural / layout / contrast gaps. It is
    computed on luminance, so it is largely **blind to equal-luminance COLOR or
    STYLE shifts** — e.g. a terminal prompt that is green/blue on the VM but plain
    on the container reads ~1.0 SSIM (the very gap we target). Necessary, not sufficient.
  * **ΔE (CIEDE2000, in CIELAB)** — perceptual COLOR distance per pixel; catches
    exactly the chromatic/palette/theme differences SSIM misses. Report mean + p95.

Plus a side-by-side **contact sheet** (orig | replay | amplified abs-diff) as the
§5 evidence artifact — and, for subtle font/hinting/style the pixel metrics still
miss, feed that sheet to a human or a VLM judge. Frames MUST already be the same
pixel size (both substrates render 1920×1080) — asserted, never rescaled.

Run (two dirs of flattened `turn_NNNN.png` bridge frames, or sample dirs whose
turn images can be read by `lite.infer.debug.log_layout`, aligned by turn name):
    uv run python devs/envs/lite.osworld/bridge/visual_diff.py ORIG_DIR REPLAY_DIR -o /tmp/diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from lite.infer.debug.log_layout import last_turn_image, turn_dirs

# (x0, y0, x1, y1) regions ignored on BOTH frames (taskbar clock, cursor glyph, …).
# Tune to the real 1920×1080 layout once measured; the same-substrate smoke showed
# the clock is the dominant non-determinism.
Rect = tuple[int, int, int, int]
DEFAULT_MASKS: tuple[Rect, ...] = ()

SSIM_FLAG = 0.98   # below → structural gap
DE_FLAG = 3.0      # p95 ΔE above → chromatic/style gap (CIEDE2000 ≈2-3 is a JND)


def _assert_same_size(a: Image.Image, b: Image.Image) -> None:
    if a.size != b.size:
        raise ValueError(
            f"frame size mismatch {a.size} vs {b.size} — assert equal resolution "
            f"(both substrates render 1920×1080); do NOT rescale (it hides real gaps)"
        )


def _keep(shape: tuple[int, int], masks: tuple[Rect, ...]) -> np.ndarray:
    """Boolean H×W: True = score this pixel, False = masked-out (EXCLUDED from the
    stat, not zeroed — zeroing would count a masked region as a perfect match and
    dilute the metric)."""
    keep = np.ones(shape, dtype=bool)
    for x0, y0, x1, y1 in masks:
        keep[y0:y1, x0:x1] = False
    return keep


def ssim(orig_png: str | Path, replay_png: str | Path,
         masks: tuple[Rect, ...] = DEFAULT_MASKS) -> float:
    """Grayscale (luminance) SSIM in [-1, 1]; 1.0 == identical. STRUCTURE only —
    pair with :func:`delta_e` for color/style."""
    o = Image.open(orig_png).convert("L")
    r = Image.open(replay_png).convert("L")
    _assert_same_size(o, r)
    # data_range=255 required for uint8 in recent skimage (else ValueError).
    _, s_map = structural_similarity(np.asarray(o), np.asarray(r), data_range=255, full=True)
    keep = _keep(s_map.shape, masks)          # exclude masked pixels from the mean
    return float(s_map[keep].mean())


def delta_e(orig_png: str | Path, replay_png: str | Path,
            masks: tuple[Rect, ...] = DEFAULT_MASKS) -> tuple[float, float]:
    """Perceptual COLOR distance (CIEDE2000 over CIELAB), masked. Returns
    ``(mean, p95)`` ΔE — catches equal-luminance palette/theme/style shifts that
    grayscale SSIM is blind to."""
    o = Image.open(orig_png).convert("RGB")
    r = Image.open(replay_png).convert("RGB")
    _assert_same_size(o, r)
    lab_o = rgb2lab(np.asarray(o) / 255.0)
    lab_r = rgb2lab(np.asarray(r) / 255.0)
    de = deltaE_ciede2000(lab_o, lab_r)[_keep(  # H×W → 1-D over kept pixels
        np.asarray(o).shape[:2], masks)]
    return float(de.mean()), float(np.percentile(de, 95))


def contact_sheet(orig_png: str | Path, replay_png: str | Path,
                  out_png: str | Path) -> Path:
    """Write ``orig | replay | amplified-abs-diff`` side by side as evidence
    (also the artifact to hand a VLM/human for subtle style calls)."""
    o = Image.open(orig_png).convert("RGB")
    r = Image.open(replay_png).convert("RGB")
    _assert_same_size(o, r)
    diff = np.abs(np.asarray(o, dtype=int) - np.asarray(r, dtype=int)).sum(axis=2)
    d = Image.fromarray(np.clip(diff, 0, 255).astype("uint8")).convert("RGB")
    w, h = o.size
    sheet = Image.new("RGB", (w * 3, h), "black")
    for i, im in enumerate((o, r, d)):
        sheet.paste(im, (i * w, 0))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def diff_dirs(orig_dir: str | Path, replay_dir: str | Path, out_dir: str | Path,
              masks: tuple[Rect, ...] = DEFAULT_MASKS) -> list[dict]:
    """Diff aligned turns from flat bridge dirs or current sample dirs on SSIM
    **and** ΔE. Returns per-turn
    ``{turn, ssim, de_mean, de_p95, flagged, sheet}``; a contact sheet is written
    (and ``flagged=True``) when SSIM < flag OR ΔE-p95 > flag — so a color-only gap
    is caught even at SSIM ≈ 1.0."""
    orig_dir, replay_dir, out_dir = Path(orig_dir), Path(replay_dir), Path(out_dir)
    frames = _aligned_frames(orig_dir, replay_dir)
    rows: list[dict] = []
    for name, orig_png, replay_png in frames:
        s = ssim(orig_png, replay_png, masks)
        de_mean, de_p95 = delta_e(orig_png, replay_png, masks)
        flagged = s < SSIM_FLAG or de_p95 > DE_FLAG
        sheet = str(contact_sheet(orig_png, replay_png, out_dir / f"diff_{name}.png")) if flagged else None
        rows.append({"turn": name, "ssim": s, "de_mean": de_mean,
                     "de_p95": de_p95, "flagged": flagged, "sheet": sheet})
    return rows


def _turn_name_key(name: str) -> tuple[int, str]:
    return (int(name.split("_")[1].split(".")[0]), name)


def _frames_by_turn(path: Path) -> dict[str, Path]:
    flat = {
        p.stem: p
        for p in path.glob("turn_*.png")
        if p.is_file()
    }
    if flat:
        return flat

    out: dict[str, Path] = {}
    for turn_dir in turn_dirs(path):
        image = last_turn_image(turn_dir)
        if image is not None:
            out[turn_dir.name] = image
    return out


def _aligned_frames(orig_dir: Path, replay_dir: Path) -> list[tuple[str, Path, Path]]:
    orig = _frames_by_turn(orig_dir)
    replay = _frames_by_turn(replay_dir)
    names = sorted(orig.keys() & replay.keys(), key=_turn_name_key)
    return [(name, orig[name], replay[name]) for name in names]


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("orig_dir")
    ap.add_argument("replay_dir")
    ap.add_argument("-o", "--out-dir", default="/tmp/bridge_diff")
    args = ap.parse_args()
    rows = diff_dirs(args.orig_dir, args.replay_dir, args.out_dir)
    if not rows:
        print("no aligned turn frames in both dirs")
        return
    for row in rows:
        flag = "  <-- GAP" + (f" ({row['sheet']})" if row["sheet"] else "") if row["flagged"] else ""
        print(f"  {row['turn']}: SSIM={row['ssim']:.4f}  ΔE(mean/p95)={row['de_mean']:.2f}/{row['de_p95']:.2f}{flag}")
    worst_ssim = min(rows, key=lambda r: r["ssim"])
    worst_de = max(rows, key=lambda r: r["de_p95"])
    print(f"worst SSIM: {worst_ssim['turn']} {worst_ssim['ssim']:.4f} (flag < {SSIM_FLAG})")
    print(f"worst ΔE-p95: {worst_de['turn']} {worst_de['de_p95']:.2f} (flag > {DE_FLAG})")


if __name__ == "__main__":
    _main()
