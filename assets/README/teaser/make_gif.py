"""
Paper Airplane Screensaver -> Wallpaper GIF Animation
=====================================================

Narrative:
1. Monitor shows paper airplane screensaver — airplane floats with a breathing motion
2. Cursor slides in from right — hidden behind the monitor bezel until it enters the screen
3. Cursor clicks (yellow sparkle)
4. Screen transitions to pixel-art landscape wallpaper
5. Cursor stays in place

Layer order (bottom -> top):
  1. White background
  2. Screen content (black + airplane, or wallpaper)
  3. Cursor + sparkle
  4. Bezel overlay — hides cursor outside the screen area

Requirements:
    pip install Pillow numpy scipy

Usage:
    python make_gif.py

Output:
    animation.gif
"""

import math
import os
import subprocess
import sys
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIZE = 512
SCALE = SIZE / 1024.0

# Crop top/bottom whitespace from final frames (monitor occupies rows ~107-397).
# Leave a small padding so the monitor is not flush against the edge.
CROP_TOP = 95
CROP_BOTTOM = 103
OUT_H = SIZE - CROP_TOP - CROP_BOTTOM

CURSOR_SCALE = 1.2   # new 40x60 cursor; 0.5*1.2 -> ~24x36 in the gif (0.8x of prior)
AIRPLANE_SCALE = 1.1
SPARKLE_SCALE = 0.6

BREATH_AMPLITUDE = 9      # pixels up/down
BREATH_PERIOD = 30         # frames per full sine cycle

USE_IMAGEMAGICK = True


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(SCRIPT_DIR, "assets")
OUTPUT = os.path.join(SCRIPT_DIR, "animation.gif")
FRAMES_DIR = os.path.join(SCRIPT_DIR, "_frames")

# Screen rectangle (scaled from 1024-space)
SL = int(195 * SCALE)
SR = int(835 * SCALE)
ST = int(294 * SCALE)
SB = int(648 * SCALE)
SW, SH = SR - SL, SB - ST


def load_and_scale():
    monitor = Image.open(os.path.join(ASSETS, "mid_frame.png")).convert("RGBA")
    monitor = monitor.resize((SIZE, SIZE), Image.LANCZOS)
    # The asset's background is (254,254,254), not pure white, which leaks
    # into the bezel and tints the final canvas. Snap any near-white pixel
    # (min channel >= 254) to pure white. Leaves the monitor body and the
    # shadow (which sits at 252 and below) untouched.
    arr = np.array(monitor)
    near_white = arr[:, :, :3].min(axis=2) >= 254
    arr[near_white, :3] = 255
    monitor = Image.fromarray(arr)

    wallpaper = Image.open(os.path.join(ASSETS, "wallpaper.png")).convert("RGBA")

    def scale(img, extra):
        return img.resize(
            (int(img.width * SCALE * extra), int(img.height * SCALE * extra)),
            Image.LANCZOS,
        )

    cursor = scale(Image.open(os.path.join(ASSETS, "cursor.png")).convert("RGBA"), CURSOR_SCALE)
    sparkle = scale(Image.open(os.path.join(ASSETS, "sparkle.png")).convert("RGBA"), SPARKLE_SCALE)
    airplane = scale(Image.open(os.path.join(ASSETS, "airplane.png")).convert("RGBA"), AIRPLANE_SCALE)

    return monitor, wallpaper, cursor, sparkle, airplane


def build_layers(monitor, wallpaper):
    """Build reusable layers."""
    # Bezel overlay: monitor with screen area punched transparent
    bezel = monitor.copy()
    arr = np.array(bezel)
    arr[ST:SB, SL:SR, 3] = 0
    bezel = Image.fromarray(arr)

    # White screen (only the screen rectangle is filled)
    white_screen = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    arr_bs = np.array(white_screen)
    arr_bs[ST:SB, SL:SR] = [255, 255, 255, 255]
    white_screen = Image.fromarray(arr_bs)

    # Wallpaper screen
    wp_screen = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    arr_ws = np.array(wp_screen)
    wp_fit = wallpaper.resize((SW, SH), Image.LANCZOS)
    arr_ws[ST:SB, SL:SR, :3] = np.array(wp_fit)[:, :, :3]
    arr_ws[ST:SB, SL:SR, 3] = 255
    wp_screen = Image.fromarray(arr_ws)

    return bezel, white_screen, wp_screen


def compose(screen, airplane, airplane_y_off, cursor, cursor_pos, sparkle, show_sparkle, bezel, is_wallpaper):
    """
    Compose one frame with correct layer order.
    Bezel on top hides the cursor when it is outside the screen area.
    """
    frame = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))

    # Layer 2: screen
    frame.paste(screen, (0, 0), screen)

    # Layer 3: airplane (screensaver only)
    if not is_wallpaper:
        ax = SL + (SW - airplane.width) // 2
        ay = ST + (SH - airplane.height) // 2 + airplane_y_off
        frame.paste(airplane, (ax, ay), airplane)

    # Layer 4: cursor + sparkle
    if cursor_pos is not None:
        cx, cy = cursor_pos
        frame.paste(cursor, (cx, cy), cursor)
        if show_sparkle:
            sx = cx + int(-37 * SCALE * SPARKLE_SCALE)
            sy = cy + int(-32 * SCALE * SPARKLE_SCALE)
            frame.paste(sparkle, (sx, sy), sparkle)

    # Layer 5: bezel — clips cursor outside screen
    frame.paste(bezel, (0, 0), bezel)

    return frame


def breath(frame_idx):
    return int(BREATH_AMPLITUDE * math.sin(2 * math.pi * frame_idx / BREATH_PERIOD))


def generate_frames(airplane, cursor, sparkle, bezel, white_screen, wp_screen):
    frames = []   # (RGB image, delay_centiseconds)
    gf = 0        # global frame counter for breathing phase

    click_x = SL + int(SW * 0.6)
    click_y = ST + int(SH * 0.5)
    start_cx, start_cy = SIZE + 10, int(200 * SCALE)

    def add(img, delay):
        bg = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
        bg.paste(img, (0, 0), img)
        bg = bg.crop((0, CROP_TOP, SIZE, SIZE - CROP_BOTTOM))
        frames.append((bg, delay))

    # Phase 1: Screensaver — breathing airplane (2.4 s)
    for _ in range(12):
        f = compose(white_screen, airplane, breath(gf), cursor, None, sparkle, False, bezel, False)
        add(f, 10); gf += 1

    # Phase 2: Cursor enters from right (0.7 s)
    for i in range(10):
        t = 1 - (1 - i / 9) ** 3
        cx = int(start_cx + (click_x - start_cx) * t)
        cy = int(start_cy + (click_y - start_cy) * t)
        f = compose(white_screen, airplane, breath(gf), cursor, (cx, cy), sparkle, False, bezel, False)
        add(f, 7); gf += 1

    # Phase 3: Pause (0.1 s)
    for _ in range(1):
        f = compose(white_screen, airplane, breath(gf), cursor, (click_x, click_y), sparkle, False, bezel, False)
        add(f, 10); gf += 1

    # Phase 4: Click sparkle (0.3 s)
    for _ in range(3):
        f = compose(white_screen, airplane, breath(gf), cursor, (click_x, click_y), sparkle, True, bezel, False)
        add(f, 10); gf += 1

    # Phase 5: Transition to wallpaper
    f = compose(wp_screen, airplane, 0, cursor, (click_x, click_y), sparkle, True, bezel, True)
    add(f, 10)
    f = compose(wp_screen, airplane, 0, cursor, (click_x, click_y), sparkle, False, bezel, True)
    add(f, 12)

    # Phase 6: Wallpaper hold — cursor stays (2.0 s)
    for _ in range(8):
        f = compose(wp_screen, airplane, 0, cursor, (click_x, click_y), sparkle, False, bezel, True)
        add(f, 25)

    return frames


def save_gif_imagemagick(frames, output_path):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    cmd = ["convert", "-loop", "0"]
    for idx, (img, delay) in enumerate(frames):
        p = os.path.join(FRAMES_DIR, f"f_{idx:03d}.png")
        img.save(p)
        cmd.extend(["-delay", str(delay), p])
    cmd.append(output_path)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("ImageMagick 'convert' not found on PATH", file=sys.stderr)
        return False
    if r.returncode != 0:
        print(f"ImageMagick error: {r.stderr}", file=sys.stderr)
    return r.returncode == 0


def save_gif_pillow(frames, output_path):
    # Build a single global palette from all frames combined, then quantize
    # every frame to it. Without this, Pillow re-quantizes each frame
    # independently, so high-saturation pixels (e.g. monitor buttons) get
    # rounded to different palette entries when the screen content changes,
    # producing a visible color flicker even in regions that are pixel-
    # identical across frames.
    imgs = [f.convert("RGB") for f, _ in frames]
    w, h = imgs[0].size
    # Bias the palette heavily toward the first (screensaver) frame so that
    # the vibrant monitor button colors survive median-cut quantization
    # instead of being averaged away by the wallpaper's color distribution.
    FIRST_WEIGHT = 20
    tiles = [imgs[0]] * FIRST_WEIGHT + imgs
    combined = Image.new("RGB", (w, h * len(tiles)))
    for i, im in enumerate(tiles):
        combined.paste(im, (0, i * h))
    palette_img = combined.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [
        im.quantize(palette=palette_img, dither=Image.Dither.NONE) for im in imgs
    ]
    # Median-cut maps the (255,255,255) background to a near-white entry
    # (e.g. 252,252,252), tinting the canvas. Putpalette before quantize
    # doesn't fix it (Pillow's nearest-neighbor still picks the original
    # entry). Workaround: find the palette index used by the corner pixel
    # and rewrite that entry to pure white on every frame's palette.
    bg_idx = int(np.array(quantized[0])[0, 0])
    for q in quantized:
        p = list(q.getpalette())
        p[bg_idx * 3 : bg_idx * 3 + 3] = [255, 255, 255]
        q.putpalette(p)
    durs = [d * 10 for _, d in frames]
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=durs,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return True


def main():
    print("Loading assets...")
    monitor, wallpaper, cursor, sparkle, airplane = load_and_scale()

    print("Building layers...")
    bezel, white_screen, wp_screen = build_layers(monitor, wallpaper)

    print("Generating frames...")
    frames = generate_frames(airplane, cursor, sparkle, bezel, white_screen, wp_screen)
    total_ms = sum(d * 10 for _, d in frames)
    print(f"  {len(frames)} frames, {total_ms} ms ({total_ms / 1000:.1f} s)")

    print(f"Saving -> {OUTPUT}")
    ok = False
    if USE_IMAGEMAGICK:
        ok = save_gif_imagemagick(frames, OUTPUT)
        if not ok:
            print("  ImageMagick failed, falling back to Pillow...")
    if not ok:
        save_gif_pillow(frames, OUTPUT)

    print(f"Done! {os.path.getsize(OUTPUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
