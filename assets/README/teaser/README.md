# Paper Airplane Screensaver → Wallpaper GIF

Pixel-art animation: a paper airplane screensaver on a retro monitor, then a cursor clicks and the screen changes to a landscape wallpaper.

## Files

```
├── make_gif.py           # Main script — generates animation.gif
├── extract_assets.py     # (Optional) re-extract cursor/sparkle/airplane from decomposed.png
├── assets/
│   ├── mid_frame.png     # Source: monitor + airplane + cursor composited
│   ├── decomposed.png    # Source: elements on checkerboard background
│   ├── wallpaper.png     # Source: pixel-art landscape
│   ├── cursor.png        # Extracted arrow cursor
│   ├── sparkle.png       # Extracted click sparkle
│   └── airplane.png      # Extracted paper airplane
└── animation.gif         # Output
```

## Usage

```bash
pip install Pillow numpy scipy
python make_gif.py
```

The script uses **ImageMagick** (`convert`) for proper per-frame timing. If not installed, it falls back to Pillow (which may merge duplicate frames).

To re-extract assets from `decomposed.png`:

```bash
python extract_assets.py
```

## Config (in make_gif.py)

| Variable | Default | Description |
|---|---|---|
| `SIZE` | 512 | Output canvas size |
| `CURSOR_SCALE` | 0.6 | Cursor size multiplier |
| `AIRPLANE_SCALE` | 1.3 | Airplane size multiplier |
| `SPARKLE_SCALE` | 0.6 | Sparkle size multiplier |
