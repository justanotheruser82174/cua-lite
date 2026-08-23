"""
Extract cursor, sparkle, and airplane from the decomposed image.

Run this if you need to re-extract assets/cursor.png, assets/sparkle.png,
and assets/airplane.png from assets/decomposed.png.

Requirements:
    pip install Pillow numpy scipy
"""

import os
import numpy as np
from PIL import Image
from scipy import ndimage

ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def extract_element(arr, region, dark_thresh=130, dilate_iter=3, connect_iter=1):
    """
    Extract a pixel-art element from a checkerboard background.

    1. Find dark outline pixels (avg brightness < dark_thresh)
    2. Dilate to connect gaps in pixel-art outlines
    3. Find largest connected component
    4. Fill holes to get complete shape
    5. Return RGBA with transparent background
    """
    y0, y1, x0, x1 = region
    crop = arr[y0:y1, x0:x1]

    avg = crop[:, :, :3].mean(axis=2)
    dark = avg < dark_thresh

    # Connect outline segments
    dilated = ndimage.binary_dilation(dark, iterations=dilate_iter)
    labeled, num = ndimage.label(dilated)
    sizes = ndimage.sum(dilated, labeled, range(1, num + 1))
    largest = np.argmax(sizes) + 1
    mask = labeled == largest

    # Get precise outline within the component
    outline = dark & ndimage.binary_dilation(mask, iterations=1)
    connected = ndimage.binary_dilation(outline, iterations=connect_iter)
    filled = ndimage.binary_fill_holes(connected)

    # Build RGBA
    rgba = np.zeros((*crop.shape[:2], 4), dtype=np.uint8)
    rgba[filled, :3] = crop[filled, :3]
    rgba[filled, 3] = 255

    # Trim to bounding box
    ys, xs = np.where(rgba[:, :, 3] > 0)
    if len(ys) == 0:
        return Image.fromarray(rgba)
    trimmed = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(trimmed)


def extract_sparkle(arr, region):
    """Extract yellow sparkle pixels from a region."""
    y0, y1, x0, x1 = region
    crop = arr[y0:y1, x0:x1]

    r, g, b = crop[:, :, 0].astype(int), crop[:, :, 1].astype(int), crop[:, :, 2].astype(int)
    yellow = (r > 170) & (g > 130) & (b < 120)
    not_gray = (np.abs(r - g) > 10) | (np.abs(g - b) > 15)
    mask = yellow & not_gray

    rgba = np.zeros((*crop.shape[:2], 4), dtype=np.uint8)
    rgba[mask, :3] = crop[mask, :3]
    rgba[mask, 3] = 255

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return Image.fromarray(rgba)
    trimmed = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(trimmed)


def main():
    src = os.path.join(ASSETS, "decomposed.png")
    print(f"Loading {src} ...")
    arr = np.array(Image.open(src).convert("RGBA"))

    # --- Arrow cursor (bottom-right of decomposed image) ---
    print("Extracting cursor...")
    cursor = extract_element(arr, region=(555, 870, 755, 955))
    cursor.save(os.path.join(ASSETS, "cursor.png"))
    print(f"  → cursor.png  {cursor.size}")

    # --- Click sparkle (near cursor tip) ---
    print("Extracting sparkle...")
    sparkle = extract_sparkle(arr, region=(555, 870, 755, 955))
    sparkle.save(os.path.join(ASSETS, "sparkle.png"))
    print(f"  → sparkle.png  {sparkle.size}")

    # --- Paper airplane (top-right of decomposed image) ---
    print("Extracting airplane...")
    airplane = extract_element(arr, region=(150, 490, 680, 990), dark_thresh=140, dilate_iter=3, connect_iter=2)
    airplane.save(os.path.join(ASSETS, "airplane.png"))
    print(f"  → airplane.png  {airplane.size}")

    print("Done!")


if __name__ == "__main__":
    main()
