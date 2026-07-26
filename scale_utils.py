"""
scale_utils.py — Relative pixel→metre scale estimation  (fixes Issue A)
=======================================================================
The pipeline used to assume a fixed scale of 0.01 (1 px = 1 cm). Real
floor-plan images are drawn at arbitrary resolutions, so a 550 px plan
became a 3.3 m "building" with 3 m walls: tower-slab walls, roofs whose
knowledge-base exemplars all fail their `fit.area` envelope, and a
diffusion conditioning mask far outside the training distribution.

This module derives the scale PER PLAN, in priority order:

  1. OCR of printed dimension text (needs `pytesseract` + the Tesseract
     binary; entirely optional). Residential plans print their overall
     dimensions (36', 50'-0", 12.5m, 12000mm); the largest printed
     length is taken as the building's long side.
  2. Footprint normalisation: the longer side of the detected wall
     bounding box is mapped to `target_long_m` (default 14 m — the
     median long side of a small detached house, and the value that
     keeps the synthetic demo plans at exactly their historical 0.01).

The returned scale multiplies pixel coordinates everywhere downstream
(detect → roof engines → builder → diffusion mask). Wall height, door
heights and pitches stay absolute in metres.
"""
import re
import numpy as np

FT = 0.3048


def footprint_px_extent(walls):
    """Width/depth in pixels of the wall-box bounding rectangle."""
    if not walls:
        return 0.0, 0.0
    x0 = min(w["pos"][0] - w["w"] / 2 for w in walls)
    x1 = max(w["pos"][0] + w["w"] / 2 for w in walls)
    y0 = min(w["pos"][1] - w["h"] / 2 for w in walls)
    y1 = max(w["pos"][1] + w["h"] / 2 for w in walls)
    return x1 - x0, y1 - y0


def dimensions_from_image(image_path):
    """OCR printed dimension annotations → list of lengths in metres.
    Silently returns [] when OCR is unavailable or finds nothing, so the
    caller can fall back to footprint normalisation."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return []
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception:
        return []
    vals = []
    # imperial: 36'  36'-0"  36 ft  36ft-6in
    for m in re.finditer(
            r"(\d{1,3})\s*(?:'|ft)\s*(?:-?\s*(\d{1,2})\s*(?:\"|in))?", text):
        ft = int(m.group(1)) + int(m.group(2) or 0) / 12.0
        if 3 <= ft <= 120:
            vals.append(ft * FT)
    # metric metres: 12.5m  12,5 m
    for m in re.finditer(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*m\b", text, re.I):
        v = float(m.group(1).replace(",", "."))
        if 2 <= v <= 60:
            vals.append(v)
    # metric millimetres: 3600mm  12000 mm
    for m in re.finditer(r"(\d{4,5})\s*mm\b", text, re.I):
        v = int(m.group(1)) / 1000.0
        if 2 <= v <= 60:
            vals.append(v)
    return vals


def estimate_scale(walls, image_path=None, target_long_m=14.0,
                   clamp=(0.004, 0.08), verbose=True):
    """px→metre scale for THIS plan (replaces the fixed 0.01).

    `clamp` guards against OCR misreads and degenerate detections:
    0.004 ≈ a 3.5 kpx plan of a 14 m house, 0.08 ≈ a 175 px thumbnail.
    """
    w_px, d_px = footprint_px_extent(walls)
    long_px = max(w_px, d_px)
    if long_px <= 0:
        return 0.01
    long_m, source = target_long_m, "target-normalised"
    if image_path:
        dims = dimensions_from_image(image_path)
        if dims:
            long_m = max(dims)
            source = f"OCR of printed dimensions ({len(dims)} found)"
    s = float(np.clip(long_m / long_px, *clamp))
    if verbose:
        print(f"📏 Scale: {s * 100:.3f} cm/px [{source}] → footprint "
              f"≈ {w_px * s:.1f} × {d_px * s:.1f} m")
    return s
