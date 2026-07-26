"""
kb_mining.py — Grow the RAG knowledge base from real roof datasets
==================================================================
Turns roof-geometry datasets into KNOWLEDGE_BASE exemplars so retrieval
is grounded in measured distributions instead of only hand-authored
recipes. Supported sources:

  · Houses3K            — modular house meshes grouped by style folders
  · Bonn Roof Geometry  — meshes organised by roof-type class folders
                          (gabled, flat, skillion, hipped, gambrel,
                          pyramidal, ...)
  · Zeitview ZRG        — per-property 3D rooftop wireframe annotations
                          (JSON with vertices + faces); plane pitches and
                          face counts are measured per building

Usage
-----
    python kb_mining.py --bonn path/to/BonnRoofs --zrg path/to/zrg \
                        --houses3k path/to/Houses3K --out roof_kb_mined.json

`roof_knowledge.py` auto-loads `roof_kb_mined.json` from the project
root at import time, appending the mined exemplars to KNOWLEDGE_BASE —
no other code changes needed. Delete the JSON to fall back to the
curated 50.

The miner produces STATISTICS-BACKED exemplars: each dataset yields
(style, frequency weight, pitch percentile range) tuples which are
instantiated with architecturally sane fit envelopes. Class counts set
retrieval weights (log-damped so a 10 000-gable dataset doesn't drown
every other style).
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

# dataset label → engine style vocabulary
STYLE_MAP = {
    "gable": "gable", "gabled": "gable", "gable_roof": "gable",
    "cross_gable": "gable", "crossgable": "gable",
    "hip": "hip", "hipped": "hip", "hip_roof": "hip",
    "half_hip": "dutch-gable", "half-hipped": "dutch-gable",
    "dutch": "dutch-gable", "dutch_gable": "dutch-gable",
    "flat": "flat", "flat_roof": "flat",
    "skillion": "skillion", "shed": "skillion", "mono": "skillion",
    "monopitch": "skillion", "lean_to": "skillion",
    "gambrel": "gambrel", "barn": "gambrel",
    "mansard": "mansard",
    "pyramid": "pyramid", "pyramidal": "pyramid", "tent": "pyramid",
    "butterfly": "butterfly",
    "saltbox": "saltbox",
}

# sane default envelopes per style when the dataset gives geometry-less
# class labels (folder scans); pitch in degrees
STYLE_DEFAULTS = {
    "gable":       dict(aspect=(1.2, 3.2), convexity=(0.86, 1.0), pitch=(24, 42)),
    "hip":         dict(aspect=(1.0, 2.6), convexity=(0.6, 1.0),  pitch=(20, 32)),
    "dutch-gable": dict(aspect=(1.2, 2.4), convexity=(0.86, 1.0), pitch=(24, 36)),
    "flat":        dict(aspect=(1.0, 2.4), convexity=(0.7, 1.0),  pitch=(0, 0)),
    "skillion":    dict(aspect=(1.2, 3.6), convexity=(0.86, 1.0), pitch=(6, 18)),
    "gambrel":     dict(aspect=(1.4, 3.0), convexity=(0.88, 1.0), pitch=(30, 40)),
    "mansard":     dict(aspect=(1.0, 1.9), convexity=(0.85, 1.0), pitch=(60, 72)),
    "pyramid":     dict(aspect=(1.0, 1.35), convexity=(0.9, 1.0), pitch=(24, 38)),
    "butterfly":   dict(aspect=(1.0, 2.4), convexity=(0.88, 1.0), pitch=(7, 14)),
    "saltbox":     dict(aspect=(1.3, 2.8), convexity=(0.88, 1.0), pitch=(28, 42)),
}

MESH_EXT = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}


def _norm_label(name):
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    for token in sorted(STYLE_MAP, key=len, reverse=True):
        if token in key:
            return STYLE_MAP[token]
    return None


# ────────────────────────────────────────────────────────────────────
#  Folder-organised mesh datasets (Bonn, Houses3K)
# ────────────────────────────────────────────────────────────────────
def scan_class_folders(root):
    """Count mesh files under each class-named subfolder →
    {engine_style: count}. Unrecognised folder names are skipped."""
    counts = Counter()
    if not os.path.isdir(root):
        return counts
    for dirpath, _dirs, files in os.walk(root):
        style = _norm_label(os.path.basename(dirpath))
        if style is None:
            continue
        n = sum(1 for f in files if os.path.splitext(f)[1].lower() in MESH_EXT)
        if n:
            counts[style] += n
    return counts


# ────────────────────────────────────────────────────────────────────
#  ZRG-style wireframe JSONs: measure pitch + classify by plane count
# ────────────────────────────────────────────────────────────────────
def _wireframe_stats(data):
    """One wireframe {vertices, faces} → (style_guess, median_pitch_deg).
    Tolerant to key naming; returns None when unusable."""
    verts = data.get("vertices") or data.get("verts") or data.get("points")
    faces = data.get("faces") or data.get("planes") or data.get("polygons")
    if not verts or not faces:
        return None
    V = np.asarray(verts, dtype=float)
    if V.ndim != 2 or V.shape[1] < 3:
        return None
    pitches = []
    for f in faces:
        idx = [i for i in f if isinstance(i, (int, np.integer)) and i < len(V)]
        if len(idx) < 3:
            continue
        p = V[idx]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        cos_h = abs(n[2]) / nn            # angle of plane vs horizontal
        pitch = np.degrees(np.arccos(np.clip(cos_h, 0, 1)))
        if pitch < 75:                    # ignore near-vertical walls
            pitches.append(pitch)
    if not pitches:
        return None
    med = float(np.median(pitches))
    sloped = [p for p in pitches if p > 4]
    if not sloped:
        return "flat", med
    n_sloped = len(sloped)
    if n_sloped == 1:
        return "skillion", med
    if n_sloped == 2:
        return "gable", med
    if n_sloped >= 4 and med < 40:
        return "hip", med
    return "gable", med


def scan_zrg(root, limit=None):
    """{style: count}, {style: [pitches]} from a folder of wireframe
    JSON/GeoJSON files."""
    counts, pitches = Counter(), defaultdict(list)
    if not os.path.isdir(root):
        return counts, pitches
    n_seen = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".json", ".geojson")):
                continue
            try:
                with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            st = _wireframe_stats(data)
            if st is None:
                continue
            style, pitch = st
            counts[style] += 1
            pitches[style].append(pitch)
            n_seen += 1
            if limit and n_seen >= limit:
                return counts, pitches
    return counts, pitches


# ────────────────────────────────────────────────────────────────────
#  Stats → exemplars
# ────────────────────────────────────────────────────────────────────
def exemplars_from_stats(counts, source, pitches=None, base_weight=1.0):
    """Instantiate one full-footprint exemplar per observed style.
    Weight ∝ log-damped frequency; pitch range from measured 20–80th
    percentiles when available, else style defaults."""
    out = []
    total = sum(counts.values())
    if not total:
        return out
    for style, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        d = STYLE_DEFAULTS.get(style)
        if d is None:
            continue
        if pitches and pitches.get(style) and len(pitches[style]) >= 5:
            arr = np.asarray(pitches[style])
            plo, phi = np.percentile(arr, [20, 80])
            pitch = (round(float(max(plo, 4)), 1), round(float(max(phi, plo + 3)), 1))
        else:
            pitch = d["pitch"]
        w = base_weight * (0.5 + 0.5 * np.log1p(n) / np.log1p(total))
        zone = dict(region=dict(type="full"), style=style)
        if style == "flat":
            zone["parapet"] = True
        else:
            zone["pitch"] = pitch
        tags = ["mined", source]
        tags.append("modern" if style in ("flat", "skillion", "butterfly")
                    else "traditional")
        out.append(dict(
            id=f"{source}_{style}",
            name=f"{style} ({source}, n={n})",
            tags=tags, weight=round(float(w), 2),
            fit=dict(aspect=d["aspect"], convexity=d["convexity"],
                     area=(35, 450)),
            zones=[zone],
            materials=None,        # filled by STYLE_MATERIALS at build time
            features=dict(),
            provenance=dict(source=source, count=int(n)),
        ))
    return out


def mine(bonn=None, houses3k=None, zrg=None, zrg_limit=None):
    exemplars = []
    if bonn:
        c = scan_class_folders(bonn)
        print(f"Bonn: {dict(c)}")
        exemplars += exemplars_from_stats(c, "bonn")
    if houses3k:
        c = scan_class_folders(houses3k)
        print(f"Houses3K: {dict(c)}")
        exemplars += exemplars_from_stats(c, "houses3k")
    if zrg:
        c, p = scan_zrg(zrg, limit=zrg_limit)
        print(f"ZRG: {dict(c)} (pitch-measured)")
        exemplars += exemplars_from_stats(c, "zrg", pitches=p)
    return exemplars


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bonn", help="Bonn Roof Geometry root folder")
    ap.add_argument("--houses3k", help="Houses3K root folder")
    ap.add_argument("--zrg", help="ZRG wireframe folder")
    ap.add_argument("--zrg-limit", type=int, default=None,
                    help="max wireframes to scan (ZRG is 20k+ properties)")
    ap.add_argument("--out", default="roof_kb_mined.json")
    args = ap.parse_args()
    ex = mine(args.bonn, args.houses3k, args.zrg, args.zrg_limit)
    if not ex:
        print("No exemplars mined — check dataset paths.")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(ex, f, indent=1)
        print(f"✅ {len(ex)} mined exemplars → {args.out} "
              f"(auto-loaded by roof_knowledge.py)")
