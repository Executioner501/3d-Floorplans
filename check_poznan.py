"""
check_poznan.py — Verify a freshly-unzipped PoznanRD download before training.

    python check_poznan.py datasets/PoznanRD

Prints the detected layout, pair count, and decodes a few samples to confirm
the uint16 height encoding (metres = px / 256) and footprint masks look sane.
Exit code 0 = ready to train.
"""
import os
import sys
import glob
import numpy as np


def main(root):
    print(f"Inspecting: {os.path.abspath(root)}\n")
    if not os.path.isdir(root):
        print("!! Not a directory."); return 1

    # Show top 2 levels so the layout is obvious
    print("Top-level contents:")
    for p in sorted(glob.glob(os.path.join(root, "*")))[:20]:
        tag = "/" if os.path.isdir(p) else ""
        print(f"  {os.path.basename(p)}{tag}")
        if os.path.isdir(p):
            for q in sorted(glob.glob(os.path.join(p, "*")))[:6]:
                print(f"      {os.path.basename(q)}{'/' if os.path.isdir(q) else ''}")
    print()

    flists = glob.glob(os.path.join(root, "*.flist"))
    if flists:
        print("flist files found:", [os.path.basename(f) for f in flists])
    gt_dirs = glob.glob(os.path.join(root, "**", "roof_gt"), recursive=True)
    if gt_dirs:
        print(f"roof_gt folders found: {len(gt_dirs)} "
              f"(e.g. {os.path.relpath(gt_dirs[0], root)})")
    print()

    from ml_roof_diffusion.dataset import PoznanRDDataset, H_MAX
    try:
        ds = PoznanRDDataset(root, preload=False)
    except AssertionError as e:
        print(f"!! {e}\n")
        # Diagnosis: dump what's actually inside any roof_img/footprint dirs
        for dirpath, dirnames, _ in os.walk(root):
            for dn in dirnames:
                if dn.lower() in ("roof_img", "footprint", "roof_gt",
                                  "roof_footprint", "img"):
                    d = os.path.join(dirpath, dn)
                    entries = sorted(os.listdir(d))
                    n_img = len(PoznanRDDataset._imgs_under(d))
                    print(f"  {os.path.relpath(d, root)}: {len(entries)} "
                          f"entries, {n_img} image files (recursive). "
                          f"First entries: {entries[:5]}")
        print("\nPaste the lines above back to diagnose the exact layout.")
        return 1

    from PIL import Image
    idx = np.linspace(0, len(ds) - 1, min(5, len(ds)), dtype=int)
    over = 0
    for i in idx:
        raw = np.asarray(Image.open(ds.gt[i]))
        fp = np.asarray(Image.open(ds.fp[i])) > 0
        h = raw.astype(np.float32) / 256.0
        hin = h[fp] if fp.any() else h
        rise = float(hin.max() - np.percentile(hin, 2)) if hin.size else 0.0
        if rise > H_MAX:
            over += 1
        print(f"  [{i:5d}] {os.path.basename(ds.gt[i]):<24} {raw.shape} "
              f"dtype={raw.dtype}  roof rise ≈ {rise:5.2f} m  "
              f"footprint {100 * fp.mean():4.1f}% of image")
        x = ds[int(i)]
        assert x.shape == (2, 128, 128), "unexpected tensor shape"

    print(f"\nOK: {len(ds)} training pairs, tensors (2,128,128) float32.")
    if over:
        print(f"NOTE: {over}/{len(idx)} sampled roofs rise above H_MAX={H_MAX} m "
              f"and will be clipped; that's fine for typical houses, but if "
              f"most samples exceed it, raise H_MAX in ml_roof_diffusion/"
              f"dataset.py (train + sample use the same constant).")
    print("Ready:  python -m ml_roof_diffusion.train --data poznan "
          f"--root {root} --resume ckpt/roofdiff_last.pt --lr 2e-5 --epochs 30")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "datasets/PoznanRD"))