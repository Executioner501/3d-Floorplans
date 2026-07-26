"""
dataset.py — Training data for footprint-conditioned roof diffusion.

Sources (curriculum: pretrain on 1, finetune on 2 or on the mix):

1. SyntheticRoofDataset — domain-randomized pairs from this project's own
   procedural engine. With a pregenerated .npz cache it PRELOADS the whole
   cache to RAM (uint8 masks + float16 heightmaps, ~1 GB for 20k) so an
   epoch costs zero disk I/O.

2. PoznanRDDataset — 13k REAL roofs from RoofDiffusion (ECCV 2024).
   Height PNGs: uint16, metres = px / 256. Preprocessing per roof:
     · rebase heights to the 2nd percentile inside the footprint (eave≈0)
     · ZERO everything outside the footprint (kills neighbouring
       buildings / trees leaking into training targets)
     · 3×3 median filter (kills LiDAR pit/spike noise)
     · downsample to RES, re-mask
   With preload=True (default) all pairs are decoded ONCE (multiprocess)
   and kept in RAM (~650 MB) — afterwards epochs are I/O-free.

3. MixedRoofDataset — concat of the two, for forgetting-free finetuning.

All samples are (2, R, R) float32 tensors in [-1, 1]:
   channel 0 = heightmap  (metres above eave, normalized by H_MAX)
   channel 1 = footprint mask (−1 outside, +1 inside)
"""
import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H_MAX = 8.0   # metres of roof rise mapped to [-1, 1]
RES   = 128


def normalize(hm):
    return np.clip(hm / H_MAX, 0, 1) * 2 - 1


def denormalize(x):
    return (np.clip(x, -1, 1) + 1) / 2 * H_MAX


# ────────────────────────────────────────────────────────────────────
#  1. SYNTHETIC (procedural-engine) SOURCE
# ────────────────────────────────────────────────────────────────────
def _rand_plan(rng):
    """Random rect/L/T footprint expressed as the pipeline's wall dicts."""
    from demo_roofs import rect_plan, l_plan, t_plan, square_plan
    kind = rng.choice(["rect", "square", "L", "T"], p=[0.35, 0.15, 0.3, 0.2])
    if kind == "rect":
        return rect_plan(W=int(rng.uniform(1000, 1900)),
                         D=int(rng.uniform(600, 1100)))
    if kind == "square":
        return square_plan(S=int(rng.uniform(700, 1200)))
    if kind == "L":
        W = int(rng.uniform(1200, 1900)); D = int(rng.uniform(800, 1300))
        return l_plan(W=W, D=D, cut_w=int(W * rng.uniform(0.3, 0.55)),
                      cut_d=int(D * rng.uniform(0.3, 0.55)))
    W = int(rng.uniform(1200, 1900)); D = int(rng.uniform(500, 800))
    return t_plan(W=W, D=D, stem_w=int(W * rng.uniform(0.28, 0.45)),
                  stem_d=int(rng.uniform(350, 650)))


def synth_sample(seed, res=RES, style_pref="mixed", pad=1.5):
    """One (mask, heightmap) training pair from the procedural engine."""
    from roof_ai import design_roof_rag
    from roof_generator import (generate_roof, extract_footprint,
                                rasterize_roof_heightmap)
    from matplotlib.path import Path

    rng = np.random.default_rng(seed)
    walls = _rand_plan(rng)
    params = design_roof_rag(walls, seed=seed, style_pref=style_pref,
                             temperature=1.2)          # extra-diverse
    params["has_dormers"] = params["has_chimney"] = False   # roof surface only
    params["has_skylight"] = False
    meshes, _ = generate_roof(walls, params, wall_h=0.0)   # heights rel. wall top

    poly = extract_footprint(walls)
    x0, y0, x1, y1 = poly.bounds
    bounds = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    hm = rasterize_roof_heightmap(meshes, bounds, res=res)

    xs = np.linspace(bounds[0], bounds[2], res)
    ys = np.linspace(bounds[1], bounds[3], res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mask = Path(np.asarray(poly.exterior.coords)).contains_points(pts)
    mask = mask.reshape(res, res).astype(np.float32)
    return mask, hm.astype(np.float32), params["recipe_id"]


def _pregen_one(args):
    seed, out_dir, res = args
    m, h, rid = synth_sample(seed, res)
    np.savez_compressed(os.path.join(out_dir, f"{seed:07d}.npz"),
                        mask=m, hm=h, recipe=rid)
    return seed


def pregenerate(n, out_dir="synthetic_cache", res=RES, start_seed=0, workers=1):
    """Cache N samples to .npz so training doesn't pay mesh+raster cost.
    workers>1 parallelizes across CPU cores (recommended: os.cpu_count())."""
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(start_seed + i, out_dir, res) for i in range(n)]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(_pregen_one, jobs,
                                                      chunksize=16)):
                if k % 200 == 0:
                    print(f"  {k}/{n}")
    else:
        for k, j in enumerate(jobs):
            _pregen_one(j)
            if k % 50 == 0:
                print(f"  {k}/{n}")
    print(f"cached {n} samples → {out_dir}/")


class SyntheticRoofDataset:
    """torch-style Dataset over the pregenerated cache.

    preload=True (default): decompress the WHOLE cache into RAM once
    (~1 GB for 20k @128²: uint8 masks + float16 heightmaps). Per-item
    cost drops from ~150 ms (npz inflate) to ~0.2 ms → use num_workers=0.
    """
    def __init__(self, size=10000, res=RES, cache_dir=None,
                 style_pref="mixed", preload=True):
        self.size, self.res, self.style_pref = size, res, style_pref
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.npz"))) \
            if cache_dir else []
        self.masks = self.hms = None
        if self.files and preload:
            n = len(self.files)
            print(f"preloading {n} synthetic samples to RAM ...", flush=True)
            z0 = np.load(self.files[0]); r = z0["mask"].shape[0]
            self.masks = np.empty((n, r, r), np.uint8)
            self.hms = np.empty((n, r, r), np.float16)
            for i, f in enumerate(self.files):
                z = np.load(f)
                self.masks[i] = z["mask"] > 0.5
                self.hms[i] = z["hm"]
                if i % 2000 == 0:
                    print(f"  {i}/{n}", flush=True)
            print(f"preloaded ({self.masks.nbytes + self.hms.nbytes >> 20} MB)")

    @property
    def ram_resident(self):
        return self.masks is not None

    def __len__(self):
        return len(self.files) or self.size

    def __getitem__(self, i):
        import torch
        if self.masks is not None:
            mask = self.masks[i].astype(np.float32)
            hm = self.hms[i].astype(np.float32)
        elif self.files:
            z = np.load(self.files[i % len(self.files)])
            mask, hm = z["mask"], z["hm"]
        else:
            mask, hm, _ = synth_sample(i, self.res, self.style_pref)
        x = np.stack([normalize(hm), mask * 2 - 1]).astype(np.float32)
        return torch.from_numpy(x)


# ────────────────────────────────────────────────────────────────────
#  2. PoznanRD (RoofDiffusion release) — REAL roofs
# ────────────────────────────────────────────────────────────────────
def _poznan_load_pair(args):
    """Decode + clean ONE real roof (top-level for multiprocessing).
    Returns (hm float16 RES², mask bool RES², rise_p98_metres)."""
    gt_path, fp_path, res = args
    from PIL import Image
    from scipy.ndimage import median_filter
    hm = np.asarray(Image.open(gt_path), dtype=np.float32) / 256.0
    fp_img = Image.open(fp_path)
    if fp_img.size != (hm.shape[1], hm.shape[0]):
        fp_img = fp_img.resize((hm.shape[1], hm.shape[0]), Image.NEAREST)
    fp = np.asarray(fp_img)
    if fp.ndim == 3:
        fp = fp[..., 0]
    fp = fp > 0
    if fp.any():
        hm -= np.percentile(hm[fp], 2)           # eave → 0
    hm = np.clip(hm, 0, None)
    hm[~fp] = 0.0                                # kill off-footprint clutter
    hm = median_filter(hm, size=3)               # kill LiDAR pits/spikes
    hm_i = np.array(Image.fromarray(hm).resize((res, res), Image.BILINEAR),
                    dtype=np.float32)
    fp_i = np.asarray(Image.fromarray(fp.astype(np.uint8) * 255)
                      .resize((res, res), Image.NEAREST)) > 127
    hm_i[~fp_i] = 0.0
    rise = float(np.percentile(hm_i[fp_i], 98)) if fp_i.any() else 0.0
    return hm_i.astype(np.float16), fp_i, rise


class PoznanRDDataset:
    """Real roofs from the RoofDiffusion (ECCV 2024) release.
    Height PNGs are uint16, metres = px / 256.

    preload=True (default): every pair is decoded + cleaned ONCE (optionally
    in parallel with `workers`) and held in RAM (~650 MB for 13k @128²);
    training then reads pure numpy → use num_workers=0.

    Auto-detects any of these layouts under --root (priority order):
      C) <root>/*img*.flist + <root>/*footprint*.flist
      D) roof_img/ + footprint/ sibling dirs at root or one level down
         (recursive inside, any raster extension, stem-normalized pairing)
      A) <root>/roof_gt/*.png + <root>/roof_footprint/*.png
      B) <root>/**/roof_gt/*.png
    """
    def __init__(self, root, res=RES, preload=True, workers=0):
        self.res = res
        self.gt, self.fp, mode = self._discover(root)
        pairs = [(g, f) for g, f in zip(self.gt, self.fp) if os.path.isfile(f)]
        dropped = len(self.gt) - len(pairs)
        self.gt = [g for g, _ in pairs]
        self.fp = [f for _, f in pairs]
        assert self.gt, (
            f"No PoznanRD samples found under {root}. Looked for "
            f"(1) *img*.flist pairs, (2) roof_img/ + footprint/ siblings "
            f"(recursive, any image extension), (3) roof_gt/ + "
            f"roof_footprint/ pairs. Run `python check_poznan.py {root}` "
            f"for a folder-by-folder diagnosis.")
        print(f"PoznanRDDataset: {len(self.gt)} pairs via {mode}"
              + (f" ({dropped} gt files without matching footprint skipped)"
                 if dropped else ""))
        self.hms = self.masks = None
        if preload:
            self._preload(workers)

    def _preload(self, workers):
        n = len(self.gt)
        print(f"preloading {n} real roofs to RAM "
              f"({'%d workers' % workers if workers > 1 else 'single process'})"
              " ...", flush=True)
        jobs = [(g, f, self.res) for g, f in zip(self.gt, self.fp)]
        self.hms = np.empty((n, self.res, self.res), np.float16)
        self.masks = np.empty((n, self.res, self.res), bool)
        rises = np.empty(n, np.float32)
        if workers > 1:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                for i, (h, m, r) in enumerate(
                        pool.imap(_poznan_load_pair, jobs, chunksize=32)):
                    self.hms[i], self.masks[i], rises[i] = h, m, r
                    if i % 1000 == 0:
                        print(f"  {i}/{n}", flush=True)
        else:
            for i, j in enumerate(jobs):
                self.hms[i], self.masks[i], rises[i] = _poznan_load_pair(j)
                if i % 1000 == 0:
                    print(f"  {i}/{n}", flush=True)
        over = int((rises > H_MAX).sum())
        print(f"preloaded ({(self.hms.nbytes + self.masks.nbytes) >> 20} MB). "
              f"roof rise p50={np.median(rises):.1f} m  p98="
              f"{np.percentile(rises, 98):.1f} m; {over}/{n} exceed "
              f"H_MAX={H_MAX} m" + (" — consider raising H_MAX!"
                                    if over > 0.1 * n else ""))

    @property
    def ram_resident(self):
        return self.hms is not None

    @staticmethod
    def _read_flist(path):
        base = os.path.dirname(os.path.abspath(path))
        out = []
        with open(path) as f:
            for line in f:
                p = line.strip().replace("\\", os.sep)
                if not p:
                    continue
                q = p[2:] if p.startswith("./") else p
                cands = [p, os.path.join(base, q)]
                for pref in ("dataset/PoznanRD/", "PoznanRD/", "dataset/"):
                    if q.startswith(pref):
                        cands.append(os.path.join(base, q[len(pref):]))
                for c in cands:
                    if os.path.isfile(c):
                        out.append(c)
                        break
        return out

    IMG_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

    @classmethod
    def _imgs_under(cls, d):
        out = []
        for dirpath, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(cls.IMG_EXTS):
                    out.append(os.path.join(dirpath, f))
        return sorted(out)

    @classmethod
    def _stem_key(cls, path):
        s = os.path.splitext(os.path.basename(path))[0].lower()
        for suf in ("_footprint", "_fp", "_mask", "_roof", "_img",
                    "_height", "_gt"):
            if s.endswith(suf):
                s = s[: -len(suf)]
        return s

    @classmethod
    def _discover(cls, root):
        img_fl = sorted(glob.glob(os.path.join(root, "*img*.flist")))
        fp_fl = sorted(glob.glob(os.path.join(root, "*footprint*.flist")))
        pick = lambda ls: ([p for p in ls if "train" in os.path.basename(p)]
                           or ls)[0] if ls else None
        fi, ff = pick(img_fl), pick(fp_fl)
        if fi and ff:
            gt, fp = cls._read_flist(fi), cls._read_flist(ff)
            if gt and len(gt) == len(fp):
                return gt, fp, f"flist ({os.path.basename(fi)})"
        for cand in [root] + sorted(p for p in glob.glob(os.path.join(root, "*"))
                                    if os.path.isdir(p)):
            ri = os.path.join(cand, "roof_img")
            fd = os.path.join(cand, "footprint")
            if os.path.isdir(ri) and os.path.isdir(fd):
                gt = cls._imgs_under(ri)
                fpm = {}
                for q in cls._imgs_under(fd):
                    fpm.setdefault(cls._stem_key(q), q)
                pairs = [(g, fpm[cls._stem_key(g)]) for g in gt
                         if cls._stem_key(g) in fpm]
                rel = os.path.relpath(cand, root)
                if pairs:
                    return ([a for a, _ in pairs], [b for _, b in pairs],
                            f"roof_img/ + footprint/ under '{rel}'")
                print(f"  note: found {ri} ({len(gt)} images) and {fd} "
                      f"({len(fpm)} images) but 0 filename matches")
        gt = sorted(glob.glob(os.path.join(root, "roof_gt", "*.png")))
        if gt:
            return (gt, [p.replace("roof_gt", "roof_footprint") for p in gt],
                    "roof_gt/ + roof_footprint/")
        gt = sorted(glob.glob(os.path.join(root, "**", "roof_gt", "*.png"),
                              recursive=True))
        sep = os.sep
        fp = [p.replace(f"{sep}roof_gt{sep}", f"{sep}roof_footprint{sep}")
              for p in gt]
        return gt, fp, "nested roof_gt/ folders"

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, i):
        import torch
        if self.hms is not None:
            hm = self.hms[i].astype(np.float32)
            fp = self.masks[i]
        else:
            hm16, fp, _ = _poznan_load_pair((self.gt[i], self.fp[i], self.res))
            hm = hm16.astype(np.float32)
        x = np.stack([normalize(hm), fp.astype(np.float32) * 2 - 1])
        return torch.from_numpy(x.astype(np.float32))


# ────────────────────────────────────────────────────────────────────
#  MIX — synthetic + real in one epoch (forgetting-free finetune)
# ────────────────────────────────────────────────────────────────────
class MixedRoofDataset:
    """Concatenation of any RAM-resident datasets (same tensor format)."""
    def __init__(self, *datasets):
        self.ds = [d for d in datasets if len(d)]
        self.cum = np.cumsum([len(d) for d in self.ds])
        print("MixedRoofDataset:", " + ".join(str(len(d)) for d in self.ds),
              f"= {self.cum[-1]} samples")

    @property
    def ram_resident(self):
        return all(getattr(d, "ram_resident", False) for d in self.ds)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.cum, i, side="right"))
        return self.ds[k][i - (int(self.cum[k - 1]) if k else 0)]


# ────────────────────────────────────────────────────────────────────
#  3. SYNBUILD-3D roof point clouds → heightmap
# ────────────────────────────────────────────────────────────────────
def synbuild_to_heightmap(points, res=RES, pad=1.5):
    """points: (N, 3) roof point cloud (Modality III). Splat max-z per
    cell; mask = occupied cells morphologically closed."""
    from scipy.ndimage import binary_closing, grey_closing
    p = np.asarray(points, dtype=float)
    z0 = np.percentile(p[:, 2], 2)
    x0, y0 = p[:, 0].min() - pad, p[:, 1].min() - pad
    x1, y1 = p[:, 0].max() + pad, p[:, 1].max() + pad
    hm = np.zeros((res, res), np.float32)
    ix = np.clip(((p[:, 0] - x0) / (x1 - x0) * (res - 1)).astype(int), 0, res - 1)
    iy = np.clip(((p[:, 1] - y0) / (y1 - y0) * (res - 1)).astype(int), 0, res - 1)
    np.maximum.at(hm, (iy, ix), p[:, 2] - z0)
    hm = grey_closing(hm, size=3)
    mask = binary_closing(hm > 0.05, iterations=2).astype(np.float32)
    return mask, hm


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregen", type=int, default=0)
    ap.add_argument("--out", default="synthetic_cache")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers for --pregen (try os.cpu_count())")
    a = ap.parse_args()
    if a.pregen:
        pregenerate(a.pregen, a.out, workers=a.workers)
    else:
        m, h, r = synth_sample(0)
        print("sample ok:", m.shape, h.shape, "recipe:", r,
              "max height %.2f m" % h.max())
