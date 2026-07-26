"""
sample.py — Generate roofs with the trained diffusion model and plug
them straight into the reconstruction pipeline.

    python -m ml_roof_diffusion.sample --ckpt ckpt/roofdiff_last.pt --n 4
    python -m ml_roof_diffusion.sample --ckpt ckpt/roofdiff_last.pt \
           --n 4 --plan rectangle --steps 100

or from the pipeline:

    from ml_roof_diffusion.sample import generate_roof_diffusion
    roof_meshes = generate_roof_diffusion(walls, "ckpt/roofdiff_last.pt",
                                          n=1, seed=7)

Notes:
  · New checkpoints store {"model", "ema"}; sampling uses the EMA weights
    (much cleaner). Old raw-state-dict checkpoints still load.
  · The raw heightmap gets a 3×3 median filter (removes residual diffusion
    speckle) before mesh_from_heightmap's gaussian smoothing.
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .dataset import RES, denormalize
from .model import RoofUNet, RoofDiffusion


def load_net(ckpt, use_ema=True):
    import torch
    net = RoofUNet()
    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        sd = obj.get("ema", obj["model"]) if use_ema else obj["model"]
        which = "ema" if (use_ema and "ema" in obj) else "model"
        print(f"loaded {ckpt} [{which} weights, epoch {obj.get('epoch', '?')}]")
    else:
        sd = obj
        print(f"loaded {ckpt} [legacy raw state_dict]")
    net.load_state_dict(sd)
    net.eval()
    return net


def footprint_mask_from_walls(walls, scale=0.01, res=RES, pad=1.5):
    from roof_generator import extract_footprint
    from matplotlib.path import Path
    poly = extract_footprint(walls, scale)
    x0, y0, x1, y1 = poly.bounds
    bounds = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    xs = np.linspace(bounds[0], bounds[2], res)
    ys = np.linspace(bounds[1], bounds[3], res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mask = Path(np.asarray(poly.exterior.coords)).contains_points(pts)
    return mask.reshape(res, res).astype(np.float32), bounds


def generate_roof_diffusion(walls, ckpt, n=1, seed=None, steps=80,
                            wall_h=3.0, scale=0.01, material=None,
                            use_ema=True):
    """Returns a list of n roof meshes (one per sample) for the given
    walls — the learned counterpart of roof_generator.generate_roof."""
    import torch
    from scipy.ndimage import median_filter
    from roof_generator import mesh_from_heightmap, ROOF_MATERIALS

    mask, bounds = footprint_mask_from_walls(walls, scale)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_net(ckpt, use_ema)
    diff = RoofDiffusion(net, device=dev)

    m = torch.from_numpy(mask * 2 - 1)[None, None].to(dev)
    meshes = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        s = int(rng.integers(0, 2**31)) if seed is None else seed + i
        x0 = diff.sample(m, steps=steps, seed=s)
        hm = denormalize(x0[0, 0].cpu().numpy())
        hm = median_filter(hm, size=3)            # de-speckle
        hm[mask < 0.5] = 0.0
        color = ROOF_MATERIALS[material] if material else None
        meshes.append(mesh_from_heightmap(hm, mask, bounds,
                                          base_z=wall_h, color=color))
    return meshes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--plan", default="L-shape",
                    help="rectangle | square | L-shape | T-shape")
    ap.add_argument("--raw", action="store_true",
                    help="sample from raw (non-EMA) weights")
    ap.add_argument("--out", default="diffusion_roofs")
    a = ap.parse_args()

    from demo_roofs import PLANS
    os.makedirs(a.out, exist_ok=True)
    walls = PLANS[a.plan]
    for i, mesh in enumerate(generate_roof_diffusion(
            walls, a.ckpt, n=a.n, seed=a.seed, steps=a.steps,
            use_ema=not a.raw)):
        mesh.export(os.path.join(a.out, f"roof_{i}.glb"))
        print(f"→ {a.out}/roof_{i}.glb")


if __name__ == "__main__":
    main()
