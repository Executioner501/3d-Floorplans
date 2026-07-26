# ml_roof_diffusion — Learned roof generator (footprint-conditioned diffusion)

A dedicated generative model that takes the reconstructed **roofless house's
footprint** and synthesizes a roof **heightmap**, converted to a mesh and
merged into the pipeline. Same architecture family as RoofDiffusion
(ECCV 2024): conditional DDPM over 128×128 heightmaps, footprint mask
injected at every UNet scale, DDIM sampling. Different noise seeds on the
same footprint → different realistic roofs.

## Why heightmaps
A roof is (almost always) a height function over the footprint. Diffusion on
this 2.5D representation is dramatically easier to train than mesh/point-cloud
generation, is footprint-exact by construction, and converts losslessly to a
watertight mesh (`roof_generator.mesh_from_heightmap`).

## Data (curriculum)

| Stage | Source | How |
|---|---|---|
| Pretrain | **Synthetic** — this repo's own procedural engine renders unlimited (mask, heightmap) pairs across all 26 knowledge-base compositions | `python -m ml_roof_diffusion.dataset --pregen 20000` |
| Finetune | **PoznanRD** — 13k real complex roofs released with RoofDiffusion | download from github.com/kylelo/RoofDiffusion, use `--data poznan --root ...` |
| Augment | **SYNBUILD-3D** roof point clouds (6.2M buildings, github.com/kdmayer/SYNBUILD-3D) | `dataset.synbuild_to_heightmap(points)`; note their roofs are straight-skeleton-generated, so treat as scale augmentation |

## Train (single consumer GPU)

```bash
pip install torch pillow scipy tqdm
python -m ml_roof_diffusion.dataset --pregen 20000 --out synthetic_cache --workers 8
# stage 1 — synthetic pretrain. Run the FULL 60 epochs: the loss plateaus
# after ~4 epochs but sample quality keeps improving for tens of epochs.
python -m ml_roof_diffusion.train --data synthetic --cache synthetic_cache \
       --epochs 60 --bs 128 --lr 2e-4
# stage 2 — finetune on real roofs MIXED with synthetic (prevents the
# model from collapsing onto noisy LiDAR texture / forgetting clean roofs)
python -m ml_roof_diffusion.train --data mix --cache synthetic_cache \
       --root datasets/PoznanRD --resume ckpt/roofdiff_last.pt \
       --lr 2e-5 --epochs 30 --bs 128
# evaluate (samples from the EMA weights; --plan rectangle|square|L-shape|T-shape)
python -m ml_roof_diffusion.sample --ckpt ckpt/roofdiff_last.pt --n 4
```

Checkpoints are `{"model", "ema", "epoch"}` dicts; sampling uses the EMA
weights (dramatically cleaner than raw online weights). Old raw-state-dict
checkpoints still `--resume` and sample fine.

## Use in the pipeline

Set `ENGINE = "diffusion"` in `main.py`, or:

```python
from ml_roof_diffusion.sample import generate_roof_diffusion
roofs = generate_roof_diffusion(walls, "ckpt/roofdiff_last.pt", n=4, seed=0)
```

## Shortcut: pretrained weights
RoofDiffusion publishes pretrained footprint-conditioned weights
(`pretrained/w_footprint/260_Network.pth` in their repo). Their network class
differs from ours, but their inference script can be driven with the footprint
mask produced by `sample.footprint_mask_from_walls` (feed an empty/sparse
height hint), then pass the resulting heightmap into
`roof_generator.mesh_from_heightmap` — a zero-training path to a learned roof
generator.
