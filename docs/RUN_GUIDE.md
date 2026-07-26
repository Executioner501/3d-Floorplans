# Run Guide — Construct-AI roof pipeline + diffusion training (RTX 5070 Ti)

Everything below is run **from this folder** (the project root, where this file
lives). Never `cd` into `ml_roof_diffusion/` — the `-m` module form below is
what makes imports and the `ckpt/` path work.

## What changed in this version
- **Fixed:** the 90° gable-wing rotation bug in `roof_generator.py`
  (`_place_wing_prism` inverted the prism's native ridge axis). All demo GLBs
  and gallery PNGs in this zip are regenerated with the fix.
- `ml_roof_diffusion/dataset.py`: `PoznanRDDataset` now auto-detects any
  PoznanRD layout (flist pair / `roof_gt`+`roof_footprint` flat or nested),
  and `--pregen` gained a `--workers` flag.
- New `check_poznan.py` verification script.
- `synthetic_cache/` ships **empty** — the old 20k cache was generated with
  the bug and is invalid. Step 3 regenerates it.

## 0. Environment (once)
Python 3.10–3.12. The RTX 5070 Ti is a Blackwell GPU (sm_120), so you need a
**CUDA 12.8 build of PyTorch** — older cu121/cu124 wheels will NOT run on it:

    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt

Verify the GPU is seen (must print `True ... RTX 5070 Ti`):

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

If it prints False or a kernel error mentions sm_120, your torch wheel is too
old — reinstall with the cu128 index URL above (torch >= 2.7).

## 1. Sanity-check the base pipeline (no GPU needed)

    python demo_roofs.py        # galleries + 3 demo GLBs (roofs correctly oriented)
    python main.py              # full floorplan -> GLB pipeline (RAG engine default)

## 2. Place the PoznanRD data
Extract your `PoznanRD.zip` (from the Google Drive link in the RoofDiffusion
README) into `datasets/PoznanRD/`, then verify:

    python check_poznan.py datasets/PoznanRD

It prints the detected layout, the number of training pairs (~13k expected),
and height sanity stats, ending with the exact training command. If it warns
that most roofs exceed `H_MAX = 8 m`, raise `H_MAX` in
`ml_roof_diffusion/dataset.py` BEFORE step 3 (train and sample share it).

## 3. Regenerate the synthetic cache (fixed engine, ~10–20 min)

    python -m ml_roof_diffusion.dataset --pregen 20000 --out synthetic_cache --workers 8

(Use your CPU core count for `--workers`.)

## 4. Stage 1 — pretrain on synthetic roofs (GPU, overnight-ish)

    python -m ml_roof_diffusion.train --data synthetic --cache synthetic_cache --epochs 60 --bs 128 --lr 2e-4

With 16 GB VRAM and a 4.2M-param UNet at 128×128, batch 128 fits easily
(bs 32 @ lr 1e-4 is the conservative fallback). Checkpoints land in
`ckpt/roofdiff_last.pt` (+ periodic epoch snapshots).

## 5. Stage 2 — finetune on real PoznanRD roofs

    python -m ml_roof_diffusion.train --data poznan --root datasets/PoznanRD --resume ckpt/roofdiff_last.pt --lr 2e-5 --epochs 30 --bs 128

## 6. Use the trained model

    python -m ml_roof_diffusion.sample --ckpt ckpt/roofdiff_last.pt --n 4

or set `ENGINE = "diffusion"` in `main.py` to use it inside the full pipeline.
(RAG remains the default engine and needs no training or API.)

## Expected folder layout when everything is in place

    3d-Floorplans-main/
    ├── RUN_GUIDE.md  main.py  builder.py  roof_generator.py  roof_ai.py ...
    ├── check_poznan.py
    ├── ml_roof_diffusion/            (dataset.py, model.py, train.py, sample.py)
    ├── synthetic_cache/              <- filled by step 3 (20,000 .npz)
    ├── datasets/PoznanRD/            <- your extracted PoznanRD zip (step 2)
    └── ckpt/                         <- created by training (step 4)

## Troubleshooting
- "No PoznanRD samples found" → run `python check_poznan.py <folder>`; its
  output shows the layout it sees and what it looked for.
- CUDA out of memory → lower `--bs` (64, then 32).
- Windows + DataLoader hang → set `num_workers=0` in
  `ml_roof_diffusion/train.py`'s DataLoader as a quick fix.
- Loss plateaus in stage 2 → normal after big stage-1 drops; judge by sampled
  outputs (`sample.py`), not loss alone.
