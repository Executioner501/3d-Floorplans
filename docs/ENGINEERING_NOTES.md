# ENGINEERING NOTES — Construct-AI (3d-Floorplans)

> **Historical document.** This is the running issue log kept while the
> diffusion engine was being debugged. It is preserved for the diagnoses
> and the "pitfalls already hit" sections, which are still the best guide
> to that part of the codebase. Some counts and file references are
> superseded: the knowledge base is now **44 exemplars** (not 37 or 50),
> the `3d-Floorplans-fixed-v3.zip` it mentions no longer exists — this
> repository is the source of truth — and RECONSTRUCTION_PLAN.md is the
> forward roadmap.

Issue-focused handoff. Goal: a fully working, good-looking diffusion roof
engine inside the floor-plan → 3D-house pipeline.

## 1. System in one paragraph
`main.py`: floor-plan PNG → YOLOv8 (CubiCasa5k-finetuned, mAP@0.5 0.731)
detects walls/doors → roof engine (`ENGINE = "rag" | "llm" | "prior" |
"diffusion"`) → `builder.py` assembles walls/doors/floor/roof → GLB/OBJ.
RAG engine: 37-exemplar composition KB (`roof_knowledge.py`) + procedural
executor (`roof_generator.py`). Diffusion engine (`ml_roof_diffusion/`):
footprint-mask-conditioned DDPM (RoofUNet 4.2M, 128×128 heightmaps,
H_MAX = 8 m, DDIM sampling, EMA weights); `sample.generate_roof_diffusion`
converts sampled heightmap → mesh. Machine: Windows, venv, RTX 5070 Ti
**Laptop (12 GB VRAM)**, torch 2.11.0+cu128 (cuda verified). **bs 64 max**
— bs 128 silently spills into sysmem and runs ~50× slower.

## 2. ISSUE A — FIXED 2026-07-24 (see RECONSTRUCTION_PLAN.md Phase 0)
Implemented `scale_utils.estimate_scale`: per-plan scale from OCR of
printed dimensions (optional pytesseract) else long-side normalisation
to `TARGET_LONG_M` (14 m), clamped to [0.004, 0.08]. Wired through
main.py → all engines + builder (`export_to_obj(scale=)`) + diffusion
mask. Verified: f1.jpg now reconstructs at 10.1×14.0 m (was ~3×4.6 m);
synthetic demo plans keep exactly 0.01. Original description follows
for context:

### (historical) fixed pixel→metre scale breaks real plans
Everything assumes `scale = 0.01` (1 px = 1 cm). Real test plan
(M.A Constructions duplex, 36'×50'): image ~550 px wide, building ~330 px
→ footprint 3.3 × 5 m, while `wall_h` stays 3.0 m absolute.
Consequences observed:
- Walls render as tower slabs ("everything fit into a small portion").
- Roof missing/"hollow" top on real plans: at ~15 mÂ² most KB exemplars
  fail their `fit.area` (30–90 m² minimums) and `gen_composite` skips
  zones with area < 2.5 m² → roof degenerates or vanishes. (Distinct from
  the composite-skillion direction bug, which is FIXED — see §5.)
- **Breaks the diffusion engine too**: the conditioning mask is built in
  metres with pad = 1.5 m. Training plans were 10–19 m wide → mask fills
  most of the 128² grid. A 3.3 m building + 1.5 m pad → tiny centred blob
  = out-of-distribution conditioning → garbage samples on real plans even
  with a good checkpoint. Scale fix is a PREREQUISITE for diffusion.
Fix (not yet implemented), in `main.py`/detect before anything downstream:
1. Default: `scale = TARGET_W_M / footprint_px_width`, TARGET_W_M ≈ 12–15.
2. Better: OCR the plan's dimension text (this plan prints 36' and 50')
   → exact scale (36 ft = 10.97 m / building px width).
3. Apply the SAME scale in detect→walls, `extract_footprint`, builder,
   and the diffusion mask; keep wall_h/doors/pitches in metres.
Acceptance test: rerun the M.A plan → footprint ≈ 11 × 15 m, walls 3 m,
roof present; diffusion mask fills ~70–90% of grid width like training.

## 3. ISSUE B: diffusion sample quality — diagnosed, partial fix, plan to finish
Training is DONE and healthy (stage 1: 60 ep synthetic, loss 0.0306→0.0004;
stage 2: 30 ep mixed w/ 13,037 cleaned PoznanRD, 0.0028→0.0008). A/B on a
rectangle plan, 4 samples each (pairwise mean |dh| inside footprint):
- `ckpt/roofdiff_ep60.pt` (pure stage-1): **diverse & tall** — |dh|
  0.29–2.02 m, rises 2.0–4.2 m, distinct design families. Current best.
- `ckpt/roofdiff_last.pt` (stage-2): **flattened** — |dh| 0.05–0.66 m,
  rises 1.8–2.3 m, one near-duplicate. Cause: PoznanRD rise p50 = 2.8 m
  (half the real roofs are near-flat) → finetune collapsed variety.
Remaining gap to "perfect": ep60 samples are valid but rounded/averaged
vs the crisp procedural engine (MSE mode-averaging on a small UNet).

### Roadmap to the final diffusion model (in order; each step testable)
1. **Fix Issue A first** (diffusion conditioning depends on it).
2. **Regenerate the synthetic cache** — the KB now has 37 compositions
   (12 new 2026-trend ones), so the cache is more diverse than the one
   ep60 was trained on:
   `python -m ml_roof_diffusion.dataset --pregen 20000 --out synthetic_cache --workers 8`
3. **Retrain stage 1 from scratch, larger UNet**: bump `base=64 → 96` in
   `model.py` (≈9–10M params; incompatible with old ckpts — that's fine).
   `--data synthetic --epochs 80 --bs 64 --lr 2e-4`. On this GPU with the
   RAM-preloaded cache an epoch is fast; 80 ep is cheap. More capacity is
   the main lever against mode-averaged/rounded outputs.
4. **Curated stage 2** (realism WITHOUT flattening):
   - Filter PoznanRD to rise p98 > 2.5 m at preload (one line where
     `rises` is computed in `dataset.py`) → drops the near-flat half.
   - Weight synthetic 2–3× (repeat the synthetic dataset in
     `MixedRoofDataset` or concat it twice).
   - Short finetune only: `--resume ... --lr 1e-5 --epochs 10 --bs 64`.
5. **Evaluate after EVERY stage with the built-in tools** (never loss —
   it plateaus in ~10 epochs while quality improves for the whole run):
   `python -m ml_roof_diffusion.sample --ckpt ckpt/roofdiff_last.pt --n 4
   --plan rectangle --out diffusion_roofs/eval` → PNG heightmap previews
   + diversity report. Pass criteria: pairwise |dh| mostly > 0.3 m; max
   rises spanning ≥ 2–4.5 m; visible straight ridge/valley lines in PNGs;
   repeat for L-shape and T-shape.
6. **Sampling knobs for final quality**: `--steps 120–200`, `--eta 1.0`
   (0 = deterministic → collapses seeds toward the conditional mean),
   `--smooth 0.4–0.6` (1.0 rounds ridges into humps). EMA weights are
   used automatically; `--raw` exists only for debugging.
7. **Wire in**: `ENGINE="diffusion"`, `DIFF_CKPT` → the winning ckpt.
   Optional polish afterwards: small eave overhang (dilate mask ~2 px at
   sample time and train with matching dilation — do NOT dilate only at
   inference, the model would see OOD masks).
Until 3–4 are done: **use `ckpt/roofdiff_ep60.pt`** (already better than
stage-2 `last`); keep `ENGINE="rag"` for demos.

### Diffusion pitfalls already hit — do not repeat
- 4+2 epochs "training" → noise; judge by samples, not loss.
- Uncleaned Poznan: off-footprint buildings/trees trained in (now the
  loader zeroes outside footprint, rebases to eave, median-filters).
- No EMA + eta=0 → every seed pulled to the conditional mean.
- bs 128 on 12 GB → silent sysmem spill (set NVIDIA "Prefer No Sysmem
  Fallback" to make it fail loudly). num_workers=0 when RAM-resident.
- Stage-2 checkpoints overwrite `roofdiff_ep10/20/30.pt` — back up
  stage-1 finals before finetuning (ep60 survived only by luck).

## 4. Training/eval command reference
```powershell
python -m ml_roof_diffusion.dataset --pregen 20000 --out synthetic_cache --workers 8
python -m ml_roof_diffusion.train --data synthetic --cache synthetic_cache --epochs 80 --bs 64 --lr 2e-4
python -m ml_roof_diffusion.train --data mix --cache synthetic_cache --root datasets/PoznanRD --resume ckpt/roofdiff_last.pt --lr 1e-5 --epochs 10 --bs 64
python -m ml_roof_diffusion.sample --ckpt ckpt/roofdiff_last.pt --n 4 --plan L-shape --out diffusion_roofs/eval
python check_poznan.py datasets/PoznanRD
```
Delete `__pycache__` + `ml_roof_diffusion/__pycache__` after any file drop-in.
Checkpoints are `{"model","ema","epoch"}` dicts; legacy raw state_dicts
still load. PoznanRD heights: metres = uint16 px / 256; H_MAX = 8 m is
shared by train + sample (change in `dataset.py` only, then retrain).

## 5c. Refinement pass 2026-07-24 (later same day)
- Windows: detector cls 2 now flows detect.py → builder.py (framed
  glass panes). `process_yolo_results` returns (walls, doors, windows).
- GLB export rewritten: Scene + PBR materials (metallic 0.05, rough
  0.85) per part — vertex-color GLBs rendered near-black in viewers.
- Facade: framed door, plinth, charcoal roofline trim, paved pad +
  lawn context. Dormers now seat ON the slope (were hovering past the
  eave). `preview_render.py` renders lit PNG previews of any GLB.
- Dataset-mined RAG: `kb_mining.py` (Houses3K / Bonn class folders,
  ZRG wireframes → measured pitch percentiles) → `roof_kb_mined.json`,
  auto-loaded by roof_knowledge.py. Verified with synthetic stubs.

## 5b. Fixed 2026-07-24 in the WORKING DIR (zip is now stale)
- Issue A (scale) — see §2.
- Composite-skillion direction bug re-fixed in this copy (the v3 zip fix
  had not been applied here); soffit layer + support posts + porch
  grammar added (`porch=dict(side, depth)`, `posts=[edges]` zone keys).
- KB 25 → 50 exemplars; new materials matte-black/solar-black/green-roof.
- Roof coverage self-check in `generate_roof`: >3% unroofed footprint →
  rebuilt as safe hip, `info["repaired"]=True`.
- Diffusion note: regenerate the synthetic cache from the 50-exemplar KB
  before retraining (Â§3 step 2 count is outdated).

## 5. Fixed earlier (verified — don't redebug)
- Composite-skillion direction bug: `+short`/`-short` resolved against
  each zone's own bounds; split zones flip aspect → "opposing" planes
  came out perpendicular with an open wedge into the house. Fixed in
  `gen_composite` (resolve vs full footprint). 0.00% uncovered cells,
  slope signs verified opposite; all 37 KB exemplars watertight on
  rect/square/L/T plans.
- `main.py` `DIFF_CKPT` pointed at nonexistent `ml_roof_diffusion/ckpt/`
  → now `ckpt/roofdiff_last.pt` (change to ep60/winner per §3).
- Slow epochs: RAM preload for both datasets (npz 153 ms → 0.2 ms/item;
  Poznan decoded once, multiprocess), tqdm + device print + grad clip.
- Legacy checkpoint compatibility, check_poznan fast path
  (`preload=False`).

## 6. Current artifacts
`3d-Floorplans-fixed-v3.zip` = source of truth (all fixes, 37-exemplar
KB, new materials matte-black/solar-black/green-roof, rewritten
ml_roof_diffusion, regenerated showcase GLB, rag_2026_gallery.png).
Checkpoints on the user's machine: `roofdiff_ep60.pt` (stage-1 final,
current best), `roofdiff_last.pt` = `roofdiff_stage2_final.pt` (backup).
Deck `genai_proposal_final.pptx` + `architecture_diagram.png` are done
except: update "26-exemplar" → 37 (3 places) and write the one-page
project summary the rubric requires.
