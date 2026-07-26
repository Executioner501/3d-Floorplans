# RECONSTRUCTION PLAN — Construct-AI (floor plan → 3D house)

Definite roadmap for (a) increasing output variety and (b) minimizing
reconstruction errors until real plans reliably produce good models.
Each phase has concrete steps and an acceptance test. Phases are ordered
by leverage: earlier phases unblock later ones.

## Phase 0 — DONE (2026-07-24)

| Fix | Where | Verified by |
|---|---|---|
| Per-plan px→metre scale (Issue A) | `scale_utils.py`, wired through `main.py`, `builder.export_to_obj(scale=)`, all engines incl. diffusion mask | simulated ⅓-res plan → 14×8 m (was 4.6×2.6 m); real `f1.jpg` → 10.1×14.0 m footprint |
| Composite-skillion direction bug (hollow rakes) | `gen_composite` resolves directions vs full footprint | slope-sign check: opposing planes on same axis, 0.00 % uncovered |
| Structure under floating roofs | soffit layer + support posts (`_soffit_plate`, `_posts_along_edge`), porch grammar | galleries |
| Roof coverage self-check | `generate_roof` rasterizes the built roof; >3 % unroofed footprint → rebuilt as distance-field hip (`info["repaired"]`) | deliberately broken half-roof repaired in test |
| KB 25 → **50 exemplars** | `roof_knowledge.py` (+ porch/posts/material zone keys, matte-black/solar-black/green-roof) | 50×2 seeds: 0 failures, 0 coverage leaks |

## Phase 1 — Detection robustness (biggest current error source)

The YOLO stage feeds everything; footprint noise becomes roof noise.

1. **Axis-snap + merge wall boxes** before `extract_footprint`: snap
   near-horizontal/vertical boxes to axis, merge collinear overlapping
   boxes (IoU-on-axis > 0.5). New `detect.postprocess_walls(walls)`.
2. ~~**Use the windows class.**~~ **DONE 2026-07-24** — `detect.py`
   returns windows (cls 2); `builder.py` renders framed glass panes.
3. **Door–wall association**: orient each door by its nearest wall
   segment instead of its own box aspect (fixes sliver detections).
4. **Confidence sweep**: evaluate conf ∈ {0.25, 0.3, 0.4, 0.5} on 10
   held-out CubiCasa5k plans; pick per-class thresholds.

**Acceptance:** IoU between extracted footprint and a hand-traced mask
≥ 0.90 on 8/10 test plans; doors flush with walls; windows visible.

## Phase 2 — Exact scale from printed dimensions

`scale_utils.dimensions_from_image` already OCRs dimension text when
`pytesseract` is installed; today the largest printed length is assumed
to be the long side.

1. Install Tesseract + `pytesseract` in the venv (optional dep).
2. Associate each OCR'd dimension with the nearest long footprint edge
   (position of the text box vs footprint bbox) instead of taking max.
3. Prefer OCR when ≥2 dimensions agree with the footprint aspect ratio
   (within 10 %); else fall back to target normalisation (current
   behaviour, `TARGET_LONG_M = 14`).

**Acceptance:** M.A duplex plan (36'×50') reconstructs to
10.97 × 15.24 m within ±3 %; walls 3 m; roof present.

## Phase 3 — Variety (beyond 50 exemplars)

1. **Mine real distributions**: OSM `roof:shape` statistics per region →
   reweight KB `weight` fields; SYNBUILD-3D wireframes → new multi-zone
   exemplars (target: 80 total, keep every `fit` envelope honest).
2. **Expose retrieval knobs** in `main.py` config: `TEMPERATURE`
   (0.6 = safe, 1.2 = adventurous) and a session `avoid` list so batch
   runs never repeat a recipe (`design_variations_rag` already does
   this — surface it).
3. **Accent materials**: more zone-level `material` overrides (two-tone
   houses read as far more "designed").
4. **Accessory richness**: gutters + ridge caps (thin boxes along
   eave/ridge lines), window bands on clerestory walls.

**Acceptance:** 20 random seeds on one plan → ≥ 12 distinct recipe ids,
no recipe > 20 % of picks; gallery renders pass eyeball review.

## Phase 4 — Error minimization / quality gates

1. Extend the self-check beyond coverage: `trimesh.Trimesh.is_watertight`
   per part, post-inside-building detection, zone-overlap volume check.
2. **Regression harness**: keep `rag_check`-style scripts as
   `tests/test_roofs.py` (smoke all exemplars × 4 plans × 2 seeds +
   coverage assertion). Run before every KB/generator change.
3. **Fallback ladder** (already partially in place):
   composite → repaired hip → legacy slab. Log which rung fired;
   a rising repair-rate is the early-warning signal for KB bugs.

**Acceptance:** 0 failures / 0 leaks on the full matrix; repair rate
< 5 % across 200 random seed/plan combos.

## Phase 5 — Diffusion engine (see issues.md §3 for full details)

Now unblocked by the scale fix (conditioning masks are in-distribution).

1. Regenerate the synthetic cache from the **50-exemplar** KB:
   `python -m ml_roof_diffusion.dataset --pregen 20000 --out synthetic_cache --workers 8`
2. Retrain stage 1 from scratch with `base=96` UNet, 80 epochs.
3. Curated stage 2 (filter near-flat PoznanRD, weight synthetic 2–3×,
   short finetune only).
4. Judge by sample diversity (pairwise |dh| > 0.3 m, rises 2–4.5 m),
   never by loss.

**Acceptance:** issues.md §3.5 criteria on rectangle, L and T plans.

## Phase 6 — Presentation-quality output

1. ~~PBR materials in the GLB~~ **DONE 2026-07-24** — `_finish` exports
   a Scene with one non-metallic rough PBRMaterial per part (fixes the
   near-black renders vertex-color GLBs produced in viewers).
2. ~~Ground plane + camera script~~ **DONE 2026-07-24** — paved pad +
   lawn in `builder.py`; `preview_render.py <glb> [png] [azim]` renders
   consistent well-lit previews.
3. Facade detail shipped: framed windows from the detector, framed
   door, plinth, roofline trim. Still open below.
4. Multi-storey massing: the reference photos are two-storey; the
   pipeline currently extrudes one storey. Needs per-floor plan input
   (or storey-count parameter replicating the wall layer) + stacked
   roof volumes. This is the main remaining gap to the reference look.
5. Window/door cutouts via boolean subtraction (manifold3d backend);
   real textures (brick/wood normal maps) on the PBR materials;
   render through Blender for marketing-quality images — the GLB now
   imports there with correct materials.

**Acceptance:** side-by-side renders match the quality bar of the
reference collages used to design the KB.
