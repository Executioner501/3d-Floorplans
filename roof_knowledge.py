"""
roof_knowledge.py — RAG knowledge base + retriever for roof design
==================================================================
Replaces the LLM designer. This is Retrieval-Augmented Generation for
GEOMETRY:

  · KNOWLEDGE BASE — a curated library of real-world residential roof
    COMPOSITIONS (not single styles): opposing skillions with a
    clerestory band, sawtooth steps, stacked flat volumes, hip+gable
    mixes, butterfly-with-garage-wing, etc. Each exemplar encodes the
    zone layout, pitch ranges, material palette and the footprint
    conditions it architecturally suits.

  · RETRIEVER — embeds the query building's measured footprint
    (aspect, convexity, area, wing count) and each exemplar's
    compatibility envelope into a common feature space, scores them,
    and samples from the top-k with temperature → diverse but
    footprint-appropriate designs, never a fixed template.

  · GENERATOR — roof_generator.py executes the retrieved composition,
    guaranteeing watertight, structurally valid geometry.

Extend the library freely: add entries mined from SYNBUILD-3D
wireframes, OSM `roof:shape` statistics, or your own reference photos —
each entry is just a dict.
"""

import json
import os

import numpy as np

# ────────────────────────────────────────────────────────────────────
#  Zone-region grammar (resolved against the real footprint at build
#  time by roof_generator.gen_composite):
#    {"type": "full"}                          whole footprint
#    {"type": "wing",  "index": i}             i-th decomposed wing
#    {"type": "split", "axis": "x"|"y"|"long"|"short", "f0": a, "f1": b}
#                                              fractional slice of plan
# ────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    # ═════════════ MODERN COMPOSITIONS ═════════════
    dict(id="opposing_skillions_clerestory",
         name="Opposing skillions with clerestory band",
         tags=["modern"], weight=1.6,
         fit=dict(aspect=(1.2, 3.2), convexity=(0.86, 1.0), area=(55, 400)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.58),
                  style="skillion", pitch=(10, 16), direction="-short"),
             dict(region=dict(type="split", axis="long", f0=0.58, f1=1.0),
                  style="skillion", pitch=(8, 14), direction="+short",
                  h_offset=(0.7, 1.2), clerestory=True),
         ],
         materials=["metal-seam", "metal-green", "membrane-white"],
         features=dict(skylight_p=0.4)),

    dict(id="echelon_split_skillion",
         name="Echelon split-level skillion",
         tags=["modern"], weight=1.3,
         fit=dict(aspect=(1.3, 3.5), convexity=(0.84, 1.0), area=(60, 380)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.5),
                  style="skillion", pitch=(9, 15), direction="+long"),
             dict(region=dict(type="split", axis="long", f0=0.5, f1=1.0),
                  style="skillion", pitch=(9, 15), direction="+long",
                  h_offset=(0.8, 1.3), clerestory=True),
         ],
         materials=["metal-seam", "metal-red", "membrane-white"],
         features=dict(skylight_p=0.3)),

    dict(id="sawtooth_steps",
         name="Stepped sawtooth (three parallel sheds)",
         tags=["modern"], weight=1.0,
         fit=dict(aspect=(1.5, 3.8), convexity=(0.9, 1.0), area=(80, 420)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.34),
                  style="skillion", pitch=(12, 17), direction="-long"),
             dict(region=dict(type="split", axis="long", f0=0.34, f1=0.67),
                  style="skillion", pitch=(12, 17), direction="-long",
                  h_offset=(0.45, 0.7), clerestory=True),
             dict(region=dict(type="split", axis="long", f0=0.67, f1=1.0),
                  style="skillion", pitch=(12, 17), direction="-long",
                  h_offset=(0.95, 1.4), clerestory=True),
         ],
         materials=["metal-seam", "metal-green"],
         features=dict(skylight_p=0.5)),

    dict(id="butterfly_statement",
         name="Mid-century butterfly",
         tags=["modern"], weight=1.1,
         fit=dict(aspect=(1.0, 2.4), convexity=(0.9, 1.0), area=(50, 260)),
         zones=[dict(region=dict(type="full"), style="butterfly",
                     pitch=(8, 14))],
         materials=["metal-seam", "membrane-white", "metal-green"],
         features=dict(skylight_p=0.35)),

    dict(id="butterfly_flat_wing",
         name="Butterfly living wing + flat garage wing",
         tags=["modern"], weight=1.0,
         fit=dict(aspect=(1.4, 3.0), convexity=(0.8, 1.0), area=(90, 380),
                  wings=(1, 3)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.62),
                  style="butterfly", pitch=(8, 13)),
             dict(region=dict(type="split", axis="long", f0=0.62, f1=1.0),
                  style="flat", parapet=True),
         ],
         materials=["metal-seam", "membrane-white"],
         features=dict(skylight_p=0.3)),

    dict(id="stacked_boxes",
         name="Stacked modernist volumes",
         tags=["modern"], weight=1.4,
         fit=dict(aspect=(1.0, 2.6), convexity=(0.75, 1.0), area=(70, 420)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.55),
                  style="flat", parapet=True),
             dict(region=dict(type="split", axis="long", f0=0.55, f1=1.0),
                  style="flat", parapet=True, h_offset=(1.0, 1.6),
                  clerestory=True),
         ],
         materials=["membrane-white", "metal-seam", "asphalt-gray"],
         features=dict(skylight_p=0.5)),

    # "flat_roof_deck" and "flat_cantilever" removed on user request —
    # single-zone full-footprint flat slabs read as a plain lid dropped
    # on the walls. Flat zones survive only INSIDE articulated
    # compositions (stepped terraces, carport wings, clerestory boxes),
    # never as the whole roof. See also the removals further down.

    dict(id="mono_skillion",
         name="Single dramatic skillion plane",
         tags=["modern"], weight=1.0,
         fit=dict(aspect=(1.3, 4.0), convexity=(0.88, 1.0), area=(40, 300)),
         zones=[dict(region=dict(type="full"), style="skillion",
                     pitch=(10, 20), direction="+short")],
         materials=["metal-seam", "metal-red", "metal-green"],
         features=dict(skylight_p=0.3)),

    dict(id="scandi_black_gable",
         name="Scandinavian minimal gable (black metal)",
         tags=["modern"], weight=0.8,   # plain prism — downweighted
         fit=dict(aspect=(1.4, 3.4), convexity=(0.9, 1.0), area=(40, 260)),
         zones=[dict(region=dict(type="full"), style="gable",
                     pitch=(38, 50), overhang=(0.15, 0.3))],
         materials=["asphalt-dark", "metal-seam"],
         features=dict(chimney_p=0.5)),

    dict(id="japanese_low_hip",
         name="Japanese-modern low hip, deep eaves",
         tags=["modern"], weight=1.0,
         fit=dict(aspect=(1.0, 2.2), convexity=(0.82, 1.0), area=(60, 340)),
         zones=[dict(region=dict(type="full"), style="hip",
                     pitch=(15, 22), overhang=(0.8, 1.2))],
         materials=["asphalt-dark", "metal-seam", "slate-blue"],
         features=dict()),

    # "desert_parapet" removed on user request (plain flat lid).

    # ═════════════ TRADITIONAL COMPOSITIONS ═════════════
    # "classic_gable" (plain single gable prism) removed on user request —
    # too generic; double_gable_cross below covers the gabled family with
    # an actual composition.
    dict(id="double_gable_cross",
         name="Double gable — cross-gabled volumes",
         tags=["traditional", "modern"], weight=1.5,
         fit=dict(aspect=(1.1, 3.0), convexity=(0.78, 1.0), area=(50, 380)),
         # single full-region cross-gable rides the simple engine path,
         # which SYNTHESIZES a crossing wing on rectangular plans and
         # follows the real wings on L/T plans.
         zones=[dict(region=dict(type="full"), style="cross-gable",
                     pitch=(28, 40))],
         materials=["asphalt-dark", "asphalt-gray", "slate-blue",
                    "metal-seam", "matte-black"],
         features=dict(chimney_p=0.5, dormer_p=0.25)),

    dict(id="cross_gable_family",
         name="Cross-gable over L/T wings",
         tags=["traditional"], weight=1.5,
         fit=dict(aspect=(1.0, 2.6), convexity=(0.55, 0.92), area=(70, 420),
                  wings=(2, 4)),
         zones=[dict(region=dict(type="wing", index=0), style="gable",
                     pitch=(28, 40)),
                dict(region=dict(type="wing", index=1), style="gable",
                     pitch=(28, 40))],
         materials=["asphalt-dark", "asphalt-gray", "slate-blue", "cedar-shake"],
         features=dict(chimney_p=0.7, dormer_p=0.4)),

    dict(id="hip_gable_mix",
         name="Hip main volume + gabled wing",
         tags=["traditional"], weight=1.2,
         fit=dict(aspect=(1.0, 2.6), convexity=(0.6, 0.95), area=(80, 420),
                  wings=(2, 4)),
         zones=[dict(region=dict(type="wing", index=0), style="hip",
                     pitch=(24, 32)),
                dict(region=dict(type="wing", index=1), style="gable",
                     pitch=(28, 38))],
         materials=["terracotta", "asphalt-dark", "slate-blue"],
         features=dict(chimney_p=0.6)),

    dict(id="hip_estate",
         name="Hipped estate roof",
         tags=["traditional"], weight=0.8,   # plain prism — downweighted
         fit=dict(aspect=(1.0, 2.4), convexity=(0.6, 1.0), area=(80, 500)),
         zones=[dict(region=dict(type="full"), style="hip", pitch=(24, 34))],
         materials=["terracotta", "clay-brown", "slate-blue", "asphalt-dark"],
         features=dict(chimney_p=0.7)),

    dict(id="mediterranean_hip",
         name="Mediterranean low hip, terracotta",
         tags=["traditional"], weight=0.8,   # plain prism — downweighted
         fit=dict(aspect=(1.0, 2.2), convexity=(0.7, 1.0), area=(60, 400)),
         zones=[dict(region=dict(type="full"), style="hip",
                     pitch=(18, 24), overhang=(0.5, 0.8))],
         materials=["terracotta", "clay-brown"],
         features=dict(chimney_p=0.4)),

    dict(id="pyramid_pavilion",
         name="Pyramidal pavilion",
         tags=["traditional"], weight=0.9,
         fit=dict(aspect=(1.0, 1.3), convexity=(0.92, 1.0), area=(30, 180)),
         zones=[dict(region=dict(type="full"), style="pyramid", pitch=(26, 38))],
         materials=["terracotta", "slate-blue", "asphalt-dark"],
         features=dict(chimney_p=0.3)),

    dict(id="dutch_colonial",
         name="Dutch gable (gablet)",
         tags=["traditional"], weight=0.9,
         fit=dict(aspect=(1.2, 2.4), convexity=(0.9, 1.0), area=(60, 300)),
         zones=[dict(region=dict(type="full"), style="dutch-gable",
                     pitch=(26, 36))],
         materials=["terracotta", "clay-brown", "cedar-shake"],
         features=dict(chimney_p=0.6)),

    dict(id="gambrel_barn",
         name="Gambrel farmhouse",
         tags=["traditional"], weight=0.9,
         fit=dict(aspect=(1.5, 3.2), convexity=(0.9, 1.0), area=(60, 320)),
         zones=[dict(region=dict(type="full"), style="gambrel", pitch=(30, 40))],
         materials=["asphalt-dark", "cedar-shake", "metal-red"],
         features=dict(chimney_p=0.5, dormer_p=0.55)),

    dict(id="saltbox_newengland",
         name="New-England saltbox",
         tags=["traditional"], weight=0.8,
         fit=dict(aspect=(1.4, 2.8), convexity=(0.9, 1.0), area=(45, 240)),
         zones=[dict(region=dict(type="full"), style="saltbox", pitch=(30, 42))],
         materials=["cedar-shake", "asphalt-gray", "asphalt-dark"],
         features=dict(chimney_p=0.8)),

    dict(id="mansard_french",
         name="French mansard",
         tags=["traditional"], weight=0.7,
         fit=dict(aspect=(1.0, 1.8), convexity=(0.85, 1.0), area=(90, 420)),
         zones=[dict(region=dict(type="full"), style="mansard", pitch=(62, 72))],
         materials=["slate-blue", "asphalt-dark"],
         features=dict(chimney_p=0.6, dormer_p=0.7)),

    dict(id="farmhouse_metal_gable",
         name="Modern-farmhouse standing-seam gable",
         tags=["traditional", "modern"], weight=0.9,   # plain prism
         fit=dict(aspect=(1.3, 3.0), convexity=(0.75, 1.0), area=(60, 360)),
         zones=[dict(region=dict(type="full"), style="gable", pitch=(30, 42),
                     overhang=(0.4, 0.6))],
         materials=["metal-seam", "metal-red", "metal-green"],
         features=dict(chimney_p=0.5, dormer_p=0.3)),

    dict(id="craftsman_low_gable",
         name="Craftsman low-pitch gable, deep eaves",
         tags=["traditional"], weight=0.9,
         fit=dict(aspect=(1.2, 2.4), convexity=(0.8, 1.0), area=(55, 260)),
         zones=[dict(region=dict(type="full"), style="gable",
                     pitch=(20, 26), overhang=(0.7, 1.0))],
         materials=["cedar-shake", "asphalt-dark", "asphalt-gray"],
         features=dict(chimney_p=0.6, dormer_p=0.4)),

    dict(id="cottage_steep_gable",
         name="Storybook steep gable cottage",
         tags=["traditional"], weight=0.7,
         fit=dict(aspect=(1.2, 2.2), convexity=(0.88, 1.0), area=(30, 160)),
         zones=[dict(region=dict(type="full"), style="gable", pitch=(45, 56))],
         materials=["cedar-shake", "slate-blue", "asphalt-dark"],
         features=dict(chimney_p=0.85, dormer_p=0.5)),

    # ═════════════ 2026 REFERENCE-INSPIRED COMPOSITIONS ═════════════
    # Mined from contemporary small-house references: shed roofs with
    # covered porches on posts, carport wings, monitor clerestories,
    # matte-black metal, green roofs.  Zone grammar additions:
    #   porch=dict(side, depth)  roof extends past that side, on posts
    #   posts=[edges]            explicit pillar edges ("auto" default)
    dict(id="shed_porch_modern",
         name="Shed roof with covered entry porch (posts)",
         tags=["modern"], weight=1.5,
         fit=dict(aspect=(1.1, 2.8), convexity=(0.86, 1.0), area=(45, 300)),
         zones=[dict(region=dict(type="full"), style="skillion",
                     pitch=(8, 14), direction="+long", overhang=(0.5, 0.8),
                     porch=dict(side="-long", depth=(1.5, 2.2)))],
         materials=["metal-seam", "matte-black", "metal-green"],
         features=dict(skylight_p=0.35)),

    dict(id="skillion_carport_wing",
         name="Raised clerestory skillion + carport wing",
         tags=["modern"], weight=1.2,
         fit=dict(aspect=(1.3, 3.2), convexity=(0.84, 1.0), area=(70, 380)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.62),
                  style="skillion", pitch=(9, 14), direction="+short",
                  h_offset=(0.6, 1.0), clerestory=True),
             dict(region=dict(type="split", axis="long", f0=0.62, f1=1.0),
                  style="skillion", pitch=(6, 10), direction="+short",
                  overhang=(0.6, 0.9), posts=["+long"]),
         ],
         materials=["metal-seam", "solar-black", "membrane-white"],
         features=dict(skylight_p=0.3)),

    dict(id="dual_skillion_porch",
         name="High-ceiling opposing skillions + verandah",
         tags=["modern"], weight=1.3,
         fit=dict(aspect=(1.2, 3.0), convexity=(0.86, 1.0), area=(60, 380)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.55),
                  style="skillion", pitch=(10, 16), direction="-short",
                  porch=dict(side="-short", depth=(1.3, 1.9))),
             dict(region=dict(type="split", axis="long", f0=0.55, f1=1.0),
                  style="skillion", pitch=(9, 14), direction="+short",
                  h_offset=(1.0, 1.5), clerestory=True),
         ],
         materials=["metal-seam", "matte-black", "metal-red"],
         features=dict(skylight_p=0.35)),

    dict(id="gable_portico_farmhouse",
         name="Gable volume + flat entry canopy on posts",
         tags=["traditional", "modern"], weight=1.1,
         fit=dict(aspect=(1.2, 2.8), convexity=(0.86, 1.0), area=(55, 320)),
         zones=[
             dict(region=dict(type="split", axis="short", f0=0.28, f1=1.0),
                  style="gable", pitch=(30, 40), overhang=(0.4, 0.6)),
             dict(region=dict(type="split", axis="short", f0=0.0, f1=0.28),
                  style="flat", parapet=False, overhang=(0.5, 0.8),
                  posts=["-short"]),
         ],
         materials=["asphalt-dark", "matte-black", "cedar-shake"],
         features=dict(chimney_p=0.5)),

    dict(id="modern_asym_gable",
         name="Asymmetric-ridge modern gable (matte black)",
         tags=["modern"], weight=1.0,
         fit=dict(aspect=(1.2, 2.6), convexity=(0.88, 1.0), area=(45, 260)),
         zones=[dict(region=dict(type="full"), style="saltbox",
                     pitch=(20, 30), overhang=(0.4, 0.7))],
         materials=["matte-black", "metal-seam"],
         features=dict(chimney_p=0.2)),

    dict(id="parapet_skillion_mix",
         name="Flat parapet volume + rising skillion wing",
         tags=["modern"], weight=1.1,
         fit=dict(aspect=(1.2, 2.8), convexity=(0.8, 1.0), area=(60, 360)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.45),
                  style="flat", parapet=True),
             dict(region=dict(type="split", axis="long", f0=0.45, f1=1.0),
                  style="skillion", pitch=(8, 13), direction="+long",
                  h_offset=(0.3, 0.6), clerestory=True),
         ],
         materials=["membrane-white", "metal-seam", "matte-black"],
         features=dict(skylight_p=0.4)),

    # "green_roof_terrace" removed on user request (plain flat lid); the
    # green-roof material stays available to flat zones in compositions.

    dict(id="solar_ranch_skillion",
         name="Low solar-array skillion ranch",
         tags=["modern"], weight=1.0,
         fit=dict(aspect=(1.6, 4.0), convexity=(0.88, 1.0), area=(60, 340)),
         zones=[dict(region=dict(type="full"), style="skillion",
                     pitch=(5, 9), direction="+short",
                     overhang=(0.4, 0.7))],
         materials=["solar-black", "matte-black"],
         features=dict(skylight_p=0.15)),

    dict(id="butterfly_carport",
         name="Butterfly living wing + open carport slab",
         tags=["modern"], weight=0.9,
         fit=dict(aspect=(1.4, 3.0), convexity=(0.8, 1.0), area=(80, 360)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.6),
                  style="butterfly", pitch=(8, 13)),
             dict(region=dict(type="split", axis="long", f0=0.6, f1=1.0),
                  style="flat", parapet=False, overhang=(0.6, 0.9),
                  posts=["+long"]),
         ],
         materials=["metal-seam", "membrane-white"],
         features=dict(skylight_p=0.3)),

    dict(id="wing_mix_hip_skillion",
         name="Hip main wing + skillion secondary wing",
         tags=["modern"], weight=1.1,
         fit=dict(aspect=(1.0, 2.6), convexity=(0.55, 0.92), area=(80, 420),
                  wings=(2, 4)),
         zones=[dict(region=dict(type="wing", index=0), style="hip",
                     pitch=(20, 26)),
                dict(region=dict(type="wing", index=1), style="skillion",
                     pitch=(9, 14), direction="+long")],
         materials=["matte-black", "metal-seam", "slate-blue"],
         features=dict()),

    dict(id="nordic_gable_porch",
         name="Steep Nordic gable + covered porch strip",
         tags=["modern", "traditional"], weight=1.0,
         fit=dict(aspect=(1.2, 2.6), convexity=(0.88, 1.0), area=(40, 240)),
         zones=[
             dict(region=dict(type="full"), style="gable",
                  pitch=(40, 52), overhang=(0.25, 0.45)),
             dict(region=dict(type="split", axis="short", f0=0.0, f1=0.22),
                  style="flat", parapet=False, overhang=(0.5, 0.8),
                  posts=["-short"]),
         ],
         materials=["matte-black", "cedar-shake", "asphalt-dark"],
         features=dict(chimney_p=0.6)),

    dict(id="clerestory_monitor",
         name="Monitor roof — raised glazed centre band",
         tags=["modern"], weight=1.1,
         fit=dict(aspect=(1.2, 3.0), convexity=(0.86, 1.0), area=(70, 380)),
         zones=[
             dict(region=dict(type="split", axis="short", f0=0.0, f1=0.33),
                  style="skillion", pitch=(8, 12), direction="+short"),
             dict(region=dict(type="split", axis="short", f0=0.67, f1=1.0),
                  style="skillion", pitch=(8, 12), direction="-short"),
             dict(region=dict(type="split", axis="short", f0=0.33, f1=0.67),
                  style="flat", parapet=False, h_offset=(0.8, 1.2),
                  clerestory=True),
         ],
         materials=["metal-seam", "matte-black", "membrane-white"],
         features=dict(skylight_p=0.4)),

    # ═════════════ FOOTPRINT-COVERAGE EXPANSION ═════════════
    # Recipes for plan shapes the base library under-served:
    # concave L/T/U wings, compact squares, very long & narrow bars.
    dict(id="dogtrot_breezeway",
         name="Dogtrot — twin gables + open breezeway on posts",
         tags=["traditional", "modern"], weight=1.0,
         fit=dict(aspect=(1.6, 3.4), convexity=(0.88, 1.0), area=(60, 320)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.40),
                  style="gable", pitch=(30, 40)),
             dict(region=dict(type="split", axis="long", f0=0.60, f1=1.0),
                  style="gable", pitch=(30, 40)),
             dict(region=dict(type="split", axis="long", f0=0.40, f1=0.60),
                  style="flat", parapet=False, overhang=(0.3, 0.5),
                  posts=["+short", "-short"], material="membrane-white"),
         ],
         materials=["metal-seam", "cedar-shake", "metal-red"],
         features=dict(chimney_p=0.5)),

    dict(id="l_court_skillions",
         name="L-court — two outward skillion wings",
         tags=["modern"], weight=1.2,
         fit=dict(aspect=(1.0, 2.4), convexity=(0.5, 0.9), area=(70, 420),
                  wings=(2, 4)),
         zones=[
             dict(region=dict(type="wing", index=0), style="skillion",
                  pitch=(9, 14), direction="+long"),
             dict(region=dict(type="wing", index=1), style="skillion",
                  pitch=(9, 14), direction="+short"),
         ],
         materials=["metal-seam", "matte-black", "metal-green"],
         features=dict(skylight_p=0.3)),

    # "u_flat_court" and "veranda_pavilion_flat" removed on user request
    # (plain flat lids; the colonnade did not rescue the roof itself).
    # Concave U/L plans are now served by cross_gable_family,
    # hip_gable_mix, tudor_cross_steep, l_court_skillions and
    # wing_mix_hip_skillion.

    dict(id="longhouse_clerestory",
         name="Longhouse split-pitch with clerestory band",
         tags=["modern"], weight=1.1,
         fit=dict(aspect=(1.8, 4.2), convexity=(0.88, 1.0), area=(60, 360)),
         zones=[
             dict(region=dict(type="split", axis="short", f0=0.0, f1=0.55),
                  style="skillion", pitch=(9, 14), direction="+short"),
             dict(region=dict(type="split", axis="short", f0=0.55, f1=1.0),
                  style="skillion", pitch=(8, 12), direction="-short",
                  h_offset=(0.5, 0.8), clerestory=True),
         ],
         materials=["metal-seam", "solar-black", "metal-green"],
         features=dict(skylight_p=0.35)),

    dict(id="stepped_terrace_flats",
         name="Three stepped flat terraces",
         tags=["modern"], weight=1.0,
         fit=dict(aspect=(1.3, 3.0), convexity=(0.82, 1.0), area=(80, 440)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.4),
                  style="flat", parapet=True),
             dict(region=dict(type="split", axis="long", f0=0.4, f1=0.7),
                  style="flat", parapet=True, h_offset=(0.7, 1.0)),
             dict(region=dict(type="split", axis="long", f0=0.7, f1=1.0),
                  style="flat", parapet=True, h_offset=(1.4, 2.0),
                  clerestory=True),
         ],
         materials=["membrane-white", "asphalt-gray", "matte-black"],
         features=dict(skylight_p=0.5, deck_railing=True)),

    dict(id="carport_gable_combo",
         name="Gable main + open flat carport on posts",
         tags=["traditional", "modern"], weight=1.2,
         fit=dict(aspect=(1.3, 3.0), convexity=(0.84, 1.0), area=(60, 360)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.68),
                  style="gable", pitch=(26, 36)),
             dict(region=dict(type="split", axis="long", f0=0.68, f1=1.0),
                  style="flat", parapet=False, overhang=(0.5, 0.8),
                  posts=["+long"]),
         ],
         materials=["asphalt-dark", "asphalt-gray", "metal-seam"],
         features=dict(chimney_p=0.5)),

    dict(id="queenslander_veranda",
         name="Hip cottage with front veranda strip",
         tags=["traditional"], weight=1.0,
         fit=dict(aspect=(1.0, 2.2), convexity=(0.8, 1.0), area=(55, 320)),
         zones=[
             dict(region=dict(type="full"), style="hip",
                  pitch=(22, 30), overhang=(0.6, 0.9)),
             dict(region=dict(type="split", axis="short", f0=0.0, f1=0.2),
                  style="flat", parapet=False, overhang=(0.4, 0.6),
                  posts=["-short"]),
         ],
         materials=["terracotta", "asphalt-dark", "metal-seam"],
         features=dict(chimney_p=0.5)),

    dict(id="tudor_cross_steep",
         name="Steep Tudor cross-gables over wings",
         tags=["traditional"], weight=0.9,
         fit=dict(aspect=(1.0, 2.4), convexity=(0.55, 0.9), area=(70, 380),
                  wings=(2, 4)),
         zones=[dict(region=dict(type="wing", index=0), style="gable",
                     pitch=(42, 52)),
                dict(region=dict(type="wing", index=1), style="gable",
                     pitch=(42, 52))],
         materials=["slate-blue", "cedar-shake", "asphalt-dark"],
         features=dict(chimney_p=0.85, dormer_p=0.5)),

    dict(id="hip_dormer_manor",
         name="Dormered hip manor",
         tags=["traditional"], weight=1.0,
         fit=dict(aspect=(1.0, 2.2), convexity=(0.7, 1.0), area=(100, 500)),
         zones=[dict(region=dict(type="full"), style="hip", pitch=(26, 34))],
         materials=["slate-blue", "asphalt-dark", "clay-brown"],
         features=dict(chimney_p=0.8, dormer_p=0.8)),

    dict(id="alpine_deep_gable",
         name="Alpine chalet gable, very deep eaves",
         tags=["traditional"], weight=0.8,
         fit=dict(aspect=(1.1, 2.2), convexity=(0.86, 1.0), area=(45, 260)),
         zones=[dict(region=dict(type="full"), style="gable",
                     pitch=(18, 24), overhang=(0.9, 1.3))],
         materials=["cedar-shake", "asphalt-dark", "metal-red"],
         features=dict(chimney_p=0.7)),

    dict(id="split_butterfly_clerestory",
         name="Butterfly wing + raised flat clerestory box",
         tags=["modern"], weight=0.9,
         fit=dict(aspect=(1.3, 2.8), convexity=(0.84, 1.0), area=(80, 380)),
         zones=[
             dict(region=dict(type="split", axis="long", f0=0.0, f1=0.6),
                  style="butterfly", pitch=(9, 14)),
             dict(region=dict(type="split", axis="long", f0=0.6, f1=1.0),
                  style="flat", parapet=True, h_offset=(0.8, 1.2),
                  clerestory=True),
         ],
         materials=["metal-seam", "membrane-white"],
         features=dict(skylight_p=0.4)),

    dict(id="black_barn_leanto",
         name="Modern black barn + lean-to porch",
         tags=["modern"], weight=1.1,
         fit=dict(aspect=(1.3, 2.8), convexity=(0.86, 1.0), area=(55, 320)),
         zones=[
             dict(region=dict(type="split", axis="short", f0=0.25, f1=1.0),
                  style="gable", pitch=(38, 48)),
             dict(region=dict(type="split", axis="short", f0=0.0, f1=0.25),
                  style="skillion", pitch=(12, 18), direction="+short",
                  overhang=(0.4, 0.6), posts=["-short"]),
         ],
         materials=["matte-black", "metal-seam"],
         features=dict(chimney_p=0.3)),
]


def is_plain_flat(ex):
    """True for a recipe whose ENTIRE roof is one flat slab over the
    whole footprint — the 'lid dropped on the walls' look the curated
    library deliberately excludes. Flat zones inside a multi-zone
    composition (stepped terraces, carports, clerestory boxes) are fine
    because something else gives the roof its shape."""
    zones = ex.get("zones") or []
    return (len(zones) == 1
            and zones[0].get("style") == "flat"
            and zones[0].get("region", {}).get("type") == "full")


# ────────────────────────────────────────────────────────────────────
#  DATASET-MINED EXEMPLARS  (kb_mining.py → roof_kb_mined.json)
#  Statistics measured from Houses3K / Bonn / ZRG extend the curated
#  library at import time. Delete the JSON to run curated-only.
# ────────────────────────────────────────────────────────────────────
def _load_mined(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "roof_kb_mined.json")):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"⚠️  roof_kb_mined.json unreadable ({e}); using curated KB only.")
        return []
    from roof_generator import STYLE_MATERIALS
    out, skipped = [], 0
    for e in raw:
        if not (isinstance(e, dict) and e.get("id") and e.get("zones")
                and e.get("fit")):
            continue
        if is_plain_flat(e):
            # datasets are full of flat roofs; mining them back in would
            # undo the curated removal of plain full-footprint lids
            skipped += 1
            continue
        if not e.get("materials"):
            st = e["zones"][0].get("style", "gable")
            key = "flat-modern" if st == "flat" else st
            e["materials"] = STYLE_MATERIALS.get(key, ["asphalt-dark"])
        out.append(e)
    if out or skipped:
        print(f"📚 RAG KB: +{len(out)} dataset-mined exemplars "
              f"({len(KNOWLEDGE_BASE)} curated"
              f"{f', {skipped} plain-flat skipped' if skipped else ''}).")
    return out


KNOWLEDGE_BASE += _load_mined()


# ════════════════════════════════════════════════════════════════════
#  RETRIEVER
# ════════════════════════════════════════════════════════════════════
def _range_score(v, lo, hi, soft=0.35):
    """1.0 inside [lo, hi]; smooth gaussian falloff outside."""
    if lo <= v <= hi:
        return 1.0
    d = (lo - v) if v < lo else (v - hi)
    span = max(hi - lo, 1e-6)
    return float(np.exp(-((d / (soft * span)) ** 2)))


def score_exemplar(ex, metrics, n_wings, style_pref="mixed"):
    fit = ex["fit"]
    s = ex.get("weight", 1.0)
    s *= _range_score(metrics["aspect"], *fit.get("aspect", (0.5, 5.0)))
    s *= _range_score(metrics["convexity"], *fit.get("convexity", (0.5, 1.0)))
    s *= _range_score(metrics["area"], *fit.get("area", (20, 600)), soft=0.8)
    if "wings" in fit:
        s *= _range_score(n_wings, *fit["wings"], soft=1.0)
    # concave plans shouldn't get full-footprint gable/gambrel prisms
    if metrics["convexity"] < 0.85 and any(
            z["region"]["type"] == "full" and z["style"] in ("gable", "gambrel",
                                                             "saltbox")
            for z in ex["zones"]):
        s *= 0.15
    if style_pref == "modern":
        s *= 3.0 if "modern" in ex["tags"] else 0.15
    elif style_pref == "traditional":
        s *= 3.0 if "traditional" in ex["tags"] else 0.15
    return s


def retrieve(metrics, n_wings, k=6, style_pref="mixed", avoid=None):
    """Rank the knowledge base for this footprint; returns top-k
    (exemplar, score) with any `avoid` recipe-ids down-weighted —
    the anti-repetition mechanism across a session."""
    avoid = set(avoid or [])
    scored = []
    for ex in KNOWLEDGE_BASE:
        s = score_exemplar(ex, metrics, n_wings, style_pref)
        if ex["id"] in avoid:
            s *= 0.10
        scored.append((ex, s))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


def sample_exemplar(metrics, n_wings, rng, k=6, temperature=0.8,
                    style_pref="mixed", avoid=None):
    """Softmax-sample one exemplar from the top-k retrieval —
    controlled randomness at the composition level."""
    top = retrieve(metrics, n_wings, k, style_pref, avoid)
    ws = np.array([max(s, 1e-9) for _, s in top])
    p = np.exp(np.log(ws) / max(temperature, 0.05))
    p /= p.sum()
    idx = int(rng.choice(len(top), p=p))
    return top[idx][0]
