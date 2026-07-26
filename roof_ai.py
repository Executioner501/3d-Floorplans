"""
roof_ai.py — Generative roof DESIGN layer
=========================================
Decides WHAT roof to build; roof_generator.py decides HOW to build it.

Two cooperating sources of intelligence:

1. LLM designer (Gemini, via ask_gemini.get_roof_parameters) — reads the
   actual floor-plan image and proposes a style + parameters from the
   full architectural vocabulary.

2. Local footprint-conditioned prior — analyses the extracted footprint
   (aspect ratio, convexity, wing count, area) and samples from a
   distribution of architecturally plausible styles. Used to
   (a) validate/repair LLM output, (b) run fully offline, and
   (c) generate N distinct variations for the same plan.

Controlled randomness: every design carries a `seed`; two calls with
different seeds on the SAME floor plan produce different-but-valid
roofs, satisfying the "similar plans, distinct roofs" requirement.
"""

import numpy as np
from roof_generator import (extract_footprint, footprint_metrics,
                            decompose_wings, STYLE_MATERIALS,
                            NEW_ENGINE_STYLES)

# legacy styles remain valid — they route to builder.py's original code
LEGACY_STYLES = ["flat", "split-level", "mono-pitch", "shed"]
ALL_STYLES = NEW_ENGINE_STYLES + LEGACY_STYLES

# architecturally sane pitch ranges (degrees) per style
PITCH_RANGE = {
    "gable":       (24, 45), "cross-gable": (26, 42), "hip": (22, 35),
    "pyramid":     (24, 38), "dutch-gable": (26, 38), "gambrel": (30, 40),
    "mansard":     (60, 72), "saltbox":     (28, 42), "butterfly": (7, 14),
    "skillion":    (8, 20),  "flat-modern": (0, 0),
    "flat": (0, 0), "split-level": (0, 0), "mono-pitch": (5, 18), "shed": (8, 20),
}


# ────────────────────────────────────────────────────────────────────
#  FOOTPRINT-CONDITIONED STYLE PRIOR
# ────────────────────────────────────────────────────────────────────
def style_prior(metrics, n_wings, region="mixed"):
    """Return {style: weight} conditioned on plan geometry.
    Encodes real residential design patterns:
      · L/T/U plans → cross-gable & hip dominate
      · long narrow plans → gable / saltbox / skillion
      · compact squares → pyramid / hip / flat-modern
      · large plans → hip / mansard read better than a single huge gable
    """
    a, cvx, area = metrics["aspect"], metrics["convexity"], metrics["area"]
    w = {s: 1.0 for s in NEW_ENGINE_STYLES}

    if cvx < 0.90 or n_wings >= 2:            # concave, multi-wing plan
        w["cross-gable"] *= 4.0; w["hip"] *= 3.0; w["dutch-gable"] *= 1.6
        w["pyramid"] *= 0.1; w["gambrel"] *= 0.4; w["saltbox"] *= 0.4
    if a > 1.9:                                # long & narrow
        w["gable"] *= 3.0; w["saltbox"] *= 2.0; w["skillion"] *= 1.8
        w["gambrel"] *= 1.6; w["pyramid"] *= 0.05
    if a < 1.25 and cvx > 0.93:                # compact square-ish
        w["pyramid"] *= 3.5; w["hip"] *= 2.0; w["flat-modern"] *= 1.6
    if area > 140:                             # large footprint
        w["hip"] *= 1.8; w["mansard"] *= 1.5; w["butterfly"] *= 0.6
    if area < 60:                              # small cottage
        w["gable"] *= 1.6; w["saltbox"] *= 1.3; w["mansard"] *= 0.5

    if region == "modern":
        for s in ("flat-modern", "skillion", "butterfly"):
            w[s] *= 3.0
        for s in ("gambrel", "mansard", "dutch-gable"):
            w[s] *= 0.3
    elif region == "traditional":
        for s in ("gable", "hip", "gambrel", "dutch-gable", "mansard", "saltbox"):
            w[s] *= 2.2
        for s in ("flat-modern", "butterfly", "skillion"):
            w[s] *= 0.3
    return w


def _sample(weights, rng):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    p /= p.sum()
    return str(rng.choice(keys, p=p))


# ────────────────────────────────────────────────────────────────────
#  PARAMETER SAMPLER  (controlled randomness)
# ────────────────────────────────────────────────────────────────────
def sample_parameters(style, metrics, rng):
    lo, hi = PITCH_RANGE.get(style, (20, 40))
    pitch = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
    params = {
        "roof_style":   style,
        "pitch_angle":  round(pitch, 1),
        "overhang":     round(float(rng.uniform(0.30, 0.70)), 2),
        "material":     str(rng.choice(STYLE_MATERIALS.get(style, ["asphalt-dark"]))),
        "has_dormers":  bool(rng.random() < (0.45 if style in
                            ("gable", "gambrel", "cross-gable", "mansard", "saltbox") else 0.0)),
        "dormer_count": int(rng.integers(1, 4)),
        "has_chimney":  bool(rng.random() < 0.55),
        "has_skylight": bool(rng.random() < (0.5 if style in
                            ("flat-modern", "skillion", "butterfly") else 0.1)),
        # legacy-path fields so builder.py stays happy either way
        "slab_thickness": 0.20,
        "has_parapet":  style in ("flat-modern", "flat", "split-level"),
        "parapet_height": round(float(rng.uniform(0.4, 0.7)), 2),
        "has_canopy":   bool(rng.random() < 0.5),
        "canopy_depth": round(float(rng.uniform(1.2, 2.2)), 2),
        "has_railing":  style in ("flat", "split-level"),
    }
    return params


def validate_and_repair(params, metrics, n_wings, rng):
    """Clamp LLM output into structurally valid ranges; fix bad combos."""
    style = str(params.get("roof_style", "gable")).lower().strip()
    aliases = {"cross gable": "cross-gable", "crossgable": "cross-gable",
               "dutch gable": "dutch-gable", "dutchgable": "dutch-gable",
               "flat modern": "flat-modern", "modern-flat": "flat-modern",
               "shed-modern": "skillion", "mono": "mono-pitch",
               "hipped": "hip", "hip-and-valley": "hip"}
    style = aliases.get(style, style)
    if style not in ALL_STYLES:
        style = _sample(style_prior(metrics, n_wings), np.random.default_rng())
    params["roof_style"] = style

    lo, hi = PITCH_RANGE.get(style, (20, 40))
    p = params.get("pitch_angle", (lo + hi) / 2)
    params["pitch_angle"] = float(np.clip(p, lo, max(hi, lo)))
    params["overhang"] = float(np.clip(params.get("overhang", 0.45), 0.15, 1.0))

    # pyramid on a strongly elongated / concave plan is invalid → hip
    if style == "pyramid" and (metrics["aspect"] > 1.5 or metrics["convexity"] < 0.9):
        params["roof_style"] = "hip"
    # butterfly on concave multi-wing plans tends to self-intersect → skillion
    if style == "butterfly" and metrics["convexity"] < 0.85:
        params["roof_style"] = "skillion"
    if "material" in params and params["material"] not in \
            STYLE_MATERIALS.get(params["roof_style"], []):
        params["material"] = str(rng.choice(
            STYLE_MATERIALS.get(params["roof_style"], ["asphalt-dark"])))
    return params


# ────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ────────────────────────────────────────────────────────────────────
def design_roof(walls, scale=0.01, seed=None, region="mixed",
                use_llm=False, image_path=None, llm_hint=None):
    """Produce one complete, validated roof-parameter dict.

    use_llm=True → ask Gemini first (needs API key + image), then
    validate/repair its answer against the actual footprint geometry.
    Otherwise the local footprint-conditioned prior designs the roof.
    """
    rng = np.random.default_rng(seed)
    poly = extract_footprint(walls, scale)
    if poly is None:
        return {"roof_style": "gable", "pitch_angle": 30, "overhang": 0.45,
                "seed": seed}
    metrics = footprint_metrics(poly)
    n_wings = len(decompose_wings(poly))

    llm_params = None
    if use_llm:
        try:
            from ask_gemini import get_roof_parameters
            llm_params = get_roof_parameters(image_path or "floorplan.png",
                                             footprint_hint={
                                                 "aspect": round(metrics["aspect"], 2),
                                                 "convexity": round(metrics["convexity"], 2),
                                                 "area_m2": round(metrics["area"], 1),
                                                 "wings": n_wings})
        except Exception as e:
            print(f"⚠️  LLM designer unavailable ({e}); using local prior.")

    if llm_params:
        params = validate_and_repair(dict(llm_params), metrics, n_wings, rng)
        # LLM decides style; sampler fills anything it left out + jitter
        defaults = sample_parameters(params["roof_style"], metrics, rng)
        for k, v in defaults.items():
            params.setdefault(k, v)
    else:
        if llm_hint and llm_hint in ALL_STYLES:
            style = llm_hint
        else:
            style = _sample(style_prior(metrics, n_wings, region), rng)
        params = sample_parameters(style, metrics, rng)

    params["seed"] = int(seed) if seed is not None else int(rng.integers(0, 2**31))
    return params


def design_variations(walls, n=4, scale=0.01, base_seed=0, region="mixed",
                      unique_styles=True):
    """N distinct plausible roofs for the SAME floor plan."""
    out, used = [], set()
    s = base_seed
    while len(out) < n and s < base_seed + 60:
        d = design_roof(walls, scale=scale, seed=s, region=region)
        if not unique_styles or d["roof_style"] not in used:
            used.add(d["roof_style"])
            out.append(d)
        s += 1
    return out


# ════════════════════════════════════════════════════════════════════
#  RAG DESIGNER  (default engine — replaces the Gemini/LLM designer)
# ════════════════════════════════════════════════════════════════════
def _resolve_recipe(ex, rng):
    """Sample concrete values from an exemplar's parameter ranges."""
    def _pick(v, default):
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return round(float(rng.uniform(*v)), 2)
        return v if v is not None else default

    zones = []
    for z in ex["zones"]:
        porch = z.get("porch")
        if porch:
            porch = {**porch, "depth": _pick(porch.get("depth"), 1.8)}
        zones.append({
            "region":    z["region"],
            "style":     z["style"],
            "pitch":     _pick(z.get("pitch"), 14),
            "overhang":  _pick(z.get("overhang"), round(float(rng.uniform(0.3, 0.55)), 2)),
            "h_offset":  _pick(z.get("h_offset"), 0.0),
            "direction": z.get("direction", "+short"),
            "clerestory": bool(z.get("clerestory", False)),
            "parapet":   bool(z.get("parapet", True)),
            "posts":     z.get("posts", "auto"),
            "porch":     porch,
            "material":  z.get("material"),
        })
    material = str(rng.choice(ex["materials"]))
    feats = ex.get("features", {})
    return zones, material, feats


def design_roof_rag(walls, scale=0.01, seed=None, style_pref="mixed",
                    temperature=0.9, avoid=None, k=12,
                    history_file=".rag_history.json", recipe=None):
    """Retrieval-Augmented roof design — the Gemini replacement.

    1. Embed the measured footprint (aspect, convexity, area, wings).
    2. Retrieve the top-k real-world roof compositions from
       roof_knowledge.KNOWLEDGE_BASE that architecturally fit it.
    3. Softmax-sample one (temperature controls diversity) and resolve
       its parameter ranges with the seeded RNG.
    4. roof_generator executes it with guaranteed-valid geometry.

    `avoid`: list of recipe ids already used this session → forces
    fresh compositions across consecutive houses.
    `style_pref`: "modern" | "traditional" | "mixed".

    Anti-repetition ACROSS RUNS: when seed is None (explore mode) the
    last few picked recipe ids are persisted to `history_file` and
    down-weighted on the next call, so repeatedly running main.py walks
    through the library instead of cycling the same top scorers.
    Reproducible runs (seed set) ignore the history. Delete the file to
    reset. `k` widens the candidate pool per pick (was 6 — the cause of
    "am I only getting 2-3 models?").
    """
    import json
    import os
    from roof_knowledge import sample_exemplar
    rng = np.random.default_rng(seed)
    poly = extract_footprint(walls, scale)
    if poly is None:
        return sample_parameters("gable", {"aspect": 1.5, "convexity": 1,
                                           "area": 100}, rng)
    metrics = footprint_metrics(poly)
    n_wings = len(decompose_wings(poly))

    use_hist = history_file and seed is None
    hist = []
    if use_hist and os.path.exists(history_file):
        try:
            with open(history_file, encoding="utf-8") as f:
                hist = [h for h in json.load(f) if isinstance(h, str)][-8:]
        except Exception:
            hist = []

    ex = sample_exemplar(metrics, n_wings, rng, k=k,
                         temperature=temperature, style_pref=style_pref,
                         avoid=list(avoid or []) + hist)

    # explicit recipe override (`recipe="double_gable_cross"`): build
    # exactly this composition instead of sampling
    if recipe:
        from roof_knowledge import KNOWLEDGE_BASE
        match = next((e for e in KNOWLEDGE_BASE if e["id"] == recipe), None)
        if match is not None:
            ex = match
            use_hist = False
        else:
            print(f"⚠️  RECIPE '{recipe}' not in KNOWLEDGE_BASE; sampling.")

    if use_hist:
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump((hist + [ex["id"]])[-10:], f)
        except Exception:
            pass
    zones, material, feats = _resolve_recipe(ex, rng)

    # Single full-footprint zones can ride the simple style path — EXCEPT
    # skillions and anything carrying porch/post structure, whose
    # direction/porch/posts only the composite executor honours.
    single = (len(zones) == 1
              and zones[0]["region"].get("type") == "full"
              and zones[0]["style"] != "skillion"
              and not zones[0].get("porch")
              and not isinstance(zones[0].get("posts"), (list, tuple)))
    params = {
        "roof_style": zones[0]["style"] if single else "composite",
        "zones": zones,
        "material": material,
        "pitch_angle": zones[0]["pitch"],
        "overhang": zones[0]["overhang"],
        "recipe_id": ex["id"],
        "recipe_name": ex["name"],
        "has_dormers": bool(rng.random() < feats.get("dormer_p", 0.0)),
        "dormer_count": int(rng.integers(1, 4)),
        "has_chimney": bool(rng.random() < feats.get("chimney_p", 0.0)),
        "has_skylight": bool(rng.random() < feats.get("skylight_p", 0.0)),
        "has_canopy": bool(rng.random() < 0.45),
        "canopy_depth": round(float(rng.uniform(1.2, 2.2)), 2),
        # legacy-path fields (unused by new engine, kept for safety)
        "slab_thickness": 0.2, "has_parapet": False,
        "parapet_height": 0.5, "has_railing": False,
        "seed": int(seed) if seed is not None else int(rng.integers(0, 2**31)),
    }
    # map full-footprint single styles onto the existing simple path
    if single and params["roof_style"] == "flat":
        params["roof_style"] = "flat-modern"
    return params


def design_variations_rag(walls, n=6, scale=0.01, base_seed=0,
                          style_pref="mixed"):
    """N compositionally distinct roofs for the same plan: each pick is
    added to the avoid-list so retrieval must move down the ranking."""
    out, used = [], []
    for i in range(n):
        d = design_roof_rag(walls, scale=scale, seed=base_seed + i,
                            style_pref=style_pref, avoid=used)
        used.append(d["recipe_id"])
        out.append(d)
    return out
