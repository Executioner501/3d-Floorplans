"""
roof_llm.py — Gemini designs the roof COMPOSITION, not a style label.
=====================================================================
The knowledge base in `roof_knowledge.py` stores a design grammar: zone
layouts, split axes, pitch ranges, height offsets, clerestory bands,
porches, material palettes, and the footprint envelope each composition
suits. That grammar is an action space, so an LLM can be asked to author a
NEW entry in it rather than pick one off the shelf.

    floor plan + measured footprint
        -> Gemini emits a composition in the zone grammar
        -> validate_exemplar() clamps it into buildable ranges
        -> roof_generator executes it
        -> the coverage self-check verifies it
        -> anything that fails at any step falls back to RAG

The generated composition goes through `roof_ai._resolve_recipe`, the same
function that resolves hand-written knowledge-base entries, so there is one
code path for both and no second geometry surface to keep correct.

WHY THE FALLBACK IS UNCONDITIONAL
Output quality must never regress relative to the RAG engine. Every failure
mode — no API key, network error, malformed JSON, a style outside the
vocabulary, zones that do not tile the footprint, geometry that trips the
coverage self-check — resolves to the RAG design that would have been
produced anyway. The LLM can only ever ADD a design that survives the same
verification the curated library passes.

No new dependencies: the REST endpoint is called with urllib. The
`google-generativeai` SDK pulls grpcio, protobuf and google-api-core, about
40 MB uncompressed, against ~52 MB of headroom in the serverless bundle.

Setup:  set GEMINI_API_KEY (see .env.example). Without it this module
returns None immediately and the caller uses RAG.
"""
import base64
import json
import os
import urllib.error
import urllib.request

import numpy as np

ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:generateContent")
DEFAULT_MODEL = "gemini-2.5-flash"
TIMEOUT = 25

# Styles the geometry engine can execute inside a zone. "composite" is the
# multi-zone container itself, so it is not a zone style.
ZONE_STYLES = ["gable", "cross-gable", "hip", "pyramid", "dutch-gable",
               "gambrel", "mansard", "saltbox", "butterfly", "skillion",
               "flat"]
MATERIALS = ["terracotta", "clay-brown", "asphalt-dark", "asphalt-gray",
             "slate-blue", "metal-seam", "metal-green", "metal-red",
             "cedar-shake", "membrane-white", "matte-black", "solar-black",
             "green-roof"]
DIRECTIONS = ["+short", "-short", "+long", "-long"]
AXES = ["x", "y", "long", "short"]

# Per-style pitch envelopes. Anything outside is clamped rather than
# rejected — a plausible composition with a silly pitch is still useful.
PITCH = {
    "gable": (22, 45), "cross-gable": (24, 42), "hip": (20, 38),
    "pyramid": (24, 40), "dutch-gable": (24, 40), "gambrel": (55, 72),
    "mansard": (60, 74), "saltbox": (26, 44), "butterfly": (6, 16),
    "skillion": (6, 22), "flat": (0, 4),
}

PROMPT = """You are an architect. Design a roof COMPOSITION for this building
and return it as JSON.

You are not picking a style name. You are authoring a composition in the
grammar below: one or more zones, each covering part of the footprint, each
with its own roof form, pitch and height offset. Multi-zone compositions are
what make a house read as designed rather than as a shed — a lower carport
wing beside a main gable, two opposing skillions with a clerestory band of
glazing where they meet, a stepped terrace.

SCHEMA
{
  "id": "snake_case_identifier",
  "name": "Short human-readable name",
  "tags": ["modern"] or ["traditional"],
  "rationale": "one sentence on why this suits THIS footprint",
  "zones": [
    {
      "region": {"type": "full"}
               | {"type": "split", "axis": "long"|"short"|"x"|"y",
                  "f0": 0.0, "f1": 0.55}
               | {"type": "wing", "index": 0},
      "style": one of %(styles)s,
      "pitch": number,
      "overhang": 0.15 to 1.0,
      "h_offset": 0.0 to 2.5,       // how far this zone's eave sits ABOVE
                                     // the wall top. Offsets are relative:
                                     // give the LOWEST zone 0.0 and raise
                                     // the others. Never negative.
      "direction": one of %(dirs)s,  // which way a skillion rises
      "clerestory": true|false,      // glazed band where this zone steps up
      "parapet": true|false,
      "material": one of %(mats)s    // optional per-zone override
    }
  ],
  "materials": ["one or more of %(mats)s"],
  "features": {"dormer_p": 0.0-1.0, "chimney_p": 0.0-1.0, "skylight_p": 0.0-1.0}
}

RULES
- 1 to 4 zones.
- "split" slices the footprint along an axis; f0/f1 are fractions of that
  axis. Splits on the same axis MUST tile it without gaps or overlap:
  0.0-0.55 then 0.55-1.0, never 0.0-0.6 then 0.5-1.0.
- Use "full" only for a single-zone composition. Two "full" zones overlap.
- "wing" only makes sense on a concave plan; the wing count is given below.
- A single flat zone covering the whole footprint is forbidden — it reads as
  a lid dropped on the walls. Flat zones are fine as PART of a composition.
- h_offset is what creates the stepped silhouette. Use it. To put a low
  carport beside a tall main volume, give the carport 0.0 and the main
  volume 0.8 — do not use negative offsets.
- clerestory only where a zone steps up above its neighbour.

Match the composition to the measured footprint, not to taste: elongated
plans suit gable, saltbox or a long skillion; concave multi-wing plans suit
cross-gable or a hip-and-wing mix; compact squares suit pyramid, hip or a
stacked flat composition.
"""


def _metrics_block(metrics, n_wings, extra=None):
    lines = [
        f"  width x depth : {metrics.get('width', 0):.1f} x {metrics.get('depth', 0):.1f} m",
        f"  area          : {metrics.get('area', 0):.0f} m2",
        f"  aspect        : {metrics.get('aspect', 0):.2f}  (long side / short side)",
        f"  convexity     : {metrics.get('convexity', 0):.2f}  (1.0 = rectangular, "
        f"below 0.85 = concave / L or T shaped)",
        f"  wings         : {n_wings}",
    ]
    out = "\nMEASURED FOOTPRINT (authoritative — trust this over the image)\n"
    out += "\n".join(lines) + "\n"
    if extra:
        out += f"\nTHE CLIENT ASKS FOR: {extra}\n"
    return out


# ════════════════════════════════════════════════════════════════════
#  VALIDATION  — the LLM is never trusted
# ════════════════════════════════════════════════════════════════════
def _num(v, lo, hi, default):
    try:
        return float(np.clip(float(v), lo, hi))
    except (TypeError, ValueError):
        return default


def validate_exemplar(ex, n_wings):
    """Clamp a generated composition into something buildable.

    Returns (exemplar, None) or (None, reason). Clamps whatever can be
    clamped and rejects only what cannot be repaired — a bad pitch is
    fixable, two overlapping "full" zones are not.
    """
    if not isinstance(ex, dict):
        return None, "not an object"
    zones_in = ex.get("zones")
    if not isinstance(zones_in, list) or not 1 <= len(zones_in) <= 4:
        return None, f"needs 1-4 zones, got {len(zones_in) if isinstance(zones_in, list) else 'none'}"

    zones, full_count = [], 0
    for i, z in enumerate(zones_in):
        if not isinstance(z, dict):
            return None, f"zone {i} is not an object"
        style = str(z.get("style", "")).strip().lower()
        style = {"flat-modern": "flat", "shed": "skillion",
                 "mono-pitch": "skillion", "hipped": "hip"}.get(style, style)
        if style not in ZONE_STYLES:
            return None, f"zone {i}: unknown style {style!r}"

        region = z.get("region")
        if not isinstance(region, dict):
            return None, f"zone {i}: missing region"
        rtype = str(region.get("type", "")).lower()
        if rtype == "full":
            full_count += 1
            region = {"type": "full"}
        elif rtype == "split":
            axis = str(region.get("axis", "long")).lower()
            if axis not in AXES:
                axis = "long"
            f0 = _num(region.get("f0"), 0.0, 1.0, 0.0)
            f1 = _num(region.get("f1"), 0.0, 1.0, 1.0)
            if f1 - f0 < 0.12:
                return None, f"zone {i}: split {f0:.2f}-{f1:.2f} is too thin to build"
            region = {"type": "split", "axis": axis, "f0": f0, "f1": f1}
        elif rtype == "wing":
            idx = int(_num(region.get("index"), 0, max(n_wings - 1, 0), 0))
            region = {"type": "wing", "index": idx}
        else:
            return None, f"zone {i}: unknown region type {rtype!r}"

        lo, hi = PITCH[style]
        mat = z.get("material")
        zones.append({
            "region": region,
            "style": style,
            "pitch": _num(z.get("pitch"), lo, hi, (lo + hi) / 2),
            "overhang": _num(z.get("overhang"), 0.15, 1.0, 0.4),
            # h_offset floors at 0. The coverage self-check measures roof
            # presence ABOVE the wall top, so a zone dropped below it reads
            # as unroofed and the whole composition gets rejected — measured:
            # every offset < 0 trips the check, every offset >= 0 passes.
            # Nothing expressive is lost, because offsets are relative: a low
            # carport beside a tall gable is 0.0 / +0.8 rather than
            # -0.8 / 0.0. None of the 44 curated recipes use a negative
            # offset either, so this keeps the LLM inside the same envelope
            # the knowledge base already lives in.
            "h_offset": _num(z.get("h_offset"), 0.0, 2.5, 0.0),
            "direction": (z.get("direction") if z.get("direction") in DIRECTIONS
                          else "+short"),
            "clerestory": bool(z.get("clerestory", False)),
            "parapet": bool(z.get("parapet", True)),
            "posts": "auto",
            "porch": None,
            "material": mat if mat in MATERIALS else None,
        })

    if full_count and len(zones) > 1:
        return None, "a 'full' zone cannot be combined with others — they overlap"
    if len(zones) == 1 and zones[0]["style"] == "flat" \
            and zones[0]["region"]["type"] == "full":
        return None, "a single flat zone over the whole footprint is a lid, not a roof"

    # Splits on one axis must tile it. Sort and close small gaps rather than
    # rejecting: the model gets the ordering right far more often than it
    # gets the arithmetic exact.
    for axis in AXES:
        band = [z for z in zones if z["region"].get("axis") == axis]
        if len(band) < 2:
            continue
        band.sort(key=lambda z: z["region"]["f0"])
        band[0]["region"]["f0"] = 0.0
        band[-1]["region"]["f1"] = 1.0
        for a, b in zip(band, band[1:]):
            if abs(a["region"]["f1"] - b["region"]["f0"]) > 1e-6:
                mid = (a["region"]["f1"] + b["region"]["f0"]) / 2
                a["region"]["f1"] = b["region"]["f0"] = float(np.clip(mid, 0.0, 1.0))
        for z in band:
            if z["region"]["f1"] - z["region"]["f0"] < 0.12:
                return None, "splits collapsed to a sliver after tiling"

    mats = [m for m in (ex.get("materials") or []) if m in MATERIALS]
    feats = ex.get("features") if isinstance(ex.get("features"), dict) else {}
    tags = [t for t in (ex.get("tags") or []) if t in ("modern", "traditional")]

    return {
        "id": str(ex.get("id") or "llm_composition")[:64],
        "name": str(ex.get("name") or "AI-designed composition")[:120],
        "tags": tags or ["modern"],
        "weight": 1.0,
        "fit": dict(aspect=(0.4, 6.0), convexity=(0.4, 1.0), area=(10, 900)),
        "zones": zones,
        "materials": mats or ["asphalt-dark"],
        "features": {k: _num(feats.get(k), 0.0, 1.0, 0.0)
                     for k in ("dormer_p", "chimney_p", "skylight_p")},
        "rationale": str(ex.get("rationale") or "")[:300],
    }, None


# ════════════════════════════════════════════════════════════════════
#  GEMINI CALL
# ════════════════════════════════════════════════════════════════════
def _api_key(explicit=None):
    if explicit:
        return explicit
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        except ImportError:
            pass
    return key


def ask_gemini_composition(metrics, n_wings, image_bytes=None, mime="image/png",
                           request=None, api_key=None, model=DEFAULT_MODEL,
                           timeout=TIMEOUT, temperature=1.0):
    """One Gemini call. Returns the raw parsed dict, or None."""
    key = _api_key(api_key)
    if not key:
        return None, "no GEMINI_API_KEY"

    prompt = PROMPT % dict(styles=ZONE_STYLES, dirs=DIRECTIONS, mats=MATERIALS)
    parts = [{"text": prompt + _metrics_block(metrics, n_wings, request)}]
    if image_bytes:
        parts.append({"inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(image_bytes).decode("ascii")}})

    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": temperature,
        },
    }).encode("utf-8")

    url = ENDPOINT.format(model=model) + f"?key={key}"
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode("utf-8", "replace")
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:                                  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"

    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text), None
    except Exception as e:                                  # noqa: BLE001
        return None, f"unreadable response ({type(e).__name__}: {e})"


# ════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════
def design_roof_llm(walls, scale=0.01, seed=None, image_bytes=None,
                    mime="image/png", request=None, style_pref="mixed",
                    api_key=None, model=DEFAULT_MODEL, timeout=TIMEOUT,
                    verify=True):
    """Ask Gemini for a composition; fall back to RAG on any failure.

    Returns a params dict in exactly the shape design_roof_rag returns,
    plus:
        params["source"]     "llm" or "rag"
        params["llm_note"]   why the LLM result was not used, when it wasn't
        params["rationale"]  the model's one-line justification, when used

    `request` is optional natural language ("two-storey modern with a
    carport") which is passed through to the model.
    """
    from roof_ai import design_roof_rag, _resolve_recipe
    from roof_generator import extract_footprint, footprint_metrics, decompose_wings

    def _fallback(note):
        p = design_roof_rag(walls, scale=scale, seed=seed,
                            style_pref=style_pref, history_file=None)
        p["source"] = "rag"
        p["llm_note"] = note
        return p

    poly = extract_footprint(walls, scale)
    if poly is None:
        return _fallback("no footprint could be extracted")
    metrics = footprint_metrics(poly)
    n_wings = len(decompose_wings(poly))

    raw, err = ask_gemini_composition(
        metrics, n_wings, image_bytes=image_bytes, mime=mime, request=request,
        api_key=api_key, model=model, timeout=timeout)
    if raw is None:
        return _fallback(err)

    ex, why = validate_exemplar(raw, n_wings)
    if ex is None:
        return _fallback(f"rejected: {why}")

    rng = np.random.default_rng(seed)
    zones, material, feats = _resolve_recipe(ex, rng)
    single = (len(zones) == 1 and zones[0]["region"].get("type") == "full"
              and zones[0]["style"] != "skillion")
    style = zones[0]["style"] if single else "composite"
    if single and style == "flat":
        style = "flat-modern"

    params = {
        "roof_style": style,
        "zones": zones,
        "material": material,
        "pitch_angle": zones[0]["pitch"],
        "overhang": zones[0]["overhang"],
        "recipe_id": ex["id"],
        "recipe_name": ex["name"],
        "rationale": ex.get("rationale", ""),
        "has_dormers": bool(rng.random() < feats.get("dormer_p", 0.0)),
        "dormer_count": int(rng.integers(1, 4)),
        "has_chimney": bool(rng.random() < feats.get("chimney_p", 0.0)),
        "has_skylight": bool(rng.random() < feats.get("skylight_p", 0.0)),
        "has_canopy": bool(rng.random() < 0.45),
        "canopy_depth": round(float(rng.uniform(1.2, 2.2)), 2),
        "slab_thickness": 0.2, "has_parapet": False,
        "parapet_height": 0.5, "has_railing": False,
        "seed": int(seed) if seed is not None else int(rng.integers(0, 2 ** 31)),
        "source": "llm",
        "llm_note": None,
    }

    if verify:
        # Build it for real before accepting. This is the step that makes the
        # feature safe: a composition that trips the coverage self-check
        # (`repaired`) or throws is discarded in favour of RAG, so the LLM
        # can only ever add a design that passes the same verification the
        # curated library passes.
        try:
            from roof_generator import generate_roof
            meshes, info = generate_roof(walls, params, scale=scale, wall_h=3.0)
        except Exception as e:                              # noqa: BLE001
            return _fallback(f"generated composition failed to build: "
                             f"{type(e).__name__}: {e}")
        if not meshes:
            return _fallback("generated composition produced no geometry")
        if info.get("repaired"):
            return _fallback("generated composition left the footprint "
                             "uncovered (coverage self-check)")
    return params


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from demo_roofs import PLANS

    plan = sys.argv[1] if len(sys.argv) > 1 else "L-shape"
    ask = sys.argv[2] if len(sys.argv) > 2 else None
    p = design_roof_llm(PLANS[plan], seed=1, request=ask)
    print(f"source   : {p['source']}")
    if p.get("llm_note"):
        print(f"note     : {p['llm_note']}")
    print(f"recipe   : {p['recipe_name']} ({p['recipe_id']})")
    print(f"style    : {p['roof_style']}  zones={len(p.get('zones') or [])}")
    if p.get("rationale"):
        print(f"rationale: {p['rationale']}")
