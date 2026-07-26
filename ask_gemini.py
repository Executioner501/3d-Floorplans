"""
ask_gemini.py — optional LLM roof designer (ENGINE = "llm")
===========================================================
Sends the floor-plan image, grounded by the measured footprint metrics,
to Gemini and asks for a roof style plus parameters. The answer is NOT
trusted: `roof_ai.validate_and_repair` clamps it into structurally valid
ranges before `roof_generator` builds anything.

This module is entirely optional. Without `GEMINI_API_KEY` set it
returns None and `roof_ai.design_roof` falls back to the local
footprint-conditioned prior, which needs no network and no key.

Setup:
    cp .env.example .env      # then put your key in it
    # or: export GEMINI_API_KEY=...

The default engine is "rag" (see roof_knowledge.py), which retrieves
real architectural compositions offline and generally produces richer
multi-volume roofs than a single-style LLM answer. Prefer it unless you
specifically want the LLM in the loop.
"""
import os
import json

# Style vocabulary the geometry engine can actually build. Keep this in
# sync with roof_generator.NEW_ENGINE_STYLES (minus "composite", which is
# reserved for the multi-zone RAG path) plus roof_ai.LEGACY_STYLES.
STYLES = [
    "gable", "cross-gable", "hip", "pyramid", "dutch-gable", "gambrel",
    "mansard", "saltbox", "butterfly", "skillion", "flat-modern",
    "flat", "split-level", "mono-pitch", "shed",
]

MATERIALS = [
    "terracotta", "clay-brown", "asphalt-dark", "asphalt-gray", "slate-blue",
    "metal-seam", "metal-green", "metal-red", "cedar-shake", "membrane-white",
    "matte-black", "solar-black", "green-roof",
]

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "roof_style":    {"type": "STRING", "enum": STYLES},
        "material":      {"type": "STRING", "enum": MATERIALS},
        "pitch_angle":   {"type": "NUMBER"},
        "overhang":      {"type": "NUMBER"},
        "has_dormers":   {"type": "BOOLEAN"},
        "dormer_count":  {"type": "INTEGER"},
        "has_chimney":   {"type": "BOOLEAN"},
        "has_skylight":  {"type": "BOOLEAN"},
        "has_canopy":    {"type": "BOOLEAN"},
        "canopy_depth":  {"type": "NUMBER"},
        "rationale":     {"type": "STRING"},
    },
    "required": ["roof_style", "material", "pitch_angle", "overhang"],
}

_PROMPT = """You are an experienced residential architect. Study this floor
plan and design the roof for it.

Choose exactly one `roof_style` from the allowed list:
  gable, cross-gable   — pitched prisms; cross-gable suits L/T plans
  hip, pyramid         — hipped surfaces; pyramid only for near-square plans
  dutch-gable          — hip with a small gablet at the ridge
  gambrel, mansard     — barn / French profiles, steep lower slopes
  saltbox              — asymmetric ridge, one long rear slope
  butterfly            — inverted V, central valley; modern statement
  skillion             — single sloping plane; modern, low pitch
  flat-modern          — flat with a parapet
  flat, split-level, mono-pitch, shed  — legacy slab styles

Then decide:
  pitch_angle    degrees; 0 for flat styles, 7-14 butterfly, 8-20 skillion,
                 22-35 hip, 24-45 gable, 60-72 mansard
  overhang       eave projection past the wall, 0.15-1.0 m
  material       one of the allowed palette entries
  has_dormers    dormers only make sense on gable/gambrel/mansard/saltbox
  dormer_count   1-3
  has_chimney    plausible on pitched traditional roofs
  has_skylight   plausible on flat-modern / skillion / butterfly
  has_canopy     entrance canopy over the front door
  canopy_depth   0.8-2.5 m
  rationale      one sentence on why this roof suits this plan

Match the style to the FOOTPRINT GEOMETRY given below, not just to taste:
elongated plans favour gable/saltbox/skillion, concave multi-wing plans
favour cross-gable/hip, compact squares favour pyramid/hip/flat-modern.
"""


def _hint_block(footprint_hint):
    if not footprint_hint:
        return ""
    lines = "\n".join(f"  {k}: {v}" for k, v in footprint_hint.items())
    return f"\n\nMEASURED FOOTPRINT (authoritative — trust these over the image):\n{lines}\n"


def get_roof_parameters(image_path="floorplan.png", footprint_hint=None,
                        model_name="gemini-2.5-flash"):
    """Ask Gemini for a roof design. Returns a params dict, or None so the
    caller falls back to the local prior.

    `footprint_hint` is the measured geometry (aspect, convexity, area_m2,
    wings) that roof_ai extracts before calling — it grounds the model in
    the real plan instead of leaving it to eyeball the drawing.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ℹ️  GEMINI_API_KEY not set; using the local roof prior instead.")
        return None

    if not os.path.exists(image_path):
        print(f"⚠️  Could not find {image_path}; using the local roof prior.")
        return None

    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError as e:
        print(f"ℹ️  LLM designer unavailable ({e}); using the local roof prior.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    print("🧠  Asking Gemini to design the roof...")
    try:
        response = model.generate_content(
            [Image.open(image_path), _PROMPT + _hint_block(footprint_hint)],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=_SCHEMA))
        data = json.loads(response.text)
    except Exception as e:
        print(f"❌  Gemini API error ({e}); using the local roof prior.")
        return None

    why = data.pop("rationale", "")
    print(f"✅  Gemini chose {data.get('roof_style')} in "
          f"{data.get('material')}" + (f" — {why}" if why else ""))
    return data


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "floorplan.png"
    print(get_roof_parameters(img, footprint_hint={
        "aspect": 1.4, "convexity": 0.97, "area_m2": 141.0, "wings": 1}))
