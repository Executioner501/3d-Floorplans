import os
import json
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
os.environ["GEMINI_API_KEY"] = "paste-your-gemini-api-key-here"  # Replace with your actual Gemini API key


def get_roof_parameters(image_path="floorplan.png"):
    """
    Sends the floorplan to Gemini and asks for modern roof parameters.
    Returns a rich dict that builder.py can use directly.
    """
    api_key = os.getenv("api_key")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)

    if not os.path.exists(image_path):
        print(f"⚠️  Could not find {image_path}")
        return _fallback()

    img   = Image.open(image_path)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """
    You are an award-winning modern architect. Study this floorplan carefully and design
    a CONTEMPORARY roof system. Choose ONE of these styles that best suits the footprint:

    • "flat"        – Flat roof with a parapet lip. Clean, minimalist. Common on square/
                      rectangular plans.
    • "split-level" – Two flat sections at different heights (right wing 1 m taller).
                      Creates a dramatic stepped silhouette. Best for L-shaped or wide plans.
    • "mono-pitch"  – Single sloping plane, one edge higher. Sleek, modern shed look.
                      Good for narrow or rectangular plans.
    • "shed"        – Steep mono-pitch with a very large front overhang. Bold and dramatic.

    Also decide:
    - overhang:       How far the slab extends past the walls (0.3 – 0.8 m).
    - slab_thickness: Roof slab depth (0.15 – 0.30 m).
    - pitch_angle:    Slope in degrees for mono-pitch/shed styles (5 – 20). Set 0 for flat.
    - has_parapet:    true/false — raised border around flat roofs.
    - parapet_height: Height of the parapet lip (0.3 – 0.8 m).
    - has_canopy:     true/false — entrance canopy projecting from front wall.
    - canopy_depth:   How deep the canopy extends outward (0.8 – 2.5 m).
    - has_railing:    true/false — rooftop terrace railings on flat-roof parapets.

    Favour SPLIT-LEVEL or MONO-PITCH to create a modern, architecturally interesting result.
    Only choose "flat" for very simple square footprints.
    """

    schema = {
        "type": "OBJECT",
        "properties": {
            "roof_style":      {"type": "STRING"},
            "overhang":        {"type": "NUMBER"},
            "slab_thickness":  {"type": "NUMBER"},
            "pitch_angle":     {"type": "NUMBER"},
            "has_parapet":     {"type": "BOOLEAN"},
            "parapet_height":  {"type": "NUMBER"},
            "has_canopy":      {"type": "BOOLEAN"},
            "canopy_depth":    {"type": "NUMBER"},
            "has_railing":     {"type": "BOOLEAN"},
        },
        "required": [
            "roof_style", "overhang", "slab_thickness", "pitch_angle",
            "has_parapet", "parapet_height", "has_canopy", "canopy_depth",
            "has_railing"
        ]
    }

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=schema
    )

    print("🧠  Asking Gemini to design the roof...")
    try:
        response   = model.generate_content(
            [img, prompt],
            generation_config=generation_config
        )
        roof_data  = json.loads(response.text)
        print(f"✅  Gemini chose: {roof_data}")
        return roof_data

    except Exception as e:
        print(f"❌  Gemini API Error: {e}")
        return _fallback()


def _fallback():
    """Safe defaults if Gemini fails — still looks modern."""
    return {
        "roof_style":     "split-level",
        "overhang":        0.4,
        "slab_thickness":  0.2,
        "pitch_angle":     0,
        "has_parapet":     True,
        "parapet_height":  0.55,
        "has_canopy":      True,
        "canopy_depth":    1.6,
        "has_railing":     True,
    }


if __name__ == "__main__":
    print(get_roof_parameters("floorplan.png"))