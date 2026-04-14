from detect import get_raw_lines
from refine import groq_straighten, math_snap
from builder import export_to_obj

# --- CONFIG ---
API_KEY = "your_groq_key_here"
INPUT_IMG = "floorplan.png"

def run_pipeline():
    # 1. Detect
    print("🔍 Extracting lines from image...")
    raw = get_raw_lines(INPUT_IMG)
    
    # 2. Refine (AI Straighten then Math Snap)
    print(f"🤖 AI Straightening {len(raw)} lines...")
    straight = groq_straighten(raw, API_KEY)
    
    print("📏 Snapping gaps with math...")
    final = math_snap(straight)
    
    # 3. Build
    print("🏗️ Generating 3D model...")
    export_to_obj(final)

if __name__ == "__main__":
    run_pipeline()