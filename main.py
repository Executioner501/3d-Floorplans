
from refine import groq_straighten, math_snap,identify_door_gaps
from builder import export_to_obj
from ultralytics import YOLO
from detect import process_yolo_results
from ask_gemini import get_roof_parameters
# --- CONFIG ---
API_KEY = "paste-your-groq-api-key-here"  # Replace with your actual Gemini API key
INPUT_IMG = "floorplan.png"

def run_pipeline():
    # 1. Detect
    print("🔍 Extracting lines from image...")
    model=YOLO('best_doors.pt')
    results = model.predict(source=INPUT_IMG, conf=0.3)
    
    # 2. Process Data
    walls, doors = process_yolo_results(results)
    print(f"📊 Found {len(walls)} walls and {len(doors)} doors.")

    roof_params = get_roof_parameters(INPUT_IMG)
    
    # 3. 3D Build
    export_to_obj(walls, doors,roof_params=roof_params)

if __name__ == "__main__":
    run_pipeline()