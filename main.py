
from refine import groq_straighten, math_snap,identify_door_gaps
from builder import export_to_obj
from ultralytics import YOLO
from detect import process_yolo_results
# --- CONFIG ---
API_KEY = "your_groq_key_here"
INPUT_IMG = "floorplan.png"

def run_pipeline():
    # 1. Detect
    print("🔍 Extracting lines from image...")
    model=YOLO('best_doors.pt')
    results = model.predict(source=INPUT_IMG, conf=0.3)
    
    # 2. Process Data
    walls, doors = process_yolo_results(results)
    print(f"📊 Found {len(walls)} walls and {len(doors)} doors.")
    
    # 3. 3D Build
    export_to_obj(walls, doors)

if __name__ == "__main__":
    run_pipeline()