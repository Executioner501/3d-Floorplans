import json
import math
from groq import Groq
import os

load_dotenv()
def groq_straighten(raw_lines, api_key):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""
    You are a precision geometry engine. I will provide a list of 2D line segments [x1, y1, x2, y2].
    Your goal is to RECTIFY these lines into a perfect architectural grid.
    
    STRICT RULES:
    1. ORTHOGONAL ONLY: Every line MUST be perfectly horizontal (y1 == y2) or perfectly vertical (x1 == x2). 
    2. SNAP TO GRID: If x1 and x2 are close, make them identical. If y1 and y2 are close, make them identical.
    3. CLOSE GAPS: If endpoints are within 20 units of each other, snap them to the exact same coordinate.
    4. PRESERVE STRUCTURE: Do not delete walls; just straighten them.
    
    Return ONLY a JSON object: {{"rectified_lines": [[x1, y1, x2, y2], ...]}}
    
    DATA:
    {json.dumps(raw_lines)}
    """

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0, # Zero randomness for math!
            response_format={"type": "json_object"}
        )
        response = json.loads(completion.choices[0].message.content)
        return response.get("rectified_lines", raw_lines)
    except Exception as e:
        print(f"Groq failed: {e}")
        return raw_lines
def math_snap(lines, threshold=25):
    """
    Snaps endpoints together while attempting to preserve horizontal/vertical alignment.
    """
    # 1. Force coordinates to floats to prevent math errors
    snapped = [[float(coord) for coord in l] for l in lines]
    
    for i in range(len(snapped)):
        for j in range(len(snapped)):
            if i == j: continue
            
            # Line I (Current line being moved)
            # Line J (The anchor line we are snapping TO)
            
            for idx_i in [0, 2]: # Endpoints of line I
                p_i = (snapped[i][idx_i], snapped[i][idx_i+1])
                
                for idx_j in [0, 2]: # Endpoints of line J
                    p_j = (snapped[j][idx_j], snapped[j][idx_j+1])
                    
                    dist = math.hypot(p_i[0] - p_j[0], p_i[1] - p_j[1])
                    
                    if 0 < dist < threshold:
                        # 📐 SMART SNAP LOGIC:
                        # Check if line I is Horizontal (y1 == y2) or Vertical (x1 == x2)
                        is_vert_i = (snapped[i][0] == snapped[i][2])
                        is_horiz_i = (snapped[i][1] == snapped[i][3])
                        
                        if is_vert_i:
                            # If vertical, only move the Y coordinate to meet the corner
                            # Keeping X exactly the same prevents the "slant"
                            snapped[i][idx_i+1] = p_j[1] 
                        elif is_horiz_i:
                            # If horizontal, only move the X coordinate
                            snapped[i][idx_i] = p_j[0]
                        else:
                            # Fallback: if it's already slanted, just snap fully
                            snapped[i][idx_i], snapped[i][idx_i+1] = p_j
                            
    return snapped