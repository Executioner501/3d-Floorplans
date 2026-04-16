import json
import math
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  
def groq_straighten(raw_lines, api_key):
    client = Groq(api_key=os.getenv("api"))
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
def identify_door_gaps(lines, min_gap=70, max_gap=130):
    """
    Analyzes gaps between collinear wall segments to find potential doors.
    Returns a list of door dictionaries with position, width, and rotation.
    """
    doors = []
    # Using a copy to avoid mutating the original wall list
    processed_pairs = set()

    for i, line_a in enumerate(lines):
        for j, line_b in enumerate(lines):
            if i >= j: continue
            
            # Check if lines are on the same axis (Collinear)
            is_horiz = (line_a[1] == line_a[3] == line_b[1] == line_b[3])
            is_vert = (line_a[0] == line_a[2] == line_b[0] == line_b[2])
            
            if not (is_horiz or is_vert):
                continue

            # Find the distance between the closest endpoints
            # We check all 4 endpoint combinations between Line A and Line B
            points_a = [(line_a[0], line_a[1]), (line_a[2], line_a[3])]
            points_b = [(line_b[0], line_b[1]), (line_b[2], line_b[3])]
            
            best_dist = float('inf')
            midpoint = (0, 0)

            for p1 in points_a:
                for p2 in points_b:
                    d = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    if d < best_dist:
                        best_dist = d
                        midpoint = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

            # If the gap falls within standard door sizes (scaled units)
            if min_gap <= best_dist <= max_gap:
                doors.append({
                    "pos": midpoint,
                    "width": best_dist,
                    "orientation": "horizontal" if is_horiz else "vertical",
                    "angle": 0 if is_horiz else math.pi/2
                })
                
    return doors