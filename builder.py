import numpy as np
import trimesh
import os

# Renamed 'cleaned_lines' to 'walls' to match your internal logic
def export_to_obj(walls, doors=None, output_file="apartment.obj"):
    # --- CONFIGURATION ---
    SCALE = 0.01
    WALL_H = 3.0
    FLOOR_THICKNESS = 0.05
    
    # Colors (R, G, B, A)
    WALL_COLOR = [255, 253, 208, 255]  # Light Greyish-White
    FLOOR_COLOR = [50, 50, 50, 255]    # Dark Charcoal
    
    components = []
    all_points = [] 

    # 1. GENERATE WALLS FROM AI BOXES
    for w_data in walls:
        # Scale coordinates and dimensions
        cx, cy = w_data['pos'][0] * SCALE, w_data['pos'][1] * SCALE
        width, thick = w_data['w'] * SCALE, w_data['h'] * SCALE
        
        # Create the box based on AI detected width/thickness
        wall = trimesh.creation.box(extents=(width, thick, WALL_H))
        wall.visual.face_colors = WALL_COLOR
        
        # Move to position
        wall.apply_translation([cx, cy, WALL_H/2])
        components.append(wall)
        
        # Collect points for floor calculation
        all_points.append(w_data['pos'])

    # 2. GENERATE DOORS
    if doors and os.path.exists("DOOR.obj"):
        try:
            base_door = trimesh.load("DOOR.obj")
            orig_width = base_door.bounding_box.extents[0]
            
            for d in doors:
                door_instance = base_door.copy()
                
                # Stand door up (Local Fix)
                lift = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
                door_instance.apply_transform(lift)
                
                # Scale to fit AI's detected width
                target_w = max(d['w'], d['h']) * SCALE
                door_instance.apply_scale(target_w / orig_width)
                
                # Rotate to match wall orientation
                rot = trimesh.transformations.rotation_matrix(d['angle'], [0, 0, 1])
                door_instance.apply_transform(rot)
                
                # Translate to position
                door_instance.apply_translation([d['pos'][0]*SCALE, d['pos'][1]*SCALE, 0])
                components.append(door_instance)
        except Exception as e:
            print(f"⚠️ Door Error: {e}")

    # 3. GENERATE FLOOR
    if all_points:
        pts = np.array(all_points) * SCALE
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        
        width = (max_x - min_x) + 1.0  # Slightly more padding
        depth = (max_y - min_y) + 1.0
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        floor = trimesh.creation.box(extents=(width, depth, FLOOR_THICKNESS))
        floor.visual.face_colors = FLOOR_COLOR
        floor.apply_translation([center_x, center_y, -FLOOR_THICKNESS/2])
        components.append(floor)

    # 4. EXPORT
    if components:
        full_model = trimesh.util.concatenate(components)
        
        # Stand the entire house up for Blender/Viewers
        stand_up_mat = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        full_model.apply_transform(stand_up_mat)
        
        full_model.export(output_file)
        print(f"✅ Enhanced Model saved: {output_file}")
    else:
        print("❌ No geometry generated.")