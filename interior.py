import numpy as np
import trimesh
import os

def export_interior(walls, doors=None, output_file="no_roof.obj"):
    SCALE = 0.01
    WALL_H = 3.0
    FLOOR_THICKNESS = 0.05
    
    # RGB Colors
    WALL_COLOR = [255, 253, 208, 255] # Cream
    FLOOR_COLOR = [50, 50, 50, 255]   # Dark Charcoal
    
    components = []
    
    # We will track the bounding box manually for the floor
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # 1. GENERATE WALLS
    for w_data in walls:
        cx, cy = w_data['pos'][0] * SCALE, w_data['pos'][1] * SCALE
        w_val, h_val = w_data['w'] * SCALE, w_data['h'] * SCALE
        
        # Create wall
        wall = trimesh.creation.box(extents=(w_val, h_val, WALL_H))
        
        # Apply Vertex Colors (more reliable than face_colors for OBJ)
        wall.visual = trimesh.visual.ColorVisuals(mesh=wall, vertex_colors=[WALL_COLOR]*len(wall.vertices))
        
        # Move wall UP so it sits ON the Z=0 plane
        wall.apply_translation([cx, cy, WALL_H/2])
        components.append(wall)
        
        # Track boundaries
        min_x = min(min_x, cx - w_val/2)
        max_x = max(max_x, cx + w_val/2)
        min_y = min(min_y, cy - h_val/2)
        max_y = max(max_y, cy + h_val/2)

    # 2. GENERATE DOORS
    if doors and os.path.exists("door.obj"):
        try:
            base_door = trimesh.load("door.obj")
            orig_w = base_door.bounding_box.extents[0]
            for d in doors:
                door = base_door.copy()
                # Stand door up
                lift = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
                door.apply_transform(lift)
                
                # Scale
                target_w = max(d['w'], d['h']) * SCALE
                door.apply_scale(target_w / orig_w)
                
                # Rotate & Translate (sitting at Z=0)
                rot = trimesh.transformations.rotation_matrix(d['angle'], [0, 0, 1])
                door.apply_transform(rot)
                door.apply_translation([d['pos'][0]*SCALE, d['pos'][1]*SCALE, 0])
                components.append(door)
        except Exception as e:
            print(f"⚠️ Door Error: {e}")

    # 3. GENERATE FLOOR (Anchor at Z=0)
    if components:
        f_w = (max_x - min_x) + 1.0
        f_d = (max_y - min_y) + 1.0
        f_cx = (min_x + max_x) / 2
        f_cy = (min_y + max_y) / 2

        floor = trimesh.creation.box(extents=(f_w, f_d, FLOOR_THICKNESS))
        floor.visual = trimesh.visual.ColorVisuals(mesh=floor, vertex_colors=[FLOOR_COLOR]*len(floor.vertices))
        
        # Place the floor slightly BELOW Z=0 so it doesn't "fight" with the walls
        floor.apply_translation([f_cx, f_cy, -FLOOR_THICKNESS/2])
        components.append(floor)

    # 4. EXPORT
    if components:
        full_model = trimesh.util.concatenate(components)
        
        # Stand the entire house up for Blender/Viewers
        # We use a standard 90-degree rotation
        stand_up = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        full_model.apply_transform(stand_up)
        
        # Exporting with specific flags to force color embedding
        full_model.export(output_file, include_color=True)
        print(f"✅ Model saved: {output_file}")