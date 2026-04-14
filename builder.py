import numpy as np
import trimesh

def export_to_obj(cleaned_lines, output_file="apartment.obj"):
    SCALE, WALL_H, WALL_T = 0.01, 3.0, 0.15
    walls_3d = []

    for l in cleaned_lines:
        x1, y1, x2, y2 = map(float, l)
        lx1, ly1, lx2, ly2 = x1*SCALE, y1*SCALE, x2*SCALE, y2*SCALE
        length = np.sqrt((lx2-lx1)**2 + (ly2-ly1)**2)
        
        if length < 0.1: continue 

        cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        wall = trimesh.creation.box(extents=(length, WALL_T, WALL_H))
        
        angle = np.arctan2((ly2-ly1), (lx2-lx1))
        rot = trimesh.transformations.rotation_matrix(angle, [0,0,1])
        wall.apply_transform(rot)
        wall.apply_translation([cx, cy, WALL_H/2])
        walls_3d.append(wall)

    if walls_3d:
        full_model = trimesh.util.concatenate(walls_3d)
        full_model.export(output_file)
        print(f"✅ Blender-ready file saved: {output_file}")
    else:
        print("❌ No walls to build.")