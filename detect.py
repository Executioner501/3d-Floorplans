import numpy as np

def process_yolo_results(results, scale_factor=0.01):
    """
    Converts YOLO bounding boxes into structured data for the 3D builder.
    """
    walls = []
    doors = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # xywh = [center_x, center_y, width, height]
            cx, cy, w, h = box.xywh[0].tolist()
            
            # Determine orientation (Vertical if height > width)
            is_vert = h > w
            angle = 1.5708 if is_vert else 0  # 90 degrees in radians
            
            data = {
                "pos": (cx, cy),
                "w": w,
                "h": h,
                "angle": angle
            }

            if cls == 1: # 'wall' class in your YAML
                walls.append(data)
            elif cls == 0: # 'door' class in your YAML
                doors.append(data)
                
    return walls, doors