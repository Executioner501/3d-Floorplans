import cv2
import numpy as np

def get_raw_lines(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Floorplan image not found.")
    
    img = cv2.resize(img, (800, 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Cleanup furniture
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(clean, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=50)
    
    if lines is not None:
        return [line[0].tolist() for line in lines]
    return []