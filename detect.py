import cv2
import numpy as np
import math

# ---- Load Image ----
img = cv2.imread("floorplan.png")
if img is None:
    print("Error: Image not found")
    exit()

# Resize for consistency
img = cv2.resize(img, (800, 800))

# ---- Convert to grayscale ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Smooth noise
gray = cv2.GaussianBlur(gray, (5,5), 0)

# ---- Threshold ----
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# ---- Morphological operations ----
kernel = np.ones((5,5), np.uint8)

# Remove small noise
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Reconnect walls
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

# ---- Strengthen walls (IMPORTANT FIX) ----
kernel_dilate = np.ones((3,3), np.uint8)
clean = cv2.dilate(clean, kernel_dilate, iterations=1)

# ---- Edge detection ----
edges = cv2.Canny(clean, 50, 150)

# ---- Hough Line Transform ----
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=60,        
    minLineLength=40,    
    maxLineGap=25       
)

# ---- Draw filtered lines ----
line_img = img.copy()

ANGLE_THRESHOLD = 10  # allow near horizontal/vertical

if lines is not None:
    print("Filtered wall lines:")

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Compute angle
        angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))

        # Keep only horizontal & vertical lines
        if not (angle < ANGLE_THRESHOLD or abs(angle-90) < ANGLE_THRESHOLD):
            continue

        # Draw line
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        print(f"({x1},{y1}) → ({x2},{y2})")

else:
    print("No lines detected")

# ---- Show results ----
cv2.imshow("Original", img)
cv2.imshow("Threshold", thresh)
cv2.imshow("Clean", clean)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Walls", line_img)

# ---- Exit on ESC ----
while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()