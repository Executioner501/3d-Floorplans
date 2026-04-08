import cv2
import numpy as np
import math
import trimesh

# ---- Load Image ----
img = cv2.imread("floorplan.png")
if img is None:
    print("Error: Image not found")
    exit()

img = cv2.resize(img, (800, 800))

# ---- Convert to grayscale ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# THRESHOLD:
# Value 50 ONLY pure black pixels become walls. 
# The gray furniture fill is completely ignored!
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# ERremove furniture lines:
# A 3x3 kernel will erase any line thinner than 3 pixels.
# The thick walls will survive, but the thin furniture lines will vanish.
kernel_open = np.ones((3, 3), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)


# Reconnect any small gaps in the walls caused by the previous step
kernel_close = np.ones((5, 5), np.uint8)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)

# ---- Edge detection ----
edges = cv2.Canny(clean, 50, 150)

# ---- Hough Line Transform ----
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=30,
    minLineLength=15,
    maxLineGap=50
)

# ---- Draw + STORE lines ----
line_img = img.copy()

ANGLE_THRESHOLD = 10
filtered_lines = []  

if lines is not None:
    print("Filtered wall lines:")

    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))

        if not (angle < ANGLE_THRESHOLD or abs(angle-90) < ANGLE_THRESHOLD):
            continue

        filtered_lines.append((x1, y1, x2, y2))


        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        print(f"({x1},{y1}) → ({x2},{y2})")

else:
    print("No lines detected")

# ---- Show 2D ----
cv2.imshow("Detected Walls", line_img)

# ---- 3D GENERATION  ----
SCALE = 0.01
WALL_HEIGHT = 3
WALL_THICKNESS = 0.1

walls_3d = []

for x1, y1, x2, y2 in filtered_lines:

    # scale down
    x1, y1, x2, y2 = [v * SCALE for v in (x1, y1, x2, y2)]

    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    if length < 0.05:
        continue

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    wall = trimesh.creation.box(
        extents=(length, WALL_THICKNESS, WALL_HEIGHT)
    )

    # rotate
    angle = np.arctan2((y2-y1), (x2-x1))
    rot = trimesh.transformations.rotation_matrix(angle, [0,0,1])
    wall.apply_transform(rot)

    # position
    wall.apply_translation([cx, cy, WALL_HEIGHT/2])

    walls_3d.append(wall)

# ---- Show 3D ----
if len(walls_3d) > 0:
    scene = trimesh.util.concatenate(walls_3d)
    scene.apply_translation(-scene.centroid)  # center view
    scene.show()
else:
    print("No walls for 3D")

# ---- Exit ----
while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
