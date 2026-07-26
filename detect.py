import numpy as np


def _dist_to_wall(cx, cy, w):
    """Distance from a point to a wall's rectangle (0 when inside)."""
    dx = max(abs(cx - w["pos"][0]) - w["w"] / 2, 0.0)
    dy = max(abs(cy - w["pos"][1]) - w["h"] / 2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def snap_openings_to_walls(openings, walls, max_dist=45):
    """Clean up door/window detections before 3D placement:

    1. ATTACH — each opening is assigned to its nearest wall; anything
       farther than `max_dist` px from every wall is a false positive
       and is dropped.
    2. ALIGN — the opening is re-centred on that wall's centreline and
       oriented along the WALL's axis (its own bbox aspect is unreliable
       when the detector includes part of the swing arc).
    3. DEDUPE — YOLO often emits two boxes for one door (NMS misses
       near-identical overlaps); after snapping, openings on the same
       axis that overlap along the wall are merged (first kept).
    """
    snapped = []
    for o in openings:
        cx, cy = o["pos"]
        wi = min(range(len(walls)),
                 key=lambda i: _dist_to_wall(cx, cy, walls[i]), default=None)
        if wi is None or _dist_to_wall(cx, cy, walls[wi]) > max_dist:
            continue
        wall = walls[wi]
        length = max(o["w"], o["h"])
        horiz = wall["w"] >= wall["h"]
        # `wall` / `wall_t` let builder.py cut this opening out of exactly
        # the wall it belongs to instead of guessing by proximity again
        if horiz:
            lo = wall["pos"][0] - wall["w"] / 2 + length / 2
            hi = wall["pos"][0] + wall["w"] / 2 - length / 2
            cx = float(np.clip(cx, lo, max(hi, lo)))
            snapped.append({"pos": (cx, wall["pos"][1]),
                            "w": length, "h": wall["h"], "angle": 0,
                            "wall": wi, "wall_t": wall["h"]})
        else:
            lo = wall["pos"][1] - wall["h"] / 2 + length / 2
            hi = wall["pos"][1] + wall["h"] / 2 - length / 2
            cy = float(np.clip(cy, lo, max(hi, lo)))
            snapped.append({"pos": (wall["pos"][0], cy),
                            "w": wall["w"], "h": length, "angle": 1.5708,
                            "wall": wi, "wall_t": wall["w"]})
    deduped = []
    for o in snapped:
        o_len = max(o["w"], o["h"])
        dup = False
        for e in deduped:
            if o["angle"] != e["angle"]:
                continue
            dx = abs(o["pos"][0] - e["pos"][0])
            dy = abs(o["pos"][1] - e["pos"][1])
            along, cross = (dx, dy) if o["angle"] == 0 else (dy, dx)
            if cross < 14 and along < (o_len + max(e["w"], e["h"])) / 2 * 0.7:
                dup = True
                break
        if not dup:
            deduped.append(o)
    return deduped


def process_yolo_results(results, scale_factor=0.01):
    """
    Converts YOLO bounding boxes into structured data for the 3D builder.
    Model classes: 0 = door, 1 = wall, 2 = window.
    Returns (walls, doors, windows).
    """
    walls = []
    doors = []
    windows = []

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
            elif cls == 2: # 'window'
                windows.append(data)

    # attach openings to walls, fix orientation, drop dupes/strays
    doors = snap_openings_to_walls(doors, walls)
    windows = snap_openings_to_walls(windows, walls)

    return walls, doors, windows