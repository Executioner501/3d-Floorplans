import numpy as np
import trimesh
from shapely.geometry import box, Polygon, LineString
from shapely.ops import unary_union
import os

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
WALL_COLOR    = [245, 245, 245, 255]
FLOOR_COLOR   = [230, 225, 220, 255]
ROOF_COLOR    = [55,  58,  60,  255]
PARAPET_COLOR = [38,  40,  42,  255]
COLUMN_COLOR  = [200, 195, 190, 255]
RAIL_COLOR    = [80,  80,  80,  255]
STAIR_COLOR   = [210, 205, 200, 255]


# ─────────────────────────────────────────────
#  HELPER: TILTED SLAB
# ─────────────────────────────────────────────
def _make_tilted_slab(poly, slab_thickness, base_z,
                      min_x, max_x, pitch_deg, tilt_dir):
    slab = trimesh.creation.extrude_polygon(poly, height=slab_thickness)
    slab.apply_translation([0, 0, base_z])
    angle_rad = np.radians(pitch_deg)
    pivot_x   = max_x if tilt_dir > 0 else min_x
    pivot_y   = poly.centroid.y
    rot = trimesh.transformations.rotation_matrix(
        angle_rad * tilt_dir, [0, 1, 0],
        point=[pivot_x, pivot_y, base_z])
    slab.apply_transform(rot)
    return slab


# ─────────────────────────────────────────────
#  HELPER: RAILING  (outer perimeter only, with gap zones)
# ─────────────────────────────────────────────
def _make_railing(poly, base_z,
                  post_h=0.9, post_w=0.05, spacing=0.8,
                  gap_zones=None):
    """
    gap_zones: list of (cx, cy, radius) — posts inside any circle are skipped.
               Use this to punch a hole in the railing where stairs land.
    """
    parts  = []
    coords = list(poly.exterior.coords)
    gap_zones = gap_zones or []

    def _in_gap(px, py):
        for (gx, gy, gr) in gap_zones:
            if np.hypot(px - gx, py - gy) < gr:
                return True
        return False

    for i in range(len(coords) - 1):
        x0, y0 = coords[i];  x1, y1 = coords[i + 1]
        seg_len = np.hypot(x1 - x0, y1 - y0)
        n_posts = max(2, int(seg_len / spacing))
        for k in range(n_posts):
            t  = k / n_posts
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)
            if _in_gap(px, py):
                continue
            post = trimesh.creation.box(extents=(post_w, post_w, post_h))
            post.visual.face_colors = RAIL_COLOR
            post.apply_translation([px, py, base_z + post_h / 2])
            parts.append(post)

    # Continuous top rail — thin extruded ring
    try:
        inner     = poly.buffer(-post_w, join_style=2)
        rail_ring = poly.difference(inner)
        rail_mesh = trimesh.creation.extrude_polygon(rail_ring, height=post_w)
        rail_mesh.visual.face_colors = RAIL_COLOR
        rail_mesh.apply_translation([0, 0, base_z + post_h])
        parts.append(rail_mesh)
    except Exception:
        pass
    return parts


# ─────────────────────────────────────────────
#  HELPER: ENTRANCE CANOPY  (door-aligned)
# ─────────────────────────────────────────────
def _make_canopy(cx, front_y, base_z,
                 width=2.4, depth=2.0, thickness=0.12, col_size=0.20):
    parts  = []
    half_w = width / 2
    slab   = trimesh.creation.box(extents=(width, depth, thickness))
    slab.visual.face_colors = ROOF_COLOR
    slab.apply_translation([cx, front_y - depth / 2, base_z - thickness / 2])
    parts.append(slab)
    col_h = base_z
    for side_x in [cx - half_w + col_size / 2,
                   cx + half_w - col_size / 2]:
        col = trimesh.creation.box(extents=(col_size, col_size, col_h))
        col.visual.face_colors = COLUMN_COLOR
        col.apply_translation([side_x,
                                front_y - depth + col_size / 2,
                                col_h / 2])
        parts.append(col)
    return parts


# ─────────────────────────────────────────────
#  HELPER: STEP RISER between slab levels
# ─────────────────────────────────────────────
def _make_step_riser(x_edge, min_y, max_y, low_z, high_z, thickness=0.25):
    height = high_z - low_z
    depth  = max_y  - min_y
    riser  = trimesh.creation.box(extents=(thickness, depth, height))
    riser.visual.face_colors = WALL_COLOR
    riser.apply_translation([x_edge, (min_y + max_y) / 2,
                              low_z + height / 2])
    return riser


# ─────────────────────────────────────────────
#  HELPER: EXTERIOR STAIRCASE
# ─────────────────────────────────────────────
def _make_stairs(start_x, start_y, top_z,
                 direction="left",
                 step_w=1.2, step_d=0.30, step_h=0.18,
                 wall_thickness=0.12):
    """
    A single-flight exterior staircase rising from ground (z=0) to top_z.
    direction: "left" | "right" — which side of the building.

    Returns (parts, gap_zone) where gap_zone=(cx, cy, radius) tells
    the railing function to leave a gap at the stair landing.
    """
    parts   = []
    n_steps = max(3, int(top_z / step_h))
    actual_step_h = top_z / n_steps

    for i in range(n_steps):
        # Each step is a box; steps extend outward in the -Y direction
        step_depth_offset = (n_steps - 1 - i) * step_d
        z_base  = i * actual_step_h
        step    = trimesh.creation.box(
            extents=(step_w, step_d, actual_step_h))
        step.visual.face_colors = STAIR_COLOR

        # Position: extend outward from the building face
        if direction == "left":
            step.apply_translation([
                start_x,
                start_y - step_depth_offset - step_d / 2,
                z_base + actual_step_h / 2
            ])
        else:
            step.apply_translation([
                start_x,
                start_y + step_depth_offset + step_d / 2,
                z_base + actual_step_h / 2
            ])
        parts.append(step)

    # Side walls (thin slabs following the stair slope)
    stringer_h   = top_z
    stringer_len = n_steps * step_d
    for side_offset in [step_w / 2, -step_w / 2]:
        stringer = trimesh.creation.box(
            extents=(wall_thickness, stringer_len, stringer_h / 2))
        stringer.visual.face_colors = WALL_COLOR
        if direction == "left":
            stringer.apply_translation([
                start_x + side_offset,
                start_y - stringer_len / 2,
                stringer_h / 4
            ])
        else:
            stringer.apply_translation([
                start_x + side_offset,
                start_y + stringer_len / 2,
                stringer_h / 4
            ])
        parts.append(stringer)

    # Gap zone centre = landing point at top of stairs
    if direction == "left":
        gap_cx = start_x
        gap_cy = start_y
    else:
        gap_cx = start_x
        gap_cy = start_y
    gap_zone = (gap_cx, gap_cy, step_w * 0.9)

    return parts, gap_zone


# ─────────────────────────────────────────────
#  HELPER: ELEVATED PAVILION on circular pillars
# ─────────────────────────────────────────────
def _make_elevated_pavilion(bounds, pillar_base_z,
                             slab_thickness,
                             pillar_gap=1.5,
                             pillar_radius=0.20,
                             inset=0.6):
    """
    Floating slab on 4 circular (cylinder) pillars.
    NO railing on top — as requested.
    Pillars are inset from the pavilion corners so they sit visually inside the slab.
    """
    parts = []
    min_x, max_x, min_y, max_y = bounds

    pav_min_x = min_x + inset
    pav_max_x = max_x - inset
    pav_min_y = min_y + inset
    pav_max_y = max_y - inset

    if pav_max_x <= pav_min_x or pav_max_y <= pav_min_y:
        return parts

    pavilion_z = pillar_base_z + pillar_gap

    # ── 4 CIRCULAR corner pillars (inset from pavilion edge) ──────
    col_inset = pillar_radius + 0.10   # push column centre away from corner
    corners = [
        (pav_min_x + col_inset, pav_min_y + col_inset),
        (pav_max_x - col_inset, pav_min_y + col_inset),
        (pav_min_x + col_inset, pav_max_y - col_inset),
        (pav_max_x - col_inset, pav_max_y - col_inset),
    ]
    for px, py in corners:
        pillar = trimesh.creation.cylinder(
            radius=pillar_radius,
            height=pillar_gap,
            sections=16)        # smooth circle
        pillar.visual.face_colors = COLUMN_COLOR
        pillar.apply_translation([px, py, pillar_base_z + pillar_gap / 2])
        parts.append(pillar)

    # ── Pavilion slab ─────────────────────────────────────────────
    pav_poly = box(pav_min_x, pav_min_y, pav_max_x, pav_max_y)
    pav_slab = trimesh.creation.extrude_polygon(pav_poly, height=slab_thickness)
    pav_slab.visual.face_colors = ROOF_COLOR
    pav_slab.apply_translation([0, 0, pavilion_z])
    parts.append(pav_slab)

    # ── Thin parapet ONLY (no railing) ────────────────────────────
    try:
        inner        = pav_poly.buffer(-0.15, join_style=2)
        parapet_ring = pav_poly.difference(inner)
        parapet      = trimesh.creation.extrude_polygon(parapet_ring, height=0.35)
        parapet.visual.face_colors = PARAPET_COLOR
        parapet.apply_translation([0, 0, pavilion_z + slab_thickness])
        parts.append(parapet)
    except Exception:
        pass

    # ── NO railing on pavilion (as requested) ─────────────────────

    return parts


# ─────────────────────────────────────────────
#  MAIN EXPORT
# ─────────────────────────────────────────────
def export_to_obj(walls, doors=None, roof_params=None,
                  output_file="apartment.obj"):

    SCALE           = 0.01
    WALL_H          = 3.0
    FLOOR_THICKNESS = 0.05

    components = []
    all_points = []

    # ── 1. WALLS ─────────────────────────────
    for w in walls:
        cx = w['pos'][0] * SCALE;  cy = w['pos'][1] * SCALE
        ww = w['w']      * SCALE;  wh = w['h']      * SCALE
        wall = trimesh.creation.box(extents=(ww, wh, WALL_H))
        wall.visual.face_colors = WALL_COLOR
        wall.apply_translation([cx, cy, WALL_H / 2])
        components.append(wall)
        all_points.append(w['pos'])

    # ── 2. DOORS ─────────────────────────────
    if doors and os.path.exists("DOOR.obj"):
        try:
            base_door  = trimesh.load("DOOR.obj")
            orig_width = base_door.bounding_box.extents[0]
            for d in doors:
                di = base_door.copy()
                di.apply_transform(
                    trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                target_w = max(d['w'], d['h']) * SCALE
                di.apply_scale(target_w / orig_width)
                di.apply_transform(
                    trimesh.transformations.rotation_matrix(d['angle'], [0, 0, 1]))
                di.apply_translation([d['pos'][0] * SCALE,
                                       d['pos'][1] * SCALE, 0])
                components.append(di)
        except Exception as e:
            print(f"⚠️ Door Error: {e}")

    # ── 3. FLOOR ─────────────────────────────
    if not all_points:
        _finish(components, output_file)
        return

    pts    = np.array(all_points) * SCALE
    gmin_x = float(pts[:, 0].min());  gmax_x = float(pts[:, 0].max())
    gmin_y = float(pts[:, 1].min());  gmax_y = float(pts[:, 1].max())

    floor = trimesh.creation.box(
        extents=((gmax_x - gmin_x) + 1.0,
                 (gmax_y - gmin_y) + 1.0,
                 FLOOR_THICKNESS))
    floor.visual.face_colors = FLOOR_COLOR
    floor.apply_translation([(gmin_x + gmax_x) / 2,
                               (gmin_y + gmax_y) / 2,
                               -FLOOR_THICKNESS / 2])
    components.append(floor)

    # ── 4. ROOF ──────────────────────────────
    if not roof_params:
        _finish(components, output_file)
        return

    overhang       = roof_params.get("overhang",       0.4)
    slab_thickness = roof_params.get("slab_thickness", 0.2)
    parapet_height = roof_params.get("parapet_height", 0.55)
    style          = roof_params.get("roof_style",     "split-level")
    pitch_angle    = roof_params.get("pitch_angle",    8)
    has_parapet    = roof_params.get("has_parapet",    True)
    has_canopy     = roof_params.get("has_canopy",     True)
    canopy_depth   = roof_params.get("canopy_depth",   1.8)
    has_railing    = roof_params.get("has_railing",    True)

    # ── Wing split ────────────────────────────────────────────────
    all_cx      = [w['pos'][0] * SCALE for w in walls]
    mid_x       = sum(all_cx) / len(all_cx)
    left_walls  = [w for w in walls if w['pos'][0] * SCALE <= mid_x]
    right_walls = [w for w in walls if w['pos'][0] * SCALE >  mid_x]

    volumes = [
        {"name": "Left",  "walls": left_walls,  "h_offset": 0.0},
        {"name": "Right", "walls": right_walls, "h_offset": 0.0},
    ]
    vol_bounds   = {}
    slab_polys   = []    # collect all wing polys for unified parapet/railing

    for vol in volumes:
        if not vol["walls"]:
            continue

        vmin_x = min(w['pos'][0]*SCALE - w['w']*SCALE/2 for w in vol["walls"])
        vmax_x = max(w['pos'][0]*SCALE + w['w']*SCALE/2 for w in vol["walls"])
        vmin_y = min(w['pos'][1]*SCALE - w['h']*SCALE/2 for w in vol["walls"])
        vmax_y = max(w['pos'][1]*SCALE + w['h']*SCALE/2 for w in vol["walls"])

        if vol["name"] == "Left":
            vmax_x = max(vmax_x, mid_x)
        else:
            vmin_x = min(vmin_x, mid_x)

        vol_bounds[vol["name"]] = (vmin_x, vmax_x, vmin_y, vmax_y)
        wing_box  = box(vmin_x, vmin_y, vmax_x, vmax_y)
        roof_poly = wing_box.buffer(overhang, join_style=2)
        slab_polys.append(roof_poly)
        base_z    = WALL_H + vol["h_offset"]

        try:
            # ── A. SLAB ────────────────────────────────────
            if style in ("mono-pitch", "shed"):
                tilt_dir     = 1 if vol["name"] == "Left" else -1
                actual_pitch = pitch_angle * (1.6 if style == "shed" else 1.0)
                slab = _make_tilted_slab(roof_poly, slab_thickness, base_z,
                                         vmin_x, vmax_x, actual_pitch, tilt_dir)
            else:
                slab = trimesh.creation.extrude_polygon(
                    roof_poly, height=slab_thickness)
                slab.apply_translation([0, 0, base_z])

            slab.visual.face_colors = ROOF_COLOR
            components.append(slab)

        except Exception as e:
            print(f"⚠️ {vol['name']} Slab Error: {e}")

    # ── B. UNIFIED PARAPET + RAILING around full combined footprint ──
    #   Merge both wing polygons → single outer shell → one parapet, one railing.
    #   This eliminates all inner intersection artefacts.
    if has_parapet and style in ("flat", "split-level") and slab_polys:
        try:
            combined_poly  = unary_union(slab_polys)
            # Simplify to remove shared-edge artefacts
            combined_poly  = combined_poly.simplify(0.02, preserve_topology=True)

            inner_poly     = combined_poly.buffer(-0.22, join_style=2)
            parapet_ring   = combined_poly.difference(inner_poly)

            parapet_top_z  = WALL_H + slab_thickness
            parapet = trimesh.creation.extrude_polygon(
                parapet_ring, height=parapet_height)
            parapet.visual.face_colors = PARAPET_COLOR
            parapet.apply_translation([0, 0, parapet_top_z])
            components.append(parapet)

            # ── C. STAIRS on the left side of the building ────────
            # Position stairs at left-centre of the building
            stair_x = gmin_x - 1    # flush with left wall
            stair_y = (gmin_y + gmax_y) / 2  # mid-depth of building
            stair_top_z = WALL_H + slab_thickness + parapet_height

            stair_parts, stair_gap = _make_stairs(
                start_x   = stair_x,
                start_y   = stair_y,
                top_z     = stair_top_z,
                direction = "left",
                step_w    = 1.2,
                step_d    = 0.30,
                step_h    = 0.18)
            for p in stair_parts:
                components.append(p)

            # ── D. SINGLE UNIFIED RAILING with stair gap ──────────
            if has_railing:
                # Railing sits on top of the parapet
                rail_base_z  = parapet_top_z + parapet_height
                # Use inner_poly as the railing path (top of parapet inner edge)
                railing_poly = inner_poly.buffer(-0.04, join_style=2)
                for part in _make_railing(
                        railing_poly, rail_base_z,
                        post_h    = 0.90,
                        spacing   = 0.70,
                        gap_zones = [stair_gap]):
                    components.append(part)

        except Exception as e:
            print(f"⚠️ Unified Parapet/Railing Error: {e}")

    # ── E. ELEVATED PAVILION on the right wing (circular pillars) ─
    if "Right" in vol_bounds and style in ("flat", "split-level"):
        lower_slab_top = WALL_H + slab_thickness
        for part in _make_elevated_pavilion(
                bounds         = vol_bounds["Right"],
                pillar_base_z  = lower_slab_top,
                slab_thickness = slab_thickness,
                pillar_gap     = 1.5,
                pillar_radius  = 0.20,   # smooth cylinder columns
                inset          = 0.55):
            components.append(part)

    # ── F. SEAM RISER ─────────────────────────────────────────────
    if "Left" in vol_bounds and "Right" in vol_bounds:
        _, lmax_x, lmin_y, lmax_y = vol_bounds["Left"]
        rmin_x, _, rmin_y, rmax_y  = vol_bounds["Right"]
        join_x       = (lmax_x + rmin_x) / 2
        shared_min_y = min(lmin_y, rmin_y)
        shared_max_y = max(lmax_y, rmax_y)
        riser = _make_step_riser(
            join_x, shared_min_y, shared_max_y,
            WALL_H, WALL_H + slab_thickness, thickness=0.20)
        components.append(riser)

    # ── G. DOOR-ALIGNED ENTRANCE CANOPY ──────────────────────────
    if has_canopy:
        canopy_cx    = (gmin_x + gmax_x) / 2
        canopy_width = max((gmax_x - gmin_x) * 0.30, 2.0)

        if doors:
            front_door   = min(doors, key=lambda d: d['pos'][1])
            canopy_cx    = front_door['pos'][0] * SCALE
            door_w       = max(front_door['w'], front_door['h']) * SCALE
            canopy_width = max(door_w * 2.8, 2.2)

        canopy_z = WALL_H - 0.25
        for part in _make_canopy(
                cx=canopy_cx, front_y=gmin_y, base_z=canopy_z,
                width=canopy_width, depth=roof_params.get("canopy_depth", 1.8),
                thickness=0.12, col_size=0.20):
            components.append(part)

    _finish(components, output_file)


# ─────────────────────────────────────────────
def _finish(components, output_file):
    if not components:
        print("❌ No geometry generated.")
        return
    full_model = trimesh.util.concatenate(components)
    up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    full_model.apply_transform(up)
    full_model.export(output_file)
    print(f"✅ Model saved → {output_file}")