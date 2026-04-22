"""
builder.py  –  Final fixed version
Fixes applied:
  1. Canopy: use gmin_y (actual front-wall face) for Y, not door-centre Y
  2. Doors: procedural fallback box when DOOR.obj is absent
  3. Stairs: subtract overhang from X so stairs never clip the roof edge
  4. Rotation: -np.pi/2 (confirmed correct by user)
"""
import numpy as np
import trimesh
from shapely.geometry import box as shp_box
from shapely.ops import unary_union
import os

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
WALL_COLOR    = [255, 253, 208, 255]   # warm cream
FLOOR_COLOR   = [50,  50,  50,  255]   # dark charcoal
ROOF_COLOR    = [55,  58,  60,  255]   # dark slate
PARAPET_COLOR = [38,  40,  42,  255]   # near-black
COLUMN_COLOR  = [200, 195, 190, 255]   # light concrete
RAIL_COLOR    = [90,  90,  90,  255]   # steel
STAIR_COLOR   = [210, 205, 200, 255]   # pale stone
DOOR_COLOR    = [120, 80,  50,  255]   # warm wood brown


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
#  Tilted slab (mono-pitch / shed)
# ─────────────────────────────────────────────
def _make_tilted_slab(poly, slab_thickness, base_z,
                      min_x, max_x, pitch_deg, tilt_dir):
    slab = trimesh.creation.extrude_polygon(poly, height=slab_thickness)
    slab.apply_translation([0, 0, base_z])
    rot = trimesh.transformations.rotation_matrix(
        np.radians(pitch_deg) * tilt_dir, [0, 1, 0],
        point=[(max_x if tilt_dir > 0 else min_x), poly.centroid.y, base_z])
    slab.apply_transform(rot)
    return slab


# ─────────────────────────────────────────────
#  Railing with gap zones
# ─────────────────────────────────────────────
def _make_railing(poly, base_z,
                  post_h=0.90, post_w=0.05, spacing=0.75,
                  gap_zones=None):
    """
    Posts around poly exterior + thin continuous top rail.
    gap_zones: [(cx, cy, radius)]  — posts inside any circle are skipped.
    """
    parts     = []
    gap_zones = gap_zones or []
    coords    = list(poly.exterior.coords)

    def _in_gap(px, py):
        return any(np.hypot(px - gx, py - gy) < gr for gx, gy, gr in gap_zones)

    for i in range(len(coords) - 1):
        x0, y0 = coords[i];  x1, y1 = coords[i + 1]
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / spacing))
        for k in range(n):
            t = k / n
            px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            if _in_gap(px, py):
                continue
            post = trimesh.creation.box(extents=(post_w, post_w, post_h))
            post.visual.face_colors = RAIL_COLOR
            post.apply_translation([px, py, base_z + post_h / 2])
            parts.append(post)

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
#  Exterior staircase with diagonal handrail
# ─────────────────────────────────────────────
def _make_stairs(stair_cx, stair_start_y, top_z,
                 step_w=1.30, step_d=0.28, rail_h=0.90):
    """
    Steps run in +Y direction (front-of-building → back).
    stair_cx must already be fully outside the building + overhang.
    Returns (parts, gap_zone, stair_end_y).
    """
    parts   = []
    n_steps = max(5, round(top_z / 0.175))
    sh      = top_z / n_steps
    run     = n_steps * step_d

    # Stacked tread boxes
    for i in range(n_steps):
        h_box = (i + 1) * sh
        tread = trimesh.creation.box(extents=(step_w, step_d, h_box))
        tread.visual.face_colors = STAIR_COLOR
        tread.apply_translation([stair_cx,
                                  stair_start_y + (i + 0.5) * step_d,
                                  h_box / 2])
        parts.append(tread)

    # Landing pad at top
    landing = trimesh.creation.box(extents=(step_w, 0.55, 0.06))
    landing.visual.face_colors = STAIR_COLOR
    landing.apply_translation([stair_cx,
                                stair_start_y + run + 0.275,
                                top_z + 0.03])
    parts.append(landing)

    # Posts + diagonal handrail on each side stringer
    POST_W = 0.07
    for sx in [stair_cx - step_w / 2, stair_cx + step_w / 2]:
        for i in range(n_steps + 1):
            post = trimesh.creation.box(extents=(POST_W, POST_W, rail_h))
            post.visual.face_colors = RAIL_COLOR
            post.apply_translation([sx,
                                     stair_start_y + i * step_d,
                                     i * sh + rail_h / 2])
            parts.append(post)

        # True diagonal handrail
        d_y   = run
        d_z   = top_z - sh
        diag  = np.hypot(d_y, d_z)
        angle = np.arctan2(d_z, d_y)
        hr    = trimesh.creation.box(extents=(POST_W * 0.7, diag, POST_W * 0.7))
        hr.apply_transform(
            trimesh.transformations.rotation_matrix(angle, [1, 0, 0]))
        hr.visual.face_colors = RAIL_COLOR
        hr.apply_translation([sx,
                               stair_start_y + run / 2,
                               sh + d_z / 2 + rail_h])
        parts.append(hr)

    stair_end_y = stair_start_y + run
    gap_zone    = (stair_cx, stair_end_y, step_w * 1.10)
    return parts, gap_zone, stair_end_y


# ─────────────────────────────────────────────
#  Entrance canopy  (FIXED Y position)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  HELPER: ENTRANCE CANOPY  (door-aligned)
# ─────────────────────────────────────────────
def _make_canopy(canopy_cx, front_wall_y, base_z,
                 width=2.4, depth=2.0, thickness=0.13, col_size=0.20):
    parts  = []
    half_w = width / 2

    # Slab: back edge touches front_wall_y, projects forward (+Y)
    slab = trimesh.creation.box(extents=(width, depth, thickness))
    slab.visual.face_colors = ROOF_COLOR
    slab.apply_translation([canopy_cx,
                             front_wall_y + depth / 2,  # <--- PLUS SIGN
                             base_z - thickness / 2])
    parts.append(slab)

    # Two support columns from ground to slab underside
    for col_x in [canopy_cx - half_w + col_size / 2,
                  canopy_cx + half_w - col_size / 2]:
        col = trimesh.creation.box(extents=(col_size, col_size, base_z - thickness))
        col.visual.face_colors = COLUMN_COLOR
        col.apply_translation([col_x,
                                front_wall_y + depth - col_size / 2, # <--- PLUS SIGN
                                (base_z - thickness) / 2])
        parts.append(col)
    return parts


# ─────────────────────────────────────────────
#  Elevated pavilion on 4 circular pillars (no railing on top)
# ─────────────────────────────────────────────
def _make_elevated_pavilion(bounds, pillar_base_z, slab_thickness,
                             pillar_gap=1.5, pillar_radius=0.22, inset=0.55):
    parts = []
    min_x, max_x, min_y, max_y = bounds
    px0, px1 = min_x + inset, max_x - inset
    py0, py1 = min_y + inset, max_y - inset
    if px1 <= px0 or py1 <= py0:
        return parts

    pav_z = pillar_base_z + pillar_gap
    ci    = pillar_radius + 0.12

    for (cx, cy) in [(px0 + ci, py0 + ci), (px1 - ci, py0 + ci),
                     (px0 + ci, py1 - ci), (px1 - ci, py1 - ci)]:
        pillar = trimesh.creation.cylinder(radius=pillar_radius,
                                           height=pillar_gap, sections=18)
        pillar.visual.face_colors = COLUMN_COLOR
        pillar.apply_translation([cx, cy, pillar_base_z + pillar_gap / 2])
        parts.append(pillar)

    pav_poly = shp_box(px0, py0, px1, py1)
    pav_slab = trimesh.creation.extrude_polygon(pav_poly, height=slab_thickness)
    pav_slab.visual.face_colors = ROOF_COLOR
    pav_slab.apply_translation([0, 0, pav_z])
    parts.append(pav_slab)

    try:
        inner_p  = pav_poly.buffer(-0.15, join_style=2)
        par_ring = pav_poly.difference(inner_p)
        parapet  = trimesh.creation.extrude_polygon(par_ring, height=0.35)
        parapet.visual.face_colors = PARAPET_COLOR
        parapet.apply_translation([0, 0, pav_z + slab_thickness])
        parts.append(parapet)
    except Exception:
        pass
    return parts


# ─────────────────────────────────────────────
#  Seam riser
# ─────────────────────────────────────────────
def _make_step_riser(x_edge, min_y, max_y, low_z, high_z, thickness=0.20):
    h     = high_z - low_z
    riser = trimesh.creation.box(extents=(thickness, max_y - min_y, h))
    riser.visual.face_colors = WALL_COLOR
    riser.apply_translation([x_edge, (min_y + max_y) / 2, low_z + h / 2])
    return riser


# ══════════════════════════════════════════════════════════════════
#  MAIN EXPORT
# ══════════════════════════════════════════════════════════════════

def export_to_obj(walls, doors=None, roof_params=None,
                  output_file=None):
    if output_file is None:
        output_file = "apartment.glb"
        print(f"ℹ️  No output file specified. Defaulting to '{output_file}'.")
    else:
        print(f"ℹ️  Exporting model to '{output_file}'...")

    SCALE           = 0.01
    WALL_H          = 3.0
    FLOOR_THICKNESS = 0.05

    components = []
    all_points = []

    # ════════════════════════════════════════
    #  1. WALLS  (exterior + interior partitions)
    # ════════════════════════════════════════
    for w_data in walls:
        cx    = w_data['pos'][0] * SCALE
        cy    = w_data['pos'][1] * SCALE
        width = w_data['w']      * SCALE
        thick = w_data['h']      * SCALE

        wall = trimesh.creation.box(extents=(width, thick, WALL_H))
        wall.visual.face_colors = WALL_COLOR
        wall.apply_translation([cx, cy, WALL_H / 2])
        components.append(wall)
        all_points.append(w_data['pos'])

    # ════════════════════════════════════════
    #  2. DOORS
    #     FIX: when DOOR.obj is absent (the common case) generate a
    #     procedural door box so doors are always visible.
    # ════════════════════════════════════════
    if doors:
        door_loaded = False
        base_door   = None
        orig_width  = None

        if os.path.exists("DOOR.obj"):
            try:
                base_door  = trimesh.load("DOOR.obj")
                orig_width = base_door.bounding_box.extents[0]
                door_loaded = True
            except Exception as e:
                print(f"⚠️ Could not load DOOR.obj: {e}")

        for d in doors:
            dw = max(d['w'], d['h']) * SCALE
            dh = WALL_H * 0.85          # door height ≈ 85 % of wall height
            dt = 0.08                   # door thickness (thin panel)

            if door_loaded and base_door is not None:
                # ── Use the OBJ asset ────────────────────────────
                di = base_door.copy()
                di.apply_transform(
                    trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                di.apply_scale(dw / orig_width)
                di.apply_transform(
                    trimesh.transformations.rotation_matrix(d['angle'], [0, 0, 1]))
                di.apply_translation([d['pos'][0] * SCALE,
                                       d['pos'][1] * SCALE, 0])
                components.append(di)
            else:
                # ── Procedural door box fallback ─────────────────
                # Orient by detected angle: vertical door → swap w/h dims
                is_vert = d['h'] > d['w']
                ext_x   = dt  if is_vert else dw
                ext_y   = dw  if is_vert else dt
                door_box = trimesh.creation.box(extents=(ext_x, ext_y, dh))
                door_box.visual.face_colors = DOOR_COLOR
                door_box.apply_translation([d['pos'][0] * SCALE,
                                             d['pos'][1] * SCALE,
                                             dh / 2])
                components.append(door_box)

    # ════════════════════════════════════════
    #  3. FLOOR
    # ════════════════════════════════════════
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

    # ════════════════════════════════════════
    #  4. ROOF  (skip gracefully when no params given)
    # ════════════════════════════════════════
    if not roof_params:
        _finish(components, output_file)
        return

    overhang       = roof_params.get("overhang",       0.40)
    slab_thickness = roof_params.get("slab_thickness", 0.20)
    parapet_height = roof_params.get("parapet_height", 0.55)
    style          = roof_params.get("roof_style",     "split-level")
    pitch_angle    = roof_params.get("pitch_angle",    8)
    has_parapet    = roof_params.get("has_parapet",    True)
    has_canopy     = roof_params.get("has_canopy",     True)
    canopy_depth   = roof_params.get("canopy_depth",   1.80)
    has_railing    = roof_params.get("has_railing",    True)

    # ── Left / Right wing split by X midpoint ────────────────────
    all_cx      = [w['pos'][0] * SCALE for w in walls]
    mid_x       = sum(all_cx) / len(all_cx)
    left_walls  = [w for w in walls if w['pos'][0] * SCALE <= mid_x]
    right_walls = [w for w in walls if w['pos'][0] * SCALE >  mid_x]

    volumes = [
        {"name": "Left",  "walls": left_walls,  "h_offset": 0.0},
        {"name": "Right", "walls": right_walls, "h_offset": 0.0},
    ]
    vol_bounds = {}
    slab_polys = []

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
        wing_box  = shp_box(vmin_x, vmin_y, vmax_x, vmax_y)
        roof_poly = wing_box.buffer(overhang, join_style=2)
        slab_polys.append(roof_poly)
        base_z    = WALL_H + vol["h_offset"]

        try:
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

    # ── Unified parapet + stairs + railing ────────────────────────
    if has_parapet and style in ("flat", "split-level") and slab_polys:
        try:
            combined_poly = unary_union(slab_polys)
            combined_poly = combined_poly.simplify(0.02, preserve_topology=True)
            inner_poly    = combined_poly.buffer(-0.22, join_style=2)
            par_ring      = combined_poly.difference(inner_poly)
            par_top_z     = WALL_H + slab_thickness

            parapet = trimesh.creation.extrude_polygon(par_ring, height=parapet_height)
            parapet.visual.face_colors = PARAPET_COLOR
            parapet.apply_translation([0, 0, par_top_z])
            components.append(parapet)

            # ── Staircase ─────────────────────────────────────────
            # FIX: subtract overhang so the stair never clips the roof slab edge
            STEP_W        = 1.30
            stair_top_z   = par_top_z + parapet_height
            stair_cx      = gmin_x - overhang - STEP_W / 2 - 0.08  # fully outside
            stair_start_y = gmin_y + 0.30

            stair_parts, stair_gap, stair_end_y = _make_stairs(
                stair_cx      = stair_cx,
                stair_start_y = stair_start_y,
                top_z         = stair_top_z,
                step_w        = STEP_W,
                step_d        = 0.28)
            for p in stair_parts:
                components.append(p)

            # ── Railing with gap at stair landing ─────────────────
            if has_railing:
                railing_gap = (gmin_x - overhang + 0.22,
                               stair_end_y,
                               STEP_W * 1.20)
                rail_poly = inner_poly.buffer(-0.04, join_style=2)
                for part in _make_railing(
                        rail_poly, par_top_z + parapet_height,
                        post_h=0.90, spacing=0.72,
                        gap_zones=[railing_gap]):
                    components.append(part)

        except Exception as e:
            print(f"⚠️ Parapet/Railing Error: {e}")

    # ── Elevated pavilion (right wing) ───────────────────────────
    if "Right" in vol_bounds and style in ("flat", "split-level"):
        for part in _make_elevated_pavilion(
                bounds         = vol_bounds["Right"],
                pillar_base_z  = WALL_H + slab_thickness,
                slab_thickness = slab_thickness,
                pillar_gap     = 1.50,
                pillar_radius  = 0.22,
                inset          = 0.55):
            components.append(part)

    # ── Seam riser ───────────────────────────────────────────────
    if "Left" in vol_bounds and "Right" in vol_bounds:
        _, lmax_x, lmin_y, lmax_y = vol_bounds["Left"]
        rmin_x, _, rmin_y, rmax_y  = vol_bounds["Right"]
        components.append(_make_step_riser(
            (lmax_x + rmin_x) / 2,
            min(lmin_y, rmin_y), max(lmax_y, rmax_y),
            WALL_H, WALL_H + slab_thickness))

    # ════════════════════════════════════════
    #  ENTRANCE CANOPY
    #  FIX: canopy_front_y = gmin_y  (actual front-wall face in model space)
    #       canopy_cx      = door's X centre  (for correct left/right alignment)
    #  The old code used door's Y as canopy_front_y which buried the slab
    #  inside the building and made it invisible.
    # ════════════════════════════════════════
    # ════════════════════════════════════════
    #  ENTRANCE CANOPY
    # ════════════════════════════════════════
    if has_canopy:
        # Always use the real front-wall face for Y (highest Y value)
        canopy_front_y = gmax_y

        if doors:
            # Grab the door with the HIGHEST Y pixel value (the main front door)
            front_door   = max(doors, key=lambda d: d['pos'][1])
            canopy_cx    = front_door['pos'][0] * SCALE
            door_w       = max(front_door['w'], front_door['h']) * SCALE
            canopy_width = max(door_w * 2.60, 2.00)
        else:
            canopy_cx    = (gmin_x + gmax_x) / 2
            canopy_width = max((gmax_x - gmin_x) * 0.35, 2.00)

        for part in _make_canopy(
                canopy_cx    = canopy_cx,
                front_wall_y = canopy_front_y,   # ← always gmax_y
                base_z       = WALL_H - 0.25,
                width        = canopy_width,
                depth        = canopy_depth,
                thickness    = 0.13,
                col_size     = 0.20):
            components.append(part)

    _finish(components, output_file)


# ══════════════════════════════════════════════════════════════════
def _finish(components, output_file):
    if not components:
        print("❌ No geometry generated.")
        return

    # Bake face colors → vertex colors on each mesh before concatenating.
    # This embeds all color data directly in the mesh so no .mtl sidecar
    # is needed. Works with any export format that supports vertex colors
    # (e.g. .glb, .ply); .obj is silently downgraded to no-color.
    for mesh in components:
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        try:
            fc = mesh.visual.face_colors          # (F, 4) RGBA uint8
            if fc is not None and len(fc) == len(mesh.faces):
                # Expand face colors to per-vertex (each vertex gets the
                # color of its first incident face — good enough for flat
                # architectural surfaces).
                vc = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
                for fi, face in enumerate(mesh.faces):
                    for vi in face:
                        vc[vi] = fc[fi]
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh, vertex_colors=vc)
        except Exception:
            pass  # leave visual untouched if anything goes wrong

    full_model = trimesh.util.concatenate(components)

    # -np.pi/2 confirmed correct by user (avoids inverted output)
    full_model.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    # Auto-switch to .glb if caller passed a .obj path, since .obj cannot
    # carry vertex colors without a .mtl file. .glb is a single binary file.
    out = output_file
    if out.lower().endswith(".obj"):
        out = out[:-4] + ".glb"
        print(f"ℹ️  Output switched to '{out}' (GLB embeds colors; OBJ cannot)")

    full_model.export(out)
    print(f"✅ Model saved → {out}")