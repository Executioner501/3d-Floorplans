"""
builder.py — assembles the final building and exports it
========================================================
Takes the detected walls/doors/windows plus a roof-parameter dict and
produces a GLB: walls with boolean-cut openings, framed joinery, floor
slab, site context, facade trim, and the roof (delegated to
roof_generator for every modern style; the legacy slab path below is a
fallback only).

Notes worth keeping in mind when editing:
  · Canopy uses gmin_y (the actual front-wall face), not the door centre.
  · Doors fall back to a procedural box when `door.obj` is absent.
  · Stairs subtract the overhang from X so they never clip the roof edge.
  · `_finish` rotates by -pi/2 about X (Z-up build → Y-up glTF).
  · Export is always GLB: vertex-colour meshes render near-black in most
    viewers, so each part gets an explicit non-metallic PBR material.
"""
import numpy as np
import trimesh
from shapely.geometry import box as shp_box
from shapely.ops import unary_union
import os

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
WALL_COLOR    = [245, 242, 230, 255]   # lighter warm cream (clean walls)
FLOOR_COLOR   = [140, 140, 140, 255]   # medium gray (visible, not distracting)
ROOF_COLOR    = [90,  95,  100, 255]   # lighter slate (not black)
PARAPET_COLOR = [80,  85,  90,  255]   # slightly darker than roof
COLUMN_COLOR  = [220, 215, 210, 255]   # brighter concrete
RAIL_COLOR    = [120, 120, 120, 255]   # lighter steel
STAIR_COLOR   = [230, 225, 220, 255]   # very light stone (stands out)
DOOR_COLOR    = [150, 110, 70,  255]   # slightly brighter wood


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
                  output_file=None, scale=0.01, windows=None):
    if output_file is None:
        output_file = "apartment.glb"
        print(f"ℹ️  No output file specified. Defaulting to '{output_file}'.")
    else:
        print(f"ℹ️  Exporting model to '{output_file}'...")

    # px→metre scale is now PER PLAN (see scale_utils.estimate_scale);
    # 0.01 kept as default for the synthetic demo plans.
    SCALE           = float(scale)
    WALL_H          = 3.0
    FLOOR_THICKNESS = 0.05

    components = []
    all_points = []

    # ════════════════════════════════════════
    #  1. WALLS  (with real door/window openings cut out)
    #     Openings are BOOLEAN-SUBTRACTED from the wall boxes, then
    #     re-filled with framed joinery. Two reasons this matters:
    #     a solid wall makes a door read as a plaque glued on, and at
    #     real wall thickness (~0.13 m) the old fixed 0.14 m frame was
    #     effectively coplanar with the wall face → z-fighting.
    #     Joinery depths are derived from each wall's own thickness.
    # ════════════════════════════════════════
    DOOR_H      = min(2.15, WALL_H * 0.72)
    SILL, WIN_H = 0.95, 1.30

    def _ext3(is_vert, along, across, height):
        """Extents for a part aligned with its wall's axis."""
        return (across, along, height) if is_vert else (along, across, height)

    def _shift(is_vert, along, across=0.0):
        """(dx, dy) offset along / across the wall axis."""
        return (across, along) if is_vert else (along, across)

    openings = []
    for _d in (doors or []):
        openings.append({"o": _d, "kind": "door", "z0": 0.0, "z1": DOOR_H})
    for _w in (windows or []):
        openings.append({"o": _w, "kind": "window",
                         "z0": SILL, "z1": SILL + WIN_H})

    def _belongs(entry, wi, w_data):
        """Openings carry the wall index detect.py snapped them to;
        fall back to geometric containment for hand-built inputs."""
        o = entry["o"]
        if o.get("wall") is not None:
            return o["wall"] == wi
        cx_, cy_ = o["pos"]
        return (abs(cx_ - w_data["pos"][0]) <= w_data["w"] / 2 + 2.0 and
                abs(cy_ - w_data["pos"][1]) <= w_data["h"] / 2 + 2.0)

    for wi, w_data in enumerate(walls):
        cx    = w_data['pos'][0] * SCALE
        cy    = w_data['pos'][1] * SCALE
        width = w_data['w']      * SCALE
        thick = w_data['h']      * SCALE
        wall_t = min(width, thick)          # true thickness of this wall

        wall = trimesh.creation.box(extents=(width, thick, WALL_H))
        wall.apply_translation([cx, cy, WALL_H / 2])

        cutters = []
        for entry in openings:
            if not _belongs(entry, wi, w_data):
                continue
            o = entry["o"]
            is_vert = bool(o.get("angle", 0))
            ln = max(o['w'], o['h']) * SCALE
            oh = entry["z1"] - entry["z0"]
            cut = trimesh.creation.box(
                extents=_ext3(is_vert, ln, wall_t + 0.60, oh))
            cut.apply_translation([o['pos'][0] * SCALE, o['pos'][1] * SCALE,
                                   (entry["z0"] + entry["z1"]) / 2])
            cutters.append(cut)

        if cutters:
            try:
                wall = trimesh.boolean.difference([wall] + cutters)
            except Exception as e:
                print(f"⚠️ Opening cut failed on wall {wi} ({e}); solid wall.")

        wall.visual.face_colors = WALL_COLOR
        components.append(wall)
        all_points.append(w_data['pos'])

    # ════════════════════════════════════════
    #  2. JOINERY — door leaves and glazed panes seated in the reveals
    #     cut above. Frames stand 15 mm proud of each wall face (never
    #     coplanar), leaves/panes sit recessed inside the wall.
    # ════════════════════════════════════════
    # DOOR.obj is authored Y-up; force='mesh' collapses it to a single
    # Trimesh (plain load returns a Scene, which _finish drops — that
    # silently deleted every door leaf and left bare frames).
    # NB: the asset ships as lowercase `door.obj`. Windows/macOS resolve the
    # name case-insensitively, so an uppercase literal here worked locally
    # while silently falling back to procedural leaves on Linux (and in CI).
    base_door = None
    _door_asset = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "door.obj")
    if doors and os.path.exists(_door_asset):
        try:
            cand = trimesh.load(_door_asset, force="mesh")
            cand.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            if isinstance(cand, trimesh.Trimesh) and len(cand.faces) and \
                    np.all(cand.extents > 1e-6):
                base_door = cand
            else:
                print("⚠️ door.obj unusable; using procedural door leaves.")
        except Exception as e:
            print(f"⚠️ Could not load door.obj ({e}); procedural leaves.")

    def _wall_t_of(o):
        """Thickness in metres of the wall this opening sits in."""
        if o.get("wall_t"):
            return float(o["wall_t"]) * SCALE
        if o.get("wall") is not None and o["wall"] < len(walls):
            w = walls[o["wall"]]
            return min(w['w'], w['h']) * SCALE
        return min(o['w'], o['h']) * SCALE

    for entry in openings:
        o       = entry["o"]
        kind    = entry["kind"]
        is_vert = bool(o.get("angle", 0))
        ox, oy  = o['pos'][0] * SCALE, o['pos'][1] * SCALE
        ln      = max(o['w'], o['h']) * SCALE
        wt      = max(_wall_t_of(o), 0.08)
        frame_c = wt + 0.03                 # 15 mm proud either side
        jw      = 0.07                      # jamb / head section

        def _put(mesh, along, z, across=0.0):
            dx, dy = _shift(is_vert, along, across)
            mesh.apply_translation([ox + dx, oy + dy, z])
            components.append(mesh)

        if kind == "door":
            dh = entry["z1"]
            leaf_t = max(0.05, wt * 0.45)
            if base_door is not None:
                # fit the asset to THIS opening: width × wall depth ×
                # door height, base seated on the floor
                di = base_door.copy()
                tgt = np.array([ln - 0.05, leaf_t, dh - 0.04])
                di.apply_scale(tgt / di.extents)
                di.apply_translation(-di.bounds[0])       # min corner → origin
                di.apply_translation([-tgt[0] / 2, -tgt[1] / 2, 0])
                if is_vert:
                    di.apply_transform(
                        trimesh.transformations.rotation_matrix(
                            np.pi / 2, [0, 0, 1]))
                di.apply_translation([ox, oy, 0])
                # replace (not mutate) the visual: the OBJ arrives with
                # TextureVisuals, whose face_colors setter is a no-op
                di.visual = trimesh.visual.ColorVisuals(
                    mesh=di, face_colors=np.tile(DOOR_COLOR, (len(di.faces), 1)))
                components.append(di)
            else:
                leaf = trimesh.creation.box(
                    extents=_ext3(is_vert, ln - 0.05, leaf_t, dh - 0.04))
                leaf.visual.face_colors = DOOR_COLOR
                _put(leaf, 0.0, (dh - 0.04) / 2)
                knob = trimesh.creation.box(
                    extents=_ext3(is_vert, 0.10, leaf_t + 0.06, 0.05))
                knob.visual.face_colors = [196, 190, 176, 255]
                _put(knob, ln / 2 - 0.16, 1.05)
            for sgn in (-1, 1):             # jambs
                jamb = trimesh.creation.box(
                    extents=_ext3(is_vert, jw, frame_c, dh + jw))
                jamb.visual.face_colors = [52, 54, 58, 255]
                _put(jamb, sgn * (ln / 2 + jw / 2), (dh + jw) / 2)
            head = trimesh.creation.box(
                extents=_ext3(is_vert, ln + 2 * jw, frame_c, jw))
            head.visual.face_colors = [52, 54, 58, 255]
            _put(head, 0.0, dh + jw / 2)

        else:                               # window
            zc = (entry["z0"] + entry["z1"]) / 2
            glass = trimesh.creation.box(
                extents=_ext3(is_vert, ln - 0.03, 0.03, WIN_H - 0.03))
            glass.visual.face_colors = [155, 195, 220, 150]
            _put(glass, 0.0, zc)
            for sgn in (-1, 1):             # jambs
                jamb = trimesh.creation.box(
                    extents=_ext3(is_vert, jw, frame_c, WIN_H + 2 * jw))
                jamb.visual.face_colors = [50, 52, 56, 255]
                _put(jamb, sgn * (ln / 2 + jw / 2), zc)
            head = trimesh.creation.box(
                extents=_ext3(is_vert, ln + 2 * jw, frame_c, jw))
            head.visual.face_colors = [50, 52, 56, 255]
            _put(head, 0.0, entry["z1"] + jw / 2)
            sill = trimesh.creation.box(                    # sill projects
                extents=_ext3(is_vert, ln + 2 * jw, frame_c + 0.06, jw))
            sill.visual.face_colors = [214, 210, 202, 255]
            _put(sill, 0.0, entry["z0"] - jw / 2)

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

    # ── site context: paved pad + lawn (grounds the model visually) ──
    cx0, cy0 = (gmin_x + gmax_x) / 2, (gmin_y + gmax_y) / 2
    pad = trimesh.creation.box(
        extents=((gmax_x - gmin_x) + 6.0, (gmax_y - gmin_y) + 6.0, 0.05))
    pad.visual.face_colors = [200, 198, 192, 255]
    pad.apply_translation([cx0, cy0, -FLOOR_THICKNESS - 0.030])
    components.append(pad)
    lawn = trimesh.creation.box(
        extents=((gmax_x - gmin_x) + 16.0, (gmax_y - gmin_y) + 16.0, 0.04))
    lawn.visual.face_colors = [128, 152, 100, 255]
    lawn.apply_translation([cx0, cy0, -FLOOR_THICKNESS - 0.075])
    components.append(lawn)

    # ── facade detailing from the closed footprint outline ──────────
    #    plinth at the base + slim charcoal trim at the roofline
    try:
        from roof_generator import extract_footprint
        fpoly = extract_footprint(walls, SCALE)
        if fpoly is not None and fpoly.area > 2.0:
            plinth_ring = fpoly.buffer(0.10, join_style=2).difference(
                fpoly.buffer(-0.06, join_style=2))
            plinth = trimesh.creation.extrude_polygon(plinth_ring, height=0.30)
            plinth.visual.face_colors = [168, 165, 158, 255]
            plinth.apply_translation([0, 0, -0.02])
            components.append(plinth)

            trim_ring = fpoly.buffer(0.07, join_style=2).difference(
                fpoly.buffer(-0.09, join_style=2))
            trim = trimesh.creation.extrude_polygon(trim_ring, height=0.28)
            trim.visual.face_colors = [72, 76, 82, 255]
            trim.apply_translation([0, 0, WALL_H - 0.28])
            components.append(trim)
    except Exception:
        pass

    # ════════════════════════════════════════
    #  4. ROOF  (skip gracefully when no params given)
    # ════════════════════════════════════════
    if not roof_params:
        _finish(components, output_file)
        return

    # ── NEW GENERATIVE ROOF ENGINE ───────────────────────────────
    # Styles like gable / cross-gable / hip / pyramid / dutch-gable /
    # gambrel / mansard / saltbox / butterfly / skillion / flat-modern
    # are built by roof_generator.py (footprint-aware, style-diverse).
    # Legacy slab styles (flat, split-level, mono-pitch, shed) fall
    # through to the original code below, unchanged.
    _style = str(roof_params.get("roof_style", "")).lower()
    try:
        from roof_generator import generate_roof, NEW_ENGINE_STYLES
        if _style in NEW_ENGINE_STYLES:
            roof_meshes, roof_info = generate_roof(
                walls, roof_params, scale=SCALE, wall_h=WALL_H)
            if roof_meshes:
                components.extend(roof_meshes)
                print(f"🏠 Generative roof: {roof_info.get('style')} "
                      f"({roof_info.get('material')}, "
                      f"pitch {roof_info.get('pitch'):.0f}°, "
                      f"{roof_info.get('wings')} wing(s))")
                # optional entrance canopy still applies to any style
                if roof_params.get("has_canopy"):
                    pts_r  = np.array(all_points) * SCALE
                    _gmaxy = float(pts_r[:, 1].max())
                    if doors:
                        _fd  = max(doors, key=lambda d: d['pos'][1])
                        _ccx = _fd['pos'][0] * SCALE
                        _cw  = max(max(_fd['w'], _fd['h']) * SCALE * 2.6, 2.0)
                    else:
                        _ccx = float(pts_r[:, 0].mean()); _cw = 2.2
                    for part in _make_canopy(
                            canopy_cx=_ccx, front_wall_y=_gmaxy,
                            base_z=WALL_H - 0.25, width=_cw,
                            depth=roof_params.get("canopy_depth", 1.6)):
                        components.append(part)
                _finish(components, output_file)
                return
    except Exception as e:
        print(f"⚠️ Generative roof engine failed ({e}); "
              f"falling back to legacy slab roof.")

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

    # Export as a Scene with one PBR material per part instead of baked
    # vertex colors. Vertex-color GLBs render near-black in most viewers
    # (they get the default fully-metallic material); an explicit
    # non-metallic rough baseColor lights correctly everywhere —
    # three.js, Windows 3D Viewer, Blender.
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene = trimesh.Scene()
    for mesh in components:
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        try:
            rgba = np.array(mesh.visual.face_colors[0], dtype=float) / 255.0
        except Exception:
            rgba = np.array([0.8, 0.8, 0.8, 1.0])
        # Rebuild instead of mesh.copy(). Trimesh.copy() deep-copies the
        # visuals, and ColorVisuals.copy() materialises vertex colours via
        # faces_sparse -> scipy.sparse.coo_matrix — pulling in a 114 MB
        # dependency to produce colours that are discarded two lines later
        # when the PBR material replaces the visual. process=False keeps the
        # geometry byte-identical (no vertex merging or reordering).
        m = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
            faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
            process=False)
        mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba.tolist(),
            metallicFactor=0.05,
            roughnessFactor=0.85)
        if rgba[3] < 0.99:
            mat.alphaMode = "BLEND"
        m.visual = trimesh.visual.texture.TextureVisuals(material=mat)
        # -np.pi/2 confirmed correct by user (avoids inverted output)
        m.apply_transform(rot)
        # Vertex normals are deliberately NOT computed or cached. trimesh's
        # glTF writer emits a NORMAL accessor only when one is already in the
        # cache, so leaving it empty keeps the file ~40 % smaller and lets the
        # viewer derive normals itself — which is the better look here, since
        # angle-weighted normals across a box's shared corner vertices round
        # off edges that should stay crisp.
        scene.add_geometry(m)

    out = output_file
    if out.lower().endswith(".obj"):
        out = out[:-4] + ".glb"
        print(f"ℹ️  Output switched to '{out}' (GLB embeds materials; OBJ cannot)")

    scene.export(out)
    print(f"✅ Model saved → {out}")