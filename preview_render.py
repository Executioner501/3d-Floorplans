"""
preview_render.py — well-lit PNG preview of an exported GLB
===========================================================
    python preview_render.py apartment.glb [out.png] [azim]

Renders the PBR-material scene with a simple sun-shaded painter
renderer (no GPU/pyrender needed) on a light background — what the
model actually looks like in a properly lit viewer.
"""
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BG = "#eef1f5"


def _parts(path):
    sc = trimesh.load(path)
    if isinstance(sc, trimesh.Scene):
        return sc.dump()
    return [sc]


def _rgba(g):
    try:
        c = np.array(g.visual.material.baseColorFactor, dtype=float)
        if c.max() > 1.001:
            c = c / 255.0
        return c
    except Exception:
        pass
    try:
        return np.array(g.visual.face_colors[0], dtype=float) / 255.0
    except Exception:
        return np.array([0.72, 0.72, 0.72, 1.0])


def render(path, out_png, elev=20, azim=-55):
    parts = _parts(path)
    # GLB is y-up; rotate back to z-up for plotting
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    light = np.array([0.45, 0.3, 0.85])
    light /= np.linalg.norm(light)
    el, az = np.radians(elev), np.radians(azim)
    cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                    np.sin(el)])

    fig = plt.figure(figsize=(12, 7.5), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d", facecolor=BG)
    allv, spans = [], []
    all_tris, all_cols = [], []
    for g in parts:
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        g = g.copy()
        g.apply_transform(R)
        rgba = _rgba(g)
        allv.append(g.vertices)
        spans.append((g.vertices.max(0) - g.vertices.min(0)).max())
        # subdivide long edges: the painter sort works per-triangle mean
        # depth, so huge lawn/roof triangles otherwise bleed through walls
        try:
            vv, ff = trimesh.remesh.subdivide_to_size(
                g.vertices, g.faces, max_edge=0.8, max_iter=7)
            g = trimesh.Trimesh(vertices=vv, faces=ff, process=False)
        except Exception:
            pass
        # back-face culling: kills z-fighting between the coincident
        # front/back faces of thin boxes (walls, pads, panes)
        keep = g.face_normals @ cam > 0.03
        # the timber soffit is an underside layer — its top face is never
        # legitimately visible, but painter ties let it bleed through the
        # roof plane above it; cull it by its palette color
        if np.allclose(rgba[:3], np.array([176, 140, 102]) / 255.0,
                       atol=0.02):
            keep &= g.face_normals[:, 2] < 0.5
        if not keep.any():
            continue
        tris = g.vertices[g.faces[keep]]
        shade = 0.58 + 0.42 * np.clip(g.face_normals[keep] @ light, 0, 1)
        cols = np.clip(rgba[None, :3] * shade[:, None], 0, 1)
        cols = np.hstack([cols, np.full((len(cols), 1), rgba[3])])
        all_tris.append(tris)
        all_cols.append(cols)
    # ONE collection → matplotlib depth-sorts all triangles together
    # (separate collections are layered by draw order, hiding the walls)
    ax.add_collection3d(Poly3DCollection(np.concatenate(all_tris),
                                         facecolors=np.concatenate(all_cols),
                                         edgecolor="none"))
    # frame the BUILDING: ignore site pads (the widest flat parts)
    max_span = max(spans)
    bld = [v for v, s in zip(allv, spans) if s < 0.72 * max_span]
    v = np.vstack(bld if bld else allv)
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2
    r = float((hi - lo).max()) * 0.55
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(lo[2], lo[2] + 0.9 * r)
    try:
        ax.set_box_aspect((1, 1, 0.45))
    except Exception:
        pass
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"🖼  {out_png}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "apartment.glb"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + "_preview.png"
    az = float(sys.argv[3]) if len(sys.argv) > 3 else -55
    render(src, dst, azim=az)
