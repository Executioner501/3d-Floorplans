"""
demo_roofs.py — Test / showcase for the generative roof module.

Runs WITHOUT YOLO or the Gemini API: synthetic wall layouts stand in
for the detection stage so the roof engine can be validated in
isolation.

  python demo_roofs.py            → styles gallery + variations gallery
                                    (PNG previews + a few .glb models)
"""
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from roof_generator import generate_roof, NEW_ENGINE_STYLES
from roof_ai import design_roof, design_variations, sample_parameters
from roof_generator import extract_footprint, footprint_metrics
from builder import export_to_obj


# ────────────────────────────────────────────────────────────────────
#  Synthetic wall layouts (pixel space, SCALE = 0.01 → metres)
# ────────────────────────────────────────────────────────────────────
def _wall(cx, cy, w, h):
    return {"pos": (cx, cy), "w": w, "h": h,
            "angle": 1.5708 if h > w else 0}


def rect_plan(W=1400, D=800, t=25):
    return [
        _wall(W / 2, t / 2, W, t), _wall(W / 2, D - t / 2, W, t),
        _wall(t / 2, D / 2, t, D), _wall(W - t / 2, D / 2, t, D),
        _wall(W / 2, D / 2, W, t),          # interior partition
    ]


def square_plan(S=900, t=25):
    return rect_plan(S, S, t)


def l_plan(W=1500, D=1000, cut_w=650, cut_d=450, t=25):
    """L-shape: big rect with the top-right corner removed."""
    walls = [
        _wall(W / 2, t / 2, W, t),                                    # bottom
        _wall(t / 2, D / 2, t, D),                                    # left
        _wall((W - cut_w) / 2, D - t / 2, W - cut_w, t),              # top-left
        _wall(W - t / 2, (D - cut_d) / 2, t, D - cut_d),              # right lower
        _wall(W - cut_w + t / 2, D - cut_d / 2, t, cut_d),            # notch vert
        _wall(W - cut_w / 2, D - cut_d - t / 2, cut_w, t),            # notch horiz
        _wall((W - cut_w) / 2, (D - cut_d) / 2, W - cut_w, t),        # partition
    ]
    return walls


def t_plan(W=1500, D=700, stem_w=520, stem_d=520, t=25):
    """T-shape: wide bar + stem projecting down from the middle."""
    sx0 = (W - stem_w) / 2
    walls = [
        _wall(W / 2, D + stem_d - t / 2, W, t),                       # top bar
        _wall(t / 2, D / 2 + stem_d, t, D),                           # bar left
        _wall(W - t / 2, D / 2 + stem_d, t, D),                       # bar right
        _wall(sx0 / 2, stem_d + t / 2, sx0, t),                       # underside L
        _wall(W - sx0 / 2, stem_d + t / 2, sx0, t),                   # underside R
        _wall(sx0 + t / 2, stem_d / 2, t, stem_d),                    # stem left
        _wall(W - sx0 - t / 2, stem_d / 2, t, stem_d),                # stem right
        _wall(W / 2, t / 2, stem_w, t),                               # stem bottom
    ]
    return walls


PLANS = {"rectangle": rect_plan(), "square": square_plan(),
         "L-shape": l_plan(), "T-shape": t_plan()}


# ────────────────────────────────────────────────────────────────────
#  Lightweight matplotlib 3-D preview (no GPU needed)
# ────────────────────────────────────────────────────────────────────
def render_meshes(ax, meshes, title=""):
    light = np.array([0.4, 0.3, 0.85])
    light = light / np.linalg.norm(light)
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        tris = mesh.vertices[mesh.faces]
        base = np.array(mesh.visual.face_colors[0][:3]) / 255.0
        n = mesh.face_normals
        shade = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
        cols = np.clip(base[None, :] * shade[:, None], 0, 1)
        pc = Poly3DCollection(tris, facecolors=cols, edgecolor="none")
        ax.add_collection3d(pc)
    allv = np.vstack([m.vertices for m in meshes
                      if isinstance(m, trimesh.Trimesh) and len(m.vertices)])
    c = allv.mean(0); r = (allv.max(0) - allv.min(0)).max() / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(0, 2 * r)
    ax.set_axis_off(); ax.set_title(title, fontsize=9)
    ax.view_init(elev=24, azim=-58)


def house_meshes(walls, roof_params, wall_h=3.0, scale=0.01):
    """Walls + floor + generative roof (mirrors builder.py geometry)."""
    comps = []
    for w in walls:
        b = trimesh.creation.box(extents=(w["w"] * scale, w["h"] * scale, wall_h))
        b.visual.face_colors = [245, 242, 230, 255]
        b.apply_translation([w["pos"][0] * scale, w["pos"][1] * scale, wall_h / 2])
        comps.append(b)
    roof, info = generate_roof(walls, roof_params, scale=scale, wall_h=wall_h)
    return comps + roof, info


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # ── 1. STYLE GALLERY: every style on a suitable plan ───────────
    style_plan = {
        "gable": "rectangle",   "cross-gable": "L-shape",
        "hip": "L-shape",       "pyramid": "square",
        "dutch-gable": "rectangle", "gambrel": "rectangle",
        "mansard": "square",    "saltbox": "rectangle",
        "butterfly": "rectangle", "skillion": "rectangle",
        "flat-modern": "square",
    }
    fig = plt.figure(figsize=(16, 12))
    for i, (style, plan_name) in enumerate(style_plan.items(), 1):
        walls = PLANS[plan_name]
        poly = extract_footprint(walls)
        params = sample_parameters(style, footprint_metrics(poly),
                                   np.random.default_rng(i * 11))
        params["seed"] = i * 11
        meshes, info = house_meshes(walls, params)
        ax = fig.add_subplot(3, 4, i, projection="3d")
        render_meshes(ax, meshes,
                      f"{style}  ·  {plan_name}\n{info['material']}, "
                      f"{info['pitch']:.0f}°")
        print(f"✔ {style:<12s} on {plan_name:<9s} → {len(meshes)} meshes, "
              f"{info['wings']} wing(s)")
    fig.suptitle("Generative Roof Engine — Style Gallery", fontsize=14)
    fig.tight_layout()
    fig.savefig("roof_styles_gallery.png", dpi=110)
    print("🖼  roof_styles_gallery.png saved")

    # ── 2. VARIATION GALLERY: same L-plan, different seeds ─────────
    walls = PLANS["L-shape"]
    variations = design_variations(walls, n=6, base_seed=100)
    fig2 = plt.figure(figsize=(15, 8))
    for i, params in enumerate(variations, 1):
        meshes, info = house_meshes(walls, params)
        ax = fig2.add_subplot(2, 3, i, projection="3d")
        render_meshes(ax, meshes,
                      f"seed {params['seed']}: {info['style']}\n"
                      f"{info['material']}, {info['pitch']:.0f}°")
    fig2.suptitle("Same Floor Plan (L-shape) — Six Distinct Generated Roofs",
                  fontsize=14)
    fig2.tight_layout()
    fig2.savefig("roof_variations_gallery.png", dpi=110)
    print("🖼  roof_variations_gallery.png saved")

    # ── 3. Full-pipeline export through builder.py (GLB output) ───
    for style, plan_name, seed in [("cross-gable", "L-shape", 3),
                                   ("hip", "T-shape", 5),
                                   ("gable", "rectangle", 8)]:
        walls = PLANS[plan_name]
        params = design_roof(walls, seed=seed, llm_hint=style)
        export_to_obj(walls, doors=None, roof_params=params,
                      output_file=f"demo_{style}_{plan_name}.glb")
    print("✅ Demo complete.")
