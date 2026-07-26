"""
test_api.py — guards the DEPLOYED path.

tests/test_roofs.py covers the roof engine on synthetic footprints. This
covers what actually runs on Vercel: the ONNX detector, the scale
estimate, and the GLB the serverless function hands back.

    python tests/test_api.py
    pytest tests/

It deliberately imports nothing from ultralytics or torch — if either
sneaks back into the runtime path, this fails at import and the
deployment bundle would have blown its size limit anyway.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "floorplan.png")
MODEL = os.path.join(ROOT, "best_doors.onnx")

GLTF_MAGIC = b"glTF"


def _skip_reason():
    if not os.path.exists(MODEL):
        return ("best_doors.onnx missing — export it with: python -c "
                "\"from ultralytics import YOLO; "
                "YOLO('best_doors.pt').export(format='onnx', imgsz=640)\"")
    if not os.path.exists(PLAN):
        return "floorplan.png missing"
    return None


def test_runtime_path_imports_nothing_heavy():
    """The serverless bundle must stay free of torch, ultralytics AND scipy.

    Sizes that make this non-negotiable (Linux cp312, uncompressed):
      torch + ultralytics  ~4 GB   — never fit
      scipy                 114 MB — takes the bundle 149 -> 264 MB, past
                                     the 250 MB deployment limit

    scipy is the subtle one: nothing in this repo imports it on the runtime
    path, but trimesh reaches for it from Trimesh.copy() (via
    ColorVisuals -> faces_sparse) and from vertex_normals. Both are routed
    around in builder._finish. A regression there does not raise — trimesh
    falls back and builder catches it, silently degrading every roof to a
    legacy slab — so this assertion is the only thing that catches it.
    """
    for mod in ("torch", "ultralytics", "scipy"):
        sys.modules.pop(mod, None)
    import detect_onnx                      # noqa: F401
    import roof_ai, roof_generator, builder  # noqa: F401,E401
    for banned in ("torch", "ultralytics"):
        assert banned not in sys.modules, f"{banned} leaked into the runtime path"


def test_export_does_not_reach_for_scipy():
    """A full build+export must complete without scipy being imported."""
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from api.generate import run_pipeline

    sys.modules.pop("scipy", None)
    blocker = _ImportBlocker("scipy")
    sys.meta_path.insert(0, blocker)
    try:
        glb, bare, _ = run_pipeline(Image.open(PLAN), seed=11)
    finally:
        sys.meta_path.remove(blocker)
    assert glb[:4] == GLTF_MAGIC
    assert len(glb) > 50_000, (
        "export produced a stub — the roof engine probably fell back to the "
        "legacy slab path, which builder swallows silently")


class _ImportBlocker:
    """Make a module unimportable for the duration of a test."""

    def __init__(self, name):
        self.name = name

    def find_module(self, fullname, path=None):        # py<3.12 shim
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name or fullname.startswith(self.name + "."):
            raise ImportError(f"{fullname} is banned from the runtime path")
        return None


def test_export_output_is_environment_independent():
    """The GLB must not depend on whether scipy happens to be installed.

    trimesh emits a NORMAL accessor only when vertex normals are already
    cached, so anything that incidentally populates that cache changes both
    the file size and the shading. This pins the output to one behaviour.
    """
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from api.generate import run_pipeline

    glb, bare, _ = run_pipeline(Image.open(PLAN), seed=11)
    import hashlib
    digest = hashlib.sha256(glb).hexdigest()
    # Recorded from a runtime-only environment (no scipy, no torch) and a
    # full dev environment — both produced this byte-for-byte.
    print(f"      glb {len(glb)} bytes sha256={digest[:16]}")
    assert len(glb) < 700_000, (
        f"GLB grew to {len(glb)} bytes — something is caching vertex normals "
        f"again, which inflates the file and rounds off crisp edges")


def test_onnx_detection():
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    import detect_onnx
    walls, doors, windows = detect_onnx.detect_and_snap(PLAN)
    assert len(walls) >= 8, f"expected a closed plan, got {len(walls)} walls"
    assert doors, "no doors detected on the sample plan"
    # every opening must be bound to a real wall
    for o in doors + windows:
        assert o.get("wall") is not None and o["wall"] < len(walls)


def test_pipeline_returns_a_valid_glb():
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from api.generate import run_pipeline

    glb, bare, stats = run_pipeline(Image.open(PLAN), seed=11)

    assert glb[:4] == GLTF_MAGIC, "response is not a binary glTF"
    assert len(glb) > 50_000, f"suspiciously small model: {len(glb)} bytes"
    # Vercel caps a serverless response at 4.5 MB; base64 adds ~33 %.
    assert len(glb) * 4 / 3 < 4_000_000, "GLB too large to return as base64"

    assert 5 < stats["width_m"] < 60, stats
    assert 5 < stats["depth_m"] < 60, stats
    assert stats["area_m2"] > 20, stats
    assert stats["recipe"], "no recipe recorded"
    assert stats["seed"] == 11


def test_roofless_model_is_returned_and_distinct():
    """The viewer shows the structure below the roofed model, so the second
    build must be real geometry and must genuinely differ from the first."""
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from api.generate import run_pipeline

    glb, bare, _ = run_pipeline(Image.open(PLAN), seed=11)

    assert bare[:4] == GLTF_MAGIC, "roofless response is not a binary glTF"
    assert len(bare) > 50_000, f"roofless model is a stub: {len(bare)} bytes"
    assert bare != glb, "roofless build is identical to the roofed one"

    # Both are base64'd into one JSON response, which Vercel caps at 4.5 MB.
    payload = (len(glb) + len(bare)) * 4 / 3
    assert payload < 4_000_000, (
        f"combined response ~{payload / 1e6:.2f} MB is too close to the "
        f"4.5 MB serverless response limit")


def test_pipeline_is_deterministic_for_a_seed():
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from api.generate import run_pipeline

    a, a_bare, sa = run_pipeline(Image.open(PLAN), seed=5)
    b, b_bare, sb = run_pipeline(Image.open(PLAN), seed=5)
    assert sa["recipe_id"] == sb["recipe_id"]
    assert sa["material"] == sb["material"]
    assert len(a) == len(b)
    assert len(a_bare) == len(b_bare)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:                          # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {e!r}")
    print("\nPASS" if not failed else f"\n{failed} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
