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


def test_no_torch_in_the_runtime_path():
    """The serverless bundle must stay torch-free."""
    for mod in ("torch", "ultralytics"):
        sys.modules.pop(mod, None)
    import detect_onnx                      # noqa: F401
    import roof_ai, roof_generator, builder  # noqa: F401,E401
    assert "torch" not in sys.modules, "torch was imported by the runtime path"
    assert "ultralytics" not in sys.modules, "ultralytics leaked into runtime"


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

    glb, stats = run_pipeline(Image.open(PLAN), seed=11)

    assert glb[:4] == GLTF_MAGIC, "response is not a binary glTF"
    assert len(glb) > 50_000, f"suspiciously small model: {len(glb)} bytes"
    # Vercel caps a serverless response at 4.5 MB; base64 adds ~33 %.
    assert len(glb) * 4 / 3 < 4_000_000, "GLB too large to return as base64"

    assert 5 < stats["width_m"] < 60, stats
    assert 5 < stats["depth_m"] < 60, stats
    assert stats["area_m2"] > 20, stats
    assert stats["recipe"], "no recipe recorded"
    assert stats["seed"] == 11


def test_pipeline_is_deterministic_for_a_seed():
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from api.generate import run_pipeline

    a, sa = run_pipeline(Image.open(PLAN), seed=5)
    b, sb = run_pipeline(Image.open(PLAN), seed=5)
    assert sa["recipe_id"] == sb["recipe_id"]
    assert sa["material"] == sb["material"]
    assert len(a) == len(b)


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
