"""
test_llm.py — the LLM composition designer must never make output worse.

The whole safety argument for roof_llm is that every failure resolves to
the RAG design that would have been produced anyway. These tests hold that
argument up, without ever calling the network: `ask_gemini_composition` is
stubbed, so this runs offline and in CI.

    python tests/test_llm.py
    pytest tests/
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import roof_llm                                              # noqa: E402

WALL_T = 12


def _rect(w, h, t=WALL_T):
    return [
        dict(pos=(w / 2, t / 2), w=w, h=t, angle=0),
        dict(pos=(w / 2, h - t / 2), w=w, h=t, angle=0),
        dict(pos=(t / 2, h / 2), w=t, h=h, angle=1.5708),
        dict(pos=(w - t / 2, h / 2), w=t, h=h, angle=1.5708),
    ]


def _l_shape(w, h, t=WALL_T):
    return _rect(w, h, t) + [
        dict(pos=(w * 0.60, h * 0.45), w=w * 0.85, h=t, angle=0),
        dict(pos=(w * 0.60, h * 0.22), w=t, h=h * 0.50, angle=1.5708),
    ]


PLANS = {
    "rectangle": _rect(1400, 900),
    "square": _rect(1000, 1000),
    "elongated": _rect(1600, 700),
    "l_shape": _l_shape(1200, 900),
}

GOOD = {
    "id": "test_comp", "name": "Opposing skillions", "tags": ["modern"],
    "zones": [
        {"region": {"type": "split", "axis": "long", "f0": 0.0, "f1": 0.55},
         "style": "skillion", "pitch": 13, "direction": "-short"},
        {"region": {"type": "split", "axis": "long", "f0": 0.55, "f1": 1.0},
         "style": "skillion", "pitch": 11, "direction": "+short",
         "h_offset": 1.0, "clerestory": True},
    ],
    "materials": ["metal-seam"], "features": {"skylight_p": 0.3},
    "rationale": "the long axis suits two opposing planes",
}


def _stub(result, err=None):
    roof_llm.ask_gemini_composition = lambda *a, **k: (result, err)


def _restore():
    import importlib
    importlib.reload(roof_llm)


# ── the fallback contract ───────────────────────────────────────────
def test_missing_api_key_falls_back_to_rag():
    _restore()
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(var, None)
    p = roof_llm.design_roof_llm(PLANS["rectangle"], seed=3)
    assert p["source"] == "rag", p
    assert p["recipe_id"], "fallback produced no design"


def test_fallback_matches_rag_exactly():
    """A fallback must be the SAME design RAG would have produced alone —
    not a degraded variant."""
    _restore()
    _stub(None, "simulated network failure")
    from roof_ai import design_roof_rag
    walls = PLANS["l_shape"]
    got = roof_llm.design_roof_llm(walls, seed=7)
    want = design_roof_rag(walls, seed=7, history_file=None)
    assert got["source"] == "rag"
    assert got["recipe_id"] == want["recipe_id"]
    assert got["roof_style"] == want["roof_style"]
    assert got["material"] == want["material"]


def test_every_failure_mode_falls_back():
    _restore()
    from roof_ai import design_roof_rag
    baseline = design_roof_rag(PLANS["rectangle"], seed=1, history_file=None)
    bad_cases = {
        "network error": (None, "HTTP 503"),
        "not an object": ("just a string", None),
        "no zones": ({"id": "x", "zones": []}, None),
        "unknown style": ({"zones": [{"region": {"type": "full"},
                                      "style": "geodesic", "pitch": 30}]}, None),
        "overlapping full zones": ({"zones": [
            {"region": {"type": "full"}, "style": "gable", "pitch": 30},
            {"region": {"type": "full"}, "style": "hip", "pitch": 25}]}, None),
        "flat lid": ({"zones": [{"region": {"type": "full"},
                                 "style": "flat", "pitch": 0}]}, None),
        "sliver split": ({"zones": [
            {"region": {"type": "split", "axis": "long", "f0": 0.0, "f1": 0.02},
             "style": "gable", "pitch": 30}]}, None),
    }
    for label, (result, err) in bad_cases.items():
        _stub(result, err)
        p = roof_llm.design_roof_llm(PLANS["rectangle"], seed=1)
        assert p["source"] == "rag", f"{label}: did not fall back"
        assert p["recipe_id"] == baseline["recipe_id"], \
            f"{label}: fallback differs from plain RAG"
        assert p["llm_note"], f"{label}: fell back without saying why"


# ── the accept path ─────────────────────────────────────────────────
def test_valid_composition_is_used_and_builds():
    _restore()
    _stub(GOOD)
    import roof_generator as G
    for name, walls in PLANS.items():
        p = roof_llm.design_roof_llm(walls, seed=2)
        assert p["source"] == "llm", f"{name}: rejected a valid composition"
        meshes, info = G.generate_roof(walls, p, wall_h=3.0)
        assert meshes, f"{name}: accepted design produced no geometry"
        assert not info.get("repaired"), \
            f"{name}: accepted design needed a coverage repair"


def test_unbuildable_composition_is_caught_by_verification():
    """Schema-valid but geometrically wrong must still fall back — this is
    what the build-and-verify step exists for."""
    _restore()
    # Two splits on different axes: each is individually legal, together
    # they do not tile the footprint.
    _stub({"id": "bad", "name": "gappy", "tags": ["modern"], "zones": [
        {"region": {"type": "split", "axis": "long", "f0": 0.0, "f1": 0.30},
         "style": "gable", "pitch": 30},
        {"region": {"type": "split", "axis": "short", "f0": 0.0, "f1": 0.25},
         "style": "flat", "pitch": 0}],
        "materials": ["asphalt-dark"], "features": {}})
    p = roof_llm.design_roof_llm(PLANS["rectangle"], seed=1)
    assert p["source"] == "rag", "an uncovering composition was accepted"
    assert "coverage" in (p["llm_note"] or "") or "uncovered" in (p["llm_note"] or "")


def test_clamping_keeps_absurd_values_buildable():
    _restore()
    ex, why = roof_llm.validate_exemplar({"zones": [
        {"region": {"type": "full"}, "style": "gable",
         "pitch": 900, "overhang": 50, "h_offset": -99}]}, 1)
    assert ex is not None, why
    z = ex["zones"][0]
    assert 22 <= z["pitch"] <= 45, z
    assert 0.15 <= z["overhang"] <= 1.0, z
    # negative offsets trip the coverage check, so they floor at 0
    assert z["h_offset"] >= 0.0, z


def test_params_shape_matches_the_rag_designer():
    """Downstream code must not need to care which designer ran."""
    _restore()
    _stub(GOOD)
    from roof_ai import design_roof_rag
    llm = roof_llm.design_roof_llm(PLANS["rectangle"], seed=5)
    rag = design_roof_rag(PLANS["rectangle"], seed=5, history_file=None)
    missing = [k for k in rag if k not in llm]
    assert not missing, f"LLM params missing keys the RAG path provides: {missing}"


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
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {e!r}")
    print("\nPASS" if not failed else f"\n{failed} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
