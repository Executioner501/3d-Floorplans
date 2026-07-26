"""
test_roofs.py — regression harness for the RAG roof engine
==========================================================
RECONSTRUCTION_PLAN Phase 4.2. Run this before every knowledge-base or
generator change; it is the cheapest signal that a composition edit has
broken geometry somewhere in the library.

    python tests/test_roofs.py        # standalone, prints a report
    pytest tests/                     # same assertions under pytest

What it covers:
  · every exemplar in KNOWLEDGE_BASE, forced by id (no sampling), so a
    broken recipe cannot hide behind retrieval scores
  · four footprint archetypes (rectangle, square, elongated, L-shape)
  · two seeds each, because pitch/overhang/direction are sampled ranges

Assertions:
  · every build returns geometry
  · no build trips the coverage self-check (`info["repaired"]`) — a rising
    repair rate is the early-warning signal for KB bugs
  · retrieval stays diverse (Phase 3 acceptance: many distinct recipes
    across seeds, no single recipe dominating)
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import roof_ai                                    # noqa: E402
import roof_generator as G                        # noqa: E402
from roof_knowledge import KNOWLEDGE_BASE         # noqa: E402

WALL_T = 12          # wall box thickness in pixels
SCALE = 0.01         # px -> m for these synthetic plans (1 px = 1 cm)


def _rect(w, h, t=WALL_T):
    """Four wall boxes forming a closed rectangular plan."""
    return [
        dict(pos=(w / 2, t / 2), w=w, h=t, angle=0),
        dict(pos=(w / 2, h - t / 2), w=w, h=t, angle=0),
        dict(pos=(t / 2, h / 2), w=t, h=h, angle=1.5708),
        dict(pos=(w - t / 2, h / 2), w=t, h=h, angle=1.5708),
    ]


def _l_shape(w, h, t=WALL_T):
    """Rectangle plus an interior return, giving a concave multi-wing plan."""
    return _rect(w, h, t) + [
        dict(pos=(w * 0.60, h * 0.45), w=w * 0.85, h=t, angle=0),
        dict(pos=(w * 0.60, h * 0.22), w=t, h=h * 0.50, angle=1.5708),
    ]


PLANS = {
    "rectangle_12x8": _rect(1200, 800),
    "square_10x10": _rect(1000, 1000),
    "elongated_16x7": _rect(1600, 700),
    "l_shape_12x9": _l_shape(1200, 900),
}
SEEDS = (1, 2)


def build_matrix():
    """Build every exemplar on every plan. Returns (failures, repairs, runs)."""
    failures, repairs, runs = [], [], 0
    for plan_name, walls in PLANS.items():
        for ex in KNOWLEDGE_BASE:
            for seed in SEEDS:
                runs += 1
                label = f"{plan_name}/{ex['id']}/seed{seed}"
                try:
                    params = roof_ai.design_roof_rag(
                        walls, scale=SCALE, seed=seed,
                        recipe=ex["id"], history_file=None)
                    meshes, info = G.generate_roof(
                        walls, params, scale=SCALE, wall_h=3.0)
                except Exception as e:                    # noqa: BLE001
                    failures.append((label, repr(e)))
                    continue
                if not meshes:
                    failures.append((label, "no geometry returned"))
                elif info.get("repaired"):
                    repairs.append(label)
    return failures, repairs, runs


def retrieval_spread(n_seeds=30, plan="rectangle_12x8"):
    """Distinct recipe ids sampled across seeds, and the most common one."""
    walls = PLANS[plan]
    counts = {}
    for seed in range(n_seeds):
        p = roof_ai.design_roof_rag(walls, scale=SCALE, seed=seed,
                                    history_file=None)
        counts[p["recipe_id"]] = counts.get(p["recipe_id"], 0) + 1
    return counts


# ── pytest entry points ─────────────────────────────────────────────
def test_every_exemplar_builds():
    failures, _, _ = build_matrix()
    assert not failures, "builds failed:\n" + "\n".join(
        f"  {lbl}: {err}" for lbl, err in failures)


def test_no_coverage_repairs():
    _, repairs, runs = build_matrix()
    rate = len(repairs) / max(runs, 1)
    assert rate < 0.05, (
        f"coverage self-check repair rate {rate:.1%} exceeds 5%:\n  "
        + "\n  ".join(repairs[:20]))


def test_retrieval_is_diverse():
    counts = retrieval_spread()
    total = sum(counts.values())
    assert len(counts) >= 8, f"only {len(counts)} distinct recipes over {total} seeds"
    top = max(counts.values()) / total
    assert top <= 0.25, f"one recipe took {top:.0%} of picks (max 25%)"


# ── standalone runner ───────────────────────────────────────────────
def main():
    print(f"knowledge base: {len(KNOWLEDGE_BASE)} exemplars")
    print(f"matrix: {len(KNOWLEDGE_BASE)} exemplars x {len(PLANS)} plans "
          f"x {len(SEEDS)} seeds\n")

    failures, repairs, runs = build_matrix()
    print(f"builds   : {runs}")
    print(f"failures : {len(failures)}")
    print(f"repairs  : {len(repairs)} ({len(repairs) / max(runs, 1):.1%})")
    for label, err in failures[:20]:
        print(f"  FAIL {label}: {err}")
    for label in repairs[:20]:
        print(f"  REPAIRED {label}")

    counts = retrieval_spread()
    total = sum(counts.values())
    top_id, top_n = max(counts.items(), key=lambda kv: kv[1])
    print(f"\nretrieval: {len(counts)} distinct recipes over {total} seeds; "
          f"most common '{top_id}' at {top_n / total:.0%}")

    ok = (not failures
          and len(repairs) / max(runs, 1) < 0.05
          and len(counts) >= 8
          and top_n / total <= 0.25)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
