from builder import export_to_obj
from ultralytics import YOLO
from detect import process_yolo_results
from scale_utils import estimate_scale

# --- CONFIG ---
INPUT_IMG = "floorplan.png"
ENGINE    = "rag"     # "rag"   : retrieval-augmented designer (DEFAULT, no API key,
                      #           retrieves real architectural compositions)
                      # "llm"   : Gemini designer, validated by roof_ai
                      # "prior" : pure local footprint-conditioned sampler
                      # "diffusion": trained model (needs ml_roof_diffusion ckpt)
STYLE_PREF = "mixed"  # "modern" | "traditional" | "mixed"
SEED       = None     # set an int for reproducible output; None → fresh each run
RECIPE     = None     # force a specific KB recipe id (e.g. "double_gable_cross",
                      # "clerestory_monitor"); None → sample with variety.
                      # Recent picks persist in .rag_history.json and are
                      # avoided on the next runs — delete the file to reset.
DIFF_CKPT  = "ckpt/roofdiff_ep60.pt"   # stage-1 final: the best sampler we
                      # have. `roofdiff_last.pt` is the stage-2 finetune,
                      # which PoznanRD's near-flat half flattened (pairwise
                      # |dh| 0.05-0.66 m vs 0.29-2.02 m for ep60). See
                      # docs/ENGINEERING_NOTES.md before switching.
TARGET_LONG_M = 14.0  # assumed long side of the building when the plan
                      # prints no readable dimensions (see scale_utils)


def run_pipeline():
    # 1. Detect
    print("🔍 Extracting lines from image...")
    model = YOLO('best_doors.pt')
    results = model.predict(source=INPUT_IMG, conf=0.3)

    # 2. Process Data
    walls, doors, windows = process_yolo_results(results)
    print(f"📊 Found {len(walls)} walls, {len(doors)} doors, "
          f"{len(windows)} windows.")

    # 2b. Per-plan px→metre scale (Issue A fix): OCR printed dimensions
    #     when possible, else normalise the long side to TARGET_LONG_M.
    #     ONE scale flows through every downstream stage.
    scale = estimate_scale(walls, image_path=INPUT_IMG,
                           target_long_m=TARGET_LONG_M)

    # 3. Generative roof design
    if ENGINE == "rag":
        from roof_ai import design_roof_rag
        roof_params = design_roof_rag(walls, scale=scale, seed=SEED,
                                      style_pref=STYLE_PREF, recipe=RECIPE)
        print(f"🎨 RAG design: {roof_params['recipe_name']} "
              f"({roof_params['material']}, seed {roof_params['seed']})")
    elif ENGINE == "llm":
        from roof_ai import design_roof
        roof_params = design_roof(walls, scale=scale, seed=SEED, use_llm=True,
                                  image_path=INPUT_IMG)
    elif ENGINE == "prior":
        from roof_ai import design_roof
        roof_params = design_roof(walls, scale=scale, seed=SEED,
                                  region=STYLE_PREF)
    elif ENGINE == "diffusion":
        # learned model: sample a roof mesh directly and merge with the
        # roofless build (see ml_roof_diffusion/README.md to train)
        from ml_roof_diffusion.sample import generate_roof_diffusion
        export_to_obj(walls, doors, roof_params=None,
                      output_file="no_roof.glb", scale=scale,
                      windows=windows)
        import trimesh
        base = trimesh.load("no_roof.glb")
        roof = generate_roof_diffusion(walls, DIFF_CKPT, n=1, seed=SEED,
                                       scale=scale)[0]
        import numpy as np
        roof.apply_transform(trimesh.transformations.rotation_matrix(
            -np.pi / 2, [1, 0, 0]))
        trimesh.util.concatenate([base] + [roof]).export("apartment.glb")
        print("✅ Model saved → apartment.glb")
        return
    else:
        raise ValueError(f"unknown ENGINE {ENGINE}")

    # 4. 3D Build
    export_to_obj(walls, doors, roof_params=roof_params, scale=scale,
                  windows=windows)
    export_to_obj(walls, doors, roof_params=None, output_file="no_roof.glb",
                  scale=scale, windows=windows)


if __name__ == "__main__":
    run_pipeline()
