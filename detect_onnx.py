"""
detect_onnx.py — torch-free wall/door/window detection
======================================================
Runs the same custom-trained YOLOv8 detector as `main.py`, but through
ONNX Runtime instead of ultralytics + torch. Dependencies are PIL,
numpy and onnxruntime (~48 MB of wheels vs ~4 GB for the torch stack),
which is what makes the pipeline deployable as a serverless function.

Export the model once, locally:

    python -c "from ultralytics import YOLO; \
               YOLO('best_doors.pt').export(format='onnx', imgsz=640, simplify=True)"

Agreement with the ultralytics path on floorplan.png: 28 of 30 boxes
match at IoU > 0.7, median IoU 0.96. The residual difference is the
letterbox — this export has a fixed 640x640 input so the image is padded
to a square, while ultralytics uses rectangular inference padded to a
stride multiple. `detect.snap_openings_to_walls` absorbs the difference
(it drops strays and merges duplicates regardless of their origin).

Output format is identical to `detect.process_yolo_results`, so every
downstream stage is unchanged.
"""
import os

import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "best_doors.onnx")
INPUT_SIZE = 640
PAD_VALUE = 114                     # ultralytics' letterbox grey

# Model classes, from the CubiCasa5k training YAML.
CLS_DOOR, CLS_WALL, CLS_WINDOW = 0, 1, 2

_SESSION = None


def _session(model_path=None):
    """Lazily build (and cache) the inference session.

    On a serverless platform the module stays warm between invocations,
    so this pays the ~200 ms session setup once per cold start rather
    than once per request.
    """
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2          # serverless vCPU budget
        _SESSION = ort.InferenceSession(model_path or MODEL_PATH, opts,
                                        providers=["CPUExecutionProvider"])
    return _SESSION


def letterbox(img, size=INPUT_SIZE, pad=PAD_VALUE):
    """Resize preserving aspect ratio, then pad to a square.

    Returns (image, ratio, dx, dy) so boxes can be mapped back to the
    original pixel space.
    """
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (dx, dy))
    return canvas, r, dx, dy


def nms(boxes, scores, iou_thr):
    """Greedy non-max suppression on xyxy boxes.

    ONNX export emits raw predictions — unlike the .pt path, suppression
    is the caller's job.
    """
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def detect(image, conf=0.3, iou=0.7, model_path=None):
    """Detect walls, doors and windows.

    `image` is a PIL.Image or a path. Returns (walls, doors, windows) as
    lists of {"pos", "w", "h", "angle"} dicts in ORIGINAL image pixels —
    the same contract as detect.process_yolo_results, before snapping.
    """
    img = image if isinstance(image, Image.Image) else Image.open(image)
    W, H = img.size
    lb, r, dx, dy = letterbox(img)

    x = np.asarray(lb, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    sess = _session(model_path)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]   # (1, 4+nc, N)

    pred = out[0].T                                  # (N, 4+nc)
    scores_all = pred[:, 4:]
    conf_v = scores_all.max(1)
    keep = conf_v > conf
    if not keep.any():
        return [], [], []
    pred, conf_v = pred[keep], conf_v[keep]
    cls_id = scores_all[keep].argmax(1)

    # cxcywh in letterboxed pixels -> xyxy in original image pixels
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xyxy = np.stack([(cx - bw / 2 - dx) / r, (cy - bh / 2 - dy) / r,
                     (cx + bw / 2 - dx) / r, (cy + bh / 2 - dy) / r], 1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)

    walls, doors, windows = [], [], []
    bucket = {CLS_DOOR: doors, CLS_WALL: walls, CLS_WINDOW: windows}
    for c in np.unique(cls_id):                      # suppress per class
        m = cls_id == c
        for i in np.flatnonzero(m)[nms(xyxy[m], conf_v[m], iou)]:
            x1, y1, x2, y2 = xyxy[i]
            bw_, bh_ = x2 - x1, y2 - y1
            if bw_ < 1 or bh_ < 1:
                continue
            bucket.get(int(c), []).append({
                "pos": (float((x1 + x2) / 2), float((y1 + y2) / 2)),
                "w": float(bw_), "h": float(bh_),
                "angle": 1.5708 if bh_ > bw_ else 0,
            })
    return walls, doors, windows


def detect_and_snap(image, conf=0.3, iou=0.7, model_path=None):
    """detect() followed by the same opening cleanup main.py applies:
    attach each door/window to its nearest wall, re-orient it along that
    wall's axis, drop strays and merge duplicate boxes."""
    from detect import snap_openings_to_walls
    walls, doors, windows = detect(image, conf, iou, model_path)
    return (walls,
            snap_openings_to_walls(doors, walls),
            snap_openings_to_walls(windows, walls))


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "floorplan.png"
    w, d, wi = detect_and_snap(src)
    print(f"{len(w)} walls, {len(d)} doors, {len(wi)} windows")
