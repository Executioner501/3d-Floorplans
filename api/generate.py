"""
api/generate.py — Vercel serverless function: floor plan in, 3D house out.

POST { "image": "data:image/png;base64,...", "seed": null, "style": "mixed" }
 ->  { "glb": "<base64>", "stats": { ... } }

The whole pipeline runs here except the diffusion engine (which needs
torch and cannot fit in a serverless bundle). Runtime deps are
onnxruntime + numpy + PIL + trimesh + shapely — see requirements.txt.

Two things differ from a local `python main.py` run, both forced by the
read-only serverless filesystem:
  · design_roof_rag is called with history_file=None. The cross-run
    anti-repetition file cannot be written, so variety comes from a
    random seed per request instead.
  · builder writes the GLB into tempfile.gettempdir() (/tmp), the only
    writable location.
"""
import base64
import io
import json
import mimetypes
import os
import posixpath
import sys
import tempfile
import traceback

# The pipeline modules live at the repository root, one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from http.server import BaseHTTPRequestHandler   # noqa: E402

MAX_UPLOAD_BYTES = 8 * 1024 * 1024      # 8 MB of decoded image
MAX_EDGE = 2400                          # downscale huge plans before detect

# Reported by the health check so a deploy that dropped the weights from the
# bundle is diagnosable without sending an image through.
_MODEL_PATH = os.path.join(ROOT, "best_doors.onnx")

# Static root. Vercel normally serves these from the CDN and they never reach
# the function, but the Python runtime can hand unmatched routes to the
# entrypoint. Serving them here too means the site works either way.
PUBLIC = os.path.join(ROOT, "public")


def _load_image(data_url):
    from PIL import Image
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("image larger than 8 MB")
    img = Image.open(io.BytesIO(raw))
    img.load()
    if max(img.size) > MAX_EDGE:                 # keep detection bounded
        scale = MAX_EDGE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.LANCZOS)
    return img.convert("RGB")


def run_pipeline(image, seed=None, style_pref="mixed", recipe=None):
    """Detect -> scale -> design -> build. Returns (glb_bytes, stats)."""
    import numpy as np
    import detect_onnx
    from scale_utils import estimate_scale
    from roof_ai import design_roof_rag
    from roof_generator import extract_footprint, footprint_metrics
    from builder import export_to_obj

    walls, doors, windows = detect_onnx.detect_and_snap(image)
    if len(walls) < 3:
        raise ValueError(
            f"only {len(walls)} walls detected — this does not look like a "
            f"floor plan, or the lines are too faint to detect")

    scale = estimate_scale(walls, image_path=None, verbose=False)

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2 ** 31)
    # history_file=None: the serverless filesystem is read-only, so the
    # cross-run avoid-list cannot persist. A fresh random seed per request
    # gives the same practical variety.
    params = design_roof_rag(walls, scale=scale, seed=seed,
                             style_pref=style_pref, recipe=recipe,
                             history_file=None)

    out_path = os.path.join(tempfile.gettempdir(), f"house_{seed}.glb")
    export_to_obj(walls, doors, roof_params=params, output_file=out_path,
                  scale=scale, windows=windows)
    with open(out_path, "rb") as f:
        glb = f.read()
    try:
        os.remove(out_path)
    except OSError:
        pass

    poly = extract_footprint(walls, scale)
    m = footprint_metrics(poly) if poly is not None else {}
    stats = {
        "walls": len(walls), "doors": len(doors), "windows": len(windows),
        "recipe": params.get("recipe_name"),
        "recipe_id": params.get("recipe_id"),
        "material": params.get("material"),
        "style": params.get("roof_style"),
        "pitch": round(float(params.get("pitch_angle", 0)), 1),
        "seed": seed,
        "scale_cm_per_px": round(scale * 100, 3),
        "width_m": round(float(m.get("width", 0)), 1),
        "depth_m": round(float(m.get("depth", 0)), 1),
        "area_m2": round(float(m.get("area", 0)), 1),
        "zones": len(params.get("zones", []) or []),
        "size_kb": round(len(glb) / 1024),
    }
    return glb, stats


class handler(BaseHTTPRequestHandler):

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        """Path without query string, trailing slash removed."""
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _drain(self):
        """Consume the request body before answering.

        Replying to a POST without reading its body makes the client see a
        connection reset instead of the status code that explains why.
        """
        try:
            n = int(self.headers.get("Content-Length") or 0)
            while n > 0:
                chunk = self.rfile.read(min(n, 65536))
                if not chunk:
                    break
                n -= len(chunk)
        except Exception:
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()

    def _static_path(self, route):
        """Resolve a URL path to a file under public/, or None.

        Rejects anything that escapes PUBLIC after normalisation — `..`
        segments and absolute paths included — so a request can never read
        the pipeline source or the model weights.
        """
        if not os.path.isdir(PUBLIC):
            return None
        rel = posixpath.normpath(route.lstrip("/"))
        if rel in ("", ".", "/"):
            rel = "index.html"
        if rel.startswith("..") or os.path.isabs(rel):
            return None
        target = os.path.realpath(os.path.join(PUBLIC, *rel.split("/")))
        if os.path.commonpath([target, os.path.realpath(PUBLIC)]) != \
                os.path.realpath(PUBLIC):
            return None
        if os.path.isdir(target):
            target = os.path.join(target, "index.html")
        if not os.path.isfile(target):
            # Vercel serves /viewer and /viewer.html interchangeably
            alt = target + ".html"
            if os.path.isfile(alt):
                return alt
            return None
        return target

    def _send_file(self, path):
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in (
                "application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self._route()
        if route in ("/api/health", "/health"):
            return self._send(200, {"ok": True,
                                    "model": os.path.exists(_MODEL_PATH),
                                    "static": os.path.isdir(PUBLIC),
                                    "endpoint": "POST /api/generate"})
        if route.startswith("/api/"):
            return self._send(404, {"error": "not found"})
        target = self._static_path(route)
        if target:
            return self._send_file(target)
        if route == "/":
            # In production the CDN serves public/ and those files are NOT in
            # the function bundle (/api/health reports "static": false), so
            # the index cannot be read here — but "/" alone does not resolve
            # to index.html and falls through to this function. Point at it
            # explicitly. No redirect loop: /index.html is served by the CDN.
            self.send_response(302)
            self.send_header("Location", "/index.html")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._route().endswith("/generate"):
            self._drain()
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 12 * 1024 * 1024:
                return self._send(413, {"error": "request body too large"})
            req = json.loads(self.rfile.read(length))
        except Exception:
            return self._send(400, {"error": "malformed request body"})

        if not req.get("image"):
            return self._send(400, {"error": "no image supplied"})

        try:
            image = _load_image(req["image"])
        except Exception as e:
            return self._send(400, {"error": f"could not read image: {e}"})

        try:
            seed = req.get("seed")
            glb, stats = run_pipeline(
                image,
                seed=int(seed) if seed not in (None, "") else None,
                style_pref=req.get("style") or "mixed",
                recipe=req.get("recipe") or None)
        except ValueError as e:
            return self._send(422, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": f"generation failed: {e}"})

        self._send(200, {"glb": base64.b64encode(glb).decode("ascii"),
                         "stats": stats})
