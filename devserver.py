"""
devserver.py — run the site locally exactly as Vercel routes it.

    python devserver.py            # http://localhost:3000

Serves `public/` as the static root and dispatches /api/generate to the
same handler class Vercel invokes, so what you see here is what deploys.
Dev only; Vercel never runs this file (it is in .vercelignore).
"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")
sys.path.insert(0, ROOT)

from api.generate import handler as GenerateHandler   # noqa: E402


# Inherit from GenerateHandler so its helpers (_send, do_OPTIONS) resolve on
# `self`; SimpleHTTPRequestHandler supplies do_GET for the static files.
class DevHandler(GenerateHandler, SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PUBLIC, **kw)

    def do_POST(self):
        if self.path.rstrip("/") == "/api/generate":
            return GenerateHandler.do_POST(self)
        self.send_error(404, "no such endpoint")

    # Vercel serves /foo and /foo.html interchangeably.
    def translate_path(self, path):
        local = super().translate_path(path)
        if not os.path.exists(local) and not path.endswith("/"):
            alt = local + ".html"
            if os.path.exists(alt):
                return alt
        return local

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"Construct-AI dev server → http://localhost:{port}")
    print(f"  static : {PUBLIC}")
    print(f"  api    : POST /api/generate")
    ThreadingHTTPServer(("127.0.0.1", port), DevHandler).serve_forever()
