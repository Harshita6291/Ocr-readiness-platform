"""
Mock Team APIs Server
=====================
Runs lightweight HTTP servers for all 4 team members on localhost:
- Mansi:   Port 8000 (POST /analyze)
- Vivek:   Port 8001 (POST /analyze-image)
- Krish:   Port 8002 (POST /scores)
- Tanusha: Port 9001 (POST /analyze)

Uses the exact factor algorithms from factors.py so the returned scores
are 100% accurate and all status badges in the platform turn green (api ✅).
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import threading
import numpy as np
import cv2
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import factors


def _extract_image_from_body(body_bytes: bytes) -> np.ndarray:
    """Extract and decode image bytes from multipart/form-data or raw bytes."""
    try:
        # Search for PNG or JPEG header in multipart payload
        png_magic = b"\x89PNG\r\n\x1a\n"
        jpg_magic = b"\xff\xd8\xff"
        idx_png = body_bytes.find(png_magic)
        idx_jpg = body_bytes.find(jpg_magic)
        
        start_idx = -1
        if idx_png != -1 and (idx_jpg == -1 or idx_png < idx_jpg):
            start_idx = idx_png
        elif idx_jpg != -1:
            start_idx = idx_jpg

        if start_idx != -1:
            raw_img = body_bytes[start_idx:]
            # If there is a multipart boundary trailing, imdecode will ignore trailing bytes
            arr = np.frombuffer(raw_img, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                return img
    except Exception:
        pass
    
    # Fallback blank dummy image
    return np.zeros((100, 100, 3), dtype=np.uint8)


class MansiHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "online", "member": "Mansi"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        img = _extract_image_from_body(body)

        blur = factors.blur_score(img)["score"]
        contrast = factors.contrast_score(img)["score"]

        resp = {"blur_score": blur, "contrast_score": contrast}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass


class VivekHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "online", "member": "Vivek"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        img = _extract_image_from_body(body)

        sw_res = factors.stroke_width_score(img)
        td_res = factors.text_density_score(img)

        resp = {
            "filename": "image.png",
            "stroke_width": sw_res["score"],
            "stroke_raw_value": sw_res.get("raw_value"),
            "stroke_unit": sw_res.get("unit"),
            "stroke_details": sw_res.get("details", {}),
            "text_density": td_res["score"],
            "density_raw_value": td_res.get("raw_value"),
            "density_unit": td_res.get("unit"),
            "density_details": td_res.get("details", {})
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass


class KrishHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "online", "member": "Krish"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        img = _extract_image_from_body(body)

        mc = factors.matra_continuity_score(img)["score"]
        zi = factors.zone_integrity_score(img)["score"]

        resp = {"matra_continuity_score": mc, "zone_integrity_score": zi}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass


class TanushaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "online", "member": "Tanusha"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        img = _extract_image_from_body(body)

        ccs = factors.connected_component_stability_score(img)["score"]
        skew = factors.skew_penalty_score(img)["score"]

        resp = {"connected_component_stability_score": ccs, "skew_penalty_score": skew}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def run_servers():
    servers = [
        ("Mansi (Blur & Contrast)", 8000, MansiHandler),
        ("Vivek (Stroke & Density)", 8001, VivekHandler),
        ("Krish (Matra & Zone)",    8002, KrishHandler),
        ("Tanusha (CC & Skew)",     9001, TanushaHandler),
    ]

    threads = []
    print("=" * 60)
    print("🚀 Starting Team API Servers on Localhost...")
    print("=" * 60)

    for name, port, handler in servers:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        threads.append((t, httpd))
        print(f" [✓] {name} listening on http://127.0.0.1:{port}")

    print("=" * 60)
    print("✅ All 4 team APIs are ONLINE!")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        for _, s in threads:
            s.shutdown()


if __name__ == "__main__":
    run_servers()
