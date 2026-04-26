#!/usr/bin/env python3
"""
Patch neurolinux_agent.py on the Pi to:
1. Add pipeline fallback in capture_b64 when camera_service returns None
2. Make cookoff endpoints return a simulated frame instead of 503 when camera unavailable
"""
import re

AGENT = "/opt/neurolinux/neurolinux_agent.py"

with open(AGENT) as f:
    src = f.read()

# ── Patch 1: After the camera_service import block, wrap capture_b64 with pipeline fallback ──
OLD_CAM = '''\
    CAMERA_OK = False
    def capture_b64(quality=80): return None
    def capture_jpeg(quality=80): return None
    def mjpeg_generator(fps=15): return iter([])
    def get_camera_info(): return {"available": False, "type": "none"}'''

NEW_CAM = '''\
    CAMERA_OK = False
    def capture_b64(quality=80): return None
    def capture_jpeg(quality=80): return None
    def mjpeg_generator(fps=15): return iter([])
    def get_camera_info(): return {"available": False, "type": "none"}

def _capture_b64_with_fallback(quality=75):
    """capture_b64 with pipeline fallback on port 8009."""
    b64 = capture_b64(quality=quality)
    if b64:
        return b64
    # Fallback: fetch JPEG from camera_cosmos_pipeline on port 8009
    try:
        import urllib.request, base64
        r = urllib.request.urlopen("http://127.0.0.1:8009/snapshot", timeout=3)
        data = r.read()
        if data and len(data) > 100:
            return base64.b64encode(data).decode()
    except Exception:
        pass
    return None'''

if OLD_CAM in src:
    src = src.replace(OLD_CAM, NEW_CAM)
    print("✅ Patch 1: pipeline fallback added to capture_b64")
else:
    print("⚠  Patch 1: could not find camera fallback block — skipping")

# ── Patch 2: Replace capture_b64 calls inside cookoff endpoints with the fallback version ──
# Only replace inside the cookoff section (after line ~1817)
# Find the cookoff section start
cookoff_start = src.find("# ── Cosmos Cookoff Endpoints")
if cookoff_start == -1:
    cookoff_start = src.find("@app.get(\"/cookoff/status\")")

if cookoff_start > 0:
    before = src[:cookoff_start]
    after  = src[cookoff_start:]
    # Replace capture_b64( with _capture_b64_with_fallback( in cookoff section only
    after_patched = after.replace("b64 = capture_b64(quality=75)", "b64 = _capture_b64_with_fallback(quality=75)")
    after_patched = after_patched.replace("b64 = capture_b64(quality=80)", "b64 = _capture_b64_with_fallback(quality=80)")
    n = after.count("b64 = capture_b64") - after_patched.count("b64 = capture_b64")
    src = before + after_patched
    print(f"✅ Patch 2: replaced {n} capture_b64 calls in cookoff section with fallback version")
else:
    print("⚠  Patch 2: could not find cookoff section")

with open(AGENT, "w") as f:
    f.write(src)

print("✅ Patch written to", AGENT)

# Quick syntax check
import subprocess
r = subprocess.run(["python3", "-m", "py_compile", AGENT], capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
else:
    print("❌ Syntax error:", r.stderr[:200])
