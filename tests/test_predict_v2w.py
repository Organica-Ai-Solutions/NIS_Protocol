#!/usr/bin/env python3
"""Test Predict2.5 video2world via SSH tunnel."""
import json, time, urllib.request, subprocess

PREDICT = "http://localhost:8200"

def post(url, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), time.time() - start, None
    except urllib.request.HTTPError as e:
        return None, time.time() - start, f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:
        return None, time.time() - start, str(e)[:200]

print("=== Generating seed image on H100 ===")
r = subprocess.run(
    ["ssh", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=10", "awesome-gpu-name",
     "/data/organica-ai/NIS_Protocol/venv/bin/python /tmp/gen_seed.py"],
    capture_output=True, text=True, timeout=30
)
seed = r.stdout.strip()
if not seed:
    # Upload gen_seed.py first
    import subprocess as sp
    sp.run(["ssh", "-o", "ConnectTimeout=30", "awesome-gpu-name",
            "cat > /tmp/gen_seed.py << 'PYEOF'\n"
            "import base64, io\n"
            "from PIL import Image\n"
            "import numpy as np\n"
            "arr = np.zeros((480, 848, 3), dtype=np.uint8)\n"
            "arr[:, :, :] = [80, 100, 120]\n"
            "buf = io.BytesIO()\n"
            "Image.fromarray(arr).save(buf, format='JPEG', quality=85)\n"
            "print(base64.b64encode(buf.getvalue()).decode())\n"
            "PYEOF"],
           timeout=15)
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=10", "awesome-gpu-name",
         "/data/organica-ai/NIS_Protocol/venv/bin/python /tmp/gen_seed.py"],
        capture_output=True, text=True, timeout=20
    )
    seed = r.stdout.strip()

if not seed:
    print(f"FAIL: could not generate seed image: {r.stderr[:100]}")
    exit(1)

print(f"  Seed: {len(seed)} chars b64")

print()
print("=== POST /video2world ===")
d, elapsed, err = post(f"{PREDICT}/video2world", {
    "prompt": "A robot arm picks up a red cube from a table and places it in a bin",
    "image_b64": seed,
    "num_frames": 25,
    "fps": 10,
    "height": 480,
    "width": 848,
    "num_inference_steps": 20,
    "guidance_scale": 7.0,
    "seed": 42,
}, timeout=120)

if err:
    print(f"  FAIL: {err}")
else:
    vid = d.get("video_b64", "")
    img = d.get("image_b64", "")
    size_kb = len(vid) * 3 // 4 // 1024
    print(f"  Video: {size_kb} KB")
    print(f"  Preview frame: {len(img)*3//4//1024} KB")
    print(f"  Resolution: {d.get('resolution')}  Frames: {d.get('num_frames')}  FPS: {d.get('fps')}")
    print(f"  Server latency: {d.get('latency_ms')}ms  Total: {elapsed:.1f}s")
    print(f"  RESULT: PASS")
