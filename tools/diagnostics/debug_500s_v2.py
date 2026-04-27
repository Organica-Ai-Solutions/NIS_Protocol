#!/usr/bin/env python3
"""Get full 500 error bodies from Predict2.5 and Transfer2.5."""
import urllib.request, json, base64, io

def post_verbose(url, payload, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.request.HTTPError as e:
        body = e.read().decode(errors="replace")
        return None, f"HTTP {e.code}: {body[:800]}"
    except Exception as e:
        return None, str(e)[:300]

# Build a proper 848x480 seed image
from PIL import Image as _PIL
import numpy as _np
arr = _np.zeros((480, 848, 3), dtype=_np.uint8)
arr[:, :] = [80, 100, 120]
buf = io.BytesIO()
_PIL.fromarray(arr).save(buf, format="JPEG", quality=85)
seed_b64 = base64.b64encode(buf.getvalue()).decode()
print(f"Seed image: {len(seed_b64)} chars b64")

print()
print("=== Predict2.5 /video2world full error ===")
d, err = post_verbose("http://localhost:8200/video2world", {
    "prompt": "A robot arm picks up a red cube",
    "image_b64": seed_b64,
    "num_frames": 25,
    "fps": 10,
    "height": 480,
    "width": 848,
    "num_inference_steps": 20,
    "guidance_scale": 7.0,
    "seed": 42,
}, timeout=60)
if err:
    print(err)
else:
    print("OK:", str(d)[:200])

print()
print("=== Transfer2.5 /transfer full error ===")
d, err = post_verbose("http://localhost:8300/transfer", {
    "demo": "car_edge",
    "control_type": "edge",
    "guidance": 3.0,
}, timeout=60)
if err:
    print(err)
else:
    print("OK:", str(d)[:200])
