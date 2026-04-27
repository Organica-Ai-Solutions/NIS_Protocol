#!/usr/bin/env python3
"""Run the full Cosmos stack test on H100 directly (single SSH session)."""
import subprocess, sys

SSH = [
    "ssh", "-o", "ConnectTimeout=90", "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=20", "-o", "TCPKeepAlive=yes",
    "awesome-gpu-name"
]

SCRIPT = r"""
import json, time, urllib.request, base64, io, sys

def post(url, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), time.time()-start, None
    except urllib.request.HTTPError as e:
        return None, time.time()-start, f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}"
    except Exception as e:
        return None, time.time()-start, str(e)[:200]

PASS = 0; FAIL = 0

# ── TEST 1: Health ──────────────────────────────────────────────────────────
print("="*60)
print("TEST 1: Health Checks")
print("="*60)
for port, name in [(8000,"NIS"),(8100,"Reason2"),(8200,"Predict2.5"),(8300,"Transfer2.5"),(8400,"Demo")]:
    try:
        r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
        d = json.loads(r.read())
        print(f"  :{port} {name}: {d.get('status','ok')}")
    except Exception as e:
        print(f"  :{port} {name}: FAIL - {str(e)[:60]}")
print()

# ── TEST 2: Reason2 ─────────────────────────────────────────────────────────
print("="*60)
print("TEST 2: Reason2 - Physics Question")
print("="*60)
d, t, err = post("http://localhost:8100/reason", {
    "query": "A robot arm holds a full glass of water and rotates 90 degrees. What happens?",
    "max_tokens": 150, "temperature": 0.7
}, timeout=60)
if err:
    print(f"  FAIL: {err}"); FAIL+=1
else:
    ans = d.get("reasoning", d.get("response", str(d)))[:200]
    print(f"  Answer: {ans}")
    print(f"  Latency: {d.get('latency_ms','?')}ms  Total: {t:.1f}s")
    print(f"  RESULT: PASS"); PASS+=1
print()

# ── TEST 3: Predict2.5 video2world ──────────────────────────────────────────
print("="*60)
print("TEST 3: Predict2.5 Video2World")
print("="*60)
try:
    from PIL import Image
    import numpy as np
    arr = np.zeros((480, 848, 3), dtype=np.uint8); arr[:,:,:]=[80,100,120]
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    seed_b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"  Seed image: {len(seed_b64)} chars")
except Exception as e:
    print(f"  FAIL: PIL unavailable: {e}"); FAIL+=1; seed_b64=None

if seed_b64:
    d, t, err = post("http://localhost:8200/video2world", {
        "prompt": "A robot arm picks up a red cube from a table and places it in a bin",
        "image_b64": seed_b64,
        "num_frames": 25, "fps": 10,
        "height": 480, "width": 848,
        "num_inference_steps": 20,
        "guidance_scale": 7.0, "seed": 42,
    }, timeout=120)
    if err:
        print(f"  FAIL: {err}"); FAIL+=1
    else:
        vid = d.get("video_b64","")
        print(f"  Video: {len(vid)*3//4//1024} KB  latency: {d.get('latency_ms')}ms  total: {t:.1f}s")
        print(f"  RESULT: PASS"); PASS+=1
print()

# ── TEST 4: Transfer2.5 ─────────────────────────────────────────────────────
print("="*60)
print("TEST 4: Transfer2.5 - Car Edge Demo")
print("="*60)
print("  Running inference (3-8 min)...")
d, t, err = post("http://localhost:8300/transfer", {
    "demo": "car_edge", "control_type": "edge", "guidance": 3.0
}, timeout=600)
if err:
    print(f"  FAIL: {err}"); FAIL+=1
else:
    vid = d.get("video_b64","")
    if vid:
        print(f"  Video: {len(vid)*3//4//1024} KB  latency: {d.get('latency_ms')}ms  total: {t:.1f}s")
        print(f"  All videos: {d.get('all_videos',{})}")
        print(f"  RESULT: PASS"); PASS+=1
    else:
        print(f"  FAIL: no video in response: {str(d)[:200]}"); FAIL+=1
print()

# ── TEST 5: NIS /cosmos/reason ──────────────────────────────────────────────
print("="*60)
print("TEST 5: NIS /cosmos/reason (image decode fix)")
print("="*60)
try:
    from PIL import Image
    import numpy as np
    arr2 = np.zeros((480, 640, 3), dtype=np.uint8); arr2[:,:,:]=[100,120,80]
    buf2 = io.BytesIO(); Image.fromarray(arr2).save(buf2, format="JPEG", quality=75)
    img_b64 = base64.b64encode(buf2.getvalue()).decode()
except Exception as e:
    img_b64 = None; print(f"  PIL unavailable: {e}")

if img_b64:
    d, t, err = post("http://localhost:8000/cosmos/reason", {
        "task": "Pick up the red cube",
        "image_data": img_b64,
        "context": {"robot": "arm"}
    }, timeout=60)
    if err:
        print(f"  FAIL: {err}"); FAIL+=1
    else:
        print(f"  Plans: {d.get('plans',d.get('plan','?'))[:150]}")
        print(f"  Total: {t:.1f}s  RESULT: PASS"); PASS+=1
print()

print("="*60)
print(f"RESULTS: {PASS} PASS, {FAIL} FAIL out of {PASS+FAIL} tests")
print("="*60)
"""

print("Connecting to H100 and running full Cosmos stack test...")
print("(Transfer2.5 takes 3-8 min — please wait)\n")

r = subprocess.run(
    SSH + [f"/data/organica-ai/NIS_Protocol/venv/bin/python -c {repr(SCRIPT)}"],
    timeout=900,
    encoding="utf-8",
    errors="replace"
)
sys.exit(r.returncode)
