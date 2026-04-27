#!/usr/bin/env python3
"""Full Cosmos stack test — runs ON H100 directly."""
import json, time, urllib.request, base64, io

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

# TEST 1: Health
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

# TEST 2: Reason2
print("="*60)
print("TEST 2: Reason2 - Physics Question")
print("="*60)
d, t, err = post("http://localhost:8100/reason", {
    "query": "A robot arm holds a full glass of water and rotates 90 degrees. What happens?",
    "max_tokens": 150, "temperature": 0.7
}, timeout=60)
if err:
    print(f"  FAIL: {err}"); FAIL += 1
else:
    ans = d.get("reasoning", d.get("response", str(d)))[:200]
    print(f"  Answer: {ans}")
    print(f"  Latency: {d.get('latency_ms','?')}ms  Total: {t:.1f}s")
    print(f"  RESULT: PASS"); PASS += 1
print()

# TEST 3: Predict2.5 video2world
print("="*60)
print("TEST 3: Predict2.5 Video2World")
print("="*60)
seed_b64 = None
try:
    from PIL import Image
    import numpy as np
    arr = np.zeros((480, 848, 3), dtype=np.uint8)
    arr[:, :, :] = [80, 100, 120]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    seed_b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"  Seed image: {len(seed_b64)} chars")
except Exception as e:
    print(f"  FAIL: PIL unavailable: {e}"); FAIL += 1

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
        print(f"  FAIL: {err}"); FAIL += 1
    else:
        vid = d.get("video_b64", "")
        print(f"  Video: {len(vid)*3//4//1024} KB  latency: {d.get('latency_ms')}ms  total: {t:.1f}s")
        print(f"  RESULT: PASS"); PASS += 1
print()

# TEST 4: Transfer2.5 — submit job then poll (avoids long-connection timeout)
print("="*60)
print("TEST 4: Transfer2.5 - Car Edge Demo")
print("="*60)
print("  Submitting inference job...")
t4_start = time.time()
d, t, err = post("http://localhost:8300/transfer/submit", {
    "demo": "car_edge", "control_type": "edge", "guidance": 3.0
}, timeout=30)
if err:
    print(f"  FAIL (submit): {err}"); FAIL += 1
else:
    job_id = d.get("job_id")
    print(f"  Job submitted: {job_id} — polling every 20s (up to 20 min)...")
    t4_done = False
    for attempt in range(60):
        time.sleep(20)
        try:
            r = urllib.request.urlopen(
                f"http://localhost:8300/transfer/status/{job_id}", timeout=10)
            s = json.loads(r.read())
            elapsed = time.time() - t4_start
            if s.get("status") == "running":
                print(f"    [{attempt+1}] still running... {elapsed:.0f}s elapsed")
                continue
            vid = s.get("video_b64", "")
            if vid:
                print(f"  Video: {len(vid)*3//4//1024} KB  latency: {s.get('latency_ms')}ms  total: {elapsed:.1f}s")
                print(f"  All videos: {s.get('all_videos', {})}")
                print(f"  RESULT: PASS"); PASS += 1
            else:
                print(f"  FAIL: {s}"); FAIL += 1
            t4_done = True
            break
        except Exception as e:
            print(f"    [{attempt+1}] poll error: {e}")
    if not t4_done:
        print(f"  FAIL: timed out after 20 min"); FAIL += 1
print()

# TEST 5: NIS /cosmos/reason (image decode fix)
print("="*60)
print("TEST 5: NIS /cosmos/reason (image decode fix)")
print("="*60)
img_b64 = None
try:
    from PIL import Image
    import numpy as np
    arr2 = np.zeros((480, 640, 3), dtype=np.uint8)
    arr2[:, :, :] = [100, 120, 80]
    buf2 = io.BytesIO()
    Image.fromarray(arr2).save(buf2, format="JPEG", quality=75)
    img_b64 = base64.b64encode(buf2.getvalue()).decode()
except Exception as e:
    print(f"  PIL unavailable: {e}"); FAIL += 1

if img_b64:
    d, t, err = post("http://localhost:8000/cosmos/reason", {
        "task": "Pick up the red cube",
        "image_data": img_b64,
        "context": {"robot": "arm"}
    }, timeout=60)
    if err:
        print(f"  FAIL: {err}"); FAIL += 1
    else:
        print(f"  Plans: {str(d.get('plans', d.get('plan', d)))[:200]}")
        print(f"  Total: {t:.1f}s  RESULT: PASS"); PASS += 1
print()

print("="*60)
print(f"RESULTS: {PASS} PASS, {FAIL} FAIL out of {PASS+FAIL} tests")
print("="*60)
