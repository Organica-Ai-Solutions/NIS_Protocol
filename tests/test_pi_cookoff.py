#!/usr/bin/env python3
"""Quick cookoff smoke test from the Pi itself."""
import urllib.request, json, time

BASE = "http://localhost:8085"

def get(path, timeout=10):
    try:
        r = urllib.request.urlopen(BASE + path, timeout=timeout)
        return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)[:120]

def post(path, body, timeout=90):
    try:
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)[:120]

print("=" * 55)
print("Pi NeuroLinux Agent — Cookoff Smoke Test")
print("=" * 55)

# 1. Agent health
d, err = get("/health")
if d:
    print(f"[health]  v{d.get('version')}  xarm={d.get('xarm')}  cam={d.get('camera',{}).get('available')}")
else:
    print(f"[health]  FAIL: {err}")

# 2. Cookoff status
d, err = get("/cookoff/status")
if d:
    print(f"[cookoff] mode={d.get('mode')}  initialized={d.get('initialized')}")
else:
    print(f"[cookoff] FAIL: {err}")

# 3. Single cosmos reason (Pi -> NIS -> H100)
print("\n[reason]  calling /cookoff/cosmos/reason (Pi->NIS->H100)...")
t0 = time.time()
d, err = post("/cookoff/cosmos/reason", {"query": "What do you see? What should the robot arm do?"})
elapsed = time.time() - t0
if d:
    src  = d.get("source", d.get("last_source", "?"))
    ok   = d.get("ok", "?")
    scene = str(d.get("scene", d.get("scene_description", "")))[:100]
    think = str(d.get("thinking", d.get("reasoning", "")))[:80]
    print(f"  ok={ok}  source={src}  {elapsed:.1f}s")
    if scene: print(f"  scene:   {scene}")
    if think: print(f"  think:   {think}...")
else:
    print(f"  FAIL: {err}  {elapsed:.1f}s")

# 4. Transfer endpoint (just check it's reachable — don't wait for full job)
print("\n[transfer] checking /cookoff/transfer endpoint...")
d, err = post("/cookoff/transfer", {"type": "edge", "strength": 0.7}, timeout=35)
if d:
    src = d.get("source", "?")
    has_video = bool(d.get("video_base64") or d.get("video_b64"))
    has_img   = bool(d.get("result_image") or d.get("transferred_image"))
    print(f"  source={src}  video={has_video}  image={has_img}")
    if d.get("error"):
        print(f"  error: {d['error'][:80]}")
else:
    print(f"  FAIL: {err}")

print("\n" + "=" * 55)
print("Done.")
