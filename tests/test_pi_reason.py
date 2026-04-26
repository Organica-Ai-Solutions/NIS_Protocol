#!/usr/bin/env python3
"""Test Pi cookoff/cosmos/reason endpoint properly."""
import urllib.request, json, time

BASE = "http://localhost:8085"

def post(path, body=None, timeout=90):
    data = json.dumps(body or {}).encode() if body is not None else b"{}"
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        try:
            body_err = e.read().decode()
        except Exception:
            body_err = ""
        return None, f"HTTP {e.code}: {body_err[:120]}"
    except Exception as e:
        return None, str(e)[:120]

def get(path, timeout=10):
    try:
        r = urllib.request.urlopen(BASE + path, timeout=timeout)
        return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)[:120]

print("=" * 55)
print("Pi Cookoff — Full Smoke Test")
print("=" * 55)

# 1. Agent health
d, err = get("/health")
print(f"[health]   {'OK v'+d.get('version') if d else 'FAIL: '+str(err)}")

# 2. Cookoff status
d, err = get("/cookoff/status")
print(f"[cookoff]  {'mode='+d.get('mode','?') if d else 'FAIL: '+str(err)}")

# 3. cosmos/reason — with proper JSON body
print("\n[reason]   POST /cookoff/cosmos/reason ...")
t0 = time.time()
d, err = post("/cookoff/cosmos/reason", {"query": "What do you see? What should the robot arm do?"})
elapsed = time.time() - t0
if d:
    src   = d.get("source", "?")
    ok    = d.get("ok", "?")
    scene = str(d.get("scene", d.get("scene_description", "")))[:100]
    think = str(d.get("thinking", d.get("reasoning", "")))[:80]
    print(f"  ok={ok}  source={src}  {elapsed:.1f}s")
    if scene: print(f"  scene: {scene}")
    if think: print(f"  think: {think}...")
else:
    print(f"  FAIL: {err}  ({elapsed:.1f}s)")

# 4. cosmos/trajectory
print("\n[traj]     POST /cookoff/cosmos/trajectory ...")
t0 = time.time()
d, err = post("/cookoff/cosmos/trajectory", {"task": "pick up the nearest object"})
elapsed = time.time() - t0
if d:
    traj = d.get("trajectory", [])
    src  = d.get("source", d.get("model", "?"))
    print(f"  ok={d.get('ok')}  source={src}  traj_pts={len(traj)}  {elapsed:.1f}s")
else:
    print(f"  FAIL: {err}  ({elapsed:.1f}s)")

# 5. transfer — with a dummy source frame (camera snapshot)
print("\n[transfer] POST /cookoff/transfer (edge, 0.7) ...")
# First capture a frame
snap, serr = get("/camera/snapshot", timeout=10)
if snap and snap.get("image_base64"):
    b64 = snap["image_base64"]
    print(f"  snapshot: {len(b64)} chars")
    t0 = time.time()
    d, err = post("/cookoff/transfer", {
        "type": "edge", "strength": 0.7,
        "source_image": b64, "target_image": b64
    }, timeout=35)
    elapsed = time.time() - t0
    if d:
        src = d.get("source", "?")
        has_v = bool(d.get("video_base64") or d.get("video_b64"))
        has_i = bool(d.get("result_image") or d.get("transferred_image"))
        print(f"  source={src}  video={has_v}  image={has_i}  {elapsed:.1f}s")
        if d.get("error"): print(f"  error: {d['error'][:80]}")
    else:
        print(f"  FAIL: {err}  ({elapsed:.1f}s)")
else:
    print(f"  snapshot failed: {serr} — skipping transfer test")

print("\n" + "=" * 55)
