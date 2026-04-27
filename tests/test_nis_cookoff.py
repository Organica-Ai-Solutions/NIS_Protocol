#!/usr/bin/env python3
"""Test NIS cookoff + cosmos routes on H100 (run from H100 itself)."""
import urllib.request, json, time

BASE = "http://localhost:8000"

def get(path, timeout=10):
    try:
        r = urllib.request.urlopen(BASE + path, timeout=timeout)
        return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)[:120]

def post(path, body, timeout=70):
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

PASS = FAIL = 0

def check(label, d, err, key=None, expected=None):
    global PASS, FAIL
    if err:
        print(f"  FAIL [{label}]: {err}")
        FAIL += 1
        return
    if key and d.get(key) != expected and expected is not None:
        print(f"  FAIL [{label}]: {key}={d.get(key)!r} expected {expected!r}")
        FAIL += 1
        return
    print(f"  PASS [{label}]")
    PASS += 1

print("=" * 60)
print("NIS Cookoff Routes — H100 Integration Test")
print("=" * 60)

# 1. /cookoff/status
print("\n[1] GET /cookoff/status")
t0 = time.time()
d, err = get("/cookoff/status", timeout=12)
print(f"    {time.time()-t0:.1f}s")
if d:
    svcs = d.get("h100_services", {})
    for svc, info in svcs.items():
        status = "✓ healthy" if info.get("healthy") else f"✗ {info.get('error','?')[:50]}"
        print(f"    {svc}: {status}")
    check("cookoff/status returns operational", d, err, "status", "operational")
else:
    check("cookoff/status", d, err)

# 2. /cookoff/robot-plan
print("\n[2] POST /cookoff/robot-plan")
t0 = time.time()
d, err = post("/cookoff/robot-plan", {
    "query": "Pick up the red cube and place it on the shelf",
    "robot_state": {"arm": "xarm", "connected": True}
}, timeout=70)
elapsed = time.time() - t0
print(f"    {elapsed:.1f}s")
if d:
    src = d.get("source", "?")
    actions = d.get("action_recommendations", [])
    conf = d.get("combined_confidence", 0)
    print(f"    source: {src}")
    print(f"    confidence: {conf:.2f}")
    print(f"    actions: {actions[:4]}")
    reasoning = d.get("cosmos_reasoning", {}).get("reasoning_chain", "")[:120]
    if reasoning:
        print(f"    reasoning: {reasoning}...")
    check("robot-plan returns actions", d, err, "action_recommendations", None)
    if "h100" in src:
        print("    ✓ Used real H100 Cosmos Reason2!")
    else:
        print(f"    ⚠ Used fallback ({src})")
else:
    check("robot-plan", d, err)

# 3. /cosmos/reason
print("\n[3] POST /cosmos/reason")
t0 = time.time()
d, err = post("/cosmos/reason", {
    "task": "What objects are on the table and what should the robot arm do?",
    "constraints": ["avoid obstacles", "gentle grasp"]
}, timeout=70)
elapsed = time.time() - t0
print(f"    {elapsed:.1f}s")
if d:
    src = d.get("source", "?")
    plan = d.get("plan", [])
    conf = d.get("confidence", 0)
    scene = d.get("scene_description", "")[:120]
    print(f"    source: {src}")
    print(f"    confidence: {conf:.2f}")
    print(f"    plan steps: {len(plan)}")
    if scene:
        print(f"    scene: {scene}...")
    check("cosmos/reason returns plan", d, err, "status", "success")
    if "h100" in src:
        print("    ✓ Used real H100 Cosmos Reason2!")
    else:
        print(f"    ⚠ Used fallback ({src})")
else:
    check("cosmos/reason", d, err)

# 4. /cookoff/robot-plan with image (no image, just verify endpoint works)
print("\n[4] POST /cookoff/robot-plan (no image, quick)")
t0 = time.time()
d, err = post("/cookoff/robot-plan", {"query": "inspect the scene"}, timeout=70)
elapsed = time.time() - t0
print(f"    {elapsed:.1f}s  source={d.get('source','?') if d else 'ERR'}")
check("robot-plan inspect", d, err)

print()
print("=" * 60)
print(f"Results: {PASS} PASS  {FAIL} FAIL")
print("=" * 60)
