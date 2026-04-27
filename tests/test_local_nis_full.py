#!/usr/bin/env python3
"""
Full test of local PC NIS (192.168.1.160:8000) → H100 via SSH tunnel.
This is the path the Pi uses: Pi → PC NIS → H100 Cosmos.
"""
import urllib.request, json, time

NIS = "http://localhost:8000"
PASS = 0
FAIL = 0

def get(path, timeout=10):
    try:
        r = urllib.request.urlopen(NIS + path, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:80]}"
    except Exception as e:
        return None, str(e)[:80]

def post(path, body=None, timeout=90):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        NIS + path, data=data,
        headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:80]}"
    except Exception as e:
        return None, str(e)[:80]

def check(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ PASS [{label}]{(' — '+detail) if detail else ''}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL [{label}]{(' — '+detail) if detail else ''}")

print("=" * 60)
print("PC NIS → H100 Full Chain Test")
print(f"NIS: {NIS}")
print("=" * 60)

# 1. NIS health
print("\n[1] NIS health")
d, err = get("/health")
check("nis-health", d and d.get("status") == "healthy",
      f"v{d.get('version')} {d.get('modular_routes')} routes" if d else err)

# 2. Cookoff status (checks H100 via tunnel)
print("\n[2] Cookoff status (H100 via tunnel)")
d, err = get("/cookoff/status")
if d:
    svcs = d.get("h100_services", {})
    all_healthy = all(v.get("healthy") for v in svcs.values())
    check("cookoff-status", all_healthy,
          " ".join(f"{k}={'✓' if v.get('healthy') else '✗'}" for k,v in svcs.items()))
else:
    check("cookoff-status", False, err)

# 3. Cosmos reason (PC NIS → H100 Reason2)
print("\n[3] /cosmos/reason (PC NIS → H100 Reason2)")
t0 = time.time()
d, err = post("/cosmos/reason", {
    "task": "A red cube is on a table. What should the robot arm do?",
    "constraints": []
})
elapsed = time.time() - t0
ok = d and d.get("status") == "success"
src = d.get("source", "?") if d else "?"
conf = d.get("confidence", "?") if d else "?"
plan = d.get("plan", []) if d else []
check("cosmos-reason", ok, f"source={src} conf={conf} plan_steps={len(plan)} {elapsed:.1f}s")
if plan:
    print(f"     plan[0]: {plan[0].get('description','')[:80]}")

# 4. Robot plan (PC NIS → H100 Reason2 /robot-plan)
print("\n[4] /cookoff/robot-plan (PC NIS → H100)")
t0 = time.time()
d, err = post("/cookoff/robot-plan", {
    "query": "Pick up the red cube and place it in the bowl"
})
elapsed = time.time() - t0
ok = d and (d.get("action_recommendations") or d.get("source"))
src = d.get("source", "?") if d else "?"
conf = d.get("combined_confidence", "?") if d else "?"
actions = d.get("action_recommendations", []) if d else []
check("robot-plan", ok, f"source={src} conf={conf} actions={len(actions)} {elapsed:.1f}s")
if actions:
    print(f"     actions: {actions[:3]}")

# 5. Transfer (PC NIS → H100 Transfer2.5 via tunnel)
print("\n[5] /cookoff/transfer (PC NIS → H100 Transfer2.5)")
t0 = time.time()
d, err = post("/cookoff/transfer", {
    "type": "edge",
    "strength": 0.7
}, timeout=120)
elapsed = time.time() - t0
ok = d and (d.get("ok") or d.get("video_base64") or d.get("transferred_image"))
src = d.get("source", "?") if d else "?"
check("transfer25", ok, f"source={src} {elapsed:.1f}s")
if d and d.get("error"):
    print(f"     error: {str(d['error'])[:80]}")

print("\n" + "=" * 60)
print(f"Results: {PASS} PASS  {FAIL} FAIL")
print("=" * 60)
if FAIL == 0:
    print("\n✅ Full chain working: Pi → PC NIS → H100")
    print("   Keep start_tunnel.ps1 running whenever Pi is in use.")
