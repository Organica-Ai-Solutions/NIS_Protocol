#!/usr/bin/env python3
"""Quick end-to-end test of H100 Cosmos endpoints from the H100 itself."""
import json, urllib.request, time

BASE = "http://localhost"

def post(port, path, body):
    url = f"{BASE}:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        t0 = time.time()
        r = urllib.request.urlopen(req, timeout=60)
        d = json.loads(r.read())
        return d, round((time.time()-t0)*1000)
    except Exception as e:
        return {"error": str(e)}, 0

def get(port, path):
    url = f"{BASE}:{port}{path}"
    try:
        r = urllib.request.urlopen(url, timeout=10)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

print("=" * 60)
print("H100 Cosmos Stack — End-to-End Test")
print("=" * 60)

# Health checks
for name, port in [("Reason2", 8100), ("Predict2.5", 8200), ("Transfer2.5", 8300)]:
    h = get(port, "/health")
    if not isinstance(h, dict):
        print(f"  {name} :{port} — {'healthy' if 'healthy' in str(h) else 'FAIL'}  raw={str(h)[:80]}")
        continue
    ok = "healthy" if h.get("status") == "healthy" or h.get("ready") else "FAIL"
    print(f"  {name} :{port} — {ok}  {h.get('model','?')}  GPU free: {h.get('gpu',{}).get('free_gb','?') if isinstance(h.get('gpu'),dict) else '?'} GB")

print()

# Reason2 — robot-plan
print("--- Reason2: /robot-plan ---")
d, ms = post(8100, "/robot-plan", {"command": "Pick up the red cube and place it on the shelf", "robot_type": "xarm"})
if d.get("error"):
    print(f"  FAIL: {d['error']}")
else:
    steps = d.get("steps", [])
    print(f"  OK  {ms}ms  steps={len(steps)}")
    for s in steps[:4]:
        print(f"    • {s.get('action','?')}: {s.get('description','')[:60]}")
    print(f"  answer: {d.get('answer','')[:120]}")

print()

# Reason2 — /reason
print("--- Reason2: /reason ---")
d, ms = post(8100, "/reason", {"query": "What objects are on the table and what should the robot arm do?", "max_tokens": 300, "use_think": True})
if d.get("error"):
    print(f"  FAIL: {d['error']}")
else:
    print(f"  OK  {ms}ms")
    print(f"  reasoning: {d.get('reasoning','')[:200]}")
    print(f"  answer:    {d.get('answer','')[:120]}")

print()

# Predict2.5 — health only (inference takes too long for quick test)
print("--- Predict2.5: /health ---")
h = get(8200, "/health")
print(f"  {'OK' if h.get('status')=='healthy' or h.get('ready') else 'FAIL'}  {h}")

print()

# Transfer2.5 — submit job
print("--- Transfer2.5: /transfer/submit ---")
d, ms = post(8300, "/transfer/submit", {"demo": "car_edge", "control_type": "edge", "guidance": 3.0})
if d.get("error"):
    print(f"  FAIL: {d['error']}")
else:
    job_id = d.get("job_id", "?")
    print(f"  OK  {ms}ms  job_id={job_id}")
    # Poll once to confirm job is running
    import time; time.sleep(3)
    s = get(8300, f"/transfer/status/{job_id}")
    print(f"  status after 3s: {s.get('status','?')}")

print()
print("=" * 60)
print("Test complete.")
