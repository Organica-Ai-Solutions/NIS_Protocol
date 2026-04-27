#!/usr/bin/env python3
"""
IMMEDIATE ARM TEST — run this the second Pi comes online.
Tests xArm movement directly, then Cosmos reason -> arm action.

Usage: python test_arm_now.py
       python test_arm_now.py --ip 192.168.1.X   (if neurolinux.local doesn't resolve)
"""
import urllib.request, json, time, sys

# Auto-detect Pi IP
PI_CANDIDATES = ["http://neurolinux.local:8085", "http://192.168.1.100:8085",
                 "http://192.168.1.101:8085", "http://192.168.1.102:8085",
                 "http://192.168.1.103:8085"]

if "--ip" in sys.argv:
    idx = sys.argv.index("--ip")
    PI_CANDIDATES = [f"http://{sys.argv[idx+1]}:8085"]

BASE = None
for candidate in PI_CANDIDATES:
    try:
        r = urllib.request.urlopen(candidate + "/health", timeout=3)
        d = json.loads(r.read())
        if d.get("version"):
            BASE = candidate
            print(f"✅ Pi found at {BASE}  v{d.get('version')}  xarm={d.get('xarm')}")
            break
    except Exception:
        pass

if not BASE:
    print("❌ Pi not reachable. Try: python test_arm_now.py --ip <pi-ip>")
    sys.exit(1)

PASS = FAIL = 0

def post(path, body=None, timeout=60):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
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
    sym = "✅" if cond else "❌"
    print(f"  {sym} [{label}]{(' — '+detail) if detail else ''}")
    if cond: PASS += 1
    else: FAIL += 1

print("\n" + "="*55)
print("xArm Direct Control Test")
print("="*55)

# ── 1. Home position ──────────────────────────────────────────
print("\n[1] HOME — all servos to 0°")
d, err = post("/agent/chat", {"message": "xarm home"})
check("home", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(2)

# ── 2. Ready position ─────────────────────────────────────────
print("\n[2] READY position")
d, err = post("/agent/chat", {"message": "ready"})
check("ready", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(2)

# ── 3. Wave ───────────────────────────────────────────────────
print("\n[3] WAVE")
d, err = post("/agent/chat", {"message": "wave"})
check("wave", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(2)

# ── 4. Inspect ────────────────────────────────────────────────
print("\n[4] INSPECT position")
d, err = post("/agent/chat", {"message": "inspect"})
check("inspect", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(2)

# ── 5. Open gripper ───────────────────────────────────────────
print("\n[5] OPEN gripper")
d, err = post("/agent/chat", {"message": "open gripper"})
check("open-gripper", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(1)

# ── 6. Close gripper ──────────────────────────────────────────
print("\n[6] CLOSE gripper")
d, err = post("/agent/chat", {"message": "close gripper"})
check("close-gripper", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(1)

# ── 7. Pick sequence ──────────────────────────────────────────
print("\n[7] PICK sequence (reach → lower → grab → lift)")
d, err = post("/agent/chat", {"message": "pick up"})
check("pick", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(3)

# ── 8. Home again ─────────────────────────────────────────────
print("\n[8] HOME (return)")
d, err = post("/agent/chat", {"message": "xarm home"})
check("home-return", d and d.get("ok"), d.get("response","") if d else err)
time.sleep(2)

print("\n" + "="*55)
print(f"Arm test: {PASS} PASS  {FAIL} FAIL")
print("="*55)

if FAIL > 0:
    print("\n⚠  Some moves failed. Check:")
    print("   - xArm USB cable connected to Pi")
    print("   - xArm powered on")
    print("   - Run: curl http://neurolinux.local:8085/health | grep xarm")
    sys.exit(1)

# ── 9. Cosmos reason → arm (full pipeline) ────────────────────
print("\n" + "="*55)
print("Cosmos Reason → Arm Action Pipeline")
print("="*55)

print("\n[9] Cosmos reason (Pi NIS :8000 → H100)...")
t0 = time.time()
NIS_BASE = BASE.replace(":8085", ":8000")
data = json.dumps({"task": "What do you see? What should the robot arm do next?"}).encode()
req = urllib.request.Request(NIS_BASE + "/cosmos/reason", data=data,
                             headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=90)
    d = json.loads(r.read())
    err = None
except urllib.error.HTTPError as e:
    d, err = None, f"HTTP {e.code}"
except Exception as e:
    d, err = None, str(e)[:80]
elapsed = time.time() - t0
if d:
    src = d.get("source","?")
    scene = str(d.get("scene", d.get("scene_description", d.get("response",""))))[:80]
    recs = d.get("action_recommendations", d.get("actions", []))
    check("cosmos-reason", True, f"source={src} {elapsed:.1f}s")
    if scene: print(f"     scene: {scene}")
    if recs:  print(f"     recs:  {recs[:3]}")

    # ── 10. Execute first recommended action ──────────────────
    if recs:
        print(f"\n[10] Executing arm action: '{recs[0]}'")
        d2, err2 = post("/agent/chat", {"message": recs[0]})
        check("cosmos-arm-action", d2 and d2.get("ok"),
              d2.get("response","") if d2 else err2)
else:
    check("cosmos-reason", False, f"{err} ({elapsed:.1f}s)")

print("\n" + "="*55)
print(f"Full pipeline: {PASS} PASS  {FAIL} FAIL")
if FAIL == 0:
    print("\n🎉 ARM IS MOVING + COSMOS PIPELINE WORKING!")
print("="*55)
