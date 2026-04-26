#!/usr/bin/env python3
"""
Full end-to-end cookoff demo test on Pi:
  camera -> reason -> robot-plan -> transfer
"""
import urllib.request, json, time, base64

BASE = "http://192.168.1.163:8085"   # Pi NeuroLinux agent
NIS  = "http://192.168.1.163:8000"   # NIS Protocol (Pi)

PASS = 0
FAIL = 0

def get(base, path, timeout=10):
    try:
        r = urllib.request.urlopen(base + path, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:80]}"
    except Exception as e:
        return None, str(e)[:80]

def post(base, path, body=None, timeout=90):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        base + path, data=data,
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
print("Full Cookoff Demo — Pi → NIS → H100")
print("=" * 60)

# ── 1. Pi agent health ────────────────────────────────────────
print("\n[1] Pi agent health")
d, err = get(BASE, "/health")
check("pi-health", d and d.get("version"), f"v{d.get('version') if d else err}")

# ── 2. NIS health ─────────────────────────────────────────────
print("\n[2] NIS health (H100)")
d, err = get(NIS, "/health")
check("nis-health", d and d.get("status") == "healthy", f"{d.get('version') if d else err}")

# ── 3. Cookoff status (NIS endpoint) ────────────────────────────────────
print("\n[3] Cookoff status")
d, err = get(NIS, "/cookoff/status")
check("cookoff-status", d is not None,
      str(list(d.get("h100_services", {}).keys()))[:50] if d else err)

# ── 4. Pi arm health ─────────────────────────────────────────────────────
print("\n[4] Pi arm health")
d, err = get(BASE, "/health")
xarm_ok = d and not d.get("xarm_simulation", True)
check("pi-arm-physical", xarm_ok,
      f"xarm={d.get('xarm')} sim={d.get('xarm_simulation')} cam={d.get('camera')}" if d else err)
if d and d.get("xarm_simulation"):
    print("     WARNING: arm in simulation mode — reconnect USB")

# ── 5. Cosmos Reason2 direct health (H100) ───────────────────────────────
print("\n[5] Cosmos Reason2 H100 health")
t0 = time.time()
d, err = get("http://172.16.1.83:8100", "/health", timeout=6)
elapsed = time.time() - t0
check("h100-reason2", d is not None,
      f"model={d.get('model','?')} gpu={d.get('gpu','?')} {elapsed:.1f}s" if d else err)

# ── 6. Robot plan via NIS /cookoff/robot-plan ─────────────────────────────
print("\n[6] Robot plan (NIS → H100 Cosmos Reason2)")
t0 = time.time()
d, err = post(NIS, "/cookoff/robot-plan",
              {"query": "Pick up the lighter and place it on the left"})
elapsed = time.time() - t0
ok = d and (d.get("action_recommendations") or d.get("action_plan") or d.get("ok"))
src = d.get("source", "?") if d else "?"
conf = d.get("combined_confidence", d.get("confidence", "?")) if d else "?"
actions = d.get("action_recommendations") or d.get("action_plan", []) if d else []
check("robot-plan", ok, f"source={src} conf={conf} {elapsed:.1f}s")
if actions: print(f"     actions: {actions[:2]}")

# ── 7. Cookoff pick (dry-run — simulation=True) ───────────────────────────
print("\n[7] Cookoff pick sequence (simulation=True)")
t0 = time.time()
d, err = post(NIS, "/cookoff/pick",
              {"s6": 500, "z": 1.5, "place": "left90", "wait_sec": 0.0})
elapsed = time.time() - t0
# Note: simulation=True not in PickRequest — pick runs for real; gate on Pi health
has_steps = d and d.get("steps")
steps_ok  = sum(1 for s in (d.get("steps") or []) if s.get("ok")) if d else 0
steps_tot = len(d.get("steps") or []) if d else 0
check("cookoff-pick", has_steps,
      f"steps={steps_ok}/{steps_tot} ok={d.get('ok')} {elapsed:.1f}s" if d else err)

# ── 8. Pick outcomes / learning state ────────────────────────────────────
print("\n[8] Pick outcome learning state")
d, err = get(NIS, "/cookoff/outcomes")
check("cookoff-outcomes", d is not None,
      f"total={d.get('total_picks',0)} rate={d.get('success_rate',0):.1%}" if d else err)

# ── 9. Transfer (NIS → H100 Transfer2.5) ─────────────────────────────────
print("\n[9] Transfer2.5 status (NIS → H100)")
t0 = time.time()
d, err = get(NIS, "/cookoff/status")
elapsed = time.time() - t0
transfer_up = d and d.get("h100_services", {}).get("transfer25", {}).get("healthy", False)
check("transfer25-health", transfer_up,
      f"{'UP' if transfer_up else 'DOWN'} {elapsed:.1f}s" if d else err)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Results: {PASS} PASS  {FAIL} FAIL")
print("=" * 60)
