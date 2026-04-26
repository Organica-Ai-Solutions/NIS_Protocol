"""
live_demo_monitor.py — Run the live demo and print detailed step-by-step feedback.
Shows arm positions after each step so you can verify gripping happens.
"""
import requests, json, time, threading

AGENT = "http://192.168.1.163:8085"
NIS   = "http://192.168.1.163:8000"

def read_pos():
    try:
        r = requests.get(f"{AGENT}/arm/status", timeout=4)
        p = r.json().get("positions", {})
        return "  ".join(f"S{k}:{int(p.get(str(k), 0)):3d}" for k in "123456")
    except:
        return "?"

# Poll arm position in background during demo
_pos_log = []
_polling = True

def poll_positions():
    while _polling:
        p = read_pos()
        ts = round(time.time(), 1)
        _pos_log.append((ts, p))
        time.sleep(0.8)

print("=== LIVE DEMO MONITOR ===")
print("This calls the full NIS Protocol /demo/run with physical arm execution.\n")

# Check connectivity
print("Pre-flight checks...")
r = requests.get(f"{NIS}/health", timeout=8)
d = r.json()
print(f"  NIS Protocol: {d.get('status')} v={d.get('version')}")
r2 = requests.get(f"{AGENT}/arm/status", timeout=8)
d2 = r2.json()
print(f"  Agent:        connected={d2.get('connected')} sim={d2.get('simulation')}")
print(f"  Arm at:       {read_pos()}")

print("\nSending arm home first...")
requests.post(f"{AGENT}/arm/home", json={}, timeout=10)
time.sleep(2.5)
print(f"  Home:         {read_pos()}")

print("\nStarting position poller...")
t = threading.Thread(target=poll_positions, daemon=True)
t.start()

print("\nLaunching LIVE demo (execute_arm=True, simulation=False)...")
print("Watch the arm!\n")
t0 = time.time()

try:
    r = requests.post(
        f"{NIS}/demo/run",
        json={
            "task": "pick up the object on the table and place it in the bin",
            "simulation": False,
            "execute_arm": True,
        },
        timeout=120,
    )
    elapsed = round(time.time() - t0, 1)
    d = r.json()
except Exception as e:
    _polling = False
    print(f"ERROR: {e}")
    raise

_polling = False
time.sleep(0.5)

print(f"\n{'='*50}")
print(f"Demo complete in {elapsed}s")
print(f"ok={d.get('ok')} | plan_source={d.get('plan_source')} | steps={d.get('steps_ok')}/{d.get('steps_total')}")
print(f"goal_complete={d.get('goal_complete')} | camera={d.get('camera_used')}")
print(f"\nActions: {d.get('action_plan')}")
if d.get('reasoning'):
    print(f"H100 reasoning: {d.get('reasoning')[:200]}")
print(f"\nExecution steps:")
for s in d.get("execution", {}).get("results", []):
    print(f"  {s['step']}. {s['action']:20s} ok={s.get('ok')} ep={s.get('endpoint','?')} {s.get('latency_ms',0)}ms")

print(f"\nArm position log ({len(_pos_log)} samples):")
if _pos_log:
    prev = None
    for ts, pos in _pos_log:
        if pos != prev:
            print(f"  {ts:.1f}s  {pos}")
            prev = pos

print(f"\nFinal arm position: {read_pos()}")
