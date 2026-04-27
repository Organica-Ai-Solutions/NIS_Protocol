"""Quick demo simulation test to verify the full pipeline."""
import requests, json, time

PI = "http://192.168.1.163:8000"
print("=== NIS Protocol Demo Pipeline Test ===\n")

# 1. Health check
r = requests.get(f"{PI}/health", timeout=8)
d = r.json()
status = d.get("status")
version = d.get("version")
routes = d.get("routes_loaded")
print(f"Health: {status} | v={version} | routes={routes}")

# 2. Check Pi agent connectivity
r2 = requests.get("http://192.168.1.163:8085/arm/status", timeout=8)
d2 = r2.json()
print(f"Agent: connected={d2.get('connected')} sim={d2.get('simulation')} pos={d2.get('positions')}")

# 3. Simulation demo - verify full sequence logic
print("\nRunning simulation demo (no arm movement)...")
t0 = time.time()
r3 = requests.post(
    f"{PI}/demo/run",
    json={"task": "pick up the red cube and place it in the bin", "simulation": True},
    timeout=120
)
elapsed_ms = round((time.time() - t0) * 1000)
d3 = r3.json()

print(f"  ok={d3.get('ok')} | plan_source={d3.get('plan_source')} | {elapsed_ms}ms")
print(f"  actions: {d3.get('action_plan')}")
print(f"  steps: {d3.get('steps_ok')}/{d3.get('steps_total')}")
print(f"  goal_complete: {d3.get('goal_complete')}")
print(f"  camera_used: {d3.get('camera_used')}")
print(f"  avg_plausibility: {d3.get('avg_plausibility')}")
print(f"  pipeline: {d3.get('pipeline')}")

exec_results = d3.get("execution", {}).get("results", [])
print("\nStep trace:")
for s in exec_results:
    print(f"  {s.get('step')}. {s.get('action'):20s} ok={s.get('ok')} src={s.get('source')}")

if d3.get("ok") and d3.get("steps_ok", 0) >= 3:
    print("\nSIMULATION PASSED — arm sequence verified!")
    print("Ready to run live: python run_demo_now.py")
else:
    print(f"\nISSUE: only {d3.get('steps_ok')} steps completed")
    print("Raw response:", json.dumps(d3, indent=2)[:500])
