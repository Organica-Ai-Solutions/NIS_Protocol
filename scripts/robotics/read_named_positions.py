"""
read_named_positions.py — Read actual servo values for all named positions.
Moves arm to each position and reads hardware values.
Returns home after each move.
"""
import requests, json, time

AGENT = "http://192.168.1.163:8085"
POSITIONS_TO_READ = ["home", "inspect", "pick_table", "place_bin", "reach_forward",
                     "reach_left", "reach_right", "wave_up"]

def read_pos():
    r = requests.get(f"{AGENT}/arm/status", timeout=8)
    return r.json().get("positions", {})

def move_named(name):
    r = requests.post(f"{AGENT}/arm/named/{name}", json={}, timeout=15)
    return r.json().get("ok", False)

print("Reading all named position servo values...\n")
print(f"{'Position':<15} {'S1':>5} {'S2':>5} {'S3':>5} {'S4':>5} {'S5':>5} {'S6':>5}")
print("-" * 50)

results = {}

for name in POSITIONS_TO_READ:
    ok = move_named(name)
    time.sleep(2.2)
    p = read_pos()
    results[name] = p
    vals = [int(p.get(str(i), 0)) for i in range(1, 7)]
    print(f"{name:<15} {vals[0]:>5} {vals[1]:>5} {vals[2]:>5} {vals[3]:>5} {vals[4]:>5} {vals[5]:>5}  ok={ok}")
    # Return home after each
    move_named("home")
    time.sleep(1.5)

print("\n-- Raw JSON for tune_pick.py --")
for name, pos in results.items():
    vals_str = ", ".join(f"'{k}':{int(v)}" for k, v in sorted(pos.items()))
    print(f"{name.upper():15} = {{{vals_str}}}")

print("\nDone. Arm is at home.")
