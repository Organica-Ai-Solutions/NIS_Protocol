"""
test_pick_sequence.py — Step-through test of pick_and_place pipeline.
Verifies each step with position readback, no full autonomous run.
Press Enter to advance between steps.
"""
import requests, json, time, sys

AGENT = "http://192.168.1.163:8085"
HOME = {1: 350, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500}

def pos():
    r = requests.get(f"{AGENT}/arm/status", timeout=8)
    p = r.json().get("positions", {})
    return "  ".join(f"S{k}:{int(p[str(k)]):3d}" for k in "123456")

def call(ep, data=None, timeout=20):
    r = requests.post(f"{AGENT}{ep}", json=data or {}, timeout=timeout)
    return r.json()

def pause(msg=""):
    print(f"\n[READY] {msg}")
    print("Press Enter to continue (or type 'q' to quit):", end=" ")
    ans = input().strip().lower()
    if ans == 'q':
        print("Quitting. Arm returning home...")
        call("/arm/home")
        sys.exit(0)

print("=== PICK & PLACE STEP TEST ===")
print("This will move the real arm! Make sure workspace is clear.\n")
print(f"Current position: {pos()}")

pause("Starting — arm will go HOME first")

# Step 1: Home
print("[1] HOME")
r = call("/arm/home", timeout=10)
time.sleep(2.5)
print(f"    pos: {pos()}  ok={r.get('ok')}")

pause("About to OPEN GRIPPER")

# Step 2: Open gripper
print("[2] OPEN GRIPPER")
r = call("/arm/gripper/open", timeout=8)
time.sleep(1.2)
print(f"    ok={r.get('ok')}")

pause("About to move to INSPECT position")

# Step 3: Inspect
print("[3] INSPECT")
r = call("/arm/named/inspect", json={}, timeout=12)
time.sleep(2.5)
print(f"    pos: {pos()}  ok={r.get('ok')}")
print("    ^ Camera view should show workspace from above")

pause("About to move to PICK_TABLE position (arm will descend toward table)")

# Step 4: Pick table
print("[4] PICK_TABLE")
r = call("/arm/named/pick_table", json={}, timeout=12)
time.sleep(3.0)
print(f"    pos: {pos()}  ok={r.get('ok')}")
print("    ^ IS GRIPPER AT RIGHT HEIGHT TO GRAB OBJECT?")
print("      If too high: run 'python tune_pick.py --s2 -20 --test'")
print("      If too low:  run 'python tune_pick.py --s2 +20 --test'")

pause("About to CLOSE GRIPPER (grasping)")

# Step 5: Close gripper
print("[5] CLOSE GRIPPER")
r = call("/arm/gripper/close", timeout=8)
time.sleep(1.5)
print(f"    pos after grip: {pos()}")
print("    ^ S6 change indicates gripper state")

pause("About to LIFT back to INSPECT (with object)")

# Step 6: Lift to inspect
print("[6] LIFT TO INSPECT")
r = call("/arm/named/inspect", json={}, timeout=12)
time.sleep(2.5)
print(f"    pos: {pos()}")

pause("About to move to PLACE_BIN position")

# Step 7: Place bin
print("[7] PLACE_BIN")
r = call("/arm/named/place_bin", json={}, timeout=12)
time.sleep(2.5)
print(f"    pos: {pos()}")

pause("About to OPEN GRIPPER (releasing object into bin)")

# Step 8: Release
print("[8] OPEN GRIPPER")
r = call("/arm/gripper/open", timeout=8)
time.sleep(1.0)
print(f"    ok={r.get('ok')}")

pause("About to return HOME")

# Step 9: Home
print("[9] HOME")
r = call("/arm/home", timeout=10)
time.sleep(2.5)
print(f"    final pos: {pos()}")

print("\n=== SEQUENCE COMPLETE ===")
print("If the arm grabbed the object and placed it in the bin, calibration is good!")
print("If not, adjust with: python tune_pick.py --s2 <delta> --test")
