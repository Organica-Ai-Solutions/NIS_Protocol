"""
IK-Based Calibration & Home Fix
================================
Uses official Hiwonder IK equations to:
  1. Move arm to CORRECT home: ki_move(0, 17, 20.5, 0) -- gripper level, straight forward
  2. Find lighter X position via camera sweep (IK-based)
  3. Confirm and save calibrated pick pose

Usage:
  python ik_calibrate.py --home          # Set correct home position
  python ik_calibrate.py --sweep         # Sweep X to find lighter
  python ik_calibrate.py --demo          # Run full 4-step pick-and-place demo
  python ik_calibrate.py --verify        # Show IK for all pipeline poses
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.kinematics.hiwonder_ik import ik_to_servos, Pose, ki_move_http, ik_solve

PI_URL   = "http://192.168.1.163:8085"
COSMOS   = "http://192.168.1.100:8000"
CALIB_DIR = Path("data/calib_frames")
CALIB_DIR.mkdir(parents=True, exist_ok=True)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def pi_get(path, timeout=10):
    try:
        r = urllib.request.urlopen(PI_URL + path, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def pi_post(path, body=None, timeout=12):
    try:
        data = json.dumps(body or {}).encode()
        req  = urllib.request.Request(
            PI_URL + path, data=data,
            headers={"Content-Type": "application/json"}
        )
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def cosmos_post(path, body, timeout=30):
    try:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            COSMOS + path, data=data,
            headers={"Content-Type": "application/json"}
        )
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


# ── Movement ───────────────────────────────────────────────────────────────────

def ki_move(x, y, z, pitch, duration_ms=1200, gripper=None):
    """Move arm to XYZ position using IK, send via HTTP to Pi."""
    servos = ik_to_servos(x, y, z, pitch)
    if servos is None:
        print(f"  [IK] No solution for ({x},{y},{z},{pitch}deg)")
        return False
    if gripper is not None:
        servos["1"] = max(100, min(900, gripper))

    body = {"positions": servos, "duration_ms": duration_ms}
    r = pi_post("/arm/group_move", body)

    s_str = " ".join(f"S{k}={v}" for k, v in sorted(servos.items()))
    ok = "error" not in r
    status = "OK" if ok else f"FAIL: {r.get('error','?')}"
    print(f"  ki_move({x:+.1f},{y:.1f},{z:.1f},{pitch:+.0f}) -> {s_str}  [{status}]")
    return ok


def gripper_open(duration_ms=500):
    pi_post("/arm/group_move", {"positions": {"1": 100}, "duration_ms": duration_ms})
    print("  Gripper OPEN (S1=100)")


def gripper_close(duration_ms=500):
    pi_post("/arm/group_move", {"positions": {"1": 500}, "duration_ms": duration_ms})
    print("  Gripper CLOSE (S1=500)")


def snapshot(filename="snap.jpg"):
    """Capture a frame from Pi camera, save to data/calib_frames/."""
    path = str(CALIB_DIR / filename)
    try:
        r = urllib.request.urlopen(PI_URL + "/camera/snapshot", timeout=15)
        with open(path, "wb") as f:
            f.write(r.read())
        size = os.path.getsize(path)
        print(f"  Snapshot saved: {path} ({size} bytes)")
        return path
    except Exception as e:
        print(f"  Snapshot failed: {e}")
        return None


def wait(seconds, msg=""):
    if msg:
        print(f"  Waiting {seconds}s — {msg}")
    else:
        print(f"  Waiting {seconds}s...")
    time.sleep(seconds)


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_verify():
    """Show IK for all pipeline positions — verify the math is correct."""
    print("\n=== IK VERIFICATION (Hiwonder documentation reference) ===\n")
    print("Official ARM coordinate system:")
    print("  Origin: base servo center")
    print("  Y+: forward (standard pick is y=17cm)")
    print("  X+: right (lighter is approximately x=+6cm)")
    print("  Z+: up (home is z=20.5cm)")
    print()
    print("Reference:  ki_move(0, 17, 20.5, 0)    -> HOME  (gripper level, 17cm forward)")
    print("Reference:  ki_move(0, 17, 1.2,  -71)  -> PICK  (standard front pick)")
    print()

    poses = {
        "HOME          ": Pose.HOME,
        "INSPECT       ": Pose.INSPECT,
        "PICK_FRONT    ": Pose.PICK_FRONT,
        "PICK_LIGHTER  ": Pose.PICK_LIGHTER,
        "LIFT          ": Pose.LIFT,
        "PLACE_BIN     ": Pose.PLACE_BIN,
        "DROP_BIN      ": Pose.DROP_BIN,
        "PLACE_LEFT_90 ": Pose.PLACE_LEFT_90,
        "DROP_LEFT_90  ": Pose.DROP_LEFT_90,
    }

    print(f"{'Pose':<15} {'Coords':^30} {'Servo positions'}")
    print("-" * 80)
    for name, (x, y, z, p) in poses.items():
        servos = ik_to_servos(x, y, z, p)
        if servos:
            s = " ".join(f"S{k}={v}" for k, v in sorted(servos.items()))
            print(f"  {name} ({x:+5.1f},{y:4.1f},{z:4.1f},{p:+3.0f}deg)  {s}")
        else:
            print(f"  {name} ({x:+5.1f},{y:4.1f},{z:4.1f},{p:+3.0f}deg)  NO SOLUTION")


def cmd_home():
    """Move arm to CORRECT documented home: (0, 17, 20.5, 0)."""
    print("\n=== SETTING CORRECT HOME POSITION ===")
    print("Documented home: ki_move(0, 17, 20.5, 0)")
    print("  x=0: straight forward (not rotated)")
    print("  y=17cm: standard forward reach")
    print("  z=20.5cm: raised position")
    print("  pitch=0: gripper level (horizontal)")
    print()
    print("NOTE: Stored arm S6=350 was WRONG (arm rotated left).")
    print("      Correct S6=500 (facing straight forward).")
    print()

    # Check Pi connectivity
    health = pi_get("/health")
    if "error" in health:
        print(f"[FAIL] Pi not reachable: {health['error']}")
        return

    print("Moving to HOME in 2 seconds... (clear the workspace!)")
    wait(2, "moving to home")

    # First open gripper
    gripper_open(500)
    wait(0.6)

    # Move to home
    ok = ki_move(*Pose.HOME, duration_ms=1500)
    wait(1.5, "settling at home")

    if ok:
        snapshot("ik_home.jpg")
        print()
        print("[OK] Arm is at IK HOME (0, 17, 20.5, 0)")
        print("     Take a photo — gripper should be level, arm pointing straight forward")
        print("     If correct, this IS your real home position.")
        print()
        save = input("Save this as arm memory home? (y/n): ").strip().lower()
        if save == "y":
            servos = ik_to_servos(*Pose.HOME)
            pi_post("/arm/touch_poses", {"home": servos})
            print("[OK] Home saved to arm memory.")
    else:
        print("[FAIL] Could not move arm to home")


def cmd_sweep():
    """Sweep X position while capturing frames — find lighter's exact X."""
    print("\n=== X SWEEP TO FIND LIGHTER POSITION ===")
    print("The lighter is to the RIGHT of center.")
    print("We will sweep X from 0 to +12cm and capture frames.")
    print("Look at the images to find which X aligns the gripper with the lighter.")
    print()

    health = pi_get("/health")
    if "error" in health:
        print(f"[FAIL] Pi not reachable: {health['error']}")
        return

    # Start from home
    print("Going to home first...")
    gripper_open(500)
    wait(0.5)
    ki_move(*Pose.HOME, duration_ms=1500)
    wait(1.5)

    # Sweep inspect height at different X values
    x_values = [0, 2, 4, 6, 8, 10, 12]
    results = {}

    print(f"\nSweeping X from 0 to 12cm at y=17, z=15 (lower than home to see workspace):")
    for x in x_values:
        print(f"\n  X={x:+.0f}cm:")
        ok = ki_move(x, 17, 15, 0, duration_ms=1000)
        wait(1.2, "settling")
        if ok:
            fname = f"sweep_x{x:+.0f}.jpg"
            path = snapshot(fname)
            results[x] = path
            print(f"    Saved {fname}")

    # Return to home
    print("\nReturning to home...")
    ki_move(*Pose.HOME, duration_ms=1200)
    wait(1.5)

    print("\n=== SWEEP COMPLETE ===")
    print("Review these frames in data/calib_frames/:")
    for x, path in results.items():
        print(f"  X={x:+.0f}cm -> {path}")
    print()
    print("Find the frame where the lighter is centered under the gripper.")
    print("That X value is your pick offset.")

    try:
        x_pick = float(input("\nEnter the X value (cm) where lighter is centered: "))
    except ValueError:
        print("Invalid input, using x=6 (from S6 sweep estimate)")
        x_pick = 6.0

    servos = ik_to_servos(x_pick, 17, 1.2, -71)
    print(f"\nCorrected PICK position: ki_move({x_pick}, 17, 1.2, -71)")
    if servos:
        s = " ".join(f"S{k}={v}" for k, v in sorted(servos.items()))
        print(f"Servo positions: {s}")

    save = input("Save this as pick_table in arm memory? (y/n): ").strip().lower()
    if save == "y":
        pi_post("/arm/touch_poses", {"pick_table": servos})
        print(f"[OK] pick_table saved with x={x_pick}cm")

    return x_pick


def cmd_demo():
    """Run a full pick-and-place demo using IK coordinates."""
    print("\n=== IK PICK-AND-PLACE DEMO ===")
    print("This runs the full Cosmos cookoff pipeline using inverse kinematics.")
    print()
    print("Sequence:")
    print("  1. HOME     - ki_move(0, 17, 20.5, 0)    gripper open")
    print("  2. INSPECT  - ki_move(0, 17, 15.0, 0)    lower to inspect workspace")
    print("  3. PICK     - ki_move(+6, 17, 1.2, -71)  move over lighter (x=6cm right)")
    print("  4. GRIP     - close gripper")
    print("  5. LIFT     - ki_move(+6, 17, 20.5, 0)   lift with object")
    print("  6. SWING    - ki_move(-17, 0, 20.5, 0)   rotate to bin side")
    print("  7. DROP     - ki_move(-19.5, 0, 2.8, -60) lower to bin")
    print("  8. RELEASE  - open gripper")
    print("  9. HOME     - ki_move(0, 17, 20.5, 0)    return home")
    print()

    health = pi_get("/health")
    if "error" in health:
        print(f"[FAIL] Pi not reachable: {health['error']}")
        return

    input("Press ENTER to start demo (make sure lighter is at x=+6cm right of center)...")

    print("\nStep 1: HOME")
    gripper_open(500)
    ki_move(*Pose.HOME, duration_ms=1500)
    wait(1.5)

    print("\nStep 2: INSPECT (lower to see workspace)")
    ki_move(0, 17, 15.0, 0, duration_ms=1200)
    wait(1.2)
    snapshot("demo_inspect.jpg")

    print("\nStep 3: PICK LIGHTER (x=+6cm right)")
    ki_move(*Pose.PICK_LIGHTER, duration_ms=1200)
    wait(1.2)
    snapshot("demo_pick_approach.jpg")

    print("\nStep 4: GRIP")
    gripper_close(600)
    wait(0.8)
    snapshot("demo_grip.jpg")

    print("\nStep 5: LIFT")
    ki_move(*Pose.LIFT_LIGHTER, duration_ms=1000)
    wait(1.0)
    snapshot("demo_lift.jpg")

    print("\nStep 6: SWING to bin")
    ki_move(*Pose.PLACE_BIN, duration_ms=1200)
    wait(1.2)

    print("\nStep 7: DROP into bin")
    ki_move(*Pose.DROP_BIN, duration_ms=1200)
    wait(1.2)
    snapshot("demo_drop.jpg")

    print("\nStep 8: RELEASE")
    gripper_open(500)
    wait(0.8)

    print("\nStep 9: HOME")
    ki_move(*Pose.PLACE_BIN, duration_ms=800)
    wait(0.8)
    ki_move(*Pose.HOME, duration_ms=1500)
    wait(1.5)
    snapshot("demo_final_home.jpg")

    print("\n=== DEMO COMPLETE ===")
    print("Frames saved to data/calib_frames/")


def cmd_cosmos_calibrate():
    """Send inspection frame to Cosmos Reason2 for spatial analysis."""
    print("\n=== COSMOS REASON2 CALIBRATION ===")

    health = pi_get("/health")
    if "error" in health:
        print(f"[FAIL] Pi not reachable: {health['error']}")
        return

    # Go to inspect pose
    print("Moving to inspect position...")
    gripper_open(500)
    ki_move(*Pose.HOME, duration_ms=1500)
    wait(1.5)
    ki_move(0, 17, 15.0, 0, duration_ms=1200)
    wait(1.5)
    img_path = snapshot("cosmos_inspect.jpg")

    if not img_path:
        print("[FAIL] Could not capture frame")
        return

    # Try Cosmos
    print("Sending to Cosmos Reason2...")
    import base64
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = """You are analyzing a robot arm workspace for a pick-and-place task.

SETUP:
- The gripper is at position (x=0, y=17cm, z=15cm) — centered, 15cm above workspace
- A green LIGHTER is somewhere in the workspace, likely to the RIGHT of center
- Workspace labels: "Left", "Right", "Front Right", "Back" are visible

TASK: Find the lighter's position relative to the arm's coordinate origin:
  - X axis: positive = RIGHT, negative = LEFT (in cm)
  - Y axis: forward from arm base (standard is 17cm for pick)
  - Estimated X offset in cm (positive = right of center)

Answer with:
1. Where is the lighter? (describe position)
2. Estimated X offset in cm from center (e.g. "+6cm" or "-3cm")
3. Estimated Y distance (forward), typically 15-20cm
4. Recommended gripper X for picking: ki_move(X, 17, 1.2, -71)"""

    r = cosmos_post("/v1/reason", {
        "prompt": prompt,
        "image_base64": b64,
        "max_tokens": 400,
    })

    if "error" in r:
        print(f"Cosmos offline: {r['error']}")
        print("Review the frame manually: {img_path}")
    else:
        text = r.get("text", r.get("response", str(r)))
        print("\nCosmos Reason2 analysis:")
        print("-" * 50)
        print(text)
        print("-" * 50)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IK-based xArm calibration")
    parser.add_argument("--verify",  action="store_true", help="Show IK for all poses")
    parser.add_argument("--home",    action="store_true", help="Set correct home position")
    parser.add_argument("--sweep",   action="store_true", help="Sweep X to find lighter")
    parser.add_argument("--demo",    action="store_true", help="Run pick-and-place demo")
    parser.add_argument("--cosmos",  action="store_true", help="Cosmos Reason2 calibration")
    args = parser.parse_args()

    if args.verify:
        cmd_verify()
    elif args.home:
        cmd_home()
    elif args.sweep:
        cmd_sweep()
    elif args.demo:
        cmd_demo()
    elif args.cosmos:
        cmd_cosmos_calibrate()
    else:
        # Interactive menu
        print("\n=== IK CALIBRATION TOOL ===")
        print("Uses official Hiwonder inverse kinematics.")
        print()
        print("Key discovery:")
        print("  HOME = ki_move(0, 17, 20.5, 0)   [S6=500 facing forward]")
        print("  Stored home had S6=350 = arm rotated left = WRONG")
        print()
        print("1. verify  - Show IK for all pipeline poses")
        print("2. home    - Move arm to correct home position")
        print("3. sweep   - Sweep X to find lighter (then save pick pose)")
        print("4. demo    - Run full pick-and-place demo")
        print("5. cosmos  - Cosmos Reason2 analysis of workspace")
        print()
        choice = input("Choose (1-5): ").strip()
        if choice == "1":
            cmd_verify()
        elif choice == "2":
            cmd_home()
        elif choice == "3":
            cmd_sweep()
        elif choice == "4":
            cmd_demo()
        elif choice == "5":
            cmd_cosmos_calibrate()
        else:
            cmd_verify()


if __name__ == "__main__":
    main()
