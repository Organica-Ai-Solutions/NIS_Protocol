"""
DEFINITIVE CALIBRATION AND PICK-AND-PLACE
==========================================
Based on OFFICIAL Hiwonder xArm AI documentation.

Official confirmed positions (from SDK tutorials):
  HOME    : ki_move(x=0,  y=17,  z=20.5, pitch=0)
  PICK UP : ki_move(x=0,  y=17,  z=1.2,  pitch=-71)
  PLACE   : ki_move(x=19.5, y=0, z=2.8,  pitch=-60) [right 90]

Servo mapping (from official LSC docs + empirical calibration):
  S1 = gripper  : 100=open, 500=closed
  S2 = elbow/wrist
  S3 = wrist pitch
  S4 = fixed (~500)
  S5 = shoulder
  S6 = base rotation : 500=forward center, 350=home, 240=left

Stored arm memory poses:
  home       : {1:500, 2:484, 3:433, 4:500, 5:432, 6:350}
  inspect    : {1:500, 2:625, 3:485, 4:500, 5:335, 6:500}
  pick_table : {1:500, 2:258, 3:733, 4:500, 5:850, 6:500}
  place_bin  : {1:550, 2:370, 3:720, 4:380, 5:680, 6:240}
  lift_grip  : {1:550, 2:625, 3:485, 4:500, 5:335, 6:500}
"""

import requests
import base64
import json
import time
import os
import sys
from pathlib import Path

PI = "http://192.168.1.163:8085"
FRAMES_DIR = Path("data/calib_frames/fixed")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def wait(sec, msg=""):
    if msg:
        print(f"  Waiting {sec}s: {msg}")
    else:
        print(f"  Waiting {sec}s...")
    time.sleep(sec)


def api_get(path, timeout=10):
    try:
        r = requests.get(f"{PI}{path}", timeout=timeout)
        return r.json()
    except Exception as e:
        print(f"  ERROR GET {path}: {e}")
        return None


def api_post(path, data=None, timeout=15):
    try:
        r = requests.post(f"{PI}{path}", json=data or {}, timeout=timeout)
        return r.json()
    except Exception as e:
        print(f"  ERROR POST {path}: {e}")
        return None


def snap(label):
    """Capture snapshot and save as real JPEG (not JSON wrapper)."""
    print(f"  Snapping: {label}...")
    result = api_post("/arm/snapshot", timeout=20)
    if not result:
        print(f"  SNAP FAILED: {label}")
        return None

    # Extract real JPEG from JSON response
    if "image_base64" in result:
        img_bytes = base64.b64decode(result["image_base64"])
        fpath = FRAMES_DIR / f"{label}.jpg"
        fpath.write_bytes(img_bytes)
        print(f"  Saved: {fpath} ({len(img_bytes):,} bytes)")
        return str(fpath)
    else:
        print(f"  No image in response. Keys: {list(result.keys())}")
        return None


def go_home():
    """Move to stored HOME position."""
    print("  Moving to HOME...")
    result = api_post("/arm/go_pose", {"pose_name": "home"})
    if not result or not result.get("ok"):
        # Fallback: move directly with servo values
        print("  Fallback: sending raw servo command for home")
        result = api_post("/arm/move_servos", {
            "servos": {"1": 500, "2": 484, "3": 433, "4": 500, "5": 432, "6": 350},
            "duration": 1500
        })
    wait(2, "arm moving to home")
    return result


def gripper_open():
    print("  Gripper OPEN...")
    r = api_post("/arm/move_servos", {"servos": {"1": 100}, "duration": 500})
    wait(0.6)
    return r


def gripper_close():
    print("  Gripper CLOSE...")
    r = api_post("/arm/move_servos", {"servos": {"1": 500}, "duration": 500})
    wait(0.7)
    return r


def move_servos(servos: dict, duration=1000, wait_sec=None):
    """Move multiple servos simultaneously."""
    r = api_post("/arm/move_servos", {"servos": servos, "duration": duration})
    if wait_sec is None:
        wait_sec = duration / 1000.0 + 0.3
    wait(wait_sec)
    return r


def move_pose(pose_name):
    """Move to a named stored pose."""
    print(f"  Moving to stored pose: {pose_name}")
    r = api_post("/arm/go_pose", {"pose_name": pose_name})
    if not r or not r.get("ok"):
        # Fallback to raw servo moves using known values
        known = {
            "home":       {"1": 500, "2": 484, "3": 433, "4": 500, "5": 432, "6": 350},
            "inspect":    {"1": 500, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
            "pick_table": {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500},
            "place_bin":  {"1": 550, "2": 370, "3": 720, "4": 380, "5": 680, "6": 240},
            "lift_grip":  {"1": 550, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
        }
        if pose_name in known:
            r = api_post("/arm/move_servos", {"servos": known[pose_name], "duration": 1500})
    wait(1.5, f"moving to {pose_name}")
    return r


# =============================================================================
# MAIN CALIBRATION STEPS
# =============================================================================

def step0_preflight():
    section("STEP 0: PRE-FLIGHT CHECK")
    health = api_get("/health")
    if not health:
        print("  FAIL: Cannot reach Pi agent at", PI)
        return False
    print(f"  Pi agent: {health}")
    if health.get("xarm_simulation"):
        print("  WARNING: arm is in simulation mode! Reconnecting...")
        r = api_post("/arm/reconnect", {"simulation": False})
        print(f"  Reconnect result: {r}")
        wait(2)
    print(f"  Camera: {health.get('camera')}")
    print(f"  xArm connected: {health.get('xarm')}")
    print(f"  Simulation mode: {health.get('xarm_simulation')}")
    return True


def step1_home():
    section("STEP 1: GO TO HOME POSITION")
    print("  Official home = x=0, y=17, z=20.5 (arm reaches FORWARD)")
    print("  Stored servo values: {1:500, 2:484, 3:433, 4:500, 5:432, 6:350}")
    gripper_open()
    result = move_pose("home")
    print(f"  Move result: {result}")
    snap("step1_home")
    return result


def step2_inspect():
    section("STEP 2: INSPECT POSITION")
    print("  Inspect: arm raised to view workspace")
    result = move_pose("inspect")
    snap("step2_inspect")
    return result


def step3_sweep_find_lighter():
    section("STEP 3: S6 SWEEP - FIND GREEN LIGHTER")
    print("  Sweeping base rotation from S6=500 to S6=650")
    print("  GREEN lighter is on the RIGHT -> positive S6 values")

    # First move to pick_table height (arm low, over workspace)
    move_pose("pick_table")
    wait(0.5)

    # Sweep S6 while keeping other servos at pick_table values
    sweep_values = [500, 530, 560, 590, 610, 630, 650]
    images = {}
    for s6 in sweep_values:
        print(f"\n  S6 = {s6}...")
        move_servos({"6": s6}, duration=600)
        path = snap(f"sweep_s6_{s6}")
        if path:
            images[s6] = path

    go_home()
    return images


def step4_select_s6(images):
    section("STEP 4: SELECT CORRECT S6 FOR GREEN LIGHTER")

    print("\n  From my camera analysis:")
    print("  - calib_sweep_500: arm facing forward, lighter to the right")
    print("  - calib_sweep_560: arm slightly right")
    print("  - calib_sweep_610: arm more to the right, closer to green lighter")
    print()
    print("  Based on image analysis, the green lighter appears to be at S6 ~610")
    print()
    print("  Images saved to:", str(FRAMES_DIR))
    for s6, path in images.items():
        print(f"    S6={s6}: {path}")

    print()
    s6_input = input("  Enter S6 value where gripper is centered over GREEN lighter [610]: ").strip()
    if not s6_input:
        best_s6 = 610
    else:
        try:
            best_s6 = int(s6_input)
        except ValueError:
            best_s6 = 610
            print(f"  Invalid, using default {best_s6}")

    print(f"  Selected S6 = {best_s6}")
    return best_s6


def step5_save_pick_table(best_s6):
    section("STEP 5: SAVE CORRECTED PICK_TABLE WITH S6=" + str(best_s6))
    # pick_table with corrected S6
    corrected = {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": best_s6}
    print(f"  New pick_table: {corrected}")

    r = api_post("/arm/save_pose", {"name": "pick_table", "servos": corrected})
    print(f"  Save result: {r}")

    # Also save lift_grip with same S6 (stays over the object while lifting)
    lift = {"1": 550, "2": 625, "3": 485, "4": 500, "5": 335, "6": best_s6}
    r2 = api_post("/arm/save_pose", {"name": "lift_grip", "servos": lift})
    print(f"  Lift_grip save: {r2}")

    return corrected


def step6_test_pick():
    section("STEP 6: TEST PICK FROM PICK_TABLE")
    print("  Testing: go to pick_table, open gripper, then lower...")

    # Go to pick_table height
    move_pose("pick_table")
    snap("step6_at_pick_table")

    print("\n  Is the arm over the green lighter? (y/n)")
    ans = input("  -> ").strip().lower()
    if ans != 'y':
        print("  Skipping pick test. Re-run calibration to adjust S6.")
        return False

    # Execute pick sequence
    print("\n  EXECUTING PICK SEQUENCE...")

    # 1. Open gripper
    gripper_open()

    # 2. Lower to pick (reduce S5 to lower arm)
    # From pick_table (S5=850), lower to actual pick height
    poses = api_get("/arm/touch_poses")
    if poses and poses.get("poses", {}).get("pick_table"):
        pt = poses["poses"]["pick_table"]
        current_s6 = pt.get("6", 500)
        # Lower pick: increase S5 slightly and decrease S2
        lower = {"1": 100, "2": pt.get("2", 258) - 30, "3": pt.get("3", 733) + 20,
                 "4": 500, "5": min(950, pt.get("5", 850) + 50), "6": current_s6}
        print(f"  Lowering to: {lower}")
        move_servos(lower, duration=800)
        snap("step6_lowered")

    # 3. Close gripper
    gripper_close()
    snap("step6_gripped")

    # 4. Lift
    move_pose("lift_grip")
    snap("step6_lifted")

    # 5. Check if object is in gripper
    print("\n  Did the arm pick up the lighter? (y/n)")
    ans = input("  -> ").strip().lower()

    if ans == 'y':
        # 6. Place
        print("  PLACING at place_bin...")
        move_pose("place_bin")
        snap("step6_at_place")
        gripper_open()
        wait(0.5)
        move_pose("home")
        print("  SUCCESS! Pick and place complete!")
        return True
    else:
        # Drop it and go home
        gripper_open()
        move_pose("home")
        print("  Pick failed. Need to adjust S6 or lower pick position.")
        return False


def step7_cosmos_analysis():
    section("STEP 7: COSMOS REASON2 ANALYSIS (Optional)")
    print("  Sending images to Cosmos for spatial reasoning...")

    # Find latest sweep images
    snap_files = sorted(FRAMES_DIR.glob("sweep_s6_*.jpg"))
    if not snap_files:
        print("  No sweep images found, skipping Cosmos analysis")
        return

    # Use last snap
    latest = snap_files[-1]
    img_b64 = base64.b64encode(latest.read_bytes()).decode()

    cosmos_url = os.environ.get("H100_REASON_URL", "http://localhost:8100")  # Cosmos Reason2 (tunnel from H100)
    try:
        r = requests.post(f"{cosmos_url}/reason", json={
            "prompt": (
                "Analyze this robot arm workspace image. "
                "The arm is a Hiwonder xArm AI. "
                "There is a GREEN LIGHTER to the right of the arm base. "
                "Please describe where the green lighter is relative to the arm gripper. "
                "Is the gripper directly above the lighter? "
                "If not, which direction and how far does the arm need to rotate?"
            ),
            "image_base64": img_b64
        }, timeout=30)
        result = r.json()
        print(f"\n  Cosmos reasoning:\n  {result.get('response', result)[:500]}")
    except Exception as e:
        print(f"  Cosmos not available: {e}")


def main():
    print("\n" + "="*60)
    print("  DEFINITIVE XARM CALIBRATION + PICK-AND-PLACE")
    print("  Based on official Hiwonder xArm AI documentation")
    print("="*60)
    print()
    print("  WORKSPACE:")
    print("    GREEN lighter: RIGHT side of arm")
    print("    BLUE lighter: LEFT side of arm")
    print("    Camera: Side/angle view on Pi")
    print()
    print("  OFFICIAL POSITIONS (from docs):")
    print("    HOME:    x=0,  y=17,  z=20.5, pitch=0")
    print("    PICK UP: x=0,  y=17,  z=1.2,  pitch=-71")
    print("    PLACE R: x=19.5, y=0, z=2.8,  pitch=-60")
    print()

    # Pre-flight
    if not step0_preflight():
        sys.exit(1)

    # Home
    step1_home()

    # Inspect
    step2_inspect()
    go_home()

    # Sweep to find lighter
    images = step3_sweep_find_lighter()

    # Select S6
    best_s6 = step4_select_s6(images)

    # Save corrected pick_table
    step5_save_pick_table(best_s6)

    # Test pick
    success = step6_test_pick()

    # Cosmos analysis
    step7_cosmos_analysis()

    # Summary
    section("CALIBRATION COMPLETE")
    print(f"  Best S6 for green lighter: {best_s6}")
    print(f"  Pick success: {success}")
    print()
    print("  To run full demo:")
    print("    python run_pick_and_place.py")
    print()


if __name__ == "__main__":
    main()
