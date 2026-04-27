"""
FULL PICK-AND-PLACE DEMO
========================
Official Hiwonder xArm AI positions confirmed from SDK documentation.

Picks GREEN lighter (RIGHT side) and places it in place_bin.

Run this AFTER calibrate_and_pick.py has found the correct S6.
Usage:
  python run_pick_and_place.py [--s6 610]
"""

import requests
import base64
import json
import time
import sys
import argparse
from pathlib import Path

PI = "http://192.168.1.163:8085"
OUT_DIR = Path("data/demo_frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def api_post(path, data=None, timeout=15):
    try:
        r = requests.post(f"{PI}{path}", json=data or {}, timeout=timeout)
        return r.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def api_get(path, timeout=10):
    try:
        r = requests.get(f"{PI}{path}", timeout=timeout)
        return r.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def snap(label):
    """Snapshot -> save real JPEG (extracts from JSON wrapper)."""
    result = api_post("/arm/snapshot", timeout=20)
    if result and "image_base64" in result:
        img_bytes = base64.b64decode(result["image_base64"])
        fpath = OUT_DIR / f"{label}.jpg"
        fpath.write_bytes(img_bytes)
        print(f"  [CAM] {fpath.name} ({len(img_bytes):,}b)")
        return str(fpath)
    return None


def move(servos: dict, ms=1200):
    """Move servos and wait."""
    api_post("/arm/move_servos", {"servos": servos, "duration": ms})
    time.sleep(ms / 1000.0 + 0.3)


def pose(name, ms=1500):
    """Go to named pose."""
    r = api_post("/arm/go_pose", {"pose_name": name})
    if not r or not r.get("ok"):
        # Hardcoded fallback values
        known = {
            "home":       {"1": 500, "2": 484, "3": 433, "4": 500, "5": 432, "6": 350},
            "inspect":    {"1": 500, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
            "lift_grip":  {"1": 550, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
            "place_bin":  {"1": 550, "2": 370, "3": 720, "4": 380, "5": 680, "6": 240},
        }
        if name in known:
            api_post("/arm/move_servos", {"servos": known[name], "duration": ms})
    time.sleep(ms / 1000.0 + 0.3)


def gripper_open():
    move({"1": 100}, ms=500)
    time.sleep(0.3)


def gripper_close():
    move({"1": 500}, ms=600)
    time.sleep(0.4)


def get_pick_pose(s6):
    """Get pick_table servos with the calibrated S6."""
    # Load from arm memory first
    poses = api_get("/arm/touch_poses")
    if poses and poses.get("poses", {}).get("pick_table"):
        pt = dict(poses["poses"]["pick_table"])
        pt["6"] = s6
        return pt
    # Fallback to hardcoded values
    return {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": s6}


def get_pick_low(pick_pose, s6):
    """Lower pick position (go slightly lower than pick_table to grab object)."""
    # From pick_table, go a bit lower to actually grab
    return {
        "1": 100,   # gripper open
        "2": max(200, int(pick_pose.get("2", 258)) - 30),
        "3": min(800, int(pick_pose.get("3", 733)) + 25),
        "4": 500,
        "5": min(950, int(pick_pose.get("5", 850)) + 40),
        "6": s6
    }


def run_demo(s6=610):
    print(f"\n{'='*60}")
    print(f"  PICK-AND-PLACE DEMO  |  S6 = {s6}")
    print(f"{'='*60}")

    # Check health
    h = api_get("/health")
    if not h:
        print("ERROR: Pi agent unreachable!")
        return False
    print(f"  Agent: {h.get('service')} | sim={h.get('xarm_simulation')} | cam={h.get('camera')}")

    if h.get("xarm_simulation"):
        print("  WARNING: Arm in simulation! Reconnecting...")
        api_post("/arm/reconnect", {"simulation": False})
        time.sleep(2)

    # Get pick pose with calibrated S6
    pick_pose = get_pick_pose(s6)
    pick_low = get_pick_low(pick_pose, s6)
    print(f"\n  Pick pose (S6={s6}): {pick_pose}")
    print(f"  Pick low:            {pick_low}")

    print("\n--- STEP 1: HOME + OPEN GRIPPER ---")
    gripper_open()
    pose("home")
    snap("01_home")

    print("\n--- STEP 2: INSPECT (view workspace) ---")
    pose("inspect")
    snap("02_inspect")

    print("\n--- STEP 3: APPROACH above lighter ---")
    move(pick_pose, ms=1200)
    snap("03_above_lighter")

    print("\n--- STEP 4: LOWER to grab lighter ---")
    move(pick_low, ms=800)
    snap("04_at_lighter")

    print("\n--- STEP 5: GRIP - close gripper ---")
    gripper_close()
    snap("05_gripped")

    print("\n--- STEP 6: LIFT with lighter ---")
    pose("lift_grip")
    snap("06_lifted")

    print("\n--- STEP 7: PLACE at place_bin ---")
    pose("place_bin")
    snap("07_at_place")

    print("\n--- STEP 8: RELEASE - open gripper ---")
    gripper_open()
    snap("08_released")
    time.sleep(0.3)

    print("\n--- STEP 9: RETURN HOME ---")
    pose("home")
    snap("09_home_return")

    print(f"\n{'='*60}")
    print("  PICK-AND-PLACE COMPLETE!")
    print(f"  Frames saved to: {OUT_DIR}")
    print(f"{'='*60}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s6", type=int, default=610,
                        help="S6 base rotation value for green lighter (from calibration)")
    args = parser.parse_args()
    run_demo(s6=args.s6)
