"""
PICK THE GREEN LIGHTER - COMPLETE PIPELINE
==========================================
Uses CORRECT Pi agent API endpoints discovered from /openapi.json

REAL API (confirmed working):
  GET  /health                    - check status
  GET  /camera/snapshot           - get photo
  GET  /vision/detect             - detect objects
  GET  /vision/scene              - scene context  
  GET  /calibration/map           - pixel->pose calibration
  GET  /arm/status                - current servo positions
  GET  /arm/touch_poses           - user-saved poses
  POST /arm/home                  - go to home
  POST /arm/named/{name}          - go to named pose
  POST /arm/group_move            - move specific servos
  POST /arm/gripper/open          - open gripper
  POST /arm/gripper/close         - close gripper
  POST /arm/pick_and_place        - full pick sequence

Workspace:
  GREEN lighter = RIGHT side of arm (positive X in arm coords)
  BLUE lighter  = LEFT side of arm
  HOME servo values: {1:500, 2:484, 3:433, 4:500, 5:432, 6:350}
  pick_table:         {1:500, 2:258, 3:733, 4:500, 5:850, 6:500}
"""

import requests
import base64
import time
import sys
from pathlib import Path

PI = "http://192.168.1.163:8085"
OUT = Path("data/pick_frames")
OUT.mkdir(parents=True, exist_ok=True)


def get(path, timeout=10):
    r = requests.get(f"{PI}{path}", timeout=timeout)
    if r.status_code == 200:
        return r.json()
    print("  GET " + path + " -> " + str(r.status_code) + " " + r.text[:100])
    return None


def post(path, data=None, timeout=15):
    r = requests.post(f"{PI}{path}", json=data or {}, timeout=timeout)
    if r.status_code == 200:
        return r.json()
    print("  POST " + path + " -> " + str(r.status_code) + " " + r.text[:100])
    return None


def snap(label, timeout=25):
    """Take snapshot and save as real JPEG."""
    d = get("/camera/snapshot", timeout=timeout)
    if d and "image_base64" in d:
        img = base64.b64decode(d["image_base64"])
        p = OUT / (label + ".jpg")
        p.write_bytes(img)
        print("  [CAM] " + p.name + " (" + str(len(img)) + "b)")
        return str(p)
    print("  [CAM] FAILED: " + label)
    return None


def wait(sec, msg=""):
    if msg:
        print("  Wait " + str(sec) + "s: " + msg)
    time.sleep(sec)


def group_move(servos, ms=1200):
    """Move specific servos using correct API."""
    str_servos = {str(k): v for k, v in servos.items()}
    r = post("/arm/group_move", {"positions": str_servos, "duration_ms": ms})
    wait(ms / 1000.0 + 0.3)
    return r


def go_named(name, timeout=15):
    """Move to named position."""
    r = post("/arm/named/" + name, {}, timeout=timeout)
    wait(1.8, "-> " + name)
    return r


def gripper_open():
    r = post("/arm/gripper/open", {})
    wait(0.6)
    return r


def gripper_close():
    r = post("/arm/gripper/close", {})
    wait(0.7)
    return r


def get_poses():
    """Get all user-saved touch poses."""
    d = get("/arm/touch_poses")
    if d and d.get("ok"):
        return d.get("poses", {})
    return {}


def get_status():
    """Get current servo positions."""
    d = get("/arm/status")
    if d:
        return d.get("positions", {})
    return {}


# =============================================================================
# STEP 1: CHECK EVERYTHING IS READY
# =============================================================================

def preflight():
    print("\n[1] PRE-FLIGHT CHECK")
    h = get("/health")
    if not h:
        print("  FAIL: Pi agent unreachable!")
        return False

    print("  Agent: " + h.get("service", "?") + " v" + h.get("version", "?"))
    print("  xArm: connected=" + str(h.get("xarm")) + " sim=" + str(h.get("xarm_simulation")))
    print("  Camera: " + str(h.get("camera")))

    if h.get("xarm_simulation"):
        print("  WARNING: ARM IN SIMULATION! Reconnecting to real hardware...")
        post("/arm/reconnect", {"simulation": False})
        wait(2)
        h2 = get("/health")
        if h2 and h2.get("xarm_simulation"):
            print("  STILL IN SIMULATION. Check USB connection!")
            return False
        print("  OK: Now in hardware mode")

    return True


# =============================================================================
# STEP 2: HOME
# =============================================================================

def go_home():
    print("\n[2] GOING HOME")
    gripper_open()
    r = post("/arm/home", {})
    wait(2.0, "arm moving to home")
    snap("home_start")
    status = get_status()
    print("  Servos at home: " + str(status))
    return r


# =============================================================================
# STEP 3: FIND GREEN LIGHTER WITH S6 SWEEP
# =============================================================================

def sweep_find_lighter():
    print("\n[3] S6 SWEEP - FINDING GREEN LIGHTER (RIGHT SIDE)")
    
    # First go to inspect
    go_named("inspect")
    snap("sweep_inspect")
    
    # Go to pick_table height (S6=500 = forward center)
    # We'll sweep S6 to rotate toward the right lighter
    poses = get_poses()
    pick = poses.get("pick_table", {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500})
    
    print("  pick_table base: " + str(pick))
    print("  Sweeping S6 from 500 to 660 (rotating right toward green lighter)...")
    print()
    print("  NOTE: TURN ON A LIGHT so the camera can see the lighters!")
    print()
    
    # Move to pick_table height
    group_move({"1": 100, "2": int(pick.get("2", 258)), "3": int(pick.get("3", 733)),
                "4": int(pick.get("4", 500)), "5": int(pick.get("5", 850)), "6": 500}, ms=1500)
    snap("sweep_center_s6_500")

    # Sweep
    best_frames = {}
    for s6 in [530, 560, 590, 610, 630, 650, 670]:
        print("  S6=" + str(s6) + "...")
        group_move({"6": s6}, ms=500)
        path = snap("sweep_s6_" + str(s6))
        best_frames[s6] = path
        wait(0.3)

    # Return to home
    go_named("home")
    return best_frames


# =============================================================================
# STEP 4: SELECT S6 FROM IMAGES
# =============================================================================

def select_s6(frames):
    print("\n[4] SELECT S6 FOR GREEN LIGHTER")
    print("  Open these images and find where the gripper is above the GREEN lighter:")
    for s6, path in frames.items():
        if path:
            print("    S6=" + str(s6) + ": " + path)
    print()
    print("  From previous analysis: S6~610 appears closest to green lighter")
    print()
    ans = input("  Enter S6 value (press Enter for 610): ").strip()
    if ans and ans.isdigit():
        return int(ans)
    return 610


# =============================================================================
# STEP 5: EXECUTE PICK AND PLACE
# =============================================================================

def pick_and_place(s6):
    print("\n[5] PICK AND PLACE - S6=" + str(s6))
    
    poses = get_poses()
    pick = poses.get("pick_table", {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500})
    lift = poses.get("lift_grip", {"1": 550, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500})
    place = poses.get("place_bin", {"1": 550, "2": 370, "3": 720, "4": 380, "5": 680, "6": 240})
    
    # Override S6 with calibrated value
    pick_s6 = dict(pick)
    pick_s6["6"] = s6
    pick_s6["1"] = 100  # gripper open
    lift_s6 = dict(lift)
    lift_s6["6"] = s6

    print("\n  Servo values:")
    print("    approach: " + str(pick_s6))
    print("    lift:     " + str(lift_s6))
    print("    place:    " + str(place))
    print()

    # --- Step A: Home + open gripper ---
    print("  A) Going to HOME...")
    post("/arm/home", {})
    gripper_open()
    wait(2.0)
    snap("pick_A_home")

    # --- Step B: Inspect position ---
    print("  B) Inspect position...")
    go_named("inspect")
    snap("pick_B_inspect")

    # --- Step C: Approach pick position (high) ---
    print("  C) Moving above lighter (pick_table height with S6=" + str(s6) + ")...")
    group_move(pick_s6, ms=1500)
    snap("pick_C_above")

    # --- Step D: Lower to pick ---
    print("  D) Lowering to grab lighter...")
    # Lower the arm slightly for actual contact
    lower = dict(pick_s6)
    lower["2"] = max(200, int(pick_s6.get("2", 258)) - 25)
    lower["3"] = min(800, int(pick_s6.get("3", 733)) + 20)
    lower["5"] = min(980, int(pick_s6.get("5", 850)) + 50)
    group_move(lower, ms=700)
    snap("pick_D_lowered")

    # --- Step E: Close gripper ---
    print("  E) Closing gripper...")
    gripper_close()
    wait(0.5)
    snap("pick_E_gripped")

    # --- Step F: Lift ---
    print("  F) Lifting with lighter...")
    group_move(lift_s6, ms=1000)
    snap("pick_F_lifted")

    # --- Step G: Verify lift ---
    print()
    ans = input("  Did the arm pick up the lighter? (y/n): ").strip().lower()
    if ans != "y":
        print("  Pick failed. Opening gripper and returning home.")
        gripper_open()
        post("/arm/home", {})
        wait(2)
        return False

    # --- Step H: Move to place ---
    print("  G) Moving to place_bin...")
    group_move(place, ms=1500)
    snap("pick_G_at_place")

    # --- Step I: Release ---
    print("  H) Releasing lighter...")
    gripper_open()
    wait(0.5)
    snap("pick_H_released")

    # --- Step J: Return home ---
    print("  I) Returning home...")
    post("/arm/home", {})
    wait(2.5)
    snap("pick_I_done")

    print("\n  *** PICK AND PLACE COMPLETE! ***")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("  PICK THE GREEN LIGHTER - CALIBRATION + DEMO")
    print("=" * 60)
    print()
    print("  Workspace setup:")
    print("    GREEN lighter: RIGHT side of arm base")
    print("    BLUE lighter:  LEFT side of arm base")
    print("    S6=500: arm faces forward (center)")
    print("    S6=600+: arm rotates RIGHT (toward green lighter)")
    print()
    print("  Official home position (from Hiwonder docs):")
    print("    x=0, y=17cm, z=20.5cm (arm reaching FORWARD)")
    print()

    # Run steps
    if not preflight():
        sys.exit(1)

    go_home()

    # Option 1: Skip sweep and use known S6
    ans = input("\n  Skip sweep and use S6=610 directly? (y/n): ").strip().lower()
    if ans == "y":
        best_s6 = 610
    else:
        frames = sweep_find_lighter()
        best_s6 = select_s6(frames)

    # Execute pick
    success = pick_and_place(best_s6)

    print()
    print("=" * 60)
    print("  RESULT: " + ("SUCCESS!" if success else "NEEDS ADJUSTMENT"))
    print("  Calibrated S6: " + str(best_s6))
    print("  Frames saved to: " + str(OUT))
    print("=" * 60)


if __name__ == "__main__":
    main()
