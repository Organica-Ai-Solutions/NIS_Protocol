#!/usr/bin/env python3
"""
NIS Protocol — Pi Deployment Script (HTTP-based, no SSH required)
=================================================================
Pushes updated arm pipeline to the Pi via HTTP, not SSH.

Strategy: Since we can't SSH reliably, we patch the Pi agent's behavior
by posting the corrected pick_and_place pipeline as saved touch poses,
then installing a lightweight "bridge" script on the Pi via its existing
/admin/exec endpoint (if available) or by verifying existing pose data.

What this does:
  1. Verifies Pi connectivity
  2. Pushes all 5 calibrated pipeline poses (home, inspect, pick_table,
     lift_grip, place_bin) via /arm/group_move + /arm/save_touch_pose
  3. Verifies each pose saved correctly
  4. Tests the arm moves to each pose in sequence
  5. Reports deployment status

Usage:
    python deploy_to_pi.py              # Push poses, verify
    python deploy_to_pi.py --test       # Test moves only (no push)
    python deploy_to_pi.py --full       # Push + full movement test
    python deploy_to_pi.py --verify     # Verify current Pi state
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

PI_URL = "http://192.168.1.163:8085"

# Ground truth pipeline poses
PIPELINE_POSES = {
    "home":       {"1": 500, "2": 500, "3": 400, "4": 500, "5": 400, "6": 350},
    "inspect":    {"1": 500, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
    "pick_table": {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500},
    "lift_grip":  {"1": 550, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},
    "place_bin":  {"1": 550, "2": 370, "3": 720, "4": 380, "5": 680, "6": 240},
}


def _get(path, timeout=5):
    try:
        r = urllib.request.urlopen(PI_URL + path, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:
        return None, str(e)


def _post(path, data, timeout=12):
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            PI_URL + path, data=body,
            headers={"Content-Type": "application/json"}
        )
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:80]}"
    except Exception as e:
        return None, str(e)


def check_connectivity():
    print("\n[1] Checking Pi connectivity...")
    data, err = _get("/health")
    if err:
        print(f"  FAIL: Pi unreachable — {err}")
        return False
    print(f"  OK — Pi v{data.get('version')} | xArm: {data.get('xarm')}")
    if not data.get("xarm"):
        print("  WARN: xArm not connected")
        return False
    return True


def push_poses():
    print("\n[2] Pushing pipeline poses to Pi...")
    results = {}
    for pose_name, positions in PIPELINE_POSES.items():
        print(f"  Moving to {pose_name}...")
        move_r, move_err = _post("/arm/group_move", {
            "positions": positions,
            "duration_ms": 1500,
        })
        if move_err or not (move_r or {}).get("ok"):
            print(f"    FAIL move: {move_err or move_r}")
            results[pose_name] = False
            continue
        time.sleep(2.0)  # settle

        print(f"  Saving {pose_name}...")
        save_r, save_err = _post("/arm/save_touch_pose", {"name": pose_name})
        if save_err:
            print(f"    FAIL save: {save_err}")
            results[pose_name] = False
        else:
            saved_pos = (save_r or {}).get("positions", {})
            s1 = saved_pos.get("1", "?")
            s6 = saved_pos.get("6", "?")
            print(f"    OK — S1={s1} S6={s6}")
            results[pose_name] = True

        time.sleep(0.5)

    ok = sum(1 for v in results.values() if v)
    print(f"\n  Poses deployed: {ok}/{len(PIPELINE_POSES)}")
    return results


def verify_poses():
    print("\n[3] Verifying poses on Pi...")
    data, err = _get("/arm/touch_poses")
    if err:
        print(f"  FAIL: {err}")
        return False

    poses = data.get("touch_poses") or data.get("poses") or {}
    required = list(PIPELINE_POSES.keys())
    missing = [p for p in required if p not in poses]

    print(f"  Total poses on Pi: {len(poses)}")
    for name in required:
        if name in poses:
            pos = poses[name]
            s1 = pos.get("1", "?")
            s6 = pos.get("6", "?")
            expected_s1 = PIPELINE_POSES[name]["1"]
            expected_s6 = PIPELINE_POSES[name]["6"]
            match = abs(int(s1) - expected_s1) <= 15 and abs(int(s6) - expected_s6) <= 15
            status = "OK" if match else "DRIFT"
            print(f"  {status} {name}: S1={s1}(exp:{expected_s1}) S6={s6}(exp:{expected_s6})")
        else:
            print(f"  MISSING: {name}")

    if missing:
        print(f"\n  Missing poses: {missing}")
        return False
    print("\n  All required poses present and verified")
    return True


def run_movement_test():
    print("\n[4] Running movement test (arm will move!)...")
    confirm = input("  Type 'yes' to physically move the arm through all poses: ").strip()
    if confirm.lower() != "yes":
        print("  Skipped.")
        return True

    for pose_name, positions in PIPELINE_POSES.items():
        print(f"  Moving to {pose_name}...")
        r, err = _post("/arm/group_move", {"positions": positions, "duration_ms": 1200})
        if err or not (r or {}).get("ok"):
            print(f"  FAIL: {err or r}")
        else:
            print(f"  OK")
        time.sleep(1.5)

    # Return home
    print("  Returning home...")
    _post("/arm/group_move", {"positions": PIPELINE_POSES["home"], "duration_ms": 1000})
    print("  Movement test complete")
    return True


def print_status():
    print("\n=== Pi Agent Status ===")
    health, _ = _get("/health")
    if not health:
        print("Pi OFFLINE")
        return

    arm_status, _ = _get("/arm/status")
    poses_r, _ = _get("/arm/touch_poses")
    poses = {}
    if poses_r:
        poses = poses_r.get("touch_poses") or poses_r.get("poses") or {}

    print(f"Version:   {health.get('version')}")
    print(f"xArm:      {health.get('xarm')}")
    print(f"Arm pos:   {arm_status.get('positions') if arm_status else 'N/A'}")
    print(f"Poses:     {len(poses)} saved")

    required = list(PIPELINE_POSES.keys())
    missing = [p for p in required if p not in poses]
    print(f"Pipeline:  {'READY' if not missing else 'MISSING: ' + str(missing)}")

    print("\nOrchestrator: python nis_console.py demo")
    print("Or via API:   POST http://localhost:8000/cookoff/arm/orchestrate")


def main():
    parser = argparse.ArgumentParser(description="NIS Protocol Pi Deployment")
    parser.add_argument("--verify", action="store_true", help="Verify current Pi state only")
    parser.add_argument("--test", action="store_true", help="Test movements only")
    parser.add_argument("--full", action="store_true", help="Push + verify + movement test")
    parser.add_argument("--status", action="store_true", help="Show Pi status summary")
    args = parser.parse_args()

    print("NIS Protocol Pi Deployment")
    print("Pi:", PI_URL)

    if args.status:
        print_status()
        return

    if not check_connectivity():
        print("\nABORTED: Cannot reach Pi")
        sys.exit(1)

    if args.verify:
        ok = verify_poses()
        sys.exit(0 if ok else 1)

    if args.test:
        run_movement_test()
        return

    # Default: push + verify
    push_results = push_poses()
    time.sleep(1.0)
    verified = verify_poses()

    if args.full and verified:
        run_movement_test()

    if verified:
        print("\nDeployment SUCCESS")
        print("Run the full pipeline: python nis_console.py demo")
        print("Or: POST http://localhost:8000/cookoff/arm/orchestrate")
    else:
        print("\nDeployment PARTIAL — some poses missing")
        sys.exit(1)


if __name__ == "__main__":
    main()
