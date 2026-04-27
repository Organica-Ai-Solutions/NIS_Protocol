#!/usr/bin/env python3
"""
push_poses_via_move.py
======================
Pushes calibrated touch poses to a Raspberry Pi running the OLD NeuroLinux agent
(which lacks the /arm/load_calibration endpoint).

Strategy: For each pose, MOVE the arm to that position via /arm/group_move,
wait for it to settle, then call /arm/save_touch_pose to capture the hardware position.

Usage:
    python push_poses_via_move.py                      # push all required poses
    python push_poses_via_move.py --pi 192.168.1.163   # specify Pi IP
    python push_poses_via_move.py --dry-run            # show plan without moving
    python push_poses_via_move.py --verify             # show current Pi poses
    python push_poses_via_move.py --poses home inspect # push only specific poses

WARNING: This PHYSICALLY MOVES the arm through all calibrated positions.
         Make sure the arm workspace is clear before running!
"""
import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PI   = "192.168.1.163"
DEFAULT_PORT = 8085
CAL_FILE     = Path(__file__).parent / "servo_calibration_result.json"

# Poses required for the pick-and-place pipeline (in execution order)
PIPELINE_POSES = ["home", "inspect", "pick_table", "lift_grip", "place_bin"]
# All poses (pipeline + extras)
ALL_POSES = PIPELINE_POSES + ["ready", "reach_left", "reach_right", "wave_up", "wave_side"]

# Movement settle time in seconds
SETTLE_S = 1.5

# ── Load calibration ──────────────────────────────────────────────────────────
def load_calibration(file: Path) -> dict:
    with open(file) as f:
        raw = json.load(f)
    poses = {}
    for name, data in raw.items():
        if name.startswith("_"):
            continue
        servo_vals = {k: int(v) for k, v in data.items()
                      if k.isdigit() or (isinstance(k, str) and str(k).lstrip('-').isdigit())}
        if servo_vals:
            poses[name] = servo_vals
    return poses

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def get_pi(base_url: str, path: str, timeout: int = 8) -> dict:
    r = requests.get(f"{base_url}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def post_pi(base_url: str, path: str, body: dict, timeout: int = 10) -> dict:
    r = requests.post(f"{base_url}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()

def move_arm(base_url: str, positions: dict, duration_ms: int = 900) -> bool:
    """Move arm to servo unit positions."""
    body = {"positions": {str(k): int(v) for k, v in positions.items()},
            "duration_ms": duration_ms}
    resp = post_pi(base_url, "/arm/group_move", body)
    return resp.get("ok", False)

def save_pose(base_url: str, name: str) -> dict:
    """Save current hardware position as named pose."""
    resp = post_pi(base_url, "/arm/save_touch_pose", {"name": name})
    return resp

def delete_pose(base_url: str, name: str) -> bool:
    """Delete a pose from Pi (if the endpoint exists)."""
    try:
        resp = post_pi(base_url, "/arm/delete_pose", {"name": name})
        return resp.get("ok", False)
    except Exception:
        return False

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Push arm poses to Pi (move-then-save)")
    parser.add_argument("--pi",       default=DEFAULT_PI, help="Pi IP address")
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT)
    parser.add_argument("--file",     default=str(CAL_FILE), help="Calibration JSON file")
    parser.add_argument("--dry-run",  action="store_true", help="Show plan, don't move")
    parser.add_argument("--verify",   action="store_true", help="Show current Pi poses and exit")
    parser.add_argument("--poses",    nargs="*", help="Specific poses to push (default: all pipeline)")
    parser.add_argument("--settle",   type=float, default=SETTLE_S, help="Settle time in seconds")
    parser.add_argument("--cleanup",  action="store_true", help="Delete old wrong poses after push")
    args = parser.parse_args()

    base_url = f"http://{args.pi}:{args.port}"
    cal_file = Path(args.file)

    # ── Verify mode ───────────────────────────────────────────────────────────
    if args.verify:
        print(f"\nCurrent Pi poses at {base_url}:")
        try:
            resp = get_pi(base_url, "/arm/touch_poses")
            poses = resp.get("poses", resp.get("touch_poses", {}))
            if not poses:
                print("  (none saved)")
            for name, vals in sorted(poses.items()):
                if name.startswith("_"):
                    continue
                s = " ".join(f"S{k}={v}" for k, v in sorted(vals.items(), key=lambda x: int(x[0])))
                status = "OK" if name in PIPELINE_POSES else "extra"
                print(f"  [{status}] {name}: {s}")
            required = set(PIPELINE_POSES)
            have = set(poses.keys())
            missing = required - have
            if missing:
                print(f"\nMISSING required poses: {', '.join(sorted(missing))}")
            else:
                print(f"\nAll {len(PIPELINE_POSES)} pipeline poses present.")
        except Exception as e:
            print(f"Cannot reach Pi: {e}")
        return

    # ── Load calibration ──────────────────────────────────────────────────────
    if not cal_file.exists():
        print(f"Calibration file not found: {cal_file}")
        sys.exit(1)

    all_cal = load_calibration(cal_file)
    target_names = args.poses if args.poses else PIPELINE_POSES

    print(f"\nCalibration file: {cal_file}")
    print(f"Pi target:        {base_url}")
    print(f"Poses to push:    {target_names}")
    print(f"Settle time:      {args.settle}s\n")

    # Check which poses are in the calibration file
    available = {n: all_cal[n] for n in target_names if n in all_cal}
    missing_from_file = [n for n in target_names if n not in all_cal]
    if missing_from_file:
        print(f"WARNING: These poses are NOT in calibration file: {missing_from_file}")
    if not available:
        print("No poses to push.")
        sys.exit(0)

    # ── Preview ───────────────────────────────────────────────────────────────
    print("Plan:")
    for name, vals in available.items():
        s = " ".join(f"S{k}={v}" for k, v in sorted(vals.items()))
        print(f"  {name}: {s}")

    if args.dry_run:
        print("\n[DRY RUN] No movements will be made.")
        return

    # ── Safety confirmation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WARNING: The arm will PHYSICALLY MOVE through each position.")
    print("Make sure the workspace is CLEAR before continuing!")
    print("=" * 60)
    answer = input("\nType 'yes' to continue, anything else to abort: ").strip().lower()
    if answer != "yes":
        print("Aborted.")
        return

    # ── Push each pose ────────────────────────────────────────────────────────
    success = []
    failed = []

    # Always go home first for safety
    if "home" in all_cal and "home" not in available:
        print("\nMoving to home first for safety...")
        move_arm(base_url, all_cal["home"], duration_ms=1200)
        time.sleep(1.5)

    for name, positions in available.items():
        print(f"\n── {name} ──────────────────────")
        s = " ".join(f"S{k}={v}" for k, v in sorted(positions.items()))
        print(f"   Moving to: {s}")
        try:
            ok = move_arm(base_url, positions, duration_ms=900)
            if not ok:
                print(f"   MOVE FAILED")
                failed.append(name)
                continue
            print(f"   Settling ({args.settle}s)...")
            time.sleep(args.settle)
            resp = save_pose(base_url, name)
            saved_positions = resp.get("positions", {})
            s2 = " ".join(f"S{k}={v}" for k, v in sorted(saved_positions.items()))
            print(f"   Saved:   {s2}")

            # Validate: check saved values are close to intended
            mismatches = []
            for k, intended_v in positions.items():
                saved_v = saved_positions.get(str(k), saved_positions.get(k))
                if saved_v is None:
                    continue
                diff = abs(int(saved_v) - int(intended_v))
                if diff > 30:
                    mismatches.append(f"S{k}: intended={intended_v}, got={saved_v}, diff={diff}")
            if mismatches:
                print(f"   WARN: position drift detected:")
                for m in mismatches:
                    print(f"         {m}")
            else:
                print(f"   OK — within tolerance")
            success.append(name)
        except Exception as e:
            print(f"   ERROR: {e}")
            failed.append(name)

    # ── Return home ───────────────────────────────────────────────────────────
    if "home" in all_cal:
        print("\nReturning to home...")
        move_arm(base_url, all_cal["home"], duration_ms=1200)
        time.sleep(1.5)

    # ── Cleanup old wrong poses ───────────────────────────────────────────────
    if args.cleanup:
        old_poses = ["pick_blue", "place_closed", "_audit_test", "_test_hw"]
        print("\nCleaning up old poses...")
        for old in old_poses:
            ok = delete_pose(base_url, old)
            print(f"  delete '{old}': {'ok' if ok else 'not available'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"DONE: {len(success)} poses pushed, {len(failed)} failed")
    if success:
        print(f"  Pushed:  {', '.join(success)}")
    if failed:
        print(f"  Failed:  {', '.join(failed)}")
    print("=" * 60)

    # Final verify
    print("\nFinal Pi pose list:")
    try:
        resp = get_pi(base_url, "/arm/touch_poses")
        poses = resp.get("poses", resp.get("touch_poses", {}))
        for n, v in sorted(poses.items()):
            if not n.startswith("_"):
                print(f"  {n}: {v}")
    except Exception as e:
        print(f"  (could not fetch: {e})")


if __name__ == "__main__":
    main()
