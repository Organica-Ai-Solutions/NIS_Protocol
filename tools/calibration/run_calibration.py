#!/usr/bin/env python3
"""
NIS Protocol — Live Calibration Runner
======================================
Runs the full Cosmos Reason2 auto-calibration:
  1. Reads ALL poses from arm memory (ground truth)
  2. Moves to each pose
  3. Captures camera frame
  4. Sends to Cosmos Reason2 for spatial analysis
  5. Computes servo corrections
  6. Saves corrected poses back to arm memory

Also saves every frame to data/calib_frames/ for visual review.

Usage:
    python run_calibration.py              # calibrate inspect + pick_table + place_bin
    python run_calibration.py --all        # all poses including home
    python run_calibration.py --dry-run    # move and capture but do NOT save corrections
    python run_calibration.py --pose pick_table   # single pose only
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("calibration")

PI_URL     = "http://192.168.1.163:8085"
COSMOS_URL = "http://localhost:8100"
FRAME_DIR  = Path("data/calib_frames")

FRAME_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def pi_get(path, timeout=15):
    try:
        r = urllib.request.urlopen(PI_URL + path, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def pi_post(path, data, timeout=15):
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(PI_URL + path, data=body,
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def cosmos_post(payload, timeout=30):
    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(COSMOS_URL + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def capture_frame(label):
    """Capture camera frame, save to disk, return base64 string."""
    snap = pi_get("/camera/snapshot", timeout=15)
    img = snap.get("image_base64", "")
    if not img:
        logger.warning(f"  No image for {label}")
        return None
    raw = base64.b64decode(img)
    path = FRAME_DIR / f"{label}.jpg"
    path.write_bytes(raw)
    logger.info(f"  Frame saved: {path} ({len(raw)//1024}KB)")
    return img   # return base64 string for Cosmos

def read_arm_poses():
    """Read poses from arm memory. Key is 'poses' in this Pi version."""
    r = pi_get("/arm/touch_poses")
    poses = r.get("touch_poses") or r.get("poses") or {}
    return {name: {str(k): int(v) for k, v in pos.items()}
            for name, pos in poses.items() if isinstance(pos, dict)}

def move_to(pose_dict, duration_ms=1500):
    r = pi_post("/arm/group_move", {"positions": pose_dict, "duration_ms": duration_ms})
    return r.get("ok", False)

def save_pose(name, positions):
    """Move to positions and save as named pose in arm memory."""
    move_to(positions, duration_ms=1000)
    time.sleep(1.2)
    r = pi_post("/arm/save_touch_pose", {"name": name})
    return "positions" in r or r.get("ok", False)

# ── Cosmos Reason2 analysis ───────────────────────────────────────────────────

COSMOS_PROMPTS = {
    "home": (
        "This is a 6DOF robotic arm at HOME position viewed from above at a slight angle. "
        "The workspace (17cm wide x 20.5cm deep) has handwritten labels on the table: "
        "'Left Right', 'Front Right', 'Back', 'Back Right'. "
        "A BLUE marker is on the LEFT side. A GREEN lighter is on the RIGHT side. "
        "Analyze: Is the arm centered and upright? Is this a safe home position? "
        "Return JSON only: {arm_centered: bool, arm_upright: bool, safe: bool, "
        "notes: string, confidence: float}"
    ),
    "inspect": (
        "This is a 6DOF robotic arm at INSPECT position viewed from above at a slight angle. "
        "Workspace labels: 'Left Right' (upper-left), 'Front Right' (upper-right), "
        "'Back' (lower-center), 'Back Right' (lower-right). "
        "BLUE marker is LEFT, GREEN lighter is RIGHT of the arm. "
        "The arm should be raised high to see the workspace from above. "
        "Analyze the gripper position relative to the two objects. "
        "Return JSON only: {gripper_visible: bool, blue_marker_visible: bool, "
        "green_lighter_visible: bool, gripper_lateral_offset_from_center_mm: float, "
        "good_inspect_height: bool, notes: string, confidence: float}"
    ),
    "pick_table": (
        "This is a 6DOF robotic arm at PICK TABLE position — the gripper has descended "
        "to table level to pick an object. "
        "Workspace labels: 'Left Right' (upper-left), 'Front Right' (upper-right), "
        "'Back' (lower-center). "
        "BLUE marker is on the LEFT. GREEN lighter is on the RIGHT — the GREEN LIGHTER "
        "is the PICK TARGET. "
        "The gripper (yellow/orange cylindrical tip) should be directly above the GREEN lighter. "
        "Analyze alignment: Is the gripper over the green lighter? "
        "If not, estimate the error in mm. "
        "Return JSON only: {gripper_over_lighter: bool, "
        "lateral_error_mm: float, "
        "depth_error_mm: float, "
        "gripper_too_high_mm: float, "
        "gripper_too_low: bool, "
        "recommended_S6_delta: int, "
        "recommended_S2_delta: int, "
        "notes: string, confidence: float}"
    ),
    "place_bin": (
        "This is a 6DOF robotic arm at PLACE BIN position — the arm has rotated left "
        "to drop the object into a bin. "
        "Workspace labels visible on the table. "
        "Analyze: Is the arm safely positioned over the bin area (left side)? "
        "Return JSON only: {arm_over_bin_area: bool, safe_to_drop: bool, "
        "lateral_error_mm: float, notes: string, confidence: float}"
    ),
}

def analyze_with_cosmos(pose_name, frame_b64):
    """Send frame to Cosmos Reason2. Returns parsed JSON or None."""
    prompt = COSMOS_PROMPTS.get(pose_name, f"Analyze the robotic arm at {pose_name} position. Return JSON.")
    payload = {
        "model": "cosmos-reason2",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame_b64
                    if not frame_b64.startswith("data:") else frame_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 512,
        "temperature": 0.1,
    }
    r = cosmos_post(payload, timeout=30)
    if "error" in r:
        logger.warning(f"  Cosmos error: {r['error'][:80]}")
        return None
    if "choices" not in r:
        logger.warning("  Cosmos: no choices in response")
        return None

    text = r["choices"][0]["message"]["content"]
    logger.info(f"  Cosmos raw: {text[:300]}")

    import re
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"raw": text}

# ── Correction computation ─────────────────────────────────────────────────────

def compute_correction(pose_name, original, cosmos):
    """Given Cosmos analysis, compute corrected pose dict."""
    if not cosmos:
        return original, {}

    corrected = dict(original)
    delta = {}

    if pose_name == "pick_table":
        # S6 lateral correction (base rotation): positive = rotate right
        s6_d = cosmos.get("recommended_S6_delta", 0)
        if s6_d == 0:
            # Compute from lateral error
            lat = cosmos.get("lateral_error_mm", 0)
            s6_d = int(-lat * 3.5)
        s6_d = max(-120, min(120, s6_d))
        if abs(s6_d) > 5:
            corrected["6"] = max(100, min(900, int(original.get("6", 500)) + s6_d))
            delta["6"] = s6_d

        # S2 height correction
        s2_d = cosmos.get("recommended_S2_delta", 0)
        if s2_d == 0:
            too_high = cosmos.get("gripper_too_high_mm", 0)
            if too_high > 10:
                s2_d = -int((too_high - 5) * 3)
            elif cosmos.get("gripper_too_low", False):
                s2_d = 20
        s2_d = max(-60, min(40, s2_d))
        if abs(s2_d) > 3:
            corrected["2"] = max(200, min(900, int(original.get("2", 258)) + s2_d))
            delta["2"] = s2_d

    elif pose_name == "inspect":
        lat = cosmos.get("gripper_lateral_offset_from_center_mm", 0)
        if abs(lat) > 5:
            s6_d = int(-lat * 3.5)
            s6_d = max(-60, min(60, s6_d))
            corrected["6"] = max(100, min(900, int(original.get("6", 500)) + s6_d))
            delta["6"] = s6_d

    elif pose_name == "place_bin":
        lat = cosmos.get("lateral_error_mm", 0)
        if abs(lat) > 5:
            s6_d = int(-lat * 3.5)
            s6_d = max(-60, min(60, s6_d))
            corrected["6"] = max(100, min(900, int(original.get("6", 240)) + s6_d))
            delta["6"] = s6_d

    return corrected, delta

# ── Main calibration loop ──────────────────────────────────────────────────────

def calibrate_pose(pose_name, original_pose, cosmos_online, dry_run):
    print(f"\n{'='*50}")
    print(f"  CALIBRATING: {pose_name}")
    print(f"  Original: {original_pose}")
    print(f"{'='*50}")

    # Move to pose
    print(f"  Moving to {pose_name}...")
    ok = move_to(original_pose, duration_ms=1800)
    time.sleep(2.5)

    # Capture frame
    frame_b64 = capture_frame(f"cal_{pose_name}")
    if not frame_b64:
        print("  ERROR: no frame captured")
        return original_pose, {}

    # Cosmos Reason2 analysis
    cosmos_result = None
    if cosmos_online:
        print("  Sending to Cosmos Reason2...")
        cosmos_result = analyze_with_cosmos(pose_name, frame_b64)
        if cosmos_result:
            print(f"  Cosmos result: {json.dumps(cosmos_result, indent=4)}")
        else:
            print("  Cosmos: no result")
    else:
        print("  Cosmos offline — skipping visual analysis")

    # Compute correction
    corrected, delta = compute_correction(pose_name, original_pose, cosmos_result)

    if delta:
        print(f"\n  Correction: {delta}")
        print(f"  Original:  {original_pose}")
        print(f"  Corrected: {corrected}")
    else:
        print("\n  No correction needed (or Cosmos offline)")

    # Save
    if not dry_run and delta:
        print(f"  Saving corrected {pose_name} to arm memory...")
        saved = save_pose(pose_name, corrected)
        print(f"  Saved: {saved}")
    elif dry_run:
        print("  [DRY RUN] Not saving")

    return corrected, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Calibrate all poses")
    parser.add_argument("--dry-run", action="store_true", help="Don't save corrections")
    parser.add_argument("--pose", default=None, help="Single pose name to calibrate")
    args = parser.parse_args()

    print("\nNIS Protocol — Cosmos Reason2 Auto-Calibration")
    print("=" * 50)

    # Check Pi
    health = pi_get("/health")
    if health.get("error"):
        print("ERROR: Pi agent offline")
        return
    print(f"Pi: v{health.get('version')} | xArm: {health.get('xarm')}")

    # Check Cosmos
    cosmos_online = False
    try:
        r = urllib.request.urlopen(COSMOS_URL + "/health", timeout=4)
        cosmos_online = True
        print("Cosmos Reason2: ONLINE")
    except Exception:
        print("Cosmos Reason2: OFFLINE (will capture frames but skip AI analysis)")

    # Read arm memory
    poses = read_arm_poses()
    print(f"\nPoses in arm memory: {list(poses.keys())}")

    # Select poses to calibrate
    if args.pose:
        to_calibrate = [args.pose]
    elif args.all:
        to_calibrate = ["home", "inspect", "pick_table", "place_bin"]
    else:
        to_calibrate = ["inspect", "pick_table", "place_bin"]

    print(f"Will calibrate: {to_calibrate}")

    # Read home for returns between poses
    home = poses.get("home", {"1":500,"2":484,"3":433,"4":500,"5":432,"6":350})
    print(f"\nHome (from arm memory): {home}")

    # Calibrate each pose
    results = {}
    for pose_name in to_calibrate:
        if pose_name not in poses:
            print(f"\nSkipping {pose_name}: not in arm memory")
            continue

        corrected, delta = calibrate_pose(
            pose_name, poses[pose_name], cosmos_online, args.dry_run
        )
        results[pose_name] = {"original": poses[pose_name], "corrected": corrected, "delta": delta}

        # Return home between poses
        if pose_name != to_calibrate[-1]:
            print(f"\n  Returning home...")
            move_to(home, duration_ms=1200)
            time.sleep(1.5)

    # Final home
    print("\n\nReturning to home...")
    move_to(home, duration_ms=1500)
    time.sleep(1.5)
    capture_frame("final_home")

    # Summary
    print("\n" + "="*50)
    print("CALIBRATION SUMMARY")
    print("="*50)
    for name, r in results.items():
        d = r["delta"]
        if d:
            delta_str = "  ".join(f"S{k}: {'+' if v>0 else ''}{v}" for k,v in d.items())
            print(f"  {name:<15} CORRECTED: {delta_str}")
        else:
            print(f"  {name:<15} no change")

    print(f"\nFrames saved to: {FRAME_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
