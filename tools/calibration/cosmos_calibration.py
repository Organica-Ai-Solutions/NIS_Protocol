#!/usr/bin/env python3
"""
cosmos_calibration.py — Cosmos Reason2 + Transfer + Predict spatial calibration
for the Hiwonder xArm on the 17×20.5 cm NIS Protocol workspace.

Flow:
  1. Snapshot from Pi camera (angled, 35-45° for depth perception)
  2. Cosmos Reason2  → VLM spatial analysis: workspace zones, object positions
  3. Cosmos Transfer → sim-to-real coordinate mapping
  4. Cosmos Predict  → trajectory validation: each pose generates a predicted next-frame
  5. Servo calibration update → save validated poses to servo_calibration_result.json

Usage:
  python cosmos_calibration.py                # full calibration
  python cosmos_calibration.py --snapshot     # just take and analyze snapshot
  python cosmos_calibration.py --pose home    # calibrate a single pose
  python cosmos_calibration.py --read-memory  # read arm controller position 1 (true home)
  python cosmos_calibration.py --dry-run      # plan only, no arm movement
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# ─── Config ─────────────────────────────────────────────────────────────────
PI_BASE        = "http://192.168.1.163:8085"
COSMOS_REASON2 = "http://localhost:8100"   # Cosmos Reason2 (VLM) via SSH tunnel
COSMOS_PREDICT = "http://localhost:8200"   # Cosmos Predict   via SSH tunnel
COSMOS_TRANSFER= "http://localhost:8300"   # Cosmos Transfer  via SSH tunnel
H100_BASE      = "http://localhost:8080"   # H100 NIS API relay
NIS_BASE       = "http://localhost:8000"   # local NIS Protocol

CALIB_DIR   = Path(__file__).parent / "calibration"
CALIB_JSON  = Path(__file__).parent / "servo_calibration_result.json"
CALIB_DIR.mkdir(exist_ok=True)

# Physical workspace: Hiwonder xArm reach zone
WORKSPACE_CM = {"width": 17.0, "depth": 20.5}

# Servo map: S1=Gripper, S2=Shoulder, S3=Elbow, S4=WristYaw, S5=WristPitch, S6=BaseRotate
SERVO_MAP = {1: "Gripper", 2: "Shoulder", 3: "Elbow",
             4: "WristYaw", 5: "WristPitch", 6: "BaseRotate"}

# ── Safe limits from physical testing ────────────────────────────────────────
SERVO_LIMITS = {
    1: (100, 900),   # Gripper    100=open, 550=closed
    2: (150, 850),   # Shoulder
    3: (150, 850),   # Elbow
    4: (300, 700),   # WristYaw
    5: (200, 900),   # WristPitch
    6: (100, 850),   # BaseRotate
}

# ── Named pipeline poses (starting reference from last physical calibration) ─
PIPELINE_POSES = {
    # HOME: 17×20.5 cm workspace-ready — arm leaning slightly forward,
    #        gripper above workspace center. NOT the vertical attention pose.
    #        S2<500 = shoulder tilted forward; S5=400 = wrist slightly down.
    "home": {"1": 500, "2": 484, "3": 433, "4": 500, "5": 432, "6": 350},

    # ATTENTION: straight-up mechanical center (safe transit pose)
    "attention": {"1": 500, "2": 500, "3": 400, "4": 500, "5": 400, "6": 350},

    # INSPECT: arm raised and forward — camera looks at workspace from arm POV
    "inspect": {"1": 500, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},

    # PICK_TABLE: lowered to object, gripper neutral (open separately before)
    "pick_table": {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500},

    # LIFT_GRIP: arm raised holding object (gripper closed separately)
    "lift_grip": {"1": 550, "2": 625, "3": 485, "4": 500, "5": 335, "6": 500},

    # PLACE_BIN: over bin, still gripping (gripper opened separately after)
    "place_bin": {"1": 550, "2": 370, "3": 720, "4": 380, "5": 680, "6": 240},
}

PICK_PIPELINE = ["home", "inspect", "pick_table", "lift_grip", "place_bin", "home"]

# ─── Helpers ────────────────────────────────────────────────────────────────

def clamp(val: int, servo_id: int) -> int:
    lo, hi = SERVO_LIMITS[servo_id]
    return max(lo, min(hi, val))

def api(method: str, url: str, **kwargs) -> dict:
    try:
        r = httpx.request(method, url, timeout=15, **kwargs)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "json" in ct:
            return r.json()
        return {"raw": r.text, "status": r.status_code}
    except httpx.ConnectError:
        return {"error": f"CONNECTION REFUSED: {url}"}
    except httpx.TimeoutException:
        return {"error": f"TIMEOUT: {url}"}
    except Exception as e:
        return {"error": str(e)}

def pi(method: str, path: str, **kwargs) -> dict:
    return api(method, f"{PI_BASE}{path}", **kwargs)

# ─── Pi camera snapshot ──────────────────────────────────────────────────────

def take_snapshot(save_path: str = None) -> Optional[bytes]:
    """Capture a frame from Pi camera. Returns raw JPEG bytes."""
    print("  → Taking Pi camera snapshot (angled view for depth)...")
    r = pi("POST", "/camera/snapshot")
    if "error" in r:
        print(f"    FAIL: {r['error']}")
        return None

    img_data = None
    if "image" in r:
        img_data = base64.b64decode(r["image"])
    elif "frame" in r:
        img_data = base64.b64decode(r["frame"])
    elif "data" in r:
        img_data = base64.b64decode(r["data"])

    if img_data and save_path:
        Path(save_path).write_bytes(img_data)
        print(f"    Saved {len(img_data)//1024}KB → {save_path}")
    return img_data

# ─── Cosmos Reason2: spatial VLM analysis ────────────────────────────────────

REASON2_SYSTEM_PROMPT = """You are a robotic spatial reasoning assistant analyzing a camera feed
from a Hiwonder xArm robotic arm setup on a 17cm × 20.5cm workspace.
The camera is mounted at a 35-45° angle to the side to enable depth perception.

The workspace is divided into a 3×3 grid (taped on the table):
- Columns: Left, Center, Right
- Rows: Front/Near, Middle, Back/Far
- Bottom-right zone is labeled "BIN" (drop zone)
- Bottom-left zone is labeled "LEFT" (pick zone)

The arm base is at the back-center of the workspace.
Servo mapping: S1=Gripper(1=open,550=closed), S2=Shoulder(low=forward),
               S3=Elbow(low=extended), S4=WristYaw, S5=WristPitch(high=down),
               S6=BaseRotate(100=far-left, 350=center, 850=far-right).

Analyze the image and return a JSON object with:
{
  "workspace_visible": bool,
  "arm_visible": bool,
  "arm_pose_estimate": {"shoulder_angle_deg": ..., "elbow_angle_deg": ..., "position_in_workspace": "left/center/right"},
  "objects_detected": [{"label": "...", "zone": "left/center/right/front/back/bin", "confidence": 0-1, "color": "..."}],
  "depth_visible": bool,
  "recommended_pick_zone": "...",
  "recommended_pick_adjustments": {"S6_delta": int, "S2_delta": int, "S5_delta": int},
  "workspace_notes": "..."
}"""

def cosmos_reason2(img_bytes: bytes, pose_context: str = "") -> dict:
    """Send image to Cosmos Reason2 VLM for spatial analysis."""
    print("  → Sending to Cosmos Reason2 for spatial analysis...")
    b64 = base64.b64encode(img_bytes).decode()

    # Try Cosmos Reason2 endpoint formats
    endpoints = [
        (f"{COSMOS_REASON2}/v1/chat/completions", "openai_compat"),
        (f"{COSMOS_REASON2}/reason", "direct"),
        (f"{COSMOS_REASON2}/generate", "generate"),
        (f"{H100_BASE}/cosmos/reason", "nis_relay"),
        (f"{NIS_BASE}/cosmos/reason", "nis_local"),
    ]

    prompt = f"Analyze this robotic workspace image. {pose_context}\n\nReturn JSON only."

    for url, fmt in endpoints:
        if fmt == "openai_compat":
            payload = {
                "model": "cosmos-reason2",
                "messages": [
                    {"role": "system", "content": REASON2_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]}
                ],
                "max_tokens": 1024,
                "temperature": 0.1,
            }
        elif fmt in ("direct", "generate"):
            payload = {
                "prompt": f"{REASON2_SYSTEM_PROMPT}\n\nUser: {prompt}",
                "image": b64,
                "max_tokens": 1024,
            }
        else:
            payload = {
                "image_base64": b64,
                "prompt": prompt,
                "system": REASON2_SYSTEM_PROMPT,
            }

        r = api("POST", url, json=payload)
        if "error" not in r:
            print(f"    Cosmos Reason2 OK via {fmt} at {url}")
            return _parse_reason2_response(r, fmt)

    print("    Cosmos Reason2 OFFLINE — using geometric fallback")
    return _geometric_fallback()

def _parse_reason2_response(r: dict, fmt: str) -> dict:
    try:
        if fmt == "openai_compat":
            content = r["choices"][0]["message"]["content"]
        elif "result" in r:
            content = r["result"]
        elif "text" in r:
            content = r["text"]
        else:
            content = json.dumps(r)

        start = content.find("{")
        end   = content.rfind("}") + 1
        if start >= 0:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"    Parse error: {e}")
    return _geometric_fallback()

def _geometric_fallback() -> dict:
    """When Cosmos offline: use workspace geometry as prior."""
    return {
        "workspace_visible": True,
        "arm_visible": True,
        "arm_pose_estimate": {"position_in_workspace": "center"},
        "objects_detected": [],
        "depth_visible": True,
        "recommended_pick_zone": "center",
        "recommended_pick_adjustments": {"S6_delta": 0, "S2_delta": 0, "S5_delta": 0},
        "workspace_notes": "OFFLINE — geometric prior used (no Cosmos adjustment)",
        "_offline": True,
    }

# ─── Cosmos Transfer: pixel → arm-coordinate mapping ─────────────────────────

def cosmos_transfer(reason2_result: dict, current_pose: dict) -> dict:
    """
    Use Cosmos Transfer to map pixel-space object positions to arm servo coordinates.
    Falls back to a linear geometric model when offline.
    """
    print("  → Cosmos Transfer: pixel → servo coordinate mapping...")

    # Try Transfer endpoint
    payload = {
        "workspace_analysis": reason2_result,
        "current_servo_state": current_pose,
        "workspace_dims_cm": WORKSPACE_CM,
        "servo_map": SERVO_MAP,
        "coordinate_system": {
            "origin": "arm_base_center",
            "x_axis": "left_right (S6: 100=left, 850=right)",
            "y_axis": "depth (S2+S3: low=forward)",
            "z_axis": "height (S2: 500=upright, 258=forward-low)",
        },
    }

    for url in [f"{COSMOS_TRANSFER}/transfer", f"{H100_BASE}/cosmos/transfer", f"{NIS_BASE}/cosmos/transfer"]:
        r = api("POST", url, json=payload)
        if "error" not in r:
            print(f"    Transfer OK via {url}")
            return r

    print("    Cosmos Transfer OFFLINE — using linear geometric model")
    return _linear_transfer(reason2_result, current_pose)

def _linear_transfer(reason2_result: dict, current_pose: dict) -> dict:
    """
    Geometric sim-to-real mapping when Cosmos Transfer is offline.

    17cm wide × 20.5cm deep workspace:
      - S6 (base rotation) spans left(100) to right(850) → ~750 units / 17cm
      - S2 (shoulder) spans near(258) to center(484) for Y depth reach
      - The arm is mounted at the back, so objects in "front" zone = low S2 + high S3
    """
    adjs = reason2_result.get("recommended_pick_adjustments", {})
    zone = reason2_result.get("recommended_pick_zone", "center")

    # Zone → S6 base rotation mapping
    zone_s6 = {"left": 200, "center": 500, "right": 750, "bin": 240}
    s6_target = zone_s6.get(zone, 500)

    # Apply Cosmos delta adjustments on top of zone estimate
    s6_delta = adjs.get("S6_delta", 0)
    s2_delta = adjs.get("S2_delta", 0)
    s5_delta = adjs.get("S5_delta", 0)

    # Pick position based on PIPELINE_POSES["pick_table"] as reference
    ref = PIPELINE_POSES["pick_table"].copy()
    transfer = {
        "pick_servo_target": {
            "1": ref["1"],
            "2": clamp(ref["2"] + s2_delta, 2),
            "3": ref["3"],
            "4": ref["4"],
            "5": clamp(ref["5"] + s5_delta, 5),
            "6": clamp(s6_target + s6_delta, 6),
        },
        "mapping_method": "geometric_linear",
        "workspace_cm": WORKSPACE_CM,
        "zone": zone,
    }
    return transfer

# ─── Cosmos Predict: trajectory validation ───────────────────────────────────

def cosmos_predict(img_bytes: bytes, pose_name: str, servo_positions: dict) -> dict:
    """
    Use Cosmos Predict to validate that moving to servo_positions from current state
    will result in the expected arm configuration (no collisions, reach ok).
    Falls back to range-check when offline.
    """
    print(f"  → Cosmos Predict: validating trajectory to '{pose_name}'...")

    b64 = base64.b64encode(img_bytes).decode() if img_bytes else ""
    payload = {
        "current_frame_base64": b64,
        "target_pose_name": pose_name,
        "target_servo_positions": servo_positions,
        "predict_steps": 5,
        "collision_check": True,
        "workspace_bounds_cm": WORKSPACE_CM,
    }

    for url in [f"{COSMOS_PREDICT}/predict", f"{H100_BASE}/cosmos/predict", f"{NIS_BASE}/cosmos/predict"]:
        r = api("POST", url, json=payload)
        if "error" not in r:
            print(f"    Predict OK via {url}")
            return r

    print("    Cosmos Predict OFFLINE — using servo-limit validation")
    return _validate_limits(pose_name, servo_positions)

def _validate_limits(pose_name: str, positions: dict) -> dict:
    """Offline validation: check servo limits and basic geometry."""
    warnings = []
    clamped  = {}
    ok = True

    for sid, val in positions.items():
        sid_int = int(sid)
        lo, hi  = SERVO_LIMITS.get(sid_int, (0, 1000))
        if val < lo or val > hi:
            warnings.append(f"S{sid} value {val} outside safe range [{lo},{hi}]")
            ok = False
        clamped[sid] = clamp(val, sid_int)

    # Basic geometry: pick_table needs S2 < 400 (arm forward)
    if pose_name == "pick_table":
        s2 = int(positions.get("2", 500))
        if s2 > 400:
            warnings.append(f"pick_table S2={s2} looks too upright (should be <350)")

    return {
        "valid": ok,
        "clamped_positions": clamped,
        "warnings": warnings,
        "method": "limit_check_offline",
        "pose": pose_name,
    }

# ─── Arm controller memory (action group) ───────────────────────────────────

def read_arm_memory_position(group_id: int = 1) -> Optional[dict]:
    """
    Trigger the arm's built-in action group (remote-controller saved position).
    Position 1 = user's saved HOME on the remote controller.
    After triggering, reads back servo positions.
    """
    print(f"\n=== Reading arm controller memory position {group_id} (remote-saved HOME) ===")

    # Method 1: Try action group endpoint if agent was updated
    for path in [f"/arm/action_group/run/{group_id}", f"/arm/action/{group_id}/run"]:
        r = pi("POST", path, json={"count": 1})
        if "error" not in r and r.get("ok"):
            print(f"  Action group {group_id} triggered via {path}")
            time.sleep(2.5)
            state = pi("GET", "/arm/status")
            return state.get("positions") or state.get("servo_state")

    # Method 2: Send raw HID bytes for CMD_ACTION_GROUP_RUN (0x06)
    # Packet: 55 AA 06 06 [group_id] 01 00 checksum
    checksum = (0x06 + 0x06 + group_id + 0x01 + 0x00) & 0xFF
    raw_bytes = [0x55, 0xAA, 0x06, 0x06, group_id, 0x01, 0x00, checksum]
    r = pi("POST", "/arm/raw_hid", json={"bytes": raw_bytes})
    if "error" not in r:
        print(f"  Raw HID action group {group_id} sent")
        time.sleep(2.5)
        state = pi("GET", "/arm/status")
        return state.get("positions") or state.get("servo_state")

    print(f"  Cannot auto-trigger memory position {group_id}.")
    print("  ACTION REQUIRED: Use the xArm remote controller to send arm to")
    print(f"  memory position {group_id} (your saved HOME), then press ENTER.")
    input("  Press ENTER after arm reaches memory position 1...")

    time.sleep(0.5)
    state = pi("GET", "/arm/status")
    pos = state.get("positions") or state.get("servo_state")
    if pos:
        print(f"  Read from hardware: {pos}")
    return pos

# ─── Main calibration routines ───────────────────────────────────────────────

def calibrate_pose(pose_name: str, dry_run: bool = False) -> dict:
    """
    Full Cosmos-powered calibration for a single named pose.
    Returns the validated servo positions.
    """
    print(f"\n{'='*60}")
    print(f" Calibrating: {pose_name}")
    print(f"{'='*60}")

    ref = PIPELINE_POSES.get(pose_name, {})
    if not ref:
        print(f"  No reference pose for '{pose_name}'")
        return {}

    # Move arm to reference position (unless dry_run)
    if not dry_run:
        print(f"  Moving arm to reference position...")
        r = pi("POST", "/arm/group_move", json={"positions": ref, "duration_ms": 1500})
        if r.get("ok") or r.get("status") == "ok":
            time.sleep(2.0)
        else:
            print(f"  Move failed: {r}")

    # 1. Take snapshot
    snap_path = str(CALIB_DIR / f"cosmos_calib_{pose_name}.jpg")
    img_bytes = take_snapshot(snap_path)
    if not img_bytes:
        print("  No snapshot — using reference pose as-is")
        return ref

    # 2. Cosmos Reason2 — spatial analysis
    reason = cosmos_reason2(img_bytes, pose_context=f"The arm is moving to '{pose_name}' pose.")
    print(f"  Reason2: zone={reason.get('recommended_pick_zone','?')} "
          f"objects={len(reason.get('objects_detected',[]))} "
          f"depth={reason.get('depth_visible','?')}")
    if reason.get("workspace_notes"):
        print(f"  Notes: {reason['workspace_notes']}")

    # 3. Cosmos Transfer — pixel→servo mapping (only for pick_table)
    if pose_name == "pick_table":
        transfer = cosmos_transfer(reason, ref)
        pick_target = transfer.get("pick_servo_target", ref)
        print(f"  Transfer → pick target S6={pick_target.get('6')} S2={pick_target.get('2')}")
    else:
        pick_target = ref

    # 4. Cosmos Predict — validate trajectory
    prediction = cosmos_predict(img_bytes, pose_name, pick_target)
    if prediction.get("warnings"):
        for w in prediction["warnings"]:
            print(f"  WARN: {w}")
    if not prediction.get("valid", True):
        print(f"  Prediction invalid — falling back to reference pose")
        pick_target = ref

    # Use clamped positions if provided
    if "clamped_positions" in prediction:
        final = {str(k): v for k, v in prediction["clamped_positions"].items()}
    else:
        final = {str(k): clamp(int(v), int(k)) for k, v in pick_target.items()}

    # 5. If not dry_run, move to final position and read hardware
    if not dry_run and pose_name != "home":
        print(f"  Moving to Cosmos-calibrated position...")
        r = pi("POST", "/arm/group_move", json={"positions": final, "duration_ms": 1200})
        time.sleep(1.8)
        state = pi("GET", "/arm/status")
        hw_pos = state.get("positions") or state.get("servo_state") or {}
        if hw_pos:
            hw_str = " ".join(f"S{k}={v}" for k, v in sorted(hw_pos.items()))
            print(f"  Hardware read-back: {hw_str}")
            # Use hardware positions as ground truth (removes HID rounding errors)
            final = {str(k): v for k, v in hw_pos.items()}

    print(f"  Final: {final}")
    return final

def full_calibration(dry_run: bool = False, read_memory: bool = True):
    """Run Cosmos-powered calibration for all pipeline poses."""
    print("\n" + "="*60)
    print(" NIS Protocol — Cosmos Calibration Pipeline")
    print(f" Workspace: {WORKSPACE_CM['width']}cm × {WORKSPACE_CM['depth']}cm")
    print("="*60)

    # Check connectivity
    cosmos_up = _check_cosmos()
    pi_up     = _check_pi()

    if not pi_up:
        print("ERROR: Raspberry Pi agent not reachable at", PI_BASE)
        sys.exit(1)

    results = {}

    # ── Home position: read from arm controller memory ────────────────────
    print("\n--- HOME POSITION ---")
    print("The 17×20.5 cm workspace-ready home is NOT the vertical attention pose.")
    print("It is saved as Position 1 in the xArm remote controller memory.")

    if read_memory:
        hw_home = read_arm_memory_position(group_id=1)
        if hw_home:
            results["home"] = {str(k): v for k, v in hw_home.items()}
            print(f"  Using arm memory position 1 as HOME: {results['home']}")
        else:
            results["home"] = PIPELINE_POSES["home"]
            print(f"  Using geometric reference as HOME: {results['home']}")
    else:
        results["home"] = PIPELINE_POSES["home"]
        print(f"  Using geometric reference as HOME (skip memory read).")

    # Save attention pose separately (straight-up transit)
    results["attention"] = PIPELINE_POSES["attention"]

    # ── Calibrate remaining pipeline poses with Cosmos ────────────────────
    for pose_name in ["inspect", "pick_table", "lift_grip", "place_bin"]:
        calibrated = calibrate_pose(pose_name, dry_run=dry_run)
        results[pose_name] = calibrated

        # Return arm to home between poses
        if not dry_run:
            print(f"\n  Returning to home before next pose...")
            pi("POST", "/arm/group_move",
               json={"positions": results["home"], "duration_ms": 1500})
            time.sleep(2.0)

    # ── Save results ──────────────────────────────────────────────────────
    existing = {}
    if CALIB_JSON.exists():
        existing = json.loads(CALIB_JSON.read_text())

    existing.update({
        "_comment": "Cosmos-calibrated xArm poses. S1=Gripper, S6=BaseRotate.",
        "_cosmos_calibrated": True,
        "_calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_workspace_cm": WORKSPACE_CM,
        "_servo_map": {str(k): v for k, v in SERVO_MAP.items()},
    })
    existing.update(results)

    CALIB_JSON.write_text(json.dumps(existing, indent=2))
    print(f"\n=== Calibration saved → {CALIB_JSON} ===")
    print(json.dumps(results, indent=2))

    # Push to Pi
    if not dry_run:
        _push_poses_to_pi(results)

    return results

def _push_poses_to_pi(poses: dict):
    """Push all calibrated poses to Pi via /arm/save_touch_pose (move-then-save)."""
    print("\n--- Pushing calibrated poses to Pi ---")
    for name, pos in poses.items():
        if name.startswith("_") or name == "attention":
            continue
        r = pi("POST", "/arm/group_move", json={"positions": pos, "duration_ms": 1200})
        time.sleep(2.0)
        r2 = pi("POST", "/arm/save_touch_pose", json={"name": name, "positions": pos})
        ok = r2.get("ok") or r2.get("saved") or r2.get("status") == "ok"
        print(f"  {name}: {'OK' if ok else 'FAIL'} {r2}")

def _check_cosmos() -> bool:
    print("\n--- Cosmos H100 connectivity ---")
    up = False
    for svc, url in [("Reason2", COSMOS_REASON2), ("Predict", COSMOS_PREDICT),
                     ("Transfer", COSMOS_TRANSFER)]:
        r = api("GET", f"{url}/health")
        ok = "error" not in r
        print(f"  {svc}: {'UP' if ok else 'OFFLINE'} ({url})")
        if ok:
            up = True
    if not up:
        print("  Cosmos offline. Start keep_tunnel.py first for full GPU reasoning.")
        print("  Falling back to geometric calibration model.")
    return up

def _check_pi() -> bool:
    print("--- Pi connectivity ---")
    r = api("GET", f"{PI_BASE}/health")
    ok = "error" not in r
    print(f"  Pi agent: {'UP' if ok else 'DOWN'} ({PI_BASE})")
    return ok

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Cosmos-powered xArm calibration")
    ap.add_argument("--snapshot", action="store_true", help="Take and analyze snapshot only")
    ap.add_argument("--pose", help="Calibrate a single pose by name")
    ap.add_argument("--read-memory", action="store_true", help="Read arm controller memory position 1")
    ap.add_argument("--dry-run", action="store_true", help="Plan only — no arm movement")
    ap.add_argument("--push-only", action="store_true", help="Push current CALIB_JSON poses to Pi")
    ap.add_argument("--no-memory", action="store_true", help="Skip reading arm controller memory")
    args = ap.parse_args()

    if args.snapshot:
        img = take_snapshot(str(CALIB_DIR / "cosmos_workspace.jpg"))
        if img:
            r = cosmos_reason2(img)
            print(json.dumps(r, indent=2))
        return

    if args.read_memory:
        pos = read_arm_memory_position(1)
        print("\nMemory position 1 (true HOME):", pos)
        return

    if args.pose:
        pos = calibrate_pose(args.pose, dry_run=args.dry_run)
        print(json.dumps({args.pose: pos}, indent=2))
        return

    if args.push_only:
        existing = json.loads(CALIB_JSON.read_text()) if CALIB_JSON.exists() else {}
        _push_poses_to_pi(existing)
        return

    full_calibration(dry_run=args.dry_run, read_memory=not args.no_memory)

if __name__ == "__main__":
    main()
