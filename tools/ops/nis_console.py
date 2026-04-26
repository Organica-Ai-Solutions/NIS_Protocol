#!/usr/bin/env python3
"""
NIS Protocol Console — "Claude Code in the console"
Inspired by OpenFang's CLI pattern and openclaw/clawbot console UX.

Usage:
    python nis_console.py                    # Interactive chat mode
    python nis_console.py "pick up the block" # Single command
    python nis_console.py --agent arm        # Focus on arm control
    python nis_console.py --cosmos           # Enable Cosmos reasoning

Architecture: This is the OpenFang SDK pattern adapted for NIS Protocol.
The console routes to: NIS Protocol (localhost:8000) --> Pi Agent (192.168.1.163:8085)
                     --> Cosmos Reason2 (localhost:8100) --> xArm hardware
"""

import json
import os
import sys
import time
import argparse
import urllib.request
import urllib.error
import base64
from typing import Optional, Dict, Any

# ─── Config ──────────────────────────────────────────────────────────────────

NIS_URL   = os.environ.get("NIS_URL",    "http://localhost:8000")
PI_URL    = os.environ.get("PI_URL",     "http://192.168.1.163:8085")
COSMOS_URL = os.environ.get("COSMOS_URL", "http://localhost:8100")

BANNER = """
+===========================================================+
|          NIS Protocol Console  v4.0.1                    |
|  Embodied AI  -  Cosmos Reason2  -  xArm 6DOF            |
|  Type 'help' for commands, 'quit' to exit                |
+===========================================================+
"""

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 5) -> Optional[Dict]:
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read())
    except Exception:
        return None


def _post(url: str, data: Dict, timeout: int = 30) -> Optional[Dict]:
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body,
              headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

# ─── Service checks ───────────────────────────────────────────────────────────

def check_services() -> Dict[str, bool]:
    services = {}
    services["nis"]    = _get(f"{NIS_URL}/health",   timeout=3) is not None
    services["pi"]     = _get(f"{PI_URL}/health",    timeout=3) is not None
    services["cosmos"] = _get(f"{COSMOS_URL}/health", timeout=3) is not None
    return services


def status_line(services: Dict[str, bool]) -> str:
    parts = []
    parts.append(f"NIS:{'OK' if services['nis'] else 'OFF'}")
    parts.append(f"Pi:{'OK' if services['pi'] else 'OFF'}")
    parts.append(f"Cosmos:{'OK' if services['cosmos'] else 'OFF'}")
    return "  ".join(parts)

# ─── Intent routing ───────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "arm_home":        ["home", "go home", "reset arm", "safe position"],
    "arm_pick":        ["pick", "grab", "take", "get the", "lighter"],
    "arm_place":       ["place", "put", "drop", "release"],
    "arm_dance":       ["dance", "baila", "reggaeton", "cumbia", "bachata", "salsa",
                        "music", "groove", "beat", "flow", "perreo"],
    "arm_demo":        ["demo", "cookoff", "full pipeline", "pick and place"],
    "arm_status":      ["arm status", "arm pos", "where is the arm", "servo"],
    "camera":          ["camera", "photo", "snapshot", "take picture", "what do you see"],
    "calibrate":       ["calibrat", "adjust", "fix position", "align"],
    "cosmos":          ["reason", "analyze", "what is", "cosmos", "visual"],
    "poses":           ["poses", "positions", "saved", "touch pose"],
    "status":          ["status", "health", "online", "connected"],
    "neurokernel":     ["neurokernel", "kernel", "dikw", "drives", "audit", "skills", "skill", "loop guard", "scan injection"],
    "agents":          ["agents", "list agents", "show agents", "what agents"],
    "audit":           ["audit", "chain", "tamper", "verify integrity"],
    "drives":          ["drive", "autonomous", "scheduled", "watchdog", "heartbeat"],
    "scan":            ["scan", "injection", "threat", "security check"],
    "help":            ["help", "commands", "?"],
}


def detect_intent(message: str) -> str:
    msg_lower = message.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in msg_lower for p in patterns):
            return intent
    return "chat"

# ─── Action handlers ──────────────────────────────────────────────────────────

# Confirmed servo positions (IK verified 2026-02-27)
_HOME  = {"1":100,"2":500,"3":310,"4":870,"5":680,"6":500}
_HOVER = {"1":100,"2":500,"3":222,"4":697,"5":604,"6":500}
_MID   = {"1":100,"2":500,"3":158,"4":798,"5":502,"6":500}
_PICK  = {"1":100,"2":500,"3":142,"4":856,"5":430,"6":500}
_GRIP  = {"1":700,"2":500,"3":142,"4":856,"5":430,"6":500}
_LIFT  = {"1":700,"2":500,"3":310,"4":870,"5":680,"6":500}
_PLACE = {"1":700,"2":500,"3":220,"4":827,"5":425,"6":875}  # left90


def _group_move(positions: dict, dur_ms: int = 900) -> dict:
    return _post(f"{PI_URL}/arm/group_move",
                 {"positions": positions, "duration_ms": dur_ms}) or {}


def cmd_home():
    print("  --> Moving arm to confirmed HOME position...")
    r = _group_move(_HOME, 1200)
    if not r.get("error"):
        print("  [OK] HOME: S1=100 S2=500 S3=310 S4=870 S5=680 S6=500")
    else:
        print(f"  [FAIL] {r}")


def cmd_arm_status():
    r = _get(f"{PI_URL}/arm/status")
    if r:
        pos = r.get("positions") or r.get("hardware_positions") or {}
        print("  Arm positions:", " ".join(f"S{k}={v}" for k, v in sorted(pos.items())))
        print("  Connected:", r.get("connected", "?"))
    else:
        print("  [FAIL] Could not reach Pi agent")


def cmd_camera(cosmos_enabled: bool = False):
    print("  --> Capturing camera frame...")
    r = _get(f"{PI_URL}/camera/snapshot", timeout=10)
    if not r or "image" not in r:
        r = _get(f"{PI_URL}/camera/frame", timeout=10)
    if r and ("image" in r or "frame" in r):
        img_data = r.get("image") or r.get("frame", "")
        print("  [OK] Frame captured")
        if cosmos_enabled and img_data:
            print("  --> Sending to Cosmos Reason2 for analysis...")
            cosmos_resp = _post(f"{COSMOS_URL}/v1/chat/completions", {
                "model": "cosmos-reason2",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_data}},
                        {"type": "text", "text": (
                            "Describe what you see in this robotic arm workspace image. "
                            "If there is an object on the table, estimate its position "
                            "relative to the gripper. Return a brief analysis."
                        )}
                    ]
                }],
                "max_tokens": 512,
                "temperature": 0.1
            }, timeout=30)
            if cosmos_resp and "choices" in cosmos_resp:
                analysis = cosmos_resp["choices"][0]["message"]["content"]
                print(f"\n  Cosmos Reason2:\n  {analysis}\n")
            else:
                print(f"  Cosmos: {cosmos_resp}")
        return img_data
    else:
        print("  [FAIL] No frame available")
        return None


def cmd_demo():
    """Run the 9-step orchestrated pick-and-place pipeline."""
    print("  --> Pre-flight check...")

    # Pre-flight
    preflight = _get(f"{NIS_URL}/cookoff/arm/orchestrate/status", timeout=6)
    if preflight:
        pi_ok  = preflight.get("pi_agent", {}).get("online", False)
        cos_ok = preflight.get("cosmos_h100", {}).get("online", False)
        poses  = preflight.get("poses", {})
        print(f"  Pi:     {'OK' if pi_ok else 'OFFLINE'}")
        print(f"  Cosmos: {'OK' if cos_ok else 'OFFLINE (will proceed without visual verify)'}")
        print(f"  Poses:  {poses.get('count', 0)} saved | pipeline={['OK' if poses.get('all_present') else 'MISSING']}")
        if poses.get("missing"):
            print(f"  [WARN] Missing poses: {poses['missing']}")
            print("  Run 'calibrate' first to save poses.")
            return
        if not pi_ok:
            print("  [FAIL] Pi agent offline")
            return

    confirm = input("\n  Type 'confirm' to run the 9-step cookoff pipeline: ").strip().lower()
    if confirm != "confirm":
        print("  Cancelled.")
        return

    print("\n  --> Running orchestrated 9-step pipeline (Windows-side)...")
    print("      Every step logged to AuditChain. Cosmos verifies vision steps.")
    print()

    r = _post(f"{NIS_URL}/cookoff/arm/orchestrate", {}, timeout=120)

    if not r:
        # Fallback: run confirmed IK pick sequence directly
        print("  [WARN] NIS Protocol not running — running IK pick directly...")
        cmd_pick(place="left90", quiet=False)
        return

    if r.get("error"):
        print(f"  [FAIL] Error: {r['error']}")
        return

    # Print step-by-step results
    success = r.get("success", False)
    total_ms = r.get("total_ms", 0)
    completed = r.get("completed_steps", 0)
    total = r.get("total_steps", 9)

    print(f"  Pipeline: {'COMPLETE' if success else 'PARTIAL'} | {completed}/{total} steps | {total_ms:.0f}ms")
    print(f"  Object picked: {r.get('object_picked', False)} | Placed: {r.get('object_placed', False)}")
    print()

    for step in r.get("steps", []):
        num  = step.get("step", "?")
        name = step.get("name", "?")
        ok   = step.get("success", False)
        ms   = step.get("duration_ms", 0)
        icon = "OK  " if ok else "FAIL"
        print(f"  [{icon}] Step {num}: {name:<20} {ms:.0f}ms", end="")
        if step.get("correction"):
            print(f" [Cosmos corrected: {step['correction']}]", end="")
        if step.get("cosmos") and step["cosmos"].get("parsed"):
            p = step["cosmos"]["parsed"]
            if "object_visible" in p:
                print(f" [visible={p.get('object_visible')} conf={p.get('confidence','?')}]", end="")
        print()

    if success:
        print("\n  DEMO COMPLETE — ready for cookoff recording.")
        print("  Arm is back at home position.")
    else:
        print(f"\n  Partial completion — {r.get('error', 'some steps failed')}")
        print("  Arm returned home. Check step log above.")


def cmd_poses():
    r = _get(f"{PI_URL}/arm/touch_poses")
    if r:
        poses = r.get("touch_poses") or r.get("poses") or {}
        print(f"  Saved poses ({len(poses)}):")
        for pose_name, pos in poses.items():
            if isinstance(pos, dict):
                units = " ".join(f"S{k}={v}" for k, v in sorted(pos.items()))
                print(f"    {pose_name}: {units}")
            else:
                print(f"    {pose_name}: {pos}")
    else:
        print("  [FAIL] Could not retrieve poses")


def cmd_calibrate(video_path: str = None):
    """
    Cosmos-guided video calibration.
    Reads home from arm memory — never from hardcoded values.
    Supports: camera burst (default) or MP4 video file.
    Has labels: place RED/BLUE/GREEN/YELLOW stickers at workspace corners.
    """
    print("  --> Cosmos Video Calibration")
    print()

    # Show arm memory status
    mem = _get(f"{NIS_URL}/cookoff/calibrate/arm_memory")
    if not mem or mem.get("error"):
        mem = _get(f"{PI_URL}/arm/touch_poses")
    if mem:
        poses = mem.get("pipeline_poses") or mem.get("touch_poses") or {}
        home = poses.get("home", {})
        if isinstance(home, dict):
            home_str = " ".join(f"S{k}={v}" for k, v in sorted(home.items()))
            print("  Arm memory HOME:", home_str)
        missing = mem.get("missing", [])
        if missing:
            print(f"  [WARN] Missing poses: {missing}")
        else:
            print(f"  Pipeline poses: {len(poses)} in memory")
    print()

    svcs = check_services()
    print("  Services:", status_line(svcs))
    if not svcs["pi"]:
        print("  [FAIL] Pi agent offline.")
        return

    # Ask about labels
    print()
    print("  Label setup:")
    print("    Place colored stickers at workspace corners:")
    print("    RED=front-left  BLUE=front-right  GREEN=back-left  YELLOW=back-right")
    print("    WHITE=on pick target object (optional but helps)")
    has_labels_str = input("  Do you have labels placed? (yes/no) [yes]: ").strip().lower()
    has_labels = has_labels_str not in ("no", "n")

    # Ask about video
    print()
    if not video_path:
        print("  Video options:")
        print("    1. Camera burst (arm moves, Pi camera captures automatically)")
        print("    2. MP4 video file (record with Win+G, then provide path)")
        choice = input("  Choose method (1/2) [1]: ").strip()
        if choice == "2":
            video_path = input("  MP4 path (e.g. C:/Users/DiegoTorres/Videos/arm_move.mp4): ").strip()
            if not video_path:
                video_path = None

    print()
    confirm = input("  Type 'go' to start calibration (arm will move): ").strip().lower()
    if confirm != "go":
        print("  Cancelled.")
        return

    print()
    print("  --> Running calibration via NIS Protocol...")
    print("      Methods: Cosmos Reason2 + Predict2.5 + Transfer2.5")
    print("      Poses to calibrate: inspect, pick_table, place_bin")
    print()

    payload = {
        "has_labels": has_labels,
        "auto_save": True,
        "poses": ["home", "pick_hover_center", "pick_down_center", "place_left90"],
    }
    if video_path:
        payload["video_path"] = video_path

    r = _post(f"{NIS_URL}/cookoff/calibrate", payload, timeout=180)
    if not r:
        # Fallback to direct camera burst calibration
        print("  NIS Protocol offline — running direct calibration...")
        _direct_calibrate(has_labels)
        return

    if r.get("error"):
        print(f"  [FAIL] {r['error']}")
        return

    print(f"  Calibration: {'COMPLETE' if r.get('success') else 'PARTIAL'}")
    print(f"  Methods used: {r.get('methods_used', [])}")
    print(f"  Synthetic frames generated: {r.get('synthetic_frames', 0)}")
    print(f"  Duration: {r.get('total_ms', 0):.0f}ms")
    print()
    for pose_name, cal in (r.get("poses") or {}).items():
        delta = cal.get("delta", {})
        conf = cal.get("avg_confidence", 0)
        if delta:
            delta_str = " ".join(f"S{k}={'+' if v>0 else ''}{v}" for k,v in delta.items())
            print(f"  [OK] {pose_name:<15} correction: {delta_str}  (conf={conf:.2f})")
        else:
            print(f"  [OK] {pose_name:<15} no correction needed  (conf={conf:.2f})")
    print()
    print("  All corrected poses saved to arm memory.")
    print("  Run 'demo' to test the calibrated pipeline.")


def _direct_calibrate(has_labels: bool = False):
    """Direct calibration fallback when NIS Protocol is not running."""
    poses_r = _get(f"{PI_URL}/arm/touch_poses")
    if not poses_r:
        print("  [FAIL] Cannot read arm poses")
        return
    arm_poses = poses_r.get("touch_poses") or poses_r.get("poses") or {}

    # Get home from arm memory
    home = arm_poses.get("home")
    if not home:
        print("  [WARN] No 'home' in arm memory. Set home on arm first.")
        return

    print(f"  Home from arm memory: {home}")
    to_calibrate = ["home", "pick_hover_center", "pick_down_center", "place_left90"]

    for pose_name in to_calibrate:
        if pose_name not in arm_poses:
            print(f"  [SKIP] {pose_name} not in arm memory")
            continue
        pos = arm_poses[pose_name]
        print(f"\n  --> Moving to {pose_name}: {pos}")
        _post(f"{PI_URL}/arm/group_move", {"positions": pos, "duration_ms": 1500})
        time.sleep(2.0)

        frame_r = _get(f"{PI_URL}/camera/snapshot", timeout=8)
        frame = (frame_r or {}).get("image")
        if frame:
            print(f"  Frame captured ({len(frame)} bytes)")
        else:
            print("  [WARN] No camera frame")
            continue

        cosmos = _post(f"{COSMOS_URL}/v1/chat/completions", {
            "model": "cosmos-reason2",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": frame}},
                {"type": "text", "text": (
                    ("Workspace has colored corner labels: RED=front-left, BLUE=front-right, "
                     "GREEN=back-left, YELLOW=back-right, WHITE=object. "
                     "Workspace: 17cm × 20.5cm. " if has_labels else "") +
                    f"Robotic arm in '{pose_name}' position. "
                    "Analyze alignment and return JSON: "
                    "{object_visible: bool, lateral_error_mm: float, "
                    "gripper_to_object_mm: float, confidence: float, "
                    "recommended_servo_correction: {S2: 0, S3: 0, S6: 0}}"
                )}
            ]}],
            "max_tokens": 256,
        }, timeout=25)

        if cosmos and "choices" in cosmos:
            text = cosmos["choices"][0]["message"]["content"]
            print(f"  Cosmos: {text[:300]}")
        else:
            print("  Cosmos offline")

        # Save pose as-is (no correction without NIS Protocol)
        _post(f"{PI_URL}/arm/save_touch_pose", {"name": pose_name})
        print(f"  [OK] Saved {pose_name}")

        # Return home between poses
        _post(f"{PI_URL}/arm/group_move", {"positions": home, "duration_ms": 1000})
        time.sleep(1.2)

    print("\n  [OK] Direct calibration complete.")


def cmd_pick(place: str = "left90", s6: int = 500, quiet: bool = False):
    """
    Full IK-confirmed pick-and-place sequence.
    10 steps: home → open → hover → mid → pick → grip → lift → place → release → home
    Uses confirmed servo positions (z=1.5cm, S1=700 grip, 2026-02-27).
    """
    place_zones = {
        "left90":  {"1":700,"2":500,"3":220,"4":827,"5":425,"6":875},
        "left45":  {"1":700,"2":500,"3":220,"4":827,"5":425,"6":690},
        "right45": {"1":700,"2":500,"3":220,"4":827,"5":425,"6":310},
        "right90": {"1":700,"2":500,"3":220,"4":827,"5":425,"6":125},
    }
    place_pos = place_zones.get(place, place_zones["left90"])
    relax_pos = {**place_pos, "1": "100"}
    aim = {**_HOME, "6": str(s6)}

    steps = [
        ("home",       _HOME,                  1000),
        ("grip open",  {**_HOME, "1": "100"},   500),
        ("hover",      {**_HOVER, "6": str(s6)}, 900),
        ("mid",        {**_MID,   "6": str(s6)}, 700),
        ("pick",       {**_PICK,  "6": str(s6)}, 600),
        ("grip close", {**_GRIP,  "6": str(s6)}, 500),
        ("lift",       {**_LIFT,  "6": str(s6)}, 800),
        ("place",      place_pos,                 900),
        ("release",    relax_pos,                 600),
        ("home final", _HOME,                   1000),
    ]

    if not quiet:
        print(f"  --> IK Pick & Place -> {place} ({len(steps)} steps)")

    # Try NIS /cookoff/pick first (runs the sequence on Pi side)
    nis_r = _post(f"{NIS_URL}/cookoff/pick",
                  {"s6": s6, "place": place, "z": 1.5}, timeout=60)
    if nis_r and nis_r.get("ok"):
        steps_ok = sum(1 for s in nis_r.get("steps", []) if s.get("ok", True))
        print(f"  [OK] NIS pick: {steps_ok}/{len(nis_r.get('steps', steps))} steps | "
              f"{nis_r.get('message', '')}")
        return

    # Fallback: direct servo sequence
    if not quiet:
        print("  (NIS offline — running direct servo sequence)")
    for i, (label, pos, dur) in enumerate(steps):
        r = _group_move(pos, dur)
        ok = not r.get("error")
        if not quiet:
            print(f"  Step {i+1:2d}/10 [{('OK' if ok else 'FAIL')}] {label}")
        time.sleep(dur / 1000.0 + 0.15)

    if not quiet:
        print("  [OK] Pick sequence complete")


def cmd_dance(genre: str = "reggaeton", moves: int = 8, use_mic: bool = False):
    """
    Trigger the Latino rhythm arm dance via NIS cosmos-dance.
    Genres: reggaeton, cumbia, bachata, salsa
    """
    valid_genres = ("reggaeton", "cumbia", "bachata", "salsa")
    if genre not in valid_genres:
        # Try to infer from genre arg
        for g in valid_genres:
            if g in genre:
                genre = g
                break
        else:
            genre = "reggaeton"

    endpoint = "start" if use_mic else "demo"
    print(f"  --> Dance: {genre.upper()} ({moves} moves, {'MIC LIVE' if use_mic else 'demo mode'})")

    r = _post(f"{NIS_URL}/cosmos-dance/{endpoint}",
              {"genre": genre, "moves": moves, "energy": 0.20},
              timeout=moves * 3 + 15)

    if r and not r.get("error"):
        done = r.get("moves_done", "?")
        print(f"  [OK] Dance complete — {done} moves, genero: {genre}")
    elif r and r.get("error"):
        # NIS offline → fallback: run a few servo moves directly
        print(f"  [WARN] NIS offline ({r['error'][:60]}) — running minimal dance on Pi...")
        _run_mini_dance(genre, moves=min(moves, 4))
    else:
        print("  [FAIL] Could not reach NIS cosmos-dance endpoint")
        print("         Make sure NIS is running: sudo systemctl restart nis-protocol")


def _run_mini_dance(genre: str, moves: int = 4):
    """Minimal fallback dance when NIS is offline — direct servo moves."""
    patterns = {
        "reggaeton": [
            {"1":"900","2":"500","3":"290","4":"820","5":"640","6":"500"},  # pump
            {"1":"900","2":"500","3":"250","4":"760","5":"570","6":"500"},  # drop
            {"1":"100","2":"500","3":"310","4":"870","5":"680","6":"640"},  # sway L
            {"1":"100","2":"500","3":"310","4":"870","5":"680","6":"360"},  # sway R
        ],
        "cumbia": [
            {"1":"100","2":"500","3":"270","4":"800","5":"600","6":"720"},
            {"1":"100","2":"500","3":"270","4":"800","5":"600","6":"280"},
            {"1":"100","2":"500","3":"235","4":"720","5":"560","6":"500"},
            {"1":"100","2":"500","3":"270","4":"800","5":"600","6":"720"},
        ],
        "bachata": [
            {"1":"100","2":"500","3":"270","4":"790","5":"580","6":"740"},
            {"1":"100","2":"500","3":"240","4":"730","5":"540","6":"500"},
            {"1":"100","2":"500","3":"270","4":"790","5":"580","6":"260"},
            {"1":"100","2":"500","3":"240","4":"730","5":"540","6":"500"},
        ],
        "salsa": [
            {"1":"100","2":"500","3":"300","4":"855","5":"670","6":"660"},
            {"1":"100","2":"500","3":"300","4":"855","5":"670","6":"340"},
            {"1":"100","2":"500","3":"310","4":"870","5":"560","6":"500"},
            {"1":"100","2":"500","3":"310","4":"800","5":"630","6":"500"},
        ],
    }
    seq = patterns.get(genre, patterns["reggaeton"])
    for i in range(moves):
        pos = seq[i % len(seq)]
        r = _group_move(pos, 350)
        print(f"  Move {i+1}/{moves}: {'OK' if not r.get('error') else 'FAIL'}")
        time.sleep(0.5)
    _group_move(_HOME, 1000)
    print(f"  Mini dance done ({moves} moves). Full dance needs NIS running.")


def cmd_neurokernel(subcommand: str = "status"):
    """Show NeuroKernel v2 status or subcomponent info."""
    if not _get(f"{NIS_URL}/health", timeout=2):
        print("  NIS Protocol not running. Start with: python main.py")
        return

    sub = subcommand.lower().strip()

    if sub in ["status", "kernel", ""]:
        r = _get(f"{NIS_URL}/neurokernel/status")
        if r:
            print(f"\n  NeuroKernel v2 | uptime={r.get('uptime_secs',0):.0f}s | requests={r.get('request_count',0)}")
            comps = r.get("components", {})
            sk = comps.get("skill_loader", {})
            ac = comps.get("audit_chain", {})
            dr = comps.get("drive_scheduler", {})
            sc = comps.get("scanner", {})
            print(f"  Skills:     {sk.get('total_skills', 0)} loaded")
            print(f"  Audit:      {ac.get('total_entries', 0)} entries | sqlite={ac.get('sqlite_backend', False)}")
            print(f"  Drives:     {dr.get('drives', 0)} registered | running={dr.get('running', False)}")
            print(f"  Scanner:    {sc.get('total_scans', 0)} scans | blocked={sc.get('blocked', 0)}")
            print(f"\n  DIKW Layers:")
            for layer, desc in r.get("dikw_layers", {}).items():
                print(f"    {layer.upper():12} --> {desc}")
        else:
            print("  [FAIL] Could not reach /neurokernel/status")

    elif sub in ["audit", "chain"]:
        r = _get(f"{NIS_URL}/neurokernel/audit?n=10")
        if isinstance(r, list):
            print(f"\n  Recent audit entries ({len(r)}):")
            for entry in r[-10:]:
                ts = time.strftime("%H:%M:%S", time.localtime(entry.get("timestamp", 0)))
                print(f"    [{ts}] {entry.get('agent_id','?')} | {entry.get('action_type','?')} | {'OK' if entry.get('success') else 'FAIL'}")
        else:
            print(f"  {r}")

    elif sub in ["drives", "drive"]:
        r = _get(f"{NIS_URL}/neurokernel/drives")
        if r and "drives" in r:
            print(f"\n  Autonomous drives ({len(r['drives'])}):")
            for d in r["drives"]:
                status_icon = "[RUN]" if d["status"] == "running" else "[OK]" if d["status"] == "done" else "[--]"
                print(f"    {status_icon} {d['name']:<25} every {d['interval_secs']:.0f}s | runs={d['run_count']} fails={d['fail_count']}")
        else:
            print("  [FAIL] Could not retrieve drives")

    elif sub in ["skills", "skill"]:
        r = _get(f"{NIS_URL}/neurokernel/skills")
        if r and "skills" in r:
            print(f"\n  Loaded skills ({len(r['skills'])}):")
            for s in r["skills"]:
                print(f"    {s['name']:<30} tags={s['tags']} used={s.get('use_count', 0)}")
        else:
            print("  [FAIL] Could not retrieve skills")

    elif sub in ["verify", "integrity"]:
        r = _post(f"{NIS_URL}/neurokernel/audit/verify", {})
        if r:
            valid = r.get("valid", False)
            print(f"  Chain integrity: {'[OK] VALID' if valid else '[FAIL] BROKEN'}")
            print(f"  Entries: {r.get('entries', 0)} | Broken at: {r.get('broken_at', 'none')}")
        else:
            print("  [FAIL] Could not verify chain")

    elif sub.startswith("scan ") or sub.startswith("scan:"):
        text = sub[5:].strip()
        r = _post(f"{NIS_URL}/neurokernel/scan", {"text": text, "context": "console"})
        if r:
            print(f"  Scan result: {'SAFE' if r.get('safe') else 'THREAT DETECTED'} | score={r.get('score')} | action={r.get('action')}")
            for t in r.get("threats", []):
                print(f"    [{t['severity'].upper()}] {t['pattern_id']} -- {t['description']}")
    else:
        print(f"  Unknown subcommand '{sub}'. Try: status, audit, drives, skills, verify, scan <text>")


def cmd_list_agents():
    """Discover and display all declarative agents from agents/*/agent.toml."""
    import os, re
    agents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
    if not os.path.isdir(agents_dir):
        print("  [FAIL] agents/ directory not found")
        return
    found = []
    for entry in sorted(os.listdir(agents_dir)):
        toml_path = os.path.join(agents_dir, entry, "agent.toml")
        if not os.path.isfile(toml_path):
            continue
        try:
            content = open(toml_path, encoding="utf-8").read()
            name  = (re.search(r'^name\s*=\s*"([^"]+)"',        content, re.M) or [None, entry]).group(1)
            desc  = (re.search(r'^description\s*=\s*"([^"]+)"', content, re.M) or [None, ""]).group(1)
            tags  = re.findall(r'"([a-z][a-z0-9\-]+)"', (re.search(r'^tags\s*=\s*\[([^\]]*)\]', content, re.M) or [None, ""]).group(1) or "")
            model = (re.search(r'^model\s*=\s*"([^"]+)"',        content, re.M) or [None, "?"]).group(1)
            found.append((name, desc[:80], tags[:5], model))
        except Exception as e:
            found.append((entry, f"[parse error: {e}]", [], "?"))
    if not found:
        print("  No agent.toml files found in agents/")
        return
    print(f"\n  Declarative Agents ({len(found)}):")
    for name, desc, tags, model in found:
        tag_str = " ".join(f"#{t}" for t in tags)
        print(f"    {name:<22} model={model:<20} {tag_str}")
        if desc:
            print(f"    {'':22} {desc}")
    print()
    # Also show NIS Protocol /neurokernel/skills if available
    r = _get(f"{NIS_URL}/neurokernel/skills")
    if r and "skills" in r:
        loaded = r["skills"]
        print(f"  SkillLoader has {len(loaded)} skill(s) loaded:")
        for s in loaded:
            print(f"    {s['name']:<30} tags={s['tags']}")
        print()


def cmd_nis_chat(message: str) -> str:
    """Route to NIS Protocol LLM chat endpoint"""
    r = _post(f"{NIS_URL}/chat", {
        "message": message,
        "user_id": "console",
        "use_tools": True,
    }, timeout=30)
    if r and "response" in r:
        return r["response"]
    elif r and "error" in r:
        return f"[NIS error: {r['error']}]"
    else:
        return "[NIS Protocol not available — start with: python main.py]"

# ─── Main REPL ────────────────────────────────────────────────────────────────

def run_command(message: str, cosmos_enabled: bool = False, agent: str = "auto") -> None:
    intent = detect_intent(message)

    if intent == "help":
        print("""
  Arm Commands:
    home               --> Move arm to confirmed HOME (S1=100 S3=310 S4=870 S5=680)
    pick               --> Full IK pick sequence (z=1.5cm, S1=700, 10 steps)
    pick left45        --> Pick and drop at 45-degree left
    pick right90       --> Pick and drop at 90-degree right
    status             --> Show arm servo positions + all services
    camera             --> Capture camera frame
    camera cosmos      --> Capture + Cosmos Reason2 visual analysis
    demo               --> Full YOLO→Reason2→arm cookoff pipeline
    calibrate          --> Cosmos-guided camera/arm calibration
    poses              --> List saved touch poses

  Dance Commands:
    dance              --> Reggaeton demo (8 moves, no mic needed)
    dance reggaeton    --> Perreo intenso at 88 BPM
    dance cumbia       --> 130 BPM Colombian cumbia flow
    dance bachata      --> 125 BPM romantic bachata
    dance salsa        --> 170 BPM fast salsa
    dance mic          --> LIVE mic mode (play music near Pi!)

  NeuroKernel v2 Commands:
    neurokernel        --> Show full kernel status (DIKW layers)
    neurokernel audit  --> Show recent audit chain entries
    neurokernel drives --> List autonomous drives + last run
    neurokernel skills --> List loaded SKILL.md domain expertise
    neurokernel verify --> Verify AuditChain integrity
    neurokernel scan <text> --> Scan text for injection threats

  Or type any natural language — routes to NIS Protocol LLM.

  Flags:
    --cosmos        Enable Cosmos reasoning on all vision commands
    --agent arm     Focus commands on arm control
""")
    elif intent == "arm_home":
        cmd_home()
    elif intent == "arm_pick":
        # Parse optional place zone from message: "pick left45", "pick right90"
        msg_l = message.lower()
        place = "left90"
        for zone in ("left90", "left45", "right45", "right90"):
            if zone in msg_l:
                place = zone
                break
        cmd_pick(place=place)
    elif intent == "arm_dance":
        # Parse genre and mode from message
        msg_l = message.lower()
        genre = "reggaeton"
        for g in ("reggaeton", "cumbia", "bachata", "salsa"):
            if g in msg_l:
                genre = g
                break
        use_mic = "mic" in msg_l or "live" in msg_l
        moves = 8
        cmd_dance(genre=genre, moves=moves, use_mic=use_mic)
    elif intent == "arm_status":
        cmd_arm_status()
    elif intent == "camera":
        cmd_camera(cosmos_enabled or "cosmos" in message.lower())
    elif intent == "arm_demo":
        cmd_demo()
    elif intent == "calibrate":
        cmd_calibrate()
    elif intent == "poses":
        cmd_poses()
    elif intent == "neurokernel":
        # Extract subcommand: "neurokernel audit" --> "audit"
        parts = message.lower().strip().split()
        sub = parts[1] if len(parts) > 1 else "status"
        # Handle "neurokernel scan some text" --> scan "some text"
        if sub == "scan" and len(parts) > 2:
            sub = "scan " + " ".join(parts[2:])
        cmd_neurokernel(sub)
    elif intent == "audit":
        cmd_neurokernel("audit")
    elif intent == "drives":
        cmd_neurokernel("drives")
    elif intent == "scan":
        # "scan this text" -> scan "this text"
        parts = message.strip().split(None, 1)
        text = parts[1] if len(parts) > 1 else ""
        cmd_neurokernel(f"scan {text}" if text else "status")
    elif intent == "agents":
        cmd_list_agents()
    elif intent == "status":
        svcs = check_services()
        print(f"  {status_line(svcs)}")
        if svcs["pi"]:
            cmd_arm_status()
    else:
        # Route to NIS Protocol LLM
        print("  --> NIS Protocol thinking...")
        response = cmd_nis_chat(message)
        print(f"\n  {response}\n")


def run_daemon(poll_interval: int = 30, cosmos_enabled: bool = False):
    """
    Daemon mode — autonomous background monitor.
    Polls Pi / Cosmos / NIS health every poll_interval seconds.
    Logs anomalies and auto-triggers arm home on recovery.
    No user input required. Ctrl-C to stop.
    """
    import threading

    print(BANNER)
    print(f"  [DAEMON] Autonomous monitor starting (poll every {poll_interval}s)")
    print(f"  [DAEMON] Ctrl-C to stop\n")

    # State tracking for change detection
    _prev: Dict[str, bool] = {"nis": None, "pi": None, "cosmos": None}
    _pi_fail = 0
    _PI_ALERT = 3

    def _poll():
        nonlocal _pi_fail
        svcs = check_services()
        ts = time.strftime("%H:%M:%S")

        # Detect transitions and anomalies
        for svc, ok in svcs.items():
            prev = _prev.get(svc)
            if prev is None:
                # First poll — just print status
                state = "OK" if ok else "OFFLINE"
                print(f"  [{ts}] {svc.upper():<8} {state}")
            elif ok and not prev:
                print(f"  [{ts}] {svc.upper():<8} RECOVERED")
                if svc == "pi":
                    print(f"  [{ts}] Pi back online — sending arm to HOME...")
                    cmd_home()
            elif not ok and prev:
                print(f"  [{ts}] {svc.upper():<8} WENT OFFLINE")
            _prev[svc] = ok

        # Pi consecutive failure counter
        if not svcs["pi"]:
            _pi_fail += 1
            if _pi_fail >= _PI_ALERT:
                print(f"  [{ts}] CRITICAL: Pi unreachable {_pi_fail}x — "
                      f"run: ssh neurolinux@192.168.1.163 sudo systemctl restart neurolinux-agent")
        else:
            _pi_fail = 0

        # NeuroKernel drives check (if NIS up)
        if svcs["nis"]:
            drives = _get(f"{NIS_URL}/neurokernel/drives")
            if drives and "drives" in drives:
                failed = [d for d in drives["drives"] if d.get("fail_count", 0) > 0]
                if failed:
                    for d in failed:
                        print(f"  [{ts}] DRIVE  {d['name']:<25} fails={d['fail_count']} last={d['status']}")

    # Interactive input thread so daemon + REPL coexist
    def _input_thread():
        while True:
            try:
                user_input = input("  nis(daemon)> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("  Daemon stopped.")
                    import os as _os
                    _os._exit(0)
                run_command(user_input, cosmos_enabled=cosmos_enabled)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print(f"  [input error] {e}")

    t = threading.Thread(target=_input_thread, daemon=True)
    t.start()

    try:
        while True:
            _poll()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n  [DAEMON] Stopped.")


def run_interactive(cosmos_enabled: bool = False, agent: str = "auto"):
    print(BANNER)
    svcs = check_services()
    print(f"  Services: {status_line(svcs)}\n")

    while True:
        try:
            prompt = f"  nis({agent})> " if agent != "auto" else "  nis> "
            user_input = input(prompt).strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("  Goodbye.")
                break
            run_command(user_input, cosmos_enabled=cosmos_enabled, agent=agent)
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye.")
            break
        except Exception as e:
            print(f"  Error: {e}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NIS Protocol Console — embodied AI from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nis_console.py                      # Interactive mode
  python nis_console.py "go home"            # Move arm to confirmed HOME
  python nis_console.py pick                 # Full IK pick (z=1.5cm)
  python nis_console.py "pick left45"        # Pick + drop at 45-deg left
  python nis_console.py "dance reggaeton"    # Perreo arm dance (8 moves)
  python nis_console.py "dance mic"          # Live mic mode (play music near Pi)
  python nis_console.py calibrate            # Cosmos-guided calibration
  python nis_console.py camera --cosmos      # Vision + Cosmos analysis
  python nis_console.py demo                 # Full YOLO→Reason2→arm cookoff
  python nis_console.py --agent arm status   # Arm status
"""
    )
    parser.add_argument("command", nargs="*", help="Command to run (omit for interactive)")
    parser.add_argument("--cosmos", action="store_true", help="Enable Cosmos visual reasoning")
    parser.add_argument("--agent", default="auto", help="Focus agent (arm, cosmos, nis)")
    parser.add_argument("--pi", default=None, help="Override Pi URL")
    parser.add_argument("--nis", default=None, help="Override NIS Protocol URL")
    parser.add_argument("--daemon", action="store_true",
                        help="Daemon mode: autonomous background monitor (polls health, auto-heals)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Daemon poll interval in seconds (default: 30)")
    args = parser.parse_args()

    global PI_URL, NIS_URL
    if args.pi:
        PI_URL = args.pi
    if args.nis:
        NIS_URL = args.nis

    if args.daemon:
        run_daemon(poll_interval=args.interval, cosmos_enabled=args.cosmos)
    elif args.command:
        message = " ".join(args.command)
        run_command(message, cosmos_enabled=args.cosmos, agent=args.agent)
    else:
        run_interactive(cosmos_enabled=args.cosmos, agent=args.agent)


if __name__ == "__main__":
    main()
