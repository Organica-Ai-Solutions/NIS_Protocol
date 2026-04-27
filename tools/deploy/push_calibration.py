#!/usr/bin/env python3
"""
push_calibration.py — Deploy calibrated arm positions to Pi without SSH.
=========================================================================
Reads servo_calibration_result.json and pushes all poses to the Pi agent
via POST /arm/load_calibration.  No SSH required.

Usage
-----
  python push_calibration.py                         # uses default Pi IP
  python push_calibration.py --pi 192.168.1.163
  python push_calibration.py --pi 192.168.1.163 --port 8085
  python push_calibration.py --file my_calibration.json
  python push_calibration.py --verify                # just print current poses

The Pi stores poses in /opt/neurolinux/touch_poses.json persistently.
"""

import argparse
import json
import sys
import time
import pathlib

try:
    import requests
except ImportError:
    print("[!] 'requests' not installed. Run: pip install requests")
    sys.exit(1)

DEFAULT_PI   = "192.168.1.163"
DEFAULT_PORT = 8085
CAL_FILE     = pathlib.Path(__file__).parent / "servo_calibration_result.json"


def load_calibration(filepath: pathlib.Path) -> dict:
    """Load calibration JSON, stripping comment fields."""
    data = json.loads(filepath.read_text())
    poses = {}
    for name, val in data.items():
        if name.startswith("_"):
            continue                    # skip comment fields
        if not isinstance(val, dict):
            continue
        # strip inline 'note' keys, keep only numeric servo IDs
        pose = {k: int(v) for k, v in val.items()
                if k.isdigit() or (isinstance(k, str) and k.lstrip("-").isdigit())}
        if pose:
            poses[name] = pose
    return poses


def push(pi_ip: str, port: int, poses: dict) -> bool:
    url = f"http://{pi_ip}:{port}/arm/load_calibration"
    print(f"\n  Pushing {len(poses)} poses to {url}")
    print(f"  Poses: {list(poses.keys())}")
    try:
        r = requests.post(url, json={"poses": poses}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"\n  [OK] Loaded {data['total_poses']} total poses on Pi")
            print(f"       Newly set: {data['loaded']}")
            print(f"       All poses: {data['all_poses']}")
            return True
        else:
            print(f"\n  [!] HTTP {r.status_code}: {r.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"\n  [!] Cannot connect to Pi at {pi_ip}:{port}")
        print(f"      Is the NeuroLinux agent running?  sudo systemctl status neurolinux-agent")
        return False
    except Exception as e:
        print(f"\n  [!] Error: {e}")
        return False


def verify(pi_ip: str, port: int) -> None:
    url = f"http://{pi_ip}:{port}/arm/touch_poses"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"\n  Current touch poses on Pi ({data['count']} total):")
            for name, vals in data.get("touch_poses", {}).items():
                print(f"    {name:15s}: {vals}")
        else:
            print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  [!] {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Push arm calibration to Pi via HTTP")
    p.add_argument("--pi",     default=DEFAULT_PI,  help=f"Pi IP address (default: {DEFAULT_PI})")
    p.add_argument("--port",   default=DEFAULT_PORT, type=int,
                   help=f"Agent port (default: {DEFAULT_PORT})")
    p.add_argument("--file",   default=str(CAL_FILE), help="Calibration JSON file")
    p.add_argument("--verify", action="store_true",
                   help="Only read and display current Pi poses (no push)")
    p.add_argument("--dry-run", action="store_true",
                   help="Load calibration file and print poses without pushing")
    args = p.parse_args()

    print(f"\n  NIS Protocol — Calibration Pusher")
    print(f"  Pi: {args.pi}:{args.port}")
    print(f"  File: {args.file}")

    if args.verify:
        verify(args.pi, args.port)
        return

    cal_path = pathlib.Path(args.file)
    if not cal_path.exists():
        print(f"\n  [!] Calibration file not found: {cal_path}")
        sys.exit(1)

    poses = load_calibration(cal_path)
    print(f"\n  Loaded {len(poses)} poses from file:")
    for name, vals in poses.items():
        print(f"    {name:15s}: {vals}")

    if args.dry_run:
        print("\n  [dry-run] No changes pushed.")
        return

    ok = push(args.pi, args.port, poses)
    if ok:
        print("\n  Verifying Pi state after push...")
        time.sleep(0.5)
        verify(args.pi, args.port)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
