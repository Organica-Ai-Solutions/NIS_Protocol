#!/usr/bin/env python3
"""
xArm Direct Control Test
=========================
Run after Pi is up and xArm is detected.
Tests: home position → wave → pick position → open/close gripper
"""

import urllib.request
import json
import time
import sys

NIS_URL = "http://192.168.1.160:8000"
PI_URL  = "http://NeuroLinux.local"

RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"; BOLD = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")

def post(url, body, timeout=15):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    r = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(r.read())

def get(url, timeout=8):
    r = urllib.request.urlopen(url, timeout=timeout)
    return json.loads(r.read())

print(f"\n{BOLD}{CYAN}{'═'*50}")
print("  xArm Embodiment Test — NeuroLinux Pi")
print(f"{'═'*50}{RESET}\n")

# Find xArm port on Pi
print("Searching for xArm on Pi...")
xarm_port = None
try:
    d = get(f"{PI_URL}:8080/devices", timeout=8)
    for dev in d.get("devices", []):
        if "xarm" in dev.get("type","").lower() or "hiwonder" in dev.get("model","").lower():
            xarm_port = dev.get("port")
            ok(f"xArm found: {xarm_port} — {dev.get('status','?')}")
            break
    if not xarm_port:
        print(f"  {RED}xArm not found. Plug in USB and wait 10 seconds.{RESET}")
        sys.exit(1)
except Exception as e:
    err(f"Cannot reach Pi: {e}")
    print("  Make sure Pi is booted and on WiFi")
    sys.exit(1)

MOVES = [
    ("Home position",    {"servo_id": "all", "angle": 0,  "speed": 500}),
    ("Wave up",          {"servo_id": 2,     "angle": 90, "speed": 800}),
    ("Wave down",        {"servo_id": 2,     "angle": 45, "speed": 800}),
    ("Wave up again",    {"servo_id": 2,     "angle": 90, "speed": 800}),
    ("Open gripper",     {"servo_id": 6,     "angle": 90, "speed": 500}),
    ("Close gripper",    {"servo_id": 6,     "angle": 0,  "speed": 500}),
    ("Return home",      {"servo_id": "all", "angle": 0,  "speed": 500}),
]

print(f"\nRunning xArm sequence ({len(MOVES)} moves)...\n")
for name, params in MOVES:
    try:
        result = post(f"{PI_URL}:8080/xarm/move", {
            "port": xarm_port,
            **params,
        }, timeout=10)
        ok(f"{name} — {result.get('status', 'ok')}")
        time.sleep(0.8)
    except Exception as e:
        err(f"{name}: {e}")

print(f"\n{BOLD}{GREEN}xArm test complete!{RESET}")
print(f"\nLive camera stream: http://NeuroLinux.local:8009/stream")
print(f"NeuroHub Dashboard: http://NeuroLinux.local:3000\n")
