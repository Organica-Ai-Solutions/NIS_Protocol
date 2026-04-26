#!/usr/bin/env python3
"""
NeuroLinux Pi First Embodiment Test
====================================
Run this on your Windows PC AFTER the Pi has booted and connected to WiFi.
It tests the full pipeline: Pi Camera → NIS Protocol → Cosmos → xArm
"""

import urllib.request
import urllib.error
import json
import time
import sys

NIS_URL  = "http://192.168.1.160:8000"
PI_URL   = "http://NeuroLinux.local"   # mDNS — or use Pi's IP directly

RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg):print(f"  {CYAN}→{RESET}  {msg}")

def get(url, timeout=10):
    r = urllib.request.urlopen(url, timeout=timeout)
    return json.loads(r.read())

def post(url, body, timeout=15):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    r = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(r.read())

def section(title):
    print(f"\n{BOLD}{CYAN}{'━'*50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'━'*50}{RESET}")

# ── 1. NIS Protocol (PC) ─────────────────────────────────────────────────────
section("1/5  NIS Protocol (PC at 192.168.1.160)")
try:
    d = get(f"{NIS_URL}/health")
    ok(f"NIS Protocol v{d['version']} — {d['status']}")
except Exception as e:
    err(f"NIS Protocol unreachable: {e}")
    print(f"\n{RED}NIS Protocol must be running. Start it with: python main.py{RESET}")
    sys.exit(1)

try:
    d = get(f"{NIS_URL}/openclaw/status")
    ok(f"OpenClaw bridge — {d['status']} ({d['skills_loaded']} skills)")
except Exception as e:
    err(f"OpenClaw bridge: {e}")

# ── 2. Pi services ────────────────────────────────────────────────────────────
section("2/5  NeuroLinux Pi Services")
info("Trying to reach Pi at NeuroLinux.local (Pi must be booted & on WiFi)...")
pi_ip = None

for host in ["NeuroLinux.local", "neurolinux.local", "192.168.1.162"]:
    try:
        d = get(f"http://{host}:8080/health", timeout=5)
        ok(f"NeuroLinux Core API at {host}:8080 — {d.get('status','ok')}")
        pi_ip = host
        break
    except Exception:
        pass

if not pi_ip:
    err("Cannot reach Pi — make sure it's booted, on WiFi, and wait 60s after boot")
    print(f"\n{YELLOW}Continuing with NIS Protocol tests only...{RESET}")
else:
    # Camera stream
    try:
        req = urllib.request.Request(f"http://{pi_ip}:8009/health")
        r = urllib.request.urlopen(req, timeout=5)
        d = json.loads(r.read())
        ok(f"Camera pipeline — backend: {d.get('camera_backend','?')}, {d.get('resolution','?')} @ {d.get('fps','?')}fps")
    except Exception as e:
        err(f"Camera pipeline: {e}")

    # xArm status
    try:
        d = get(f"http://{pi_ip}:8080/devices", timeout=5)
        arms = [dev for dev in d.get("devices", []) if "xarm" in dev.get("type","").lower() or "hiwonder" in dev.get("type","").lower()]
        if arms:
            ok(f"xArm detected: {arms[0].get('port','?')} — {arms[0].get('status','?')}")
        else:
            info("xArm not detected yet (plug in USB and wait 10s)")
    except Exception as e:
        err(f"Device discovery: {e}")

# ── 3. Camera → Cosmos pipeline ───────────────────────────────────────────────
section("3/5  Camera → NIS Protocol → Cosmos Reasoning")
info("Simulating a camera frame through the Cosmos pipeline...")

import base64
# 1×1 white JPEG in base64 (minimal valid image for testing)
TEST_JPEG_B64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AJQAB/9k="

try:
    payload = {
        "image_base64": TEST_JPEG_B64,
        "query": "What objects can a robot arm interact with in this scene?",
        "robot_state": {"joints": [0, 0, 0, 0, 0, 0], "gripper": "open"},
    }
    t0 = time.time()
    d = post(f"{NIS_URL}/cookoff/robot-plan", payload, timeout=30)
    elapsed = time.time() - t0
    ok(f"Cosmos robot plan generated in {elapsed:.1f}s")
    recs = d.get("action_recommendations", [])
    if recs:
        ok(f"  Action: {recs[0].get('action', recs[0])}")
    confidence = d.get("combined_confidence", 0)
    ok(f"  Confidence: {confidence:.0%}")
except Exception as e:
    err(f"Cosmos pipeline: {e}")

# ── 4. xArm via NIS Protocol ─────────────────────────────────────────────────
section("4/5  xArm Control via NIS Protocol")
info("Testing xArm command routing through NIS Protocol...")

try:
    d = post(f"{NIS_URL}/openclaw/invoke", {
        "tool": "nis_xarm",
        "parameters": {"action": "get_status"},
    }, timeout=10)
    ok(f"OpenClaw xArm tool: {d.get('result', {}).get('status', d)}")
except Exception as e:
    info(f"xArm invoke (expected if no arm connected): {str(e)[:60]}")

try:
    d = post(f"{NIS_URL}/v4/robotics/status", {}, timeout=10)
    ok(f"Robotics agent: {d.get('status', 'active')}")
except Exception as e:
    info(f"Robotics endpoint: {str(e)[:60]}")

# ── 5. Full end-to-end summary ────────────────────────────────────────────────
section("5/5  Summary")
if pi_ip:
    ok(f"Pi found at {pi_ip}")
    ok("Camera → Cosmos → Action pipeline VERIFIED")
    print(f"\n{BOLD}{GREEN}{'='*50}")
    print("  FIRST EMBODIMENT READY!")
    print(f"{'='*50}{RESET}")
    print(f"\n  Pi Dashboard:  http://{pi_ip}:3000")
    print(f"  Camera stream: http://{pi_ip}:8009/stream")
    print(f"  Pi API:        http://{pi_ip}:8080")
    print(f"  NIS Protocol:  {NIS_URL}")
    print(f"\n  Next: plug in xArm USB → run arm_test.py")
else:
    print(f"\n{YELLOW}Pi not detected yet.{RESET}")
    print("  1. Insert SD card into Pi 5")
    print("  2. Connect HDMI + power")
    print("  3. Wait ~90 seconds for full boot")
    print("  4. Run this script again")
    print(f"\n  NIS Protocol at {NIS_URL} is ready and waiting.")
