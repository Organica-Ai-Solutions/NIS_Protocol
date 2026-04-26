#!/usr/bin/env python3
"""
pi_status.py — Quick Pi / NIS health dashboard
===============================================
Usage:
  python pi_status.py          # one-shot status snapshot
  python pi_status.py --watch  # refresh every 10s until Ctrl-C
  python pi_status.py --ping   # wait for NIS to come up (useful after manual restart)

What it checks:
  • neurolinux-agent  (192.168.1.163:8085)
  • NIS Protocol      (192.168.1.163:8000)
  • Camera feed
  • xArm connection (physical vs simulation)
  • NIS endpoints: /cookoff, /cosmos-dance, /health
"""

import sys
import time
import json
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AGENT = "http://192.168.1.163:8085"
NIS   = "http://192.168.1.163:8000"
H100  = "http://172.16.1.83:8100"
WATCH = "--watch" in sys.argv
PING  = "--ping"  in sys.argv


def _get(url, timeout=6):
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:
        return None, str(e)[:60]


def tick(label, ok, detail=""):
    sym = "✓" if ok else "✗"
    col = ""  # no ANSI color on bare PowerShell — use symbols instead
    print(f"  [{sym}] {label:<28} {detail}")


def check_agent():
    d, err = _get(f"{AGENT}/health")
    if not d:
        tick("neurolinux-agent :8085", False, f"DOWN  ({err})")
        return False
    sim  = d.get("xarm_simulation", True)
    cam  = d.get("camera", False)
    mode = "SIMULATION" if sim else "PHYSICAL HW"
    tick("neurolinux-agent :8085", True,
         f"v{d.get('version','?')}  xArm={mode}  cam={'yes' if cam else 'NO'}")

    # Pull extended system info including service states
    sys_d, _ = _get(f"{AGENT}/system", timeout=8)
    if sys_d:
        svcs = sys_d.get("services", {})
        cpu  = sys_d.get("cpu_pct", "?")
        mem  = sys_d.get("mem_pct", "?")
        temp = sys_d.get("temp_c", "?")
        print(f"       CPU={cpu}%  MEM={mem}%  TEMP={temp}°C")
        for svc, state in svcs.items():
            ok = state == "active"
            tick(f"  svc: {svc}", ok, state)

    return True


def check_nis():
    d, err = _get(f"{NIS}/health")
    if not d:
        tick("NIS Protocol     :8000", False, f"DOWN  ({err})")
        return False
    tick("NIS Protocol     :8000", True,
         f"status={d.get('status','?')}  build={d.get('build','?')}")
    return True


def check_h100():
    d, err = _get(f"{H100}/health", timeout=6)
    if not d:
        tick("H100 Cosmos Reason2 :8100", False, f"DOWN  ({err})")
        return False
    tick("H100 Cosmos Reason2 :8100", True,
         f"model={d.get('model','?')}  gpu={d.get('gpu','?')}  status={d.get('status','?')}")
    return True


def check_nis_endpoints():
    for path, label in [
        ("/cookoff/status",      "/cookoff"),
        ("/cookoff/outcomes",    "/cookoff/outcomes"),
        ("/cosmos-dance/status", "/cosmos-dance"),
        ("/events/stream",        "/events/stream"),
    ]:
        d, err = _get(f"{NIS}{path}")
        if err and "HTTP" in str(err):
            tick(f"  NIS {label}", True, "(endpoint exists)")
        elif d is not None:
            tick(f"  NIS {label}", True, str(list(d.keys()))[:40])
        else:
            tick(f"  NIS {label}", False, str(err)[:40])


def check_camera():
    d, err = _get(f"{AGENT}/camera/snapshot", timeout=10)
    if not d:
        tick("Camera snapshot", False, str(err)[:40])
        return
    img = d.get("image_base64") or d.get("image") or ""
    kb  = len(img) * 3 // 4 // 1024
    tick("Camera snapshot", bool(img), f"{kb} KB image")


def snapshot():
    print()
    print("=" * 56)
    print(f"  NIS PROTOCOL STATUS  —  {time.strftime('%H:%M:%S')}")
    print("=" * 56)

    agent_ok = check_agent()
    nis_ok   = check_nis()
    h100_ok  = check_h100()

    if nis_ok:
        check_nis_endpoints()

    if agent_ok:
        check_camera()

    print()
    if not nis_ok:
        print("  ACTION NEEDED:")
        print("    On the Pi, run:  sudo systemctl restart nis-protocol")
        print("    Or SSH:  ssh neurolinux@192.168.1.163")
        print("             sudo systemctl restart nis-protocol")
        print()
        print("  Auto-deploy watcher is active — will deploy once NIS is up.")
    else:
        print("  All systems operational.")
        print("  Run: python nis_console.py     — interactive robot console")
        print("  Run: python arm_dance.py       — Latino arm dance (BPM mode)")
        print("  Run: python vision_pick.py     — autonomous vision pick")

    print("=" * 56)


if PING:
    print("Waiting for NIS Protocol to come online...")
    while True:
        d, err = _get(f"{NIS}/health")
        if d:
            print(f"\nNIS is UP! status={d.get('status')}")
            snapshot()
            break
        print(".", end="", flush=True)
        time.sleep(5)

elif WATCH:
    try:
        while True:
            snapshot()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopped.")

else:
    snapshot()
