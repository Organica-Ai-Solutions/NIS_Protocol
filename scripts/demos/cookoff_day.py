#!/usr/bin/env python3
"""
cookoff_day.py — Cosmos Cookoff Demo Day Runbook
=================================================
NIS Protocol + NeuroLinux xArm + Cosmos Reason2 on H100

Usage:
  python cookoff_day.py             # full status + dry-run check
  python cookoff_day.py --pick      # run one real pick (left90)
  python cookoff_day.py --demo      # full YOLO->Reason2->arm demo
  python cookoff_day.py --dance     # Latino arm dance
  python cookoff_day.py --blue      # pick blue (right side -> place left)
  python cookoff_day.py --s6 450    # custom S6 pick position
  python cookoff_day.py --status    # status check only

All physical operations require Pi + NIS to be live.
H100 Cosmos improves pick accuracy but is NOT required (IK fallback active).

Endpoints used:
  Pi agent:  http://192.168.1.163:8085
  NIS:       http://192.168.1.163:8000
  H100:      http://172.16.1.83:8100
"""

import sys
import json
import time
import base64
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
PI    = "http://192.168.1.163:8085"
NIS   = "http://192.168.1.163:8000"
H100  = "http://172.16.1.83:8100"

PICK  = "--pick"   in sys.argv
DEMO  = "--demo"   in sys.argv
DANCE = "--dance"  in sys.argv
BLUE  = "--blue"   in sys.argv
STATUS_ONLY = "--status" in sys.argv

S6 = 500
if "--s6" in sys.argv:
    try:
        S6 = int(sys.argv[sys.argv.index("--s6") + 1])
    except (IndexError, ValueError):
        pass

GENRE = "reggaeton"
for g in ("reggaeton", "cumbia", "bachata", "salsa"):
    if f"--{g}" in sys.argv:
        GENRE = g


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _get(url, timeout=8):
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:
        return None, str(e)[:60]


def _post(url, body=None, timeout=60):
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:
            detail = {}
        return None, f"HTTP {e.code}: {detail.get('detail', '')[:60]}"
    except Exception as e:
        return None, str(e)[:80]


# ── UI helpers ────────────────────────────────────────────────────────────────
def sep(char="=", n=62):
    print(char * n)


def hdr(msg):
    print()
    sep()
    print(f"  {msg}")
    sep()


def ok(label, detail=""):
    suf = f"  {detail}" if detail else ""
    print(f"  [OK] {label:<32}{suf}")


def fail(label, detail=""):
    suf = f"  {detail}" if detail else ""
    print(f"  [!!] {label:<32}{suf}")


def info(msg):
    print(f"       {msg}")


def step(n, msg):
    print(f"\n  [{n:02d}] {msg}")


# ── Status checks ─────────────────────────────────────────────────────────────
def check_all():
    hdr(f"NIS PROTOCOL  x  COSMOS COOKOFF  —  {time.strftime('%H:%M:%S')}")

    # Pi agent
    d, err = _get(f"{PI}/health")
    if d:
        sim = d.get("xarm_simulation", True)
        cam = d.get("camera", False)
        ok("Pi NeuroLinux :8085",
           f"v{d.get('version','?')}  xArm={'PHYSICAL' if not sim else 'SIM'}  cam={'yes' if cam else 'NO'}")
        if sim:
            fail("  xArm in SIMULATION", "reconnect USB or run: POST /arm/reconnect")
    else:
        fail("Pi NeuroLinux :8085", f"DOWN — {err}")

    # NIS Protocol
    d, err = _get(f"{NIS}/health")
    if d:
        ok("NIS Protocol  :8000", f"status={d.get('status','?')}")
    else:
        fail("NIS Protocol  :8000", f"DOWN — {err}")
        info("Run on Pi:  sudo systemctl restart nis-protocol")

    # H100 Cosmos Reason2
    d, err = _get(f"{H100}/health", timeout=6)
    if d:
        ok("H100 Reason2  :8100",
           f"model={d.get('model','cosmos-reason2')}  gpu={d.get('gpu',0)}")
    else:
        fail("H100 Reason2  :8100", f"DOWN — {err}  (IK fallback active)")

    # Cookoff pipeline
    d, err = _get(f"{NIS}/cookoff/status")
    if d:
        svcs = d.get("h100_services", {})
        r2   = svcs.get("reason2", {}).get("healthy", False)
        pred = svcs.get("predict25", {}).get("healthy", False)
        xfer = svcs.get("transfer25", {}).get("healthy", False)
        ok("Cookoff pipeline",
           f"reason2={'UP' if r2 else 'down'}  predict={'UP' if pred else 'down'}  transfer={'UP' if xfer else 'down'}")
    else:
        fail("Cookoff pipeline", str(err)[:50])

    # Pick outcomes (learning)
    d, err = _get(f"{NIS}/cookoff/outcomes")
    if d and d.get("available"):
        ok("Pick learning state",
           f"total={d.get('total_picks',0)}  rate={d.get('success_rate',0):.0%}  "
           f"recent10={d.get('recent_10_rate',0):.0%}")
    elif d:
        info("Pick learning: no data yet (0 picks recorded)")

    # SSE channel
    d, err = _get(f"{NIS}/events/topics")
    if d:
        ok("SSE event stream", f"topics={d.get('topics',[])}  conns={d.get('active_connections',0)}")
    else:
        fail("SSE event stream", str(err)[:40])

    # Camera snapshot
    d, err = _get(f"{PI}/camera/snapshot", timeout=15)
    if d:
        img = d.get("image_base64") or d.get("image") or ""
        kb  = len(img) * 3 // 4 // 1024 if img else 0
        ok("Camera snapshot", f"{kb} KB  {'OK' if kb > 5 else 'empty?'}")
    else:
        fail("Camera snapshot", str(err)[:40])

    print()


# ── Pre-flight gate ───────────────────────────────────────────────────────────
def preflight():
    """Return True only if Pi + NIS are live and arm is physical."""
    d_pi,  _ = _get(f"{PI}/health")
    d_nis, _ = _get(f"{NIS}/health")
    if not d_pi:
        fail("PRE-FLIGHT", "Pi agent DOWN — cannot run pick")
        return False
    if d_pi.get("xarm_simulation", True):
        fail("PRE-FLIGHT", "arm in SIMULATION — reconnect USB and retry")
        _post(f"{PI}/arm/reconnect")
        time.sleep(2.5)
        d2, _ = _get(f"{PI}/health")
        if d2 and not d2.get("xarm_simulation", True):
            ok("arm reconnected", "physical mode active")
        else:
            return False
    if not d_nis:
        fail("PRE-FLIGHT", "NIS Protocol DOWN — cannot route cookoff")
        return False
    return True


# ── Pick sequence ─────────────────────────────────────────────────────────────
def run_pick(s6=500, place="left90"):
    hdr(f"COOKOFF PICK  S6={s6}  place={place}")

    if not preflight():
        sys.exit(1)

    # Camera warmup — 3 dummy snaps so sensor is fully open
    info("Camera warmup (3 snaps)...")
    for i in range(3):
        _get(f"{PI}/camera/snapshot", timeout=8)
        time.sleep(0.4)

    # Pre-pick Cosmos inspection (optional — fallback if H100 down)
    cosmos_correction = 0
    d_r2, _ = _get(f"{H100}/health", timeout=4)
    if d_r2:
        snap, _ = _get(f"{PI}/camera/snapshot", timeout=12)
        img = (snap or {}).get("image_base64", "")
        if img:
            body = {
                "query": (
                    "Arm at HOVER position (z~6cm). "
                    "Workspace is 17x20.5cm wooden table. "
                    "Object (lighter) is at center-front approx (x=0, y=17cm). "
                    "Q: Is the lighter visible? "
                    "Q: Estimate offset from arm centerline in cm (positive=right). "
                    "Reply JSON only: {\"object_visible\": bool, \"object_x_cm\": number, "
                    "\"confidence\": 0-1}"
                ),
                "image_base64": img,
                "max_tokens": 80,
                "use_think": False,
            }
            d_reason, _ = _post(f"{H100}/reason", body, timeout=15)
            if d_reason:
                raw = d_reason.get("response", "")
                try:
                    import re
                    m = re.search(r'\{.*\}', raw, re.DOTALL)
                    parsed = json.loads(m.group()) if m else {}
                    x_cm = float(parsed.get("object_x_cm", 0))
                    conf = float(parsed.get("confidence", 0))
                    if abs(x_cm) > 1.5 and conf >= 0.70:
                        cosmos_correction = -round(x_cm * (375 / 90))
                        s6_new = 500 + cosmos_correction
                        info(f"Cosmos correction: x={x_cm:+.1f}cm  S6 {s6} -> {s6_new}  (conf={conf:.2f})")
                        s6 = s6_new
                    else:
                        info(f"Cosmos: x={x_cm:+.1f}cm conf={conf:.2f}  no correction needed")
                except Exception:
                    info(f"Cosmos response: {raw[:80]}")

    # Execute IK pick via NIS
    info(f"Executing IK pick  S6={s6}  place={place}...")
    t0 = time.time()
    d, err = _post(f"{NIS}/cookoff/pick",
                   {"s6": s6, "z": 1.5, "place": place, "wait_sec": 0.0},
                   timeout=90)
    elapsed = time.time() - t0

    if not d:
        fail("pick failed", str(err))
        return False

    steps = d.get("steps", [])
    n_ok  = sum(1 for s in steps if s.get("ok"))
    n_tot = len(steps)
    print()
    for s in steps:
        sym = "OK" if s.get("ok") else "!!"
        print(f"  [{sym}] {s.get('step','?'):<14}")
    print()
    if d.get("ok"):
        ok(f"PICK COMPLETE  {n_ok}/{n_tot} steps", f"{elapsed:.1f}s")
    else:
        fail(f"pick partial  {n_ok}/{n_tot} steps", f"{elapsed:.1f}s")

    # Post-pick verify with Cosmos
    time.sleep(1.0)
    snap2, _ = _get(f"{PI}/camera/snapshot", timeout=12)
    img2 = (snap2 or {}).get("image_base64", "")
    if img2 and d_r2:
        body2 = {
            "query": (
                "Task: pick lighter and place in drop zone. "
                "Look at the current scene — is the task complete? "
                "Reply YES or NO with one sentence."
            ),
            "image_base64": img2,
            "max_tokens": 60,
            "use_think": False,
        }
        d_verify, _ = _post(f"{H100}/reason", body2, timeout=15)
        if d_verify:
            verdict = d_verify.get("response", "")
            sym = "OK" if "yes" in verdict.lower() else "!!"
            print(f"  [{sym}] Cosmos verify: {verdict[:80]}")

    return d.get("ok", False)


# ── Full demo ─────────────────────────────────────────────────────────────────
def run_demo():
    hdr("COOKOFF DEMO  (YOLO -> Cosmos Reason2 -> IK arm)")

    if not preflight():
        sys.exit(1)

    task = "Pick up the lighter and place it on the left"
    info(f"Task: {task}")

    d, err = _post(f"{NIS}/cookoff/demo",
                   {"task": task, "execute_arm": True, "simulation": False},
                   timeout=120)
    if not d:
        fail("demo failed", str(err))
        return

    print()
    ok("Demo complete",
       f"steps={d.get('steps_ok')}/{d.get('steps_total')}  "
       f"goal={'YES' if d.get('goal_complete') else 'no'}  "
       f"{d.get('latency_ms',0):.0f}ms")
    if d.get("cosmos_reasoning"):
        info(f"R2 reasoning: {d['cosmos_reasoning'][:100]}")
    if d.get("goal_verify"):
        info(f"Goal verify:  {d['goal_verify'][:80]}")


# ── Dance ─────────────────────────────────────────────────────────────────────
def run_dance(genre="reggaeton"):
    hdr(f"COSMOS DANCE  genre={genre.upper()}")

    if not preflight():
        sys.exit(1)

    info(f"Starting {genre} dance (24 moves)...")
    d, err = _post(f"{NIS}/cookoff/dance",
                   {"genre": genre, "moves": 24, "energy": 0.20, "use_mic": False},
                   timeout=80)
    if not d:
        fail("dance failed", str(err))
        return

    ok(f"{genre} DANCE COMPLETE",
       f"moves={d.get('moves_done',0)}/{d.get('moves_requested',24)}  "
       f"{d.get('latency_ms',0):.0f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sep()
    print("  NIS PROTOCOL  x  COSMOS COOKOFF  —  DEMO DAY RUNBOOK")
    sep()
    print(f"  Pi:   {PI}")
    print(f"  NIS:  {NIS}")
    print(f"  H100: {H100}")
    print()

    # Always show status first
    check_all()

    if STATUS_ONLY:
        return

    if DANCE:
        run_dance(GENRE)
        return

    if DEMO:
        run_demo()
        return

    if PICK:
        place = "left90"
        if BLUE:
            # Blue lighter is on left side, place to right
            s6_pick = 685   # left45
            place   = "right90"
        else:
            # Green/default lighter is center-front
            s6_pick = S6
        success = run_pick(s6=s6_pick, place=place)
        sys.exit(0 if success else 1)

    # Default: status only (already printed above)
    print("  Flags:  --pick  --demo  --dance  --status  --blue  --s6 NNN")
    print("          --reggaeton / --cumbia / --bachata / --salsa")
    print()
    print("  Quick picks:")
    print("    python cookoff_day.py --pick              # center pick -> left90")
    print("    python cookoff_day.py --pick --s6 450     # right-shifted pick")
    print("    python cookoff_day.py --pick --blue       # blue lighter -> right")
    print("    python cookoff_day.py --demo              # full YOLO+R2 demo")
    print("    python cookoff_day.py --dance --cumbia    # cumbia dance")
    print()


if __name__ == "__main__":
    main()
