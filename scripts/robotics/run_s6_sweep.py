"""
S6 Sweep — Find lighter's exact S6 value.
Keeps all other servos at stored pick_table values.
Only sweeps S6 from center (500) toward right (650).
"""
import json, time, urllib.request, sys, os
sys.path.insert(0, '.')

PI = 'http://192.168.1.163:8085'
os.makedirs('data/calib_frames', exist_ok=True)


def pi_get(path):
    r = urllib.request.urlopen(PI + path, timeout=10)
    return json.loads(r.read())


def pi_post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(PI + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req, timeout=15)
    return json.loads(r.read())


def snap(name):
    r = urllib.request.urlopen(PI + '/camera/snapshot', timeout=15)
    p = 'data/calib_frames/' + name
    with open(p, 'wb') as f:
        f.write(r.read())
    sz = os.path.getsize(p)
    print("  Saved " + name + " (" + str(sz) + " bytes)")
    return p


def move(servos, ms=1200):
    r = pi_post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    s = ' '.join('S' + k + '=' + str(v) for k, v in sorted(servos.items()))
    print("  -> " + s + "  [" + str(r.get('ok', '?')) + "]")


# ── Load stored poses from arm memory ─────────────────────────────────────────
print("Reading arm memory...")
data = pi_get('/arm/touch_poses')
poses = data.get('touch_poses') or data.get('poses') or {}

home      = {str(k): int(v) for k, v in poses['home'].items()}
inspect   = {str(k): int(v) for k, v in poses['inspect'].items()}
pick      = {str(k): int(v) for k, v in poses['pick_table'].items()}
place_bin = {str(k): int(v) for k, v in poses['place_bin'].items()}

print("HOME:       " + ' '.join('S'+k+'='+str(v) for k,v in sorted(home.items())))
print("PICK_TABLE: " + ' '.join('S'+k+'='+str(v) for k,v in sorted(pick.items())))
print()

# ── Step 1: Go to home ────────────────────────────────────────────────────────
print("Step 1: Moving to HOME (stored, S6=350)...")
move({**home, '1': 100}, ms=2000)
time.sleep(2.5)
snap('s6sweep_home.jpg')
print()

# ── Step 2: Move to inspect to see workspace ──────────────────────────────────
print("Step 2: Moving to INSPECT...")
move({**inspect, '1': 100}, ms=1500)
time.sleep(2.0)
snap('s6sweep_inspect.jpg')
print()

# ── Step 3: Move to pick height at S6=500 (center) for baseline ───────────────
print("Step 3: Moving to PICK height at S6=500 (center)...")
move({**pick, '6': 500, '1': 100}, ms=1200)
time.sleep(1.5)
snap('s6sweep_pick_500.jpg')
print()

# ── Step 4: Sweep S6 from 500 to 650 in steps of 25 ──────────────────────────
print("Step 4: S6 SWEEP — watch the arm rotate right to find lighter...")
print("        Lighter should be somewhere S6=560 to S6=640")
print()

for s6 in [525, 550, 575, 600, 620, 640]:
    label = 'S6=' + str(s6)
    print(label + ':')
    move({**pick, '6': s6, '1': 100}, ms=800)
    time.sleep(1.2)
    fname = 's6sweep_pick_' + str(s6) + '.jpg'
    snap(fname)
    print()

# ── Step 5: Return to home ────────────────────────────────────────────────────
print("Returning to HOME...")
move({**pick, '6': 500, '1': 100}, ms=800)
time.sleep(0.8)
move({**home, '1': 100}, ms=1500)
time.sleep(2.0)

print()
print("=== SWEEP COMPLETE ===")
print("Frames saved to data/calib_frames/")
print()
print("Look at s6sweep_pick_*.jpg and find which S6 centers the gripper over the lighter.")
print("Then run: python set_pick_s6.py <S6_VALUE>")
print()
print("Current pick_table S6: " + str(pick.get('6', '?')))
print("Likely correct S6:     600-620 (from previous sweep observation)")
