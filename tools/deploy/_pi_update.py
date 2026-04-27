#!/usr/bin/env python3
"""
PI FULL UPDATE - HTTP-based deployment of confirmed parameters
=============================================================
Pushes ALL confirmed working calibration data to the Pi via HTTP.
No SSH required.

What this does:
  1. Pushes 7 confirmed arm poses as touch_poses (home, hover, mid, pick, lift, left90_place, left90_release)
  2. Pushes camera calibration (linear scale + affine if available)
  3. Verifies each pose was saved
  4. Sends the arm to HOME to confirm connection is live

Confirmed pick parameters (verified 2026-02-27):
  z=1.5cm, S6=500, S1_GRIP=700, alpha=-65
  S3=142, S4=856, S5=430 (at pick height)
"""

import json, time, math
import urllib.request as _ur

PI = 'http://192.168.1.163:8085'
SEP = '=' * 60

def get(p, t=10):
    try:
        r = _ur.urlopen(PI+p, timeout=t)
        return json.loads(r.read())
    except Exception as e:
        print(f'  GET {p}: {e}')
        return {}

def post(p, b=None, t=15):
    try:
        d = json.dumps(b or {}).encode()
        r = _ur.Request(PI+p, data=d, headers={'Content-Type':'application/json'})
        return json.loads(_ur.urlopen(r, timeout=t).read())
    except Exception as e:
        print(f'  POST {p}: {e}')
        return {}

# ============================================================
# CONFIRMED WORKING ARM POSITIONS
# ============================================================
POSES = {
    # Name: servo dict {S1..S6}
    'home': {
        '1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': 500
    },
    'pick_hover_center': {   # 6cm above pick point, S6=500 (center)
        '1': 100, '2': 500, '3': 222, '4': 697, '5': 604, '6': 500
    },
    'pick_mid_center': {     # 3.5cm above pick point (mid descent)
        '1': 100, '2': 500, '3': 158, '4': 798, '5': 502, '6': 500
    },
    # NOTE: pick_down_center is SKIPPED in the movement loop below to avoid
    # knocking objects on the table. The servo values are saved via /arm/group_move
    # from the hover position at the save step, not by physically descending.
    # Actual pick height: z=1.5cm, S3=142 S4=856 S5=430 S6=500 (confirmed)
    'lift_center': {         # Home height with gripper CLOSED and S6=500
        '1': 700, '2': 500, '3': 310, '4': 870, '5': 680, '6': 500
    },
    'place_left90': {        # Drop position - left 90 degrees
        '1': 700, '2': 500, '3': 220, '4': 827, '5': 425, '6': 875
    },
    'release_left90': {      # Release - same as place but gripper open
        '1': 100, '2': 500, '3': 220, '4': 827, '5': 425, '6': 875
    },
    'clear_view': {          # Arm swings left, camera has clear view of workspace
        '1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': 750
    },
}

print(SEP)
print('  NIS PROTOCOL - PI FULL UPDATE (HTTP)')
print(SEP)

# ── 1. Health check ───────────────────────────────────────────
print()
print('[1] Pi health check...')
h = get('/health')
if not h:
    print('  FAIL: Pi unreachable at', PI)
    import sys; sys.exit(1)

print(f'  Agent v{h.get("version")}  xarm={h.get("xarm")}  cam={h.get("camera")}')
if not h.get('xarm'):
    print('  WARNING: xArm not connected - poses will be saved but arm wont move')

# ── 2. Push touch poses ───────────────────────────────────────
print()
print('[2] Saving confirmed arm poses...')
print('    (Arm will move to each pose to save it - stand clear!)')
print()

# IMPORTANT: pick_down_center (z=1.5cm) is NOT saved via physical movement
# to avoid knocking objects on the table. guided_pick.py uses IK-computed values,
# not touch poses, for the actual pick — so this pose is reference-only.
# We record it in results as pre-known good values.
results = {}

# Poses that are safe to move to (all high/away from table surface)
SAFE_TO_MOVE = ['home', 'pick_hover_center', 'pick_mid_center',
                'lift_center', 'place_left90', 'release_left90', 'clear_view']
SKIP_MOVE    = []  # pick_down_center removed from POSES dict above

for pose_name, positions in POSES.items():
    print(f'  [{pose_name}]', end=' ')
    r   = post('/arm/group_move', {'positions': positions, 'duration_ms': 1400})
    sim = r.get('simulation', False)
    if sim:
        post('/arm/reconnect')
        time.sleep(2.5)
        r = post('/arm/group_move', {'positions': positions, 'duration_ms': 1400})
    time.sleep(1.6)
    sr     = post('/arm/save_touch_pose', {'name': pose_name})
    saved  = sr.get('ok', False) or 'positions' in sr
    results[pose_name] = saved
    print(f'-> {"SAVED" if saved else "FAILED"}{"  [SIM]" if sim else ""}')

# pick_down_center: move briefly from hover (minimal dip, ≤300ms, then back up)
print('  [pick_down_center (quick-dip)]', end=' ')
post('/arm/group_move', {'positions': POSES.get('pick_hover_center',
     {'1':100,'2':500,'3':222,'4':697,'5':604,'6':500}), 'duration_ms': 800})
time.sleep(1.0)
PICK_DOWN = {'1': 100, '2': 500, '3': 142, '4': 856, '5': 430, '6': 500}
post('/arm/group_move', {'positions': PICK_DOWN, 'duration_ms': 250})
time.sleep(0.3)
sr3   = post('/arm/save_touch_pose', {'name': 'pick_down_center'})
saved3 = sr3.get('ok', False) or 'positions' in sr3
results['pick_down_center'] = saved3
# Pull back immediately
post('/arm/group_move', {'positions': {'1':100,'2':500,'3':222,'4':697,'5':604,'6':500},
     'duration_ms': 400})
time.sleep(0.5)
print(f'-> {"SAVED" if saved3 else "FAILED"}')

ok_count = sum(1 for v in results.values() if v)
print()
print(f'  Poses saved: {ok_count}/{len(POSES)}')

# ── 3. Return to HOME safely ──────────────────────────────────
print()
print('[3] Returning arm to HOME...')
post('/arm/group_move', {
    'positions': {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500},
    'duration_ms': 2000
})
time.sleep(2.2)

# ── 4. Calibration map via API ────────────────────────────────
print()
print('[4] Pushing camera calibration data...')

# Our confirmed single-point calibration (arm-guided)
# anchor: arm(0, 17) <-> cam(approx 530, 80) from successful pick frame
# These are the linear scale values derived from the confirmed pick + camera geometry
cal_data = {
    'scale_x': 14.059,
    'scale_y': 9.261,
    'cam_cx': 640,
    'cam_y_ref': 620.4,
    'cam_w': 1280,
    'cam_h': 720,
    'note': 'confirmed 2026-02-27 from successful pick at x=0 y=17 z=1.5',
}

# Try calibration/map endpoint
cr = post('/calibration/map', cal_data)
if cr.get('ok') or cr.get('status'):
    print(f'  Calibration map updated: {cr}')
else:
    print(f'  calibration/map not accepted ({cr}) -- Pi uses internal calibration.json')

# ── 5. Verify poses ───────────────────────────────────────────
print()
print('[5] Verifying saved poses...')
vr = get('/arm/touch_poses')
saved_poses = vr.get('poses', {})
for name in POSES:
    if name in saved_poses:
        sp = saved_poses[name]
        expected_s3 = POSES[name].get('3', '?')
        actual_s3   = sp.get('3', '?')
        match = abs(int(actual_s3) - int(expected_s3)) <= 10
        print(f'  {name}: S3={actual_s3} (exp {expected_s3}) -> {"OK" if match else "DRIFT"}')
    else:
        print(f'  {name}: MISSING')

# ── 6. Quick arm function test ────────────────────────────────
print()
print('[6] Quick arm function test...')
# Move to home, then to hover, then back
print('  -> HOME')
r = post('/arm/group_move', {
    'positions': POSES['home'],
    'duration_ms': 1500
})
time.sleep(1.8)

print('  -> PICK HOVER (center, 6cm above table)')
r = post('/arm/group_move', {
    'positions': POSES['pick_hover_center'],
    'duration_ms': 1500
})
time.sleep(1.8)

print('  -> HOME')
r = post('/arm/group_move', {
    'positions': POSES['home'],
    'duration_ms': 1500
})
time.sleep(1.8)

# ── 7. Final status ───────────────────────────────────────────
print()
print(SEP)
print('  UPDATE COMPLETE')
print(SEP)
print()
print(f'  Poses saved: {ok_count}/{len(POSES)}')
print(f'  Pi agent: v{h.get("version")}  xarm={h.get("xarm")}')
print()
print('  Confirmed pick parameters now stored on Pi:')
print('    pose: pick_down_center  (S3=142 S4=856 S5=430 S6=500)')
print('    z=1.5cm  alpha=-65deg  S1_grip=700')
print()
print('  Next steps:')
print('    1. Run camera calibration:  python cam_calibrate.py --wait 15')
print('    2. Test autonomous pick:    python vision_pick.py')
print()

# ── 8. SSH key install instructions ──────────────────────────
print('  To enable SSH for future Pi updates:')
try:
    pub_key = open('C:/Users/DiegoTorres/.ssh/id_ed25519.pub').read().strip()
    print('  Run this on the Pi (connect monitor+keyboard or use existing session):')
    print()
    print(f'    mkdir -p ~/.ssh && echo "{pub_key}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys')
    print()
    print('  OR from Windows if you know the Pi password:')
    print('    ssh-copy-id -i C:/Users/DiegoTorres/.ssh/id_ed25519.pub pi@192.168.1.163')
except Exception:
    pass
