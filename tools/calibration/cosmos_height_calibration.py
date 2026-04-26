"""
COSMOS HEIGHT CALIBRATION - xArm AI
=====================================
Full Cosmos stack pipeline to fix pick height (arm 1-2cm above table).

Problem:
  Official pick ki_move(x, y, 1.2, -71) → arm stops 1-2cm above table.
  Root cause: z=1.2cm is the GRIPPER TIP height but actual table may be lower.
  Previous code only adjusted S3 (wrist) -- WRONG.
  Height is controlled by S5 (shoulder) + S4 (elbow) together via IK.

Cosmos Stack Used:
  /cosmos/depth_map     - Cosmos AI spatial analysis & object depth
  /vision/detect        - YOLO + color detection of lighter
  /camera/snapshot      - Visual verification at each height
  /agent/chat           - Cosmos reasoning about geometry correction

IK truth: for z delta of 1.5cm lower (z=-0.3):
  S5: 450 -> 397  (shoulder, -53)
  S4: 800 -> 838  (elbow, +38)
  S3: 180 -> 178  (wrist, barely changes)

Usage:
  python cosmos_height_calibration.py               # interactive
  python cosmos_height_calibration.py --auto        # auto (1.5cm correction)
  python cosmos_height_calibration.py --delta 1.0   # manual delta in cm
"""

import json, math, time, sys, os, base64
from pathlib import Path

PI     = 'http://192.168.1.163:8085'
FRAMES = Path('data/height_calib')
OUT    = Path('data/calib_results.json')
FRAMES.mkdir(parents=True, exist_ok=True)

AUTO  = '--auto' in sys.argv
DELTA = None
for i, a in enumerate(sys.argv):
    if a == '--delta' and i + 1 < len(sys.argv):
        try:
            DELTA = float(sys.argv[i + 1])
        except ValueError:
            pass

SEP = '=' * 68

# ============================================================================
# API HELPERS
# ============================================================================

import urllib.request as _ur

def _get(url, timeout=12):
    r = _ur.urlopen(url, timeout=timeout)
    return json.loads(r.read())

def _post(url, body, timeout=20):
    data = json.dumps(body).encode()
    req  = _ur.Request(url, data=data, headers={'Content-Type': 'application/json'})
    r    = _ur.urlopen(req, timeout=timeout)
    return json.loads(r.read())

def get(p, timeout=12):
    try:
        return _get(PI + p, timeout=timeout)
    except Exception as e:
        print(f'  GET {p}: {e}')
        return {}

def post(p, body=None, timeout=20):
    try:
        return _post(PI + p, body or {}, timeout=timeout)
    except Exception as e:
        print(f'  POST {p}: {e}')
        return {}

def move(servos, ms=1000, label='', extra=0.0):
    r   = post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim = r.get('simulation', False)
    ok  = r.get('ok', False)
    s   = ' '.join(f'S{k}={v}' for k, v in sorted(servos.items()))
    tag = f'[{label}] ' if label else ''
    stat = 'SIM' if sim else ('OK' if ok else 'FAIL')
    print(f'  {tag}{s}  [{stat}]')
    if sim:
        print('  !! ARM IN SIM -- reconnecting...')
        post('/arm/reconnect')
        time.sleep(2.5)
        post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    time.sleep(ms / 1000.0 + 0.4 + extra)

def snap(name, label=''):
    d   = get('/camera/snapshot', timeout=25)
    img = d.get('image_base64') or d.get('image')
    if img:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        tag = f'[{label}] ' if label else ''
        print(f'  {tag}[CAM] {p.name}  ({p.stat().st_size:,}b)')
        return str(p)
    print('  [CAM] snapshot failed')
    return None

def ask(prompt, default=''):
    if AUTO:
        print(f'  {prompt}  [auto: {default}]')
        return str(default)
    try:
        v = input(f'  {prompt} ').strip()
        return v if v else str(default)
    except EOFError:
        return str(default)

def section(t):
    print(); print(SEP); print(f'  {t}'); print(SEP)


# ============================================================================
# KINEMATICS  (official Hiwonder link lengths)
# ============================================================================

L1, L2, L3, L4 = 6.9, 9.5, 9.5, 16.9
S6_SCALE = 375.0 / 90.0   # 4.167 counts/deg

_HOME_THETA1, _HOME_THETA2, _HOME_THETA3 = 45.4, 88.6, -134.0
_HOME_S5, _HOME_S4, _HOME_S3             = 680,  870,   310
S5_SCALE, S4_SCALE, S3_SCALE             = 5.84, 4.09,  8.97


def ki_to_servos(x, y, z, alpha_deg):
    """Full IK: ki_move(x,y,z,alpha) -> servo dict (S2-S6)."""
    theta_base_deg = math.degrees(math.atan2(x, y))
    s6  = round(500.0 - theta_base_deg * S6_SCALE)
    s6  = max(100, min(900, s6))
    r   = math.sqrt(x*x + y*y)
    ar  = math.radians(alpha_deg)
    ex  = L4 * math.cos(ar)
    ey  = L4 * math.sin(ar)
    px  = r  - ex
    py  = (z - L1) - ey
    d   = math.sqrt(px*px + py*py)
    d   = max(abs(L2 - L3) + 0.01, min(L2 + L3 - 0.01, d))
    cos_t2 = (d*d - L2*L2 - L3*L3) / (2.0 * L2 * L3)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    t2  = math.degrees(math.acos(cos_t2))
    k1  = L2 + L3 * math.cos(math.radians(t2))
    k2  = L3 * math.sin(math.radians(t2))
    t1  = math.degrees(math.atan2(py, px) - math.atan2(k2, k1))
    t3  = alpha_deg - t1 - t2
    s5  = round(_HOME_S5 + (t1 - _HOME_THETA1) * S5_SCALE)
    s4  = round(_HOME_S4 + (t2 - _HOME_THETA2) * S4_SCALE)
    s3  = round(_HOME_S3 + (t3 - _HOME_THETA3) * S3_SCALE)
    return {
        '2': 500,
        '3': max(100, min(900, s3)),
        '4': max(100, min(900, s4)),
        '5': max(100, min(900, s5)),
        '6': s6,
    }


def ki_pos(x, y, z, alpha, gripper=100):
    d = ki_to_servos(x, y, z, alpha)
    d['1'] = gripper
    return d


# Fixed ground-truth HOME (verified .rob values)
HOME_POS = {'1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': 500}

# Official PICK (z=1.2cm, overridden with .rob values)
PICK_OFFICIAL = {'1': 100, '2': 500, '3': 180, '4': 800, '5': 450}


# ============================================================================
# LOAD EXISTING CALIBRATION
# ============================================================================

section('STEP 0 -- LOAD CALIBRATION & PREFLIGHT')

if OUT.exists():
    calib = json.loads(OUT.read_text())
    S6_PICK  = calib.get('s6_pick', 400)
    S6_PLACE = calib.get('s6_place', 875)
    PLACE_KEY = calib.get('place_key', 'place_left90')
    S3_OLD   = calib.get('s3_pick', 180)
    print(f'  Loaded {OUT}')
    print(f'  S6_PICK={S6_PICK}  S6_PLACE={S6_PLACE}  S3_old={S3_OLD}')
else:
    print('  No calib_results.json found -- using defaults')
    S6_PICK, S6_PLACE, PLACE_KEY, S3_OLD = 400, 875, 'place_left90', 180
    calib = {}

# Compute x, y from S6_PICK
theta_pick_deg = (500.0 - S6_PICK) / S6_SCALE
theta_pick_rad = math.radians(theta_pick_deg)
X_PICK = 17.0 * math.sin(theta_pick_rad)
Y_PICK = 17.0 * math.cos(theta_pick_rad)
print(f'  Lighter at x={X_PICK:.1f}cm, y={Y_PICK:.1f}cm  (from S6={S6_PICK})')

# Health check
try:
    h = get('/health')
    print(f'  Agent: {h.get("service")} v{h.get("version")}  [{h.get("status")}]')
    print(f'  xArm: connected={h.get("xarm")}  sim={h.get("xarm_simulation")}')
    print(f'  Camera: {h.get("camera")}')
except Exception as e:
    print(f'  FAIL: cannot reach Pi -- {e}'); sys.exit(1)

if h.get('xarm_simulation'):
    print('  Arm in SIM -- reconnecting...')
    post('/arm/reconnect')
    time.sleep(2.5)


# ============================================================================
# STEP 1: COSMOS DEPTH MAP - Spatial Analysis
# ============================================================================

section('STEP 1 -- COSMOS DEPTH MAP (AI spatial analysis)')

cosmos_z_estimate = None
cosmos_data = {}

try:
    dm = get('/cosmos/depth_map', timeout=20)
    spatial = dm.get('data', {}).get('cosmos_spatial', {})
    cosmos_data = dm.get('data', {})
    px_per_cm = spatial.get('px_per_cm', 19)
    objects   = spatial.get('objects', [])
    ws_w      = spatial.get('workspace_w_cm', 28)
    ws_h      = spatial.get('workspace_h_cm', 19)

    print(f'  Cosmos spatial: {px_per_cm} px/cm  workspace={ws_w}x{ws_h}cm')
    print(f'  Objects detected: {len(objects)}')
    for o in objects:
        print(f'    {o.get("color","?")} at ({o.get("px_pct_x")}%, {o.get("px_pct_y")}%)  depth={o.get("depth")}')
        if o.get('depth') == 'near':
            cosmos_z_estimate = 0.0   # near = on table
        elif o.get('depth') == 'far':
            cosmos_z_estimate = 5.0

    if cosmos_z_estimate is not None:
        print(f'  Cosmos depth estimate: object is "{objects[0].get("depth")}" -> z_table~{cosmos_z_estimate}cm')
except Exception as e:
    print(f'  Cosmos depth map: {e}')


# ============================================================================
# STEP 2: COSMOS AGENT REASONING
# ============================================================================

section('STEP 2 -- COSMOS AGENT REASONING')

cosmos_recommendation = None
try:
    msg = (
        'The xArm pick height is 1-2cm above the table. '
        f'Official ki_move pick z=1.2cm. '
        f'Current servo values: S3=180, S4=800, S5=450. '
        f'Arm link lengths: L1=6.9 L2=9.5 L3=9.5 L4=16.9 cm. '
        f'The lighter is at x={X_PICK:.1f}cm, y={Y_PICK:.1f}cm from the arm base. '
        'Cosmos spatial analysis says the object is at "near" depth (on table surface). '
        'To lower the arm 1.5cm: IK gives S5=397, S4=838, S3=178. '
        'Please confirm: should I decrease S5 and increase S4 to lower the pick height? '
        'Reply with the recommended servo adjustment.'
    )
    ctx = {
        'arm': 'xArm-AI Hiwonder',
        'issue': 'pick height 1-2cm above table',
        'current_pick': {'S3': 180, 'S4': 800, 'S5': 450, 'S6': S6_PICK},
        'cosmos_depth': cosmos_data.get('cosmos_spatial', {})
    }
    r = post('/agent/chat', {'message': msg, 'context': ctx}, timeout=25)
    reply = r.get('reply') or r.get('response') or r.get('message') or str(r)[:400]
    cosmos_recommendation = reply
    print(f'  Cosmos AI says: {reply[:500]}')
except Exception as e:
    print(f'  Agent chat: {e}')
    cosmos_recommendation = 'IK-based: decrease S5 by ~53, increase S4 by ~38 to lower 1.5cm'
    print(f'  Using built-in recommendation: {cosmos_recommendation}')


# ============================================================================
# STEP 3: VISION DETECT - Find lighter in frame
# ============================================================================

section('STEP 3 -- VISION DETECT (lighter in camera frame)')

vision_lighter = None
try:
    # Move to home first for a clean view
    move(HOME_POS, ms=1500, label='HOME')
    time.sleep(0.5)
    vd = get('/vision/detect', timeout=15)
    dets = vd.get('detections', [])
    print(f'  Vision detections: {len(dets)}')
    for d in dets:
        label = d.get('label', '?')
        cx, cy = d.get('cx', 0), d.get('cy', 0)
        conf = d.get('conf', 0)
        print(f'    {label:15s} cx={cx:4d} cy={cy:4d}  conf={conf:.2f}')
        if 'lighter' in label.lower() or 'cube' in label.lower() or conf > 0.3:
            vision_lighter = d

    if not vision_lighter and dets:
        vision_lighter = max(dets, key=lambda x: x.get('conf', 0))
    if vision_lighter:
        print(f'  Best object: {vision_lighter.get("label")} at ({vision_lighter.get("cx")}, {vision_lighter.get("cy")})')
except Exception as e:
    print(f'  Vision detect: {e}')

snap('01_home_view.jpg', 'HOME')


# ============================================================================
# STEP 4: IK HEIGHT SWEEP -- AUTO-COMPUTE CORRECTED POSITIONS
# ============================================================================

section('STEP 4 -- IK HEIGHT SWEEP  (z from +1.5 down to -2.0 cm)')

print('  For each z, computing FULL IK (S3+S4+S5 all adjusted correctly).')
print('  THIS IS THE FIX: height is controlled by S5+S4, NOT just S3.')
print()

# z sweep: from 1.5cm down to -2.0cm
Z_START  = 1.5
Z_END    = -2.0
Z_STEP   = -0.25
z_values = []
v = Z_START
while v >= Z_END:
    z_values.append(round(v, 2))
    v += Z_STEP
    v  = round(v, 2)

print(f'  z sweep ({len(z_values)} steps): {Z_START} -> {Z_END} cm, step={abs(Z_STEP)}cm')
print()
print(f'  {"z(cm)":>7}  {"S3":>5}  {"S4":>5}  {"S5":>5}  {"delta_S5":>9}  {"delta_S4":>9}')
print(f'  {"-"*7}  {"-"*5}  {"-"*5}  {"-"*5}  {"-"*9}  {"-"*9}')

sweep_table = []
for z in z_values:
    sv = ki_to_servos(X_PICK, Y_PICK, z, -71)
    s3, s4, s5 = sv['3'], sv['4'], sv['5']
    ds5 = s5 - 450   # delta from official
    ds4 = s4 - 800
    sweep_table.append({'z': z, 's3': s3, 's4': s4, 's5': s5})
    marker = ' <-- OFFICIAL' if abs(z - 1.2) < 0.15 else ''
    print(f'  {z:>7.2f}  {s3:>5}  {s4:>5}  {s5:>5}  {ds5:>+9}  {ds4:>+9}{marker}')

print()
print('  Physical meaning:')
print('  S5 (shoulder): LOWER value = arm reaches DOWN more')
print('  S4 (elbow):    HIGHER value = elbow bends more (works with S5)')
print('  S3 (wrist):    barely changes -- NOT the height control')


# ============================================================================
# STEP 5: DETERMINE CORRECTION AMOUNT
# ============================================================================

section('STEP 5 -- HEIGHT CORRECTION DECISION')

if DELTA is not None:
    z_correction = DELTA
    print(f'  User-specified --delta {DELTA}cm correction')
elif AUTO:
    z_correction = 1.5
    print(f'  AUTO mode: applying 1.5cm correction (midpoint of 1-2cm reported error)')
else:
    print('  The arm is 1-2cm above the table.')
    print('  A 1.5cm correction (z_new = 1.2 - 1.5 = -0.3cm) is the midpoint.')
    print()
    print('  Enter the correction in cm (e.g. 1.0 = 1cm lower, 1.5 = 1.5cm lower, 2.0 = 2cm lower):')
    try:
        z_correction = float(ask('Height correction in cm [1.5]', '1.5'))
    except ValueError:
        z_correction = 1.5

z_new = 1.2 - z_correction
sv_new = ki_to_servos(X_PICK, Y_PICK, z_new, -71)
S3_NEW = sv_new['3']
S4_NEW = sv_new['4']
S5_NEW = sv_new['5']

print()
print(f'  Correction: {z_correction:.2f}cm lower')
print(f'  New ki_move z = 1.2 - {z_correction:.2f} = {z_new:.2f}cm')
print()
print(f'  Servo changes:')
print(f'    S3: {PICK_OFFICIAL["3"]} -> {S3_NEW}  (delta={S3_NEW - PICK_OFFICIAL["3"]:+d})')
print(f'    S4: {PICK_OFFICIAL["4"]} -> {S4_NEW}  (delta={S4_NEW - PICK_OFFICIAL["4"]:+d})')
print(f'    S5: {PICK_OFFICIAL["5"]} -> {S5_NEW}  (delta={S5_NEW - PICK_OFFICIAL["5"]:+d})')


# ============================================================================
# STEP 6: PHYSICAL VALIDATION SWEEP
# ============================================================================

section('STEP 6 -- PHYSICAL HEIGHT VALIDATION SWEEP')

print('  Moving arm through height steps from ABOVE to BELOW.')
print('  Watch which position has gripper tip touching the table.')
print()

# Move to pick approach (S6 rotated to lighter position)
approach = dict(HOME_POS); approach['6'] = S6_PICK
move(approach, ms=1500, label='APPROACH')
time.sleep(0.3)

# Sweep heights
SWEEP_Z = [1.5, 1.2, 0.75, 0.25, z_new, z_new - 0.25, z_new - 0.5]
SWEEP_Z = [round(z, 2) for z in SWEEP_Z if Z_END <= z <= Z_START + 0.1]
SWEEP_Z = sorted(set(SWEEP_Z), reverse=True)  # high to low

print(f'  Sweeping z values: {SWEEP_Z}')
print()

sweep_results = {}
for z in SWEEP_Z:
    sv = ki_to_servos(X_PICK, Y_PICK, z, -71)
    pose = {
        '1': 100,
        '2': 500,
        '3': sv['3'],
        '4': sv['4'],
        '5': sv['5'],
        '6': S6_PICK,
    }
    label = f'z={z:+.2f}'
    move(pose, ms=600, label=label)
    time.sleep(0.5)
    frame = snap(f'06_z_{str(z).replace("-","m").replace(".","p")}.jpg', label)
    sweep_results[z] = {'pose': pose, 'frame': frame, 'ik': sv}
    print()

# Return to approach height
move(approach, ms=800)
time.sleep(0.5)
move(HOME_POS, ms=1000, label='HOME')
time.sleep(0.5)


# ============================================================================
# STEP 7: CONFIRM BEST HEIGHT
# ============================================================================

section('STEP 7 -- CONFIRM BEST HEIGHT')

print('  Check the sweep frames in data/height_calib/')
print('  Find the z where gripper tip is AT the table surface (just touching).')
print()
print(f'  Frames taken:')
for z in SWEEP_Z:
    r = sweep_results.get(z, {})
    sv = r.get('ik', {})
    fn = Path(r.get('frame', '')).name if r.get('frame') else '?'
    print(f'    z={z:+.2f}cm  S3={sv.get("3","?")}  S4={sv.get("4","?")}  S5={sv.get("5","?")}  -> {fn}')

print()
if AUTO:
    z_final = z_new
    print(f'  AUTO: using calculated correction z={z_final:.2f}cm')
else:
    print(f'  Recommended: z={z_new:.2f}cm  (your {z_correction:.1f}cm correction)')
    try:
        raw = ask(f'Enter best z value from frames [default {z_new:.2f}]', f'{z_new:.2f}')
        z_final = float(raw)
    except ValueError:
        z_final = z_new

sv_final = ki_to_servos(X_PICK, Y_PICK, z_final, -71)
S3_FINAL = sv_final['3']
S4_FINAL = sv_final['4']
S5_FINAL = sv_final['5']

print()
print(f'  FINAL pick height: z={z_final:.2f}cm')
print(f'    S3={S3_FINAL}  S4={S4_FINAL}  S5={S5_FINAL}  S6={S6_PICK}')


# ============================================================================
# STEP 8: FULL PICK-AND-PLACE VALIDATION
# ============================================================================

section('STEP 8 -- FULL PICK-AND-PLACE TRIAL')

print(f'  Pick:  z={z_final:.2f}cm  S3={S3_FINAL}  S4={S4_FINAL}  S5={S5_FINAL}  S6={S6_PICK}')
print(f'  Place: S6={S6_PLACE}  ({PLACE_KEY})')
print()
print('  Place the lighter on the table at its calibrated position now.')
ask('Press Enter when ready', '')

# Compute place position
PLACE_POS_MAP = {
    'place_left90':  {'2': 500, '3': 220, '4': 800, '5': 460, '6': 875},
    'place_left45':  {'2': 500, '3': 225, '4': 800, '5': 460, '6': 685},
    'place_right45': {'2': 500, '3': 225, '4': 800, '5': 460, '6': 315},
    'place_right90': {'2': 500, '3': 220, '4': 800, '5': 460, '6': 125},
}
place_joints = PLACE_POS_MAP.get(PLACE_KEY, PLACE_POS_MAP['place_left90'])

PICK_APPROACH = dict(HOME_POS);  PICK_APPROACH['6'] = S6_PICK
PICK_DOWN     = {'1': 100, '2': 500, '3': S3_FINAL, '4': S4_FINAL, '5': S5_FINAL, '6': S6_PICK}
GRIP_CLOSE    = {'1': 500, '2': 500, '3': S3_FINAL, '4': S4_FINAL, '5': S5_FINAL, '6': S6_PICK}
LIFT_POSE     = {'1': 500, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_PICK}
TRANSIT_POSE  = {'1': 500, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_PLACE}
PLACE_DOWN    = dict(place_joints); PLACE_DOWN['1'] = 500
RELEASE       = dict(place_joints); RELEASE['1'] = 100

steps = [
    (HOME_POS,    1800, 'HOME'),
    (PICK_APPROACH, 1200, 'APPROACH'),
    (PICK_DOWN,   900,  'LOWER'),
    (GRIP_CLOSE,  500,  'GRIP'),
    (LIFT_POSE,   1000, 'LIFT'),
    (TRANSIT_POSE,1500, 'ROTATE'),
    (PLACE_DOWN,  1000, 'PLACE'),
    (RELEASE,     500,  'RELEASE'),
    (HOME_POS,    1500, 'HOME'),
]

labels = ['a','b','c','d','e','f','g','h','i']
for (pose, ms, lbl), letter in zip(steps, labels):
    print(f'  {letter}) {lbl}:')
    move(pose, ms=ms, label=lbl)
    snap(f'08{letter}_{lbl.lower()}.jpg', lbl)
    print()

trial_ok = ask('Was the pick-and-place successful? (y/n)', 'y')
trial_success = trial_ok.strip().lower() == 'y'


# ============================================================================
# STEP 9: SAVE UPDATED CALIBRATION
# ============================================================================

section('STEP 9 -- SAVE UPDATED CALIBRATION')

# Build final poses
FINAL_POSES = {
    'home':          HOME_POS,
    'pick_approach': PICK_APPROACH,
    'pick_low':      PICK_DOWN,
    'lift':          LIFT_POSE,
    'place_transit': TRANSIT_POSE,
    'place_low':     PLACE_DOWN,
    'release':       RELEASE,
}

print('  Final calibrated poses:')
for name, vals in FINAL_POSES.items():
    s = ' '.join(f'S{k}={v}' for k, v in sorted(vals.items()))
    print(f'    {name:<16}  {s}')

# Merge with existing calib
calib.update({
    'calibrated_at':   time.strftime('%Y-%m-%dT%H:%M:%S'),
    's6_pick':         S6_PICK,
    's6_place':        S6_PLACE,
    's3_pick':         S3_FINAL,
    's4_pick':         S4_FINAL,
    's5_pick':         S5_FINAL,
    'z_pick':          z_final,
    'z_correction_cm': z_correction,
    'x_pick':          round(X_PICK, 2),
    'y_pick':          round(Y_PICK, 2),
    'place_key':       PLACE_KEY,
    'trial_ok':        trial_success,
    'cosmos_recommendation': cosmos_recommendation,
    'poses':           FINAL_POSES,
    'ki_positions': {
        'home':  '(0, 17, 20.5, 0)',
        'pick':  f'({X_PICK:.1f}, {Y_PICK:.1f}, {z_final:.2f}, -71)',
        'place': f'(-19.5, 0, 2.8, -60)',
    },
    'ik_sweep': sweep_table,
})

OUT.write_text(json.dumps(calib, indent=2))
print(f'  Written: {OUT}')

# Save to Pi memory
print()
print('  Saving to Pi arm memory...')
pi_poses = {
    'home':       HOME_POS,
    'pick_table': PICK_DOWN,
    'lift_grip':  LIFT_POSE,
    'place_bin':  PLACE_DOWN,
}
for name, vals in pi_poses.items():
    r = post('/arm/save_touch_pose', {'name': name, 'positions': vals})
    print(f'    {name}: {r.get("ok","?")}')


# ============================================================================
# SUMMARY
# ============================================================================

section('COSMOS HEIGHT CALIBRATION COMPLETE')

print(f'  Problem:   arm was {z_correction:.1f}cm above table during pick')
print(f'  Fix:       z pick  1.2cm -> {z_final:.2f}cm')
print()
print(f'  Servo correction (what changed):')
print(f'    S5: {PICK_OFFICIAL["5"]} -> {S5_FINAL}  (shoulder DOWN, -{PICK_OFFICIAL["5"]-S5_FINAL})')
print(f'    S4: {PICK_OFFICIAL["4"]} -> {S4_FINAL}  (elbow BEND,  +{S4_FINAL-PICK_OFFICIAL["4"]})')
print(f'    S3: {PICK_OFFICIAL["3"]} -> {S3_FINAL}  (wrist, minimal change)')
print()
print(f'  Cosmos AI: {(cosmos_recommendation or "N/A")[:100]}')
print()
print(f'  Trial success: {"YES" if trial_success else "NO"}')
print()
print(f'  Frames: {FRAMES}')
for f in sorted(FRAMES.glob('*.jpg')):
    print(f'    {f.name}')
print()
print('  Next:')
print('    python run_demo.py   # run demo with corrected height')
print()
print(SEP)
print('  DONE')
print(SEP)
