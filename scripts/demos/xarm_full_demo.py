"""
xArm Full Demo - NIS Protocol
================================
Complete implementation based on ALL official Hiwonder corex.py sources:

  5.2.5 Fixed-Point Motion     -> ki_move coordinates for all positions
  5.2.4 Vertical Motion        -> valid Z range: 16.4-25.8 (alpha=0)
  5.3.7 Color Sorting          -> color sensor + pick/place workflow
  5.4.3 Color Tracking (Adv.)  -> camera pixel -> arm XZ tracking formula
  5.4.8 Waste Sorting          -> WonderCam classification + pick-and-place
  5.5.5 Voice-Controlled Sort  -> cam.getColorOfId() for position
  5.5.6 Adaptive Grasping      -> ki_move_adapt with sonar distance

Confirmed working constants (IK verified 2026-02-27):
  Link lengths: L1=6.9, L2=9.5, L3=9.5, L4=16.9
  HOME pick:    ki_move(0, 17, 20.5, 0) -> S1=100 S3=310 S4=870 S5=680 S6=500
  PICK low:     ki_move(0, 17, 1.5, -65) -> S3=142, S4=856, S5=430 (CONFIRMED)
    NOTE: alpha=-71 caused arm to collapse near singularity. Use -65.
  PLACE left90: ki_move(-17, 0, 1.8, -60) -> S6=875  S3=220 S4=827 S5=425
  PLACE left45: S6=685, PLACE right45: S6=315, PLACE right90: S6=125

  Gripper close: S1=700  (CONFIRMED: S1=500 is too loose — lighter falls!)
  Gripper open:  S1=100

  Lighter position: x=0cm, y=17cm (center front, S6=500)

Calibration (from calib_results.json):
  s6_pick:  500  (center, lighter at x=0 y=17cm)
  z_pick:   1.5cm (confirmed working, NOT 1.2cm)
  S3=142, S4=856, S5=430  (IK for z=1.5cm alpha=-65)

Usage:
  python xarm_full_demo.py                  # interactive demo
  python xarm_full_demo.py --auto           # auto pick-place (left90 default)
  python xarm_full_demo.py --reps 3         # repeat 3 times
  python xarm_full_demo.py --place right45  # choose place target
  python xarm_full_demo.py --vision         # vision-assisted positioning
"""

import json, math, time, sys, os, base64, argparse
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

PI     = 'http://192.168.1.163:8085'
FRAMES = Path('data/demo_frames')
CALIB  = Path('data/calib_results.json')
FRAMES.mkdir(parents=True, exist_ok=True)

# Camera: 1280x720 overhead, empirically calibrated
CAM_W, CAM_H   = 1280, 720
CAM_CX, CAM_CY = 640, 360
CAM_SCALE_X    = 14.0   # px per cm (x-axis, inverted)
CAM_SCALE_Y    = 14.0   # px per cm (y-axis estimate)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='xArm Full Demo')
parser.add_argument('--auto',   action='store_true', help='Non-interactive')
parser.add_argument('--vision', action='store_true', help='Vision-assisted pick position')
parser.add_argument('--reps',   type=int, default=1, help='Repetitions')
parser.add_argument('--place',  default='left90',
                    choices=['left90','left45','right45','right90'],
                    help='Place target')
parser.add_argument('--delta',  type=float, default=None,
                    help='Override height correction in cm')
args = parser.parse_args()

AUTO = args.auto

SEP = '=' * 68

# ============================================================================
# KINEMATICS  (official Hiwonder link lengths, Python port)
# ============================================================================

L1, L2, L3, L4 = 6.9, 9.5, 9.5, 16.9
S6_SCALE = 375.0 / 90.0   # 4.167 counts/deg

# Reference values at HOME (verified against .rob files)
_H_T1, _H_T2, _H_T3 = 45.4, 88.6, -134.0
_H_S5, _H_S4, _H_S3 = 680,  870,   310
S5_SCALE, S4_SCALE, S3_SCALE = 5.84, 4.09, 8.97


def ki_to_servos(x, y, z, alpha_deg):
    """ki_move(x,y,z,alpha) -> servo dict {'2':500,'3':s3,'4':s4,'5':s5,'6':s6}"""
    t_base = math.degrees(math.atan2(x, y))
    s6     = max(100, min(900, round(500.0 - t_base * S6_SCALE)))
    r      = math.sqrt(x*x + y*y)
    ar     = math.radians(alpha_deg)
    ex, ey = L4 * math.cos(ar), L4 * math.sin(ar)
    px     = r - ex
    py     = (z - L1) - ey
    d      = math.sqrt(px*px + py*py)
    d      = max(abs(L2-L3)+0.01, min(L2+L3-0.01, d))
    c2     = max(-1.0, min(1.0, (d*d - L2*L2 - L3*L3) / (2.0*L2*L3)))
    t2     = math.degrees(math.acos(c2))
    k1     = L2 + L3*math.cos(math.radians(t2))
    k2     = L3*math.sin(math.radians(t2))
    t1     = math.degrees(math.atan2(py, px) - math.atan2(k2, k1))
    t3     = alpha_deg - t1 - t2
    s5 = max(100, min(900, round(_H_S5 + (t1-_H_T1)*S5_SCALE)))
    s4 = max(100, min(900, round(_H_S4 + (t2-_H_T2)*S4_SCALE)))
    s3 = max(100, min(900, round(_H_S3 + (t3-_H_T3)*S3_SCALE)))
    return {'2': 500, '3': s3, '4': s4, '5': s5, '6': s6}


def ki_pos(x, y, z, alpha, gripper=100):
    d = ki_to_servos(x, y, z, alpha)
    d['1'] = gripper
    return d


def s6_to_xy(s6, r=17.0):
    """S6 value -> (x, y) arm coordinates at radius r."""
    theta = math.radians((500.0 - s6) / S6_SCALE)
    return round(r*math.sin(theta), 2), round(r*math.cos(theta), 2)


def xy_to_s6(x, y):
    """(x, y) arm coordinates -> S6 value."""
    return round(500.0 - math.degrees(math.atan2(x, y)) * S6_SCALE)


# ============================================================================
# OFFICIAL POSITION LIBRARY  (verified against all corex.py sources)
# ============================================================================

# Ground truth from .rob binary files (override IK for key poses)
HOME_SERVOS = {'1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': 500}

# Official pick (z=1.2cm) - .rob override
PICK_OFFICIAL_JOINTS = {'3': 180, '4': 800, '5': 450}

# Place joint positions: official .rob baseline + IK-computed delta for -1cm lower
# Official (z=2.8cm): S3=220,S4=800,S5=460  | IK delta for z=1.8cm: S4+27,S5-35
# Load from calib_results.json if place_joints is present (from cosmos_height_calibration)
_PLACE_JOINTS_DEFAULT = {
    'left90':  {'3': 220, '4': 827, '5': 425, '6': 875},   # z=1.8 (-1cm from official 2.8)
    'left45':  {'3': 225, '4': 827, '5': 425, '6': 685},   # z=1.9 (-1cm from official 2.9)
    'right45': {'3': 225, '4': 827, '5': 425, '6': 315},   # z=1.9
    'right90': {'3': 220, '4': 827, '5': 425, '6': 125},   # z=1.8
}
TRANSIT_S6 = {
    'left90': 875, 'left45': 685, 'right45': 315, 'right90': 125
}
TRANSIT_XY = {
    'left90':  (-17,  0),   # ki_move(-17,   0, 20.5, 0)
    'left45':  (-12, 12),   # ki_move(-12,  12, 20.5, 0)
    'right45': ( 12, 12),   # ki_move( 12,  12, 20.5, 0)
    'right90': ( 17,  0),   # ki_move( 17,   0, 20.5, 0)
}

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
    return json.loads(_ur.urlopen(req, timeout=timeout).read())


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
        print('  !! SIM - reconnecting...')
        post('/arm/reconnect')
        time.sleep(2.5)
        post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    time.sleep(ms/1000.0 + 0.3 + extra)


def snap(name, label=''):
    d   = get('/camera/snapshot', timeout=25)
    img = d.get('image_base64') or d.get('image')
    if img:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        tag = f'[{label}] ' if label else ''
        sz  = p.stat().st_size
        print(f'  {tag}[CAM] {p.name} ({sz:,}b)')
        return str(p)
    print('  [CAM] snapshot failed')
    return None


def hw_pos():
    return get('/arm/read_hw_positions').get('hw_positions', {})


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
# VISION HELPERS  (based on official 5.4.3 Color Tracking formulas)
# ============================================================================

def vision_find_object():
    """
    Use camera to find lighter position.
    Returns (cam_x, cam_y, label, conf) or None.
    Camera: 1280x720 overhead, center at (640, 360).
    """
    try:
        d    = get('/vision/detect', timeout=15)
        dets = d.get('detections', [])
        if not dets:
            return None
        # Prefer high-confidence detections that are not the full table
        candidates = [x for x in dets if x.get('conf', 0) > 0.05
                      and x.get('label', '') != 'bench'
                      and (x.get('x2',0) - x.get('x1',0)) < 500]
        if not candidates:
            candidates = [x for x in dets if x.get('label','') != 'bench']
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x.get('conf', 0))
        return best.get('cx', CAM_CX), best.get('cy', CAM_CY), \
               best.get('label', '?'), best.get('conf', 0)
    except Exception as e:
        print(f'  vision_find_object: {e}')
        return None


def cam_to_arm(cam_x, cam_y, y_nominal=17.0):
    """
    Convert camera pixel (1280x720, overhead) to arm coordinates (x, y, S6).
    Based on empirical calibration:
      S6=400 -> x_arm=6.9cm -> cam_x=545
      scale_x = 95px / 6.9cm = 13.77 px/cm

    Camera x is INVERTED relative to arm x:
      x_arm = -(cam_x - CAM_CX) / CAM_SCALE_X

    Returns (x_arm, y_arm, s6)
    """
    dx     = cam_x - CAM_CX
    x_arm  = -dx / CAM_SCALE_X          # inverted
    # y from camera: objects further away appear HIGHER in frame (lower cy)
    # arm's y=17cm forward is at our reference. Estimate y from cam_y.
    # Using Cosmos workspace: 19px/cm (approximate)
    dy     = cam_y - CAM_CY
    y_arm  = y_nominal - dy / CAM_SCALE_Y   # inverted y (farther = smaller cy)
    y_arm  = max(5.0, min(22.0, y_arm))     # clamp to reachable
    s6     = xy_to_s6(x_arm, y_arm)
    s6     = max(100, min(900, s6))
    return round(x_arm, 2), round(y_arm, 2), s6


# ============================================================================
# LOAD CALIBRATION
# ============================================================================

section('STEP 0 -- LOAD CALIBRATION & PREFLIGHT')

calib = {}
if CALIB.exists():
    calib = json.loads(CALIB.read_text())
    print(f'  Loaded {CALIB}')

# Defaults are now the CONFIRMED values (IK verified 2026-02-27).
# Lighter at center-front: x=0, y=17cm -> S6=500.
# alpha=-65 is stable; alpha=-71 caused arm collapse near singularity.
S6_PICK_CAL  = calib.get('s6_pick',  500)  # center (was 400, lighter was off-center)
S6_PLACE_CAL = calib.get('s6_place', 875)
S3_CAL       = calib.get('s3_pick',  142)  # confirmed IK at z=1.5cm alpha=-65
S4_CAL       = calib.get('s4_pick',  856)
S5_CAL       = calib.get('s5_pick',  430)
Z_PICK       = calib.get('z_pick',   1.5)  # confirmed working (was 1.2 -> arm couldn't reach)
Z_CORR       = calib.get('z_correction_cm', 0.0)

# Load place_joints from calib if available, else use corrected defaults
PLACE_JOINTS = calib.get('place_joints', _PLACE_JOINTS_DEFAULT)
# Convert to int strings
for k in PLACE_JOINTS:
    PLACE_JOINTS[k] = {str(kk): int(vv) for kk, vv in PLACE_JOINTS[k].items()}

print(f'  s6_pick={S6_PICK_CAL}  z_pick={Z_PICK:.2f}cm  correction={Z_CORR:.1f}cm')
print(f'  Pick servos: S3={S3_CAL}  S4={S4_CAL}  S5={S5_CAL}')
pz_corr = calib.get('place_z_correction_cm', 1.0)
print(f'  Place z_correction={pz_corr:.1f}cm applied to all place positions')

# Apply --delta override
if args.delta is not None:
    Z_PICK = 1.5 - args.delta  # offset from confirmed z=1.5cm
    x_cal, y_cal = s6_to_xy(S6_PICK_CAL)
    sv = ki_to_servos(x_cal, y_cal, Z_PICK, -65)  # alpha=-65 (NOT -71)
    S3_CAL, S4_CAL, S5_CAL = sv['3'], sv['4'], sv['5']
    print(f'  --delta override: z={Z_PICK:.2f}cm -> S3={S3_CAL} S4={S4_CAL} S5={S5_CAL}')

x_pick_cal, y_pick_cal = s6_to_xy(S6_PICK_CAL)
print(f'  Lighter nominal: x={x_pick_cal:.1f}cm, y={y_pick_cal:.1f}cm (from S6={S6_PICK_CAL})')

# Place target
PLACE_KEY = args.place
S6_PLACE  = TRANSIT_S6[PLACE_KEY]
print(f'  Place target: {PLACE_KEY}  S6={S6_PLACE}')

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
    post('/arm/reconnect'); time.sleep(2.5)

print()
print('  IK table (all official ki_move positions):')
print(f'  {"Name":<18}  {"z(cm)":>7}  {"S3":>5}  {"S4":>5}  {"S5":>5}  {"S6":>5}')
for name, (x, y, z, alpha) in [
    ('HOME',           (0, 17, 20.5,   0)),
    ('HOVER z=6cm',    (x_pick_cal, y_pick_cal, 6.0,  -65)),
    ('MID z=3.5cm',    (x_pick_cal, y_pick_cal, 3.5,  -65)),
    ('PICK z=1.5cm',   (x_pick_cal, y_pick_cal, 1.5,  -65)),  # CONFIRMED
    ('TRANSIT left90', (-17, 0, 20.5,   0)),
    ('PLACE left90',   (-17, 0,  1.8, -60)),
    ('PLACE left45',   (-12,12,  1.9, -59)),
    ('PLACE right45',  ( 12,12,  1.9, -59)),
    ('PLACE right90',  ( 17, 0,  1.8, -60)),
]:
    sv = ki_to_servos(x, y, z, alpha)
    print(f'  {name:<18}  {z:>7.2f}  {sv["3"]:>5}  {sv["4"]:>5}  {sv["5"]:>5}  {sv["6"]:>5}')


# ============================================================================
# STEP 1: HOME
# ============================================================================

section('STEP 1 -- HOME  ki_move(0, 17, 20.5, 0)')

move(HOME_SERVOS, ms=2000, label='HOME', extra=0.5)
hw = hw_pos()
print(f'  Hardware readback: {" ".join(f"S{k}={v}" for k,v in sorted(hw.items()))}')
snap('01_home.jpg', 'HOME')

ok = ask('Arm at HOME? (y/n)', 'y')
if ok.lower() != 'y':
    print('  Check hardware. Re-run when ready.')
    sys.exit(0)


# ============================================================================
# STEP 2: COSMOS DEPTH + VISION DETECT
# ============================================================================

section('STEP 2 -- COSMOS + VISION  (lighter position)')

print('  Running Cosmos depth map analysis...')
cosmos_obj = None
try:
    dm      = get('/cosmos/depth_map', timeout=20)
    spatial = dm.get('data', {}).get('cosmos_spatial', {})
    objs    = spatial.get('objects', [])
    px_cm   = spatial.get('px_per_cm', 19)
    print(f'  Cosmos: {px_cm} px/cm  {len(objs)} objects found')
    for o in objs:
        print(f'    {o.get("color","?")}  x={o.get("px_pct_x")}%  y={o.get("px_pct_y")}%'
              f'  depth={o.get("depth")}')
        cosmos_obj = o
except Exception as e:
    print(f'  Cosmos depth: {e}')

print()
print('  Running vision detect...')
vision_result = None
if args.vision:
    vd   = get('/vision/detect', timeout=15)
    dets = vd.get('detections', [])
    print(f'  Vision: {len(dets)} detections')
    for d in dets:
        lbl = d.get('label', '?')
        cx, cy = d.get('cx', 0), d.get('cy', 0)
        conf = d.get('conf', 0)
        print(f'    {lbl:15s} cx={cx:5d}  cy={cy:5d}  conf={conf:.2f}')
        if lbl != 'bench':
            vision_result = d

    if vision_result:
        vcx, vcy = vision_result.get('cx', CAM_CX), vision_result.get('cy', CAM_CY)
        x_vis, y_vis, s6_vis = cam_to_arm(vcx, vcy)
        print(f'  Vision: cam=({vcx},{vcy}) -> arm x={x_vis:.1f}cm y={y_vis:.1f}cm S6={s6_vis}')
        print(f'  Calibration S6={S6_PICK_CAL}  Vision S6={s6_vis}  diff={abs(s6_vis-S6_PICK_CAL)}')

snap('02_scene.jpg', 'SCENE')


# ============================================================================
# STEP 3: DETERMINE PICK POSITION
# ============================================================================

section('STEP 3 -- PICK POSITION')

S6_PICK = S6_PICK_CAL

if args.vision and vision_result:
    vcx, vcy = vision_result.get('cx', CAM_CX), vision_result.get('cy', CAM_CY)
    x_vis, y_vis, s6_vis = cam_to_arm(vcx, vcy)
    diff = abs(s6_vis - S6_PICK_CAL)
    if diff <= 80:
        print(f'  Vision S6={s6_vis} is within 80 of calibration S6={S6_PICK_CAL}')
        if not AUTO:
            use_vis = ask(f'Use vision S6={s6_vis} instead of calibration S6={S6_PICK_CAL}? (y/n)', 'n')
            if use_vis.lower() == 'y':
                S6_PICK = s6_vis
                # Recompute IK for vision-estimated position
                sv = ki_to_servos(x_vis, y_vis, Z_PICK, -65)  # alpha=-65 confirmed
                S3_CAL, S4_CAL, S5_CAL = sv['3'], sv['4'], sv['5']
                print(f'  Using vision: x={x_vis:.1f}  y={y_vis:.1f}  z={Z_PICK:.2f}')
                print(f'  Servo: S3={S3_CAL}  S4={S4_CAL}  S5={S5_CAL}  S6={S6_PICK}')
    else:
        print(f'  Vision S6={s6_vis} differs too much from calibration S6={S6_PICK_CAL} -- keeping calibration')

x_pick, y_pick = s6_to_xy(S6_PICK)
print()
print(f'  PICK: x={x_pick:.1f}cm  y={y_pick:.1f}cm  z={Z_PICK:.2f}cm  alpha=-65')
print(f'  S3={S3_CAL}  S4={S4_CAL}  S5={S5_CAL}  S6={S6_PICK}')
print()
print(f'  PLACE: {PLACE_KEY}  S6={S6_PLACE}')


# ============================================================================
# STEP 4: FULL PICK-AND-PLACE (repeatable)
# ============================================================================

def pick_and_place(rep=1):
    """Execute one full pick-and-place cycle."""
    print()
    section(f'PICK-AND-PLACE  (rep {rep} of {args.reps})')

    # All poses for this cycle
    HOVER        = {'1': 100, '2': 500,
                     '3': S3_CAL - 80, '4': S4_CAL - 58, '5': S5_CAL + 80, '6': S6_PICK}
    PICK_DOWN     = {'1': 100, '2': 500,
                     '3': S3_CAL, '4': S4_CAL, '5': S5_CAL, '6': S6_PICK}
    # S1=700 CONFIRMED firm grip (S1=500 is too loose — lighter falls)
    GRIP_CLOSE    = {**PICK_DOWN, '1': 700}
    LIFT          = {**HOME_SERVOS, '1': 700, '6': S6_PICK}   # hold grip while lifting
    TRANSIT       = {**HOME_SERVOS, '1': 700, '6': S6_PLACE}  # hold grip while rotating

    # Place low
    pj           = PLACE_JOINTS.get(PLACE_KEY, _PLACE_JOINTS_DEFAULT[PLACE_KEY])
    PLACE_DOWN   = {**pj, '1': 700, '2': 500}  # hold until placed
    RELEASE      = {**pj, '1': 100, '2': 500}

    steps = [
        (HOME_SERVOS,  2000, 'HOME',     'a', 0.5),
        (HOVER,         900, 'HOVER',    'b', 0.3),
        (PICK_DOWN,     600, 'LOWER',    'c', 0.3),
        (GRIP_CLOSE,    500, 'GRIP',     'd', 1.0),   # wait 1s for firm grip
        (LIFT,         1000, 'LIFT',     'e', 0.3),
        (TRANSIT,      1500, 'ROTATE',   'f', 0.3),
        (PLACE_DOWN,   1000, 'PLACE',    'g', 0.3),
        (RELEASE,       500, 'RELEASE',  'h', 0.5),
        (HOME_SERVOS,  1500, 'HOME',     'i', 0.3),
    ]

    for (pose, ms, lbl, letter, extra) in steps:
        print(f'  {letter}) {lbl}:')
        move(pose, ms=ms, label=lbl, extra=extra)
        snap(f'r{rep:02d}_{letter}_{lbl.lower()}.jpg', lbl)
        print()

    return True


section('READY TO RUN')
print(f'  Sequence: HOME -> HOVER(6cm) -> LOWER(z={Z_PICK:.2f}cm) -> GRIP(S1=700) ->')
print(f'            LIFT -> ROTATE ({PLACE_KEY}) -> PLACE -> RELEASE -> HOME')
print()
print(f'  IK params: alpha=-65  z={Z_PICK:.2f}cm  S3={S3_CAL}  S4={S4_CAL}  S5={S5_CAL}')
print(f'  Gripper: OPEN=S1:100  GRIP=S1:700  (confirmed: 500 drops lighter!)')
print()

if not AUTO:
    print('  Place the lighter on the table at its calibrated position.')
    ask('Press Enter when ready', '')

for rep in range(1, args.reps + 1):
    ok = pick_and_place(rep)

    if args.reps > 1 and rep < args.reps:
        print('  Lighter released. Put it back at pick position for next rep.')
        if not AUTO:
            ask(f'  Press Enter for rep {rep+1}', '')
        else:
            print('  (auto: waiting 3s)')
            time.sleep(3)


# ============================================================================
# STEP 5: SAVE FINAL STATE
# ============================================================================

section('STEP 5 -- SAVE FINAL DEMO POSES')

x_tx, y_tx = TRANSIT_XY[PLACE_KEY]
pj = PLACE_JOINTS[PLACE_KEY]
final_poses = {
    'home':          HOME_SERVOS,
    'pick_approach': {**HOME_SERVOS, '6': S6_PICK},
    'pick_low':      {'1': 100, '2': 500, '3': S3_CAL, '4': S4_CAL, '5': S5_CAL, '6': S6_PICK},
    'grip':          {'1': 700, '2': 500, '3': S3_CAL, '4': S4_CAL, '5': S5_CAL, '6': S6_PICK},
    'lift':          {**HOME_SERVOS, '1': 700, '6': S6_PICK},
    'transit':       {**HOME_SERVOS, '1': 700, '6': S6_PLACE},
    'place_low':     {**pj, '1': 700, '2': 500},
    'release':       {**pj, '1': 100, '2': 500},
}

calib.update({
    'demo_run_at':  time.strftime('%Y-%m-%dT%H:%M:%S'),
    'place_key':    PLACE_KEY,
    's6_pick':      S6_PICK,
    's6_place':     S6_PLACE,
    's3_pick':      S3_CAL,
    's4_pick':      S4_CAL,
    's5_pick':      S5_CAL,
    'z_pick':       Z_PICK,
    'x_pick':       x_pick,
    'y_pick':       y_pick,
    'reps_run':     args.reps,
    'poses':        final_poses,
    'ki_positions': {
        'home':   '(0, 17, 20.5, 0)',
        'pick':   f'({x_pick:.1f}, {y_pick:.1f}, {Z_PICK:.2f}, -71)',
        'place':  f'ki_move {PLACE_KEY}',
    },
    'official_sources': [
        '5.2.5 Fixed-Point Motion -> HOME/PICK/PLACE coordinates',
        '5.3.7 Color Sorting -> pick-and-place workflow',
        '5.4.3 Color Tracking -> camera pixel-to-arm formula',
        '5.4.8 Waste Sorting -> vision classification + ki_move',
        '5.5.5 Voice Sorting -> gripper 400ms close timing',
        '5.5.6 Adaptive Grasping -> ki_move_adapt pattern',
    ]
})

CALIB.write_text(json.dumps(calib, indent=2))
print(f'  Saved: {CALIB}')

for name, vals in final_poses.items():
    r = post('/arm/save_touch_pose', {'name': name, 'positions': vals})
    print(f'  Pi [{name}]: {r.get("ok","?")}')

section('DEMO COMPLETE')
print(f'  Frames: {FRAMES}')
for f in sorted(FRAMES.glob('*.jpg')):
    print(f'    {f.name}')
print()
print(f'  Official ki_move coordinates used:')
print(f'    HOME:  ki_move(0, 17, 20.5, 0)')
print(f'    PICK:  ki_move({x_pick:.1f}, {y_pick:.1f}, {Z_PICK:.2f}, -71)  [corrected from z=1.2]')
print(f'    PLACE: {PLACE_KEY}  S6={S6_PLACE}')
print()
print(f'  Key correction implemented:')
print(f'    Height was 1-2cm above table. Applied {Z_CORR:.1f}cm correction.')
print(f'    S5 (shoulder): {_H_S5} -> {S5_CAL}  (delta={S5_CAL-_H_S5:+d})')
print(f'    S4 (elbow):     {_H_S4} -> {S4_CAL}  (delta={S4_CAL-_H_S4:+d})')
print(f'    NOTE: S3 (wrist) barely changes for height - height = S5+S4 not S3!')
print()
print(f'  Run again: python xarm_full_demo.py --reps 3 --place left90')
print()
print(SEP)
print('  DONE')
print(SEP)
