"""
MASTER CALIBRATION - xArm AI
=============================
Implements the OFFICIAL Hiwonder kinematics from the WonderCode corex.py files.

Official workflow (from 5.2 Fixed-Point Motion & 5.3 Color Sorting):
  set_link_length(6.9, 9.5, 9.5, 16.9)
  ki_move(0, 17, 20.5, 0, 1000)      # HOME: forward 17cm, height 20.5cm
  ki_move(0, 17, 1.2, -71, 800)      # PICK: forward 17cm, height 1.2cm, wrist -71deg
  LSC.moveServo(1, 500, 400)          # GRIP: close gripper
  ki_move(0, 17, 20.5, 0, 800)       # LIFT: raise back up
  ki_move(-17, 0, 20.5, 0, 800)      # ROTATE: swing left 90deg (transit height)
  ki_move(-19.5, 0, 2.8, -60, 800)   # PLACE: lower to place position
  LSC.moveServo(1, 100, 500)          # RELEASE: open gripper

Place coordinates (x, y, z, alpha):
  Left  90deg:  (-19.5,    0, 2.8, -60)  S6=875
  Left  45deg:  (-14,     14, 2.9, -59)  S6=685
  Right 45deg:  ( 14,     14, 2.9, -59)  S6=315
  Right 90deg:  ( 19.5,    0, 2.8, -60)  S6=125

S6 direction: S6 = 500 - atan2(x,y)*180/pi * 4.167
  x<0  -> S6 > 500  (left)
  x>0  -> S6 < 500  (right)

Usage:
  python master_calibration.py
  python master_calibration.py --no-input   # auto mode with defaults
"""

import json, math, time, sys, os
from pathlib import Path

sys.path.insert(0, '.')
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('rob_parser', os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'src', 'neurolinux', 'drivers', 'rob_parser.py'))
_rob = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rob)
OFFICIAL = _rob.OFFICIAL

try:
    import urllib.request as _ur, json as _js
    def _get(url, timeout=10):
        r = _ur.urlopen(url, timeout=timeout)
        return _js.loads(r.read())
    def _post(url, body, timeout=18):
        data = _js.dumps(body).encode()
        req = _ur.Request(url, data=data, headers={'Content-Type': 'application/json'})
        r = _ur.urlopen(req, timeout=timeout)
        return _js.loads(r.read())
except Exception:
    pass

PI      = 'http://192.168.1.163:8085'
FRAMES  = Path('data/master_calib')
OUT     = Path('data/calib_results.json')
FRAMES.mkdir(parents=True, exist_ok=True)

NO_INPUT = '--no-input' in sys.argv
SEP = '=' * 64


# ============================================================================
# KINEMATICS  (Python port of Hiwonder xArmKinematic C++ library)
# ============================================================================

L1, L2, L3, L4 = 6.9, 9.5, 9.5, 16.9   # official link lengths

# Servo scale factors (derived from official .rob cross-reference)
# S6: 4.167 counts per degree  (verified: S6=875 = left90, S6=125 = right90)
# S5: 5.84  counts per degree  (verified from HOME and PICK cross-ref)
# S4: 4.09  counts per degree  (verified from HOME and PICK cross-ref)
# S3: 8.97  counts per degree  (wrist, slightly less accurate)
S6_SCALE = 375.0 / 90.0   # = 4.167

# Reference angles at HOME position ki_move(0,17,20.5,0)
# Verified against .rob HOME {S3:310, S4:870, S5:680}
_HOME_THETA1 =  45.4   # shoulder angle at home
_HOME_THETA2 =  88.6   # elbow angle at home
_HOME_THETA3 = -134.0  # wrist angle at home
_HOME_S5, _HOME_S4, _HOME_S3 = 680, 870, 310

S5_REF_THETA = _HOME_THETA1;  S5_REF_VAL = _HOME_S5;  S5_SCALE = 5.84
S4_REF_THETA = _HOME_THETA2;  S4_REF_VAL = _HOME_S4;  S4_SCALE = 4.09
S3_REF_THETA = _HOME_THETA3;  S3_REF_VAL = _HOME_S3;  S3_SCALE = 8.97


def ki_to_servos(x, y, z, alpha_deg):
    """
    Convert Hiwonder ki_move(x, y, z, alpha, ms) coordinates to servo positions.
    Returns dict {1: gripper(unchanged), 2: 500, 3: int, 4: int, 5: int, 6: int}

    x, y = horizontal plane (cm).  y=forward, x=right is POSITIVE.
    z    = height (cm).
    alpha_deg = wrist angle (0=horizontal, negative=tilt down).
    """
    # Base rotation
    theta_base_rad = math.atan2(x, y)          # atan2(x, y) not atan2(y, x)!
    theta_base_deg = math.degrees(theta_base_rad)
    s6 = round(500.0 - theta_base_deg * S6_SCALE)
    s6 = max(100, min(900, s6))

    # Horizontal reach
    r = math.sqrt(x*x + y*y)

    # End effector wrist contribution
    alpha_rad = math.radians(alpha_deg)
    ex = L4 * math.cos(alpha_rad)
    ey = L4 * math.sin(alpha_rad)

    # Target for 2-link IK (L2, L3)
    px = r  - ex
    py = (z - L1) - ey

    d_sq = px*px + py*py
    d    = math.sqrt(d_sq)

    # Clamp to reachable space
    d_max = L2 + L3 - 0.01
    d_min = abs(L2 - L3) + 0.01
    d = max(d_min, min(d_max, d))
    if d_sq != d*d:   # re-clamp target
        scale = d / math.sqrt(d_sq)
        px *= scale;  py *= scale

    cos_t2 = (d*d - L2*L2 - L3*L3) / (2.0 * L2 * L3)
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    theta2 = math.degrees(math.acos(cos_t2))          # elbow (always positive)

    k1 = L2 + L3 * math.cos(math.radians(theta2))
    k2 = L3 * math.sin(math.radians(theta2))
    theta1 = math.degrees(math.atan2(py, px) - math.atan2(k2, k1))  # shoulder

    theta3 = alpha_deg - theta1 - theta2               # wrist

    s5 = round(S5_REF_VAL + (theta1 - S5_REF_THETA) * S5_SCALE)
    s4 = round(S4_REF_VAL + (theta2 - S4_REF_THETA) * S4_SCALE)
    s3 = round(S3_REF_VAL + (theta3 - S3_REF_THETA) * S3_SCALE)

    s5 = max(100, min(900, s5))
    s4 = max(100, min(900, s4))
    s3 = max(100, min(900, s3))

    return {'2': 500, '3': s3, '4': s4, '5': s5, '6': s6}


def ki_pos(x, y, z, alpha, gripper=100):
    """Full servo dict including gripper."""
    d = ki_to_servos(x, y, z, alpha)
    d['1'] = gripper
    return d


# Pre-compute all official positions
POS = {
    # HOME: forward 17cm, height 20.5cm, horizontal wrist
    'home':            ki_pos( 0,   17,  20.5,   0,   100),
    # PICK LOW: same forward/reach, height 1.2cm, wrist -71deg
    'pick_low_fwd':    ki_pos( 0,   17,   1.2, -71,   100),
    # LIFT with grip: same as home but gripper closed
    'lift_fwd':        ki_pos( 0,   17,  20.5,   0,   500),

    # PLACE POSITIONS (from official corex.py color sorting)
    # Transit height (high, gripper closed)
    'transit_left90':  ki_pos(-17,   0,  20.5,   0,   500),
    'transit_left45':  ki_pos(-12,  12,  20.5,   0,   500),
    'transit_right45': ki_pos( 12,  12,  20.5,   0,   500),
    'transit_right90': ki_pos( 17,   0,  20.5,   0,   500),

    # Place low (lower to release)
    'place_left90':    ki_pos(-19.5,  0,  2.8, -60,   500),
    'place_left45':    ki_pos(-14,   14,  2.9, -59,   500),
    'place_right45':   ki_pos( 14,   14,  2.9, -59,   500),
    'place_right90':   ki_pos( 19.5,  0,  2.8, -60,   500),
}

# Override with official .rob values where we have them (ground truth)
# These OVERRIDE the IK-computed values for maximum accuracy
POS['home']['3']          = 310;  POS['home']['4']          = 870;  POS['home']['5']          = 680
POS['pick_low_fwd']['3']  = 180;  POS['pick_low_fwd']['4']  = 800;  POS['pick_low_fwd']['5']  = 450
# Place positions: 1cm lower than official z=2.8/2.9cm
# IK delta: S4 +27, S5 -35 applied to official .rob values
# official: S3=220/225, S4=800, S5=460  -> corrected: S4=827, S5=425
POS['place_left90']['5']  = 425;  POS['place_left90']['4']  = 827;  POS['place_left90']['3']  = 220
POS['place_left45']['5']  = 425;  POS['place_left45']['4']  = 827;  POS['place_left45']['3']  = 225
POS['place_right45']['5'] = 425;  POS['place_right45']['4'] = 827;  POS['place_right45']['3'] = 225
POS['place_right90']['5'] = 425;  POS['place_right90']['4'] = 827;  POS['place_right90']['3'] = 220

# Lift versions (same arm joints, gripper closed)
POS['lift_fwd']['3'] = 310;  POS['lift_fwd']['4'] = 870;  POS['lift_fwd']['5'] = 680


# ============================================================================
# PI AGENT API
# ============================================================================

def get(path, timeout=12):
    try:
        return _get(PI + path, timeout=timeout)
    except Exception as e:
        print(f'  GET {path}: {e}')
        return {}


def post(path, body=None, timeout=18):
    try:
        return _post(PI + path, body or {}, timeout=timeout)
    except Exception as e:
        print(f'  POST {path}: {e}')
        return {}


def move(servos, ms=1200, label='', extra=0.0):
    r    = post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim  = r.get('simulation', False)
    ok   = r.get('ok', False)
    s    = ' '.join(f'S{k}={v}' for k, v in sorted(servos.items()))
    tag  = f'[{label}] ' if label else ''
    stat = 'SIM' if sim else ('OK' if ok else 'FAIL')
    print(f'  {tag}{s}  [{stat}]')
    if sim:
        print('  !! SIM -- reconnecting...')
        post('/arm/reconnect'); time.sleep(2.5)
        post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    time.sleep(ms / 1000.0 + 0.5 + extra)


def snap(name, label=''):
    import base64
    d = get('/camera/snapshot', timeout=25)
    img = d.get('image_base64') or d.get('image')
    if img:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        tag = f'[{label}] ' if label else ''
        print(f'  {tag}[CAM] {p.name} ({p.stat().st_size:,}b)')
        return str(p)
    print('  [CAM] snapshot failed')
    return None


def hw_pos():
    """Read actual hardware servo positions."""
    d = get('/arm/read_hw_positions')
    return d.get('hw_positions', {})


def ask(prompt, default):
    if NO_INPUT:
        print(f'  {prompt}  [auto: {default}]')
        return str(default)
    return input(f'  {prompt} ').strip() or str(default)


def section(t):
    print(); print(SEP); print(f'  {t}'); print(SEP)


def ps(d):
    return ' '.join(f'S{k}={v}' for k, v in sorted(d.items()))


# ============================================================================
# STEP 0: PRE-FLIGHT
# ============================================================================

section('STEP 0 -- PRE-FLIGHT')

try:
    h = get('/health')
    print(f'  Agent : {h.get("service")} v{h.get("version")}  [{h.get("status")}]')
    print(f'  xArm  : connected={h.get("xarm")}  sim={h.get("xarm_simulation")}')
    print(f'  Camera: {h.get("camera")}')
except Exception as e:
    print(f'  FAIL: cannot reach Pi -- {e}'); sys.exit(1)

if h.get('xarm_simulation'):
    print('  Arm in SIM -- reconnecting...')
    post('/arm/reconnect'); time.sleep(2.5)
    h = get('/health')
    print(f'  After reconnect: sim={h.get("xarm_simulation")}')

print()
print('  IK-computed positions (official ki_move coordinates):')
for name, p in POS.items():
    print(f'    {name:<20}  {ps(p)}')


# ============================================================================
# STEP 1: OFFICIAL HOME
# ============================================================================

section('STEP 1 -- OFFICIAL HOME  ki_move(0, 17, 20.5, 0)')
print('  Using manufacturer ki_move coordinates -- arm should face FORWARD.')
print()

move(POS['home'], ms=2000, label='HOME', extra=0.5)
hw = hw_pos()
print(f'  Hardware readback: {ps(hw)}')
snap('01_home.jpg', 'HOME')

ok = ask('Arm looks correct at HOME? Gripper open, arm facing forward? (y/n)', 'y')
if ok.lower() != 'y':
    print('  Check hardware, then re-run.')
    sys.exit(0)


# ============================================================================
# STEP 2: PICK HEIGHT (center forward)
# ============================================================================

section('STEP 2 -- PICK HEIGHT  ki_move(0, 17, 1.2, -71)')
print('  Official pick position: 17cm forward, 1.2cm high, wrist -71deg.')
print('  THIS IS WHERE THE LIGHTER MUST BE for a center-forward pick.')
print()

move(POS['pick_low_fwd'], ms=1200, label='PICK-CTR', extra=0.3)
hw = hw_pos()
print(f'  Hardware: {ps(hw)}')
snap('02_pick_center.jpg', 'PICK-CTR')

print()
print('  Is the lighter placed at this position (directly in front, ~17cm)?')
lighter_fwd = ask('Lighter at center forward position? (y/n)', 'n')


# ============================================================================
# STEP 3: S6 SWEEP -- find lighter
# ============================================================================

section('STEP 3 -- S6 SWEEP  (find where lighter actually is)')

# S6 mapping:  S6 = 500 - atan2(x, y)*deg * 4.167
# x > 0 (RIGHT) -> S6 < 500  (sweep DOWN from 500)
# x < 0 (LEFT)  -> S6 > 500  (sweep UP from 500)

print('  S6 direction (CONFIRMED from official .rob files):')
print('    S6=500 = center forward  (y=17, x=0)')
print('    S6=875 = arm LEFT  90deg (y=0, x=-17)  [BLUE lighter side]')
print('    S6=315 = arm RIGHT 45deg (y=12, x=12)  [GREEN lighter side]')
print('    S6=125 = arm RIGHT 90deg (y=0, x=+17)')
print()
print('  Sweep order: center (500), then RIGHT (480...280), then LEFT (520...720)')
print('  Watch which frame shows the gripper CENTERED over the lighter.')
print()

# Build sweep: center first, then right, then left
SWEEP = [500] + list(range(480, 260, -20)) + list(range(520, 740, 20))

# Arm joint positions at pick height (same for all S6 values)
PICK_JOINTS = {'3': POS['pick_low_fwd']['3'], '4': POS['pick_low_fwd']['4'],
               '5': POS['pick_low_fwd']['5'], '2': 500, '1': 100}

sweep_done = {}
for s6 in SWEEP:
    pose = dict(PICK_JOINTS)
    pose['6'] = s6
    label = f'S6={s6}'
    move(pose, ms=500, label=label)
    time.sleep(0.6)
    p = snap(f'03_sweep_{s6:04d}.jpg', label)
    sweep_done[s6] = p
    print()

# Return home
move(POS['home'], ms=1000)
time.sleep(1.0)

print()
print(f'  Sweep done ({len(sweep_done)} frames). Open data/master_calib/ and look at 03_sweep_*.jpg')
print()
print('  Key reference S6 values:')
print('    RIGHT: 500(fwd)  480  460  440  420  400  380  360  340  320  300  280')
print('    LEFT:  520  540  560  580  600  620  640  660  680  700  720')
print()

try:
    raw = ask('Enter S6 where gripper is OVER the lighter (from image)', '400')
    S6_PICK = int(raw)
except ValueError:
    S6_PICK = 400
    print(f'  Using default S6={S6_PICK}')

# Compute the pick x,y from S6
theta_deg = (500.0 - S6_PICK) / S6_SCALE
theta_rad = math.radians(theta_deg)
# atan2(x, y) = theta, so x = r*sin(theta), y = r*cos(theta) where r=17cm
r_pick = 17.0
x_pick = r_pick * math.sin(theta_rad)
y_pick = r_pick * math.cos(theta_rad)

print(f'  S6={S6_PICK} -> theta={theta_deg:.1f}deg -> x={x_pick:.1f}cm, y={y_pick:.1f}cm')

# Verify
print()
print(f'  Verifying S6={S6_PICK}...')
verify_pose = dict(PICK_JOINTS)
verify_pose['6'] = S6_PICK
move(verify_pose, ms=800, label=f'VERIFY S6={S6_PICK}')
time.sleep(1.0)
snap(f'03_verify_{S6_PICK}.jpg', f'VERIFY S6={S6_PICK}')

ok_s6 = ask('Gripper centered over lighter? (y/n)', 'y')
if ok_s6.lower() != 'y':
    try:
        S6_PICK = int(ask('Enter corrected S6', S6_PICK))
    except ValueError:
        pass
    verify_pose['6'] = S6_PICK
    move(verify_pose, ms=600)
    time.sleep(1.0)
    snap(f'03_final_{S6_PICK}.jpg', f'FINAL S6={S6_PICK}')
    theta_deg = (500.0 - S6_PICK) / S6_SCALE
    theta_rad = math.radians(theta_deg)
    x_pick = r_pick * math.sin(theta_rad)
    y_pick = r_pick * math.cos(theta_rad)

move(POS['home'], ms=800)
print(f'  CONFIRMED: S6_PICK = {S6_PICK}')
print(f'  Lighter at approx (x={x_pick:.1f}, y={y_pick:.1f})cm')


# ============================================================================
# STEP 4: PICK HEIGHT FINE-TUNE  (IK-correct: adjusts S4+S5, not just S3)
# ============================================================================

section('STEP 4 -- PICK HEIGHT FINE-TUNE')
print('  Official pick: S3=180, S4=800, S5=450 (ki_move z=1.2cm)')
print()
print('  HEIGHT is controlled by S5 (shoulder) + S4 (elbow) -- NOT S3 alone.')
print('  IK sweep (lower z = lower arm):')
print(f'    {"z(cm)":>7}  {"S3":>5}  {"S4":>5}  {"S5":>5}  delta_S5   delta_S4')
print(f'    {"-"*7}  {"-"*5}  {"-"*5}  {"-"*5}  {"-"*9}  {"-"*9}')
for _z in [1.5, 1.2, 0.75, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0, -1.5]:
    _sv = ki_to_servos(x_pick, y_pick, _z, -71)
    _m  = ' <-- official' if abs(_z - 1.2) < 0.1 else ''
    print(f'    {_z:>7.2f}  {_sv["3"]:>5}  {_sv["4"]:>5}  {_sv["5"]:>5}  '
          f'{_sv["5"]-450:>+9}  {_sv["4"]-800:>+9}{_m}')
print()

# Move to pick with calibrated S6
pick_approach = dict(POS['home'])
pick_approach['6'] = S6_PICK
move(pick_approach, ms=1200, label='APPROACH')
time.sleep(0.5)

pick_low_at_s6 = dict(PICK_JOINTS)
pick_low_at_s6['6'] = S6_PICK
move(pick_low_at_s6, ms=800, label='PICK-LOW z=1.2')
snap('04_pick_height.jpg', 'PICK-HEIGHT')

print()
print('  If gripper is 1-2cm above table, enter a z_correction value (in cm).')
print('  z_correction=1.5 means "lower by 1.5cm" -> new z = 1.2 - 1.5 = -0.3cm.')
print('  z_correction=0.0 keeps official height.')
print()

try:
    raw_z = ask('z_correction in cm [0.0 = keep official, 1.5 = recommended fix]', '1.5')
    z_corr = float(raw_z)
except ValueError:
    z_corr = 0.0

z_pick = 1.2 - z_corr
_sv_adj = ki_to_servos(x_pick, y_pick, z_pick, -71)
S3_PICK = _sv_adj['3']
S4_ADJ  = _sv_adj['4']
S5_ADJ  = _sv_adj['5']

print(f'  z_correction={z_corr}cm  ->  z_pick={z_pick:.2f}cm')
print(f'  Adjusted servos: S3={S3_PICK}  S4={S4_ADJ}  S5={S5_ADJ}  (S6={S6_PICK})')

if z_corr != 0.0:
    pick_low_at_s6['3'] = S3_PICK
    pick_low_at_s6['4'] = S4_ADJ
    pick_low_at_s6['5'] = S5_ADJ
    move(pick_low_at_s6, ms=600, label=f'ADJ z={z_pick:.2f}')
    time.sleep(0.8)
    snap(f'04_pick_adj_z{z_pick:.2f}.jpg', f'z={z_pick:.2f}')

move(POS['home'], ms=1000)
print(f'  CONFIRMED: z_pick={z_pick:.2f}cm  S3={S3_PICK}  S4={S4_ADJ}  S5={S5_ADJ}')


# ============================================================================
# STEP 5: DETERMINE PLACE POSITION
# ============================================================================

section('STEP 5 -- PLACE POSITION')
print('  Official place positions (from color sorting corex.py):')
print()
print('    L  -> Left  90deg:  S6=875   ki_move(-19.5,  0, 2.8, -60)')
print('    l  -> Left  45deg:  S6=685   ki_move(-14,   14, 2.9, -59)')
print('    r  -> Right 45deg:  S6=315   ki_move( 14,   14, 2.9, -59)')
print('    R  -> Right 90deg:  S6=125   ki_move( 19.5,  0, 2.8, -60)')
print()

place_choice = ask('Select place target [L/l/r/R]', 'L').strip().upper()
PLACE_MAP = {
    'L': ('place_left90',  875, -19.5,  0),
    'LO': ('place_left45', 685,  -14,  14),
    'R':  ('place_right90', 125, 19.5,  0),
    'RO': ('place_right45', 315,  14,  14),
}
# Map single char
if place_choice == 'L':   pk = 'L'
elif place_choice == 'l': pk = 'LO'
elif place_choice == 'r': pk = 'RO'
elif place_choice == 'R': pk = 'R'
else:                      pk = 'L'

place_key, S6_PLACE, px_place, py_place = PLACE_MAP[pk]
print(f'  Place: {place_key}  S6={S6_PLACE}  ki_move({px_place}, {py_place}, 2.8, -60)')

# Verify place position
print()
print('  Testing place position...')
transit_pose = dict(POS['home'])
transit_pose['1'] = 500   # gripper closed (simulating holding object)
transit_pose['6'] = S6_PLACE
move(transit_pose, ms=1500, label='TRANSIT')
snap('05_transit.jpg', 'TRANSIT')

place_low_pose = dict(POS[place_key])
place_low_pose['6'] = S6_PLACE    # ensure correct S6
move(place_low_pose, ms=1000, label='PLACE-LOW')
snap('05_place_low.jpg', 'PLACE-LOW')

print()
ok_place = ask('Place position looks correct for the drop zone? (y/n)', 'y')
move(POS['home'], ms=1500)


# ============================================================================
# STEP 6: FULL PICK-AND-PLACE TRIAL
# ============================================================================

section('STEP 6 -- FULL PICK-AND-PLACE TRIAL')
print('  Complete sequence with actual lighter.')
print()
print(f'  Pick S6  = {S6_PICK}  (lighter position)')
print(f'  S3 pick  = {S3_PICK}  (pick height)')
print(f'  Place S6 = {S6_PLACE}  ({place_key})')
print()
print('  Place the lighter on the table now.')
ask('Press Enter when ready', '')

# Build final poses  (use IK-corrected S4/S5 for pick height)
PICK_APPROACH = dict(POS['home']); PICK_APPROACH['6'] = S6_PICK
PICK_DOWN     = {'1': 100, '2': 500, '3': S3_PICK, '4': S4_ADJ, '5': S5_ADJ, '6': S6_PICK}
GRIP_CLOSE    = {'1': 500, '2': 500, '3': S3_PICK, '4': S4_ADJ, '5': S5_ADJ, '6': S6_PICK}
LIFT_POSE     = {'1': 500, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_PICK}
TRANSIT_POSE  = {'1': 500, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_PLACE}
PLACE_DOWN    = dict(POS[place_key]); PLACE_DOWN['6'] = S6_PLACE
RELEASE       = dict(PLACE_DOWN);    RELEASE['1'] = 100

print('  a) HOME:')
move(POS['home'], ms=1800, label='HOME')
snap('06a_home.jpg', 'HOME')

print()
print('  b) APPROACH (arm rotated to lighter):')
move(PICK_APPROACH, ms=1200, label='APPROACH')
snap('06b_approach.jpg', 'APPROACH')

print()
print('  c) LOWER to pick:')
move(PICK_DOWN, ms=800, label='LOWER')
snap('06c_lower.jpg', 'LOWER')

print()
print('  d) GRIP:')
move(GRIP_CLOSE, ms=500, label='GRIP')
snap('06d_gripped.jpg', 'GRIPPED')

print()
print('  e) LIFT:')
move(LIFT_POSE, ms=1000, label='LIFT')
snap('06e_lifted.jpg', 'LIFTED')

print()
print(f'  f) ROTATE to {place_key} (S6={S6_PLACE}):')
move(TRANSIT_POSE, ms=1500, label='ROTATE')
snap('06f_rotated.jpg', 'ROTATED')

print()
print('  g) LOWER to place:')
move(PLACE_DOWN, ms=1000, label='PLACE')
snap('06g_place.jpg', 'PLACE')

print()
print('  h) RELEASE:')
move(RELEASE, ms=500, label='RELEASE')
snap('06h_released.jpg', 'RELEASED')

print()
print('  i) HOME:')
move(POS['home'], ms=1500, label='HOME')
snap('06i_done.jpg', 'DONE')

trial_ok = ask('Was the pick-and-place successful? (y/n)', 'y')


# ============================================================================
# STEP 7: SAVE
# ============================================================================

section('STEP 7 -- SAVE CALIBRATION')

FINAL = {
    'home':          POS['home'],
    'pick_approach': PICK_APPROACH,
    'pick_low':      PICK_DOWN,
    'lift':          LIFT_POSE,
    'place_transit': TRANSIT_POSE,
    'place_low':     PLACE_DOWN,
    'release':       RELEASE,
}

print('  Final calibrated poses:')
for name, vals in FINAL.items():
    print(f'    {name:<16}  {ps(vals)}')

print()
if ask('Save to Pi arm memory? (y/n)', 'y').lower() == 'y':
    pi_map = {
        'home':       FINAL['home'],
        'pick_table': FINAL['pick_low'],
        'lift_grip':  FINAL['lift'],
        'place_bin':  FINAL['place_low'],
        'inspect':    {**POS['home'], '3': 250, '4': 830, '5': 570},
    }
    for name, vals in pi_map.items():
        r = post('/arm/save_touch_pose', {'name': name, 'positions': vals})
        if not r:
            r = post('/arm/touch_poses', {name: vals})
        print(f'  Saved {name}: {r.get("ok","?")}')

# Write local calib file
calib = {
    'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    's6_green':   S6_PICK if x_pick > 0 else None,
    's6_blue':    S6_PICK if x_pick < 0 else None,
    's6_pick':    S6_PICK,
    's6_place':   S6_PLACE,
    's3_pick':    S3_PICK,
    's4_pick':    S4_ADJ,
    's5_pick':    S5_ADJ,
    'z_pick':     z_pick,
    'z_correction_cm': z_corr,
    'x_pick':     round(x_pick, 2),
    'y_pick':     round(y_pick, 2),
    'place_key':  place_key,
    'trial_ok':   trial_ok.lower() == 'y',
    'poses':      FINAL,
    'ki_positions': {
        'home':       '(0, 17, 20.5, 0)',
        'pick':       f'({x_pick:.1f}, {y_pick:.1f}, {z_pick:.2f}, -71)',
        'place':      f'({px_place}, {py_place}, 2.8, -60)',
    }
}
OUT.write_text(json.dumps(calib, indent=2))
print(f'  Written to {OUT}')


# ============================================================================
# SUMMARY
# ============================================================================

section('CALIBRATION COMPLETE')

print('  ki_move coordinates used:')
print('    HOME:    ki_move(0, 17, 20.5, 0)')
print(f'    PICK:    ki_move({x_pick:.1f}, {y_pick:.1f}, {z_pick:.2f}, -71)  [S6={S6_PICK}]')
print(f'    PLACE:   ki_move({px_place}, {py_place}, 2.8, -60)  [S6={S6_PLACE}]')
print()
print(f'  Servo summary:')
print(f'    HOME:     {ps(POS["home"])}')
print(f'    PICK-LOW: {ps(PICK_DOWN)}')
print(f'    PLACE:    {ps(PLACE_DOWN)}')
print()
print(f'  Height correction applied: {z_corr:.1f}cm  (z: 1.2 -> {z_pick:.2f}cm)')
print(f'    S5: {POS["pick_low_fwd"]["5"]} -> {S5_ADJ}  (shoulder, delta={S5_ADJ-POS["pick_low_fwd"]["5"]:+d})')
print(f'    S4: {POS["pick_low_fwd"]["4"]} -> {S4_ADJ}  (elbow, delta={S4_ADJ-POS["pick_low_fwd"]["4"]:+d})')
print()
print('  Next:')
print('    python run_demo.py              # run demo using calibration')
print('    python cosmos_cookoff_demo.py   # full cosmos demo')
print()
print('  Frames in:', FRAMES)
for f in sorted(FRAMES.glob('*.jpg')):
    print(f'    {f.name}')
print()
print(SEP)
print('  DONE')
print(SEP)
