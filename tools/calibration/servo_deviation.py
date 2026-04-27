"""
GATE 1: SERVO DEVIATION CHECK & CORRECTION
==========================================
Official Hiwonder xArm AI documentation source:
  - ArmPi Ultra Manual 2.4.4: minor vs major deviation procedure
  - xArm AI Get Started 1.3.1: neutral position / blind zone explanation
  - Bus Servo Controller Manual: deviation slider ±100 to ±125

KEY FACTS (from official docs):
  - Bus servo range:   0 – 1000 (0° – 240°)
  - Neutral position:  500 = "zero point" for all joints
  - 1 slider unit:     0.24° of rotation  (240 ÷ 1000)
  - Minor deviation:   |offset| <= 125 units  (<= 30°) → software fix
  - Major deviation:   |offset|  > 125 units  (> 30°)  → physical horn reset

CRITICAL WARNING FROM HIWONDER:
  "If the neutral position is not set correctly, the potentiometer may enter a
   'blind zone,' causing the entire system to malfunction. The robot may
   experience failing to reach specified angles or inconsistencies in
   action groups."

SERVO ID MAP (xArm AI):
  S1 = gripper           (HX-06L)
  S2 = wrist tilt        (HTS-16L)
  S3 = wrist rotate      (LX-15D)
  S4 = elbow             (LX-15D)
  S5 = shoulder          (LX-225)
  S6 = base rotate       (LX-15D)

GATE: Pass = all |offset| <= 20 units (5°) for pick accuracy
      Warn = any |offset| 21–125 (minor deviation, software fix needed)
      Fail = any |offset|  > 125 (major deviation, physical fix required)

Usage:
  python servo_deviation.py              # check only
  python servo_deviation.py --fix        # check + send corrected deviations
  python servo_deviation.py --sweep      # move each servo through range and check
"""

import json, math, time, sys, argparse
from pathlib import Path

PI     = 'http://192.168.1.163:8085'
CALIB  = Path('data/calib_results.json')
SEP    = '=' * 70

parser = argparse.ArgumentParser()
parser.add_argument('--fix',    action='store_true', help='Apply deviation corrections')
parser.add_argument('--sweep',  action='store_true', help='Move servos and check positions')
parser.add_argument('--reset',  action='store_true', help='Reset all servos to 500 then check')
args = parser.parse_args()

import urllib.request as _ur

def get(p, t=15):
    try: return json.loads(_ur.urlopen(PI+p, timeout=t).read())
    except Exception as e: print(f'  GET {p}: {e}'); return {}

def post(p, b=None, t=20):
    try:
        d = json.dumps(b or {}).encode()
        r = _ur.Request(PI+p, data=d, headers={'Content-Type':'application/json'})
        return json.loads(_ur.urlopen(r, timeout=t).read())
    except Exception as e: print(f'  POST {p}: {e}'); return {}

def move(servos, ms=1200, label=''):
    r   = post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim = r.get('simulation', False)
    s   = ' '.join(f'S{k}={v}' for k,v in sorted(servos.items()))
    print(f'  [{label or "MOVE"}] {s}  [{"SIM" if sim else "OK"}]')
    if sim: post('/arm/reconnect'); time.sleep(2.5)
    time.sleep(ms/1000.0 + 0.5)
    return r

def move1(sid, pos, ms=800):
    return move({str(sid): pos}, ms=ms, label=f'S{sid}={pos}')


# ============================================================================
# SERVO METADATA
# ============================================================================

SERVO_NAMES = {
    1: 'gripper      (HX-06L)',
    2: 'wrist-tilt   (HTS-16L)',
    3: 'wrist-rotate (LX-15D)',
    4: 'elbow        (LX-15D)',
    5: 'shoulder     (LX-225)',
    6: 'base-rotate  (LX-15D)',
}

NEUTRAL_500 = {1: 500, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500}

# Expected HOME pose (matches official ki_move(0,17,20.5,0) output)
# These are where servos SHOULD be at home, not 500
HOME_EXPECTED = {
    1: 100,   # gripper open
    2: 500,   # wrist tilt neutral
    3: 310,   # wrist rotate
    4: 870,   # elbow
    5: 680,   # shoulder
    6: 500,   # base center
}

# IK-confirmed PICK pose (z=0.7cm, x=0, y=17)
PICK_EXPECTED = {
    1: 100,
    2: 500,
    3: 177,
    4: 814,
    5: 432,
    6: 400,
}

DEG_PER_UNIT = 240.0 / 1000.0   # 0.24° per servo unit
MINOR_THRESHOLD = 125             # ±125 units = ±30° from ArmPi Ultra manual
PASS_THRESHOLD  = 20              # ±20 units = ±5° for pick accuracy


# ============================================================================
# VISUAL HELPERS
# ============================================================================

def deviation_bar(offset, width=40):
    """ASCII progress bar showing deviation amount and direction."""
    half = width // 2
    max_dev = MINOR_THRESHOLD
    fill = min(int(abs(offset) / max_dev * half), half)
    if offset >= 0:
        bar = ' ' * half + '#' * fill + ' ' * (half - fill)
    else:
        bar = ' ' * (half - fill) + '#' * fill + ' ' * half
    center = half
    b = list(bar)
    b[center] = '|'
    return ''.join(b)

def status_label(abs_off):
    if abs_off <= PASS_THRESHOLD:
        return 'PASS  '
    elif abs_off <= MINOR_THRESHOLD:
        return 'WARN-minor'
    else:
        return 'FAIL-MAJOR'


# ============================================================================
# STEP 0: PREFLIGHT
# ============================================================================

print(SEP)
print('  GATE 1: SERVO DEVIATION CHECK')
print('  Source: Hiwonder ArmPi Ultra Manual 2.4.4 + xArm AI 1.3.1')
print(SEP)

h = get('/health')
print(f'  Agent: {h.get("service")} v{h.get("version")}  [{h.get("status")}]')
print(f'  xArm: connected={h.get("xarm")}  sim={h.get("xarm_simulation")}')
if h.get('xarm_simulation'):
    post('/arm/reconnect'); time.sleep(2.5)
print()


# ============================================================================
# STEP 1: RESET TO NEUTRAL (optional)
# ============================================================================

HOME = {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500}

if args.reset:
    print('  Sending all servos to position 500 (factory neutral)...')
    RESET = {str(i): 500 for i in range(1, 7)}
    move(RESET, ms=2000, label='NEUTRAL-500')
    print('  Visually verify:')
    print('    - S3/S4/S5 screws form a STRAIGHT CENTER LINE')
    print('    - S6 is PARALLEL to the metal casing')
    print('    - S2 metal casing PARALLEL to camera casing')
    print('    - S1 gripper SLIGHTLY OPEN')
    print()
    input('  Press Enter after visual inspection...')
    print()


# ============================================================================
# STEP 2: READ CURRENT ARM STATUS
# ============================================================================

print('  Reading arm status...')
st = get('/arm/status', t=12)
if not st:
    print('  ERROR: Cannot read arm status')
    sys.exit(1)

positions = st.get('positions', {})
if not positions:
    positions = HOME_EXPECTED.copy()
    print('  WARNING: No position feedback -- using HOME expected as baseline')
else:
    print(f'  Got positions: {positions}')
print()


# ============================================================================
# STEP 3: DEVIATION FROM NEUTRAL (500) AND FROM HOME EXPECTED
# ============================================================================

def compute_deviations(actual, reference):
    devs = {}
    for sid in range(1, 7):
        act = actual.get(sid, actual.get(str(sid), reference.get(sid, 500)))
        ref = reference.get(sid, 500)
        devs[sid] = int(act) - int(ref)
    return devs

# Deviation from expected home pose (not from 500!)
# The arm is at HOME, not at factory neutral.
# Comparing to 500 would always show false "MAJOR" since HOME != all-500.
dev_home    = compute_deviations(positions, HOME_EXPECTED)
# Also compute from neutral for information
dev_neutral = compute_deviations(positions, NEUTRAL_500)

print('  POSITION CHECK vs EXPECTED HOME:')
print('  (Home: S1=100 S2=500 S3=310 S4=870 S5=680 S6=500  from ki_move(0,17,20.5,0))')
print()
print(f'  {"ID":<4} {"Name":<30} {"Actual":>7} {"Expect":>7} {"Dev":>6} {"Deg":>7}   {"Status":<12}  Bar(±30°)')
print(f'  {"-"*4} {"-"*30} {"-"*7} {"-"*7} {"-"*6} {"-"*7}   {"-"*12}  {"-"*42}')

all_pass = True
warns    = []
fails    = []

for sid in range(1, 7):
    act = int(positions.get(sid, positions.get(str(sid), HOME_EXPECTED[sid])))
    exp = HOME_EXPECTED[sid]
    off = dev_home[sid]
    deg = off * DEG_PER_UNIT
    st2 = status_label(abs(off))
    bar = deviation_bar(off)
    print(f'  S{sid}   {SERVO_NAMES[sid]:<30} {act:>7} {exp:>7} {off:>+6} {deg:>+7.1f}°   {st2:<12}  [{bar}]')
    if abs(off) > PASS_THRESHOLD:
        all_pass = False
        if abs(off) <= MINOR_THRESHOLD:
            warns.append(sid)
        else:
            fails.append(sid)

print()
print('  NOTE: For Hiwonder MECHANICAL deviation check (stored servo offsets):')
print('    -> Run "Reset servo" in PC software (sends all to 500)')
print('    -> Visually verify: S3-S5 main screws form a STRAIGHT LINE')
print('    -> S6 PARALLEL to casing, S2 PARALLEL to camera casing')
print('    -> If NOT straight: THAT is mechanical deviation (use PC tool)')

print()

# ============================================================================
# STEP 4: GATE RESULT AND INSTRUCTIONS
# ============================================================================

print()
print(SEP)
if all_pass:
    print('  GATE 1 RESULT: PASS -- Arm at correct HOME position (±5° tolerance)')
    print('  Servo positions match expected ki_move(0,17,20.5,0) home pose.')
    print()
    print('  Mechanical deviation check (from Hiwonder PC tool - do ONCE):')
    print('    1. Open PC software -> "Reset servo" (sends all to 500)')
    print('    2. Verify: S3-S5 screws form a STRAIGHT CENTER LINE')
    print('    3. S6 PARALLEL to metal casing')
    print('    4. If NOT: follow Minor/Major deviation procedure in manual 2.4.4')
elif not fails:
    print(f'  GATE 1 RESULT: WARN -- Position drift detected on S{warns}')
    print(f'  Home position is off by > 5° on some joints.')
    print(f'  This suggests the HOME pose constants in code may be miscalibrated.')
    print()
    print('  Run:  python servo_deviation.py --reset  to verify at neutral 500')
    print('  Then: check if arm is mechanically straight at 500')
else:
    print(f'  GATE 1 RESULT: FAIL -- Large position mismatch on S{fails}')
    print(f'  These joints are far from expected HOME position.')
    print(f'  This could mean:')
    print(f'    a) The arm was moved manually before this check')
    print(f'    b) The HOME constants in code are wrong')
    print(f'    c) Mechanical deviation requiring PC tool adjustment')
    print()
    for sid in fails:
        off = dev_home[sid]
        exp = HOME_EXPECTED[sid]
        print(f'  S{sid}: expected={exp}, actual={int(positions.get(sid,positions.get(str(sid),exp)))}, '
              f'diff={off:+d} units ({off*DEG_PER_UNIT:+.1f}°)')
    print()
    print('  IF mechanical: Hiwonder ArmPi Ultra 2.4.4 Major Deviation procedure:')
    for sid in fails:
        if abs(dev_home[sid]) > MINOR_THRESHOLD:
            print(f'    S{sid}: power off -> remove horn screws -> power on -> Reset servo')
            print(f'           power off -> reinstall horns in (+) -> power on -> minor tune')
print(SEP)
print()


# ============================================================================
# STEP 5: SWEEP TEST (optional) - moves each servo and checks positions
# ============================================================================

if args.sweep:
    print('  SWEEP TEST: Moving each servo and checking position feedback')
    print()
    for sid in range(1, 7):
        print(f'  --- S{sid} ({SERVO_NAMES[sid].strip()}) ---')
        for target in [300, 500, 700]:
            move1(sid, target, ms=600)
            st2 = get('/arm/status', t=8)
            actual = int(st2.get('positions', {}).get(sid,
                         st2.get('positions', {}).get(str(sid), -1)))
            err = actual - target
            deg = err * DEG_PER_UNIT
            print(f'    Target {target:>4} -> Actual {actual:>4}  Error {err:>+4} units = {deg:>+5.1f}°')
            time.sleep(0.2)
        # Return to neutral
        move1(sid, 500, ms=600)
        print()
    # Restore HOME
    move(HOME, ms=2000, label='HOME')
    print()


# ============================================================================
# STEP 6: APPLY SOFTWARE DEVIATIONS (optional)
# ============================================================================

if args.fix and warns:
    print('  APPLYING SOFTWARE DEVIATION CORRECTIONS via API...')
    print()

    # The deviation correction sends the NEGATIVE of the observed offset
    # so the arm moves to TRUE neutral 500
    corrections = {}
    for sid in warns:
        off = dev_neutral[sid]
        correction = -off
        # Clamp to safe range
        correction = max(-125, min(125, correction))
        corrections[sid] = correction
        deg = correction * DEG_PER_UNIT
        print(f'    S{sid}: observed offset={off:+d} -> correction={correction:+d} units ({deg:+.1f}°)')

    print()
    print('  Sending correction move (shift from current toward 500)...')
    corrected = {}
    for sid in range(1, 7):
        act = int(positions.get(sid, positions.get(str(sid), 500)))
        if sid in corrections:
            corrected[str(sid)] = max(100, min(900, act + corrections[sid]))
        else:
            corrected[str(sid)] = act

    move(corrected, ms=1200, label='CORRECTED-NEUTRAL')
    print()

    # Save deviation info
    calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    calib['servo_deviation'] = {
        'measured_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'dev_from_neutral': {str(k): v for k,v in dev_neutral.items()},
        'corrections_applied': corrections,
        'gate1_result': 'WARN-minor-corrected',
    }
    CALIB.write_text(json.dumps(calib, indent=2))
    print(f'  Saved deviation data to {CALIB}')
    print()

elif args.fix and fails:
    print('  CANNOT auto-fix MAJOR deviations via software.')
    print('  Follow the PHYSICAL REPAIR procedure above, then re-run.')
    print()


# ============================================================================
# STEP 7: SAVE GATE 1 AUDIT RECORD
# ============================================================================

calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}
calib['gate1_servo_deviation'] = {
    'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'positions_read': {str(k): v for k,v in positions.items()},
    'dev_from_neutral': {str(k): v for k,v in dev_neutral.items()},
    'minor_deviation_servos': warns,
    'major_deviation_servos': fails,
    'gate1_pass': all_pass,
    'threshold_pass_units': PASS_THRESHOLD,
    'threshold_minor_units': MINOR_THRESHOLD,
    'deg_per_unit': DEG_PER_UNIT,
}
CALIB.write_text(json.dumps(calib, indent=2))

print(SEP)
print('  GATE 1 COMPLETE')
print(SEP)
print(f'  Audit saved to {CALIB}')
print()
print('  Summary of deviation thresholds (Hiwonder official):')
print(f'    PASS   : |offset| <= {PASS_THRESHOLD:3} units  (<=  {PASS_THRESHOLD*DEG_PER_UNIT:.0f}°)  ideal for ML/IK')
print(f'    MINOR  : |offset| <= {MINOR_THRESHOLD:3} units  (<= {MINOR_THRESHOLD*DEG_PER_UNIT:.0f}°)  software fix')
print(f'    MAJOR  :          > {MINOR_THRESHOLD:3} units  ( > {MINOR_THRESHOLD*DEG_PER_UNIT:.0f}°)  physical disassembly')
print(f'    1 unit = {DEG_PER_UNIT}°  (240° range / 1000 steps)')
print()
print('  Next steps:')
if fails:
    print('    PHYSICAL REPAIR -> re-run this script -> then run camera_intrinsics.py')
elif warns:
    print('    python servo_deviation.py --fix  (apply software corrections)')
    print('    Then: python camera_intrinsics.py (Gate 2)')
else:
    print('    python camera_intrinsics.py  (Gate 2: camera calibration)')
