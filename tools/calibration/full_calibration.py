"""
FULL CALIBRATION -- xArm AI Pick-and-Place
==========================================
Uses OFFICIAL Hiwonder .rob action group positions as baseline.

  Official HOME:     {S1:100, S2:500, S3:310, S4:870, S5:680, S6:500}
  Official PICK LOW: {S1:100, S2:500, S3:180, S4:800, S5:450, S6:500}
  S6 direction:      DECREASING = rotate RIGHT  |  INCREASING = rotate LEFT

Calibration sequence:
  0. Pre-flight  -- ping Pi, check arm hardware
  1. Official HOME -- land at manufacturer baseline pose
  2. RIGHT sweep  -- S6: 500 -> 280 (find GREEN lighter on right)
  3. LEFT  sweep  -- S6: 500 -> 720 (find BLUE  lighter on left,  optional)
  4. Pick-height  -- fine-tune S3/S4/S5 at chosen S6
  5. Grip test    -- full open->lower->close->lift cycle
  6. Place test   -- rotate to drop zone, lower, release
  7. Save + summary

Usage:
  python full_calibration.py
  python full_calibration.py --no-input   # non-interactive (uses official defaults)
"""

import json, time, urllib.request, sys, os
from pathlib import Path

sys.path.insert(0, '.')

# Import rob_parser directly to avoid neurolinux __init__ chain
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'rob_parser',
    os.path.join(os.path.dirname(__file__), 'src', 'neurolinux', 'drivers', 'rob_parser.py')
)
_rob = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rob)
OFFICIAL     = _rob.OFFICIAL
pick_pose     = _rob.pick_pose
pick_low_pose = _rob.pick_low_pose
place_pose    = _rob.place_pose

PI      = 'http://192.168.1.163:8085'
FRAMES  = Path('data/calib_frames')
RESULTS = Path('data/calib_results.json')
FRAMES.mkdir(parents=True, exist_ok=True)

NO_INPUT = '--no-input' in sys.argv
SEP      = '=' * 64
SEP2     = '-' * 64

# -- Official baseline (from Hiwonder .rob files, decoded 2026-02-27) ----------
HOME     = {str(k): v for k, v in OFFICIAL['home'].items()}
PICK_HI  = {str(k): v for k, v in OFFICIAL['home'].items()}        # home height, S6 swappable
PICK_LO  = {str(k): v for k, v in OFFICIAL['pick_low'].items()}    # official pick depth
GRIP_ON  = {str(k): v for k, v in OFFICIAL['grip_closed'].items()} # gripper closed, arm up
PLACE_LO = {str(k): v for k, v in OFFICIAL['place_low'].items()}   # place depth

# Place targets (official S6 values)
S6_LEFT_90  = OFFICIAL['s6_left_90']   # 875
S6_LEFT_45  = OFFICIAL['s6_left_45']   # 685
S6_RIGHT_45 = OFFICIAL['s6_right_45']  # 315
S6_RIGHT_90 = OFFICIAL['s6_right_90']  # 125
S6_CENTER   = OFFICIAL['s6_center']    # 500


# -- HTTP helpers ---------------------------------------------------------------

def pi_get(path, timeout=12):
    r = urllib.request.urlopen(PI + path, timeout=timeout)
    return json.loads(r.read())


def pi_post(path, body=None, timeout=18):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        PI + path, data=data,
        headers={'Content-Type': 'application/json'}
    )
    r = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(r.read())


def move(servos, ms=1200, label='', wait_extra=0.0):
    """Send group_move, print result, wait for motion to complete."""
    r = pi_post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim = r.get('simulation', False)
    ok  = r.get('ok', False)

    tag    = f'  [{label}] ' if label else '  '
    s_str  = ' '.join(f'S{k}={v}' for k, v in sorted(servos.items()))
    status = 'SIM' if sim else ('OK' if ok else 'FAIL')
    print(f'{tag}{s_str}  [{status}]')

    if sim:
        print('  *** SIMULATION -- retrying with reconnect ***')
        try:
            pi_post('/arm/reconnect', timeout=12)
            time.sleep(2.5)
            r2 = pi_post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
            print('  Retry:', 'SIM' if r2.get('simulation') else 'OK')
        except Exception as e:
            print('  Reconnect error:', e)

    time.sleep(ms / 1000.0 + 0.5 + wait_extra)
    return ok


def snap(name, label=''):
    """Capture camera frame, save to data/calib_frames/."""
    try:
        r = urllib.request.urlopen(PI + '/camera/snapshot', timeout=20)
        p = FRAMES / name
        p.write_bytes(r.read())
        sz = p.stat().st_size
        tag = f'[{label}] ' if label else ''
        print(f'  {tag}[CAM]  {name} ({sz:,} bytes)')
        return str(p)
    except Exception as e:
        print(f'  Snapshot failed: {e}')
        return None


def ask(prompt, default=None):
    """Input with default for --no-input mode."""
    if NO_INPUT:
        print(f'  {prompt}  [auto: {default}]')
        return str(default) if default is not None else ''
    return input(f'  {prompt} ').strip()


def section(title):
    print()
    print(SEP)
    print(f'  {title}')
    print(SEP)


def ps(d):
    """Pretty-print a servo dict."""
    return ' '.join(f'S{k}={v}' for k, v in sorted(d.items()))


# -- STEP 0: Pre-flight ---------------------------------------------------------

section('STEP 0 -- PRE-FLIGHT')

try:
    health = pi_get('/health')
    print(f'  Pi agent:  {health.get("service")} v{health.get("version")}  [{health.get("status")}]')
except Exception as e:
    print(f'  FAIL: Cannot reach Pi at {PI}\n  {e}')
    sys.exit(1)

status = pi_get('/arm/status')
sim    = status.get('simulation', True)
print(f'  Arm port:  {status.get("port")}')
print(f'  Connected: {status.get("connected")}')
print(f'  Sim mode:  {sim}')

if sim:
    print('  -> Reconnecting to hardware...')
    try:
        pi_post('/arm/reconnect', timeout=14)
        time.sleep(2.5)
        status = pi_get('/arm/status')
        sim    = status.get('simulation', True)
        print(f'  After reconnect -- sim={sim}')
    except Exception as e:
        print(f'  Reconnect error: {e}')

if sim:
    print()
    print('  WARNING: arm still in simulation. Physical moves will not happen.')
    go = ask('Continue anyway? (y/n):', 'y')
    if go.lower() != 'y':
        sys.exit(0)

print()
print('  Official baseline (from .rob files):')
print(f'    HOME:     {ps(HOME)}')
print(f'    PICK LOW: {ps(PICK_LO)}')
print(f'    S6: RIGHT 90={S6_RIGHT_90}  RIGHT 45={S6_RIGHT_45}  '
      f'CENTER={S6_CENTER}  LEFT 45={S6_LEFT_45}  LEFT 90={S6_LEFT_90}')


# -- STEP 1: Official HOME -----------------------------------------------------

section('STEP 1 -- OFFICIAL HOME')
print('  Moving to official Hiwonder home position.')
print('  Arm should be upright, gripper open, facing forward.')
print()

move(HOME, ms=2000, label='HOME', wait_extra=0.5)
snap('01_official_home.jpg', 'HOME')

input_ok = ask('Does arm look correct at home? (y/n):', 'y')
if input_ok.lower() != 'y':
    print('  -> Check arm hardware before continuing.')
    ask('Press Enter when ready to continue:', '')


# -- STEP 2: S6 RIGHT SWEEP -- find GREEN lighter --------------------------------

section('STEP 2 -- S6 RIGHT SWEEP (green lighter, right side)')
print('  S6 DECREASES = arm rotates RIGHT.')
print('  Sweeping S6: 500 -> 280 in steps of 20.')
print('  Watch which frame shows gripper centered over the GREEN lighter.')
print()

# At pick height (arm high) sweep to see which S6 aligns over target
RIGHT_SWEEP = list(range(500, 260, -20))   # [500, 480, 460, ... 280]

sweep_results = {}
for s6 in RIGHT_SWEEP:
    pose = dict(PICK_HI)
    pose['6'] = s6
    label = f'S6={s6}'
    move(pose, ms=600, label=label)
    time.sleep(0.8)
    p = snap(f'02_right_{s6}.jpg', label)
    sweep_results[s6] = p
    print()

move(HOME, ms=800)   # return to center

print()
print('  Sweep complete. Open data/calib_frames/ and look at 02_right_*.jpg')
print('  Find which S6 aligns the gripper fingers directly over the GREEN lighter.')
print()
print(f'  Reference: right-45 deg = {S6_RIGHT_45}, right-90 deg = {S6_RIGHT_90}')
print(f'  (Previous wrong value was S6=610 which goes LEFT, not right)')
print()

try:
    s6_green_raw = ask(f'Enter S6 for GREEN lighter (center: 500, right-45: {S6_RIGHT_45}, right-90: {S6_RIGHT_90}):', S6_RIGHT_45)
    S6_GREEN = int(s6_green_raw)
except ValueError:
    S6_GREEN = S6_RIGHT_45
    print(f'  Using default: S6={S6_GREEN}')

# Verify
print(f'  Verifying S6={S6_GREEN}...')
pose_verify = dict(PICK_HI)
pose_verify['6'] = S6_GREEN
move(pose_verify, ms=800, label='VERIFY_S6')
time.sleep(1.0)
snap(f'02_verify_s6_{S6_GREEN}.jpg', f'VERIFY S6={S6_GREEN}')

ok_s6 = ask('Is gripper centered over GREEN lighter at this height? (y/n):', 'y')
if ok_s6.lower() != 'y':
    try:
        S6_GREEN = int(ask('Enter corrected S6:', S6_GREEN))
    except ValueError:
        pass
    pose_verify['6'] = S6_GREEN
    move(pose_verify, ms=600, label='RECHECK')
    time.sleep(1.0)
    snap(f'02_final_s6_{S6_GREEN}.jpg', f'FINAL S6={S6_GREEN}')

print(f'  OK S6_GREEN = {S6_GREEN}')


# -- STEP 3: S6 LEFT SWEEP -- find BLUE lighter ---------------------------------

section('STEP 3 -- S6 LEFT SWEEP (blue lighter, left side) [optional]')
do_blue = ask('Calibrate BLUE lighter on left side too? (y/n):', 'y')

S6_BLUE = S6_LEFT_90   # default

if do_blue.lower() == 'y':
    print()
    print('  S6 INCREASES = arm rotates LEFT.')
    print('  Sweeping S6: 500 -> 720 in steps of 20.')
    print()

    LEFT_SWEEP = list(range(500, 740, 20))   # [500, 520, ... 720]

    for s6 in LEFT_SWEEP:
        pose = dict(PICK_HI)
        pose['6'] = s6
        label = f'S6={s6}'
        move(pose, ms=600, label=label)
        time.sleep(0.8)
        snap(f'03_left_{s6}.jpg', label)
        print()

    move(HOME, ms=800)

    try:
        S6_BLUE = int(ask(f'Enter S6 for BLUE lighter (left-45: {S6_LEFT_45}, left-90: {S6_LEFT_90}):', S6_LEFT_45))
    except ValueError:
        S6_BLUE = S6_LEFT_45

    pose_blue = dict(PICK_HI)
    pose_blue['6'] = S6_BLUE
    move(pose_blue, ms=800, label='VERIFY_BLUE')
    time.sleep(1.0)
    snap(f'03_verify_blue_{S6_BLUE}.jpg', f'BLUE S6={S6_BLUE}')
    print(f'  OK S6_BLUE = {S6_BLUE}')
else:
    print(f'  Skipped -- using official left-90: S6={S6_BLUE}')

move(HOME, ms=800)


# -- STEP 4: Pick-height calibration -------------------------------------------

section('STEP 4 -- PICK HEIGHT (S3/S4/S5 fine-tune)')
print('  Official pick-low: S3=180, S4=800, S5=450')
print('  Verifying this height is correct for your table/lighter height.')
print()

print('  Phase 4a -- approach at pick height, GREEN lighter:')
pose_approach = dict(PICK_LO)
pose_approach['6'] = str(S6_GREEN)
move(pose_approach, ms=1000, label='PICK_LOW_GREEN')
time.sleep(1.0)
snap('04a_pick_low_green.jpg', f'PICK_LOW S6={S6_GREEN}')

print()
print('  Fingers should be just above the lighter (ready to grip).')
print('  Official S3=180 = max forward/down. Increase S3 to raise, decrease to go lower.')
print()

adjust = ask('Adjust S3? Enter new value (official=180, higher=up, lower=down) or Enter to keep:', '180')
try:
    S3_PICK = int(adjust) if adjust else 180
except ValueError:
    S3_PICK = 180

if S3_PICK != 180:
    pose_approach['3'] = str(S3_PICK)
    move(pose_approach, ms=600, label=f'S3={S3_PICK}')
    time.sleep(0.8)
    snap(f'04b_pick_adj_s3_{S3_PICK}.jpg', f'S3={S3_PICK}')
    print(f'  OK S3_PICK = {S3_PICK}')
else:
    S3_PICK = 180
    print(f'  OK S3_PICK = {S3_PICK} (official)')

# Build final PICK_LOW with calibrated values
PICK_LOW_FINAL = {
    '1': 100,   # gripper open
    '2': 500,
    '3': S3_PICK,
    '4': 800,
    '5': 450,
    '6': S6_GREEN,
}
move(HOME, ms=1000)


# -- STEP 5: Full grip test ----------------------------------------------------

section('STEP 5 -- FULL GRIP CYCLE TEST')
print('  Testing complete: home -> approach -> lower -> grip -> lift')
print()
print('  !!  Position GREEN lighter on the table before continuing.')
print()
ask('Press Enter when lighter is placed and ready:', '')

# 5a - Official home
print('  5a HOME:')
move(HOME, ms=1500, label='HOME')
snap('05a_home_before.jpg', 'HOME')

# 5b - Approach (home height, rotated to green lighter)
print()
print('  5b APPROACH:')
approach = dict(HOME)
approach['6'] = str(S6_GREEN)
move(approach, ms=1200, label='APPROACH')
snap('05b_approach.jpg', f'APPROACH S6={S6_GREEN}')

# 5c - Lower to pick
print()
print('  5c LOWER:')
move(PICK_LOW_FINAL, ms=800, label='LOWER')
snap('05c_lowered.jpg', 'LOWERED')

# 5d - Close gripper
print()
print('  5d GRIP:')
grip = dict(PICK_LOW_FINAL)
grip['1'] = 500    # close gripper
move(grip, ms=500, label='GRIP', wait_extra=0.3)
snap('05d_gripped.jpg', 'GRIPPED')

# 5e - Lift to home height (keep rotation)
print()
print('  5e LIFT:')
lift = dict(HOME)
lift['1'] = 500    # keep gripper closed
lift['6'] = str(S6_GREEN)
move(lift, ms=1000, label='LIFT')
snap('05e_lifted.jpg', 'LIFTED')

print()
grip_ok = ask('Is lighter gripped and lifted? (y/n):', 'y')
if grip_ok.lower() != 'y':
    print('  -> Lighter not gripped. Possible fix: lower S3_PICK (currently', S3_PICK, ')')
    print('    Try re-running with a smaller S3 value.')


# -- STEP 6: Place test --------------------------------------------------------

section('STEP 6 -- PLACE CYCLE TEST')
print('  Transport lighter to place zone and release.')
print()

# 6a - Decide place side
print('  Place options:')
print(f'    L  -- left  90 deg (S6={S6_LEFT_90})')
print(f'    l  -- left  45 deg (S6={S6_LEFT_45})')
print(f'    r  -- right 45 deg (S6={S6_RIGHT_45})')
print(f'    R  -- right 90 deg (S6={S6_RIGHT_90})')
print(f'    B  -- blue lighter position (S6={S6_BLUE})')
print()

place_choice = ask('Place target [L/l/r/R/B]:', 'L').strip()
S6_PLACE_MAP = {'L': S6_LEFT_90, 'l': S6_LEFT_45, 'r': S6_RIGHT_45, 'R': S6_RIGHT_90, 'B': S6_BLUE}
S6_PLACE = S6_PLACE_MAP.get(place_choice.upper()[0] if place_choice else 'L', S6_LEFT_90)
print(f'  Using place S6={S6_PLACE}')

# 6b - Rotate to place (keep arm high, gripper closed)
print()
print('  6a ROTATE TO PLACE:')
rotate_pose = dict(HOME)
rotate_pose['1'] = 500        # gripper closed
rotate_pose['6'] = str(S6_PLACE)
move(rotate_pose, ms=1500, label='ROTATE')
snap('06a_rotated.jpg', f'ROTATED S6={S6_PLACE}')

# 6c - Lower to place height
print()
print('  6b LOWER TO PLACE:')
place_low = dict(PLACE_LO)
place_low['1'] = 500           # gripper closed during descent
place_low['6'] = str(S6_PLACE)
move(place_low, ms=1000, label='PLACE_LOW')
snap('06b_place_low.jpg', 'PLACE_LOW')

# 6d - Open gripper
print()
print('  6c RELEASE:')
release = dict(place_low)
release['1'] = 100    # open gripper
move(release, ms=500, label='RELEASE', wait_extra=0.3)
snap('06c_released.jpg', 'RELEASED')

# 6e - Lift and home
print()
print('  6d RETREAT:')
move(HOME, ms=1500, label='HOME')
snap('06d_home_after.jpg', 'HOME_AFTER')

place_ok = ask('Was lighter placed correctly? (y/n):', 'y')


# -- STEP 7: Save calibrated poses ---------------------------------------------

section('STEP 7 -- SAVE CALIBRATED POSITIONS')

CALIBRATED = {
    'home': {
        '1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_CENTER
    },
    'pick_approach_green': {
        '1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_GREEN
    },
    'pick_low_green': PICK_LOW_FINAL,
    'lift_with_object': {
        '1': 500, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_GREEN
    },
    'pick_approach_blue': {
        '1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': S6_BLUE
    },
    'pick_low_blue': {
        '1': 100, '2': 500, '3': S3_PICK, '4': 800, '5': 450, '6': S6_BLUE
    },
    'place_left_90': {
        '1': 500, '2': 500, '3': 220, '4': 800, '5': 460, '6': S6_LEFT_90
    },
    'place_left_45': {
        '1': 500, '2': 500, '3': 225, '4': 800, '5': 460, '6': S6_LEFT_45
    },
    'place_right_45': {
        '1': 500, '2': 500, '3': 225, '4': 800, '5': 460, '6': S6_RIGHT_45
    },
    'place_right_90': {
        '1': 500, '2': 500, '3': 220, '4': 800, '5': 460, '6': S6_RIGHT_90
    },
}

print()
print('  Calibrated positions:')
for name, vals in CALIBRATED.items():
    print(f'    {name:<28}  {ps(vals)}')

print()
save = ask('Save to arm memory? (y/n):', 'y')

if save.lower() == 'y':
    # Save individual named poses the Pi knows about
    pi_pose_map = {
        'home':       CALIBRATED['home'],
        'pick_table': CALIBRATED['pick_low_green'],
        'lift_grip':  CALIBRATED['lift_with_object'],
        'place_bin':  CALIBRATED['place_left_90'],
        'inspect':    {'1': 100, '2': 500, '3': 250, '4': 830, '5': 570, '6': S6_CENTER},
    }
    for pose_name, vals in pi_pose_map.items():
        try:
            r = pi_post('/arm/touch_poses', {pose_name: vals})
            ok = r.get('ok') or r.get('saved') or True
            print(f'  Saved {pose_name}: {ok}')
        except Exception as e:
            print(f'  Save {pose_name} failed: {e}')

    # Also write to local JSON for reference
    RESULTS.write_text(json.dumps({
        'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        's6_green': S6_GREEN,
        's6_blue':  S6_BLUE,
        's3_pick':  S3_PICK,
        's6_place_default': S6_PLACE,
        'poses': CALIBRATED,
    }, indent=2))
    print(f'  Written to {RESULTS}')
else:
    print('  Skipped.')


# -- STEP 8: Summary -----------------------------------------------------------

section('CALIBRATION COMPLETE -- SUMMARY')

try:
    final = pi_get('/arm/touch_poses')
    fp = final.get('touch_poses') or final.get('poses') or {}
    print('  Arm memory poses:')
    for name in ['home', 'inspect', 'pick_table', 'lift_grip', 'place_bin']:
        if name in fp:
            vals = {str(k): int(v) for k, v in fp[name].items()}
            print(f'    {name:<14}  {ps(vals)}')
except Exception as e:
    print(f'  Could not read arm memory: {e}')

print()
print(f'  S6 calibration results:')
print(f'    GREEN lighter (right):  S6 = {S6_GREEN}')
print(f'    BLUE  lighter (left):   S6 = {S6_BLUE}')
print(f'    Pick height S3:         S3 = {S3_PICK}  (official = 180)')
print(f'    Place default:          S6 = {S6_PLACE}')

print()
frames = sorted(FRAMES.glob('*.jpg'))
print(f'  Frames saved ({len(frames)} total): {FRAMES}/')
for f in frames:
    print(f'    {f.name} ({f.stat().st_size:,} bytes)')

print()
print('  Next steps:')
print('    python cosmos_cookoff_demo.py       # run full pick-and-place demo')
print('    python run_demo.py green            # demo: pick green lighter -> place left')
print('    python run_demo.py blue             # demo: pick blue lighter -> place right')
print()
print(SEP)
print('  DONE')
print(SEP)
