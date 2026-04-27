"""
VISION PICK - xArm AI  (Camera Closed-Loop Edition)
====================================================
Uses camera feedback at every critical step:
  1. Detect object -> compute arm coords
  2. Move to approach (safe height, S6 aimed at object)
  3. CAMERA VERIFY: re-detect object, correct S6 if arm drifted
  4. 2-stage descent: high approach -> pick height (prevents table crash)
  5. CAMERA VERIFY: check arm is over object
  6. Close gripper (S1=700, firm grip, 1.5s wait)
  7. PICKUP CONFIRM: lift slightly, check if object left table
  8. Full lift + place

Gripper notes (official Hiwonder xArm AI bus servo range 100-900):
  S1=100  fully open
  S1=700  firm pick grip (use for grasping objects)
  S1=900  fully closed (use for empty-hand)

IK approach fix:
  Previous: z=0.7cm, alpha=-71  -> arm collapses near singularity
  Fixed:    z=1.5cm, alpha=-65  -> S5=~430 (confirmed working 2026-02-27)
"""

import json, math, time, sys, os, base64, argparse
from pathlib import Path

PI     = 'http://192.168.1.163:8085'
CALIB  = Path('data/calib_results.json')
CAMCAL = Path('data/camera_cal.json')
FRAMES = Path('data/vision_pick')
FRAMES.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--reps',      type=int,   default=1)
parser.add_argument('--place',     default='left90',
                    choices=['left90','left45','right45','right90'])
parser.add_argument('--watch',     action='store_true')
parser.add_argument('--dry-run',   action='store_true')
parser.add_argument('--conf',      type=float, default=0.05)
parser.add_argument('--no-verify', action='store_true',
                    help='Skip camera verification steps (faster but less accurate)')
parser.add_argument('--z-pick',    type=float, default=None,
                    help='Override pick height in cm (default: from calib, confirmed=1.5)')
args = parser.parse_args()

DRY = args.dry_run
SEP = '=' * 68

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

def snap(name, label=''):
    d   = get('/camera/snapshot', t=25)
    img = d.get('image_base64') or d.get('image')
    if img:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        tag = f'[{label}] ' if label else ''
        print(f'  {tag}[CAM] {p.name}  ({p.stat().st_size:,}b)')
        return str(p)
    print('  [CAM] failed'); return None

def move(servos, ms=1000, label='', extra=0.0):
    if DRY:
        s = ' '.join(f'S{k}={v}' for k,v in sorted(servos.items()))
        print(f'  [DRY-{label}] {s}')
        time.sleep(0.1); return
    r   = post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim = r.get('simulation', False)
    s   = ' '.join(f'S{k}={v}' for k,v in sorted(servos.items()))
    tag = f'[{label}] ' if label else ''
    print(f'  {tag}{s}  [{"SIM" if sim else "OK"}]')
    if sim:
        post('/arm/reconnect'); time.sleep(2.5)
        post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    time.sleep(ms/1000.0 + 0.3 + extra)


# ============================================================================
# KINEMATICS
# ============================================================================

L1,L2,L3,L4 = 6.9,9.5,9.5,16.9
S6_SC = 375.0/90.0
_H_T1,_H_T2,_H_T3 = 45.4,88.6,-134.0
_H_S5,_H_S4,_H_S3 = 680,870,310
S5_SC,S4_SC,S3_SC  = 5.84,4.09,8.97

# Gripper constants
S1_OPEN  = 100   # fully open
S1_GRIP  = 700   # firm grip for picking (was 500 - too loose!)
S1_CLOSE = 900   # fully closed (empty hand transport)

def ki(x, y, z, alpha, gripper=S1_OPEN):
    """Inverse kinematics. Returns servo dict."""
    tb  = math.degrees(math.atan2(x, y))
    s6  = max(100, min(900, round(500.0 - tb * S6_SC)))
    r   = math.sqrt(x*x + y*y)
    ar  = math.radians(alpha)
    ex, ey = L4*math.cos(ar), L4*math.sin(ar)
    px  = r - ex
    py  = (z - L1) - ey
    d   = max(abs(L2-L3)+0.01, min(L2+L3-0.01, math.sqrt(px*px+py*py)))
    c2  = max(-1.0, min(1.0, (d*d-L2*L2-L3*L3)/(2*L2*L3)))
    t2  = math.degrees(math.acos(c2))
    k1  = L2 + L3*math.cos(math.radians(t2))
    k2  = L3*math.sin(math.radians(t2))
    t1  = math.degrees(math.atan2(py,px) - math.atan2(k2,k1))
    t3  = alpha - t1 - t2
    s5  = max(100, min(900, round(_H_S5 + (t1-_H_T1)*S5_SC)))
    s4  = max(100, min(900, round(_H_S4 + (t2-_H_T2)*S4_SC)))
    s3  = max(100, min(900, round(_H_S3 + (t3-_H_T3)*S3_SC)))
    return {'1':gripper,'2':500,'3':s3,'4':s4,'5':s5,'6':s6}

HOME        = {'1':S1_OPEN,'2':500,'3':310,'4':870,'5':680,'6':500}
CLEAR_VIEW  = {**HOME, '6': 750}   # arm swings left, workspace visible

# Official + corrected place positions
PLACE_S6  = {'left90':875,'left45':685,'right45':315,'right90':125}
PLACE_XY  = {'left90':(-19.5,0),'left45':(-14,14),'right45':(14,14),'right90':(19.5,0)}


# ============================================================================
# LOAD CALIBRATIONS
# ============================================================================

calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}

# --- Pick height: use confirmed value from calib (z=1.5cm confirmed working) ---
# Minimum guard at 1.0cm (below this, arm presses into table at alpha=-65)
_z_from_calib = calib.get('z_pick', 1.5)
Z_PICK = args.z_pick if args.z_pick is not None else max(1.0, _z_from_calib)
PICK_ALPHA = calib.get('pick_alpha', -65)  # -65 confirmed stable (was -71, arm folded)

# Always recompute S3/S4/S5 via IK for the actual Z_PICK and PICK_ALPHA used
# This ensures the servos match the real target height, not a stale calib value
_x_pick = calib.get('x_pick', 0.0) or 0.0
_y_pick = calib.get('y_pick', 17.0) or 17.0
_ik_fresh = ki(_x_pick, _y_pick, Z_PICK, PICK_ALPHA)
S3_PICK = _ik_fresh['3']
S4_PICK = _ik_fresh['4']
S5_PICK = _ik_fresh['5']
print(f'  [pick IK] z={Z_PICK}cm alpha={PICK_ALPHA}deg -> S3={S3_PICK} S4={S4_PICK} S5={S5_PICK}')

S6_CAL   = calib.get('s6_pick', 400)   # fallback S6 if vision fails

# Place joints
PLACE_JOINTS = calib.get('place_joints', {
    'left90':  {'3':220,'4':827,'5':425,'6':875},
    'left45':  {'3':225,'4':827,'5':425,'6':685},
    'right45': {'3':225,'4':827,'5':425,'6':315},
    'right90': {'3':220,'4':827,'5':425,'6':125},
})

# Camera calibration
AFFINE_M = None
AFFINE_T = None
if CAMCAL.exists():
    cc       = json.loads(CAMCAL.read_text())
    CAM_W    = cc.get('cam_w',   1280)
    CAM_H    = cc.get('cam_h',    720)
    CAM_CX   = cc.get('cam_cx',   640)
    SCALE_X  = cc.get('scale_x',  12.6)
    SCALE_Y  = cc.get('scale_y',  12.6)
    CAM_YREF = cc.get('cam_y_ref', 620)
    if 'affine_M' in cc and 'affine_T' in cc:
        import numpy as _np
        AFFINE_M = _np.array(cc['affine_M'])
        AFFINE_T = _np.array(cc['affine_T'])
        rms = cc.get('affine_rms_cm', '?')
        print(f'  Camera cal: AFFINE (RMS={rms}cm)  [affine]')
    elif 'M' in cc and 'T' in cc:
        import numpy as _np
        AFFINE_M = _np.array(cc['M'])
        AFFINE_T = _np.array(cc['T'])
        rms = cc.get('rms_cm', '?')
        print(f'  Camera cal: AFFINE (RMS={rms}cm)  [affine]')
    else:
        print(f'  Camera cal: linear scale=({SCALE_X},{SCALE_Y}) px/cm')
else:
    CAM_W, CAM_H, CAM_CX = 1280, 720, 640
    SCALE_X = SCALE_Y = 12.6
    CAM_YREF = 620.0
    print('  Camera cal not found -- using defaults')


# ============================================================================
# VISION HELPERS
# ============================================================================

def cam_to_arm(cam_x, cam_y):
    """Convert camera pixel coordinates to arm (x, y) in cm."""
    # Try affine first
    if AFFINE_M is not None:
        import numpy as _np
        arm = AFFINE_M @ _np.array([float(cam_x), float(cam_y)]) + AFFINE_T
        xa, ya = round(float(arm[0]), 2), round(float(arm[1]), 2)
        # Sanity check: arm must be reachable (radius 4-23cm, y > 0)
        r = math.sqrt(xa**2 + ya**2)
        if 4.0 < r < 23.0 and ya > 2.0:
            return xa, ya
        # Affine gave unreachable result - warn and fall back to linear
        print(f'  [cam_to_arm] Affine result ({xa:.1f},{ya:.1f}) r={r:.1f}cm UNREACHABLE -- using linear scale')

    # Linear scale fallback (single-point calibration)
    x = -(cam_x - CAM_CX) / SCALE_X
    y =  (CAM_YREF - cam_y) / SCALE_Y
    return round(x, 2), round(y, 2)

def arm_to_pick_servos(x_arm, y_arm, z=None, alpha=None):
    z_use     = z     if z     is not None else Z_PICK
    alpha_use = alpha if alpha is not None else PICK_ALPHA
    sv = ki(x_arm, y_arm, z_use, alpha_use, gripper=S1_OPEN)
    return sv, math.sqrt(x_arm**2 + y_arm**2)

STATIC_ZONES = [(655, 475, 60)]  # known background false-positive area

def in_workspace(cx2, cy2, bw=0):
    if not (150 < cx2 < 1130 and 280 < cy2 < 650):
        return False
    if bw > 500:
        return False
    for sx, sy, sr in STATIC_ZONES:
        if (cx2-sx)**2 + (cy2-sy)**2 < sr**2:
            return False
    return True

def find_object(retries=4, move_clear=True):
    """Detect object in workspace. Moves arm to clear view first."""
    if not DRY and move_clear:
        move(CLEAR_VIEW, ms=800, label='CLEAR-VIEW')
        time.sleep(0.5)

    best = (None, None, None, 0)
    for attempt in range(retries):
        d    = get('/vision/detect', t=15)
        dets = d.get('detections', [])
        SKIP = ('bench','chair','table','couch','potted plant','bed','tv')

        objs = [x for x in dets
                if x.get('label','') not in SKIP
                and in_workspace(x.get('cx',0), x.get('cy',0),
                                  x.get('x2',0)-x.get('x1',0))]

        if objs:
            b = max(objs, key=lambda x: x.get('conf',0))
            cx, cy = b.get('cx'), b.get('cy')
            if cx and cy and b.get('conf',0) > best[3]:
                print(f'    [det-{attempt+1}] {b.get("label","?")} ({b.get("conf",0):.2f}) '
                      f'cam({cx},{cy})')
                best = (cx, cy, b.get('label','?'), b.get('conf',0))

        if best[0]:
            break
        time.sleep(0.6)

    # Cosmos depth fallback
    if not best[0]:
        try:
            dm  = get('/cosmos/depth_map', t=20)
            obs = dm.get('data',{}).get('cosmos_spatial',{}).get('objects',[])
            for o in obs:
                cx2 = int(CAM_W * o.get('px_pct_x',50) / 100)
                cy2 = int(CAM_H * o.get('px_pct_y',50) / 100)
                if in_workspace(cx2, cy2):
                    print(f'  [Cosmos] {o.get("color","?")} -> cam({cx2},{cy2})')
                    best = (cx2, cy2, 'cosmos_'+o.get('color','obj'), 0.5)
                    break
        except Exception:
            pass

    return best


# ============================================================================
# CAMERA CLOSED-LOOP FUNCTIONS
# ============================================================================

def camera_verify_alignment(target_cx, target_cy, current_s6, threshold_px=60):
    """
    Re-detect object after moving to APPROACH.
    If arm is over object (blocking view), object may not be visible - that's OK.
    Returns corrected S6 and updated arm coords, or original if object not visible.
    """
    print(f'  [VERIFY] Checking alignment (target cam={target_cx},{target_cy})...')
    # Don't move arm to clear view - we're checking from current approach position
    d    = get('/vision/detect', t=15)
    dets = d.get('detections', [])
    SKIP = ('bench','chair','table','couch','potted plant')
    objs = [x for x in dets
            if x.get('label','') not in SKIP
            and in_workspace(x.get('cx',0), x.get('cy',0),
                              x.get('x2',0)-x.get('x1',0))]

    if not objs:
        print(f'  [VERIFY] Object not visible from approach (arm may be covering it) - OK')
        return current_s6, None, None

    b   = max(objs, key=lambda x: x.get('conf',0))
    cx2 = b.get('cx')
    cy2 = b.get('cy')
    if not cx2:
        return current_s6, None, None

    dx   = cx2 - target_cx
    dy   = cy2 - target_cy
    dist = math.sqrt(dx*dx + dy*dy)
    print(f'  [VERIFY] Object now at cam({cx2},{cy2})  offset=({dx:+.0f},{dy:+.0f})  dist={dist:.0f}px')

    if dist <= threshold_px:
        print(f'  [VERIFY] ALIGNED - within {threshold_px}px threshold')
        xa2, ya2 = cam_to_arm(cx2, cy2)
        return current_s6, xa2, ya2

    # Object moved or arm shifted - recalculate
    xa2, ya2 = cam_to_arm(cx2, cy2)
    sv2, _   = arm_to_pick_servos(xa2, ya2)
    new_s6   = sv2['6']
    print(f'  [VERIFY] CORRECTING S6: {current_s6} -> {new_s6}  arm({xa2},{ya2})')
    return new_s6, xa2, ya2


def pickup_confirmed(original_cx, original_cy, retries=3):
    """
    After lifting, verify the object LEFT the table.
    Object at same position -> failed pickup.
    Object gone -> successful grab!
    """
    print(f'  [CONFIRM] Verifying pickup (object was at cam({original_cx},{original_cy}))...')

    for attempt in range(retries):
        time.sleep(0.4)
        d    = get('/vision/detect', t=15)
        dets = d.get('detections', [])
        SKIP = ('bench','chair','table','couch','potted plant')
        objs = [x for x in dets
                if x.get('label','') not in SKIP
                and in_workspace(x.get('cx',0), x.get('cy',0),
                                  x.get('x2',0)-x.get('x1',0))]

        still_there = False
        for o in objs:
            cx2, cy2 = o.get('cx',0), o.get('cy',0)
            dist = math.sqrt((cx2-original_cx)**2 + (cy2-original_cy)**2)
            if dist < 100:   # within 100px of original position = still on table
                still_there = True
                print(f'  [CONFIRM] Object STILL at cam({cx2},{cy2}) dist={dist:.0f}px -- MISSED')
                break

        if not still_there:
            print(f'  [CONFIRM] Object GONE from table -- PICKUP SUCCESSFUL!')
            return True

    return False


# ============================================================================
# WATCH MODE
# ============================================================================

if args.watch:
    print(SEP)
    print('  WATCH MODE  (Ctrl+C to stop)')
    print(SEP)
    print(f'  {"cx":>5}  {"cy":>5}  {"x_arm":>7}  {"y_arm":>7}  {"r":>6}  {"S6":>5}  label')
    while True:
        cx2, cy2, lbl, conf = find_object()
        if cx2:
            xa, ya = cam_to_arm(cx2, cy2)
            sv, r  = arm_to_pick_servos(xa, ya)
            print(f'  {cx2:>5}  {cy2:>5}  {xa:>7.2f}  {ya:>7.2f}  {r:>6.1f}  {sv["6"]:>5}  {lbl} ({conf:.2f})')
        else:
            print('  -- no detection --')
        time.sleep(0.8)


# ============================================================================
# PREFLIGHT
# ============================================================================

print(SEP)
print('  VISION PICK  (camera closed-loop)')
print(SEP)

h = get('/health')
print(f'  Agent: {h.get("service")} v{h.get("version")}  [{h.get("status")}]')
print(f'  xArm: connected={h.get("xarm")}  sim={h.get("xarm_simulation")}')
print(f'  Pick height: z={Z_PICK:.2f}cm  alpha={PICK_ALPHA}deg')
print(f'  Pick servos: S3={S3_PICK}  S4={S4_PICK}  S5={S5_PICK}')
print(f'  Gripper: open=S1={S1_OPEN}  grip=S1={S1_GRIP}')
print(f'  Place: {args.place}  S6={PLACE_S6[args.place]}')
print(f'  Camera verify: {"OFF (--no-verify)" if args.no_verify else "ON"}')
print()

if h.get('xarm_simulation'):
    post('/arm/reconnect'); time.sleep(2.5)

# Setup place pose
PLACE_KEY = args.place
S6_PLACE  = PLACE_S6[PLACE_KEY]
pj        = {str(k): int(v) for k,v in PLACE_JOINTS.get(PLACE_KEY, {
    '3':220,'4':827,'5':425,'6':S6_PLACE,
}).items()}

print('Moving to HOME...')
move(HOME, ms=2000, label='HOME', extra=0.5)

success_count = 0
fail_count    = 0

for rep in range(1, args.reps + 1):
    print()
    print(SEP)
    print(f'  REP {rep}/{args.reps}')
    print(SEP)

    # ---- STEP 1: DETECT OBJECT ----
    snap(f'r{rep:02d}_00_scene.jpg', 'SCENE')
    print('  Detecting object...')
    cx2, cy2, lbl, conf = find_object(move_clear=True)

    if cx2:
        xa, ya   = cam_to_arm(cx2, cy2)
        radius   = math.sqrt(xa**2 + ya**2)
        pick_sv, _ = arm_to_pick_servos(xa, ya)
        print(f'  Object: {lbl} ({conf:.2f}) at cam({cx2},{cy2})')
        print(f'  Arm:    x={xa:.2f}cm  y={ya:.2f}cm  r={radius:.1f}cm')
        print(f'  Pick:   S6={pick_sv["6"]}  S3={pick_sv["3"]}  S4={pick_sv["4"]}  S5={pick_sv["5"]}')

        if radius < 5.0 or radius > 23.0:
            print(f'  WARNING: radius={radius:.1f}cm out of safe range [5-23cm] -- skipping')
            fail_count += 1
            continue
    else:
        print('  No object detected -- using calibrated fallback')
        cx2, cy2 = None, None
        theta_r  = math.radians((500.0 - S6_CAL) / S6_SC)
        xa = 17.0 * math.sin(theta_r)
        ya = 17.0 * math.cos(theta_r)
        pick_sv  = {'1':S1_OPEN,'2':500,'3':S3_PICK,'4':S4_PICK,'5':S5_PICK,'6':S6_CAL}
        radius   = 17.0

    # ---- STEP 2: APPROACH (high, safe height) ----
    # Approach at a safe hover height first to avoid table collision
    # HIGH_Z = 7cm is enough to clear objects without singularity
    HIGH_Z     = 7.0
    high_sv    = ki(xa, ya, HIGH_Z, PICK_ALPHA, gripper=S1_OPEN)
    APPROACH   = {**HOME, '6': pick_sv['6']}
    HIGH_HOVER = {**high_sv, '1': S1_OPEN}

    print()
    print('  a) HOME...')
    move(HOME, ms=1500, label='HOME')

    print('  b) APPROACH (pointing at object, HOME height)...')
    move(APPROACH, ms=1200, label='APPROACH')
    snap(f'r{rep:02d}_b_approach.jpg', 'APPROACH')

    # ---- STEP 3: CAMERA VERIFY ALIGNMENT ----
    if not args.no_verify and cx2:
        corrected_s6, xa_new, ya_new = camera_verify_alignment(
            cx2, cy2, pick_sv['6'], threshold_px=80)
        if xa_new is not None and ya_new is not None:
            xa, ya = xa_new, ya_new
        if corrected_s6 != pick_sv['6']:
            pick_sv['6']   = corrected_s6
            high_sv['6']   = corrected_s6
            APPROACH['6']  = corrected_s6
            HIGH_HOVER['6'] = corrected_s6
            # Re-aim
            move(APPROACH, ms=800, label='RE-AIM')

    # ---- STEP 4: HIGH HOVER (5cm above pick) ----
    print('  c) HIGH HOVER...')
    move(HIGH_HOVER, ms=1200, label='HIGH-HOVER')
    snap(f'r{rep:02d}_c_hover.jpg', 'HIGH-HOVER')

    # ---- STEP 5: LOWER TO PICK HEIGHT ----
    PICK_DOWN = {
        '1': S1_OPEN,
        '2': 500,
        '3': pick_sv['3'],
        '4': pick_sv['4'],
        '5': pick_sv['5'],
        '6': pick_sv['6'],
    }
    # Blend from high_hover to pick in 2 steps for smoother descent
    mid_z  = (HIGH_Z + Z_PICK) / 2.0
    mid_sv = ki(xa, ya, mid_z, PICK_ALPHA, gripper=S1_OPEN)
    mid_sv['6'] = pick_sv['6']

    print(f'  d) MID DESCENT (z={mid_z:.1f}cm)...')
    move({**mid_sv, '1': S1_OPEN}, ms=800, label='MID-DESCENT')

    print(f'  e) PICK HEIGHT (z={Z_PICK:.1f}cm)...')
    move(PICK_DOWN, ms=700, label='LOWER', extra=0.2)
    snap(f'r{rep:02d}_e_lower.jpg', 'LOWER')

    # ---- STEP 6: CAMERA CHECK - ARM OVER OBJECT? ----
    if not args.no_verify and cx2:
        print('  [VERIFY] Checking arm position over object...')
        d2   = get('/vision/detect', t=10)
        dets = d2.get('detections', [])
        SKIP = ('bench','chair','table','couch','potted plant')
        obj_visible = any(
            x.get('label','') not in SKIP and
            in_workspace(x.get('cx',0), x.get('cy',0), x.get('x2',0)-x.get('x1',0)) and
            math.sqrt((x.get('cx',0)-cx2)**2 + (x.get('cy',0)-cy2)**2) < 120
            for x in dets
        )
        if obj_visible:
            print('  [VERIFY] Object visible below arm - ready to grip')
        else:
            print('  [VERIFY] Object may be covered by arm - proceeding to grip')

    # ---- STEP 7: CLOSE GRIPPER (FIRM GRIP) ----
    GRIP = {**PICK_DOWN, '1': S1_GRIP}  # S1=700, firm grip
    print(f'  f) GRIP (S1={S1_GRIP})...')
    move(GRIP, ms=600, label='GRIP', extra=1.2)   # close firmly, wait 1.8s total
    snap(f'r{rep:02d}_f_grip.jpg', 'GRIP')

    # ---- STEP 8: LIFT SLIGHTLY to check pickup ----
    LIFT_CHECK = {**HOME, '1': S1_GRIP, '6': pick_sv['6']}
    print('  g) LIFTING...')
    move(LIFT_CHECK, ms=1200, label='LIFT')
    snap(f'r{rep:02d}_g_lift.jpg', 'LIFT')

    # ---- STEP 9: PICKUP CONFIRMATION ----
    picked = True
    if not args.no_verify and cx2:
        picked = pickup_confirmed(cx2, cy2)
        if not picked:
            print('  PICKUP FAILED - object still on table')
            print('  Returning home for retry...')
            move({**LIFT_CHECK, '1': S1_OPEN}, ms=1200, label='RELEASE-FAIL')
            move(HOME, ms=1500, label='HOME-FAIL')
            fail_count += 1
            # Optionally retry
            if rep < args.reps:
                print('  Retrying next rep...')
            continue

    # ---- STEP 10: TRANSPORT TO PLACE ----
    TRANSIT    = {**HOME, '1': S1_GRIP, '6': S6_PLACE}
    PLACE_DOWN = {**pj, '1': S1_GRIP, '2': 500}
    RELEASE    = {**pj, '1': S1_OPEN, '2': 500}

    print('  h) ROTATE TO PLACE...')
    move(TRANSIT, ms=1500, label='ROTATE')
    snap(f'r{rep:02d}_h_transit.jpg', 'TRANSIT')

    print('  i) PLACE DOWN...')
    move(PLACE_DOWN, ms=1000, label='PLACE')
    snap(f'r{rep:02d}_i_place.jpg', 'PLACE')

    print(f'  j) RELEASE (S1={S1_OPEN})...')
    move(RELEASE, ms=600, label='RELEASE', extra=0.5)
    snap(f'r{rep:02d}_j_release.jpg', 'RELEASE')

    print('  k) HOME...')
    move(HOME, ms=1500, label='HOME')
    snap(f'r{rep:02d}_k_done.jpg', 'DONE')

    success_count += 1
    print(f'  REP {rep} COMPLETE - Success!')

    if args.reps > 1 and rep < args.reps:
        print('  Put lighter back at pick zone...')
        time.sleep(3.0)


# ============================================================================
# SUMMARY
# ============================================================================

print()
print(SEP)
print('  VISION PICK COMPLETE')
print(SEP)
print(f'  Success: {success_count}/{args.reps}   Fail: {fail_count}/{args.reps}')
print(f'  Pick height used: z={Z_PICK:.2f}cm  alpha={PICK_ALPHA}deg')
print(f'  Gripper grip: S1={S1_GRIP}  (open=S1={S1_OPEN})')
print()

if not DRY:
    calib.update({
        'last_vision_pick_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'z_pick_effective': Z_PICK,
        'pick_alpha': PICK_ALPHA,
        'gripper_grip_value': S1_GRIP,
        'camera_cal': {'scale_x': SCALE_X, 'scale_y': SCALE_Y, 'cam_y_ref': CAM_YREF}
    })
    CALIB.write_text(json.dumps(calib, indent=2))
    print(f'  Saved: {CALIB}')

print()
print('  If arm still goes too deep:     python vision_pick.py --z-pick 2.0')
print('  If arm too high (not grabbing): python vision_pick.py --z-pick 1.0')
print('  Skip verification:            python vision_pick.py --no-verify')
print()
print(f'  Frames: {FRAMES}')
for f in sorted(FRAMES.glob('*.jpg'))[-8:]:
    print(f'    {f.name}')
