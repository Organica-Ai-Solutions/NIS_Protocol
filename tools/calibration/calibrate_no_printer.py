"""
CAMERA CALIBRATION WITHOUT A PRINTER
======================================
Uses the arm itself as calibration target.

How it works:
  1. Move arm to a 3x3 grid of known (x, y) positions at pick height
  2. At each position, the gripper tip is at a KNOWN arm coordinate
  3. Detect the lighter/gripper in camera -> known pixel position
  4. Build full pixel-to-arm affine mapping from 9 known ground-truth pairs
  5. Save accurate scale, rotation, and offset to camera_cal.json

This is actually MORE accurate than a chessboard because:
  - Ground truth is the arm's own IK (which we trust)
  - No distortion artifacts from paper + lens
  - Covers the exact pick workspace (not a wall target)

Requirements:
  - Place the LIGHTER at the gripper tip position at each step
    (arm moves to position, you place lighter under gripper, press Enter)
  - OR: use a bright sticker on the TABLE under each grid point

Grid positions (arm x, y in cm):
  (-8, 14)  (0, 14)  (8, 14)   <- near row
  (-8, 17)  (0, 17)  (8, 17)   <- mid row  (pick zone)
  (-8, 20)  (0, 20)  (8, 20)   <- far row

Usage:
  python calibrate_no_printer.py          # full 9-point grid calibration
  python calibrate_no_printer.py --quick  # 4-point fast calibration
  python calibrate_no_printer.py --auto   # arm moves, you just confirm each point
  python calibrate_no_printer.py --verify # show live pixel->arm conversion
"""

import json, math, time, sys, base64, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--quick',  action='store_true', help='4-point calibration')
parser.add_argument('--auto',   action='store_true', help='Arm auto-moves, press Enter at each point')
parser.add_argument('--verify', action='store_true', help='Live verification mode')
args = parser.parse_args()

PI     = 'http://192.168.1.163:8085'
CAMCAL = Path('data/camera_cal.json')
CALIB  = Path('data/calib_results.json')
FRAMES = Path('data/cal_arm_frames')
FRAMES.mkdir(parents=True, exist_ok=True)
SEP    = '=' * 68

import urllib.request as _ur
import numpy as np

def get(p, t=15):
    try: return json.loads(_ur.urlopen(PI+p, timeout=t).read())
    except Exception as e: print(f'  GET {p}: {e}'); return {}

def post(p, b=None, t=20):
    try:
        d = json.dumps(b or {}).encode()
        r = _ur.Request(PI+p, data=d, headers={'Content-Type':'application/json'})
        return json.loads(_ur.urlopen(r, timeout=t).read())
    except Exception as e: print(f'  POST {p}: {e}'); return {}

def snap(name=''):
    d   = get('/camera/snapshot', t=20)
    img = d.get('image_base64') or d.get('image')
    if img and name:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        print(f'  [CAM] {name}  ({p.stat().st_size:,}b)')
    return img

def detect():
    d    = get('/vision/detect', t=15)
    dets = d.get('detections', [])
    objs = [x for x in dets if x.get('label','') != 'bench'
            and (x.get('x2',0)-x.get('x1',0)) < 600]
    if not objs:
        objs = [x for x in dets if x.get('label','') != 'bench']
    if objs:
        best = max(objs, key=lambda x: x.get('conf',0))
        return best.get('cx'), best.get('cy'), best.get('label','?'), best.get('conf',0)
    return None, None, None, 0

# ============================================================================
# KINEMATICS - move arm to known positions
# ============================================================================

L1,L2,L3,L4 = 6.9,9.5,9.5,16.9
S6_SC = 375.0/90.0
_H_T1,_H_T2,_H_T3 = 45.4,88.6,-134.0
_H_S5,_H_S4,_H_S3 = 680,870,310
S5_SC,S4_SC,S3_SC = 5.84,4.09,8.97

def ki(x, y, z, alpha):
    tb = math.degrees(math.atan2(x, y))
    s6 = max(100, min(900, round(500.0 - tb * S6_SC)))
    r  = math.sqrt(x*x + y*y)
    ar = math.radians(alpha)
    ex, ey = L4*math.cos(ar), L4*math.sin(ar)
    px = r - ex; py = (z - L1) - ey
    d  = max(abs(L2-L3)+0.01, min(L2+L3-0.01, math.sqrt(px*px+py*py)))
    c2 = max(-1.0, min(1.0, (d*d-L2*L2-L3*L3)/(2*L2*L3)))
    t2 = math.degrees(math.acos(c2))
    k1 = L2 + L3*math.cos(math.radians(t2))
    k2 = L3*math.sin(math.radians(t2))
    t1 = math.degrees(math.atan2(py,px) - math.atan2(k2,k1))
    t3 = alpha - t1 - t2
    s5 = max(100, min(900, round(_H_S5 + (t1-_H_T1)*S5_SC)))
    s4 = max(100, min(900, round(_H_S4 + (t2-_H_T2)*S4_SC)))
    s3 = max(100, min(900, round(_H_S3 + (t3-_H_T3)*S3_SC)))
    return {'1':100,'2':500,'3':s3,'4':s4,'5':s5,'6':s6}

HOME = {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500}

def move(svs, ms=1200, label=''):
    r   = post('/arm/group_move', {'positions': svs, 'duration_ms': ms})
    sim = r.get('simulation', False)
    s   = ' '.join(f'S{k}={v}' for k,v in sorted(svs.items()))
    print(f'  [{label or "MOVE"}] {s}  [{"SIM" if sim else "OK"}]')
    if sim: post('/arm/reconnect'); time.sleep(2.5); post('/arm/group_move',{'positions':svs,'duration_ms':ms})
    time.sleep(ms/1000.0 + 0.4)

# ============================================================================
# CALIBRATION GRID
# ============================================================================

calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}
Z_PICK = calib.get('z_pick', 0.7)

# Use a HOVER height (slightly above pick) so gripper tip visible from above
Z_CAL = Z_PICK + 2.0   # 2cm above table so gripper tip clearly in frame

if args.quick:
    # 4-point L-shape: gives scale_x, scale_y, origin
    GRID = [
        (0.0,  17.0,  'center'),
        (8.0,  17.0,  'right'),
        (-8.0, 17.0,  'left'),
        (0.0,  13.0,  'near'),
        (0.0,  21.0,  'far'),
    ]
else:
    # Full 3x3 grid
    GRID = [
        (-8.0, 20.0, 'far_left'),
        ( 0.0, 20.0, 'far_ctr'),
        ( 8.0, 20.0, 'far_rgt'),
        (-8.0, 17.0, 'mid_left'),
        ( 0.0, 17.0, 'mid_ctr'),
        ( 8.0, 17.0, 'mid_rgt'),
        (-8.0, 14.0, 'near_left'),
        ( 0.0, 14.0, 'near_ctr'),
        ( 8.0, 14.0, 'near_rgt'),
    ]

# ============================================================================
# PREFLIGHT
# ============================================================================

print(SEP)
print('  CAMERA CALIBRATION (NO PRINTER NEEDED)')
print('  Method: arm as calibration target, 9 known ground-truth points')
print(SEP)

h = get('/health')
print(f'  Agent: {h.get("service")} [{h.get("status")}]  xArm={h.get("xarm")}')
if h.get('xarm_simulation'):
    post('/arm/reconnect'); time.sleep(2.5)

cam = snap('preflight.jpg')
w_cam, h_cam = 1280, 720   # read from first snapshot if possible
print(f'  Camera: {w_cam}x{h_cam}')
print()
print(f'  Grid: {len(GRID)} calibration points at z={Z_CAL:.1f}cm (hover above table)')
print()
print('  WHAT YOU NEED:')
print('  - The LIGHTER (or any small bright object)')
print('  - Place it directly under the gripper tip at each position')
print('  - The camera will auto-detect it')
print()

print('  Starting in 3 seconds...')
time.sleep(3)
print()

# ============================================================================
# COLLECT CALIBRATION POINTS
# ============================================================================

print('  Moving arm to HOME...')
move(HOME, ms=2000, label='HOME')
print()

collected = []   # list of (arm_x, arm_y, cam_x, cam_y)

for i, (ax, ay, label) in enumerate(GRID):
    print(f'  [{i+1}/{len(GRID)}] Position: arm({ax:.1f}, {ay:.1f}) [{label}]')

    # Compute IK and move to calibration hover position
    sv = ki(ax, ay, Z_CAL, -71)
    move(sv, ms=1200, label=label)

    # STEP A: capture background frame BEFORE lighter is placed (arm still in hover)
    print(f'  Capturing background (arm is hovering, lighter NOT placed yet)...')
    time.sleep(0.5)
    bg_img = snap(f'bg_{i:02d}_{label}.jpg')
    bg_arr  = None
    if bg_img:
        try:
            import numpy as _n
            import base64 as _b
            raw = _n.frombuffer(_b.b64decode(bg_img), _n.uint8)
            # Simple decode to grayscale via luminance formula
            # (avoid cv2 dependency for basic operation)
            bg_arr = raw
        except Exception:
            pass

    print(f'  >>> NOW place lighter directly under gripper tip <<<')
    print(f'  Waiting 10 seconds for you to position it...')
    for s in range(10, 0, -1):
        print(f'    {s}...', end='\r', flush=True)
        time.sleep(1)
    print('    GO!       ')

    # STEP B: Rotate arm AWAY for clear view, then detect
    print('  Moving arm to CLEAR VIEW for detection...')
    CLEAR = {**HOME, '6': 750}
    move(CLEAR, ms=600, label='CLEAR')
    time.sleep(0.5)

    # STEP C: Try detection multiple times, filter out non-lighter detections
    cx2, cy2, lbl, conf = None, None, None, 0
    # Remember previous detection positions to reject stationary objects
    prev_static = []
    if collected:
        # Previous lighter positions - the CURRENT lighter should be at a NEW spot
        prev_static = [(int(p[2]), int(p[3])) for p in collected]

    for attempt in range(5):
        all_cx2, all_cy2, all_lbl, all_conf = detect()
        if all_cx2 is None:
            print(f'    Attempt {attempt+1}: no detection, retrying...')
            time.sleep(0.8)
            continue
        # Check: is this detection at a NEW position (not same as previous calibration points)?
        is_static = False
        for (px2, py2) in prev_static:
            if abs(all_cx2 - px2) < 30 and abs(all_cy2 - py2) < 30:
                is_static = True
                print(f'    Attempt {attempt+1}: detected {all_lbl} at ({all_cx2},{all_cy2}) '
                      f'-- same as previous point, likely static object, retrying...')
                break
        if not is_static:
            cx2, cy2, lbl, conf = all_cx2, all_cy2, all_lbl, all_conf
            break
        time.sleep(0.8)

    if cx2:
        print(f'  DETECTED: {lbl} ({conf:.2f}) at cam({cx2}, {cy2})')
        snap(f'cal_{i:02d}_{label}.jpg')
        collected.append((ax, ay, cx2, cy2))
        print(f'  => arm({ax:.1f}, {ay:.1f}) <-> cam({cx2}, {cy2})')
    else:
        print(f'  WARNING: No object detected at this position. Skipping.')
        print(f'  TIP: Make sure lighter is visible in top-down camera view.')

    # Return to hover position
    move(sv, ms=800, label='RETURN')
    print()

print('  Moving to HOME...')
move(HOME, ms=1500, label='HOME')
print()

print(f'  Collected {len(collected)}/{len(GRID)} calibration points')
print()

if len(collected) < 4:
    print('  ERROR: Need at least 4 points for calibration.')
    print('  Make sure the lighter is visible and brightly colored.')
    sys.exit(1)

# ============================================================================
# COMPUTE AFFINE CALIBRATION
# ============================================================================

print(SEP)
print('  COMPUTING CALIBRATION')
print(SEP)

pts = np.array(collected, dtype=float)
arm_pts = pts[:, :2]   # arm (x, y)
cam_pts = pts[:, 2:]   # camera (cx, cy)

# Fit affine transform: arm = M * cam + t
# Using least squares for robust fitting

# Build system: for each point:
#   arm_x = m00*cam_x + m01*cam_y + t0
#   arm_y = m10*cam_x + m11*cam_y + t1

N = len(collected)
A = np.zeros((2*N, 6))
b = np.zeros(2*N)

for i, (ax, ay, cx2, cy2) in enumerate(collected):
    A[2*i,   :] = [cx2, cy2, 1, 0, 0, 0]
    A[2*i+1, :] = [0, 0, 0, cx2, cy2, 1]
    b[2*i]      = ax
    b[2*i+1]    = ay

sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
m00, m01, t0, m10, m11, t1 = sol

M = np.array([[m00, m01], [m10, m11]])
T = np.array([t0, t1])

print('  Affine transform matrix:')
print(f'    x_arm = {m00:.6f}*cam_x + {m01:.6f}*cam_y + {t0:.4f}')
print(f'    y_arm = {m10:.6f}*cam_x + {m11:.6f}*cam_y + {t1:.4f}')
print()

# Residuals
print('  Verification (arm coordinates, cm):')
print(f'  {"Label":<12} {"Arm X":>7} {"Arm Y":>7} {"Pred X":>7} {"Pred Y":>7} {"Err X":>7} {"Err Y":>7}')
print(f'  {"-"*12} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*7}')

errors = []
for i, (ax, ay, cx2, cy2) in enumerate(collected):
    label = GRID[i][2] if i < len(GRID) else f'pt{i}'
    cam_v  = np.array([cx2, cy2])
    pred   = M @ cam_v + T
    err_x  = pred[0] - ax
    err_y  = pred[1] - ay
    errors.append(math.sqrt(err_x**2 + err_y**2))
    print(f'  {label:<12} {ax:>7.2f} {ay:>7.2f} {pred[0]:>7.2f} {pred[1]:>7.2f} {err_x:>+7.3f} {err_y:>+7.3f}')

rms_err = math.sqrt(sum(e**2 for e in errors) / len(errors))
max_err = max(errors)
print()
print(f'  RMS error: {rms_err:.3f} cm   Max: {max_err:.3f} cm', end='')
if rms_err < 0.5:
    print('  [EXCELLENT]')
elif rms_err < 1.0:
    print('  [GOOD]')
elif rms_err < 2.0:
    print('  [ACCEPTABLE]')
else:
    print('  [POOR - collect more points]')
print()

# Compute scale (px/cm) from matrix
# scale_x from m00 (dx_arm per dcam_x), scale_y from m11
scale_x = abs(1.0 / m00) if abs(m00) > 1e-6 else 12.6
scale_y = abs(1.0 / m11) if abs(m11) > 1e-6 else 12.6

# For backward compat with simple formulas, compute effective cam origin
# at arm(0,17): cam = M^-1 * (arm - T)
try:
    M_inv      = np.linalg.inv(M)
    cam_at_center = M_inv @ (np.array([0.0, 17.0]) - T)
    cam_cx     = float(cam_at_center[0])
    cam_cy     = float(cam_at_center[1])
    cam_at_base   = M_inv @ (np.array([0.0, 0.0]) - T)
    cam_y_ref  = float(cam_at_base[1])
except Exception:
    cam_cx   = w_cam / 2
    cam_cy   = h_cam / 2
    cam_y_ref = 620.0

print(f'  Effective scale: {scale_x:.2f} px/cm (x)  {scale_y:.2f} px/cm (y)')
print(f'  Camera center (arm origin x=0,y=0): cam({cam_cx:.0f}, {cam_y_ref:.0f})')
print()

# ============================================================================
# SAVE CALIBRATION
# ============================================================================

cc = json.loads(CAMCAL.read_text()) if CAMCAL.exists() else {}
cc.update({
    'cam_w':        w_cam,
    'cam_h':        h_cam,
    'cam_cx':       int(w_cam // 2),
    'cam_cy':       int(h_cam // 2),
    'scale_x':      round(scale_x, 3),
    'scale_y':      round(scale_y, 3),
    'cam_y_ref':    round(cam_y_ref, 1),
    'affine_M':     M.tolist(),
    'affine_T':     T.tolist(),
    'affine_rms_cm': round(rms_err, 4),
    'n_points':     len(collected),
    'known_points': [[float(ax),float(ay),int(cx2),int(cy2)] for ax,ay,cx2,cy2 in collected],
    'method':       'arm-grid-affine',
    'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
})
CAMCAL.write_text(json.dumps(cc, indent=2))
print(f'  Saved: {CAMCAL}')

# Also update CALIB with new camera info
calib['camera_affine'] = {
    'M': M.tolist(), 'T': T.tolist(),
    'rms_cm': round(rms_err, 4),
    'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
CALIB.write_text(json.dumps(calib, indent=2))
print(f'  Updated: {CALIB}')

# ============================================================================
# UPDATE vision_pick.py to use affine transform
# ============================================================================

print()
print(SEP)
print('  UPDATING vision_pick.py to use affine transform...')
print(SEP)

vp = Path('vision_pick.py')
if vp.exists():
    txt = vp.read_text()
    old_fn = '''def cam_to_arm(cam_x, cam_y):
    """
    Top-down 1280x720 camera -> arm (x, y) in cm.
    x_arm = -(cam_x - CAM_CX) / SCALE_X   [inverted: cam-left = arm-right]
    y_arm = (CAM_YREF - cam_y) / SCALE_Y  [inverted: higher in frame = farther]
    """
    x = -(cam_x - CAM_CX) / SCALE_X
    y =  (CAM_YREF - cam_y) / SCALE_Y
    return round(x, 2), round(y, 2)'''

    new_fn = '''def cam_to_arm(cam_x, cam_y):
    """
    Top-down camera -> arm (x, y) in cm.
    Uses affine transform if available (arm-grid calibration),
    falls back to single-point linear mapping.
    """
    if 'affine_M' in (cc if 'cc' in dir() else {}):
        import numpy as np
        v = np.array([cam_x, cam_y])
        arm = cc['affine_M'] @ v + cc['affine_T']
        return round(float(arm[0]), 2), round(float(arm[1]), 2)
    # Linear fallback
    x = -(cam_x - CAM_CX) / SCALE_X
    y =  (CAM_YREF - cam_y) / SCALE_Y
    return round(x, 2), round(y, 2)'''

    if old_fn in txt:
        txt = txt.replace(old_fn, new_fn)
        vp.write_text(txt)
        print('  vision_pick.py: cam_to_arm updated with affine transform')
    else:
        print('  vision_pick.py: function signature changed, update manually')
        print('  Insert this in vision_pick.py cam_to_arm():')
        print('    M = np.array(cc["affine_M"])')
        print('    T = np.array(cc["affine_T"])')
        print('    arm = M @ np.array([cam_x, cam_y]) + T')

# ============================================================================
# VERIFY MODE
# ============================================================================

if args.verify:
    print()
    print(SEP)
    print('  LIVE VERIFY -- place lighter anywhere and watch detection')
    print(SEP)

    M_arr = np.array(cc.get('affine_M', M.tolist()))
    T_arr = np.array(cc.get('affine_T', T.tolist()))

    print(f'  {"cam_x":>6} {"cam_y":>6} {"arm_x":>7} {"arm_y":>7} {"radius":>7}  label')
    for _ in range(8):
        cx2, cy2, lbl, conf = detect()
        if cx2:
            arm = M_arr @ np.array([float(cx2), float(cy2)]) + T_arr
            r   = math.sqrt(arm[0]**2 + arm[1]**2)
            s6  = round(500 - math.degrees(math.atan2(arm[0], arm[1])) * S6_SC)
            s6  = max(100, min(900, s6))
            print(f'  {cx2:>6} {cy2:>6} {arm[0]:>7.2f} {arm[1]:>7.2f} {r:>7.2f}  {lbl} ({conf:.2f}) S6={s6}')
        else:
            print('  -- no detection --')
        time.sleep(1.0)

print()
print(SEP)
print('  CALIBRATION COMPLETE')
print(SEP)
print(f'  Method:   arm-grid-affine ({len(collected)} points)')
print(f'  RMS:      {rms_err:.3f} cm')
print(f'  Scale:    {scale_x:.2f} px/cm (x)  {scale_y:.2f} px/cm (y)')
print(f'  Saved:    {CAMCAL}')
print()
print('  Affine formula (pixels -> arm cm):')
print(f'    x_arm = {m00:.6f}*cam_x + {m01:.6f}*cam_y + {t0:.4f}')
print(f'    y_arm = {m10:.6f}*cam_x + {m11:.6f}*cam_y + {t1:.4f}')
print()
print('  Run: python vision_pick.py            (pick with new calibration)')
print('  Run: python calibrate_no_printer.py --verify  (live accuracy check)')
