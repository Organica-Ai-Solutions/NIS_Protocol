"""
CAM CALIBRATE  - Arm-Guided Camera Calibration
================================================
The arm acts as a POINTER. It hovers over a known arm-coordinate, you place
the lighter directly under the gripper tip, then the arm swings away and the
camera photographs the lighter in place.

This builds a pixel -> arm-coordinate mapping with REAL data points, not guesses.

CONFIRMED ANCHOR: Point A (x=0, y=17, S6=500) was physically verified on
2026-02-27 by a successful pick. We always include this point.

5 POINTS we collect:
  A: center           x= 0.0  y=17.0  S6=500
  B: right ~8cm       x= 8.0  y=15.0  S6=380
  C: left  ~8cm       x=-8.0  y=15.0  S6=620
  D: near center      x= 0.0  y=13.5  S6=500
  E: far center       x= 0.0  y=20.5  S6=500

Usage:
  python cam_calibrate.py             # collect all 5 points  (default)
  python cam_calibrate.py --points A  # collect only point A (quick sanity check)
  python cam_calibrate.py --wait 15   # give yourself more time to place lighter
  python cam_calibrate.py --verify    # run without arm moves (just compute from saved data)

After collecting points:
  - Saves camera_cal.json with affine transform
  - Updates calib_results.json
  - Run: python vision_pick.py --dry-run  to test
"""

import json, math, time, sys, base64, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--points', default='ABCDE',
                    help='Which marker points to collect (e.g. ABCDE or just AB)')
parser.add_argument('--wait',   type=int, default=12,
                    help='Seconds to wait while placing lighter under gripper tip')
parser.add_argument('--verify', action='store_true',
                    help='Skip arm moves, recompute calibration from saved data only')
parser.add_argument('--show',   action='store_true',
                    help='Print existing saved calibration points and exit')
args = parser.parse_args()

PI     = 'http://192.168.1.163:8085'
CAMCAL = Path('data/camera_cal.json')
CALIB  = Path('data/calib_results.json')
CALPTS = Path('data/cal_points.json')
FRAMES = Path('data/cam_cal_frames')
FRAMES.mkdir(parents=True, exist_ok=True)
CALPTS.parent.mkdir(parents=True, exist_ok=True)
SEP    = '=' * 68

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

def snap_raw(name):
    d   = get('/camera/snapshot', t=25)
    img = d.get('image_base64') or d.get('image')
    if img:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        print(f'    [cam] {p.name}  ({p.stat().st_size:,}b)')
        return base64.b64decode(img)
    print('    [cam] FAILED'); return None

def move(svs, ms=1200, label='', extra=0.0):
    r   = post('/arm/group_move', {'positions': svs, 'duration_ms': ms})
    sim = r.get('simulation', False)
    s   = ' '.join(f'S{k}={v}' for k, v in sorted(svs.items()))
    print(f'    [{label}] {s}  [{"SIM" if sim else "OK"}]')
    if sim:
        post('/arm/reconnect'); time.sleep(2.5)
        post('/arm/group_move', {'positions': svs, 'duration_ms': ms})
    time.sleep(ms / 1000.0 + 0.3 + extra)

def countdown(n, msg=''):
    for t in range(n, 0, -1):
        print(f'  {msg}  {t}s ', end='\r')
        time.sleep(1)
    print()


# ============================================================================
# KINEMATICS (same as guided_pick.py)
# ============================================================================
L1, L2, L3, L4     = 6.9, 9.5, 9.5, 16.9
S6_SC               = 375.0 / 90.0
_H_T1, _H_T2, _H_T3 = 45.4, 88.6, -134.0
_H_S5, _H_S4, _H_S3 = 680, 870, 310
S5_SC, S4_SC, S3_SC  = 5.84, 4.09, 8.97

def ki(x, y, z, alpha, grip=100):
    tb  = math.degrees(math.atan2(x, y))
    s6  = max(100, min(900, round(500.0 - tb * S6_SC)))
    r   = math.sqrt(x * x + y * y)
    ar  = math.radians(alpha)
    ex, ey = L4 * math.cos(ar), L4 * math.sin(ar)
    px  = r - ex
    py  = (z - L1) - ey
    d   = max(abs(L2 - L3) + 0.01, min(L2 + L3 - 0.01, math.sqrt(px*px + py*py)))
    c2  = max(-1.0, min(1.0, (d*d - L2*L2 - L3*L3) / (2*L2*L3)))
    t2  = math.degrees(math.acos(c2))
    k1  = L2 + L3 * math.cos(math.radians(t2))
    k2  = L3 * math.sin(math.radians(t2))
    t1  = math.degrees(math.atan2(py, px) - math.atan2(k2, k1))
    t3  = alpha - t1 - t2
    s5  = max(100, min(900, round(_H_S5 + (t1 - _H_T1) * S5_SC)))
    s4  = max(100, min(900, round(_H_S4 + (t2 - _H_T2) * S4_SC)))
    s3  = max(100, min(900, round(_H_S3 + (t3 - _H_T3) * S3_SC)))
    return {'1': grip, '2': 500, '3': s3, '4': s4, '5': s5, '6': s6}


# ============================================================================
# MARKER DEFINITIONS
# ============================================================================
# Each marker: name, arm_x, arm_y, hover_z, alpha
MARKERS = {
    'A': ( 0.0, 17.0, 6.0, -65),   # center forward  (CONFIRMED WORKING)
    'B': ( 8.0, 15.0, 6.0, -65),   # right
    'C': (-8.0, 15.0, 6.0, -65),   # left
    'D': ( 0.0, 13.5, 6.0, -65),   # near
    'E': ( 0.0, 20.5, 6.0, -65),   # far
}

HOME       = {'1': 100, '2': 500, '3': 310, '4': 870, '5': 680, '6': 500}
CLEAR_VIEW = {**HOME, '6': 750}  # arm swings left, clear view of workspace


# ============================================================================
# YELLOW LIGHTER DETECTOR
# ============================================================================

def detect_yellow_in_image(img_bytes, bg_bytes=None, name='frame'):
    """
    Find the yellow lighter in an image.
    Uses HSV color filtering. If bg_bytes provided, uses background subtraction.
    Returns (cx, cy, area) or (None, None, 0) if not found.
    """
    try:
        import numpy as np
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Try to decode with cv2, fall back to basic approach
        try:
            import cv2
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('cv2 decode failed')

            # Background subtraction if background provided
            mask_bg = None
            if bg_bytes is not None:
                bg_arr = np.frombuffer(bg_bytes, np.uint8)
                bg_img = cv2.imdecode(bg_arr, cv2.IMREAD_COLOR)
                if bg_img is not None:
                    diff = cv2.absdiff(img, bg_img)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, mask_bg = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel)

            # Yellow HSV mask
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Yellow: hue 15-40, high sat, high val
            lower_y = np.array([12, 80, 80])
            upper_y = np.array([40, 255, 255])
            mask_y = cv2.inRange(hsv, lower_y, upper_y)

            # Combine masks
            if mask_bg is not None:
                mask = cv2.bitwise_and(mask_y, mask_bg)
            else:
                mask = mask_y

            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Save debug mask
            debug_path = FRAMES / f'{name}_mask.jpg'
            cv2.imwrite(str(debug_path), mask)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f'    [detect] No yellow contours found')
                return None, None, 0

            # Workspace filter (exclude top 250px = arm joints, edges)
            workspace_contours = []
            for c in contours:
                M = cv2.moments(c)
                if M['m00'] < 200:
                    continue
                cx2 = int(M['m10'] / M['m00'])
                cy2 = int(M['m01'] / M['m00'])
                if 100 < cx2 < 1180 and 250 < cy2 < 680:
                    workspace_contours.append((c, cx2, cy2, cv2.contourArea(c)))

            if not workspace_contours:
                print(f'    [detect] Yellow found but outside workspace')
                return None, None, 0

            # Take largest
            workspace_contours.sort(key=lambda t: t[3], reverse=True)
            _, cx2, cy2, area = workspace_contours[0]
            print(f'    [detect] Yellow at cam({cx2}, {cy2})  area={area:.0f}px^2')
            return cx2, cy2, area

        except ImportError:
            # No cv2 - use PIL for basic yellow detection
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            w, h = img.size
            pixels = list(img.getdata())
            yellow_px = []
            for i, (r, g, b) in enumerate(pixels):
                if r > 150 and g > 130 and b < 100 and r > b + 60 and g > b + 50:
                    y2 = i // w
                    x2 = i % w
                    if 100 < x2 < w - 100 and 250 < y2 < h - 40:
                        yellow_px.append((x2, y2))
            if len(yellow_px) < 30:
                print(f'    [detect-PIL] Only {len(yellow_px)} yellow pixels')
                return None, None, 0
            cx2 = int(sum(p[0] for p in yellow_px) / len(yellow_px))
            cy2 = int(sum(p[1] for p in yellow_px) / len(yellow_px))
            print(f'    [detect-PIL] Yellow centroid cam({cx2},{cy2})  {len(yellow_px)}px')
            return cx2, cy2, len(yellow_px)

    except Exception as e:
        print(f'    [detect] ERROR: {e}')
        return None, None, 0


# ============================================================================
# SHOW EXISTING POINTS
# ============================================================================

if args.show:
    if CALPTS.exists():
        pts = json.loads(CALPTS.read_text())
        print(SEP)
        print('  SAVED CALIBRATION POINTS')
        print(SEP)
        for name, d in pts.items():
            print(f'  {name}: arm({d["arm_x"]:.1f},{d["arm_y"]:.1f})  '
                  f'cam({d["cam_x"]},{d["cam_y"]})  area={d.get("area",0):.0f}')
        print()
    else:
        print('  No saved points. Run without --show to collect.')
    sys.exit(0)


# ============================================================================
# LOAD/MERGE EXISTING POINTS
# ============================================================================

saved_pts = {}
if CALPTS.exists():
    saved_pts = json.loads(CALPTS.read_text())
    print(f'  Loaded {len(saved_pts)} existing calibration points')

# Points to collect this run
to_collect = [c for c in args.points.upper() if c in MARKERS]
if not to_collect:
    print('ERROR: No valid markers in --points. Use A B C D E')
    sys.exit(1)


# ============================================================================
# MAIN COLLECTION LOOP
# ============================================================================

if not args.verify:
    print(SEP)
    print('  CAM CALIBRATE  - ARM GUIDED')
    print(SEP)

    h = get('/health')
    print(f'  Agent: {h.get("service")}  xArm: {h.get("xarm")}')
    if h.get('xarm_simulation'):
        post('/arm/reconnect'); time.sleep(2.5)

    print()
    print('  PROCEDURE:')
    print('  1. Arm hovers over marker position (6cm above table)')
    print('  2. You place the YELLOW LIGHTER directly under the gripper tip')
    print('  3. Arm swings away to CLEAR VIEW')
    print('  4. Camera photographs the lighter')
    print('  5. Script detects lighter pixel coordinates')
    print()

    # Start at HOME
    move(HOME, ms=2000, label='HOME')
    time.sleep(0.5)

    for pt_name in to_collect:
        arm_x, arm_y, hover_z, alpha = MARKERS[pt_name]
        hover_sv = ki(arm_x, arm_y, hover_z, alpha)

        print()
        print(SEP)
        print(f'  MARKER {pt_name}:  arm({arm_x:.1f}, {arm_y:.1f})')
        print(SEP)

        # Move arm to HOME first
        print('  -> HOME')
        move(HOME, ms=1500, label='HOME')

        # Aim S6 first
        aim_sv = {**HOME, '6': hover_sv['6']}
        print(f'  -> AIM  S6={hover_sv["6"]}')
        move(aim_sv, ms=1000, label='AIM')

        # Lower to hover height
        print(f'  -> HOVER  (6cm above table)')
        move(hover_sv, ms=1500, label='HOVER', extra=0.3)

        # Show where arm is
        hover_path = FRAMES / f'marker_{pt_name}_hover.jpg'
        snap_raw(f'marker_{pt_name}_hover.jpg')

        print()
        print('  +----------------------------------------------------------+')
        print(f'  |  PLACE THE LIGHTER UNDER THE GRIPPER TIP NOW           |')
        print(f'  |  The gripper is 6cm above the table                     |')
        print(f'  |  You have {args.wait} seconds                           |')
        print('  +----------------------------------------------------------+')
        print()
        countdown(args.wait, msg='Waiting...')

        # Capture background reference BEFORE arm moves (lighter is in scene, arm in hover)
        print('  -> Snapping with lighter in place...')
        with_lighter = snap_raw(f'marker_{pt_name}_with_lighter.jpg')

        # Move arm to CLEAR VIEW
        print('  -> CLEAR VIEW (arm swings left so camera has clear shot)')
        move(CLEAR_VIEW, ms=1200, label='CLEAR', extra=0.5)

        # Snap background (no arm over workspace)
        print('  -> Snapping clear background...')
        bg_bytes = snap_raw(f'marker_{pt_name}_bg.jpg')

        # Move arm BACK to hover to re-detect with arm removed from background
        # (lighter is still on table at the same spot)
        print('  -> Snapping lighter from clear view...')
        time.sleep(0.3)
        lighter_bytes = snap_raw(f'marker_{pt_name}_detect.jpg')

        # Detect yellow lighter
        print(f'  Detecting yellow lighter...')
        cx2, cy2, area = detect_yellow_in_image(lighter_bytes, bg_bytes,
                                                  name=f'marker_{pt_name}')

        if cx2 is None:
            # Try without background subtraction
            print('  Retrying detection without background subtraction...')
            cx2, cy2, area = detect_yellow_in_image(lighter_bytes,
                                                      name=f'marker_{pt_name}_noBG')

        if cx2 is None:
            print(f'  [SKIP] Could not detect lighter at marker {pt_name}')
            print(f'  Make sure the YELLOW LIGHTER was placed under the gripper tip!')
            print()
            # Return to HOME before next marker
            move(HOME, ms=1500, label='HOME')
            continue

        print(f'  POINT {pt_name}: arm({arm_x:.1f},{arm_y:.1f}) <-> cam({cx2},{cy2})')
        saved_pts[pt_name] = {
            'arm_x': arm_x,
            'arm_y': arm_y,
            'cam_x': cx2,
            'cam_y': cy2,
            'area': area,
            'collected_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        CALPTS.write_text(json.dumps(saved_pts, indent=2))
        print(f'  Saved point {pt_name} -> {CALPTS}')

        # Return to HOME before next marker
        move(HOME, ms=1500, label='HOME')
        time.sleep(0.5)

    print()
    print('  Collection done. Computing calibration...')


# ============================================================================
# COMPUTE CALIBRATION FROM COLLECTED POINTS
# ============================================================================

print()
print(SEP)
print('  COMPUTING CALIBRATION')
print(SEP)

pts = saved_pts
if len(pts) < 2:
    print(f'  ERROR: Need at least 2 points, have {len(pts)}. '
          f'Run arm-guided collection first.')
    sys.exit(1)

# Build arrays
arm_coords = [(d['arm_x'], d['arm_y']) for d in pts.values()]
cam_coords = [(d['cam_x'], d['cam_y']) for d in pts.values()]
names_list = list(pts.keys())

print(f'  Using {len(pts)} points: {", ".join(names_list)}')
for name, a, c in zip(names_list, arm_coords, cam_coords):
    print(f'    {name}:  arm({a[0]:+6.1f},{a[1]:+6.1f})  cam({c[0]:4d},{c[1]:3d})')

if len(pts) < 3:
    # Linear scaling only (2 points)
    print()
    print('  WARNING: Only 2 points - using linear scaling (less accurate)')
    print('  Collect more points for better accuracy (run again for remaining markers)')

    # Scale from pixel to cm based on 2-point span
    (a1x, a1y), (c1x, c1y) = arm_coords[0], cam_coords[0]
    (a2x, a2y), (c2x, c2y) = arm_coords[1], cam_coords[1]

    if abs(c2x - c1x) > 10:
        scale_x = abs(a2x - a1x) / abs(c2x - c1x) if abs(c2x - c1x) > 5 else 0.014
    else:
        scale_x = 0.014  # default ~14px/cm

    if abs(c2y - c1y) > 10:
        scale_y = abs(a2y - a1y) / abs(c2y - c1y) if abs(c2y - c1y) > 5 else 0.011
    else:
        scale_y = 0.011

    # Use point A (confirmed) as reference
    ref_arm_x = arm_coords[0][0]
    ref_arm_y = arm_coords[0][1]
    ref_cam_x = cam_coords[0][0]
    ref_cam_y = cam_coords[0][1]

    cam_cx   = ref_cam_x - ref_arm_x / scale_x
    cam_yref = ref_cam_y + ref_arm_y / scale_y

    cal = {
        'method': 'linear_2pt',
        'n_points': len(pts),
        'scale_x': round(1.0 / scale_x, 4),
        'scale_y': round(1.0 / scale_y, 4),
        'cam_cx':   round(cam_cx, 1),
        'cam_yref': round(cam_yref, 1),
        'cam_w': 1280,
        'cam_h': 720,
        'rms_cm': None,
        'points': {k: {'arm': list(a), 'cam': list(c)}
                   for k, a, c in zip(names_list, arm_coords, cam_coords)},
        'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    print(f'  scale_x={cal["scale_x"]}px/cm  scale_y={cal["scale_y"]}px/cm')
    print(f'  cam_cx={cam_cx:.1f}  cam_yref={cam_yref:.1f}')
    rms = None

else:
    # Affine transform (3+ points)
    try:
        import numpy as np

        A_rows = []
        b_rows = []
        for (arm_x, arm_y), (cam_x, cam_y) in zip(arm_coords, cam_coords):
            A_rows.append([cam_x, cam_y, 1, 0, 0, 0])
            A_rows.append([0, 0, 0, cam_x, cam_y, 1])
            b_rows.append(arm_x)
            b_rows.append(arm_y)

        A_mat = np.array(A_rows, dtype=float)
        b_vec = np.array(b_rows, dtype=float)
        params, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

        M = [[params[0], params[1]], [params[3], params[4]]]
        T = [params[2], params[5]]

        # Compute RMS on training data
        residuals = []
        for (arm_x, arm_y), (cam_x, cam_y) in zip(arm_coords, cam_coords):
            pred_x = params[0]*cam_x + params[1]*cam_y + params[2]
            pred_y = params[3]*cam_x + params[4]*cam_y + params[5]
            err = math.sqrt((pred_x-arm_x)**2 + (pred_y-arm_y)**2)
            residuals.append(err)

        rms = round(math.sqrt(sum(r**2 for r in residuals) / len(residuals)), 3)
        print(f'  Affine transform RMS = {rms} cm')
        print(f'  Per-point errors:')
        for name, err in zip(names_list, residuals):
            ok = 'OK' if err < 1.5 else 'HIGH - possible bad data'
            print(f'    {name}: {err:.3f} cm  [{ok}]')

        # Check for bad points (>2x median error)
        import statistics
        med = statistics.median(residuals)
        bad_pts = [n for n, e in zip(names_list, residuals) if e > max(2.0, 2*med)]
        if bad_pts:
            print(f'  WARNING: Points {bad_pts} have high error -- they may have bad detections')
            print(f'  Re-run with --points {"".join(bad_pts)} to re-collect them')

        # Linear fallback values (for vision_pick.py backwards compat)
        # Estimate from affine: at center (cam 640, 360) what arm coords?
        cx_center = 640
        cy_ref = cam_coords[0][1] if cam_coords else 360
        cx2_ref_arm_x = params[0]*cx_center + params[1]*cy_ref + params[2]
        scale_x_est = abs(1.0 / params[0]) if abs(params[0]) > 0.001 else 14.0
        scale_y_est = abs(1.0 / params[4]) if abs(params[4]) > 0.001 else 11.0

        cal = {
            'method': 'affine',
            'n_points': len(pts),
            'affine_M': M,
            'affine_T': T,
            'affine_rms_cm': rms,
            # Linear fallback fields (used by vision_pick.py if affine sanity check fails)
            'scale_x': round(scale_x_est, 2),
            'scale_y': round(scale_y_est, 2),
            'cam_cx':  round(float(cx_center - cx2_ref_arm_x * scale_x_est), 1),
            'cam_yref': round(float(cam_coords[0][1] + arm_coords[0][1] * scale_y_est), 1),
            'cam_w': 1280,
            'cam_h': 720,
            'points': {k: {'arm': list(a), 'cam': list(c)}
                       for k, a, c in zip(names_list, arm_coords, cam_coords)},
            'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }

    except ImportError:
        print('  numpy not available for affine -- using linear scaling')
        rms = None
        # Use point A as reference
        ref = list(pts.values())[0]
        scale_x = 14.0  # px per cm (default)
        scale_y = 11.0
        cam_cx   = ref['cam_x'] - ref['arm_x'] * scale_x
        cam_yref = ref['cam_y'] + ref['arm_y'] * scale_y
        cal = {
            'method': 'linear_no_numpy',
            'n_points': len(pts),
            'scale_x': scale_x,
            'scale_y': scale_y,
            'cam_cx':  round(cam_cx, 1),
            'cam_yref': round(cam_yref, 1),
            'cam_w': 1280, 'cam_h': 720,
            'points': {k: {'arm': list(a), 'cam': list(c)}
                       for k, a, c in zip(names_list, arm_coords, cam_coords)},
            'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }


# ============================================================================
# SAVE RESULTS
# ============================================================================

CAMCAL.write_text(json.dumps(cal, indent=2))
print()
print(f'  Saved: {CAMCAL}')

# Update calib_results.json
if CALIB.exists():
    calib = json.loads(CALIB.read_text())
else:
    calib = {}

if rms is not None:
    calib['camera_affine'] = {
        'M': cal.get('affine_M'),
        'T': cal.get('affine_T'),
        'rms_cm': rms,
    }
calib['camera_cal'] = {
    'scale_x':  cal.get('scale_x', 14.0),
    'scale_y':  cal.get('scale_y', 11.0),
    'cam_y_ref': cal.get('cam_yref', 620.0),
}
calib['cam_calibrated_at']    = cal['calibrated_at']
calib['cam_n_points']         = len(pts)
calib['cam_calibration_rms']  = rms
CALIB.write_text(json.dumps(calib, indent=2))
print(f'  Updated: {CALIB}')

print()
print(SEP)
print('  CALIBRATION COMPLETE')
print(SEP)
if rms is not None:
    qual = 'EXCELLENT' if rms < 0.8 else ('GOOD' if rms < 1.5 else ('OK' if rms < 2.5 else 'POOR'))
    print(f'  RMS error: {rms} cm  [{qual}]')
    if qual == 'POOR':
        print()
        print('  The RMS is high. This usually means:')
        print('  1. The lighter was NOT directly under the gripper tip for some points')
        print('  2. The lighter moved during the arm swing')
        print('  3. The camera detected the wrong object')
        print()
        print('  How to fix:')
        print('  - Re-collect the high-error points shown above')
        print('  - Make sure lighter is SNUG under the gripper tip before countdown ends')
        print('  - Check the _detect.jpg frames in data/cam_cal_frames/')
    else:
        print()
        print('  Ready for autonomous pick!')
        print('  Test with: python vision_pick.py --dry-run')
        print('  Live run:  python vision_pick.py')
else:
    print('  Linear calibration saved (collect 3+ points for affine)')
    print(f'  Currently have {len(pts)} points. Collect more:')
    remaining = [k for k in 'ABCDE' if k not in pts]
    if remaining:
        print(f'  python cam_calibrate.py --points {"".join(remaining)}')
print()
print(f'  Debug frames: {FRAMES}')
