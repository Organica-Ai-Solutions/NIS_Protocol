"""
TOP-DOWN CAMERA CALIBRATION - xArm AI
======================================
Self-calibrating script: sweeps arm to known ki_move positions, captures
snapshots, detects the lighter/marker in each frame, and derives the
pixel-to-cm homography for the overhead 1280x720 camera.

What it discovers:
  scale_x   px/cm  (horizontal)
  scale_y   px/cm  (vertical)
  cam_cx    pixel x that maps to arm x=0 (camera horizontal center)
  cam_y_ref pixel y that maps to arm y=0 (arm base reference)

Confirmed working parameters (2026-02-27):
  HOME:  S1=100 S2=500 S3=310 S4=870 S5=680 S6=500
  PICK:  z=1.5cm, alpha=-65 (NOT -71 — that caused arm collapse)
  Lighter position: x=0, y=17cm (center front)
  S6=500 = center facing forward

Usage:
  python camera_calibrate.py          # full interactive sweep calibration
  python camera_calibrate.py --quick  # skip sweep, use stored data
  python camera_calibrate.py --test   # verify calibration with live detect
"""

import json, math, time, sys, os, base64, struct
from pathlib import Path

PI     = 'http://192.168.1.163:8085'
CALIB  = Path('data/calib_results.json')
CAMCAL = Path('data/camera_cal.json')
FRAMES = Path('data/cam_cal_frames')
FRAMES.mkdir(parents=True, exist_ok=True)

QUICK = '--quick' in sys.argv
TEST  = '--test'  in sys.argv

import urllib.request as _ur

def _get(url, timeout=15):
    return json.loads(_ur.urlopen(url, timeout=timeout).read())

def _post(url, body, timeout=20):
    d = json.dumps(body).encode()
    r = _ur.Request(url, data=d, headers={'Content-Type':'application/json'})
    return json.loads(_ur.urlopen(r, timeout=timeout).read())

def get(p, t=12):
    try: return _get(PI+p, timeout=t)
    except Exception as e: print(f'  GET {p}: {e}'); return {}

def post(p, b=None, t=18):
    try: return _post(PI+p, b or {}, timeout=t)
    except Exception as e: print(f'  POST {p}: {e}'); return {}

def move(servos, ms=1200, label=''):
    r   = post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim = r.get('simulation', False)
    s   = ' '.join(f'S{k}={v}' for k,v in sorted(servos.items()))
    print(f'  [{label}] {s}  [{"SIM" if sim else "OK"}]')
    if sim:
        post('/arm/reconnect'); time.sleep(2.5)
        post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    time.sleep(ms/1000.0 + 0.5)

def snap(name):
    d   = get('/camera/snapshot', t=25)
    img = d.get('image_base64') or d.get('image')
    if img:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        print(f'  [CAM] {p.name}  ({p.stat().st_size:,}b)')
        return str(p)
    print('  [CAM] failed'); return None

def detect():
    d    = get('/vision/detect', t=15)
    dets = d.get('detections', [])
    objs = [x for x in dets if x.get('label','') != 'bench'
            and (x.get('x2',0)-x.get('x1',0)) < 400]
    if not objs:
        objs = [x for x in dets if x.get('label','') != 'bench']
    if not objs and dets:
        objs = dets
    if objs:
        best = max(objs, key=lambda x: x.get('conf', 0))
        return best.get('cx'), best.get('cy'), best.get('label','?'), best.get('conf',0)
    return None, None, None, 0

def cam_resolution():
    d   = get('/camera/snapshot', t=25)
    img = d.get('image_base64') or d.get('image')
    if not img: return 1280, 720
    raw = base64.b64decode(img)
    i = 2
    while i < len(raw)-1:
        if raw[i] == 0xFF:
            mk = raw[i+1]
            if mk in (0xC0, 0xC2):
                h = struct.unpack('>H', raw[i+5:i+7])[0]
                w = struct.unpack('>H', raw[i+7:i+9])[0]
                return w, h
            lg = struct.unpack('>H', raw[i+2:i+4])[0]
            i += 2+lg
        else: i += 1
    return 1280, 720


# ============================================================================
# KINEMATICS
# ============================================================================

L1,L2,L3,L4 = 6.9,9.5,9.5,16.9
S6_SC = 375.0/90.0
_H_T1,_H_T2,_H_T3 = 45.4,88.6,-134.0
_H_S5,_H_S4,_H_S3 = 680,870,310
S5_SC,S4_SC,S3_SC = 5.84,4.09,8.97

def ki_servos(x,y,z,alpha):
    tb = math.degrees(math.atan2(x,y))
    s6 = max(100,min(900,round(500.0-tb*S6_SC)))
    r  = math.sqrt(x*x+y*y)
    ar = math.radians(alpha)
    ex,ey = L4*math.cos(ar), L4*math.sin(ar)
    px = r-ex; py = (z-L1)-ey
    d  = max(abs(L2-L3)+0.01, min(L2+L3-0.01, math.sqrt(px*px+py*py)))
    c2 = max(-1.0,min(1.0,(d*d-L2*L2-L3*L3)/(2*L2*L3)))
    t2 = math.degrees(math.acos(c2))
    k1 = L2+L3*math.cos(math.radians(t2))
    k2 = L3*math.sin(math.radians(t2))
    t1 = math.degrees(math.atan2(py,px)-math.atan2(k2,k1))
    t3 = alpha-t1-t2
    s5 = max(100,min(900,round(_H_S5+(t1-_H_T1)*S5_SC)))
    s4 = max(100,min(900,round(_H_S4+(t2-_H_T2)*S4_SC)))
    s3 = max(100,min(900,round(_H_S3+(t3-_H_T3)*S3_SC)))
    return {'1':100,'2':500,'3':s3,'4':s4,'5':s5,'6':s6}

HOME = {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500}


# ============================================================================
# STEP 0: PREFLIGHT
# ============================================================================

print('='*68)
print('  TOP-DOWN CAMERA CALIBRATION')
print('='*68)

h = get('/health')
print(f'  Agent: {h.get("service")} v{h.get("version")}  [{h.get("status")}]')
print(f'  xArm: connected={h.get("xarm")}  sim={h.get("xarm_simulation")}')
if h.get('xarm_simulation'):
    post('/arm/reconnect'); time.sleep(2.5)

CAM_W, CAM_H = cam_resolution()
CAM_CX, CAM_CY = CAM_W//2, CAM_H//2
print(f'  Camera: {CAM_W}x{CAM_H}  center=({CAM_CX},{CAM_CY})')
print()


# ============================================================================
# STEP 1: LOAD EXISTING CALIBRATION DATA
# ============================================================================

calib = {}
if CALIB.exists():
    calib = json.loads(CALIB.read_text())

# Confirmed data point — lighter at center-front (S6=500, alpha=-65, z=1.5cm)
# Update these after running a manual sweep session.
KNOWN_POINTS = [
    # (arm_x, arm_y, cam_x, cam_y, note)
    # If you have a confirmed point from a previous session, put it here.
    # Example format: (arm_x_cm, arm_y_cm, pixel_x, pixel_y, 'description')
    (0.0, 17.0, 640, 420, 'center pick position estimated — run sweep to confirm'),
]


# ============================================================================
# STEP 2: QUICK MODE - use stored single point
# ============================================================================

if QUICK:
    print('  QUICK MODE: Using stored calibration point + existing data/camera_cal.json.')

    # Try to load existing calibration first
    if CAMCAL.exists():
        existing = json.loads(CAMCAL.read_text())
        scale_x   = existing.get('scale_x',  12.6)
        scale_y   = existing.get('scale_y',  12.6)
        cam_y_ref = existing.get('cam_y_ref', 620.0)
        print(f'  Loaded existing: scale_x={scale_x} scale_y={scale_y} cam_y_ref={cam_y_ref}')
    else:
        ax, ay, cx, cy, _ = KNOWN_POINTS[0]
        # scale_x: if arm x=0 maps to cam_cx, use lateral offset
        if abs(ax) > 0.5:
            scale_x = abs(cx - CAM_CX) / abs(ax)
        else:
            scale_x = 12.6  # default 12.6 px/cm for C270 at typical height
        scale_y   = scale_x
        cam_y_ref = cy + ay * scale_y
        print(f'  Estimated from known point: scale_x={scale_x:.2f} cam_y_ref={cam_y_ref:.0f}')

    print()
    print(f'  scale_x = {scale_x:.2f} px/cm')
    print(f'  scale_y = {scale_y:.2f} px/cm  (assumed = scale_x)')
    print(f'  cam_cx  = {CAM_CX}  (camera center)')
    print(f'  cam_y_ref = {cam_y_ref:.0f}  (arm base y=0 in pixels)')

    cam_cal = {
        'cam_w': CAM_W, 'cam_h': CAM_H,
        'cam_cx': CAM_CX, 'cam_cy': CAM_CY,
        'scale_x': round(scale_x, 3),
        'scale_y': round(scale_y, 3),
        'cam_y_ref': round(cam_y_ref, 1),
        'known_points': KNOWN_POINTS,
        'method': 'single-point',
        'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    CAMCAL.write_text(json.dumps(cam_cal, indent=2))
    print(f'\n  Saved: {CAMCAL}')

    # Verify
    print()
    print('  Verification:')
    for ax2,ay2,cx2,cy2,note in KNOWN_POINTS:
        x_pred = -(cx2 - CAM_CX) / scale_x
        y_pred = (cam_y_ref - cy2) / scale_y
        print(f'    cam({cx2},{cy2}) -> pred({x_pred:.1f},{y_pred:.1f})  actual({ax2},{ay2})  {note}')

    print()
    print('  Run --test to verify with live detect.')
    sys.exit(0)


# ============================================================================
# STEP 3: FULL SWEEP CALIBRATION
# ============================================================================

print('  FULL SWEEP MODE: Moving arm to known positions, detecting marker.')
print('  Place a SMALL BRIGHT MARKER (tape, sticker) directly under the gripper tip.')
print('  The arm will move to 6 positions; confirm marker visible each time.')
print()

# Known arm positions for sweep — confirmed alpha=-65, z=1.5cm
# Place a bright marker (yellow tape, lighter) under the gripper tip at each position.
# alpha=-65 is CONFIRMED stable (alpha=-71 caused arm to fold near singularity).
SWEEP_POSITIONS = [
    # label        x      y     z     alpha
    ('center',    0.0,  17.0,  1.5,  -65),  # confirmed pick position
    ('right_sm',  5.0,  16.2,  1.5,  -65),  # slight right
    ('right_lg', 10.0,  14.1,  1.5,  -65),  # more right
    ('left_sm',  -5.0,  16.2,  1.5,  -65),  # slight left
    ('left_lg', -10.0,  14.1,  1.5,  -65),  # more left
    ('far',       0.0,  19.0,  1.5,  -65),  # farther forward
    ('near',      0.0,  14.0,  1.5,  -65),  # closer
]

input('  Press Enter to start sweep (arm will move to pick height at 7 positions)...')

move(HOME, ms=2000, label='HOME')
time.sleep(0.5)

sweep_data = []
for label, x, y, z, alpha in SWEEP_POSITIONS:
    sv = ki_servos(x, y, z, alpha)
    move(sv, ms=1200, label=label)
    time.sleep(0.5)
    snap(f'sweep_{label}.jpg')
    cx2, cy2, lbl, conf = detect()
    print(f'  Vision: {lbl} at ({cx2},{cy2}) conf={conf:.2f}')
    if cx2 is not None:
        sweep_data.append({
            'label': label, 'arm_x': x, 'arm_y': y,
            'cam_x': cx2, 'cam_y': cy2, 'conf': conf
        })
        print(f'  => arm({x:.1f},{y:.1f}) <-> cam({cx2},{cy2})')
    print()
    move(HOME, ms=800, label='HOME')
    time.sleep(0.3)

print(f'  Sweep complete: {len(sweep_data)}/{len(SWEEP_POSITIONS)} points detected')
print()

if len(sweep_data) < 3:
    print('  WARNING: Not enough points for reliable calibration (need >= 3)')
    print('  Falling back to single-point quick mode...')
    ax, ay, cx, cy, _ = KNOWN_POINTS[0]
    scale_x = abs(cx - CAM_CX) / abs(ax)
    scale_y = scale_x
    cam_y_ref = cy + ay * scale_y
else:
    # Fit scale_x from x-varied points
    x_pairs = [(d['arm_x'], d['cam_x']) for d in sweep_data if abs(d['arm_x']) > 1]
    if x_pairs:
        scales_x = [abs(CAM_CX - cx2) / abs(ax2) for ax2, cx2 in x_pairs if ax2 != 0]
        scale_x = sum(scales_x) / len(scales_x)
    else:
        scale_x = abs(KNOWN_POINTS[0][2] - CAM_CX) / abs(KNOWN_POINTS[0][0])

    # Fit scale_y from y-varied points
    y_pairs = [(d['arm_y'], d['cam_y']) for d in sweep_data]
    if len(y_pairs) >= 2:
        # Linear fit: cam_y = cam_y_ref - arm_y * scale_y
        # For two points: scale_y = (cam_y1 - cam_y2) / (arm_y2 - arm_y1)
        y_scales = []
        for i in range(len(y_pairs)):
            for j in range(i+1, len(y_pairs)):
                ay1, cy1 = y_pairs[i]
                ay2, cy2 = y_pairs[j]
                if abs(ay1 - ay2) > 0.5:
                    y_scales.append(abs(cy1 - cy2) / abs(ay1 - ay2))
        scale_y = sum(y_scales) / len(y_scales) if y_scales else scale_x
        # y_ref: cam_y where y_arm=0
        cam_y_refs = [cy2 + ay2*scale_y for ay2, cy2 in y_pairs]
        cam_y_ref  = sum(cam_y_refs) / len(cam_y_refs)
    else:
        scale_y   = scale_x
        ax, ay, cx, cy, _ = KNOWN_POINTS[0]
        cam_y_ref = cy + ay * scale_y

print(f'  Calibration results:')
print(f'    scale_x  = {scale_x:.2f} px/cm')
print(f'    scale_y  = {scale_y:.2f} px/cm')
print(f'    cam_cx   = {CAM_CX}')
print(f'    cam_y_ref = {cam_y_ref:.0f}  (arm base y=0)')
print()

# Verify on all detected points
print('  Verification:')
for d in sweep_data:
    x_pred = -(d['cam_x'] - CAM_CX) / scale_x
    y_pred = (cam_y_ref - d['cam_y']) / scale_y
    ex = abs(x_pred - d['arm_x'])
    ey = abs(y_pred - d['arm_y'])
    print(f'    {d["label"]:<10} arm({d["arm_x"]:>6.1f},{d["arm_y"]:>5.1f}) '
          f'pred({x_pred:>6.1f},{y_pred:>5.1f})  err({ex:.2f},{ey:.2f})cm')

cam_cal = {
    'cam_w': CAM_W, 'cam_h': CAM_H,
    'cam_cx': CAM_CX, 'cam_cy': CAM_CY,
    'scale_x': round(scale_x, 3),
    'scale_y': round(scale_y, 3),
    'cam_y_ref': round(cam_y_ref, 1),
    'known_points': KNOWN_POINTS,
    'sweep_data': sweep_data,
    'method': 'multi-point-sweep' if len(sweep_data) >= 3 else 'single-point',
    'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
CAMCAL.write_text(json.dumps(cam_cal, indent=2))
print(f'\n  Saved: {CAMCAL}')


# ============================================================================
# TEST MODE
# ============================================================================

if TEST:
    print()
    print('='*68)
    print('  LIVE TEST: Place lighter on table, watch detection')
    print('='*68)

    if CAMCAL.exists():
        cc   = json.loads(CAMCAL.read_text())
        sx   = cc['scale_x']
        sy   = cc['scale_y']
        cyr  = cc['cam_y_ref']
        ccx  = cc['cam_cx']
    else:
        sx, sy, cyr, ccx = scale_x, scale_y, cam_y_ref, CAM_CX

    for _ in range(5):
        cx2, cy2, lbl, conf = detect()
        if cx2 is not None:
            xp = -(cx2 - ccx) / sx
            yp = (cyr - cy2) / sy
            r  = math.sqrt(xp**2 + yp**2)
            s6 = round(500 - math.degrees(math.atan2(xp, yp)) * S6_SC)
            s6 = max(100, min(900, s6))
            print(f'  cam({cx2},{cy2}) -> arm({xp:.1f},{yp:.1f})cm  r={r:.1f}cm  S6={s6}  [{lbl} {conf:.2f}]')
        else:
            print('  No object detected')
        time.sleep(1.5)

print()
print('='*68)
print('  CALIBRATION COMPLETE')
print('='*68)
print(f'  Formula (top-down 1280x720):')
print(f'    x_arm = -(cam_x - {CAM_CX}) / {scale_x:.2f}')
print(f'    y_arm = ({cam_y_ref:.0f} - cam_y) / {scale_y:.2f}')
print(f'    S6    = 500 - atan2(x_arm, y_arm) * 4.167')
print()
print('  Run: python vision_pick.py   to use in pick-and-place')
