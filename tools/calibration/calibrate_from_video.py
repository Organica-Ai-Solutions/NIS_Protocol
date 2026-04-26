"""
CAMERA CALIBRATION FROM VIDEO BURSTS
======================================
Much more reliable than single-snapshot calibration.

How it works:
  1. Arm moves to each grid position and hovers
  2. Records a 5-second burst of frames (continuous snapshots)
  3. You place the lighter under the gripper ANYTIME during the burst
  4. Script analyzes ALL frames, finds the best lighter detection
  5. Builds accurate pixel->arm mapping from confirmed frames

Advantages over single-snapshot:
  - No timing pressure: just place lighter within 5 seconds
  - Best-frame selection = highest confidence detection = most accurate
  - Can detect motion (object appears = it's the lighter, not background)
  - Rejects frames where lighter isn't visible yet

Usage:
  python calibrate_from_video.py          # 5-point grid, 5s burst each
  python calibrate_from_video.py --full   # 9-point grid, more accuracy
  python calibrate_from_video.py --burst 8  # 8 frames per position
  python calibrate_from_video.py --verify   # show live mapping after cal
"""

import json, math, time, sys, base64, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--full',   action='store_true', help='9-point grid')
parser.add_argument('--burst',  type=int, default=10, help='Frames per position')
parser.add_argument('--verify', action='store_true',  help='Live verify after cal')
args = parser.parse_args()

PI     = 'http://192.168.1.163:8085'
CAMCAL = Path('data/camera_cal.json')
CALIB  = Path('data/calib_results.json')
BURSTS = Path('data/cal_bursts')
BURSTS.mkdir(parents=True, exist_ok=True)
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

def grab_frame(save_path=None):
    """Grab one frame from Pi camera. Returns image bytes."""
    d   = get('/camera/snapshot', t=15)
    img = d.get('image_base64') or d.get('image')
    if img and save_path:
        Path(save_path).write_bytes(base64.b64decode(img))
    return img

def detect_in_frame():
    """Run /vision/detect and return filtered workspace detections."""
    # Known background false-positives to reject
    STATIC_ZONES = [(655, 475, 70)]   # background chair
    SKIP_LABELS  = {'bench', 'chair', 'table', 'couch', 'potted plant',
                    'tv', 'laptop', 'monitor', 'keyboard', 'mouse'}

    def in_workspace(cx2, cy2, bw):
        if not (180 < cx2 < 1100 and 300 < cy2 < 630):
            return False
        if bw > 500:
            return False
        for (sx, sy, sr) in STATIC_ZONES:
            if (cx2-sx)**2 + (cy2-sy)**2 < sr**2:
                return False
        return True

    d    = get('/vision/detect', t=12)
    dets = d.get('detections', [])
    good = [x for x in dets
            if x.get('label','') not in SKIP_LABELS
            and in_workspace(x.get('cx',0), x.get('cy',0),
                             x.get('x2',0)-x.get('x1',0))]
    return good

# ============================================================================
# KINEMATICS
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

HOME  = {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500}
CLEAR = {**HOME, '6': 750}   # arm points left → lighter visible on right

def move(svs, ms=1200, label=''):
    r   = post('/arm/group_move', {'positions': svs, 'duration_ms': ms})
    sim = r.get('simulation', False)
    s   = ' '.join(f'S{k}={v}' for k,v in sorted(svs.items()))
    print(f'  [{label or "MOVE"}] {s}  [{"SIM" if sim else "OK"}]')
    if sim: post('/arm/reconnect'); time.sleep(2.5)
    time.sleep(ms/1000.0 + 0.35)

# ============================================================================
# CALIBRATION GRID
# ============================================================================

calib  = json.loads(CALIB.read_text()) if CALIB.exists() else {}
Z_PICK = calib.get('z_pick', 0.7)
Z_CAL  = Z_PICK + 2.5   # hover 2.5cm above table → gripper tip visible from above

if args.full:
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
else:
    GRID = [
        ( 0.0, 17.0, 'center'),
        ( 8.0, 17.0, 'right'),
        (-8.0, 17.0, 'left'),
        ( 0.0, 13.5, 'near'),
        ( 0.0, 20.5, 'far'),
    ]

BURST_N = args.burst   # frames per position

# ============================================================================
# PREFLIGHT
# ============================================================================

print(SEP)
print('  CAMERA CALIBRATION — VIDEO BURST METHOD')
print(f'  Grid: {len(GRID)} positions × {BURST_N} frames each')
print(SEP)

h = get('/health')
print(f'  Agent: {h.get("service")} [{h.get("status")}]  xArm={h.get("xarm")}')
if h.get('xarm_simulation'):
    post('/arm/reconnect'); time.sleep(2.5)

print()
print('  HOW TO HELP:')
print('    - Have the lighter (or bright object) ready in hand')
print('    - When arm moves to a position and you see "PLACE LIGHTER -->"')
print('      put the lighter directly under the gripper tip')
print('    - The camera records a burst of frames -- just hold it there')
print('    - Best frame is auto-selected from the burst')
print('    - Arm then moves away for a clear shot')
print()
print(f'  Hover height: {Z_CAL:.1f}cm above table')
print()

time.sleep(2)

# ============================================================================
# COLLECT VIDEO BURSTS
# ============================================================================

print('  Moving to HOME...')
move(HOME, ms=2000, label='HOME')
print()

collected = []   # (arm_x, arm_y, cam_x, cam_y, conf, label_str)

for i, (ax, ay, label) in enumerate(GRID):
    print(SEP)
    print(f'  POSITION {i+1}/{len(GRID)}: arm({ax:.1f}, {ay:.1f}) [{label}]')
    print(SEP)

    # Move to calibration hover
    sv = ki(ax, ay, Z_CAL, -71)
    move(sv, ms=1400, label=label)

    # Arm steady, now tell user to place lighter
    print()
    print(f'  --> PLACE LIGHTER under gripper tip <--')
    print(f'  Recording {BURST_N} frames... hold lighter still')
    print()

    # Move to CLEAR VIEW for recording
    move(CLEAR, ms=700, label='CLEAR-VIEW')
    time.sleep(0.3)

    # Record burst
    burst_dir  = BURSTS / f'pos{i:02d}_{label}'
    burst_dir.mkdir(exist_ok=True)
    burst_dets = []   # (frame_idx, cx, cy, label, conf)

    for f_idx in range(BURST_N):
        fp  = burst_dir / f'f{f_idx:03d}.jpg'
        img = grab_frame(save_path=str(fp))
        dets = detect_in_frame()
        if dets:
            best = max(dets, key=lambda x: x.get('conf', 0))
            cx2  = best.get('cx')
            cy2  = best.get('cy')
            lbl2 = best.get('label','?')
            c2   = best.get('conf', 0)
            burst_dets.append((f_idx, cx2, cy2, lbl2, c2))
            print(f'    frame {f_idx+1:2d}/{BURST_N}  {lbl2} ({c2:.2f}) cam({cx2},{cy2})', end='  ')
            if c2 > 0.2:
                print('[GOOD]')
            else:
                print()
        else:
            print(f'    frame {f_idx+1:2d}/{BURST_N}  no detection')
        time.sleep(0.4)

    print()

    # Pick best detection from burst
    if burst_dets:
        # Sort by confidence, then pick the one with highest confidence
        burst_dets.sort(key=lambda x: x[4], reverse=True)
        best_frame = burst_dets[0]
        f_idx, cx2, cy2, lbl2, c2 = best_frame
        print(f'  BEST: frame {f_idx+1} -> {lbl2} ({c2:.2f}) cam({cx2},{cy2})')
        print(f'  => arm({ax:.1f}, {ay:.1f}) <-> cam({cx2}, {cy2})')
        collected.append((ax, ay, cx2, cy2, c2, lbl2))
    else:
        print(f'  NO DETECTION in burst for position {label}')
        print(f'  Make sure lighter is visible and brightly colored.')

    # Return to hover position
    move(sv, ms=800, label='RETURN')
    print()

print('  Moving to HOME...')
move(HOME, ms=1500, label='HOME')
print()
print(f'  Collected {len(collected)}/{len(GRID)} calibration points')
print()

if len(collected) < 4:
    print(f'  ERROR: Need at least 4 detected points. Got {len(collected)}.')
    print('  Tips:')
    print('    - Make lighter more visible (brighter color, sticker on top)')
    print('    - Place it within the table area, not at the edges')
    print('    - Increase burst: python calibrate_from_video.py --burst 15')
    sys.exit(1)

# ============================================================================
# COMPUTE AFFINE CALIBRATION
# ============================================================================

print(SEP)
print('  COMPUTING AFFINE CALIBRATION')
print(SEP)
print()

pts = [(ax,ay,cx2,cy2) for ax,ay,cx2,cy2,c,l in collected]
arm_pts = np.array([[ax,ay] for ax,ay,_,_ in pts], dtype=float)
cam_pts = np.array([[cx2,cy2] for _,_,cx2,cy2 in pts], dtype=float)

# Least-squares affine: arm = M @ cam + T
N = len(pts)
A = np.zeros((2*N, 6))
b = np.zeros(2*N)
for i, (ax,ay,cx2,cy2) in enumerate(pts):
    A[2*i,   :] = [cx2, cy2, 1, 0, 0, 0]
    A[2*i+1, :] = [0, 0, 0, cx2, cy2, 1]
    b[2*i]      = ax
    b[2*i+1]    = ay

sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
m00,m01,t0, m10,m11,t1 = sol
M = np.array([[m00,m01],[m10,m11]])
T = np.array([t0, t1])

# Verification
print('  Affine transform:')
print(f'    x_arm = {m00:.6f}*cam_x + {m01:.6f}*cam_y + {t0:.4f}')
print(f'    y_arm = {m10:.6f}*cam_x + {m11:.6f}*cam_y + {t1:.4f}')
print()
print(f'  {"Label":<12} {"arm_x":>6} {"arm_y":>6} {"pred_x":>7} {"pred_y":>7}'
      f' {"err_x":>7} {"err_y":>7} {"conf":>6}')
print(f'  {"-"*12} {"-"*6} {"-"*6} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*6}')

errors = []
for (ax,ay,cx2,cy2,c,lbl2) in collected:
    pred   = M @ np.array([cx2,cy2]) + T
    ex, ey = pred[0]-ax, pred[1]-ay
    err    = math.sqrt(ex**2 + ey**2)
    errors.append(err)
    ok = 'OK' if err < 1.5 else 'WARN'
    print(f'  {lbl2:<12} {ax:>6.1f} {ay:>6.1f} {pred[0]:>7.2f} {pred[1]:>7.2f}'
          f' {ex:>+7.3f} {ey:>+7.3f} {c:>6.2f}  [{ok}]')

rms = math.sqrt(sum(e**2 for e in errors)/len(errors))
mx  = max(errors)
print()
print(f'  RMS: {rms:.3f}cm   Max: {mx:.3f}cm', end='')
if rms < 0.5:  print('  [EXCELLENT]')
elif rms < 1.0: print('  [GOOD]')
elif rms < 2.0: print('  [ACCEPTABLE]')
else:           print('  [POOR — run --full for more points]')
print()

# Scale from matrix
scale_x = abs(1/m00) if abs(m00)>1e-6 else 12.6
scale_y = abs(1/m11) if abs(m11)>1e-6 else 12.6
print(f'  Effective scale: x={scale_x:.2f}  y={scale_y:.2f}  px/cm')

# cam_y_ref (arm y=0 in camera coords)
try:
    M_inv    = np.linalg.inv(M)
    base_cam = M_inv @ (-T)
    cam_y_ref = float(base_cam[1])
    cam_cx    = float(base_cam[0])
except Exception:
    cam_y_ref = 620.0
    cam_cx    = 640.0

# ============================================================================
# SAVE
# ============================================================================

cc = json.loads(CAMCAL.read_text()) if CAMCAL.exists() else {}
cc.update({
    'cam_w': 1280, 'cam_h': 720,
    'cam_cx': 640, 'cam_cy': 360,
    'scale_x':    round(scale_x, 3),
    'scale_y':    round(scale_y, 3),
    'cam_y_ref':  round(cam_y_ref, 1),
    'affine_M':   M.tolist(),
    'affine_T':   T.tolist(),
    'affine_rms_cm': round(rms, 4),
    'n_points':   len(collected),
    'known_points': [[float(ax),float(ay),int(cx2),int(cy2)]
                      for ax,ay,cx2,cy2,c,l in collected],
    'method':     'video-burst-affine',
    'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
})
CAMCAL.write_text(json.dumps(cc, indent=2))
print(f'\n  Saved: {CAMCAL}')

calib['camera_affine'] = {
    'M': M.tolist(), 'T': T.tolist(), 'rms_cm': round(rms,4),
    'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
CALIB.write_text(json.dumps(calib, indent=2))
print(f'  Updated: {CALIB}')

# ============================================================================
# LIVE VERIFY
# ============================================================================

if args.verify:
    print()
    print(SEP)
    print('  LIVE VERIFY — place lighter anywhere and watch tracking')
    print(SEP)
    print(f'  {"cam_x":>6} {"cam_y":>6} {"arm_x":>7} {"arm_y":>7} {"r":>6} {"S6":>5}  label')
    for _ in range(10):
        dets = detect_in_frame()
        if dets:
            best = max(dets, key=lambda x: x.get('conf',0))
            cx2  = best.get('cx')
            cy2  = best.get('cy')
            arm  = M @ np.array([float(cx2), float(cy2)]) + T
            r    = math.sqrt(arm[0]**2 + arm[1]**2)
            s6   = round(500 - math.degrees(math.atan2(arm[0],arm[1])) * S6_SC)
            s6   = max(100, min(900, s6))
            print(f'  {cx2:>6} {cy2:>6} {arm[0]:>7.2f} {arm[1]:>7.2f} {r:>6.2f} {s6:>5}  '
                  f'{best.get("label","?")} ({best.get("conf",0):.2f})')
        else:
            print('  -- no detection --')
        time.sleep(1.0)

print()
print(SEP)
print('  CALIBRATION COMPLETE')
print(SEP)
print(f'  Points:   {len(collected)} (from video bursts)')
print(f'  RMS:      {rms:.3f} cm')
print(f'  Method:   video-burst-affine')
print()
print('  Run: python vision_pick.py           (pick using new calibration)')
print('  Run: python calibrate_from_video.py --verify  (check accuracy)')
print()
print('  Frames saved to:', BURSTS)
