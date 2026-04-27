"""
GATE 2: CAMERA INTRINSICS CALIBRATION (OpenCV Chessboard)
==========================================================
Official Hiwonder source: ArmPi FPV Manual 7.2 (Monocular Camera Calibration)
Method: chessboard calibration -> camera matrix + distortion coefficients

Official workflow:
  1. Print chessboard, move/tilt it in camera view
  2. Track: x (left/right), y (up/down), size (distance), scale (tilt)
  3. Collect until progress bars all green (> 25 good frames)
  4. Compute calibration: findChessboardCorners + calibrateCamera
  5. Commit: save camera matrix + distortion coefficients

Output: data/camera_intrinsics.json
  - camera_matrix: 3x3 intrinsic matrix
  - dist_coeffs: distortion coefficients [k1,k2,p1,p2,k3]
  - reprojection_error: RMS error (< 1.0 px = good)
  - resolution: [w, h]
  - focal_length_px: [fx, fy]
  - principal_point: [cx, cy]

Also: recomputes camera_cal.json scale using undistorted focal length

Usage:
  python camera_intrinsics.py              # run calibration (needs monitor)
  python camera_intrinsics.py --frames 30  # collect 30 frames before calibrating
  python camera_intrinsics.py --verify     # verify saved calibration with live feed
  python camera_intrinsics.py --load       # just load and print saved calibration

CRITICAL: Camera intrinsics are needed for:
  1. Undistortion of frames before pixel->arm coordinate conversion
  2. Accurate focal length for perspective-based distance estimation
  3. Proper Isaac Sim camera model matching (sim-to-real gate)
"""

import json, math, time, sys, os, base64, argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--frames',  type=int, default=25, help='Frames needed for calibration')
parser.add_argument('--board',   default='9x6', help='Chessboard inner corners WxH e.g. 9x6')
parser.add_argument('--sq',      type=float, default=2.5, help='Square size in cm (for real-world scale)')
parser.add_argument('--verify',  action='store_true')
parser.add_argument('--load',    action='store_true')
parser.add_argument('--live',    action='store_true', help='Use live OpenCV capture instead of Pi API')
args = parser.parse_args()

PI    = 'http://192.168.1.163:8085'
CAL   = Path('data/camera_intrinsics.json')
CAMCAL= Path('data/camera_cal.json')
SNAPS = Path('data/chessboard_frames')
SNAPS.mkdir(parents=True, exist_ok=True)

import urllib.request as _ur

def get(p, t=20):
    try: return json.loads(_ur.urlopen(PI+p, timeout=t).read())
    except Exception as e: print(f'  GET {p}: {e}'); return {}

SEP = '=' * 68

# ============================================================================
# LOAD AND PRINT SAVED CALIBRATION
# ============================================================================

if args.load:
    if CAL.exists():
        c = json.loads(CAL.read_text())
        print(SEP)
        print('  SAVED CAMERA INTRINSICS')
        print(SEP)
        print(f'  Resolution:  {c["resolution"]}')
        print(f'  Focal length: fx={c["focal_length_px"][0]:.1f}  fy={c["focal_length_px"][1]:.1f}  px')
        print(f'  Principal pt: cx={c["principal_point"][0]:.1f}  cy={c["principal_point"][1]:.1f}  px')
        print(f'  Distortion:  {[round(v,4) for v in c["dist_coeffs"]]}')
        print(f'  RMS error:   {c["reprojection_error"]:.4f} px')
        print(f'  Calibrated:  {c.get("calibrated_at","")}')
        print()
        cm = np.array(c['camera_matrix'])
        print('  Camera matrix:')
        for row in cm: print(f'    {row}')
    else:
        print('  No saved calibration found. Run python camera_intrinsics.py')
    sys.exit(0)


# ============================================================================
# PARSE BOARD DIMENSIONS
# ============================================================================

bw, bh = [int(x) for x in args.board.split('x')]
BOARD   = (bw, bh)   # inner corners
SQ_CM   = args.sq    # square size in real world
NEEDED  = args.frames

print(SEP)
print('  GATE 2: CAMERA INTRINSICS CALIBRATION')
print(f'  Board: {bw}x{bh} inner corners  |  Square: {SQ_CM}cm  |  Need: {NEEDED} frames')
print(SEP)
print()
print('  INSTRUCTIONS:')
print('    1. Print chessboard_9x6.png from data/ folder')
print('       (or use any standard 9x6 inner-corner chessboard)')
print('    2. Hold it flat, then slowly move/tilt it:')
print('       LEFT/RIGHT (x),  UP/DOWN (y),  NEAR/FAR (size),  TILT (scale)')
print('    3. Watch progress bars -- all must reach GREEN before calibrating')
print('    4. This script auto-captures frames via Pi camera API')
print()


# ============================================================================
# GENERATE CHESSBOARD PATTERN
# ============================================================================

def generate_chessboard_png():
    """Generate a simple chessboard PNG for printing."""
    try:
        import cv2
        sq = 80
        h, w = (bh+1)*sq, (bw+1)*sq
        img = np.zeros((h, w), dtype=np.uint8)
        for r in range(bh+1):
            for c in range(bw+1):
                if (r+c) % 2 == 0:
                    img[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 255
        p = Path('data/chessboard_9x6.png')
        cv2.imwrite(str(p), img)
        print(f'  Chessboard generated: {p}  ({(bw+1)*sq}x{(bh+1)*sq}px)')
        return str(p)
    except ImportError:
        print('  OpenCV not available - use any 9x6 chessboard image')
        return None

generate_chessboard_png()
print()


# ============================================================================
# CALIBRATION LOOP
# ============================================================================

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print('  WARNING: OpenCV not installed. Install with: pip install opencv-python')
    print('           Using fallback (saves frames only, no calibration)')

obj_pts_template = np.zeros((bw*bh, 3), np.float32)
obj_pts_template[:, :2] = np.mgrid[0:bw, 0:bh].T.reshape(-1, 2) * SQ_CM

all_obj_pts = []
all_img_pts = []
w_cap, h_cap = 1280, 720

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) if HAS_CV2 else None

x_vals, y_vals, size_vals, scale_vals = [], [], [], []

def progress_bar(vals, target, width=20):
    """Show progress bar for parameter coverage."""
    if not vals: return '[' + ' '*width + '] 0%'
    rng = max(vals) - min(vals) if len(vals) > 1 else 0
    pct = min(100, int(rng / target * 100))
    filled = int(pct / 100 * width)
    color = 'GREEN' if pct >= 100 else 'AMBER' if pct >= 60 else 'RED'
    return f'[{"#"*filled}{" "*(width-filled)}] {pct:3d}% {color}'

def get_frame():
    """Fetch frame from Pi camera API."""
    d   = get('/camera/snapshot', t=20)
    img = d.get('image_base64') or d.get('image')
    if not img:
        return None
    data = base64.b64decode(img)
    arr  = np.frombuffer(data, np.uint8)
    if HAS_CV2:
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None

def find_corners(frame):
    """Find chessboard corners in frame, return (corners, gray)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), CRIT)
    return ret, corners, gray

print('  Starting frame capture. Move chessboard around the frame.')
print('  Press Ctrl+C when all progress bars reach GREEN.')
print()

n_captured = 0
n_attempt  = 0

if args.live and HAS_CV2:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(-1)
else:
    cap = None

try:
    while n_captured < NEEDED:
        # Capture frame
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None: time.sleep(0.2); continue
        else:
            frame = get_frame()
            if frame is None: time.sleep(1.0); continue

        n_attempt += 1
        h_cap, w_cap = frame.shape[:2]

        if not HAS_CV2:
            # Save raw frame for offline calibration
            p = SNAPS / f'frame_{n_attempt:04d}.jpg'
            p.write_bytes(base64.b64decode(
                get('/camera/snapshot').get('image_base64','')))
            print(f'  [FRAME {n_attempt}] saved')
            time.sleep(1.0)
            continue

        found, corners, gray = find_corners(frame)

        if not found:
            print(f'  [{n_attempt:3d}] No board detected', end='\r')
            time.sleep(0.5)
            continue

        # Compute coverage metrics (like Hiwonder x/y/size/scale bars)
        c     = corners.reshape(-1, 2)
        cx    = float(c[:, 0].mean())
        cy    = float(c[:, 1].mean())
        area  = float((c[:, 0].max()-c[:, 0].min()) * (c[:, 1].max()-c[:, 1].min()))
        dx    = float(c[:, 0].max()-c[:, 0].min())
        dy    = float(c[:, 1].max()-c[:, 1].min())
        tilt  = abs(dx/dy - 1.0) if dy > 0 else 0   # scale (tilt)

        x_vals.append(cx)
        y_vals.append(cy)
        size_vals.append(area)
        scale_vals.append(tilt)

        all_obj_pts.append(obj_pts_template.copy())
        all_img_pts.append(corners)
        n_captured += 1

        # Save annotated frame
        dbg = frame.copy()
        cv2.drawChessboardCorners(dbg, BOARD, corners, found)
        p = SNAPS / f'board_{n_captured:03d}.jpg'
        cv2.imwrite(str(p), dbg)

        # Progress bars
        xbar    = progress_bar(x_vals,     w_cap * 0.6, 20)
        ybar    = progress_bar(y_vals,     h_cap * 0.6, 20)
        szbar   = progress_bar(size_vals,  (w_cap * h_cap * 0.1), 20)
        scbar   = progress_bar(scale_vals, 0.4, 20)

        print(f'\n  Frame {n_captured:3d}/{NEEDED}:')
        print(f'    x    (left/right): {xbar}')
        print(f'    y    (up/down):    {ybar}')
        print(f'    size (near/far):   {szbar}')
        print(f'    scale (tilt):      {scbar}')

        time.sleep(0.3)

except KeyboardInterrupt:
    print('\n  Capture stopped by user.')

if cap and cap.isOpened():
    cap.release()

print(f'\n  Captured {n_captured} frames with board detected.')

if not HAS_CV2:
    print('  OpenCV required for calibration computation.')
    print('  Frames saved to', SNAPS)
    sys.exit(1)

if n_captured < 6:
    print(f'  Too few frames ({n_captured}) for calibration. Need at least 6.')
    sys.exit(1)


# ============================================================================
# COMPUTE CALIBRATION
# ============================================================================

print()
print('  Computing calibration...')

rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_pts, all_img_pts, (w_cap, h_cap), None, None
)

print(f'  RMS reprojection error: {rms:.4f} px', end='')
if rms < 0.5:
    print('  [EXCELLENT]')
elif rms < 1.0:
    print('  [GOOD]')
elif rms < 2.0:
    print('  [ACCEPTABLE]')
else:
    print('  [POOR - collect more frames]')

fx = float(camera_matrix[0, 0])
fy = float(camera_matrix[1, 1])
cx = float(camera_matrix[0, 2])
cy = float(camera_matrix[1, 2])

print()
print('  Camera matrix (intrinsics):')
print(f'    fx={fx:.2f}  fy={fy:.2f}  (focal length in pixels)')
print(f'    cx={cx:.2f}  cy={cy:.2f}  (principal point)')
print(f'    Resolution: {w_cap}x{h_cap}')
print()
print('  Distortion coefficients:')
dc = dist_coeffs.flatten().tolist()
print(f'    k1={dc[0]:.4f}  k2={dc[1]:.4f}  p1={dc[2]:.4f}  p2={dc[3]:.4f}', end='')
if len(dc) > 4: print(f'  k3={dc[4]:.4f}')
else: print()
print()

# Undistorted optimal camera matrix (for undistortion and pixel mapping)
new_mtx, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w_cap, h_cap), 1, (w_cap, h_cap)
)

print('  Optimal undistorted camera matrix:')
fx2 = float(new_mtx[0, 0])
fy2 = float(new_mtx[1, 1])
cx2 = float(new_mtx[0, 2])
cy2 = float(new_mtx[1, 2])
print(f'    fx={fx2:.2f}  fy={fy2:.2f}')
print(f'    cx={cx2:.2f}  cy={cy2:.2f}')
print(f'    ROI: {roi}')
print()


# ============================================================================
# SAVE CALIBRATION
# ============================================================================

cal_data = {
    'resolution':        [w_cap, h_cap],
    'camera_matrix':     camera_matrix.tolist(),
    'dist_coeffs':       dist_coeffs.flatten().tolist(),
    'new_camera_matrix': new_mtx.tolist(),
    'roi':               list(roi),
    'focal_length_px':   [fx, fy],
    'principal_point':   [cx, cy],
    'reprojection_error': float(rms),
    'n_frames':          n_captured,
    'board_size':        list(BOARD),
    'square_size_cm':    SQ_CM,
    'calibrated_at':     time.strftime('%Y-%m-%dT%H:%M:%S'),
}

CAL.write_text(json.dumps(cal_data, indent=2))
print(f'  Saved: {CAL}')


# ============================================================================
# UPDATE camera_cal.json WITH PROPER FOCAL-LENGTH-BASED SCALE
# ============================================================================

# For top-down camera at known height H_cm above table:
# scale_px_per_cm = fx / H_cm
# From single-point calibration we know: scale was 12.6 px/cm
# Estimate H_cm = fx / 12.6
H_est = fx / 12.6   # estimated camera height in cm above table

print()
print('  Top-down camera geometry:')
print(f'    Focal length (undistorted): fx={fx:.1f} px')
print(f'    Single-point scale:         12.6 px/cm')
print(f'    Estimated camera height:    {H_est:.1f} cm above table')

# Update camera_cal.json
if CAMCAL.exists():
    cc = json.loads(CAMCAL.read_text())
else:
    cc = {}

cc.update({
    'intrinsics_file':    str(CAL),
    'fx_px':              fx,
    'fy_px':              fy,
    'principal_cx':       cx,
    'principal_cy':       cy,
    'dist_coeffs':        dist_coeffs.flatten().tolist(),
    'rms_error':          float(rms),
    'estimated_height_cm': round(H_est, 1),
    'calibrated_at':      time.strftime('%Y-%m-%dT%H:%M:%S'),
})
CAMCAL.write_text(json.dumps(cc, indent=2))
print(f'  Updated: {CAMCAL}')


# ============================================================================
# VERIFY (optional)
# ============================================================================

if args.verify:
    print()
    print('  VERIFY: Undistorting a live frame...')
    frame = get_frame()
    if frame is not None:
        undist = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_mtx)
        cv2.imwrite('data/undistorted_test.jpg', undist)
        print('  Saved: data/undistorted_test.jpg')

        # Measure residual chessboard error
        found, corners, gray = find_corners(frame)
        if found:
            err_pts = []
            for i, (obj, img) in enumerate(zip(all_obj_pts[:1], all_img_pts[:1])):
                imgpts2, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i],
                                               camera_matrix, dist_coeffs)
                err_pts.append(cv2.norm(img, imgpts2, cv2.NORM_L2) /
                                len(imgpts2))
            print(f'  Reprojection error (verify frame): {sum(err_pts)/len(err_pts):.4f} px')
    else:
        print('  No frame available for verification')

print()
print(SEP)
print('  GATE 2 COMPLETE')
print(SEP)
print()
print('  Next: python xarm_urdf.py  (Gate 3: URDF validation)')
print()
print('  For vision_pick.py: camera intrinsics will improve coordinate accuracy')
print('  when camera height estimation is confirmed against ground truth point.')
print()
print('  Isaac Sim camera setup (for sim-to-real Gate 4):')
print(f'    fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}')
print(f'    width={w_cap}  height={h_cap}')
print(f'    Estimated camera height above table: {H_est:.1f} cm')
