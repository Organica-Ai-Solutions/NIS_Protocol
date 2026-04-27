"""
CALIBRATION FROM A RECORDED VIDEO
===================================
The easiest calibration method -- no timing pressure.

Step 1 (YOU DO THIS):
  Record a short video (~30 seconds) with your phone or the Pi camera.
  In the video:
    a) Put tape markers on the table at 5 positions (we tell you where)
    b) Place the lighter on marker 1 -> hold 3 seconds -> move to marker 2 -> hold 3s -> ...
    c) Keep the arm STILL at HOME (or move it out of the way)

Step 2 (THIS SCRIPT DOES):
  - Breaks the video into frames
  - Detects the lighter in every frame
  - Groups frames into 5 segments (one per marker position)
  - Picks the best frame from each segment
  - Computes precise affine calibration

Step 3:
  - Run python vision_pick.py and you're done

MARKER POSITIONS (tape on table):
  The arm coordinate system uses:
    x = right/left (cm)   y = forward distance from arm base (cm)

  Put tape at these ARM COORDINATES:
    A: (0, 17)    <- center (straight ahead ~17cm from base)
    B: (8, 17)    <- 8cm to the right of A
    C: (-8, 17)   <- 8cm to the left of A
    D: (0, 13.5)  <- closer (same line as A, 3.5cm nearer)
    E: (0, 20.5)  <- farther (same line as A, 3.5cm away)

  To find position A: run the arm to S6=500 (center), S4=870, S5=680
    (HOME position) and note where the gripper tip points to the table.
    Or use a ruler: 17cm forward from the arm base center.

Usage:
  python calibrate_from_recorded_video.py --video path/to/video.mp4
  python calibrate_from_recorded_video.py --video video.mp4 --n_marks 5
  python calibrate_from_recorded_video.py --capture     # record from Pi cam now
  python calibrate_from_recorded_video.py --pi_burst    # controlled Pi burst

  # If you have OpenCV installed:
  pip install opencv-python
"""

import json, math, time, sys, base64, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--video',       default='', help='Path to recorded video file')
parser.add_argument('--n_marks',     type=int, default=5, help='Number of marker positions')
parser.add_argument('--capture',     action='store_true', help='Record from Pi camera first')
parser.add_argument('--pi_burst',    action='store_true', help='Use Pi API burst (controlled)')
parser.add_argument('--arm_guided',  action='store_true',
                    help='Arm hovers over each spot; place lighter under it then arm moves away for photo')
parser.add_argument('--verify',      action='store_true')
args = parser.parse_args()

PI     = 'http://192.168.1.163:8085'
CAMCAL = Path('data/camera_cal.json')
CALIB  = Path('data/calib_results.json')
VFRAMES= Path('data/video_frames')
VFRAMES.mkdir(parents=True, exist_ok=True)
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

# ============================================================================
# MARKER POSITIONS (arm coordinates)
# ============================================================================

MARKERS = [
    ('A', 0.0,  17.0),   # center
    ('B', 8.0,  17.0),   # right
    ('C', -8.0, 17.0),   # left
    ('D', 0.0,  13.5),   # near
    ('E', 0.0,  20.5),   # far
]

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
    print(f'  [{label}] {"SIM" if sim else "OK"}')
    if sim: post('/arm/reconnect'); time.sleep(2.5)
    time.sleep(ms/1000.0 + 0.3)


# ============================================================================
# PRINT MARKER POSITIONS GUIDE
# ============================================================================

print(SEP)
print('  CALIBRATION FROM RECORDED VIDEO')
print(SEP)
print()
print('  MARKER POSITIONS (put tape/stickers on table at these spots):')
print()

calib  = json.loads(CALIB.read_text()) if CALIB.exists() else {}
Z_MARK = calib.get('z_pick', 0.7) + 2.5

for name, ax, ay in MARKERS[:args.n_marks]:
    sv = ki(ax, ay, Z_MARK, -71)
    r  = math.sqrt(ax**2 + ay**2)
    print(f'  Marker {name}: arm({ax:+.1f}cm, {ay:.1f}cm forward)  r={r:.1f}cm  S6={sv["6"]}')

print()
print('  To find position A (center, 17cm forward):')
print('    - Use a ruler from the arm base center: 17cm straight forward')
print('    - Or run the arm to S6=500 HOME and look where the gripper tip points')
print('    - B = A + 8cm to the RIGHT')
print('    - C = A - 8cm to the LEFT  (your left when facing arm)')
print('    - D = A - 3.5cm closer to arm')
print('    - E = A + 3.5cm farther from arm')
print()
print('  VIDEO RECORDING STEPS:')
print('    1. Place tape markers A-E on the table')
print('    2. Start recording with phone camera (looking down at table)')
print('       OR use the Pi top-down camera')
print('    3. Place lighter on A -> hold 4 seconds')
print('    4. Move lighter to B -> hold 4 seconds')
print('    5. Move to C -> 4s, D -> 4s, E -> 4s')
print('    6. Stop recording')
print()


# ============================================================================
# OPTION 1: PROCESS EXISTING VIDEO FILE
# ============================================================================

def process_video_file(video_path):
    """Extract calibration points from a recorded video file."""
    try:
        import cv2
    except ImportError:
        print('  ERROR: OpenCV required. Install: pip install opencv-python')
        sys.exit(1)

    import numpy as np

    print(f'  Processing: {video_path}')
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps
    print(f'  Video: {w_v}x{h_v}  {fps:.1f}fps  {total} frames  {dur:.1f}s')
    print()

    # Extract all frames, detect lighter in each
    print('  Analyzing frames...')
    all_dets = []   # (frame_idx, cx, cy, conf)

    SKIP_LABELS = {'bench','chair','table','couch','potted plant','tv'}
    STATIC = [(655, 475, 70)]   # known false positives

    def in_workspace(cx2, cy2, bw):
        if not (180 < cx2 < w_v-100 and int(h_v*0.4) < cy2 < int(h_v*0.9)):
            return False
        if bw > w_v // 3:
            return False
        for sx, sy, sr in STATIC:
            if (cx2-sx)**2 + (cy2-sy)**2 < sr**2:
                return False
        return True

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Save every 5th frame
        if frame_idx % 5 == 0:
            p = VFRAMES / f'f{frame_idx:05d}.jpg'
            cv2.imwrite(str(p), frame)

        # Detect colored object (HSV-based for lighter)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect bright colored objects (lighter is typically orange/yellow/red)
        best_cx, best_cy, best_area = None, None, 0
        for (lo, hi) in [
            # Orange-red range
            (np.array([0,100,100]),   np.array([15,255,255])),
            (np.array([160,100,100]), np.array([180,255,255])),
            # Yellow
            (np.array([15,100,100]),  np.array([35,255,255])),
            # Bright any color (high saturation)
            (np.array([0,150,150]),   np.array([180,255,255])),
        ]:
            mask = cv2.inRange(hsv, lo, hi)
            # Remove noise
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=3)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200 or area > 30000:  # too small or too large
                    continue
                M2  = cv2.moments(cnt)
                if M2['m00'] == 0: continue
                cx2 = int(M2['m10']/M2['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                bw  = cv2.boundingRect(cnt)[2]
                if in_workspace(cx2, cy2, bw) and area > best_area:
                    best_cx, best_cy, best_area = cx2, cy2, area

        if best_cx is not None:
            conf = min(1.0, best_area / 5000.0)
            all_dets.append((frame_idx, best_cx, best_cy, conf))

        if frame_idx % 50 == 0:
            print(f'    Frame {frame_idx}/{total}  detections so far: {len(all_dets)}',
                  end='\r')

    cap.release()
    print(f'\n  Analyzed {frame_idx} frames, {len(all_dets)} with lighter detected')
    print()

    if not all_dets:
        print('  No lighter detected. Try different color ranges.')
        return []

    # Group into N segments (one per marker position)
    n_marks = args.n_marks
    seg_len  = frame_idx / n_marks
    collected = []

    for seg_i in range(n_marks):
        seg_start = int(seg_i * seg_len)
        seg_end   = int((seg_i + 1) * seg_len)
        seg_dets  = [(f,cx2,cy2,c) for f,cx2,cy2,c in all_dets
                     if seg_start <= f < seg_end]

        marker_name, ax, ay = MARKERS[seg_i]
        print(f'  Segment {seg_i+1} [{marker_name}: arm({ax:+.1f},{ay:.1f})]:'
              f'  {len(seg_dets)} detections')

        if seg_dets:
            # Pick most stable detection (median cx, cy weighted by conf)
            weights = np.array([c for _,_,_,c in seg_dets])
            cx_all  = np.array([cx2 for _,cx2,_,_ in seg_dets])
            cy_all  = np.array([cy2 for _,_,cy2,_ in seg_dets])
            cx_med  = float(np.average(cx_all, weights=weights))
            cy_med  = float(np.average(cy_all, weights=weights))
            best_c  = float(max(weights))
            print(f'    -> cam({cx_med:.0f}, {cy_med:.0f})  conf={best_c:.2f}')
            collected.append((ax, ay, int(cx_med), int(cy_med), best_c, marker_name))
        else:
            print(f'    -> NO DETECTION in this segment')

    return collected


# ============================================================================
# OPTION 2: CAPTURE FROM PI CAMERA (burst recording)
# ============================================================================

def capture_pi_burst():
    """Record a burst from Pi camera while user moves lighter between markers."""
    print('  PI CAMERA BURST RECORDING')
    print('  The Pi camera will record frames continuously.')
    print('  You have 40 seconds total: ~8 seconds per marker position.')
    print()
    print('  Sequence:')
    for i, (name, ax, ay) in enumerate(MARKERS[:args.n_marks]):
        t = i * 8
        print(f'    {t:2d}s - {t+7:2d}s: Place lighter at marker {name} '
              f'(arm {ax:+.1f}cm, {ay:.1f}cm)')
    print()

    # Move arm to CLEAR VIEW (out of the way)
    print('  Moving arm to CLEAR VIEW (left, out of workspace)...')
    CLEAR = {**HOME, '6': 750}
    move(CLEAR, ms=2000, label='CLEAR')
    time.sleep(0.5)

    # Capture background reference frames BEFORE lighter is placed
    print('  Capturing background reference (DO NOT place lighter yet)...')
    bg_ref_frames = []
    for i in range(5):
        d = get('/camera/snapshot', t=10)
        img = d.get('image_base64') or d.get('image')
        if img:
            p = VFRAMES / f'bg_ref_{i:02d}.jpg'
            p.write_bytes(base64.b64decode(img))
            bg_ref_frames.append(str(p))
        time.sleep(0.3)
    print(f'  Background captured ({len(bg_ref_frames)} frames)')
    print()
    print('  RECORDING IN 3...  <- GET LIGHTER READY AT MARKER A')
    time.sleep(1)
    print('  RECORDING IN 2...')
    time.sleep(1)
    print('  RECORDING IN 1...')
    time.sleep(1)
    print('  RECORDING! Move lighter through markers A->B->C->D->E')
    print()

    frames = []
    start  = time.time()
    total_t = args.n_marks * 8 + 2

    while time.time() - start < total_t:
        elapsed = time.time() - start
        seg     = min(int(elapsed / 8), args.n_marks - 1)
        name, ax, ay = MARKERS[seg]
        print(f'  t={elapsed:.1f}s  [{name}] arm({ax:+.1f},{ay:.1f})  '
              f'grabbing frame...', end='\r')

        d   = get('/camera/snapshot', t=10)
        img = d.get('image_base64') or d.get('image')
        if img:
            t_ms = int((time.time()-start)*1000)
            p    = VFRAMES / f'burst_{t_ms:06d}.jpg'
            p.write_bytes(base64.b64decode(img))
            frames.append((t_ms, seg, str(p)))

    print(f'\n  Recorded {len(frames)} frames in {total_t:.0f}s')
    print()

    # Process frames with detection
    try:
        import cv2
        import numpy as np
    except ImportError:
        print('  OpenCV not available -- frames saved, cannot auto-detect')
        print(f'  Frames: {VFRAMES}')
        return []

    collected = []
    seg_dets = {i: [] for i in range(args.n_marks)}

    # Build background model from dedicated reference frames (arm at clear, no lighter)
    bg_ref_paths = sorted(VFRAMES.glob('bg_ref_*.jpg'))
    raw_bg = [cv2.imread(str(p)) for p in bg_ref_paths]
    bg_frames = [f for f in raw_bg if f is not None]
    if not bg_frames:
        # Fallback: use first 5 burst frames
        bg_frames = [cv2.imread(fp) for _, _, fp in frames[:5] if cv2.imread(fp) is not None]
    if bg_frames:
        bg_gray = cv2.cvtColor(
            np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in bg_frames],
                               axis=0).astype(np.float32), axis=0).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )[:, :, 0]
        print(f'  Background model built from {len(bg_frames)} reference frames')
        print(f'  Using frame-differencing to isolate moving lighter (eliminates static objects)')
    else:
        bg_gray = None
        print('  WARNING: No background model - static objects may cause false detections')

    for t_ms, seg_i, fp in frames:
        frame = cv2.imread(fp)
        if frame is None: continue
        h_v, w_v = frame.shape[:2]

        # Frame differencing: only look at pixels that CHANGED from background
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bg_gray is not None and gray.shape == bg_gray.shape:
            diff = cv2.absdiff(gray, bg_gray)
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=4)
        else:
            motion_mask = np.ones((h_v, w_v), dtype=np.uint8) * 255

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_cx, best_cy, best_area = None, None, 0
        for (lo, hi) in [
            (np.array([0,80,80]),    np.array([20,255,255])),   # orange-red
            (np.array([155,80,80]),  np.array([180,255,255])),  # red wrap
            (np.array([15,80,80]),   np.array([40,255,255])),   # yellow
            (np.array([0,120,120]),  np.array([180,255,255])),  # bright any
        ]:
            color_mask = cv2.inRange(hsv, lo, hi)
            # AND with motion mask - ONLY detect where something MOVED
            combined = cv2.bitwise_and(color_mask, motion_mask)
            combined = cv2.erode(combined, None, iterations=1)
            combined = cv2.dilate(combined, None, iterations=3)
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 150 or area > 40000: continue
                M2  = cv2.moments(cnt)
                if M2['m00'] == 0: continue
                cx2 = int(M2['m10']/M2['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                bw  = cv2.boundingRect(cnt)[2]
                ok  = (100 < cx2 < w_v-50 and
                       int(h_v*0.25) < cy2 < int(h_v*0.95) and
                       bw < w_v//2)
                if ok and area > best_area:
                    best_cx, best_cy, best_area = cx2, cy2, area

        if best_cx:
            conf = min(1.0, best_area/4000.0)
            seg_dets[seg_i].append((best_cx, best_cy, conf))

    for seg_i, dets in seg_dets.items():
        name, ax, ay = MARKERS[seg_i]
        print(f'  Segment {seg_i+1} [{name}]: {len(dets)} detections', end='')
        if dets:
            weights = np.array([c for _,_,c in dets])
            cx_avg = float(np.average([cx2 for cx2,_,_ in dets], weights=weights))
            cy_avg = float(np.average([cy2 for _,cy2,_ in dets], weights=weights))
            best_c = float(max(weights))
            print(f'  -> cam({cx_avg:.0f},{cy_avg:.0f}) conf={best_c:.2f}')
            collected.append((ax, ay, int(cx_avg), int(cy_avg), best_c, name))
        else:
            print('  -> no detection')

    return collected


# ============================================================================
# COMPUTE AND SAVE CALIBRATION
# ============================================================================

def compute_and_save(collected):
    import numpy as np

    if len(collected) < 4:
        print(f'\n  ERROR: Need >= 4 points, got {len(collected)}')
        return

    pts = [(ax,ay,cx2,cy2) for ax,ay,cx2,cy2,c,l in collected]
    N   = len(pts)
    A   = np.zeros((2*N, 6))
    b   = np.zeros(2*N)
    for i, (ax,ay,cx2,cy2) in enumerate(pts):
        A[2*i,   :] = [cx2, cy2, 1, 0, 0, 0]
        A[2*i+1, :] = [0, 0, 0, cx2, cy2, 1]
        b[2*i]      = ax
        b[2*i+1]    = ay
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    m00,m01,t0,m10,m11,t1 = sol
    M = np.array([[m00,m01],[m10,m11]])
    T = np.array([t0, t1])

    print()
    print(SEP)
    print('  CALIBRATION RESULTS')
    print(SEP)
    print(f'  x_arm = {m00:.6f}*cam_x + {m01:.6f}*cam_y + {t0:.4f}')
    print(f'  y_arm = {m10:.6f}*cam_x + {m11:.6f}*cam_y + {t1:.4f}')
    print()

    errors = []
    for ax,ay,cx2,cy2,c,lbl2 in collected:
        pred = M @ np.array([cx2,cy2]) + T
        ex, ey = pred[0]-ax, pred[1]-ay
        err = math.sqrt(ex**2+ey**2)
        errors.append(err)
        ok = 'OK' if err<1.5 else 'WARN'
        print(f'  [{lbl2}] arm({ax:+.1f},{ay:.1f}) pred({pred[0]:.2f},{pred[1]:.2f})'
              f' err({ex:+.3f},{ey:+.3f}) [{ok}]')

    rms = math.sqrt(sum(e**2 for e in errors)/len(errors))
    print(f'\n  RMS: {rms:.3f}cm', end='')
    if rms<0.5: print('  [EXCELLENT]')
    elif rms<1.0: print('  [GOOD]')
    elif rms<2.0: print('  [ACCEPTABLE]')
    else: print('  [POOR]')

    scale_x = abs(1/m00) if abs(m00)>1e-6 else 12.6
    scale_y = abs(1/m11) if abs(m11)>1e-6 else 12.6

    cc = json.loads(CAMCAL.read_text()) if CAMCAL.exists() else {}
    cc.update({
        'cam_w':1280,'cam_h':720,'cam_cx':640,'cam_cy':360,
        'scale_x': round(scale_x,3), 'scale_y': round(scale_y,3),
        'affine_M': M.tolist(), 'affine_T': T.tolist(),
        'affine_rms_cm': round(rms,4),
        'n_points': len(collected),
        'known_points': [[float(ax),float(ay),int(cx2),int(cy2)]
                          for ax,ay,cx2,cy2,c,l in collected],
        'method': 'video-recorded-affine',
        'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    })
    CAMCAL.write_text(json.dumps(cc, indent=2))
    print(f'\n  Saved: {CAMCAL}')

    calib2 = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    calib2['camera_affine'] = {'M':M.tolist(),'T':T.tolist(),'rms_cm':round(rms,4)}
    CALIB.write_text(json.dumps(calib2, indent=2))
    print(f'  Updated: {CALIB}')
    return M, T, rms


# ============================================================================
# ARM-GUIDED CALIBRATION  (new, most reliable method)
# ============================================================================

HOVER_Z   = 5.0    # hover height when pointing to marker (cm above table)
HOVER_ALP = -20.0  # approach angle for hover

def capture_arm_guided():
    """
    Most reliable calibration:
      1. Arm hovers over each marker position at low height
      2. User places lighter under the gripper tip (8 seconds)
      3. Arm moves to clear view (S6=750, looking left)
      4. Camera snaps a photo - lighter is stationary on table
      5. Detect lighter in photo
      6. Arm returns to next marker
    """
    print()
    print(SEP)
    print('  ARM-GUIDED CALIBRATION')
    print(SEP)
    print()
    print('  The arm will hover over each marker position.')
    print('  Place the lighter directly under the gripper tip.')
    print('  Then the arm moves away and takes a photo.')
    print()
    print('  You have 8 seconds per marker to place the lighter.')
    print('  The arm tip will show you exactly where to put it.')
    print()

    # Move arm to CLEAR VIEW first - background taken HERE (arm stays at clear view for all snaps)
    CLEAR = {**HOME, '6': 750}
    print('  Moving arm to CLEAR VIEW for background reference...')
    move(CLEAR, ms=2500, label='CLEAR_BG')
    time.sleep(1.0)

    # Take 3 background frames (arm at CLEAR VIEW, NO lighter on table yet)
    print('  Capturing background (DO NOT place lighter yet)...')
    bg_imgs = []
    for i in range(3):
        d   = get('/camera/snapshot', t=10)
        img = d.get('image_base64') or d.get('image')
        if img:
            p = VFRAMES / f'bg_guided_{i}.jpg'
            p.write_bytes(base64.b64decode(img))
            bg_imgs.append(str(p))
        time.sleep(0.4)

    bg_path = Path(bg_imgs[0]) if bg_imgs else None
    print(f'  Background reference saved ({len(bg_imgs)} frames).')
    print()

    collected_pts = []

    for idx, (name, ax, ay) in enumerate(MARKERS[:args.n_marks]):
        print(f'  ---- Marker {name} ({idx+1}/{args.n_marks}) ----')
        print(f'  Arm coords: x={ax:+.1f}cm, y={ay:.1f}cm')

        # Compute IK for hover position
        hover_svs = ki(ax, ay, HOVER_Z, HOVER_ALP)
        print(f'  Moving arm to hover over marker {name}...')
        move(hover_svs, ms=2000, label=f'HOVER_{name}')

        # Countdown - user places lighter under gripper
        for t in range(8, 0, -1):
            print(f'  PLACE LIGHTER UNDER GRIPPER TIP   {t}s remaining...', end='\r')
            time.sleep(1)
        print()
        print(f'  LIGHTER PLACED - moving arm to clear view...')

        # Move arm to clear view (out of the way)
        # Return arm to SAME CLEAR VIEW position as background reference
        move(CLEAR, ms=1800, label='CLEAR')
        time.sleep(0.8)  # let camera settle

        # Take 3 snapshots - arm is at same CLEAR VIEW as background, only lighter differs
        best_cx, best_cy, best_conf = None, None, 0.0
        for snap_i in range(3):
            d    = get('/camera/snapshot', t=10)
            img  = d.get('image_base64') or d.get('image')
            if not img:
                time.sleep(0.5)
                continue
            snap_path = VFRAMES / f'guided_{name}_{snap_i}.jpg'
            snap_path.write_bytes(base64.b64decode(img))

            try:
                import cv2, numpy as np
                frame = cv2.imread(str(snap_path))
                if frame is None: continue
                h_v, w_v = frame.shape[:2]

                # Background subtraction: arm at same position -> only lighter is new
                bg_mask = None
                if bg_path and bg_path.exists():
                    bg = cv2.imread(str(bg_path))
                    if bg is not None and bg.shape == frame.shape:
                        gray_f  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
                        diff    = cv2.absdiff(gray_f, gray_bg)
                        diff    = cv2.GaussianBlur(diff, (5, 5), 0)
                        _, bg_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                        bg_mask = cv2.dilate(bg_mask, None, iterations=4)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cx_best, cy_best, area_best = None, None, 0
                for lo, hi in [
                    (np.array([0,60,60]),   np.array([25,255,255])),   # orange-red
                    (np.array([150,60,60]), np.array([180,255,255])),  # red wrap
                    (np.array([15,60,60]),  np.array([45,255,255])),   # yellow-orange
                    (np.array([0,80,100]),  np.array([180,255,255])),  # bright any
                ]:
                    color_mask = cv2.inRange(hsv, lo, hi)
                    if bg_mask is not None:
                        combined = cv2.bitwise_and(color_mask, bg_mask)
                    else:
                        combined = color_mask
                    combined = cv2.erode(combined, None, iterations=1)
                    combined = cv2.dilate(combined, None, iterations=3)
                    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        area = cv2.contourArea(cnt)
                        # Lighter is 1-5cm² ≈ 100-6000px at this camera distance
                        if area < 80 or area > 8000: continue
                        M2 = cv2.moments(cnt)
                        if M2['m00'] == 0: continue
                        cx2 = int(M2['m10']/M2['m00'])
                        cy2 = int(M2['m01']/M2['m00'])
                        if (50 < cx2 < w_v-30 and
                                int(h_v*0.1) < cy2 < int(h_v*0.98) and
                                area > area_best):
                            cx_best, cy_best, area_best = cx2, cy2, area

                if cx_best:
                    conf = min(1.0, area_best/2000.0)
                    if conf > best_conf:
                        best_cx, best_cy, best_conf = cx_best, cy_best, conf
                    print(f'  Snap {snap_i+1}: DETECTED cam({cx_best},{cy_best}) '
                          f'area={area_best:.0f} conf={conf:.2f}')
                else:
                    print(f'  Snap {snap_i+1}: no detection')
            except ImportError:
                print('  (OpenCV not available - cannot auto-detect, save frames manually)')

            time.sleep(0.4)

        if best_cx is not None:
            print(f'  [GOOD] Marker {name}: cam({best_cx},{best_cy}) conf={best_conf:.2f}')
            collected_pts.append((ax, ay, best_cx, best_cy, best_conf, name))
        else:
            print(f'  [MISS] Marker {name}: could not detect lighter in any snapshot')
            print(f'         (Make sure lighter has a bright/colored surface visible to camera)')

        print()
        # Return home briefly before next marker
        if idx < args.n_marks - 1:
            move(HOME, ms=1500, label='HOME')
            time.sleep(0.3)

    move(HOME, ms=1800, label='DONE')
    print(f'  Collected {len(collected_pts)}/{args.n_marks} points')
    return collected_pts


# ============================================================================
# MAIN
# ============================================================================

collected = []

if args.video:
    collected = process_video_file(args.video)
elif args.arm_guided:
    collected = capture_arm_guided()
elif args.capture or args.pi_burst:
    collected = capture_pi_burst()
else:
    print('  Choose an option:')
    print()
    print('  BEST: Arm guides you to each spot, no timing pressure:')
    print('    python calibrate_from_recorded_video.py --arm_guided')
    print()
    print('  Option 2 -- Record using Pi camera (40-second burst):')
    print('    python calibrate_from_recorded_video.py --pi_burst')
    print()
    print('  Option 3 -- Record video with phone/camera, then run:')
    print('    python calibrate_from_recorded_video.py --video my_video.mp4')
    print()
    sys.exit(0)

if collected:
    result = compute_and_save(collected)
    if result:
        M, T, rms = result
        print()
        print(SEP)
        print('  CALIBRATION COMPLETE')
        print(SEP)
        print(f'  RMS error: {rms:.3f}cm')
        print()
        print('  Run: python vision_pick.py   (uses new calibration automatically)')


