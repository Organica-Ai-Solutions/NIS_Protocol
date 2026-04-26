"""
GUIDED PICK - Zero calibration required
========================================
The arm hovers over the pick position and SHOWS YOU exactly where to put
the lighter. You place it under the gripper tip, then the arm picks it up.

No camera math. No affine transforms. No calibration needed.

Usage:
  python guided_pick.py              # default: center forward
  python guided_pick.py --s6 420     # aim slightly right
  python guided_pick.py --s6 500     # center
  python guided_pick.py --reps 3     # repeat 3 times
  python guided_pick.py --z 2.0      # pick height (cm above table)
"""

import json, math, time, sys, base64, argparse
from pathlib import Path

PI    = 'http://192.168.1.163:8085'
CALIB = Path('data/calib_results.json')
FRAMES = Path('data/guided_pick')
FRAMES.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--s6',    type=int,   default=500,
                    help='Base rotation for pick. 500=center, 400=right24deg, 600=left24deg')
parser.add_argument('--z',     type=float, default=1.5,
                    help='Pick height cm above table (default 1.5, confirmed working)')
parser.add_argument('--place', default='left90',
                    choices=['left90','left45','right45','right90'])
parser.add_argument('--reps',  type=int,   default=1)
parser.add_argument('--wait',  type=int,   default=10,
                    help='Seconds to wait while you place lighter under gripper')
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

def snap(name, label=''):
    d   = get('/camera/snapshot', t=20)
    img = d.get('image_base64') or d.get('image')
    if img:
        p = FRAMES / name
        p.write_bytes(base64.b64decode(img))
        print(f'  [{label}] snap -> {p.name}')
        return str(p)
    return None

def move(svs, ms=1200, label='', extra=0.0):
    r   = post('/arm/group_move', {'positions': svs, 'duration_ms': ms})
    sim = r.get('simulation', False)
    s   = ' '.join(f'S{k}={v}' for k,v in sorted(svs.items()))
    print(f'  [{label}] {s}  [{"SIM" if sim else "OK"}]')
    if sim:
        post('/arm/reconnect'); time.sleep(2.5)
        post('/arm/group_move', {'positions': svs, 'duration_ms': ms})
    time.sleep(ms/1000.0 + 0.3 + extra)


# ============================================================================
# KINEMATICS
# ============================================================================
L1,L2,L3,L4   = 6.9,9.5,9.5,16.9
S6_SC          = 375.0/90.0
_H_T1,_H_T2,_H_T3 = 45.4,88.6,-134.0
_H_S5,_H_S4,_H_S3 = 680,870,310
S5_SC,S4_SC,S3_SC  = 5.84,4.09,8.97

def ki(x, y, z, alpha, grip=100):
    tb = math.degrees(math.atan2(x, y))
    s6 = max(100, min(900, round(500.0 - tb * S6_SC)))
    r  = math.sqrt(x*x + y*y)
    ar = math.radians(alpha)
    ex,ey = L4*math.cos(ar), L4*math.sin(ar)
    px = r - ex
    py = (z - L1) - ey
    d  = max(abs(L2-L3)+0.01, min(L2+L3-0.01, math.sqrt(px*px+py*py)))
    c2 = max(-1.0, min(1.0, (d*d-L2*L2-L3*L3)/(2*L2*L3)))
    t2 = math.degrees(math.acos(c2))
    k1 = L2 + L3*math.cos(math.radians(t2))
    k2 = L3*math.sin(math.radians(t2))
    t1 = math.degrees(math.atan2(py,px) - math.atan2(k2,k1))
    t3 = alpha - t1 - t2
    s5 = max(100, min(900, round(_H_S5+(t1-_H_T1)*S5_SC)))
    s4 = max(100, min(900, round(_H_S4+(t2-_H_T2)*S4_SC)))
    s3 = max(100, min(900, round(_H_S3+(t3-_H_T3)*S3_SC)))
    return {'1':grip,'2':500,'3':s3,'4':s4,'5':s5,'6':s6}

# Compute arm (x,y) from S6 angle at radius 17cm
theta  = (500.0 - args.s6) / S6_SC
theta_r = math.radians(theta)
PICK_X = round(17.0 * math.sin(theta_r), 2)
PICK_Y = round(17.0 * math.cos(theta_r), 2)
PICK_Z = args.z
ALPHA  = -65   # gentle approach angle

# Derive servo positions
HOVER_SV = ki(PICK_X, PICK_Y, 6.0,    ALPHA, grip=100)   # 6cm above table
MID_SV   = ki(PICK_X, PICK_Y, 3.5,    ALPHA, grip=100)   # mid descent
PICK_SV  = ki(PICK_X, PICK_Y, PICK_Z, ALPHA, grip=100)   # pick height
# Force S6 from user arg (override IK's auto-compute)
HOVER_SV['6'] = args.s6
MID_SV['6']   = args.s6
PICK_SV['6']  = args.s6

HOME = {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500}
PLACE_JOINTS = {
    'left90':  {'3':220,'4':827,'5':425,'6':875},
    'left45':  {'3':225,'4':827,'5':425,'6':685},
    'right45': {'3':225,'4':827,'5':425,'6':315},
    'right90': {'3':220,'4':827,'5':425,'6':125},
}
calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}
pj = {str(k): int(v) for k,v in calib.get('place_joints',{}).get(args.place,
    PLACE_JOINTS[args.place]).items()}

SEP = '=' * 60

# ============================================================================
# MAIN
# ============================================================================

print(SEP)
print('  GUIDED PICK')
print(SEP)
print(f'  Pick target: x={PICK_X:.1f}cm  y={PICK_Y:.1f}cm  z={PICK_Z}cm')
print(f'  S6={args.s6}  (500=center, 400=right, 600=left)')
print(f'  Hover: S3={HOVER_SV["3"]} S4={HOVER_SV["4"]} S5={HOVER_SV["5"]}')
print(f'  Pick:  S3={PICK_SV["3"]} S4={PICK_SV["4"]} S5={PICK_SV["5"]}')
print(f'  Place: {args.place}')
print()

h = get('/health')
print(f'  Agent: {h.get("service")}  xArm: connected={h.get("xarm")}')
print()

if h.get('xarm_simulation'):
    post('/arm/reconnect'); time.sleep(2.5)

for rep in range(1, args.reps + 1):
    print()
    print(SEP)
    print(f'  REP {rep}/{args.reps}')
    print(SEP)

    # HOME
    print('  1) Going to HOME...')
    move(HOME, ms=2000, label='HOME')
    snap(f'r{rep:02d}_a_home.jpg', 'HOME')

    # HOVER - show user where the gripper tip is
    print()
    print('  2) Moving to HOVER position (6cm above table)...')
    move({**HOME, '6': args.s6}, ms=1000, label='AIM')
    move(HOVER_SV, ms=1500, label='HOVER')
    snap(f'r{rep:02d}_b_hover.jpg', 'HOVER')

    print()
    print('  ============================================================')
    print(f'  >>> PLACE LIGHTER DIRECTLY UNDER THE GRIPPER TIP <<<')
    print(f'  >>> The gripper is 6cm above the table right now  <<<')
    print(f'  >>> You have {args.wait} seconds                          <<<')
    print('  ============================================================')
    print()

    for t in range(args.wait, 0, -1):
        print(f'  Placing... {t}s remaining', end='\r')
        time.sleep(1)
    print()
    print('  Proceeding to pick!')
    snap(f'r{rep:02d}_c_placed.jpg', 'PLACED')

    # 2-stage descent to pick height
    print()
    print(f'  3) Descending to MID (z=3.5cm)...')
    move(MID_SV, ms=800, label='MID')

    print(f'  4) PICK HEIGHT (z={PICK_Z}cm)...')
    move(PICK_SV, ms=600, label='LOWER', extra=0.3)
    snap(f'r{rep:02d}_d_lower.jpg', 'LOWER')

    # CLOSE GRIPPER - firm grip
    print()
    print('  5) CLOSING GRIPPER (S1=700, firm)...')
    GRIP = {**PICK_SV, '1': 700}
    move(GRIP, ms=600, label='GRIP', extra=1.5)
    snap(f'r{rep:02d}_e_grip.jpg', 'GRIP')

    # LIFT
    print()
    print('  6) LIFTING...')
    LIFT = {**HOME, '1': 700, '6': args.s6}
    move(LIFT, ms=1500, label='LIFT')
    snap(f'r{rep:02d}_f_lift.jpg', 'LIFT')

    # Pause so you can see if lighter was grabbed
    time.sleep(0.5)

    # ROTATE TO PLACE
    print()
    print(f'  7) ROTATING to {args.place}...')
    TRANSIT = {**HOME, '1': 700, '6': pj.get('6', 875)}
    move(TRANSIT, ms=1800, label='ROTATE')

    # PLACE DOWN
    print('  8) PLACING DOWN...')
    PLACE_DOWN = {**pj, '1': 700, '2': 500}
    move(PLACE_DOWN, ms=1000, label='PLACE')
    snap(f'r{rep:02d}_g_place.jpg', 'PLACE')

    # RELEASE
    print('  9) RELEASING...')
    RELEASE = {**pj, '1': 100, '2': 500}
    move(RELEASE, ms=500, label='RELEASE', extra=0.5)
    snap(f'r{rep:02d}_h_release.jpg', 'RELEASE')

    # HOME
    print('  10) HOME...')
    move(HOME, ms=1800, label='HOME')
    snap(f'r{rep:02d}_i_done.jpg', 'DONE')

    print()
    print('  REP DONE. Did the lighter move?')
    print('  If YES: it worked!')
    print('  If NO (lighter still in same spot):')
    print('    - The lighter was NOT directly under the gripper tip')
    print('    - Try again and look at where the arm hovers')
    print()

    if args.reps > 1 and rep < args.reps:
        print(f'  Put lighter back. Next rep in 5 seconds...')
        time.sleep(5)

print(SEP)
print('  DONE')
print(SEP)
print()
print('  Tune S6 to aim the arm at your lighter:')
print('    S6=500 -> center (straight forward)')
print('    S6=450 -> slight right (~13deg)')  
print('    S6=400 -> right (~24deg)')
print('    S6=550 -> slight left (~13deg)')
print('    S6=600 -> left (~24deg)')
print()
print('  Tune z height:')
print('    python guided_pick.py --z 3.0   (higher, less deep)')
print('    python guided_pick.py --z 2.0   (lower, more reach)')
print()
print(f'  Example: python guided_pick.py --s6 500 --z 2.5')
