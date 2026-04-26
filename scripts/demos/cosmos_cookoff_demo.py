"""
COSMOS COOKOFF DEMO -- NIS Protocol Pick & Place
================================================
NIS Protocol + NeuroLinux Agent + Hiwonder xArm AI

Confirmed working servo positions (IK verified 2026-02-27):
  HOME:     S1=100  S2=500  S3=310  S4=870  S5=680  S6=500
  PICK:     S1=100  S2=500  S3=142  S4=856  S5=430  S6=500 (z=1.5cm alpha=-65)
  GRIP:     S1=700  (CONFIRMED -- S1=500 is too loose, lighter drops!)
  PLACE L:  S6=875  S3=220  S4=827  S5=425

S6 direction (manufacturer confirmed):
  S6=500  center forward
  S6 > 500  ->  arm rotates LEFT
  S6 < 500  ->  arm rotates RIGHT

Usage:
  python cosmos_cookoff_demo.py              # pick GREEN -> place LEFT
  python cosmos_cookoff_demo.py --blue       # pick BLUE  -> place RIGHT
  python cosmos_cookoff_demo.py --s6 380     # custom S6 pick position
  python cosmos_cookoff_demo.py --dry-run    # no arm moves, camera only
"""

import sys, time, json, base64, os
from pathlib import Path

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'rob_parser',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'src', 'neurolinux', 'drivers', 'rob_parser.py')
)
_rob = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_rob)
OFFICIAL = _rob.OFFICIAL

try:
    import requests
    def _get(url, **kw):   return requests.get(url, **kw)
    def _post(url, **kw):  return requests.post(url, **kw)
except ImportError:
    import urllib.request
    class _Resp:
        def __init__(self, data): self._d = data
        def json(self): return json.loads(self._d)
        @property
        def status_code(self): return 200
    def _get(url, timeout=10, **_):
        return _Resp(urllib.request.urlopen(url, timeout=timeout).read())
    def _post(url, json=None, timeout=15, **_):
        data = __import__('json').dumps(json or {}).encode()
        req  = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        return _Resp(urllib.request.urlopen(req, timeout=timeout).read())

PI      = 'http://192.168.1.163:8085'
OUT     = Path('data/cookoff_demo')
DRY_RUN = '--dry-run' in sys.argv
BLUE    = '--blue' in sys.argv

# -- Override S6 from command line ---------------------------------------------
_S6_OVERRIDE = None
if '--s6' in sys.argv:
    idx = sys.argv.index('--s6')
    try:
        _S6_OVERRIDE = int(sys.argv[idx + 1])
    except (IndexError, ValueError):
        pass

OUT.mkdir(parents=True, exist_ok=True)

# -- Load calibration from file if available -----------------------------------
_calib_file = Path('data/calib_results.json')
_calib = {}
if _calib_file.exists():
    try:
        _calib = json.loads(_calib_file.read_text())
    except Exception:
        pass

# -- Servo positions -----------------------------------------------------------
# Official baseline from .rob files
_S6_CENTER     = OFFICIAL['s6_center']    # 500
_S6_LEFT_90    = OFFICIAL['s6_left_90']   # 875
_S6_LEFT_45    = OFFICIAL['s6_left_45']   # 685
_S6_RIGHT_45   = OFFICIAL['s6_right_45']  # 315
_S6_RIGHT_90   = OFFICIAL['s6_right_90']  # 125

# Use calibration file values if available, otherwise .rob defaults
if BLUE:
    S6_PICK  = _S6_OVERRIDE or _calib.get('s6_blue',  _S6_LEFT_45)
    S6_PLACE = _S6_RIGHT_90
    TARGET   = 'BLUE (left side -> place right)'
else:
    S6_PICK  = _S6_OVERRIDE or _calib.get('s6_green', _S6_RIGHT_45)
    S6_PLACE = _S6_LEFT_90
    TARGET   = 'GREEN (right side -> place left)'

# IK-corrected pick height: uses S4+S5 (not just S3) for height control
# If cosmos_height_calibration.py ran, s4_pick/s5_pick are in calib_results.json
# Defaults are now the CONFIRMED values (IK verified 2026-02-27).
# alpha=-65 is stable; alpha=-71 / z=1.2cm caused arm collapse.
S3_PICK = _calib.get('s3_pick', 142)  # confirmed z=1.5cm alpha=-65
S4_PICK = _calib.get('s4_pick', 856)
S5_PICK = _calib.get('s5_pick', 430)
Z_PICK  = _calib.get('z_pick',  1.5)

print(f'  Pick: z={Z_PICK:.2f}cm  S3={S3_PICK}  S4={S4_PICK}  S5={S5_PICK}  (alpha=-65)')

# Build pose dicts (string keys for Pi agent)
def _p(d): return {str(k): int(v) for k, v in d.items()}

HOME     = _p(OFFICIAL['home'])
APPROACH = _p({**OFFICIAL['home'],     '6': S6_PICK})
# Use IK-corrected S3/S4/S5 for pick height (height = S5+S4, not just S3)
PICK_LOW = _p({'1': 100, '2': 500, '3': S3_PICK, '4': S4_PICK, '5': S5_PICK, '6': S6_PICK})
HOVER    = _p({'1': 100, '2': 500, '3': 222,      '4': 697,      '5': 604,      '6': S6_PICK})  # z=6cm
MID      = _p({'1': 100, '2': 500, '3': 158,      '4': 798,      '5': 502,      '6': S6_PICK})  # z=3.5cm
GRIPPED  = _p({'1': 700, '2': 500, '3': S3_PICK, '4': S4_PICK, '5': S5_PICK, '6': S6_PICK})  # S1=700 CONFIRMED firm grip
PLACE_HI = _p({**OFFICIAL['home'],     '1': 700, '6': S6_PLACE})
# Place joints: 1cm lower than official (IK delta: S4+27, S5-35)
_place_joints_corrected = _calib.get('place_joints', {})
if S6_PLACE == 875 and 'left90' in _place_joints_corrected:
    _pj = {str(k): int(v) for k,v in _place_joints_corrected['left90'].items()}
elif S6_PLACE == 125 and 'right90' in _place_joints_corrected:
    _pj = {str(k): int(v) for k,v in _place_joints_corrected['right90'].items()}
elif S6_PLACE == 685 and 'left45' in _place_joints_corrected:
    _pj = {str(k): int(v) for k,v in _place_joints_corrected['left45'].items()}
elif S6_PLACE == 315 and 'right45' in _place_joints_corrected:
    _pj = {str(k): int(v) for k,v in _place_joints_corrected['right45'].items()}
else:
    # Fallback: apply delta to official (S4+27, S5-35)
    _pj = {**OFFICIAL['place_low'], '4': OFFICIAL['place_low']['4']+27,
           '5': OFFICIAL['place_low']['5']-35}
    _pj = {str(k): int(v) for k,v in _pj.items()}
_pj['6'] = str(S6_PLACE)

PLACE_LO = _p({**_pj, '1': 700})  # hold grip until placed
RELEASE  = _p({**_pj, '1': 100})


# -- Helpers -------------------------------------------------------------------

def get(path, timeout=12):
    try:
        r = _get(f'{PI}{path}', timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f'    GET {path} failed: {e}')
    return {}


def post(path, data=None, timeout=18):
    try:
        r = _post(f'{PI}{path}', json=data or {}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f'    POST {path} failed: {e}')
    return {}


def move(servos, ms=1200, label=''):
    if DRY_RUN:
        s = ' '.join(f'S{k}={v}' for k, v in sorted(servos.items()))
        print(f'    [DRY] {label}  {s}')
        time.sleep(0.2)
        return

    r = post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim = r.get('simulation', False)
    ok  = r.get('ok', False)
    s   = ' '.join(f'S{k}={v}' for k, v in sorted(servos.items()))
    tag = f'[{label}] ' if label else ''
    print(f'    {tag}{s}  [{"SIM" if sim else ("OK" if ok else "FAIL")}]')

    if sim:
        print('    !! SIM -- reconnecting...')
        post('/arm/reconnect')
        time.sleep(2.5)
        post('/arm/group_move', {'positions': servos, 'duration_ms': ms})

    time.sleep(ms / 1000.0 + 0.5)


def snap(label):
    if DRY_RUN:
        print(f'    [DRY] [CAM]  {label}.jpg')
        return None

    d = get('/camera/snapshot', timeout=25)
    img_b64 = d.get('image_base64') or d.get('image')
    if img_b64:
        p = OUT / f'{label}.jpg'
        p.write_bytes(base64.b64decode(img_b64))
        print(f'    [CAM]  {p.name}  ({p.stat().st_size:,} bytes)')
        return str(p)

    # Fallback: raw bytes endpoint
    try:
        import urllib.request as ur
        raw = ur.urlopen(f'{PI}/camera/snapshot', timeout=20).read()
        p = OUT / f'{label}.jpg'
        p.write_bytes(raw)
        print(f'    [CAM]  {p.name}  ({p.stat().st_size:,} bytes)')
        return str(p)
    except Exception as e:
        print(f'    [CAM]  snapshot failed: {e}')
    return None


def step(n, msg):
    print(f'\n  [{n:02d}] {msg}')


# -- Main sequence -------------------------------------------------------------

def run():
    print()
    print('=' * 64)
    print('  NIS PROTOCOL  x  COSMOS COOKOFF DEMO')
    print('  Hiwonder xArm AI  --  Pick & Place Pipeline')
    print('=' * 64)
    print(f'  Target : {TARGET}')
    print(f'  S6_PICK: {S6_PICK}  |  S6_PLACE: {S6_PLACE}')
    print(f'  S3_PICK: {S3_PICK}  (official = 180)')
    print(f'  DRY_RUN: {DRY_RUN}')
    print()

    # Pre-flight
    h = get('/health')
    if not h and not DRY_RUN:
        print('  ERROR: Pi unreachable at', PI)
        sys.exit(1)

    if h:
        print(f'  Agent : {h.get("service")} v{h.get("version")}  [{h.get("status")}]')
        print(f'  xArm  : connected={h.get("xarm")}  sim={h.get("xarm_simulation")}')
        print(f'  Camera: {h.get("camera")}')

    if h.get('xarm_simulation') and not DRY_RUN:
        print('\n  Arm in simulation -- reconnecting...')
        post('/arm/reconnect')
        time.sleep(2.5)

    print()

    # -- 1. HOME ----------------------------------------------------------------
    step(1, 'HOME  (official .rob baseline)')
    move(HOME, ms=1800, label='HOME')
    snap('01_home')

    # -- 2. APPROACH ------------------------------------------------------------
    step(2, f'APPROACH  (S6={S6_PICK}, rotate to {"right" if S6_PICK < 500 else "left"})')
    move(APPROACH, ms=1200, label='APPROACH')
    snap('02_approach')

    # -- 3. HOVER (z=6cm) -------------------------------------------------------
    step(3, f'HOVER  (z=6cm, S6={S6_PICK})')
    move(HOVER, ms=900, label='HOVER')
    snap('03_hover')

    # -- 4. MID (z=3.5cm) -------------------------------------------------------
    step(4, f'MID  (z=3.5cm descent)')
    move(MID, ms=700, label='MID')
    snap('04_mid')

    # -- 5. PICK (z=1.5cm) -------------------------------------------------------
    step(5, f'PICK  (z=1.5cm, S3={S3_PICK}, S4={S4_PICK}, S5={S5_PICK} -- IK CONFIRMED)')
    move(PICK_LOW, ms=600, label='PICK')
    snap('05_pick')

    # -- 6. GRIP ----------------------------------------------------------------
    step(6, 'GRIP  (S1: 100 -> 700  -- CONFIRMED firm, S1=500 drops lighter!)')
    grip = dict(PICK_LOW)
    grip['1'] = '700'  # S1=700 is the confirmed firm grip
    move(grip, ms=500, label='GRIP')
    snap('06_gripped')

    # -- 7. LIFT ----------------------------------------------------------------
    step(7, 'LIFT  (raise to home height, keep grip S1=700)')
    move(GRIPPED, ms=1000, label='LIFT')
    snap('07_lifted')

    # -- 8. ROTATE TO PLACE -----------------------------------------------------
    step(8, f'ROTATE  (S6: {S6_PICK} -> {S6_PLACE}, {"left" if S6_PLACE > 500 else "right"} 90 deg)')
    move(PLACE_HI, ms=1500, label='ROTATE')
    snap('08_rotated')

    # -- 9. LOWER TO PLACE ------------------------------------------------------
    step(9, 'LOWER TO PLACE  (S3=220, S4=827, S5=425)')
    move(PLACE_LO, ms=1000, label='PLACE_LOW')
    snap('09_place_low')

    # -- 10. RELEASE -------------------------------------------------------------
    step(10, 'RELEASE  (S1: 700 -> 100, open gripper)')
    move(RELEASE, ms=500, label='RELEASE')
    snap('10_released')

    # -- 11. RETURN HOME ---------------------------------------------------------
    step(11, 'RETURN HOME')
    move(HOME, ms=1500, label='HOME')
    snap('11_done')

    # -- Summary ----------------------------------------------------------------
    print()
    print('=' * 64)
    print('  PICK & PLACE COMPLETE')
    print('=' * 64)
    print()
    print('  Sequence:')
    print(f'    1. HOME          {" ".join(f"S{k}={v}" for k,v in sorted(HOME.items()))}')
    print(f'    2. APPROACH      S6={S6_PICK}  ({"RIGHT" if S6_PICK < 500 else "LEFT"} of center)')
    print(f'    3. HOVER         z=6cm  S3=222, S4=697, S5=604')
    print(f'    4. MID           z=3.5cm  S3=158, S4=798, S5=502')
    print(f'    5. PICK LOW      z=1.5cm  S3={S3_PICK}, S4={S4_PICK}, S5={S5_PICK}')
    print(f'    6. GRIP          S1=700 (CONFIRMED firm — S1=500 drops lighter!)')
    print(f'    7. LIFT          S3=310, S4=870, S5=680 (home height, grip=700)')
    print(f'    8. ROTATE        S6={S6_PLACE}  ({"LEFT" if S6_PLACE > 500 else "RIGHT"} 90 deg)')
    print(f'    9. PLACE LOW     S3=220, S4=827, S5=425')
    print(f'   10. RELEASE       S1=100 (open)')
    print(f'   11. HOME          ')
    print()
    print(f'  Frames: {OUT}/')
    for f in sorted(OUT.glob('*.jpg')):
        print(f'    {f.name}  ({f.stat().st_size:,} bytes)')
    print()


if __name__ == '__main__':
    run()
