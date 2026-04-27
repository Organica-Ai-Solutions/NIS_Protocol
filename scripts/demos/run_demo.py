"""
run_demo.py -- Single-command NIS Protocol pick-and-place runner
===============================================================
Reads calibrated positions from data/calib_results.json (written by
full_calibration.py) and runs the complete pick-and-place sequence.

Usage:
  python run_demo.py               # pick green lighter -> place left
  python run_demo.py green         # same as above
  python run_demo.py blue          # pick blue lighter  -> place right
  python run_demo.py both          # green first, then blue (full demo)
  python run_demo.py --dry-run     # print poses only, no arm movement

Flags:
  --dry-run    simulate without moving
  --no-snap    skip camera snapshots (faster)
  --reps N     repeat N times (default 1)
  --delay S    seconds between reps (default 3)
"""

import sys, json, time, base64, os
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
    import urllib.request, json as _json
    class _R:
        def __init__(self, d): self._d = d
        def json(self): return _json.loads(self._d)
        status_code = 200
    def _get(url, timeout=10, **_):
        return _R(urllib.request.urlopen(url, timeout=timeout).read())
    def _post(url, json=None, timeout=15, **_):
        data = __import__('json').dumps(json or {}).encode()
        req  = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        return _R(urllib.request.urlopen(req, timeout=timeout).read())

# -- Config --------------------------------------------------------------------
PI       = 'http://192.168.1.163:8085'
DRY      = '--dry-run' in sys.argv
NO_SNAP  = '--no-snap' in sys.argv
REPS     = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 1
DELAY    = float(sys.argv[sys.argv.index('--delay') + 1]) if '--delay' in sys.argv else 3.0

args     = [a for a in sys.argv[1:] if not a.startswith('-')]
MODE     = args[0].lower() if args else 'green'   # green | blue | both

OUT = Path('data/run_demo')
OUT.mkdir(parents=True, exist_ok=True)

# -- Load calibration ---------------------------------------------------------
_cal = {}
_cal_file = Path('data/calib_results.json')
if _cal_file.exists():
    try:
        _cal = json.loads(_cal_file.read_text())
        print(f'  Loaded calibration from {_cal_file}')
        print(f'    s6_green={_cal.get("s6_green")}  s6_blue={_cal.get("s6_blue")}  s3_pick={_cal.get("s3_pick")}')
    except Exception as e:
        print(f'  Could not load calib: {e}  -- using official defaults')
else:
    print('  No calib file found -- using official .rob defaults.')
    print('  Run python full_calibration.py first for accurate results.')

# -- Positions -----------------------------------------------------------------
S6_CENTER    = OFFICIAL['s6_center']
S6_LEFT_90   = OFFICIAL['s6_left_90']
S6_LEFT_45   = OFFICIAL['s6_left_45']
S6_RIGHT_45  = OFFICIAL['s6_right_45']
S6_RIGHT_90  = OFFICIAL['s6_right_90']

S6_GREEN = _cal.get('s6_green', S6_RIGHT_45)   # 315 default (right 45 deg)
S6_BLUE  = _cal.get('s6_blue',  S6_LEFT_45)    # 685 default (left 45 deg)
# Pick height: use IK-corrected S3/S4/S5 if available (cosmos_height_calibration sets these)
S3_PICK  = _cal.get('s3_pick',  180)
S4_PICK  = _cal.get('s4_pick',  800)
S5_PICK  = _cal.get('s5_pick',  450)

def _p(d): return {str(k): int(v) for k, v in d.items()}

HOME = _p(OFFICIAL['home'])   # {1:100, 2:500, 3:310, 4:870, 5:680, 6:500}

def make_sequence(s6_pick, s6_place, label):
    """Build 9-step sequence for given pick/place S6 values."""
    pick_low_base = dict(OFFICIAL['pick_low'])
    pick_low_base.update({'3': S3_PICK, '4': S4_PICK, '5': S5_PICK})
    return {
        'label':     label,
        's6_pick':   s6_pick,
        's6_place':  s6_place,
        'home':      _p(OFFICIAL['home']),
        'approach':  _p({**OFFICIAL['home'],     '6': s6_pick}),
        'pick_low':  _p({**pick_low_base,         '6': s6_pick}),
        'gripped':   _p({**OFFICIAL['home'],     '1': 500, '6': s6_pick}),
        'place_hi':  _p({**OFFICIAL['home'],     '1': 500, '6': s6_place}),
        'place_lo':  _p({**OFFICIAL['place_low'],'1': 500, '6': s6_place}),
        'release':   _p({**OFFICIAL['place_low'],'1': 100, '6': s6_place}),
    }

SEQUENCES = {
    'green': make_sequence(S6_GREEN, S6_LEFT_90,  'GREEN -> place LEFT 90 deg'),
    'blue':  make_sequence(S6_BLUE,  S6_RIGHT_90, 'BLUE  -> place RIGHT 90 deg'),
}


# -- API helpers ---------------------------------------------------------------

def get(path, timeout=12):
    try:
        r = _get(f'{PI}{path}', timeout=timeout)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        if not DRY:
            print(f'    !! GET {path}: {e}')
        return {}


def post(path, data=None, timeout=18):
    try:
        r = _post(f'{PI}{path}', json=data or {}, timeout=timeout)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        if not DRY:
            print(f'    !! POST {path}: {e}')
        return {}


def move(servos, ms=1200, label=''):
    s = ' '.join(f'S{k}={v}' for k, v in sorted(servos.items()))
    if DRY:
        print(f'    [DRY] {label:<14}  {s}')
        time.sleep(0.05)
        return

    r   = post('/arm/group_move', {'positions': servos, 'duration_ms': ms})
    sim = r.get('simulation', False)
    ok  = r.get('ok', False)
    tag = f'[{label:<12}]' if label else '             '
    print(f'    {tag} {s}  {"!! SIM" if sim else ("OK" if ok else "FAIL FAIL")}')

    if sim:
        print('    !! SIM -- reconnecting...')
        post('/arm/reconnect')
        time.sleep(2.5)
        post('/arm/group_move', {'positions': servos, 'duration_ms': ms})

    time.sleep(ms / 1000.0 + 0.45)


def snap(label):
    if DRY or NO_SNAP:
        if DRY:
            print(f'    [DRY] [CAM]  {label}.jpg')
        return None

    # Try JSON endpoint
    d = get('/camera/snapshot', timeout=25)
    img_b64 = d.get('image_base64') or d.get('image')
    if img_b64:
        p = OUT / f'{label}.jpg'
        p.write_bytes(base64.b64decode(img_b64))
        print(f'    [CAM]  {p.name}')
        return str(p)

    # Fallback raw bytes
    try:
        import urllib.request as _u
        raw = _u.urlopen(f'{PI}/camera/snapshot', timeout=20).read()
        p = OUT / f'{label}.jpg'
        p.write_bytes(raw)
        print(f'    [CAM]  {p.name}')
        return str(p)
    except Exception as e:
        print(f'    [CAM]  failed: {e}')
    return None


def bar(title):
    print()
    print(f'  {"-"*58}')
    print(f'  {title}')
    print(f'  {"-"*58}')


# -- Pick-and-place execution -------------------------------------------------

def run_sequence(seq, rep=1, total=1):
    label   = seq['label']
    s6_pick = seq['s6_pick']
    s6_pl   = seq['s6_place']
    prefix  = f'r{rep:02d}_' if total > 1 else ''

    bar(f'PICK & PLACE  --  {label}  (rep {rep}/{total})')
    print(f'    S6 pick={s6_pick}  place={s6_pl}  S3_pick={S3_PICK}')
    print()

    t0 = time.time()

    move(seq['home'],     ms=1800, label='HOME')
    snap(f'{prefix}01_home')

    move(seq['approach'], ms=1200, label='APPROACH')
    snap(f'{prefix}02_approach')

    move(seq['pick_low'], ms=800,  label='LOWER')
    snap(f'{prefix}03_lower')

    grip = dict(seq['pick_low']); grip['1'] = 500
    move(grip,            ms=500,  label='GRIP')
    snap(f'{prefix}04_gripped')

    move(seq['gripped'],  ms=1000, label='LIFT')
    snap(f'{prefix}05_lifted')

    move(seq['place_hi'], ms=1500, label='ROTATE')
    snap(f'{prefix}06_rotated')

    move(seq['place_lo'], ms=1000, label='PLACE_LOW')
    snap(f'{prefix}07_place_low')

    move(seq['release'],  ms=500,  label='RELEASE')
    snap(f'{prefix}08_released')

    move(seq['home'],     ms=1500, label='HOME')
    snap(f'{prefix}09_done')

    elapsed = time.time() - t0
    print()
    print(f'  OK  Sequence complete in {elapsed:.1f}s')


# -- Entry point ---------------------------------------------------------------

def main():
    print()
    print('=' * 64)
    print('  NIS PROTOCOL  x  xArm AI  --  RUN DEMO')
    print('=' * 64)
    print(f'  Mode    : {MODE}')
    print(f'  Reps    : {REPS}')
    print(f'  S6 green: {S6_GREEN}  ({"right" if S6_GREEN < 500 else "left"} of center)')
    print(f'  S6 blue : {S6_BLUE}   ({"right" if S6_BLUE  < 500 else "left"} of center)')
    print(f'  S3 pick : {S3_PICK}   (official = 180)')
    print(f'  DRY_RUN : {DRY}')
    print()

    if not DRY:
        h = get('/health')
        if not h:
            print('  ERROR: Pi unreachable at', PI)
            sys.exit(1)
        print(f'  Agent: {h.get("service")} v{h.get("version")}')
        sim = h.get('xarm_simulation', True)
        if sim:
            print('  Arm in simulation -- reconnecting...')
            post('/arm/reconnect')
            time.sleep(2.5)

    modes = ['green', 'blue'] if MODE == 'both' else [MODE]

    for rep in range(1, REPS + 1):
        for m in modes:
            if m not in SEQUENCES:
                print(f'  Unknown mode: {m}  (use: green / blue / both)')
                sys.exit(1)
            run_sequence(SEQUENCES[m], rep=rep, total=REPS)
            if m != modes[-1] or rep != REPS:
                print(f'  Waiting {DELAY}s before next run...')
                time.sleep(DELAY)

    print()
    print('=' * 64)
    print('  ALL DONE')
    print('=' * 64)
    print()
    frames = sorted(OUT.glob('*.jpg'))
    if frames:
        print(f'  {len(frames)} frames saved to {OUT}/')
        for f in frames:
            print(f'    {f.name}')
    print()


if __name__ == '__main__':
    main()
