"""
ARM DANCE  - Real-time Latino rhythm arm choreography
=====================================================
Polls the Pi mic in real-time, detects BPM and genre, drives the arm.

HOW IT WORKS:
  1. Every beat cycle: asks Pi to capture audio from its C270 mic
  2. Pi runs onset detection + inter-onset interval BPM estimation
  3. BPM mapped to genre: reggaeton / cumbia / bachata / salsa
  4. Genre + energy selects the right choreography sequence
  5. Arm executes the move — synced to the actual detected BPM

USAGE:
  python arm_dance.py                        # live mic mode (Pi captures music)
  python arm_dance.py --demo reggaeton       # fixed demo, no mic needed
  python arm_dance.py --demo salsa           # 170 BPM salsa demo
  python arm_dance.py --demo cumbia          # 130 BPM cumbia demo
  python arm_dance.py --demo bachata         # 125 BPM bachata demo
  python arm_dance.py --moves 48            # how many beats (default 32)
  python arm_dance.py --energy 0.25         # force energy level (0.0-0.5)

GENRES + BPM ranges:
  reggaeton  70-100 BPM  -> prreo pump, dembow, fist
  cumbia    105-145 BPM  -> flowing side steps, dip, groove
  bachata   113-138 BPM  -> smooth romantic sway, hip, spin
  salsa     145+ BPM     -> fast sharp salsa steps, snaps, spins
"""

import json, math, time, sys, argparse, collections, struct
import urllib.request as _ur

parser = argparse.ArgumentParser()
parser.add_argument('--demo',   default='', choices=['','reggaeton','cumbia','bachata','salsa'],
                    help='Fixed demo genre (no mic needed)')
parser.add_argument('--moves',  type=int,   default=32, help='Number of dance moves')
parser.add_argument('--energy', type=float, default=None,
                    help='Override energy 0.0-0.5 (default: from mic or demo)')
parser.add_argument('--pi',     default='http://192.168.1.163:8085', help='Pi agent URL')
parser.add_argument('--nis',    default='http://192.168.1.163:8000', help='Pi NIS server URL')
args = parser.parse_args()

PI  = args.pi
NIS = args.nis
SEP = '=' * 64

# ============================================================================
# DEMO PRESETS (fixed BPM + energy when --demo is used)
# ============================================================================
DEMO_PRESETS = {
    'reggaeton': {'bpm': 88.0,  'energy': 0.20},
    'cumbia':    {'bpm': 130.0, 'energy': 0.15},
    'bachata':   {'bpm': 125.0, 'energy': 0.14},
    'salsa':     {'bpm': 170.0, 'energy': 0.22},
}

# ============================================================================
# DANCE MOVES — confirmed IK servo positions (verified 2026-02-27)
# ============================================================================
# S6: 500=center, 875=L90, 125=R90
# S1: 100=open, 700=grip, 900=fist
MOVES = {
    'home':            {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500, '_ms':900},
    'sway_left':       {'1':100,'2':500,'3':310,'4':870,'5':680,'6':640, '_ms':450},
    'sway_right':      {'1':100,'2':500,'3':310,'4':870,'5':680,'6':360, '_ms':450},
    'slow_lean_left':  {'1':100,'2':500,'3':290,'4':840,'5':660,'6':680, '_ms':700},
    'slow_lean_right': {'1':100,'2':500,'3':290,'4':840,'5':660,'6':320, '_ms':700},
    # Cumbia
    'cumbia_L':        {'1':100,'2':500,'3':270,'4':800,'5':600,'6':720, '_ms':380},
    'cumbia_R':        {'1':100,'2':500,'3':270,'4':800,'5':600,'6':280, '_ms':380},
    'cumbia_dip':      {'1':100,'2':500,'3':235,'4':720,'5':560,'6':500, '_ms':500},
    # Salsa
    'salsa_L':         {'1':100,'2':500,'3':300,'4':855,'5':670,'6':660, '_ms':220},
    'salsa_R':         {'1':100,'2':500,'3':300,'4':855,'5':670,'6':340, '_ms':220},
    'salsa_up':        {'1':100,'2':500,'3':310,'4':800,'5':630,'6':500, '_ms':200},
    'salsa_snap':      {'1':100,'2':500,'3':310,'4':870,'5':560,'6':500, '_ms':180},
    # Reggaeton
    'prreo_pump':      {'1':900,'2':500,'3':290,'4':820,'5':640,'6':500, '_ms':280},
    'prreo_drop':      {'1':900,'2':500,'3':250,'4':760,'5':570,'6':500, '_ms':350},
    'prreo_up':        {'1':100,'2':500,'3':300,'4':840,'5':660,'6':500, '_ms':280},
    'dembow_L':        {'1':900,'2':500,'3':285,'4':800,'5':610,'6':680, '_ms':240},
    'dembow_R':        {'1':900,'2':500,'3':285,'4':800,'5':610,'6':320, '_ms':240},
    # Bachata
    'bachata_side_L':  {'1':100,'2':500,'3':270,'4':790,'5':580,'6':740, '_ms':500},
    'bachata_side_R':  {'1':100,'2':500,'3':270,'4':790,'5':580,'6':260, '_ms':500},
    'bachata_hip':     {'1':100,'2':500,'3':240,'4':730,'5':540,'6':500, '_ms':600},
    # Universal high energy
    'spin_L':          {'1':100,'2':500,'3':300,'4':845,'5':655,'6':820, '_ms':280},
    'spin_R':          {'1':100,'2':500,'3':300,'4':845,'5':655,'6':180, '_ms':280},
    'reach_high':      {'1':100,'2':500,'3':315,'4':790,'5':620,'6':500, '_ms':350},
    'groove_wide_L':   {'1':100,'2':500,'3':255,'4':760,'5':530,'6':820, '_ms':400},
    'groove_wide_R':   {'1':100,'2':500,'3':255,'4':760,'5':530,'6':180, '_ms':400},
    'fist_L':          {'1':900,'2':500,'3':295,'4':830,'5':640,'6':700, '_ms':200},
    'fist_R':          {'1':900,'2':500,'3':295,'4':830,'5':640,'6':300, '_ms':200},
    'snap_open':       {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500, '_ms':180},
}

# ============================================================================
# CHOREOGRAPHY SEQUENCES per genre+energy
# ============================================================================
CHOREO = {
    'silence':       ['home'],
    'soft':          ['sway_left','sway_right','slow_lean_left','slow_lean_right'],
    'reggaeton_mid': ['prreo_pump','prreo_drop','prreo_up','dembow_L','prreo_up','dembow_R'],
    'reggaeton_high':['dembow_L','fist_L','prreo_drop','dembow_R','fist_R','prreo_up',
                      'spin_L','prreo_pump','spin_R','snap_open'],
    'cumbia_mid':    ['cumbia_L','cumbia_R','cumbia_dip','cumbia_L','cumbia_R','reach_high'],
    'cumbia_high':   ['cumbia_L','fist_L','cumbia_dip','cumbia_R','fist_R','spin_L',
                      'groove_wide_L','cumbia_dip','groove_wide_R','spin_R'],
    'bachata_mid':   ['bachata_side_L','bachata_hip','bachata_side_R','bachata_hip'],
    'bachata_high':  ['bachata_side_L','fist_L','bachata_hip','bachata_side_R',
                      'fist_R','spin_L','bachata_hip','spin_R'],
    'salsa_mid':     ['salsa_L','salsa_R','salsa_up','salsa_snap','salsa_L','salsa_R'],
    'salsa_high':    ['salsa_L','salsa_snap','spin_L','salsa_R','salsa_snap','spin_R',
                      'fist_L','salsa_up','fist_R','snap_open'],
}

def classify(bpm: float, energy: float) -> str:
    if energy < 0.015: return 'silence'
    if energy < 0.04:  return 'soft'
    hi = energy >= 0.12
    if bpm < 105:
        return 'reggaeton_high' if hi else 'reggaeton_mid'
    elif bpm < 145:
        if 113 <= bpm <= 138:
            return 'bachata_high' if hi else 'bachata_mid'
        return 'cumbia_high' if hi else 'cumbia_mid'
    return 'salsa_high' if hi else 'salsa_mid'

# ============================================================================
# ARM CONTROL
# ============================================================================
def do_move(name: str) -> bool:
    if name not in MOVES:
        return False
    sv = dict(MOVES[name])
    ms = sv.pop('_ms', 500)
    try:
        d = json.dumps({'positions': sv, 'duration_ms': ms}).encode()
        r = _ur.Request(PI+'/arm/group_move', data=d,
                        headers={'Content-Type':'application/json'})
        _ur.urlopen(r, timeout=5)
        return True
    except Exception as e:
        print(f'  [ARM] {name} failed: {e}')
        return False

def get_beat_from_pi() -> dict:
    """Ask Pi to capture audio + return beat analysis."""
    try:
        r = _ur.urlopen(NIS+'/cosmos-dance/mic?duration=0.35', timeout=6)
        return json.loads(r.read())
    except Exception as e:
        return {'ok': False, 'bpm': 100.0, 'energy': 0.0,
                'genre': 'silence', 'vibe': f'mic error: {e}'}

# ============================================================================
# BPM SMOOTHER
# ============================================================================
_bpm_hist = collections.deque(maxlen=10)

def smooth_bpm(raw: float) -> float:
    _bpm_hist.append(raw)
    vals = sorted(_bpm_hist)
    med  = vals[len(vals)//2]
    # Correct tempo octave errors
    if raw > med * 1.7:   raw /= 2.0
    elif raw < med * 0.6: raw *= 2.0
    return max(70.0, min(220.0, raw))

# ============================================================================
# LIVE DISPLAY
# ============================================================================
BAR_WIDTH = 20

def bpm_bar(bpm: float) -> str:
    """ASCII visual of BPM in Latino range 70-220."""
    pct = (bpm - 70) / 150.0
    filled = int(pct * BAR_WIDTH)
    return '|' + '#' * filled + '-' * (BAR_WIDTH - filled) + '|'

def energy_bar(e: float) -> str:
    filled = min(BAR_WIDTH, int(e * BAR_WIDTH / 0.35))
    return '|' + '#' * filled + ' ' * (BAR_WIDTH - filled) + '|'

GENRE_EMOJI = {
    'silence': '  ....',
    'soft':    '  SUAVE',
    'reggaeton_mid':  '  REGGAETON',
    'reggaeton_high': '  PERREO INTENSO',
    'cumbia_mid':     '  CUMBIA',
    'cumbia_high':    '  CUMBIA BRAVA',
    'bachata_mid':    '  BACHATA',
    'bachata_high':   '  BACHATA FUEGO',
    'salsa_mid':      '  SALSA',
    'salsa_high':     '  SALSA BRAVA',
}

# ============================================================================
# MAIN
# ============================================================================
print(SEP)
print('  ARM DANCE  - La Maquina Tiene Flow')
print(SEP)

# Health check
try:
    h = json.loads(_ur.urlopen(PI+'/health', timeout=6).read())
    print(f'  Pi agent v{h.get("version")}  xarm={h.get("xarm")}  cam={h.get("camera")}')
except Exception as e:
    print(f'  ERROR: Pi unreachable — {e}'); sys.exit(1)

if not h.get('xarm'):
    print('  ERROR: xArm not connected'); sys.exit(1)

# Demo mode or mic mode
demo_genre = args.demo
if demo_genre:
    preset = DEMO_PRESETS[demo_genre]
    fixed_bpm    = preset['bpm']
    fixed_energy = args.energy if args.energy is not None else preset['energy']
    print(f'  MODE: DEMO  genre={demo_genre}  bpm={fixed_bpm:.0f}  energy={fixed_energy:.2f}')
else:
    print(f'  MODE: LIVE MIC  (Pi C270 mic at {NIS})')
    print('  Play music near the Pi mic or your speakers!')
    print()
    # Test mic
    print('  Testing Pi mic...', end=' ', flush=True)
    tb = get_beat_from_pi()
    if tb.get('ok'):
        print(f'OK  bpm={tb["bpm"]:.0f}  energy={tb["energy"]:.3f}')
    else:
        print(f'WARNING: {tb.get("error","unknown")}')
        print('  Will attempt to continue (mic may work once music plays)')

print()
print('  Choreography:')
for g, seq in CHOREO.items():
    print(f'    {GENRE_EMOJI.get(g, g):25s}: {", ".join(seq[:3])}{"..." if len(seq)>3 else ""}')
print()
input('  Press ENTER to start  (Ctrl+C to stop anytime)...')
print()

# HOME
print('  -> HOME')
do_move('home')
time.sleep(1.0)

choreo_step = 0
last_move   = 'home'
successes   = 0
genre_counts: dict = {}

print(f'  {"Step":>4}  {"BPM":>6}  {"Genre":20}  {"Move":18}  Energy  Latency')
print('  ' + '-'*62)

try:
    for step in range(args.moves):
        t0 = time.time()

        # ── Get beat ────────────────────────────────────────────────────────
        if demo_genre:
            bpm    = smooth_bpm(fixed_bpm)
            energy = args.energy if args.energy is not None else fixed_energy
        else:
            beat   = get_beat_from_pi()
            raw_bpm = beat.get('bpm', 100.0)
            bpm    = smooth_bpm(raw_bpm)
            energy = beat.get('energy', 0.0)
            if args.energy is not None:
                energy = args.energy

        # ── Classify genre + pick move ───────────────────────────────────────
        genre  = classify(bpm, energy)
        seq    = CHOREO.get(genre, ['home'])
        move   = seq[choreo_step % len(seq)]
        if move == last_move and len(seq) > 1:
            move = seq[(choreo_step + 1) % len(seq)]
        choreo_step += 1

        # ── Execute ──────────────────────────────────────────────────────────
        ok      = do_move(move)
        elapsed = time.time() - t0
        if ok: successes += 1
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # ── Display ──────────────────────────────────────────────────────────
        label   = GENRE_EMOJI.get(genre, genre)
        e_bar   = energy_bar(energy)
        print(f'  {step+1:4d}  {bpm:6.1f}  {label:20s}  {move:18s}  '
              f'{e_bar}  {elapsed*1000:.0f}ms')

        last_move = move

        # ── Sync to beat ─────────────────────────────────────────────────────
        beat_sec  = 60.0 / max(60.0, bpm)
        move_sec  = MOVES.get(move, {}).get('_ms', 500) / 1000.0
        remaining = max(0.05, beat_sec - elapsed)
        time.sleep(remaining)

except KeyboardInterrupt:
    print()
    print('  Stopped!')

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(SEP)
print(f'  BAILE TERMINADO  —  {successes}/{args.moves} moves')
print(SEP)
if genre_counts:
    top = sorted(genre_counts.items(), key=lambda x: -x[1])
    print()
    print('  Genre breakdown:')
    for g, c in top:
        bar = '#' * c
        print(f'    {GENRE_EMOJI.get(g,g):25s}: {bar}  ({c})')

print()
print('  -> HOME')
do_move('home')
time.sleep(1.5)
print()
print('  Tips:')
print('    Live mic: python arm_dance.py --moves 64')
print('    Reggaeton: python arm_dance.py --demo reggaeton --energy 0.28')
print('    Salsa:     python arm_dance.py --demo salsa --energy 0.25')
print('    Cumbia:    python arm_dance.py --demo cumbia')
print('    Bachata:   python arm_dance.py --demo bachata --energy 0.12')
