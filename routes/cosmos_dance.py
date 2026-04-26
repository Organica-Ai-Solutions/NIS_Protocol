"""
COSMOS DANCE — NIS Protocol Real-Time AI Choreographer
======================================================
Architecture:
  1. Pi mic (C270 webcam at hw:1,0) captures audio chunks
  2. Onset detection + inter-onset interval (IOI) → real BPM
  3. Energy envelope → vibe level
  4. Cosmos Reason2 on H100 decides EACH move based on live audio + history
     (no preprogrammed sequences — pure AI reaction to music)
  5. xArm executes via /arm/group_move (confirmed servo positions)
  6. Fallback to lookup table when H100 is offline

Endpoints:
  GET  /cosmos-dance/mic          capture 1 audio chunk, return {bpm, energy, vibe}
  POST /cosmos-dance/start        start AI dance loop (Pi mic → Reason2 → arm)
  POST /cosmos-dance/stop         stop dance loop
  POST /cosmos-dance/demo         fixed-beat demo (AI picks moves)
  GET  /cosmos-dance/status       current state + move history
  POST /cosmos-dance/music-react  browser sends PCM chunk, Pi arm reacts (1 move)
"""

import asyncio
import collections
import json
import logging
import math
import os
import struct
import subprocess
import time
from typing import Optional, List

from fastapi import APIRouter

logger = logging.getLogger("nis.cosmos_dance")

router = APIRouter(prefix="/cosmos-dance", tags=["Cosmos Dance"])

# ── Config ────────────────────────────────────────────────────────────────────
AGENT_URL    = "http://localhost:8085"
REASON2_URL  = os.getenv("H100_REASON_URL", "http://localhost:8100")  # Cosmos Reason2 (tunnel from H100)

MIC_DEVICE   = "hw:1,0"    # C270 webcam mic on Pi (arecord device)
SAMPLE_RATE  = 22050        # higher rate for better onset detection
CHUNK_SECS   = 0.35         # audio per analysis cycle (~2 per beat at 100BPM)
DEFAULT_BPM  = 100
MAX_MOVES    = 64

# ── BPM history smoother ──────────────────────────────────────────────────────
_bpm_history: collections.deque = collections.deque(maxlen=12)
_onset_times: collections.deque = collections.deque(maxlen=24)  # timestamps of detected onsets

def _smooth_bpm(raw_bpm: float) -> float:
    """Median smooth BPM over history, handle half/double tempo."""
    _bpm_history.append(raw_bpm)
    if len(_bpm_history) < 3:
        return raw_bpm
    vals = sorted(_bpm_history)
    med  = vals[len(vals) // 2]
    # Collapse if raw is ~2x or ~0.5x the median (tempo octave error)
    if raw_bpm > med * 1.7:
        raw_bpm /= 2.0
    elif raw_bpm < med * 0.6:
        raw_bpm *= 2.0
    # Clamp to Latin music range 70–220 BPM
    return max(70.0, min(220.0, raw_bpm))

# ── Confirmed servo positions (IK verified 2026-02-27) ────────────────────────
# All moves use /arm/group_move with direct servo positions
# S6 scale: 500=center, 875=left90, 125=right90, 375=right45, 625=left45
# S1: 100=open, 700=grip, 900=closed

# S6 scale: 500=center, 875=left90, 125=right90
# S1: 100=open, 700=firm grip, 900=closed fist
# All _ms values are tuned for Latino rhythm feel
ARM_SERVO_MOVES = {
    # ── Still ──────────────────────────────────────────────────────────────────
    "home":            {"1":100,"2":500,"3":310,"4":870,"5":680,"6":500, "_ms":900},

    # ── Suave / soft (salsa pause, cumbia sway) ────────────────────────────────
    "sway_left":       {"1":100,"2":500,"3":310,"4":870,"5":680,"6":640, "_ms":450},
    "sway_right":      {"1":100,"2":500,"3":310,"4":870,"5":680,"6":360, "_ms":450},
    "slow_lean_left":  {"1":100,"2":500,"3":290,"4":840,"5":660,"6":680, "_ms":700},
    "slow_lean_right": {"1":100,"2":500,"3":290,"4":840,"5":660,"6":320, "_ms":700},

    # ── Cumbia (120-140 BPM, flowing side step) ────────────────────────────────
    "cumbia_L":        {"1":100,"2":500,"3":270,"4":800,"5":600,"6":720, "_ms":380},
    "cumbia_R":        {"1":100,"2":500,"3":270,"4":800,"5":600,"6":280, "_ms":380},
    "cumbia_dip":      {"1":100,"2":500,"3":235,"4":720,"5":560,"6":500, "_ms":500},

    # ── Salsa (160-200 BPM, fast sharp accents) ────────────────────────────────
    "salsa_L":         {"1":100,"2":500,"3":300,"4":855,"5":670,"6":660, "_ms":220},
    "salsa_R":         {"1":100,"2":500,"3":300,"4":855,"5":670,"6":340, "_ms":220},
    "salsa_up":        {"1":100,"2":500,"3":310,"4":800,"5":630,"6":500, "_ms":200},
    "salsa_snap":      {"1":100,"2":500,"3":310,"4":870,"5":560,"6":500, "_ms":180},  # wrist snap

    # ── Reggaeton (80-95 BPM, heavy bounce) ───────────────────────────────────
    "prreo_pump":      {"1":900,"2":500,"3":290,"4":820,"5":640,"6":500, "_ms":280},  # fist pump
    "prreo_drop":      {"1":900,"2":500,"3":250,"4":760,"5":570,"6":500, "_ms":350},  # drop low
    "prreo_up":        {"1":100,"2":500,"3":300,"4":840,"5":660,"6":500, "_ms":280},  # up
    "dembow_L":        {"1":900,"2":500,"3":285,"4":800,"5":610,"6":680, "_ms":240},
    "dembow_R":        {"1":900,"2":500,"3":285,"4":800,"5":610,"6":320, "_ms":240},

    # ── Bachata (120-130 BPM, romantic smooth) ─────────────────────────────────
    "bachata_side_L":  {"1":100,"2":500,"3":270,"4":790,"5":580,"6":740, "_ms":500},
    "bachata_side_R":  {"1":100,"2":500,"3":270,"4":790,"5":580,"6":260, "_ms":500},
    "bachata_hip":     {"1":100,"2":500,"3":240,"4":730,"5":540,"6":500, "_ms":600},

    # ── High energy universales ────────────────────────────────────────────────
    "spin_L":          {"1":100,"2":500,"3":300,"4":845,"5":655,"6":820, "_ms":280},
    "spin_R":          {"1":100,"2":500,"3":300,"4":845,"5":655,"6":180, "_ms":280},
    "reach_high":      {"1":100,"2":500,"3":315,"4":790,"5":620,"6":500, "_ms":350},
    "groove_wide_L":   {"1":100,"2":500,"3":255,"4":760,"5":530,"6":820, "_ms":400},
    "groove_wide_R":   {"1":100,"2":500,"3":255,"4":760,"5":530,"6":180, "_ms":400},

    # ── Gripper accents (on the 1!) ────────────────────────────────────────────
    "fist_L":          {"1":900,"2":500,"3":295,"4":830,"5":640,"6":700, "_ms":200},
    "fist_R":          {"1":900,"2":500,"3":295,"4":830,"5":640,"6":300, "_ms":200},
    "snap_open":       {"1":100,"2":500,"3":310,"4":870,"5":680,"6":500, "_ms":180},
}

ARM_MOVES = {"wave": "/arm/wave", "home": "/arm/home"}

# ── Genre labels (for display + fallback) ─────────────────────────────────────
def _bpm_to_genre_energy(bpm: float, energy: float) -> str:
    """Map BPM + energy to a genre label (used for display and offline fallback)."""
    if energy < 0.015:
        return "silence"
    if energy < 0.04:
        return "soft"
    hi = energy >= 0.12
    if bpm < 105:
        return "reggaeton_high" if hi else "reggaeton_mid"
    elif bpm < 145:
        if 113 <= bpm <= 138:
            return "bachata_high" if hi else "bachata_mid"
        return "cumbia_high" if hi else "cumbia_mid"
    else:
        return "salsa_high" if hi else "salsa_mid"

# ── Offline fallback sequences (only used when H100 is down) ──────────────────
_FALLBACK_CHOREO: dict = {
    "silence"        : ["home"],
    "soft"           : ["sway_left","sway_right","slow_lean_left","slow_lean_right"],
    "reggaeton_mid"  : ["prreo_pump","prreo_drop","prreo_up","dembow_L","prreo_up","dembow_R"],
    "reggaeton_high" : ["dembow_L","fist_L","prreo_drop","dembow_R","fist_R","prreo_up","spin_L","prreo_pump","spin_R","snap_open"],
    "cumbia_mid"     : ["cumbia_L","cumbia_R","cumbia_dip","cumbia_L","cumbia_R","reach_high"],
    "cumbia_high"    : ["cumbia_L","fist_L","cumbia_dip","cumbia_R","fist_R","spin_L","groove_wide_L","cumbia_dip","groove_wide_R","spin_R"],
    "bachata_mid"    : ["bachata_side_L","bachata_hip","bachata_side_R","bachata_hip"],
    "bachata_high"   : ["bachata_side_L","fist_L","bachata_hip","bachata_side_R","fist_R","spin_L","bachata_hip","spin_R"],
    "salsa_mid"      : ["salsa_L","salsa_R","salsa_up","salsa_snap","salsa_L","salsa_R"],
    "salsa_high"     : ["salsa_L","salsa_snap","spin_L","salsa_R","salsa_snap","spin_R","fist_L","salsa_up","fist_R","snap_open"],
}
_fallback_step: int = 0


def _fallback_pick(beat: dict, history: list) -> str:
    """Offline fallback: cycle the lookup table. Used when H100 is unreachable."""
    global _fallback_step
    genre = beat.get("genre", "silence")
    seq   = _FALLBACK_CHOREO.get(genre, _FALLBACK_CHOREO["silence"])
    move  = seq[_fallback_step % len(seq)]
    last  = history[-1] if history else ""
    if move == last and len(seq) > 1:
        move = seq[(_fallback_step + 1) % len(seq)]
    _fallback_step += 1
    return move


# ── AI move picker via Cosmos Reason2 ─────────────────────────────────────────
_MOVE_LIST = list(ARM_SERVO_MOVES.keys())  # all valid moves
_R2_TIMEOUT = 2.5  # keep tight so we stay on beat

_R2_DANCE_PROMPT = """You are a real-time AI choreographer for a physical robot arm reacting to live music.

Audio analysis:
  BPM:     {bpm}
  Energy:  {energy}  (0=silence, 0.04=soft, 0.12+=loud)
  Genre:   {genre}
  Onsets:  {onsets} detected this chunk

Recent moves (last 5, avoid repeating): {history}

Available moves:
{move_list}

Pick ONE move that fits the music right now. Consider:
- Match intensity: high energy = bold fast moves (fist, spin, salsa_snap, dembow)
- Match genre: reggaeton=prreo/dembow, cumbia=cumbia_*/groove, bachata=bachata_*/slow, salsa=salsa_*
- Vary the choice — don't repeat recent moves unless they fit perfectly
- Silence/soft energy = sway or home

Respond with ONLY the move name, nothing else."""


async def _ai_pick_move(client, beat: dict, history: list) -> tuple[str, str]:
    """
    Ask Cosmos Reason2 to pick the next dance move based on live audio.
    Returns (move_name, source) where source is 'cosmos_r2' or 'fallback'.
    """
    if beat.get("energy", 0) < 0.015:
        return "home", "silence"

    recent = history[-5:] if len(history) >= 5 else history
    prompt = _R2_DANCE_PROMPT.format(
        bpm=round(beat.get("bpm", 100), 1),
        energy=round(beat.get("energy", 0), 4),
        genre=beat.get("genre", "?"),
        onsets=beat.get("onset_count", 0),
        history=", ".join(recent) if recent else "none yet",
        move_list="  " + "\n  ".join(_MOVE_LIST),
    )

    try:
        r = await client.post(
            f"{REASON2_URL}/reason",
            json={"prompt": prompt, "image_base64": None, "max_tokens": 20},
            timeout=_R2_TIMEOUT,
        )
        if r.status_code == 200:
            raw = r.json().get("answer", "").strip().lower().split()[0] if r.json().get("answer") else ""
            # Sanitize: must be a real move
            move = raw.strip(".,!?\'\"")
            if move in ARM_SERVO_MOVES:
                return move, "cosmos_r2"
    except Exception as e:
        logger.debug("R2 dance pick failed: %s", e)

    # Fallback
    return _fallback_pick(beat, history), "fallback"

# ── Audio capture ─────────────────────────────────────────────────────────────
def _capture_audio_chunk(duration_secs: float = CHUNK_SECS) -> Optional[bytes]:
    """Record PCM from C270 mic via arecord. Returns raw 16-bit LE mono bytes."""
    cmd = [
        "arecord",
        "-D", MIC_DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-t", "raw",
        "--duration", str(duration_secs),
        "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=duration_secs + 2.5)
        if result.returncode == 0 and len(result.stdout) > 0:
            return result.stdout
        logger.debug("arecord rc=%d stderr=%s", result.returncode,
                     result.stderr[:80] if result.stderr else "")
    except Exception as e:
        logger.warning("arecord failed: %s", e)
    return None


# ── Beat analysis ──────────────────────────────────────────────────────────────
def _analyze_beat(pcm_bytes: bytes) -> dict:
    """
    Full onset-detection beat analysis.
    Returns {energy, bpm, genre, vibe, onset_count}.
    """
    if not pcm_bytes or len(pcm_bytes) < 4:
        return {"energy": 0.0, "bpm": DEFAULT_BPM, "genre": "silence",
                "vibe": "silence", "onset_count": 0}

    n = len(pcm_bytes) // 2
    raw = struct.unpack_from("<%dh" % n, pcm_bytes)
    norm = [s / 32768.0 for s in raw]

    # ── 1. Global RMS energy ──────────────────────────────────────────────────
    energy = math.sqrt(sum(x*x for x in norm) / len(norm))

    # ── 2. Onset detection via spectral flux in short windows ─────────────────
    win_samples = max(1, int(SAMPLE_RATE * 0.022))   # ~22ms window
    hop_samples = max(1, win_samples // 2)

    # Compute per-window RMS
    wins = []
    for i in range(0, len(norm) - win_samples, hop_samples):
        w = norm[i:i + win_samples]
        rms = math.sqrt(sum(x*x for x in w) / len(w))
        wins.append(rms)

    if len(wins) < 4:
        return {"energy": round(energy, 4), "bpm": DEFAULT_BPM,
                "genre": "silence", "vibe": "silence", "onset_count": 0}

    # ── 3. Spectral flux: energy increase between windows ─────────────────────
    flux = [max(0.0, wins[i] - wins[i-1]) for i in range(1, len(wins))]
    mean_flux = sum(flux) / len(flux) if flux else 0.0
    std_flux  = math.sqrt(sum((f - mean_flux)**2 for f in flux) / len(flux)) if flux else 0.0
    threshold = mean_flux + 1.2 * std_flux   # adaptive threshold

    # Detect onsets (flux peaks above threshold with min spacing)
    min_onset_gap = max(1, int(SAMPLE_RATE * 0.15 / hop_samples))  # 150ms min gap
    onsets = []
    last_onset = -min_onset_gap
    for i, f in enumerate(flux):
        if f > threshold and i - last_onset >= min_onset_gap:
            onsets.append(i)
            last_onset = i

    onset_count = len(onsets)

    # ── 4. BPM from inter-onset intervals ────────────────────────────────────
    if onset_count >= 2:
        hop_sec = hop_samples / SAMPLE_RATE
        intervals = [(onsets[i] - onsets[i-1]) * hop_sec
                     for i in range(1, len(onsets))]
        # Median interval to filter outliers
        intervals_sorted = sorted(intervals)
        med_interval = intervals_sorted[len(intervals_sorted) // 2]
        raw_bpm = 60.0 / med_interval if med_interval > 0.01 else DEFAULT_BPM
        bpm = _smooth_bpm(raw_bpm)
    else:
        bpm = _smooth_bpm(DEFAULT_BPM)

    # ── 5. Onset tracking for persistent BPM ─────────────────────────────────
    now = time.time()
    _onset_times.extend([now] * onset_count)

    # ── 6. Genre / vibe label ─────────────────────────────────────────────────
    genre = _bpm_to_genre_energy(bpm, energy)
    if energy < 0.015:
        vibe = "silence"
    elif energy < 0.04:
        vibe = f"soft  {bpm:.0f}bpm"
    elif bpm < 105:
        vibe = f"reggaeton  {bpm:.0f}bpm  E={energy:.3f}"
    elif 113 <= bpm <= 138:
        vibe = f"bachata  {bpm:.0f}bpm  E={energy:.3f}"
    elif bpm < 145:
        vibe = f"cumbia  {bpm:.0f}bpm  E={energy:.3f}"
    else:
        vibe = f"salsa  {bpm:.0f}bpm  E={energy:.3f}"

    return {
        "energy":       round(energy, 4),
        "bpm":          round(bpm, 1),
        "genre":        genre,
        "vibe":         vibe,
        "onset_count":  onset_count,
    }




# ── Execute move on arm ────────────────────────────────────────────────────────
async def _execute_move(client, move: str) -> bool:
    """Execute a dance move. Prefers direct servo control, falls back to named endpoints."""
    # Try direct servo move first (our confirmed positions)
    if move in ARM_SERVO_MOVES:
        sv = dict(ARM_SERVO_MOVES[move])
        ms = sv.pop("_ms", 600)
        try:
            r = await client.post(
                f"{AGENT_URL}/arm/group_move",
                json={"positions": sv, "duration_ms": ms},
                timeout=6.0,
            )
            ok = r.status_code == 200
            logger.info("ARM servo %s ok=%s  ms=%d", move, ok, ms)
            return ok
        except Exception as e:
            logger.warning("ARM group_move %s failed: %s", move, e)

    # Fallback to named endpoint
    endpoint = ARM_MOVES.get(move)
    if endpoint:
        try:
            r = await client.post(f"{AGENT_URL}{endpoint}", json={}, timeout=5.0)
            ok = r.status_code == 200
            logger.info("ARM named %s → %s ok=%s", move, endpoint, ok)
            return ok
        except Exception as e:
            logger.warning("ARM named %s failed: %s", move, e)

    logger.warning("Unknown move: %s", move)
    return False


# ── Main dance loop ────────────────────────────────────────────────────────────
_dance_running = False
_dance_log: list = []
_last_beat: dict = {}
_last_move_source: str = "none"


async def _dance_loop(total_moves: int = MAX_MOVES, fixed_beat: Optional[dict] = None):
    global _dance_running, _dance_log, _last_beat, _last_move_source, _fallback_step
    _dance_running   = True
    _dance_log       = []
    _fallback_step   = 0
    history: list    = []

    logger.info("DANCE starting — mic=%s  fixed=%s  ai_moves=cosmos_r2", MIC_DEVICE, bool(fixed_beat))

    import httpx
    async with httpx.AsyncClient() as client:
        for i in range(total_moves):
            if not _dance_running:
                break

            t0 = time.time()

            # ── 1. Capture & analyze audio ────────────────────────────────────
            if fixed_beat:
                beat = dict(fixed_beat)  # copy so analysis fields can be enriched
            else:
                pcm  = await asyncio.get_event_loop().run_in_executor(
                    None, _capture_audio_chunk, CHUNK_SECS)
                beat = _analyze_beat(pcm or b"")
            _last_beat = beat

            # ── 2. Cosmos Reason2 picks the move (AI, not lookup table) ───────
            move, source = await _ai_pick_move(client, beat, history)
            _last_move_source = source

            # ── 3. Execute on arm (fire-and-forget, overlaps next cycle) ──────
            asyncio.ensure_future(_execute_move(client, move))

            entry = {
                "step":       i + 1,
                "move":       move,
                "source":     source,
                "bpm":        beat["bpm"],
                "energy":     beat["energy"],
                "genre":      beat.get("genre", "?"),
                "vibe":       beat.get("vibe", ""),
                "latency_ms": round((time.time() - t0) * 1000),
            }
            _dance_log.append(entry)
            history.append(move)
            logger.info("Step %d | [%s] %s | %s | bpm=%.0f E=%.3f",
                        i+1, source, move, beat.get("genre","?"), beat["bpm"], beat["energy"])

            # ── 4. Sync to beat: wait remainder of 1-beat window ──────────────
            elapsed       = time.time() - t0
            beat_duration = 60.0 / max(60.0, beat["bpm"])
            remaining     = beat_duration - elapsed
            if remaining > 0.05:
                await asyncio.sleep(remaining)

    import httpx
    async with httpx.AsyncClient() as client:
        await _execute_move(client, "home")

    _dance_running = False
    logger.info("DANCE complete — %d moves, ai_picks=%d",
                len(_dance_log),
                sum(1 for e in _dance_log if e.get("source") == "cosmos_r2"))


# ── Pydantic models ───────────────────────────────────────────────────────────
from pydantic import BaseModel

class MusicReactRequest(BaseModel):
    pcm_base64: Optional[str] = None   # base64 raw S16_LE 22050Hz mono PCM from browser
    bpm: Optional[float]       = None  # override if browser already computed BPM
    energy: Optional[float]    = None  # override if browser already computed energy
    execute: bool              = True  # actually move the arm


# ── FastAPI endpoints ──────────────────────────────────────────────────────────

@router.get("/mic")
async def mic_beat(duration: float = 0.35):
    """
    Capture one audio chunk from the Pi mic and return beat analysis.
    Windows polls this endpoint to drive the arm in real-time.

    Returns: {ok, bpm, energy, genre, vibe, onset_count}
    """
    loop = asyncio.get_event_loop()
    pcm  = await loop.run_in_executor(None, _capture_audio_chunk, duration)
    if pcm is None:
        return {
            "ok": False, "error": "mic unavailable (arecord failed)",
            "bpm": DEFAULT_BPM, "energy": 0.0,
            "genre": "silence", "vibe": "mic error", "onset_count": 0,
        }
    beat = _analyze_beat(pcm)
    return {"ok": True, **beat}


@router.post("/start")
async def dance_start(moves: int = MAX_MOVES):
    """
    Start the full mic-driven dance loop on the Pi.
    Pi mic → onset detection → BPM → genre → servo move → xArm.
    """
    global _dance_running
    if _dance_running:
        return {"ok": False, "error": "Dance already running — POST /cosmos-dance/stop first"}
    asyncio.ensure_future(_dance_loop(total_moves=moves))
    return {
        "ok": True, "status": "started", "moves": moves,
        "mic": MIC_DEVICE, "sample_rate": SAMPLE_RATE,
        "genres_supported": ["reggaeton", "cumbia", "bachata", "salsa"],
    }


@router.post("/demo")
async def dance_demo(bpm: float = 100.0, energy: float = 0.18, moves: int = 24):
    """
    Fixed-beat demo — no mic needed. Cosmos Reason2 picks every move.
    bpm=88,  energy=0.18  -> reggaeton feel
    bpm=130, energy=0.12  -> cumbia feel
    bpm=125, energy=0.15  -> bachata feel
    bpm=170, energy=0.22  -> salsa feel
    """
    global _dance_running
    if _dance_running:
        return {"ok": False, "error": "Dance already running"}
    genre = _bpm_to_genre_energy(bpm, energy)
    beat  = {"bpm": bpm, "energy": energy,
             "genre": genre, "vibe": f"demo {genre} bpm={bpm:.0f}",
             "onset_count": 4}
    asyncio.ensure_future(_dance_loop(total_moves=moves, fixed_beat=beat))
    return {
        "ok": True, "status": "demo_started",
        "ai_choreographer": "cosmos_reason2",
        "bpm": bpm, "energy": energy, "genre": genre, "moves": moves,
        "duration_sec": round(moves * 60.0 / bpm, 1),
    }


@router.post("/music-react")
async def music_react(req: MusicReactRequest):
    """
    Single-shot music reaction — browser sends one audio chunk (or pre-computed
    BPM/energy), Cosmos Reason2 picks the move, arm executes it.

    Browser flow:
      1. getUserMedia() → AudioContext → ScriptProcessor or AudioWorklet
      2. Downsample to 22050Hz mono S16_LE PCM
      3. base64-encode and POST here every ~350ms
      4. Arm moves in real-time to the music
    """
    import httpx
    import base64

    # ── Analyze audio if PCM provided ────────────────────────────────────────
    if req.pcm_base64:
        try:
            raw_pcm = base64.b64decode(req.pcm_base64)
            beat = _analyze_beat(raw_pcm)
        except Exception as e:
            return {"ok": False, "error": f"PCM decode failed: {e}"}
    elif req.bpm is not None:
        # Browser already computed BPM/energy (e.g. via Web Audio API)
        bpm    = req.bpm
        energy = req.energy or 0.1
        genre  = _bpm_to_genre_energy(bpm, energy)
        beat   = {"bpm": bpm, "energy": energy, "genre": genre,
                  "vibe": f"browser {genre} {bpm:.0f}bpm", "onset_count": 2}
    else:
        # No audio — capture from Pi mic
        loop = asyncio.get_event_loop()
        pcm  = await loop.run_in_executor(None, _capture_audio_chunk, CHUNK_SECS)
        beat = _analyze_beat(pcm or b"")

    if beat.get("energy", 0) < 0.01:
        return {"ok": True, "move": "home", "source": "silence", "beat": beat}

    # ── Cosmos Reason2 picks the move ─────────────────────────────────────────
    async with httpx.AsyncClient() as client:
        move, source = await _ai_pick_move(client, beat, _dance_log[-5:] if _dance_log else [])

        # ── Execute on arm ────────────────────────────────────────────────────
        arm_ok = False
        if req.execute:
            arm_ok = await _execute_move(client, move)

    return {
        "ok":      True,
        "move":    move,
        "source":  source,
        "arm_ok":  arm_ok,
        "beat":    beat,
    }


@router.post("/stop")
async def dance_stop():
    """Stop the dance loop after current move."""
    global _dance_running
    _dance_running = False
    return {"ok": True, "status": "stopping", "moves_completed": len(_dance_log)}


@router.get("/status")
async def dance_status():
    """Current dance state + live BPM and move log."""
    ai_picks   = sum(1 for e in _dance_log if e.get("source") == "cosmos_r2")
    total_done = len(_dance_log)
    return {
        "running":          _dance_running,
        "moves_completed":  total_done,
        "ai_picks":         ai_picks,
        "fallback_picks":   total_done - ai_picks,
        "last_move_source": _last_move_source,
        "last_beat":        _last_beat,
        "last_move":        _dance_log[-1] if _dance_log else None,
        "history":          _dance_log[-10:],
        "moves_available":  list(ARM_SERVO_MOVES.keys()),
        "ai_model":         "cosmos_reason2",
        "r2_url":           REASON2_URL,
    }
