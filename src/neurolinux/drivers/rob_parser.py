"""
Hiwonder .rob Action Group File Parser
=======================================
Decodes the binary SOBARM format used by the Hiwonder PC software.

Format:
  Header: b'SOBARM' + 10 padding bytes (16 bytes total)
  Byte 6-7 (little-endian uint16): number of frames
  Frames: each is 14 bytes:
    [0:2]  duration_ms  (uint16 LE)
    [2:4]  servo 1 pos  (uint16 LE)  - GRIPPER: 100=open, 500=closed
    [4:6]  servo 2 pos  (uint16 LE)  - always ~500 (fixed)
    [6:8]  servo 3 pos  (uint16 LE)  - elbow/shoulder joint
    [8:10] servo 4 pos  (uint16 LE)  - shoulder joint
    [10:12] servo 5 pos (uint16 LE)  - shoulder joint
    [12:14] servo 6 pos (uint16 LE)  - BASE ROTATION: 500=center, 875=left90, 125=right90

Official action groups (NO.0 - NO.9):
  0: Initialization 1  - HOME position (S6=500 center)
  1: Initialization 2  - Alternative home
  2: Grip + Place Left  90 deg
  3: Grip + Place Left  45 deg
  4: Grip + Place Right 45 deg
  5: Grip + Place Right 90 deg
  6: Place Left  90 deg (arm already has object)
  7: Place Left  45 deg
  8: Place Right 45 deg
  9: Place Right 90 deg

Key calibrated positions (from official .rob files):
  HOME:          {1:100, 2:500, 3:310, 4:870, 5:680, 6:500}
  PICK LOW:      {1:100, 2:500, 3:180, 4:800, 5:450, 6:500}
  PLACE L 90:    S6=875
  PLACE L 45:    S6=685
  PLACE R 45:    S6=315
  PLACE R 90:    S6=125

S6 direction:
  500 = center (forward)
  875 = arm's LEFT  90 degrees
  315 = arm's RIGHT 45 degrees
  125 = arm's RIGHT 90 degrees
  -> DECREASING S6 = rotating arm RIGHT
"""

import struct
from pathlib import Path
from typing import List, Dict


FRAME_SIZE = 14  # bytes per frame
HEADER_SIZE = 16  # bytes before first frame
SERVO_COUNT = 6


def parse_rob(path: str) -> List[Dict]:
    """
    Parse a .rob action group file.

    Returns list of frames, each dict:
      {'duration_ms': int, 'servos': {1: int, 2: int, ..., 6: int}}
    """
    data = Path(path).read_bytes()

    if not data.startswith(b'SOBARM'):
        raise ValueError(f"Not a SOBARM file: {path}")

    # Frame count is at byte 6 (uint16 LE)
    n_frames = struct.unpack_from('<H', data, 6)[0]
    payload = data[HEADER_SIZE:]

    frames = []
    for i in range(n_frames):
        offset = i * FRAME_SIZE
        chunk = payload[offset: offset + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            break

        duration_ms = struct.unpack_from('<H', chunk, 0)[0]
        servos = {}
        for s in range(SERVO_COUNT):
            pos = struct.unpack_from('<H', chunk, 2 + s * 2)[0]
            servos[s + 1] = pos

        frames.append({'duration_ms': duration_ms, 'servos': servos})

    return frames


def get_key_positions(rob_dir: str) -> Dict[str, Dict[int, int]]:
    """
    Load all .rob files and extract the key calibrated positions.
    Returns dict of position_name -> servo_positions.
    """
    d = Path(rob_dir)
    positions = {}

    # NO.0 = home
    f = d / "NO.0 Initialization Action 1.rob"
    if f.exists():
        frames = parse_rob(str(f))
        if frames:
            positions['home_official'] = frames[0]['servos']

    # NO.2 = grip+place left 90 -> has home, pick, place positions
    f2 = d / "NO.2 Grip and Place Left 90\u00b0.rob"
    if not f2.exists():
        # Try with degree symbol variant
        for name in d.glob("NO.2*.rob"):
            f2 = name
            break

    if f2 and f2.exists():
        frames = parse_rob(str(f2))
        if len(frames) >= 10:
            positions['pick_low'] = frames[1]['servos']   # frame 1 = lowered to pick
            positions['lift']     = frames[5]['servos']   # frame 5 = lifted
            positions['place_left_90'] = frames[9]['servos']  # frame 9 = at place left 90

    # Extract S6 values for each direction
    for fname, label in [
        ("NO.6 Place Left 90*.rob",  "s6_left_90"),
        ("NO.7 Place Left 45*.rob",  "s6_left_45"),
        ("NO.8 Place Right 45*.rob", "s6_right_45"),
        ("NO.9 Place Right 90*.rob", "s6_right_90"),
    ]:
        for f in d.glob(fname):
            frames = parse_rob(str(f))
            if frames:
                positions[label] = frames[0]['servos'][6]
            break

    return positions


# ── Hardcoded official positions (pre-parsed, always available) ──────────────

OFFICIAL = {
    'home':         {1: 100, 2: 500, 3: 310, 4: 870, 5: 680, 6: 500},
    'pick_low':     {1: 100, 2: 500, 3: 180, 4: 800, 5: 450, 6: 500},
    'grip_closed':  {1: 500, 2: 500, 3: 310, 4: 870, 5: 680, 6: 500},
    'place_low':    {1: 500, 2: 500, 3: 220, 4: 800, 5: 460, 6: 500},

    # S6 values for each rotation direction
    's6_center':    500,
    's6_left_45':   685,
    's6_left_90':   875,
    's6_right_45':  315,
    's6_right_90':  125,
}


def pick_pose(s6: int) -> Dict[int, int]:
    """Return pick approach pose with custom S6 rotation."""
    p = dict(OFFICIAL['home'])
    p[6] = s6
    return p


def pick_low_pose(s6: int) -> Dict[int, int]:
    """Return pick low (grabbing) pose with custom S6 rotation."""
    p = dict(OFFICIAL['pick_low'])
    p[6] = s6
    return p


def place_pose(s6: int) -> Dict[int, int]:
    """Return place pose with custom S6 rotation."""
    p = dict(OFFICIAL['place_low'])
    p[6] = s6
    return p


if __name__ == '__main__':
    import sys, json

    rob_dir = "docs/arm_docs/action_groups"
    if len(sys.argv) > 1:
        rob_dir = sys.argv[1]

    print("Official positions (hardcoded from .rob files):")
    for name, vals in OFFICIAL.items():
        print(f"  {name}: {vals}")

    print()
    print("Parsed from files:")
    try:
        pos = get_key_positions(rob_dir)
        for k, v in pos.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  (Could not parse files: {e})")
