"""
Hiwonder xArm Inverse Kinematics
==================================
Implements the exact IK from the official Hiwonder documentation.

Coordinate system (from docs section 5.2.2):
  Origin: bottom of servo at base platform
  Y+: forward from the arm's front
  X+: right (arm's first-person view)
  Z+: up

Link lengths (from docs, cm):
  L1 = 6.9   base to servo5 output shaft
  L2 = 9.5   servo5 to servo4 output shaft
  L3 = 9.5   servo4 to servo3 output shaft
  L4 = 16.9  servo3 to gripper tip (closed)

Reference positions (from docs):
  Home / initial: ki_move(0, 17, 20.5, 0)    — gripper level, 17cm forward
  Pick table:     ki_move(0, 17, 1.2, -71)    — gripper angled down, standard
  Left 90°:       ki_move(-17, 0, 20.5, 0)    — arm pointing left
  Left place:     ki_move(-19.5, 0, 2.8, -60) — drop at left bin

Servo mapping (measured from stored arm positions vs known IK positions):
  S6 = base rotation (left-right)
       center=500, left=240 (bin), right≈760
       ~15.3 servo units per cm of X at full reach
  S5 = shoulder (link2 joint)
  S4 = elbow (link3 joint)
  S3 = wrist pitch (link4 joint)
  S2 = wrist roll (not in IK plane)
  S1 = gripper: 100=open, 500=closed
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from typing import Dict, Optional, Tuple

logger = logging.getLogger("nis.kinematics")

PI_URL = "http://192.168.1.163:8085"

# ── Link lengths (cm) from official documentation ─────────────────────────────
L1 = 6.9    # base to first joint (vertical)
L2 = 9.5    # upper arm
L3 = 9.5    # forearm
L4 = 16.9   # wrist to gripper tip

# ── Servo calibration — empirically derived from two physical anchors ─────────
#
# ANCHOR 1 (HOME pose stored in arm memory):
#   ki_move(0, 17, 20.5, 0)  =>  IK: t1=45.29, t2=88.58, t3=-133.87
#   Stored servos: S2=484, S3=433, S4=500, S5=432, S6=350
#
# ANCHOR 2 (PICK_TABLE stored in arm memory):
#   ki_move(0, 17, 1.2, -71) =>  IK: t1=6.06, t2=71.47, t3=-148.53
#   Stored servos: S2=258, S3=733, S4=500, S5=850, S6=500
#
# Note: S6=350 at home ≠ S6=500 at pick. The user's home is slightly rotated
# left. S6=500 is the "picking center" (straight forward for pick operations).
# Home deliberately uses S6=350 (user-set physical reference).

# S5 (shoulder): scale derived from (432,45.29°) and (850,6.06°)
S5_ANCHOR_DEG  = 45.29   # IK theta1 at stored HOME
S5_ANCHOR_UNIT = 432      # S5 value at stored HOME
S5_SCALE       = 10.66    # units per degree, REVERSED: higher S5 = lower theta1

# S3 (wrist pitch): S3 tracks absolute pitch of gripper directly
# pitch=0 -> S3=433;  pitch=-71 -> S3=733  => scale = -4.23 units/deg
S3_PITCH_ZERO  = 433      # S3 when pitch_deg=0
S3_PITCH_SCALE = 4.23     # S3 units per degree magnitude (S3 increases as pitch goes negative)

# S4 (elbow): stays at 500 for all tested positions (vertical-plane motion)
S4_FIXED = 500

# S2 (wrist roll): changes with arm extension; anchored at home
S2_HOME = 484
# S2 at pick: 258; roughly tracks inverse of arm extension
# Simple approximation: S2 = 484 - (S5-432)*0.54
S2_SCALE = 0.54  # empirical

# S6 (base rotation): S6=500 = picking center (straight forward for pick ops)
# Two calibration anchors:
#   PICK_TABLE: S6=500, angle=0  (straight forward, x=0)
#   PLACE_BIN:  S6=240, angle=-90° (left 90°, x=-17cm at y=0)
# Scale = (500-240)/90 = 2.89 units/degree
# Note: S6_HOME=350 is user's rest position, slightly rotated left from pick center
S6_PICK_CENTER = 500   # S6 for x=0 in pick operations (straight forward)
S6_SCALE       = 2.89  # units per degree of base rotation (empirical, from place_bin anchor)


# ── IK solver ─────────────────────────────────────────────────────────────────

def ik_solve(
    x: float, y: float, z: float,
    pitch_deg: float
) -> Optional[Tuple[float, float, float, float]]:
    """
    Solve inverse kinematics for xArm.

    Args:
        x: lateral offset cm (+right, -left)
        y: forward distance cm (positive = forward)
        z: height cm (from base origin)
        pitch_deg: gripper pitch deg (0=horizontal, -90=straight down)

    Returns:
        (theta_base_deg, theta1_deg, theta2_deg, theta3_deg) or None
        theta_base: base rotation angle
        theta1: shoulder angle
        theta2: elbow angle
        theta3: wrist pitch angle

    Reference equations from Hiwonder docs section 5.2.1:
        m = Px - L4*cos(α)
        n = Pz - L1 - L4*sin(α)
        θ2 = arccos((m²+n² - L2²-L3²) / (2*L2*L3))
        θ1 = arctan(n/m) - arctan(L3*sin(θ2) / (L2 + L3*cos(θ2)))
        θ3 = α - θ1 - θ2
    """
    pitch = math.radians(pitch_deg)

    # Base rotation: maps X displacement to rotation angle
    reach = math.sqrt(x**2 + y**2)
    if reach < 0.001:
        reach = 0.001
    theta_base = math.atan2(x, y)   # angle from Y-axis toward X-axis

    # Project into 2D plane (forward=reach, up=z)
    # Wrist position after removing L4 contribution
    m = reach - L4 * math.cos(pitch)
    n = z - L1 - L4 * math.sin(pitch)

    # Check reachability
    D_sq = m**2 + n**2
    D_arg = (D_sq - L2**2 - L3**2) / (2 * L2 * L3)

    if D_arg < -1.0 or D_arg > 1.0:
        logger.debug(f"IK no solution: D_arg={D_arg:.3f} for ({x},{y},{z},{pitch_deg}deg)")
        return None

    # Elbow angle (take elbow-up solution)
    theta2 = math.acos(D_arg)

    # Shoulder angle
    if abs(m) < 0.001 and abs(n) < 0.001:
        return None
    theta1 = math.atan2(n, m) - math.atan2(
        L3 * math.sin(theta2),
        L2 + L3 * math.cos(theta2)
    )

    # Wrist angle
    theta3 = pitch - theta1 - theta2

    return (
        math.degrees(theta_base),
        math.degrees(theta1),
        math.degrees(theta2),
        math.degrees(theta3),
    )


def ik_to_servos(
    x: float, y: float, z: float,
    pitch_deg: float,
    gripper: int = 500
) -> Optional[Dict[str, int]]:
    """
    Full pipeline: XYZ + pitch → servo position dict {S1..S6}.

    Returns servo positions (0-1000) or None if position unreachable.
    """
    result = ik_solve(x, y, z, pitch_deg)
    if result is None:
        return None

    theta_base, theta1, theta2, theta3 = result

    # ── S6: base rotation ─────────────────────────────────────────────────────
    # S6_PICK_CENTER=500 is straight forward for pick operations.
    # Positive x (right) increases S6.
    base_angle_deg = math.degrees(math.atan2(x, y))  # angle from Y-forward toward X-right
    s6 = int(round(S6_PICK_CENTER + base_angle_deg * S6_SCALE))

    # ── S5: shoulder (theta1) — empirical scale ───────────────────────────────
    # S5 INCREASES as theta1 DECREASES (arm goes from raised to down)
    s5 = int(round(S5_ANCHOR_UNIT + (S5_ANCHOR_DEG - theta1) * S5_SCALE))

    # ── S4: elbow — fixed at 500 (no significant variation in tested range) ───
    s4 = S4_FIXED

    # ── S3: wrist pitch — tracks absolute pitch directly ─────────────────────
    # S3 increases as pitch goes more negative (gripper points more downward)
    s3 = int(round(S3_PITCH_ZERO + (-pitch_deg) * S3_PITCH_SCALE))

    # ── S2: wrist roll — approximated from arm extension ─────────────────────
    # Empirical: S2 decreases as arm extends downward (S5 increases)
    s2 = int(round(S2_HOME - (s5 - S5_ANCHOR_UNIT) * S2_SCALE))

    # ── S1: gripper ───────────────────────────────────────────────────────────
    s1 = max(100, min(900, gripper))

    # Clamp all to valid range
    servos = {
        "1": max(100, min(900, s1)),
        "2": max(200, min(900, s2)),
        "3": max(100, min(900, s3)),
        "4": max(200, min(800, s4)),
        "5": max(200, min(900, s5)),
        "6": max(100, min(900, s6)),
    }

    logger.debug(
        f"IK({x},{y},{z},{pitch_deg}deg) -> "
        f"th=({theta_base:.1f},{theta1:.1f},{theta2:.1f},{theta3:.1f})deg -> "
        f"S={servos}"
    )
    return servos


# ── HTTP move helper ───────────────────────────────────────────────────────────

def ki_move_http(
    x: float, y: float, z: float,
    pitch_deg: float,
    duration_ms: int = 1200,
    gripper: int = 500,
) -> Dict:
    """
    Compute IK and send group_move to Pi via HTTP.
    Equivalent to the on-device kinematics.ki_move() call.
    """
    servos = ik_to_servos(x, y, z, pitch_deg, gripper)
    if servos is None:
        return {"ok": False, "error": f"IK no solution for ({x},{y},{z},{pitch_deg}°)"}

    try:
        body = json.dumps({
            "positions": servos,
            "duration_ms": duration_ms,
        }).encode()
        req = urllib.request.Request(
            PI_URL + "/arm/group_move", data=body,
            headers={"Content-Type": "application/json"}
        )
        r = urllib.request.urlopen(req, timeout=12)
        result = json.loads(r.read())
        result["ik_servos"] = servos
        result["ik_angles"] = ik_solve(x, y, z, pitch_deg)
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Reference positions (from documentation) ──────────────────────────────────

class Pose:
    """Standard pipeline positions in XYZ cm + pitch degrees."""

    # From Hiwonder docs section 5.2.4 and Color Sorting examples:
    HOME          = (0,     17,    20.5,   0)    # initial position
    INSPECT       = (0,     17,    20.5,   0)    # same height, looking down via camera
    PICK_FRONT    = (0,     17,     1.2,  -71)   # standard pick (object at center-front)
    LIFT          = (0,     17,    20.5,   0)    # lift after pick
    PLACE_LEFT_90 = (-17,    0,    20.5,   0)    # rotate left 90° (transition)
    DROP_LEFT_90  = (-19.5,  0,     2.8,  -60)   # lower to drop at left 90°
    PLACE_RIGHT_90 = (17,    0,    20.5,   0)    # rotate right 90°
    DROP_RIGHT_90  = (19.5,  0,     2.8,  -60)   # lower to drop at right 90°

    # For our cookoff: lighter is to the right (~x=+6cm based on S6 sweep)
    # Adjust PICK_X to move gripper right
    PICK_LIGHTER  = (6,     17,     1.2,  -71)   # lighter at ~x=6cm right
    LIFT_LIGHTER  = (6,     17,    20.5,   0)    # lift position matching lighter
    PLACE_BIN     = (-17,   0,     20.5,   0)    # sweep to left bin
    DROP_BIN      = (-19.5, 0,      2.8,  -60)   # drop in bin


def verify_reference_positions():
    """Compute servo positions for all reference positions and print them."""
    print("\nIK Verification — reference positions from documentation:")
    print("=" * 70)

    refs = {
        "HOME          ": Pose.HOME,
        "PICK_FRONT    ": Pose.PICK_FRONT,
        "LIFT          ": Pose.LIFT,
        "PLACE_LEFT_90 ": Pose.PLACE_LEFT_90,
        "DROP_LEFT_90  ": Pose.DROP_LEFT_90,
        "PICK_LIGHTER  ": Pose.PICK_LIGHTER,
        "PLACE_BIN     ": Pose.PLACE_BIN,
        "DROP_BIN      ": Pose.DROP_BIN,
    }

    for name, (x, y, z, p) in refs.items():
        servos = ik_to_servos(x, y, z, p)
        if servos:
            s = " ".join(f"S{k}={v}" for k, v in sorted(servos.items()))
            print(f"  {name} ({x:+.1f},{y:.1f},{z:.1f},{p:+.0f}deg) -> {s}")
        else:
            print(f"  {name} -> NO SOLUTION")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    verify_reference_positions()
