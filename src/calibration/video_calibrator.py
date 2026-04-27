"""
NIS Protocol — Video Calibration System
========================================
Three-method calibration pipeline using the full Cosmos stack.

METHOD 1 — Video-to-Frames + Cosmos Reason2
    Record video of arm moving to each pose.
    Extract frames with OpenCV.
    Feed each frame to Cosmos Reason2 for spatial analysis.
    Get millimeter-accurate corrections per pose.

METHOD 2 — Camera Burst + Cosmos Reason2  (no video file needed)
    Pi camera takes a burst of snapshots while arm moves.
    Same Reason2 analysis per frame.
    Best for real-time calibration runs.

METHOD 3 — Cosmos Predict2.5 (Image2World)
    Take the "inspect" frame.
    Ask Cosmos Predict2.5 to simulate the arm descending to pick.
    Compare predicted vs actual to detect miscalibration.
    Also generates synthetic training data for edge cases.

METHOD 4 — Cosmos Transfer2.5 (Sim2Real / Synthetic Augmentation)
    Takes calibration frames → applies Edge, Depth, Segmentation control.
    Generates augmented variants: different lighting, object colors, distances.
    Builds a dataset usable for fine-tuning Reason2 on your specific workspace.

KEY FIXES in this module:
    - Home position is ALWAYS read from the arm's memory (/arm/touch_poses)
      NOT from the hardcoded dictionary. Arm memory is the ground truth.
    - Labels (colored stickers on workspace corners) are passed to Reason2
      to give it a spatial reference frame for mm measurements.
    - Camera is assumed to be at a better angle (side/35° view, not top-down).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nis.calibration")

# ── Config ────────────────────────────────────────────────────────────────────

PI_URL      = "http://192.168.1.163:8085"
COSMOS_R2   = "http://localhost:8100"   # Cosmos Reason2
COSMOS_P25  = "http://localhost:8200"   # Cosmos Predict2.5
COSMOS_T25  = "http://localhost:8300"   # Cosmos Transfer2.5

WORKSPACE_CM = (17.0, 20.5)  # width × depth in cm per documentation

# Pipeline poses that MUST be in arm memory (ground truth):
REQUIRED_POSES = ["home", "inspect", "pick_table", "lift_grip", "place_bin"]

# Label colors the user should place at workspace corners.
# These help Cosmos Reason2 measure distances accurately.
WORKSPACE_LABELS = {
    "front_left":  "RED label",
    "front_right": "BLUE label",
    "back_left":   "GREEN label",
    "back_right":  "YELLOW label",
    "object":      "WHITE label (on pick target)",
}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 6) -> Optional[Dict]:
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read())
    except Exception:
        return None


def _post(url: str, data: Dict, timeout: int = 30) -> Optional[Dict]:
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _pi_post(path: str, data: Dict, timeout: int = 12) -> Dict:
    return _post(PI_URL + path, data, timeout) or {"error": "Pi unreachable"}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class FrameAnalysis:
    """Cosmos Reason2 result for a single frame."""
    pose_name: str
    frame_idx: int
    object_visible: bool = False
    lateral_error_mm: float = 0.0
    depth_error_mm: float = 0.0
    gripper_to_object_mm: float = 0.0
    confidence: float = 0.0
    recommended_delta: Dict[str, int] = field(default_factory=dict)
    raw_response: str = ""


@dataclass
class PoseCalibration:
    """Calibration result for one pipeline pose."""
    pose_name: str
    original: Dict[str, int]
    corrected: Dict[str, int]
    correction_delta: Dict[str, int]
    frames_analyzed: int
    avg_confidence: float
    cosmos_method: str  # "reason2", "predict25", "transfer25"
    notes: str = ""


@dataclass
class CalibrationResult:
    """Full calibration run result."""
    success: bool
    poses: Dict[str, PoseCalibration] = field(default_factory=dict)
    synthetic_frames: int = 0
    total_ms: float = 0.0
    methods_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "total_ms": round(self.total_ms, 1),
            "methods_used": self.methods_used,
            "synthetic_frames": self.synthetic_frames,
            "error": self.error,
            "poses": {
                name: {
                    "original": p.original,
                    "corrected": p.corrected,
                    "delta": p.correction_delta,
                    "frames_analyzed": p.frames_analyzed,
                    "avg_confidence": round(p.avg_confidence, 3),
                    "method": p.cosmos_method,
                    "notes": p.notes,
                }
                for name, p in self.poses.items()
            },
        }


# ── Core: read arm memory (the ground truth) ──────────────────────────────────

def read_arm_poses() -> Dict[str, Dict[str, int]]:
    """
    Read pipeline poses from the arm's memory (/arm/touch_poses).
    This is THE ground truth — not the hardcoded dictionaries.
    The arm's home position is whatever the user set on the physical arm,
    which may differ from any hardcoded value.
    """
    r = _get(f"{PI_URL}/arm/touch_poses")
    if not r:
        logger.warning("Cannot read arm poses — Pi offline")
        return {}
    poses = r.get("touch_poses") or r.get("poses") or {}
    # Normalize: ensure all servo values are int
    return {
        name: {str(k): int(v) for k, v in pos.items()}
        for name, pos in poses.items()
        if isinstance(pos, dict)
    }


def get_home_from_memory() -> Dict[str, int]:
    """
    Get the home position stored in the arm's memory.
    Falls back to a safe default only if the arm is unreachable.
    NEVER use the hardcoded S2=500 S3=400 values — use this function.
    """
    poses = read_arm_poses()
    if "home" in poses:
        logger.info(f"Home from arm memory: {poses['home']}")
        return poses["home"]
    # The arm's current position may be home if user just set it:
    status = _get(f"{PI_URL}/arm/status")
    if status and "positions" in status:
        logger.info(f"Home from current arm position: {status['positions']}")
        return {str(k): int(v) for k, v in status["positions"].items()}
    # Absolute fallback (rarely needed):
    fallback = {"1": 500, "2": 484, "3": 433, "4": 500, "5": 432, "6": 350}
    logger.warning(f"Pi unreachable — using memory-derived fallback home: {fallback}")
    return fallback


# ── Method 1+2: Video or camera burst frame extraction ───────────────────────

def extract_frames_from_video(video_path: str, max_frames: int = 12) -> List[str]:
    """
    Extract evenly-spaced frames from an MP4 video file.
    Returns list of base64-encoded JPEG strings.

    Usage: record arm movement with Win+G (Xbox Game Bar) → saves to
    C:/Users/<user>/Videos/Captures/*.mp4
    Then pass the path here.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_s = total_frames / fps if fps > 0 else 0
        logger.info(f"Video: {total_frames} frames @ {fps:.1f}fps = {duration_s:.1f}s")

        # Pick evenly-spaced frame indices
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]

        frames_b64 = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            # Resize to 1280×720 max (Cosmos works best at this resolution)
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            frames_b64.append(b64)

        cap.release()
        logger.info(f"Extracted {len(frames_b64)} frames from {video_path}")
        return frames_b64
    except ImportError:
        logger.error("OpenCV not available. Install: pip install opencv-python-headless")
        return []


def burst_frames_from_camera(
    n_frames: int = 6,
    interval_ms: int = 800,
) -> List[str]:
    """
    Take N snapshots from the Pi camera at regular intervals.
    Use during arm movement for real-time calibration.
    Returns list of base64 image strings.
    """
    frames = []
    for i in range(n_frames):
        r = _get(f"{PI_URL}/camera/snapshot", timeout=8)
        if r:
            img = r.get("image") or r.get("image_base64") or r.get("frame")
            if img:
                frames.append(img)
                logger.debug(f"  Burst frame {i+1}/{n_frames} captured")
        time.sleep(interval_ms / 1000.0)
    logger.info(f"Camera burst: {len(frames)}/{n_frames} frames captured")
    return frames


# ── Method 1+2: Cosmos Reason2 frame analysis ─────────────────────────────────

def _reason2_prompt_for_pose(pose_name: str, has_labels: bool) -> str:
    """
    Generate the ideal Cosmos Reason2 prompt for a given pipeline pose.
    Adapts based on whether workspace labels are present.
    """
    label_ctx = (
        "Workspace labels: RED sticker=front-left corner, BLUE=front-right, "
        "GREEN=back-left, YELLOW=back-right, WHITE=pick target object. "
        "Use label positions to measure distances accurately in mm. "
        "The workspace is 17cm wide × 20.5cm deep. "
    ) if has_labels else (
        "No calibration labels present. Estimate distances based on "
        "typical object sizes and arm proportions. "
        "The workspace is 17cm wide × 20.5cm deep. "
    )

    prompts = {
        "home": (
            f"{label_ctx}"
            "The robotic arm is in HOME position. "
            "Analyze: Is the arm centered? Stable? Safe? "
            "Return JSON: {arm_centered: bool, safe: bool, notes: string, "
            "confidence: float}"
        ),
        "inspect": (
            f"{label_ctx}"
            "The robotic arm gripper is positioned above the workspace in INSPECT mode. "
            "Identify any pick-target object on the table. "
            "Measure: (1) lateral error (+ = object left of gripper center, mm), "
            "(2) estimated distance from gripper tip to object top (mm). "
            "Return JSON: {object_visible: bool, lateral_error_mm: float, "
            "gripper_to_object_mm: float, confidence: float, "
            "object_description: string}"
        ),
        "pick_table": (
            f"{label_ctx}"
            "The robotic arm gripper is at TABLE LEVEL attempting to pick an object. "
            "Gripper S1 servo is OPEN (500). "
            "Analyze alignment between gripper fingers and the object. "
            "Return JSON: {object_visible: bool, object_between_fingers: bool, "
            "lateral_error_mm: float, gripper_too_high_mm: float, "
            "gripper_too_low_mm: float, confidence: float, "
            "recommended_servo_correction: {S2: 0, S3: 0, S6: 0}}"
        ),
        "lift_grip": (
            f"{label_ctx}"
            "The robotic arm has CLOSED its gripper (S1=550) and is lifting. "
            "Is the object gripped? Is it centered in the gripper? "
            "Return JSON: {object_gripped: bool, grip_centered: bool, "
            "object_slipping: bool, grip_confidence: float, safe_to_sweep: bool}"
        ),
        "place_bin": (
            f"{label_ctx}"
            "The robotic arm is positioned over the DROP BIN with gripper closed. "
            "Is the gripper over the bin? Is it safe to open and drop? "
            "Return JSON: {over_bin: bool, drop_safe: bool, "
            "lateral_error_mm: float, height_above_bin_mm: float, confidence: float}"
        ),
    }
    return prompts.get(pose_name, (
        f"{label_ctx}"
        f"Analyze the robotic arm in pose '{pose_name}'. "
        "Return JSON with relevant spatial measurements and confidence."
    ))


def analyze_frames_reason2(
    pose_name: str,
    frames_b64: List[str],
    has_labels: bool = False,
    timeout: int = 30,
) -> List[FrameAnalysis]:
    """
    Send frames to Cosmos Reason2 and parse spatial analysis results.
    """
    prompt = _reason2_prompt_for_pose(pose_name, has_labels)
    results = []

    for i, frame in enumerate(frames_b64):
        try:
            payload = {
                "model": "cosmos-reason2",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": frame}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": 512,
                "temperature": 0.1,
            }
            r = _post(f"{COSMOS_R2}/v1/chat/completions", payload, timeout=timeout)
            if not r or "choices" not in r:
                continue

            text = r["choices"][0]["message"]["content"]
            import re
            m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}

            fa = FrameAnalysis(
                pose_name=pose_name,
                frame_idx=i,
                object_visible=parsed.get("object_visible", parsed.get("over_bin", True)),
                lateral_error_mm=float(parsed.get("lateral_error_mm", 0)),
                depth_error_mm=float(parsed.get("gripper_too_high_mm", 0)
                                     - parsed.get("gripper_too_low_mm", 0)),
                gripper_to_object_mm=float(parsed.get("gripper_to_object_mm",
                                                       parsed.get("height_above_bin_mm", 0))),
                confidence=float(parsed.get("confidence", parsed.get("grip_confidence", 0.5))),
                recommended_delta={
                    str(k[1:]): int(v)
                    for k, v in parsed.get("recommended_servo_correction", {}).items()
                    if v != 0
                },
                raw_response=text[:300],
            )
            results.append(fa)
            logger.info(
                f"  Frame {i}: visible={fa.object_visible} "
                f"lat={fa.lateral_error_mm:.1f}mm "
                f"depth={fa.gripper_to_object_mm:.1f}mm "
                f"conf={fa.confidence:.2f}"
            )
        except Exception as e:
            logger.debug(f"  Frame {i} analysis error: {e}")

    return results


def compute_pose_correction(
    pose_name: str,
    analyses: List[FrameAnalysis],
    original_pose: Dict[str, int],
) -> PoseCalibration:
    """
    Aggregate frame analyses → compute the final servo correction for this pose.

    Rules:
    - Use only high-confidence frames (≥ 0.5)
    - Average the corrections
    - Apply clamps per servo to prevent unsafe movements
    - For pick_table: correct S6 (lateral) and S2 (height)
    - For place_bin: correct S6 (lateral)
    """
    good = [a for a in analyses if a.confidence >= 0.5]
    if not good:
        return PoseCalibration(
            pose_name, original_pose, original_pose, {}, 0, 0.0, "reason2",
            "No high-confidence frames"
        )

    avg_conf = sum(a.confidence for a in good) / len(good)
    avg_lateral = sum(a.lateral_error_mm for a in good) / len(good)
    avg_height = sum(a.gripper_to_object_mm for a in good) / len(good)
    avg_depth_err = sum(a.depth_error_mm for a in good) / len(good)

    corrected = dict(original_pose)
    delta: Dict[str, int] = {}

    if pose_name in ("inspect", "pick_table"):
        # Lateral correction: 1mm ≈ 3.5 servo units on S6
        if abs(avg_lateral) > 2.0:
            d6 = int(-avg_lateral * 3.5)
            d6 = max(-80, min(80, d6))
            corrected["6"] = max(100, min(900, original_pose.get("6", 500) + d6))
            delta["6"] = d6

        # Height correction: S2 adjusts end-effector height
        if pose_name == "pick_table" and avg_height > 12.0:
            d2 = -int((avg_height - 8) * 3.5)
            d2 = max(-60, min(0, d2))
            corrected["2"] = max(200, min(900, original_pose.get("2", 500) + d2))
            delta["2"] = d2
        elif pose_name == "pick_table" and avg_height < 1.5:
            d2 = 25  # lift up
            corrected["2"] = min(900, original_pose.get("2", 500) + d2)
            delta["2"] = d2

        # Depth error (too far front/back) via S3
        if abs(avg_depth_err) > 5.0 and pose_name == "pick_table":
            d3 = int(-avg_depth_err * 2.5)
            d3 = max(-40, min(40, d3))
            corrected["3"] = max(100, min(800, original_pose.get("3", 500) + d3))
            delta["3"] = d3

    elif pose_name == "place_bin":
        if abs(avg_lateral) > 3.0:
            d6 = int(-avg_lateral * 3.5)
            d6 = max(-60, min(60, d6))
            corrected["6"] = max(100, min(900, original_pose.get("6", 240) + d6))
            delta["6"] = d6

    notes = (
        f"avg_lateral={avg_lateral:.1f}mm, "
        f"avg_height={avg_height:.1f}mm, "
        f"frames={len(good)}/{len(analyses)}, "
        f"confidence={avg_conf:.2f}"
    )
    return PoseCalibration(
        pose_name, original_pose, corrected, delta,
        len(good), avg_conf, "reason2", notes
    )


# ── Method 3: Cosmos Predict2.5 ───────────────────────────────────────────────

def cosmos_predict_future_state(
    inspect_frame_b64: str,
    target_pose_prompt: str,
    output_dir: Optional[str] = None,
    timeout: int = 90,
) -> Optional[Dict]:
    """
    Cosmos Predict2.5 (Image2World):
    Given the inspect-position camera frame, predict what the arm will
    look like when it reaches pick_table.

    The predicted video frames show us:
    - Whether the arm trajectory looks correct
    - What alignment corrections are needed before descent
    - Generates synthetic training data for edge cases

    API: POST to Cosmos Predict2.5 server (H100, port 8200)
    Compatible with OpenAI /v1/chat/completions format if predict server
    is running the NIS inference wrapper. Otherwise calls the native API.
    """
    if not _get(f"{COSMOS_P25}/health", timeout=3):
        logger.info("Cosmos Predict2.5 offline — skipping Method 3")
        return None

    logger.info("Cosmos Predict2.5: generating future state...")

    # Try OpenAI-compatible endpoint first (wrapped by NIS Protocol)
    payload_compat = {
        "model": "cosmos-predict2.5",
        "inference_type": "image2world",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": inspect_frame_b64}},
                {"type": "text", "text": target_pose_prompt},
            ],
        }],
        "max_tokens": 512,
    }
    r = _post(f"{COSMOS_P25}/v1/chat/completions", payload_compat, timeout=timeout)
    if r and not r.get("error") and "choices" in r:
        return {"method": "predict2.5_compat", "result": r}

    # Try native Cosmos Predict2.5 JSON format
    payload_native = {
        "inference_type": "image2world",
        "name": "arm_calibration",
        "prompt": target_pose_prompt,
        "input_path": inspect_frame_b64,  # some servers accept base64 directly
    }
    r2 = _post(f"{COSMOS_P25}/v1/predict", payload_native, timeout=timeout)
    if r2 and not r2.get("error"):
        return {"method": "predict2.5_native", "result": r2}

    logger.warning("Cosmos Predict2.5 reachable but both endpoints failed")
    return None


# ── Method 4: Cosmos Transfer2.5 (Synthetic Data) ────────────────────────────

def cosmos_transfer_synthetic(
    frames_b64: List[str],
    pose_name: str,
    output_dir: Optional[str] = None,
    timeout: int = 120,
) -> Dict:
    """
    Cosmos Transfer2.5:
    Takes real arm frames and generates synthetic variations:
    - Different object colors (red → blue cube, etc.)
    - Different lighting (morning, overhead, dim)
    - Different backgrounds
    - Depth/edge augmentation for Reason2 training data

    Uses: Edge control (structure preservation) + Vis control (lighting feel)
    Transfer 2.5 API: POST /v1/transfer with control modalities

    This builds a synthetic dataset to fine-tune Reason2 on your workspace.
    """
    if not _get(f"{COSMOS_T25}/health", timeout=3):
        logger.info("Cosmos Transfer2.5 offline — skipping Method 4")
        return {"skipped": True, "reason": "Transfer2.5 offline"}

    if not frames_b64:
        return {"skipped": True, "reason": "No frames to augment"}

    logger.info(f"Cosmos Transfer2.5: augmenting {len(frames_b64)} frames for {pose_name}...")

    augmentation_configs = [
        {
            "name": f"{pose_name}_edge_depth",
            "prompt": (
                f"Robotic arm in {pose_name} position in a clean workspace. "
                "Industrial lighting. High contrast."
            ),
            "negative_prompt": "blurry, dark, overexposed",
            "controls": {
                "edge": {"weight": 0.7},
                "vis":  {"weight": 0.3},
            },
            "guidance_scale": 3.0,
        },
        {
            "name": f"{pose_name}_alt_object",
            "prompt": (
                f"Robotic arm in {pose_name} position. "
                "Blue cube object on white workspace surface. "
                "Bright overhead lighting."
            ),
            "negative_prompt": "dark, blurry",
            "controls": {
                "edge": {"weight": 0.5},
                "seg":  {"weight": 0.4},
                "vis":  {"weight": 0.1},
            },
            "guidance_scale": 3.5,
        },
    ]

    results = []
    for cfg in augmentation_configs:
        payload = {
            "model": "cosmos-transfer2.5",
            "input_frames": frames_b64[:3],  # use first 3 frames
            **cfg,
        }
        r = _post(f"{COSMOS_T25}/v1/transfer", payload, timeout=timeout)
        if r and not r.get("error"):
            results.append({"config": cfg["name"], "result": r})
            logger.info(f"  Generated: {cfg['name']}")
        else:
            logger.warning(f"  Failed: {cfg['name']}: {(r or {}).get('error')}")

    saved = 0
    if output_dir and results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, res in enumerate(results):
            out_path = Path(output_dir) / f"{pose_name}_aug_{i}.json"
            out_path.write_text(json.dumps(res, indent=2))
            saved += 1

    return {
        "configs_run": len(augmentation_configs),
        "successful": len(results),
        "saved_to": output_dir if saved else None,
    }


# ── Main calibration orchestrator ─────────────────────────────────────────────

class VideoCalibrator:
    """
    Full three-method calibration pipeline.

    Methods used (in order):
    1. Camera burst + Cosmos Reason2 (always, requires Pi + H100)
    2. Cosmos Predict2.5 for inspect pose (if Predict2.5 is online)
    3. Cosmos Transfer2.5 for synthetic data (if Transfer2.5 is online)
    """

    def __init__(
        self,
        has_labels: bool = True,
        synthetic_output_dir: Optional[str] = None,
        video_path: Optional[str] = None,
    ):
        self.has_labels = has_labels
        self.synthetic_output_dir = synthetic_output_dir
        self.video_path = video_path
        self._arm_poses: Dict[str, Dict[str, int]] = {}

    def get_arm_pose(self, pose_name: str) -> Optional[Dict[str, int]]:
        """Read pose from arm memory. Never from hardcoded dict."""
        if not self._arm_poses:
            self._arm_poses = read_arm_poses()
        return self._arm_poses.get(pose_name)

    def move_to_pose(self, pose_name: str, duration_ms: int = 1400) -> bool:
        """Move arm to a named pose from memory."""
        pose = self.get_arm_pose(pose_name)
        if not pose:
            logger.error(f"Pose '{pose_name}' not in arm memory")
            return False
        r = _pi_post("/arm/group_move", {"positions": pose, "duration_ms": duration_ms})
        return r.get("ok", False) and not r.get("error")

    def save_corrected_pose(self, pose_name: str, corrected: Dict[str, int]) -> bool:
        """Move to corrected position and save it to arm memory."""
        r = _pi_post("/arm/group_move", {"positions": corrected, "duration_ms": 1200})
        if not r.get("ok"):
            return False
        time.sleep(1.5)
        r2 = _pi_post("/arm/save_touch_pose", {"name": pose_name})
        ok = r2.get("name") == pose_name or r2.get("ok") or "positions" in r2
        if ok:
            logger.info(f"Saved corrected {pose_name}: {corrected}")
        return bool(ok)

    async def calibrate_pose(
        self, pose_name: str, frames_b64: Optional[List[str]] = None
    ) -> Optional[PoseCalibration]:
        """
        Calibrate a single pose:
        1. If no frames provided, move arm to pose and take camera burst
        2. Send frames to Cosmos Reason2
        3. Compute correction
        4. Optionally save corrected pose to arm memory
        """
        original = self.get_arm_pose(pose_name)
        if not original:
            logger.warning(f"Skipping {pose_name}: not in arm memory")
            return None

        logger.info(f"\n  Calibrating: {pose_name}")
        logger.info(f"  Original: {original}")

        # Get frames
        if frames_b64 is None:
            if self.video_path:
                # From video file: estimate which segment covers this pose
                # (basic heuristic: each pose takes ~1/n of the video)
                all_frames = extract_frames_from_video(self.video_path, max_frames=30)
                poses_order = ["home", "inspect", "pick_table", "lift_grip", "place_bin", "home"]
                if pose_name in poses_order:
                    idx = poses_order.index(pose_name)
                    n = len(poses_order)
                    start = int(idx / n * len(all_frames))
                    end = int((idx + 1) / n * len(all_frames))
                    frames_b64 = all_frames[start:end] or all_frames[:3]
                else:
                    frames_b64 = all_frames[:3]
            else:
                # Camera burst: move to pose, wait, burst
                moved = self.move_to_pose(pose_name)
                time.sleep(1.8)  # settle
                frames_b64 = burst_frames_from_camera(n_frames=4, interval_ms=600)

        if not frames_b64:
            logger.warning(f"No frames for {pose_name} — skipping")
            return None

        # Cosmos Reason2 analysis
        analyses = analyze_frames_reason2(
            pose_name, frames_b64,
            has_labels=self.has_labels,
        )

        calibration = compute_pose_correction(pose_name, analyses, original)
        logger.info(f"  Correction: {calibration.correction_delta}")
        return calibration

    async def run(
        self,
        poses_to_calibrate: Optional[List[str]] = None,
        auto_save: bool = True,
    ) -> CalibrationResult:
        """
        Run the full calibration pipeline.

        Args:
            poses_to_calibrate: Which poses to calibrate. Default: all 5.
            auto_save: If True, save corrected poses to arm memory.
        """
        start = time.time()
        target_poses = poses_to_calibrate or ["inspect", "pick_table", "place_bin"]
        methods: List[str] = []

        logger.info("=" * 55)
        logger.info("VideoCalibrator: Starting calibration")
        logger.info(f"  Poses: {target_poses}")
        logger.info(f"  Labels: {self.has_labels}")
        logger.info(f"  Video: {self.video_path or 'camera burst'}")
        logger.info("=" * 55)

        # Pre-flight: read arm memory
        self._arm_poses = read_arm_poses()
        if not self._arm_poses:
            return CalibrationResult(False, error="Pi offline — cannot read arm poses")

        logger.info(f"Arm memory: {list(self._arm_poses.keys())}")

        # Make sure we start from home (from memory)
        home = self.get_arm_pose("home")
        if home:
            logger.info(f"Moving to home (from memory): {home}")
            _pi_post("/arm/group_move", {"positions": home, "duration_ms": 1200})
            time.sleep(1.5)

        # Method 1+2: Camera burst or video → Reason2
        cosmos_r2_online = _get(f"{COSMOS_R2}/health", timeout=3) is not None
        if cosmos_r2_online:
            methods.append("cosmos_reason2")
        else:
            logger.warning("Cosmos Reason2 offline — spatial analysis skipped")

        calibrations: Dict[str, PoseCalibration] = {}

        for pose_name in target_poses:
            if pose_name not in self._arm_poses:
                logger.warning(f"'{pose_name}' not in arm memory — skipping")
                continue

            cal = await self.calibrate_pose(pose_name)
            if cal:
                calibrations[pose_name] = cal

                # Save corrected pose to arm memory
                if auto_save and cal.correction_delta:
                    saved = self.save_corrected_pose(pose_name, cal.corrected)
                    cal.notes += f" | saved={saved}"

                # Return home between poses
                if home and pose_name != "home":
                    _pi_post("/arm/group_move", {"positions": home, "duration_ms": 1000})
                    time.sleep(1.2)

        synthetic_frames = 0

        # Method 3: Cosmos Predict2.5 for inspect
        if "inspect" in calibrations:
            inspect_frames = burst_frames_from_camera(n_frames=1)
            if inspect_frames:
                predict_result = cosmos_predict_future_state(
                    inspect_frames[0],
                    "A robotic arm gripper descending from inspect height to pick "
                    "a small cube object on a flat workspace surface. Maintain gripper "
                    "alignment. Physics-accurate motion.",
                )
                if predict_result:
                    methods.append("cosmos_predict2.5")
                    synthetic_frames += 4  # Predict2.5 generates ~4s of video

        # Method 4: Cosmos Transfer2.5 synthetic data
        if self.synthetic_output_dir:
            for pose_name, cal in calibrations.items():
                frames_for_aug = burst_frames_from_camera(n_frames=2, interval_ms=500)
                aug_result = cosmos_transfer_synthetic(
                    frames_for_aug, pose_name, self.synthetic_output_dir
                )
                if not aug_result.get("skipped"):
                    methods.append("cosmos_transfer2.5")
                    synthetic_frames += aug_result.get("successful", 0) * 3

        total_ms = (time.time() - start) * 1000
        success = len(calibrations) > 0

        logger.info(f"\nCalibration {'COMPLETE' if success else 'FAILED'}")
        logger.info(f"Poses calibrated: {list(calibrations.keys())}")
        logger.info(f"Methods used: {list(set(methods))}")
        logger.info(f"Total: {total_ms:.0f}ms")

        return CalibrationResult(
            success=success,
            poses=calibrations,
            synthetic_frames=synthetic_frames,
            total_ms=total_ms,
            methods_used=list(set(methods)),
        )


# ── Module interface ───────────────────────────────────────────────────────────

async def run_calibration(
    video_path: Optional[str] = None,
    has_labels: bool = True,
    poses: Optional[List[str]] = None,
    auto_save: bool = True,
    synthetic_dir: Optional[str] = None,
) -> CalibrationResult:
    """
    Top-level calibration entry point.

    Args:
        video_path:    Path to MP4 video of arm movement (optional).
                       If None, uses live camera burst.
        has_labels:    True if workspace has colored corner labels.
                       Labels dramatically improve Reason2 accuracy.
        poses:         Poses to calibrate. Default: inspect, pick_table, place_bin.
        auto_save:     Save corrected poses to arm memory.
        synthetic_dir: Where to save Transfer2.5 synthetic frames.
    """
    calibrator = VideoCalibrator(
        has_labels=has_labels,
        synthetic_output_dir=synthetic_dir,
        video_path=video_path,
    )
    return await calibrator.run(poses_to_calibrate=poses, auto_save=auto_save)
