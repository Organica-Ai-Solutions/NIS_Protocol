"""
NIS Protocol v4.0 - Live YOLO Vision Route
===========================================
Real-time object detection using YOLOv8 on the Pi Camera.
Auto-downloads YOLOv8n weights on first run.
Feeds labeled detections directly into Cosmos Reason2 prompts.

Endpoints:
  GET  /yolo/detect          — Snap + detect, return annotated image + detections
  GET  /yolo/stream-frame    — Single annotated JPEG (for live UI polling)
  POST /yolo/reason          — Snap + detect + ask Cosmos R2 about the scene
  GET  /yolo/status          — YOLO model status + last detection summary
  POST /yolo/cookoff-scan    — Full cookoff scan: YOLO → R2 → action plan
"""

import asyncio
import base64
import io
import logging
import threading
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.yolo_vision")

router = APIRouter(prefix="/yolo", tags=["YOLO Vision"])

AGENT_URL       = os.environ.get("AGENT_URL",       "http://localhost:8085")
H100_REASON_URL = os.environ.get("H100_REASON_URL", "http://localhost:8100")
H100_GDINO_URL  = os.environ.get("H100_GDINO_URL",  "http://localhost:8400")

# ── Model state (module-level singleton) ─────────────────────────────────────
# yolov8n = fastest, least accurate. yolov8s = ~2× slower, better accuracy. yolov8m = best, slower.
_yolo_model       = None
_yolo_model_name  = os.environ.get("YOLO_MODEL", "yolov8x.pt")
_yolo_ready       = False
_yolo_error       = ""
_last_detections: List[Dict] = []

# ── YOLO-World singleton (open-vocabulary, detects "lighter" directly) ─────────
_yolo_world_model  = None
_yolo_world_ready  = False
_yolo_world_error  = ""
_yolo_world_lock   = threading.Lock()  # thread-safe set_classes
# Prefer large model for better accuracy; fall back to small if not downloaded yet
import os as _os
_YOLO_WORLD_PATH   = os.environ.get(
    "YOLO_WORLD_MODEL",
    "/home/nvidia/yolov8l-worldv2.pt"
    if _os.path.exists("/home/nvidia/yolov8l-worldv2.pt")
    else "/home/nvidia/yolov8s-worldv2.pt"
)
# Broad robot-task classes — covers most pick-and-place scenarios.
# Loaded once at startup (no dynamic set_classes to avoid CUDA issues).
# GDINO handles truly custom per-request objects.
_YOLO_WORLD_CLASSES = [
    # Pick targets
    "lighter", "pen", "marker", "pencil", "key", "bolt", "nut", "screw",
    "tool", "screwdriver", "wrench", "pliers", "hammer",
    "block", "cube", "toy", "ball", "dice",
    "card", "chip", "coin", "token",
    "phone", "remote", "usb", "cable",
    # Place targets
    "bin", "bowl", "box", "tray", "basket", "container", "bucket",
    "cup", "mug", "glass", "bottle", "vase",
    "plate", "dish",
    # Scene context
    "table", "shelf", "surface",
    "object", "item", "thing",
]
_last_frame_b64   = ""
_last_annotated_b64 = ""
_last_detect_ts   = 0.0

# ── ByteTrack singleton (supervision) ─────────────────────────────────────────
_tracker           = None    # sv.ByteTracker instance, created lazily
_tracker_error     = ""
_smooth: Dict[int, Dict] = {}          # track_id → EMA-smoothed box dict
_EMA_ALPHA         = 0.5               # smoothing factor (0=frozen, 1=no smoothing)

def _get_tracker():
    """
    Lazy-init ByteTracker/ByteTrack — handles both old and new supervision API.
    supervision >= 0.19  → sv.ByteTrack (new name)
    supervision < 0.19   → sv.ByteTracker (old name)
    """
    global _tracker, _tracker_error
    if _tracker is not None:
        return _tracker
    try:
        import supervision as sv
        if hasattr(sv, "ByteTrack"):
            # supervision >= 0.19 API
            _tracker = sv.ByteTrack(
                track_activation_threshold = 0.35,
                lost_track_buffer          = 30,
                minimum_matching_threshold = 0.8,
                frame_rate                 = 2,
            )
            logger.info("ByteTrack (sv>=0.19) ready")
        elif hasattr(sv, "ByteTracker"):
            # supervision < 0.19 API
            _tracker = sv.ByteTracker(
                track_thresh  = 0.35,
                match_thresh  = 0.8,
                track_buffer  = 30,
                frame_rate    = 2,
            )
            logger.info("ByteTracker (sv<0.19) ready")
        else:
            raise AttributeError("supervision has no ByteTrack or ByteTracker")
    except Exception as e:
        _tracker_error = str(e)
        logger.warning("ByteTracker unavailable: %s", e)
        _tracker = False   # sentinel: tried but failed
    return _tracker



def _load_yolo_world() -> bool:
    """Load YOLO-World open-vocab model (yolov8s-worldv2.pt)."""
    global _yolo_world_model, _yolo_world_ready, _yolo_world_error
    if _yolo_world_ready:
        return True
    try:
        from ultralytics import YOLO
        import numpy as np
        if not os.path.exists(_YOLO_WORLD_PATH):
            _yolo_world_error = f"weights not found: {_YOLO_WORLD_PATH}"
            logger.warning("YOLO-World: %s", _yolo_world_error)
            return False
        _yolo_world_model = YOLO(_YOLO_WORLD_PATH)
        _yolo_world_model.to("cuda:0")  # load on GPU before set_classes
        _yolo_world_model.set_classes(_YOLO_WORLD_CLASSES)
        _yolo_world_model.to("cuda:0")  # ensure text embeds on GPU too
        blank = np.zeros((320, 320, 3), dtype="uint8")
        _yolo_world_model.predict(blank, verbose=False, conf=0.10)
        _yolo_world_ready = True
        _yolo_world_error = ""
        logger.info("YOLO-World ready — classes=%s", _YOLO_WORLD_CLASSES)
        return True
    except Exception as e:
        _yolo_world_error = str(e)
        logger.warning("YOLO-World load failed: %s", e)
        return False


async def _ensure_yolo_world():
    """Async wrapper to load YOLO-World in thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_yolo_world)

# ── GDINO availability cache ──────────────────────────────────────────────────
_gdino_last_ok   = False
_gdino_last_ping = 0.0
_GDINO_PING_TTL  = 60.0   # re-check every 60 s

# ── Stream rate limit (2fps cap) ──────────────────────────────────────────────
_last_stream_ts    = 0.0
STREAM_MIN_INTERVAL = 0.5   # seconds (= 2 fps max)

# COCO class names (80 classes) — for reference / filtering
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","lighter","box","bin","cube","block","marker"
]

# Cookoff-specific classes we care most about
COOKOFF_TARGETS = {
    "bottle", "cup", "bowl", "vase", "scissors", "knife", "spoon", "fork",
    "remote", "cell phone", "book", "clock", "mouse", "keyboard",
    "chair", "dining table", "laptop", "backpack", "handbag",
    # custom / non-COCO that we add via conf threshold tuning
    "lighter", "box", "bin", "cube", "block",
}

# COCO label → cookoff-friendly alias (YOLOv8n never predicts 'lighter'/'bin' natively)
# We relabel nearest COCO detections so the UI shows meaningful names
_COCO_ALIAS: Dict[str, str] = {
    # YOLOv8n sees these — we relabel for the cookoff UI
    "bottle":       "lighter",   # most likely object on table is the lighter
    "vase":         "lighter",
    "cup":          "cup",
    "bowl":         "bin",
    "sports ball":  "object",
    "dining table": "table",
    "cell phone":   "remote",
    # Food misclassifications of small cylinders/tubes → remap to useful names
    "hot dog":      "lighter",
    "banana":       "lighter",
    "carrot":       "lighter",
    "syringe":      "lighter",
}
_ALIAS_MIN_CONF = 0.20  # Apply alias at lower confidence too — catches dim/partial lighters

# For detect: always include these COCO classes even without target filter
_ALWAYS_DETECT = {
    "bottle", "cup", "vase", "bowl", "scissors", "knife", "spoon",
    "remote", "cell phone", "book", "laptop", "mouse", "keyboard",
    "chair", "person", "backpack", "handbag", "clock",
}

# HARD REJECT: COCO classes that should never appear in a robotics table scene.
# Reject these regardless of confidence unless they're explicitly in the targets list.
_REJECT_CLASSES = {
    # Animals
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
    # Outdoor / vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    # Sports
    "frisbee", "skis", "snowboard", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    # Food (table objects get misclassified as food — these are never actual food)
    "pizza", "donut", "cake", "sandwich", "apple", "orange",
    "broccoli",
    # Large furniture
    "bed", "couch", "toilet", "bench",
}


def _iou(a: Dict, b: Dict) -> float:
    """Intersection over union of two boxes with x1,y1,x2,y2."""
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_yolo():
    """Load YOLOv8 — auto-downloads weights on first call."""
    global _yolo_model, _yolo_ready, _yolo_error, _yolo_model_name
    if _yolo_ready:
        return True
    try:
        from ultralytics import YOLO
        # Model search order: local path → auto-download from ultralytics CDN
        model_paths = [
            f"/opt/nis-protocol/models/{_yolo_model_name}",
            f"/opt/neurolinux/models/{_yolo_model_name}",
            f"/home/neurolinux/.config/Ultralytics/{_yolo_model_name}",
            f"/tmp/{_yolo_model_name}",
            _yolo_model_name,  # ultralytics will auto-download to ~/.config/Ultralytics/
        ]
        loaded = False
        for path in model_paths:
            if os.path.exists(path):
                logger.info("YOLO: loading from %s", path)
                _yolo_model = YOLO(path)
                loaded = True
                break
        if not loaded:
            logger.info("YOLO: auto-downloading %s ...", _yolo_model_name)
            _yolo_model = YOLO(_yolo_model_name)  # triggers download
        # Warm up on a blank frame
        import numpy as np
        blank = np.zeros((320, 320, 3), dtype="uint8")
        _yolo_model.predict(blank, verbose=False, conf=0.25)
        _yolo_ready = True
        _yolo_error = ""
        logger.info("YOLO: ready — model=%s", _yolo_model_name)
        return True
    except Exception as e:
        _yolo_error = str(e)
        logger.error("YOLO: load failed: %s", e)
        return False


async def _ensure_yolo():
    """Non-blocking load — runs in thread pool so FastAPI stays responsive."""
    if _yolo_ready:
        return True
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_yolo)


async def _get_camera_frame() -> Optional[str]:
    """Grab raw JPEG base64 from Agent :8085 /camera/snapshot, or directly from /dev/video0."""
    # Try agent first
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{AGENT_URL}/camera/snapshot")
            if r.status_code == 200:
                img = r.json().get("image_base64", "")
                if img:
                    return img
    except Exception as e:
        logger.warning("Camera snapshot (agent) failed: %s", e)
    # Fallback: capture directly from /dev/video0
    try:
        import cv2, base64 as _b64
        def _grab():
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return _b64.b64encode(buf.tobytes()).decode()
        frame_b64 = await asyncio.get_event_loop().run_in_executor(None, _grab)
        if frame_b64:
            return frame_b64
    except Exception as e:
        logger.warning("Camera direct V4L2 capture failed: %s", e)
    return None


async def _gdino_detect(frame_b64: str, prompt: str = "lighter . bin .") -> List[Dict]:
    """
    Call H100 Grounding DINO server at H100_GDINO_URL/detect.
    Returns detections list on success, [] on any failure (non-blocking best-effort).
    Updates _gdino_last_ok availability cache.
    """
    global _gdino_last_ok, _gdino_last_ping
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=1.5, read=6.0, write=2.0, pool=2.0)
        ) as c:
            r = await c.post(
                f"{H100_GDINO_URL}/detect",
                json={
                    "image_base64": frame_b64,
                    "text_prompts": prompt,
                    "box_threshold": 0.22,
                    "text_threshold": 0.18,
                },
            )
            if r.status_code == 200:
                _gdino_last_ok   = True
                _gdino_last_ping = time.time()
                return r.json().get("detections", [])
    except Exception as e:
        logger.debug("GDINO call failed (non-critical): %s", e)
    _gdino_last_ok   = False
    _gdino_last_ping = time.time()
    return []


def _nms_merge(primary: List[Dict], extra: List[Dict], iou_thresh: float = 0.4) -> List[Dict]:
    """
    Merge extra detections into primary list, dropping any extra box that overlaps
    an existing primary box by more than iou_thresh. Keeps primary boxes as-is.
    """
    merged = list(primary)
    for e in extra:
        overlap = any(_iou(e, p) > iou_thresh for p in merged)
        if not overlap:
            merged.append(e)
    merged.sort(key=lambda d: d["conf"], reverse=True)
    return merged


def _apply_bytetrack(detections: List[Dict]) -> List[Dict]:
    """
    Run ByteTracker on current detections, assign track_id, apply EMA smoothing.
    Falls back gracefully if supervision is not installed.
    """
    global _smooth
    tracker = _get_tracker()
    if not tracker:
        return detections   # supervision unavailable — return untracked

    try:
        import supervision as sv
        if not detections:
            # Feed empty frame to keep tracker state advancing
            empty = sv.Detections.empty()
            tracker.update_with_detections(empty)
            _smooth.clear()
            return detections

        xyxy  = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=float)
        confs = np.array([d["conf"] for d in detections], dtype=float)
        sv_dets = sv.Detections(xyxy=xyxy, confidence=confs)

        # update_with_detections works for both ByteTrack and ByteTracker
        if hasattr(tracker, "update_with_detections"):
            tracked = tracker.update_with_detections(sv_dets)
        else:
            tracked = tracker.update(sv_dets, [])

        # Map tracked entries back to original detections by index
        tracked_ids  = (tracked.tracker_id if tracked.tracker_id is not None else [])
        for i, det in enumerate(detections):
            tid = int(tracked_ids[i]) if i < len(tracked_ids) else -1
            det["track_id"] = tid
            if tid < 0:
                continue
            prev = _smooth.get(tid)
            if prev is None:
                _smooth[tid] = dict(det)
                continue
            # EMA smooth pixel coords
            a = _EMA_ALPHA
            det["x1"] = int(prev["x1"] * (1 - a) + det["x1"] * a)
            det["y1"] = int(prev["y1"] * (1 - a) + det["y1"] * a)
            det["x2"] = int(prev["x2"] * (1 - a) + det["x2"] * a)
            det["y2"] = int(prev["y2"] * (1 - a) + det["y2"] * a)
            det["cx"] = (det["x1"] + det["x2"]) // 2
            det["cy"] = (det["y1"] + det["y2"]) // 2
            det["w"]  = det["x2"] - det["x1"]
            det["h"]  = det["y2"] - det["y1"]
            _smooth[tid] = dict(det)
    except Exception as e:
        logger.debug("ByteTrack step failed: %s", e)

    return detections


def _classify_bbox_color(np_frame: "np.ndarray", x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Classify the dominant color of a bounding-box region using HSV histogramming.
    Returns: 'red'|'orange'|'yellow'|'green'|'blue'|'purple'|'' (empty = neutral/unknown)
    Requires OpenCV — fails silently if unavailable.
    """
    try:
        import cv2
        roi = np_frame[max(0, y1):max(y2, y1 + 2), max(0, x1):max(x2, x1 + 2)]
        if roi.size < 60:
            return ""
        bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].flatten()   # OpenCV: H in [0,179]
        s_ch = hsv[:, :, 1].flatten()   # S in [0,255]
        v_ch = hsv[:, :, 2].flatten()   # V in [0,255]
        # Only count saturated, non-dark pixels
        mask = (s_ch > 55) & (v_ch > 45)
        if mask.sum() < 20:
            return ""    # mostly grey / black / white
        h = h_ch[mask]
        n = float(len(h))

        def _pct(lo: int, hi: int) -> float:
            return float(((h >= lo) & (h <= hi)).sum()) / n

        scores = {
            "red":    _pct(0, 8)   + _pct(170, 179),
            "orange": _pct(9, 18),
            "yellow": _pct(19, 35),
            "green":  _pct(36, 85),
            "blue":   _pct(86, 130),
            "purple": _pct(131, 169),
        }
        best_color, best_score = max(scores.items(), key=lambda kv: kv[1])
        return best_color if best_score > 0.30 else ""
    except Exception:
        return ""


# Labels for which we prepend the color to create "blue_lighter", "yellow_cup", etc.
_COLOR_LABEL_TARGETS = {
    "lighter", "bottle", "vase", "cup", "box", "object",
    "bin", "cube", "block", "bowl", "marker",
}


def _run_yolo(frame_b64: str, conf: float = 0.10, targets: Optional[List[str]] = None) -> Dict:
    """
    Run YOLOv8 on a base64 JPEG frame.
    Returns:
      detections      — list of {label, color, conf, cx, cy, x1,y1,x2,y2, w, h}
      annotated_b64   — JPEG with bounding boxes + labels drawn
      raw_b64         — original frame (unchanged)
      scene_context   — human-readable string for R2 prompts
    """
    global _last_detections, _last_frame_b64, _last_annotated_b64, _last_detect_ts

    if not _yolo_ready or _yolo_model is None:
        return {
            "detections": [], "annotated_b64": frame_b64, "raw_b64": frame_b64,
            "scene_context": "YOLO not ready", "n": 0,
        }

    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        # Decode frame
        img_bytes = base64.b64decode(frame_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame_w, frame_h = pil_img.size
        np_frame = np.array(pil_img)

        imgsz = int(os.environ.get("YOLO_IMGSZ", "640"))  # 640=fast, 960/1280=better small-object detection
        results = _yolo_model.predict(
            np_frame,
            conf=conf,
            verbose=False,
            imgsz=min(1280, max(320, imgsz)),
        )

        detections: List[Dict] = []
        for res in results:
            for box in (res.boxes or []):
                cls_id = int(box.cls[0])
                label  = _yolo_model.names.get(cls_id, f"cls{cls_id}")
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                w  = int(x2 - x1)
                h  = int(y2 - y1)

                label_lc = label.lower()
                targets_set = {t.lower() for t in targets} if targets else set()

                # Hard reject: food/animal/outdoor classes never appear on robotics table
                # Exception: if user explicitly lists it in targets (e.g. "apple" for demo)
                if label_lc in _REJECT_CLASSES and label_lc not in targets_set:
                    continue

                # Check alias early — if this label maps to a valid target, treat as in_targets
                alias = _COCO_ALIAS.get(label_lc)
                effective_label_lc = alias if (alias and (not targets_set or alias in targets_set)) else label_lc

                in_always  = effective_label_lc in _ALWAYS_DETECT or label_lc in _ALWAYS_DETECT
                in_targets = not targets_set or effective_label_lc in targets_set or label_lc in targets_set

                if not in_always and not in_targets:
                    # Off-target, off-always: require very high confidence
                    if confidence < 0.75:
                        continue

                # Remap COCO label to cookoff-friendly alias only when conf is decent
                display_label = label
                if targets and confidence >= _ALIAS_MIN_CONF:
                    target_lc = [t.lower() for t in targets]
                    alias = _COCO_ALIAS.get(label_lc)
                    if alias and alias in target_lc and label_lc not in target_lc:
                        display_label = alias

                # ── Color classification — makes "blue_lighter", "yellow_cup" etc. ──
                det_color = _classify_bbox_color(np_frame, int(x1), int(y1), int(x2), int(y2))
                base_noun = display_label.lower().split("_")[-1]  # strip any prior color prefix
                if det_color and base_noun in _COLOR_LABEL_TARGETS:
                    display_label = f"{det_color}_{base_noun}"

                detections.append({"_raw_label": label, "label": display_label,
                    "color":  det_color,
                    "conf":   round(confidence, 3),
                    "cx":     cx,
                    "cy":     cy,
                    "x1":     int(x1), "y1": int(y1),
                    "x2":     int(x2), "y2": int(y2),
                    "w":      w,
                    "h":      h,
                    "pct_x":  round(cx / frame_w * 100, 1),
                    "pct_y":  round(cy / frame_h * 100, 1),
                })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["conf"], reverse=True)

        # ── ByteTrack: assign track IDs + EMA smooth ─────────────────────────
        detections = _apply_bytetrack(detections)

        # ── Color-blob fallback: if YOLO found nothing, use OpenCV contours ──
        # Catches lighters, cups, small objects YOLO misses on a plain table.
        if not detections:
            try:
                import cv2
                gray = cv2.cvtColor(np_frame, cv2.COLOR_RGB2GRAY)
                # Adaptive threshold to find salient blobs
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 31, 10
                )
                # Morphological cleanup
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                blob_dets = []
                frame_area = frame_w * frame_h
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    min_area = frame_area * 0.001
                    max_area = frame_area * 0.35
                    if area < min_area or area > max_area:
                        continue
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cx_b = bx + bw // 2
                    cy_b = by + bh // 2
                    pct_area = area / frame_area
                    conf_b = round(min(0.40, pct_area * 15), 2)
                    if conf_b < 0.05:
                        continue
                    # Heuristic labels by aspect & size (improves R2 context)
                    ar = bh / bw if bw > 0 else 1
                    if ar > 1.3 and pct_area < 0.02:
                        blob_label = "lighter_candidate"  # elongated, small
                    elif ar < 0.7 and pct_area > 0.05:
                        blob_label = "bin_candidate"     # wide, larger
                    elif pct_area > 0.08:
                        blob_label = "table_region"
                    else:
                        blob_label = "object"
                    blob_dets.append({
                        "label":  blob_label,
                        "conf":   conf_b,
                        "cx":     cx_b, "cy": cy_b,
                        "x1": bx, "y1": by,
                        "x2": bx + bw, "y2": by + bh,
                        "w": bw, "h": bh,
                        "pct_x": round(cx_b / frame_w * 100, 1),
                        "pct_y": round(cy_b / frame_h * 100, 1),
                    })
                blob_dets.sort(key=lambda d: d["conf"], reverse=True)
                # NMS: drop overlapping boxes (keep higher conf)
                kept = []
                for d in blob_dets:
                    overlap = any(
                        _iou(d, k) > 0.4 for k in kept
                    )
                    if not overlap:
                        kept.append(d)
                detections = kept[:5]
                if detections:
                    logger.info("Color-blob fallback: %d objects found", len(detections))
            except Exception as _cv_err:
                logger.debug("Color-blob fallback unavailable: %s", _cv_err)

        # Draw annotations on a copy
        ann_img = pil_img.copy()
        draw = ImageDraw.Draw(ann_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
            font_sm = font

        colors = [
            "#FF3B30","#FF9500","#FFCC00","#34C759","#5AC8FA",
            "#007AFF","#5856D6","#FF2D55","#AF52DE","#00C7BE",
        ]
        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            # Box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # Label background (include track ID when available)
            tid_str = f" #{det['track_id']}" if det.get("track_id", -1) >= 0 else ""
            src_tag = " [G]" if det.get("source") == "gdino" else ""
            label_txt = f"{det['label']}{src_tag}{tid_str} {det['conf']:.0%}"
            bbox = draw.textbbox((x1, y1 - 18), label_txt, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
            draw.text((x1, y1 - 18), label_txt, fill="white", font=font)
            # Center dot
            draw.ellipse([det["cx"]-3, det["cy"]-3, det["cx"]+3, det["cy"]+3], fill=color)

        # Add HUD: object count + timestamp + tracker state
        tracker_tag = "" if _tracker is None else " | ByteTrack"
        gdino_tag   = " | GDINO✓" if _gdino_last_ok else ""
        hud = f"YOLO  {len(detections)} objects  {time.strftime('%H:%M:%S')}{tracker_tag}{gdino_tag}"
        draw.rectangle([0, 0, frame_w, 22], fill="#000000AA")
        draw.text((6, 4), hud, fill="#00FF88", font=font_sm)

        # Encode annotated
        ann_buf = io.BytesIO()
        ann_img.save(ann_buf, format="JPEG", quality=85)
        annotated_b64 = base64.b64encode(ann_buf.getvalue()).decode()

        # Build scene_context string for R2 prompts
        if detections:
            parts = []
            for d in detections[:10]:
                pos = "left" if d["pct_x"] < 33 else ("center" if d["pct_x"] < 66 else "right")
                depth = "close" if d["pct_y"] > 66 else ("mid" if d["pct_y"] > 33 else "far")
                parts.append(
                    f"{d['label']}({d['conf']:.0%}) at [{d['cx']},{d['cy']}] "
                    f"({pos},{depth}) size={d['w']}x{d['h']}px"
                )
            scene_context = "YOLO detects: " + "; ".join(parts)
        else:
            scene_context = "YOLO: no objects detected above confidence threshold"

        _last_detections   = detections
        _last_frame_b64    = frame_b64
        _last_annotated_b64 = annotated_b64
        # Clean up internal _raw_label field
        for d in detections:
            d.pop("_raw_label", None)

        _last_detect_ts    = time.time()

        return {
            "detections":    detections,
            "annotated_b64": annotated_b64,
            "raw_b64":       frame_b64,
            "scene_context": scene_context,
            "n":             len(detections),
            "frame_w":       frame_w,
            "frame_h":       frame_h,
        }

    except Exception as e:
        logger.error("YOLO inference failed: %s", e)
        return {
            "detections": [], "annotated_b64": frame_b64, "raw_b64": frame_b64,
            "scene_context": f"YOLO error: {e}", "n": 0,
        }


# ── Request models ────────────────────────────────────────────────────────────

class ReasonRequest(BaseModel):
    query:   str = Field(default="What objects do you see? What should the arm do next?")
    targets: List[str] = Field(default_factory=list)
    conf:    float = Field(default=0.25)


class CookoffScanRequest(BaseModel):
    task:        str   = Field(default="Pick up the lighter and place it on the left")
    conf:        float = Field(default=0.20)
    execute_arm: bool  = Field(default=False)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def yolo_status():
    """YOLO model status + last detection summary."""
    tracker = _get_tracker()
    return {
        "ready":           _yolo_ready,
        "model":           _yolo_model_name,
        "error":           _yolo_error,
        "n_last":          len(_last_detections),
        "last_ts":         _last_detect_ts,
        "last_scene":      _last_detections[:5],
        "agent_url":       AGENT_URL,
        "h100_url":        H100_REASON_URL,
        "gdino_available": _gdino_last_ok,
        "gdino_url":       H100_GDINO_URL,
        "bytetrack":       bool(tracker),
        "bytetrack_error": _tracker_error,
    }



def _run_yolo_world(frame_b64: str, conf: float = 0.08,
                    targets: List[str] = None) -> Dict:
    """
    Run YOLO-World open-vocabulary inference on a base64 frame.
    targets: list of text classes to detect — YOLO-World will look for these
             specific objects. If None, uses default _YOLO_WORLD_CLASSES.
    Returns detections list (same format as _run_yolo).
    """
    import numpy as np
    from PIL import Image
    import io, base64

    if not _yolo_world_ready or _yolo_world_model is None:
        return {"detections": [], "n": 0}
    try:
        # Use pre-loaded classes (set at startup on GPU, no per-request set_classes)
        # GDINO handles truly custom per-request objects
        active_classes = _YOLO_WORLD_CLASSES

        img_bytes = base64.b64decode(frame_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_frame  = np.array(pil_img)
        results   = _yolo_world_model.predict(
            np_frame, conf=max(0.05, conf * 0.8), verbose=False,
            imgsz=640, device=0
        )
        detections = []
        for res in results:
            for box in (res.boxes or []):
                wcls   = int(box.cls[0])
                wconf  = float(box.conf[0])
                wlabel = active_classes[wcls] if wcls < len(active_classes) else "object"
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                detections.append({
                    "label":  wlabel,
                    "conf":   wconf,
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "w": int(x2-x1), "h": int(y2-y1),
                    "source": "yolo_world",
                    "color": "",
                })
        return {"detections": detections, "n": len(detections)}
    except Exception as e:
        logger.warning("_run_yolo_world error: %s", e)
        return {"detections": [], "n": 0}

@router.get("/detect")
async def yolo_detect(
    targets: str = "lighter,bottle,cup,box,bin,person,chair",
    conf:    float = 0.10,
):
    """
    Snap frame → run YOLO → return annotated image + structured detections.
    targets: comma-separated label hints (all COCO classes always run, targets just shown first)
    conf: minimum confidence threshold (default 0.20 — lower = more detections)
    """
    t0 = time.time()

    # Ensure YOLO + YOLO-World are loaded
    ready = await _ensure_yolo()
    if not ready:
        return {"ok": False, "error": _yolo_error, "detections": [], "scene_context": "YOLO unavailable"}
    await _ensure_yolo_world()  # non-blocking lazy load

    # Get camera frame
    frame_b64 = await _get_camera_frame()
    if not frame_b64:
        return {"ok": False, "error": "Camera unavailable", "detections": [], "scene_context": "No camera frame"}

    # Run YOLO + GDINO concurrently (GDINO is best-effort, won't block on failure)
    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    gdino_prompt = " . ".join(target_list) + " ." if target_list else "lighter . bin ."
    loop = asyncio.get_event_loop()
    yolo_task  = loop.run_in_executor(None, _run_yolo, frame_b64, conf, target_list)
    gdino_task = _gdino_detect(frame_b64, gdino_prompt)
    world_task = loop.run_in_executor(None, _run_yolo_world, frame_b64, conf, target_list)
    result, gdino_dets, world_result = await asyncio.gather(
        yolo_task, gdino_task, world_task, return_exceptions=False
    )

    # Merge GDINO hits (NMS dedup against YOLO boxes)
    if gdino_dets:
        result["detections"] = _nms_merge(result["detections"], gdino_dets)
        result["n"]          = len(result["detections"])
        logger.info("GDINO merged %d extra detections", len(gdino_dets))

    # Merge YOLO-World hits (detects "lighter" directly, not "bottle" alias)
    if isinstance(world_result, dict):
        world_dets = world_result.get("detections", [])
        if world_dets:
            result["detections"] = _nms_merge(result["detections"], world_dets)
            result["n"]          = len(result["detections"])
            logger.info("YOLO-World merged %d detections", len(world_dets))

    # ── Fire autonomy event if high-conf detections found ────────────────────
    # Non-blocking — autonomy engine decides whether to queue a task.
    if result["detections"]:
        try:
            from routes.autonomy import get_engine as _get_autonomy
            asyncio.ensure_future(_get_autonomy().ingest_event("yolo_detection", {
                "detections": result["detections"][:8],
                "scene":      result["scene_context"],
            }))
        except Exception:
            pass

    n_gdino = len([d for d in result["detections"] if d.get("source") == "gdino"])
    return {
        "ok":              True,
        "yolo_available":  _yolo_ready,
        "gdino_available": _gdino_last_ok,
        "detections":      result["detections"],
        "annotated_b64":   result["annotated_b64"],
        "raw_b64":         result["raw_b64"],
        "scene_context":   result["scene_context"],
        "n":               result["n"],
        "n_yolo":          result["n"] - n_gdino,
        "n_gdino":         n_gdino,
        "latency_ms":      round((time.time() - t0) * 1000),
        "model":           _yolo_model_name,
        "frame_w":         result.get("frame_w", 0),
        "frame_h":         result.get("frame_h", 0),
    }


@router.get("/stream-frame")
async def yolo_stream_frame(conf: float = 0.20):
    """
    Returns a raw annotated JPEG (Content-Type: image/jpeg).
    Rate-capped at 2fps: repeated polls within 0.5s return the cached last frame.
    """
    global _last_stream_ts

    now = time.time()
    if now - _last_stream_ts < STREAM_MIN_INTERVAL and _last_annotated_b64:
        # Return cached frame without re-running inference
        return Response(content=base64.b64decode(_last_annotated_b64), media_type="image/jpeg")

    _last_stream_ts = now

    ready = await _ensure_yolo()
    frame_b64 = await _get_camera_frame()

    if not ready or not frame_b64:
        # Return a blank grey JPEG
        from PIL import Image
        img = Image.new("RGB", (640, 360), color=(30, 30, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return Response(content=buf.getvalue(), media_type="image/jpeg")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_yolo, frame_b64, conf, None)
    img_bytes = base64.b64decode(result["annotated_b64"])
    return Response(content=img_bytes, media_type="image/jpeg")


@router.post("/reason")
async def yolo_reason(request: ReasonRequest):
    """
    Snap → YOLO → inject scene labels into Cosmos Reason2 prompt → return R2 analysis.
    This is the full agentic vision pipeline:
      Camera → YOLO labels all objects with pixel coords → R2 reasons about the scene.
    """
    t0 = time.time()

    ready = await _ensure_yolo()
    frame_b64 = await _get_camera_frame()

    detections: List[Dict] = []
    scene_context = "No camera frame available"
    annotated_b64 = ""
    raw_b64 = ""

    if ready and frame_b64:
        loop = asyncio.get_event_loop()
        det_result = await loop.run_in_executor(
            None, _run_yolo, frame_b64, request.conf, request.targets or None
        )
        detections    = det_result["detections"]
        scene_context = det_result["scene_context"]
        annotated_b64 = det_result["annotated_b64"]
        raw_b64       = det_result["raw_b64"]

    # Build enriched prompt for Cosmos Reason2
    r2_prompt = (
        f"You are the vision system for an xArm 6-DOF robot on a wooden table. "
        f"Camera resolution: 1280x720 (x=0 left, y=0 top). "
        f"{scene_context}. "
        f"User query: {request.query}"
    )

    r2_response = ""
    r2_confidence = 0.0
    h100_ok = False

    try:
        reason_body: Dict[str, Any] = {
            "query":      r2_prompt,
            "max_tokens": 300,
            "use_think":  True,
        }
        if annotated_b64:
            reason_body["image_base64"] = annotated_b64
        elif raw_b64:
            reason_body["image_base64"] = raw_b64

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=15.0, write=3.0, pool=3.0)
        ) as c:
            rr = await c.post(f"{H100_REASON_URL}/reason", json=reason_body)
            if rr.status_code == 200:
                rd = rr.json()
                r2_response   = rd.get("response", rd.get("reasoning", rd.get("answer", "")))
                r2_confidence = rd.get("confidence", 0.9)
                h100_ok = True
    except Exception as e:
        logger.warning("R2 reason call failed: %s", e)
        r2_response = f"Cosmos Reason2 unavailable: {e}. YOLO scene: {scene_context}"

    return {
        "ok":           True,
        "detections":   detections,
        "n_objects":    len(detections),
        "scene_context": scene_context,
        "annotated_b64": annotated_b64,
        "r2_response":  r2_response,
        "r2_confidence": r2_confidence,
        "h100_ok":      h100_ok,
        "latency_ms":   round((time.time() - t0) * 1000),
        "model":        _yolo_model_name,
        "timestamp":    time.time(),
    }


@router.post("/cookoff-scan")
async def yolo_cookoff_scan(request: CookoffScanRequest):
    """
    Full cookoff agentic pipeline:
      1. Camera snapshot
      2. YOLO detects + labels ALL objects with pixel positions
      3. Annotated frame + scene labels → Cosmos Reason2 /robot-plan
      4. R2 returns action plan grounded in what YOLO actually sees
      5. Optionally execute on xArm

    This is the cookoff demo entry point that combines:
      - Real-time YOLO object detection (all 80 COCO classes + custom)
      - Cosmos Reason2 spatial reasoning with visual grounding
      - xArm execution of the R2-generated plan
    """
    t0 = time.time()
    logs: List[str] = []

    # Step 1: YOLO detection
    ready = await _ensure_yolo()
    frame_b64 = await _get_camera_frame()

    detections: List[Dict] = []
    scene_context = ""
    annotated_b64 = ""

    if ready and frame_b64:
        loop = asyncio.get_event_loop()
        det_result = await loop.run_in_executor(
            None, _run_yolo, frame_b64, request.conf, None
        )
        detections    = det_result["detections"]
        scene_context = det_result["scene_context"]
        annotated_b64 = det_result["annotated_b64"]
        logs.append(f"YOLO: {len(detections)} objects — {scene_context[:120]}")
    else:
        logs.append("YOLO: skipped (not ready or no camera)")

    # Step 2: Cosmos Reason2 robot-plan
    action_plan: List[str] = []
    r2_reasoning = ""
    r2_confidence = 0.0
    plan_source = "local_fallback"

    robot_plan_body: Dict[str, Any] = {
        "command":    request.task,
        "robot_type": "xarm",
        "scene_context": scene_context,
        "yolo_detections": [
            {"label": d["label"], "cx": d["cx"], "cy": d["cy"],
             "conf": d["conf"], "w": d["w"], "h": d["h"]}
            for d in detections[:12]
        ],
    }
    if annotated_b64:
        robot_plan_body["image_base64"] = annotated_b64

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=15.0, write=3.0, pool=3.0)
        ) as c:
            rr = await c.post(f"{H100_REASON_URL}/robot-plan", json=robot_plan_body)
            if rr.status_code == 200:
                rd = rr.json()
                action_plan   = rd.get("action_plan", [])
                r2_reasoning  = rd.get("reasoning", rd.get("response", ""))
                r2_confidence = rd.get("confidence", 0.9)
                plan_source   = "h100_cosmos_reason2"
                logs.append(f"R2 plan: {len(action_plan)} steps — source={plan_source}")
    except Exception as e:
        logs.append(f"R2 /robot-plan failed: {e}")

    # Fallback: derive plan from YOLO + task keywords
    if not action_plan:
        action_plan = _derive_plan_from_yolo(request.task, detections)
        plan_source = "yolo_rule_fallback"
        logs.append(f"Fallback plan: {action_plan}")

    # Step 3: Execute on xArm (if requested)
    exec_results: List[Dict] = []
    steps_ok = 0
    steps_total = len(action_plan)

    if request.execute_arm and action_plan:
        try:
            exec_body = {
                "action_plan": action_plan,
                "execute_arm": True,
                "simulation":  False,
            }
            async with httpx.AsyncClient(timeout=60.0) as c:
                er = await c.post(f"http://localhost:8000/cookoff/execute", json=exec_body)
                if er.status_code == 200:
                    ed = er.json()
                    exec_results = ed.get("results", [])
                    steps_ok     = ed.get("steps_ok", 0)
                    logs.append(f"Execution: {steps_ok}/{steps_total} steps OK")
        except Exception as e:
            logs.append(f"Execution failed: {e}")

    return {
        "ok":            True,
        "task":          request.task,
        "detections":    detections,
        "n_objects":     len(detections),
        "scene_context": scene_context,
        "annotated_b64": annotated_b64,
        "action_plan":   action_plan,
        "r2_reasoning":  r2_reasoning,
        "r2_confidence": r2_confidence,
        "plan_source":   plan_source,
        "steps_ok":      steps_ok,
        "steps_total":   steps_total,
        "exec_results":  exec_results,
        "logs":          logs,
        "latency_ms":    round((time.time() - t0) * 1000),
        "model":         _yolo_model_name,
        "timestamp":     time.time(),
    }


def _derive_plan_from_yolo(task: str, detections: List[Dict]) -> List[str]:
    """Rule-based plan when R2 is unavailable, grounded in YOLO detections."""
    t = task.lower()
    target_obj = None

    # Find most relevant detected object
    priority = ["lighter", "bottle", "cup", "bowl", "box", "remote", "phone", "scissors"]
    for p in priority:
        for d in detections:
            if p in d["label"].lower():
                target_obj = d
                break
        if target_obj:
            break
    if not target_obj and detections:
        target_obj = detections[0]

    if "pick" in t or "grab" in t:
        if target_obj:
            return [
                "Move to home position",
                f"Move arm above {target_obj['label']} at pixel [{target_obj['cx']},{target_obj['cy']}]",
                "Lower arm to pick position",
                "Close gripper",
                "Lift object",
                "Move to place position",
                "Open gripper",
                "Return to home",
            ]
        return ["Move to home", "Scan scene", "Pick object", "Place object", "Return home"]
    elif "wave" in t:
        return ["Wave greeting sequence (3 oscillations)", "Return to home"]
    elif "dance" in t:
        return ["Start dance choreography", "Execute 8 dance moves", "Return to home"]
    else:
        return [
            "Move to home position",
            f"Inspect scene — {len(detections)} objects detected",
            "Execute task based on scene",
            "Return to home",
        ]


# ── Background model preload on startup ──────────────────────────────────────

async def preload_yolo():
    """Called from main_pi.py startup to pre-warm YOLO model."""
    logger.info("YOLO: preloading model in background...")
    await _ensure_yolo()
    await _ensure_yolo_world()
    if _yolo_ready:
        logger.info("YOLO: model ready ✓")
    else:
        logger.warning("YOLO: preload failed — %s", _yolo_error)
