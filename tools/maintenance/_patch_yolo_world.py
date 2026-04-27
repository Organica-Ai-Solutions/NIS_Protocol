#!/usr/bin/env python3
"""
Patch /home/nvidia/routes/yolo_vision.py to add YOLO-World support.
Run on H100: python3 /tmp/_patch_yolo_world.py
"""
import re

PATH = "/home/nvidia/routes/yolo_vision.py"
with open(PATH) as f:
    src = f.read()

# ── 1. Add YOLO-World globals after existing model globals ─────────────────────
OLD_GLOBALS = """_yolo_model       = None
_yolo_model_name  = os.environ.get("YOLO_MODEL", "yolov8x.pt")
_yolo_ready       = False
_yolo_error       = ""
_last_detections: List[Dict] = []"""

NEW_GLOBALS = """_yolo_model       = None
_yolo_model_name  = os.environ.get("YOLO_MODEL", "yolov8x.pt")
_yolo_ready       = False
_yolo_error       = ""
_last_detections: List[Dict] = []

# ── YOLO-World singleton (open-vocabulary, detects "lighter" directly) ─────────
_yolo_world_model  = None
_yolo_world_ready  = False
_yolo_world_error  = ""
_YOLO_WORLD_PATH   = os.environ.get("YOLO_WORLD_MODEL", "/home/nvidia/yolov8s-worldv2.pt")
_YOLO_WORLD_CLASSES = ["lighter", "bin", "bottle", "cup", "box", "object"]"""

if OLD_GLOBALS in src:
    src = src.replace(OLD_GLOBALS, NEW_GLOBALS)
    print("✅ Added YOLO-World globals")
else:
    print("⚠️  Could not find globals anchor — skipping globals patch")

# ── 2. Add _ensure_yolo_world() after _ensure_yolo() ─────────────────────────
YOLO_WORLD_FUNC = '''

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
        _yolo_world_model.set_classes(_YOLO_WORLD_CLASSES)
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

'''

# Insert right before the GDINO availability cache comment
GDINO_ANCHOR = "# ── GDINO availability cache ──"
if GDINO_ANCHOR in src:
    src = src.replace(GDINO_ANCHOR, YOLO_WORLD_FUNC + GDINO_ANCHOR)
    print("✅ Added _load_yolo_world() and _ensure_yolo_world()")
else:
    print("⚠️  Could not find GDINO anchor — appending YOLO-World function at end of globals block")

# ── 3. In _run_yolo(), add YOLO-World detections after standard YOLO results ───
# Find the NMS merge section and add World results before it
OLD_NMS = '''        # ── NMS merge YOLO + GDINO ────────────────────────────────────────────
        all_dets = detections + gdino_dets'''

NEW_NMS = '''        # ── YOLO-World: run open-vocab model if lighter is a target ─────────
        world_dets: List[Dict] = []
        if _yolo_world_ready and _yolo_world_model is not None:
            try:
                world_results = _yolo_world_model.predict(
                    np_frame, conf=max(0.08, conf * 0.7), verbose=False, imgsz=640
                )
                for wres in world_results:
                    for wbox in (wres.boxes or []):
                        wcls   = int(wbox.cls[0])
                        wlabel = (_yolo_world_model.names or {}).get(wcls, "lighter")
                        if wlabel not in _YOLO_WORLD_CLASSES:
                            wlabel = _YOLO_WORLD_CLASSES[wcls] if wcls < len(_YOLO_WORLD_CLASSES) else "object"
                        wconf = float(wbox.conf[0])
                        wx1, wy1, wx2, wy2 = [float(v) for v in wbox.xyxy[0]]
                        wcx = int((wx1 + wx2) / 2)
                        wcy = int((wy1 + wy2) / 2)
                        world_dets.append({
                            "label": wlabel,
                            "conf":  wconf,
                            "cx": wcx, "cy": wcy,
                            "x1": wx1, "y1": wy1, "x2": wx2, "y2": wy2,
                            "w": int(wx2 - wx1), "h": int(wy2 - wy1),
                            "source": "yolo_world",
                            "color": "",
                        })
            except Exception as _we:
                logger.warning("YOLO-World inference failed: %s", _we)

        # ── NMS merge YOLO + GDINO ────────────────────────────────────────────
        all_dets = detections + gdino_dets + world_dets'''

if OLD_NMS in src:
    src = src.replace(OLD_NMS, NEW_NMS)
    print("✅ Added YOLO-World inference in _run_yolo()")
else:
    print("⚠️  Could not find NMS anchor — YOLO-World not wired into detection loop")

# ── 4. Kick off YOLO-World preload in startup ──────────────────────────────────
OLD_STARTUP = "@router.on_event(\"startup\")"
if OLD_STARTUP in src:
    # find the startup function body and add world preload
    OLD_STARTUP_BODY = "    asyncio.ensure_future(_ensure_yolo())"
    NEW_STARTUP_BODY = """    asyncio.ensure_future(_ensure_yolo())
    asyncio.ensure_future(_ensure_yolo_world())"""
    if OLD_STARTUP_BODY in src:
        src = src.replace(OLD_STARTUP_BODY, NEW_STARTUP_BODY)
        print("✅ Added _ensure_yolo_world() to startup")
    else:
        print("⚠️  Could not add yolo_world to startup — add manually")
else:
    print("⚠️  No @router.on_event startup found")

with open(PATH, "w") as f:
    f.write(src)

print(f"\n✅ Patch complete — {PATH}")
print("Restart NIS to activate YOLO-World.")
