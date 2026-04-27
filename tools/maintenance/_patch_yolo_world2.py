#!/usr/bin/env python3
"""
Patch 2: Wire YOLO-World into the /yolo/detect endpoint's concurrent gather.
Run on H100: python3 /tmp/_patch_yolo_world2.py
"""

PATH = "/home/nvidia/routes/yolo_vision.py"
with open(PATH) as f:
    src = f.read()

# ── 1. Add _run_yolo_world() function alongside _run_yolo ─────────────────────
# Insert before the detect endpoint
WORLD_RUN_FUNC = '''
def _run_yolo_world(frame_b64: str, conf: float = 0.08,
                    targets: List[str] = None) -> Dict:
    """
    Run YOLO-World inference on a base64 frame.
    Returns detections list (same format as _run_yolo).
    """
    import numpy as np
    from PIL import Image
    import io, base64

    if not _yolo_world_ready or _yolo_world_model is None:
        return {"detections": [], "n": 0}
    try:
        img_bytes = base64.b64decode(frame_b64)
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_frame  = np.array(pil_img)
        results   = _yolo_world_model.predict(
            np_frame, conf=max(0.07, conf * 0.7), verbose=False, imgsz=640
        )
        detections = []
        for res in results:
            for box in (res.boxes or []):
                wcls   = int(box.cls[0])
                wconf  = float(box.conf[0])
                wlabel = _YOLO_WORLD_CLASSES[wcls] if wcls < len(_YOLO_WORLD_CLASSES) else "object"
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

'''

# Find the detect endpoint definition as anchor
DETECT_ANCHOR = "@router.get(\"/detect\")"
if DETECT_ANCHOR in src:
    src = src.replace(DETECT_ANCHOR, WORLD_RUN_FUNC + DETECT_ANCHOR)
    print("✅ Added _run_yolo_world() before /detect endpoint")
else:
    print("⚠️  Could not find /detect endpoint — appending _run_yolo_world to end")
    src += WORLD_RUN_FUNC

# ── 2. Wire YOLO-World into the concurrent gather ──────────────────────────────
OLD_GATHER = """    result, gdino_dets = await asyncio.gather(yolo_task, gdino_task, return_exceptions=False)

    # Merge GDINO hits (NMS dedup against YOLO boxes)
    if gdino_dets:
        result["detections"] = _nms_merge(result["detections"], gdino_dets)
        result["n"]          = len(result["detections"])
        logger.info("GDINO merged %d extra detections", len(gdino_dets))"""

NEW_GATHER = """    world_task = loop.run_in_executor(None, _run_yolo_world, frame_b64, conf, target_list)
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
            logger.info("YOLO-World merged %d detections", len(world_dets))"""

if OLD_GATHER in src:
    src = src.replace(OLD_GATHER, NEW_GATHER)
    print("✅ Wired YOLO-World into concurrent gather")
else:
    print("⚠️  Could not find gather anchor — check manually")

# ── 3. Add YOLO-World to startup preload ──────────────────────────────────────
OLD_PRELOAD = "    await _ensure_yolo()"
NEW_PRELOAD = """    await _ensure_yolo()
    await _ensure_yolo_world()"""

# Only replace the first occurrence (in preload_yolo function)
if OLD_PRELOAD in src:
    src = src.replace(OLD_PRELOAD, NEW_PRELOAD, 1)
    print("✅ Added _ensure_yolo_world() to preload_yolo()")
else:
    print("⚠️  Could not find preload anchor")

with open(PATH, "w") as f:
    f.write(src)

# Validate syntax
import ast
try:
    ast.parse(src)
    print("✅ Python syntax OK")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR: {e}")

print(f"\n✅ Patch 2 complete — {PATH}")
