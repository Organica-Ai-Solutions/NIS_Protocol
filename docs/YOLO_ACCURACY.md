# Improving YOLO Object Detection Accuracy

YOLO object detection works but may not be accurate enough for your cookoff setup. Here are ways to improve it.

## Quick wins (no training)

### 1. Use a larger model — `yolov8s.pt`

The default `yolov8n.pt` (nano) is fastest but least accurate. On Pi 5, `yolov8s.pt` (small) gives noticeably better accuracy with ~2× latency.

**On the Pi** (in the NIS service env or systemd override):
```
YOLO_MODEL=yolov8s.pt
```

The model will auto-download (~22MB) on first run.

### 2. Increase input resolution — `YOLO_IMGSZ=960`

Larger input helps small objects (lighter, cup) at the cost of speed.

```
YOLO_IMGSZ=960
```

Valid range: 320–1280.

### 3. Tune confidence threshold

Lower `conf` = more detections but more false positives. The API accepts `?conf=0.15` or `?conf=0.20`.

## Current behavior

- **COCO aliasing:** Bottle/vase → "lighter", bowl → "bin" when confidence ≥ 0.35 (avoids relabeling noisy detections).
- **Blob fallback:** When YOLO finds nothing, OpenCV contours find salient blobs with heuristic labels: `lighter_candidate`, `bin_candidate`, `table_region`, `object`.
- **NMS:** Overlapping blob boxes (IoU > 0.4) are merged to reduce duplicates.

## Best accuracy (requires training)

For reliable lighter/bin detection, fine-tune YOLO on your own data:

1. Label 100–500 images: lighter, bin, xArm, table.
2. Export in YOLO format (one `.txt` per image).
3. Fine-tune `yolov8n.pt` or `yolov8s.pt` on that dataset.
4. Deploy the custom `.pt` to `/opt/nis-protocol/models/` and set `YOLO_MODEL` to its path.

This gives the best results but needs labeled data and a training run.
