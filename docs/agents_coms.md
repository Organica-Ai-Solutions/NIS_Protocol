# NIS Protocol — Agent Communications Reference
> Last updated: Session 20 (Feb 25 2026)

---

## 1. System Architecture

```
                        ┌─────────────────────────────────────┐
                        │         NVIDIA H100 80GB            │
                        │  :8100 Cosmos Reason2 (8B+LoRA)     │
                        │  :8200 Cosmos Predict2 (2B V2W)     │
                        │  :8300 Cosmos Transfer2.5           │
                        └───────────────┬─────────────────────┘
                                        │  SSH tunnel
                        ┌───────────────▼─────────────────────┐
                        │     PC Relay  192.168.1.160          │
                        │  :8101 → Reason2                     │
                        │  :8102 → Predict2                    │
                        │  :8103 → Transfer2.5                 │
                        └───────────────┬─────────────────────┘
                                        │  LAN
                        ┌───────────────▼─────────────────────┐
                        │     Raspberry Pi 4  192.168.1.163    │
                        │                                      │
                        │  NIS Protocol    :8000  main_pi.py  │
                        │  neurolinux-agent :8085              │
                        │  neurohub-ui     :3000               │
                        │  neurostore      :8006               │
                        └──────────────────────────────────────┘
```

---

## 2. Message Flow — Full VLA Pipeline (Session 14)

```
User Task: "pick up the mug and place it on the shelf"
    │
    ▼
NIS /demo/run  (POST :8000/demo/run)
    │
    ├─► _extract_task_objects(task) → ["mug", "shelf"]
    │
    ├─► _capture_frame(task)
    │       │
    │       ├─► GET :8085/vision/detect?targets=mug,shelf
    │       │       │
    │       │       ├─► YOLOv8n inference  (80 COCO classes)
    │       │       ├─► Red/color HSV detector
    │       │       ├─► Open-vocab remap (cup→mug, suitcase→shelf)
    │       │       └─► Returns: annotated_b64 + raw_b64 + scene_context
    │       │             "Scene: mug at [430,380] conf=0.87 | shelf at [220,450]"
    │       │
    │       └─► _predict2_video(task, raw_b64)
    │               │
    │               └─► POST :8102/video2world
    │                     prompt: "Robotic arm workspace: pick up the mug..."
    │                     image_b64: raw_b64
    │                     → Returns: {image_b64, video_b64, latency_ms}
    │                     → Cached in _latest_predicted_frame[0]
    │
    ├─► Phase 1: H100 Reasoning Loop (4 steps)
    │       │
    │       └─► _h100_next_action(step_i, completed, image)
    │               │
    │               ├─► step_cmd = "{task}. Phase N: {stage_hint}. {scene_context}"
    │               │     e.g. "pick up mug. Phase 1: Approach. Scene: mug at [430,380]"
    │               │
    │               ├─► step_i==0: image = Predict2 predicted frame (future state)
    │               ├─► step_i>0:  image = live annotated frame (current state)
    │               │
    │               └─► POST :8101/robot-plan
    │                     {command, robot_type, image_base64}
    │                     → Returns: {action, reasoning, confidence, safe_to_execute}
    │
    ├─► Phase 2: Execute sequence
    │       │
    │       └─► _extract_vla_intent(raw_action)
    │               │
    │               ├─► regex: "[px,py]" found?
    │               │     → _pixel_to_named_position(px,py)
    │               │         → calibration.json Euclidean lookup
    │               │         → e.g. [430,380] → "pick_table"
    │               │
    │               └─► keyword map fallback (wave/pick/place/grip/...)
    │
    └─► POST :8085/arm/named/{position}  (physical xArm move)
```

---

## 3. Agent Endpoints Reference

### neurolinux-agent  `:8085`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status: xarm_simulation, camera, yolo, calibration |
| GET | `/camera/snapshot` | Raw JPEG from C270 (base64) |
| GET | `/camera/status` | available, opencv, yolo, yolo_ready |
| GET | `/vision/detect?targets=mug,box` | YOLO+color → annotated_b64 + raw_b64 + scene_context |
| GET | `/vision/scene` | scene_context string only (no image) |
| POST | `/arm/named/{pos}` | Move arm to named position |
| POST | `/arm/move_group` | Move arm with raw servo dict |
| GET | `/arm/positions` | List all 13 named positions |
| POST | `/arm/reconnect` | Force USB HID reconnect |
| POST | `/arm/wave` | Wave sequence |
| POST | `/arm/pick_and_place` | Pick and place sequence |
| GET | `/calibration/status` | Auto-cal status/progress/result |
| POST | `/calibration/run?delay=N` | Trigger visual calibration |
| GET | `/calibration/lookup?px=&py=` | Find nearest named position for pixel |
| GET | `/camera/stream` | MJPEG stream (10fps) |

### NIS Protocol  `:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/demo/run` | Full VLA pipeline (YOLO→Predict2→Reason2→arm) |
| GET | `/health` | Service health |
| GET/POST | `/arm/*` | Proxy → :8085 arm endpoints |
| GET | `/cookoff/status` | H100 services health |
| POST | `/cookoff/demo` | Full cookoff demo |
| WS | `/ws/vla` | WebSocket streaming VLA |

---

## 4. Voice Chat Integration Analysis

### What's Already Built (in `main.py` / `src/voice/`)

| Component | File | Status |
|-----------|------|--------|
| Whisper STT | `src/voice/whisper_stt.py` | ✅ Built, lazy-load |
| VibeVoice TTS | `src/communication/` | ✅ 4 agent voices, <50ms |
| `/voice/transcribe` | `main.py` | ✅ Endpoint exists |
| `/communication/synthesize` | `main.py` | ✅ Returns WAV/MP3 |
| WS `/voice-chat` | `main.py` | ✅ STT→LLM→TTS pipeline |
| Frontend mic button | `static/chat_console.html` | ✅ Present |

### Integration with Robot Pipeline — **VERY EASY (2 steps)**

The voice chat system and robot arm pipeline are already connected via the NIS `/demo/run` endpoint. All that's needed:

**Step 1** — Voice → Task string:
```
User speaks → Whisper STT → text transcript
e.g. "pick up the blue bottle and put it in the box"
```

**Step 2** — Task string → Robot:
```python
# Already exists in main_pi.py:
POST /demo/run  {"task": transcript, "execute_arm": True}
```

**That's it.** The full YOLO+Predict2+Reason2+arm pipeline fires automatically from any text string.

### Voice → Robot Demo Flow (complete)

```
🎤 User speaks:  "wave hello, then grab the bottle"
        │
        ▼
WS /voice-chat → Whisper STT
        │
        ▼ transcript: "wave hello, then grab the bottle"
        │
        ▼
POST /demo/run {task: transcript, execute_arm: true}
        │
        ├─► YOLO finds "bottle" in frame (open-vocab target)
        ├─► Predict2 generates predicted trajectory
        ├─► Reason2 plans: wave → approach bottle → grasp → place
        └─► xArm executes physically
        │
        ▼
Response → TTS: "Done! I waved hello and grabbed the bottle."
        │
        ▼
🔊 User hears: AI voice confirms action
```

### To Wire It Up (minimal change to main_pi.py)

Add one endpoint:
```python
@app.post("/voice/robot-command")
async def voice_robot_command(audio_b64: str, execute_arm: bool = False):
    # 1. Transcribe
    stt = get_whisper_stt()
    result = await stt.transcribe_base64(audio_b64)
    transcript = result["text"]

    # 2. Run robot pipeline (reuse existing demo_run logic)
    req = DemoRunRequest(task=transcript, execute_arm=execute_arm)
    demo_result = await demo_run(req)

    # 3. Synthesize response
    tts_text = f"Executing: {transcript}"
    return {"transcript": transcript, "demo": demo_result, "tts_text": tts_text}
```

**Total effort: ~1 hour.** The Whisper model needs to be installed on Pi (or run STT on PC):
```bash
/opt/neurolinux/venv/bin/pip install openai-whisper
```

### Whisper on Pi vs PC

| Option | Latency | Notes |
|--------|---------|-------|
| Whisper `tiny` on Pi 4 | ~2s | Works, small model |
| Whisper `base` on Pi 4 | ~5s | Better accuracy |
| Whisper on PC (relay) | ~0.3s | **Recommended** — relay already running |
| H100 Whisper (future) | ~50ms | Via relay, if added to H100 stack |

**Recommendation**: Run Whisper `tiny` directly on Pi for offline capability, fall back to PC relay for speed.

---

## 5. Key Constants & Config

```
PI_IP         = 192.168.1.163
PC_IP         = 192.168.1.160
AGENT_PORT    = 8085
NIS_PORT      = 8000
H100_REASON2  = http://192.168.1.160:8101  (cosmos-reason2-8b + NIS LoRA)
H100_PREDICT2 = http://192.168.1.160:8102  (cosmos-predict2-2B-V2W)
H100_TRANSFER = http://192.168.1.160:8103  (Transfer2.5)
YOLO_MODEL    = /opt/neurolinux/yolov8n.pt  (6MB, 80 COCO classes)
CAL_FILE      = /opt/neurolinux/calibration.json  (13 positions)
WEBCAM        = /dev/video16  (Logitech C270)
```

---

## 6. Named Positions & Pixel Map

### Servo Values (neurolinux_agent.py NAMED_POSITIONS)

| Position | Servo {1,2,3,4,5,6} | Notes |
|----------|---------------------|-------|
| home | 500,500,500,500,500,350 | ARM STANDING TALL — gripper top-center |
| ready | 500,650,350,500,400,350 | mid-height ready pose |
| park | 500,850,150,500,200,350 | folded/parked |
| reach_forward | 500,250,750,500,600,350 | horizontal toward camera |
| reach_left | 150,280,720,300,580,350 | left side workspace |
| reach_right | 850,280,720,700,580,350 | right side workspace |
| pick_table | 500,200,820,500,750,100 | reaching table surface, grip open |
| inspect | 500,380,500,500,200,350 | looking down at table |
| wave_up | 500,200,800,500,150,350 | arm raised up |
| wave_side | 750,200,800,650,150,350 | arm up + right |
| grip_open | {6:100} | gripper fully open |
| grip_close | {6:550} | gripper closed on object |
| place_bin | 800,280,720,650,580,100 | **BIN DROP** — left side, grip open |

### Calibration Fallback Pixel Map (v2, Feb 25 2026)

| Position | cx | cy | zone | source |
|----------|----|----|------|--------|
| home | 424 | 90 | center/far/high | fallback |
| inspect | 384 | 287 | center/mid/mid | fallback |
| pick_table | 380 | 373 | center/near/low | fallback |
| reach_forward | 424 | 340 | center/mid/low | fallback |
| reach_left | 180 | 300 | left/mid/low | fallback |
| reach_right | 662 | 300 | right/mid/low | fallback |
| place_bin | 144 | 400 | left/near/low | fallback |
| wave_up | 424 | 120 | center/far/high | fallback |
| wave_side | 600 | 150 | right/far/high | fallback |

### Calibration Status Fields (v2)

```json
GET :8000/calibration/status  →
{
  "status":     "complete" | "running" | "upgrading" | "failed",
  "calibrated": 9,
  "live":       6,
  "total":      9,
  "timestamp":  1740523200.0,
  "positions": {
    "home": {
      "cx": 248, "cy": 344, "zone": "center/mid/low",
      "ok": true,
      "source": "live_yolo",          ← live_hybrid | live_reason2 | live_yolo | fallback
      "yolo": [248, 344, 0.95],       ← [cx, cy, conf] or null
      "r2":   [242, 338]              ← [cx, cy] or null
    },
    ...
  }
}
```

### pick_and_place() Sequence (CORRECT as of Session 18)
```
home → grip_open → pick_table → grip_close → ready → place_bin → grip_open → home
```

### wave() Sequence
```
home → wave_up → wave_side × 2 → home   (~8.5s total)
```

---

## 7. Session History

| Session | Key Changes |
|---------|-------------|
| 1-10 | Core NIS Protocol, xArm HID, H100 Reason2 integration |
| 11 | H100 LoRA fine-tune, cookoff pipeline, /demo/run |
| 12 | Camera→Cosmos pipeline, stage-specific prompts, dedup fix, NeuroHub |
| 13 | Logitech C270 webcam, lazy camera init, AutoCalibrator (13 pos), calibration→NIS |
| 14 | YOLOv8n open-vocab detection, Predict2 /video2world hybrid, scene_context→H100 |
| 15-17 | pick_and_place fix (place_bin not reach_right), relay_h100.py TCP relay, SSH Paramiko |
| 18 | Full verified demo (7/7 record_demo.py, 4/4 cookoff, 4/4 cosmos-demo). place_bin confirmed. |
| 19 | Calibration v2 multi-source design: YOLO primary, R2 upgrade, per-pos hints |
| **20** | **Calibration v2 2-phase rewrite: Phase1=YOLO tour (~30s, live commits), Phase2=R2 upgrade (65s/pos, best-effort). Image resize 424×240 for R2 speed. Extended coord extraction (clamp+4 patterns). Few-shot R2 prompt.** |

### Session 20 Changes (Feb 25 2026)
- **`_auto_calibrate()` v2 full rewrite** in `main_pi.py`:
  - **Phase 1** — YOLO arm tour, 15s HTTP timeout, commits each position to `_calibration_state` immediately → Pi stays responsive, `/calibration/status` updates live
  - **Phase 2** — R2 upgrade pass, 65s timeout, no arm movement, best-effort; upgrades `live_yolo` → `live_hybrid` or `live_reason2`
- **`_resize_for_r2()`** — resizes image to max 424×240 before R2 call (prevents 90-120s timeouts on 130KB images)
- **`_extract_coords()` v2** — clamps out-of-bounds coords (R2 sometimes imagines taller frame), adds 4 regex patterns: `[x,y]`, `(x,y)`, `x=N y=N`, `at N, N`
- **Few-shot R2 prompt** — concrete examples force `gripper at [x, y]` output format
- **Per-position `_POS_HINTS`** — spatial guidance per position injected into every R2 prompt
- **`_SCENE_CTX`** — global physical setup context: camera faces arm, card faces camera, home=tall, tape anchors, 848×480
- **Source labels**: `live_hybrid` 🟢 / `live_reason2` 🔵 / `live_yolo` 🟣 / `fallback` 🟡
- **`record_demo.py`** calibration display updated to show new source labels + R2/YOLO sub-coords
- **`relay_h100.py`** TCP relay: PC listens on 0.0.0.0:8101-8103, forwards to localhost SSH tunnel ports
- **`watch_cal.py`** — new polling script that watches calibration complete and prints full results

---

## 8. Quick Fire Demo Commands

```bash
# ── Calibration ────────────────────────────────────────────────────
# Check calibration status (live count, source labels)
curl http://192.168.1.163:8000/calibration/status | python -m json.tool

# Trigger fresh recalibration (Phase 1 ~30s, Phase 2 ~10min best-effort)
curl -X POST http://192.168.1.163:8000/calibration/recalibrate

# ── Demos ──────────────────────────────────────────────────────────
# Record demo trace (7 steps, ~55s)
python record_demo.py

# Fast cookoff demo (~33s)
python run_cookoff_demo.py

# Full cosmos-demo + demo/run combined
python final_demo_run.py

# Watch calibration complete (polls HTTP, then shows full results)
python watch_cal.py

# ── Push to Pi ─────────────────────────────────────────────────────
# Push main_pi.py only + restart NIS
python push_main_only.py

# Push agent + main_pi.py + cookoff.py + restart both
python push_both.py

# ── Direct curl ────────────────────────────────────────────────────
# Full VLA demo
curl -X POST http://192.168.1.163:8000/demo/run \
  -H "Content-Type: application/json" \
  -d '{"task":"wave hello then pick up the red cube and place it in the bin","execute_arm":true}'

# Cookoff demo
curl -X POST http://192.168.1.163:8000/cookoff/demo

# Vision detect
curl "http://192.168.1.163:8085/vision/detect?targets=arm,mug,bottle,box"

# Move arm to position
curl -X POST http://192.168.1.163:8085/arm/named/home
curl -X POST http://192.168.1.163:8085/arm/wave
curl -X POST http://192.168.1.163:8085/arm/pick_and_place

# ── H100 direct (requires relay_h100.py running on PC) ─────────────
curl http://192.168.1.160:8101/health   # Reason2
curl http://192.168.1.160:8102/health   # Predict2
curl http://192.168.1.160:8103/health   # Transfer2.5
```

### Key Scripts on PC Desktop

| Script | Purpose | Time |
|--------|---------|------|
| `record_demo.py` | 7-step demo trace for video recording | ~55s |
| `run_cookoff_demo.py` | Fast cookoff demo | ~33s |
| `final_demo_run.py` | cosmos-demo + demo/run combined | ~90s |
| `watch_cal.py` | Poll calibration until complete, print results | ~30s+ |
| `push_main_only.py` | Push main_pi.py → Pi, restart NIS | ~35s |
| `push_both.py` | Push agent+main_pi.py+cookoff.py, restart both | ~50s |
| `relay_h100.py` | TCP relay + SSH tunnel to H100 (must run on PC) | persistent |
| `check_r2.py` | Quick R2 health + test call | ~15s |
