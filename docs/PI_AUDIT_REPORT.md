# Raspberry Pi (NeuroLinux Agent) — Full Audit Report
> Audit date: Feb 26 2026 | Pi IP: 192.168.1.163:8085 | Agent version: 4.0.0

---

## 1. Connectivity & Hardware

| Check | Result |
|---|---|
| Pi reachable | YES |
| xArm connected | YES — USB HID (`/dev/hidraw*`) |
| xArm simulation mode | NO — real hardware |
| Camera available | YES — OpenCV + YOLO ready |
| Camera snapshot | YES — 274 KB JPEG captured |
| Agent version | 4.0.0 |

---

## 2. API Endpoint Inventory

### Working Endpoints (Pi)

| Endpoint | Method | Status |
|---|---|---|
| `/` | GET | 200 — dashboard HTML |
| `/health` | GET | 200 — OK |
| `/arm/status` | GET | 200 — arm connected |
| `/arm/touch_poses` | GET | 200 — 3 poses (outdated) |
| `/arm/named/{name}` | POST | 200 |
| `/arm/group_move` | POST | 200 |
| `/arm/gripper/open` | POST | 200 |
| `/arm/gripper/close` | POST | 200 |
| `/arm/pick_and_place` | POST | 200 |
| `/arm/save_touch_pose` | POST | 200 |
| `/arm/train` | POST | 200 |
| `/arm/train/stop` | POST | 200 |
| `/camera/status` | GET | 200 |
| `/camera/snapshot` | GET | 200 — 274 KB image |
| `/camera/stream` | GET | streaming (MJPEG) |
| `/ws` | WS | available |

### Missing Endpoints (Pi has OLD code)

| Endpoint | Impact |
|---|---|
| `POST /arm/load_calibration` | **CRITICAL** — can't push calibration remotely |
| `POST /chat` | MISSING — no NLP on Pi |
| `GET /offline/status` | MISSING — Ollama status unavailable |
| `POST /offline/chat` | MISSING — local LLM unavailable |
| `POST /system/restart` | MISSING — can't restart services remotely |
| `GET /skills` | MISSING — skill listing unavailable |
| `POST /execute` | MISSING — remote code execution unavailable |

---

## 3. Touch Poses — OUTDATED

**Current Pi poses** (from `GET /arm/touch_poses`):

| Pose name | S1 | S2 | S3 | S4 | S5 | S6 | Issue |
|---|---|---|---|---|---|---|---|
| `pick_blue` | 350 | 500 | 500 | 500 | 500 | 500 | WRONG — this is near home, not pick |
| `lift_grip` | 550 | 625 | 485 | 500 | 335 | 500 | OK |
| `place_closed` | 550 | 370 | 720 | 380 | 680 | 240 | WRONG NAME — should be `place_bin` |

**Required poses for pick-and-place pipeline:**

| Pose | Required? | On Pi? |
|---|---|---|
| `home` | YES — step 1 & 9 | **MISSING** |
| `inspect` | YES — step 2 | **MISSING** |
| `pick_table` | YES — step 4 | **MISSING** |
| `lift_grip` | YES — step 6 | present (values correct) |
| `place_bin` | YES — step 7 | **MISSING** (exists as `place_closed`) |

**Consequence:** `/arm/pick_and_place` will fail at step 1 (no `home` pose) or fall back to the angle-based preset system.

---

## 4. Key Bug: `save_touch_pose` ignores explicit positions

The Pi's running `save_touch_pose` endpoint always reads the **current hardware position** regardless of what `positions` is sent in the body.

**Test result:**
```
POST /arm/save_touch_pose {"name":"_test", "positions":{"1":500,"2":500,...}}
Response: positions: {1:350, 2:500, ...}   ← ignored body, used hardware position
```

**Root cause:** Pi is running old agent code without the body-positions fix committed Feb 2026.

---

## 5. What the Pi IS Running vs What We Have

| Feature | Pi (old) | Windows repo (new) |
|---|---|---|
| Simultaneous servo moves | Sequential loop | `set_positions()` broadcast |
| `save_touch_pose` body | Ignores positions | Uses body positions first |
| `/arm/load_calibration` | MISSING | Present |
| `/chat` endpoint | MISSING | Present (agentic) |
| `/offline/status` | MISSING | Present |
| HiwonderXArm class | MISSING | `HiwonderXArm` sync wrapper |
| Servo ID map (xarm_hid) | S1=Base, S6=Grip | **FIXED**: S1=Grip, S6=Base |
| Servo limits | Conservative | Expanded to match calibration |
| `pick_sequence()` | Wrong degrees | Servo units + correct poses |

---

## 6. Current Arm Position

When audited, arm was at:
```
S1 (Gripper):       350  ← neutral/slightly open
S2 (Shoulder):      500  ← center
S3 (Elbow):         500  ← center (slightly different from our home=400)
S4 (Wrist Yaw):     500  ← center
S5 (Wrist Pitch):   500  ← center (slightly different from our home=400)
S6 (Base):          500  ← center (our home=350)
```

The arm is near center but NOT at our calibrated home position `{1:500,2:500,3:400,4:500,5:400,6:350}`.

---

## 7. Action Plan

### Immediate (can do now over HTTP)

1. **Push calibration poses** using `push_poses_via_move.py` — moves arm to each position then saves it. Requires arm to be unobstructed.

2. **Delete wrong poses** `pick_blue` and `place_closed` after pushing correct ones.

### Requires Pi access (SSH or physical)

3. **Deploy updated agent** — the Pi needs our new `neurolinux_agent.py`:
   ```bash
   # On the Pi:
   cd /opt/neurolinux
   git pull origin main
   sudo systemctl restart neurolinux-agent
   ```
   This will bring: `/arm/load_calibration`, `/chat`, `/offline/status`, `/system/restart`, servo-fix, etc.

4. **Also deploy updated drivers:**
   ```bash
   # Drivers are at /opt/neurolinux/drivers/
   # xarm_hid.py — fixed servo IDs, set_positions(), get_positions()
   # hiwonder_xarm.py — added HiwonderXArm sync class
   ```

### After deployment

5. Run `push_calibration.py --pi 192.168.1.163` to push full calibration set via the new `/arm/load_calibration` endpoint.

---

## 8. Servo Limit Issue in Old Code

The old `xarm_hid.py` on the Pi may have conservative limits that clamp calibrated positions:
- `wave_up` pose: S2=834 (old limit: 800) → would be silently clamped to 800
- `wave_up` pose: S3=154 (old limit: 400) → would be silently clamped to 400

After deploying new drivers, all 10 calibrated poses will execute correctly.
