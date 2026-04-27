# Hiwonder xArm — Complete Calibration & Library Reference
> NIS Protocol / NeuroLinux · Last updated Feb 2026

---

## 1. Hardware Overview

| Property | Value |
|---|---|
| Model | Hiwonder xArm 1S (6-DOF) / xArm AI |
| Servo type | Bus servo (LX-15D / LX-225 / HTS-16L / HX-06L) |
| Communication | USB HID *or* UART TTL serial |
| USB VID | `0x0483` (1155) |
| USB PID | `0x5750` (22352) |
| UART baud | 9600, 8N1 |
| Position range | **0 – 1000** servo units per servo |
| Center / neutral | **500** (mid-range for rotation servos) |
| Power | 7.5 V, 6 A DC adapter |

---

## 2. Servo ID Map (Our Arm)

Confirmed by physical testing on the Hiwonder xArm 1S (USB HID `hidraw` interface):

| ID | Joint | Direction notes | Open / Close (gripper only) |
|---|---|---|---|
| **S1** | **Gripper** | 100 = fully open · 500 = neutral · 550 = closed on object | ✓ |
| S2 | Shoulder pitch | low value = forward lean · high = upright |  |
| S3 | Elbow | low = extended/straight · high = folded |  |
| S4 | Wrist yaw | 500 = center |  |
| S5 | Wrist pitch | low = tilt-up · high = tilt-down |  |
| **S6** | **Base rotation** | 100 = far left / bin side · 350 = center · 800 = far right | |

> **Note from official docs:** All servos pre-configured to neutral (500) before shipment. The `Attention` action group (No. 0) is the factory home pose.

---

## 3. Safe Motion Ranges (per servo)

From `SafeXArm` class by @maximecb (validated community source):

| Servo | Min | Center | Max | Note |
|---|---|---|---|---|
| S1 (Gripper) | 100 | 350 | 650 | 100=open, 550=obj grip |
| S2 (Shoulder) | 200 | 500 | 800 |  |
| S3 (Elbow) | 400 | 650 | 900 |  |
| S4 (Wrist Yaw) | 100 | 350 | 600 |  |
| S5 (Wrist Pitch) | 50  | 450 | 850 |  |
| S6 (Base) | 100 | 500 | 900 |  |

**Always return to home before powering off** — servos left at extreme positions cause stalling on next power-on.

---

## 4. Calibrated Named Positions (Our Arm)

Source of truth: `servo_calibration_result.json`

```json
{
  "home":       {"1":500, "2":500, "3":400, "4":500, "5":400, "6":350},
  "inspect":    {"1":500, "2":625, "3":485, "4":500, "5":335, "6":500},
  "pick_table": {"1":500, "2":258, "3":733, "4":500, "5":850, "6":500},
  "lift_grip":  {"1":550, "2":625, "3":485, "4":500, "5":335, "6":500},
  "place_bin":  {"1":550, "2":370, "3":720, "4":380, "5":680, "6":240},
  "ready":      {"1":500, "2":484, "3":433, "4":500, "5":432, "6":350},
  "reach_left": {"1":162, "2":433, "3":650, "4":375, "5":550, "6":350},
  "reach_right":{"1":859, "2":153, "3":650, "4":525, "5":550, "6":350}
}
```

> S1=500 in pick_table because the **gripper is opened separately** before the arm descends.  
> S1=550 in lift_grip/place_bin because the **grip is maintained** during transit.

---

## 5. Full Pick-and-Place Pipeline (9 steps)

```
Step  Action           Servo details
 1    home             S1=500 (neutral), arm upright
 2    inspect          Survey workspace from above
 3    open_gripper     S1 → 100 (fully open)
 4    pick_table       Lower arm to object, aligned
 5    close_gripper    S1 → 550 (grip object)
 6    lift_grip        Raise arm, maintain S1=550
 7    place_bin        Move base to bin side (S6≈240), maintain S1=550
 8    open_gripper     S1 → 100 (drop object into bin)
 9    home             Return to home
```

Triggered via:
- `POST /arm/pick_and_place` (Pi agent REST)
- `python nis_cli.py "pick and place demo"` (NIS CLI)
- `POST /chat` with message "run pick and place" (NIS Protocol REST)

---

## 6. Available Libraries

### 6.1 `xarm` — ccourson/Hiwonder-xArm1S ⭐ (recommended)

GitHub: https://github.com/ccourson/Hiwonder-xArm1S

```bash
pip install xarm
# or from source:
git clone https://github.com/ccourson/Hiwonder-xArm1S
```

**Usage (USB / serial):**
```python
import xarm

arm = xarm.Controller('USB')   # auto-detects USB HID
# arm = xarm.Controller('/dev/ttyUSB0')  # serial fallback

# Move single servo
arm.setPosition(1, 500, 1000, True)   # servo_id, position, time_ms, wait

# Move all 6 servos simultaneously
arm.setPosition([[1,500],[2,500],[3,400],[4,500],[5,400],[6,350]], 1000)

# Read position
pos = arm.getPosition(1)          # returns 0-1000
pos_deg = arm.getPosition(1, True)  # returns degrees

# Turn off servo torque (compliant mode for hand-teaching)
arm.servoOff(1)    # single servo
arm.servoOff()     # all servos

# Battery voltage (mV)
v = arm.getBatteryVoltage()
```

### 6.2 `easyhid` / `hid` — Raw USB HID (low-level)

```bash
sudo apt-get install libhidapi-hidraw0 libhidapi-libusb0
pip install hid        # preferred
# or
pip install easyhid
```

**Packet format (CMD_SERVO_MOVE 0x03):**
```
[0x55, 0x55, 8, 0x03, 1, time_lsb, time_msb, servo_id, pos_lsb, pos_msb]
```

**Read all 6 positions (CMD_MULT_SERVO_POS_READ 0x15):**
```
Send:    [0x55, 0x55, 9, 21, 6, 1, 2, 3, 4, 5, 6]
Receive: [0x55, 0x55, len, 21, count, id1, pos1_lsb, pos1_msb, id2, ...]
```

**Full minimalist class:**
```python
import hid

def itos(v):
    return v & 0xFF, v >> 8

class XArm:
    VID, PID = 1155, 22352

    def __init__(self):
        self.dev = hid.device()
        self.dev.open(self.VID, self.PID)

    def move(self, servo_id: int, pos: int, time_ms: int = 1000):
        t_l, t_h = itos(time_ms)
        p_l, p_h = itos(pos)
        self.dev.write([0x00, 0x55, 0x55, 8, 0x03, 1, t_l, t_h, servo_id, p_l, p_h])

    def move_all(self, positions: dict, time_ms: int = 1000):
        """positions: {servo_id: value_0_to_1000}"""
        for sid, val in positions.items():
            self.move(int(sid), int(val), time_ms)

    def read_all(self) -> dict:
        self.dev.write([0x00, 0x55, 0x55, 9, 21, 6, 1, 2, 3, 4, 5, 6])
        ret = self.dev.read(64)
        count = ret[4]
        return {ret[5 + 3*i]: (ret[6 + 3*i] | ret[7 + 3*i] << 8)
                for i in range(count)}

    def torque_off(self, servos=(1,2,3,4,5,6)):
        n = len(servos)
        self.dev.write([0x00, 0x55, 0x55, n+3, 0x14, n] + list(servos))

    def torque_on(self):
        """Re-engage torque by sending a move command."""
        pass  # any move command re-engages servos

    def close(self):
        self.dev.close()
```

> **Note:** `read_all()` can be slow (~450 ms). Don't call it in a tight loop.

### 6.3 `hiwonder_xarm` — Serial interface (our codebase)

Found in `src/neurolinux/drivers/hiwonder_xarm.py`. Used when arm connects over TTL serial instead of USB HID.

```python
from hiwonder_xarm import HiwonderXArm

arm = HiwonderXArm(port='/dev/ttyUSB0')
arm.connect()
arm.move_servo(servo_id=1, angle_deg=120, speed=800)
arm.disconnect()
```

> In serial mode, positions are in **degrees (0–240)**, not servo units.  
> Our codebase maps: `angle_deg = servo_unit * 240 / 1000`

### 6.4 `xarm_hid` — HID interface (our codebase)

Located at `drivers/xarm_hid.py` on the Pi. Primary driver used by `neurolinux_agent.py`.

```python
from xarm_hid import XArmHID, HOME_UNITS, GRIPPER_OPEN, GRIPPER_CLOSE

arm = XArmHID()              # auto-detect /dev/hidraw*
arm.connect()

# ── Servo-unit control (0-1000) — CANONICAL ─────────────────────────
arm.set_position(1, 500, 1000)            # single servo: id, units, ms
arm.set_positions({1:500, 6:350}, 800)   # multi-servo simultaneously

# ── Degree control (0-240) — legacy ─────────────────────────────────
arm.move(1, 120, 1000)                    # single servo: id, degrees, ms
arm.move_all({1:120, 6:84}, 800)         # multi-servo in degrees

# ── Prebuilt moves ───────────────────────────────────────────────────
arm.home()                    # HOME_UNITS = {1:500,2:500,3:400,4:500,5:400,6:350}
arm.grip(True)                # close gripper  S1 → 550
arm.grip(False)               # open gripper   S1 → 100
arm.wave()                    # wave using base rotation (S6)
arm.pick_sequence()           # full 9-step pick-and-place

# ── Position readback (slow ~450ms) ─────────────────────────────────
pos = arm.read_pos(1)         # single servo -> servo units (int) or None
all_pos = arm.get_positions() # all servos  -> {1:val, 2:val, ...}

# ── Torque control ───────────────────────────────────────────────────
arm.torque_off()              # compliant mode — arm can be moved by hand
arm.torque_on()               # re-engage torque

arm.stop()
arm.disconnect()
```

> **Critical fix (Feb 2026):** Driver previously had servo IDs backwards (S1=Base, S6=Gripper).
> Corrected to match physical testing: **S1=Gripper, S6=Base Rotation**.
> Also added `set_position()`, `set_positions()`, `get_positions()`, `torque_off()`,
> `torque_on()` — these were called by the agent but were missing from the driver.

---

## 7. Official UART Command Reference

| Command | Byte | Description |
|---|---|---|
| `CMD_SERVO_MOVE` | `0x03` | Move one or more servos to position |
| `CMD_ACTION_GROUP_RUN` | `0x06` | Execute stored action group |
| `CMD_ACTION_GROUP_STOP` | `0x07` | Stop running action group |
| `CMD_ACTION_GROUP_SPEED` | `0x0B` | Adjust action group speed (%) |
| `CMD_GET_BATTERY_VOLTAGE` | `0x0F` | Get battery voltage (mV) |
| `CMD_MULT_SERVO_UNLOAD` | `0x14` | Torque-off selected servos (compliant) |
| `CMD_MULT_SERVO_POS_READ` | `0x15` | Read positions of selected servos |
| `CMD_ACTION_GROUP_COMPLETE` | `0x08` | Controller → host: action completed |

**Packet format:**
```
[0x55, 0x55, LENGTH, CMD, param1, param2, ...]
LENGTH = number_of_params + 2
```

**Move servo 1 to position 800 in 1000 ms:**
```
55 55 08 03 01 E8 03 01 20 03
             ^^ count=1
                   ^^^^ time=1000ms (0x03E8)
                         ^^ servo_id=1
                            ^^^^^ position=800 (0x0320)
```

**Torque-off all 6 servos (training mode):**
```
55 55 09 14 06 01 02 03 04 05 06
```

**Read all 6 positions:**
```
Send:    55 55 09 15 06 01 02 03 04 05 06
Receive: 55 55 [len] 15 06 [id1 posL posH] [id2 posL posH] ...
```

---

## 8. udev Rule (Linux — avoid running as root)

```bash
# /etc/udev/rules.d/80-xarm.rules
SUBSYSTEM=="hidraw", ATTRS{product}=="LOBOT", GROUP="dialout", MODE="0660"

# Apply without reboot:
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER   # then log out/in
```

---

## 9. Known Issues & Quirks

| Issue | Detail |
|---|---|
| `read_all()` latency | Position readback takes ~450 ms over HID. Don't call in motion loop. |
| `fwupd` device conflict | On Ubuntu, `fwupd` tries to upgrade xArm firmware and blocks HID. `sudo systemctl disable fwupd` |
| Servo stalling | If joint hits mechanical stop while powered, current spikes sharply. Use safe ranges from §3. |
| Gripper drop during transit | Always maintain S1=550 from pick to place. Open gripper only at home and at bin. See §5. |
| Serial vs HID units | HID uses 0-1000 natively. Serial uses 0-240 degrees. Our mapping: `angle = unit * 240 / 1000` |
| Torque re-engagement | After `torque_off()`, any `move` command re-engages the servo. |
| Action Group No. 0 | Factory "Attention" (home) action. `CMD_ACTION_GROUP_RUN 0x06 00 01 00` |

---

## 10. Calibration Workflow (Step by Step)

### A. First-time calibration (physical teaching)

```bash
# 1. Enable compliant / training mode
curl -X POST http://192.168.1.163:8085/arm/train

# 2. Move arm by hand to desired position

# 3. Save the position
curl -X POST http://192.168.1.163:8085/arm/save_touch_pose \
  -H "Content-Type: application/json" \
  -d '{"name": "pick_table"}'

# 4. Repeat for each position: home, inspect, pick_table, lift_grip, place_bin

# 5. Exit training mode
curl -X POST http://192.168.1.163:8085/arm/train/stop
```

### B. Push calibration from laptop to Pi (no SSH)

```bash
# Edit servo_calibration_result.json, then:
python push_calibration.py --pi 192.168.1.163

# Verify what the Pi has saved:
python push_calibration.py --pi 192.168.1.163 --verify
```

### C. Fine-tune a specific position programmatically

```python
import requests

BASE = "http://192.168.1.163:8085"

# Move to approximate position first
requests.post(f"{BASE}/arm/group_move", json={
    "positions": {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500},
    "duration_ms": 900
})

# Then save it as a named pose
requests.post(f"{BASE}/arm/save_touch_pose", json={"name": "pick_table"})
```

Or push specific values directly (bypasses hardware readback):
```python
requests.post(f"{BASE}/arm/save_touch_pose", json={
    "name": "pick_table",
    "positions": {"1": 500, "2": 258, "3": 733, "4": 500, "5": 850, "6": 500}
})
```

---

## 11. Official Documentation Links

| Resource | URL |
|---|---|
| xArm AI full docs | https://docs.hiwonder.com/projects/xArm_AI/en/latest/ |
| Serial communication protocol | https://docs.hiwonder.com/projects/xArm_AI/en/latest/docs/7.Serial_Communication_Course.html |
| PC software control | https://docs.hiwonder.com/projects/xArm_AI/en/latest/docs/3.PC_software_control.html |
| Bus servo secondary dev | https://docs.hiwonder.com/projects/Bus-Servo-Controller/en/latest/ |
| ccourson/Hiwonder-xArm1S | https://github.com/ccourson/Hiwonder-xArm1S |
| DuaneOne/RPi examples | https://github.com/DuaneOne/Hiwonder-LewanSoul-xArm-1S-Python-for-RPi-examples |
| maximecb HID gist | https://gist.github.com/maximecb/7fd42439e8a28b9a74a4f7db68281071 |
| xArm-Developer SDK (industrial) | https://github.com/xArm-Developer/xArm-Python-SDK |
