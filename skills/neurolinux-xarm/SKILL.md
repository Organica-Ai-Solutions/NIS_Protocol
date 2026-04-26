---
name: neurolinux-xarm
description: Control the Hiwonder xArm 1.6 robotic arm connected to NeuroLinux via USB serial. Use when the user wants to move the arm, open/close the gripper, home the arm, get joint status, or run servo commands. Also handles xArm 1s (older model). Requires the arm to be connected and NeuroLinux running.
metadata:
  openclaw:
    emoji: "🦾"
    always: false
    primaryEnv: NEUROLINUX_URL
    requires:
      anyBins: [python3, python]
      env: []
---

# Hiwonder xArm 1.6 Control

Control the Hiwonder xArm robotic arm through NIS Protocol and NeuroLinux.

## Connection Requirements

- xArm connected via USB-to-serial (typically `/dev/ttyUSB0` on Linux, `COM3` on Windows)
- Baud rate: **9600** for xArm 1s, **115200** for xArm 1.6
- NeuroLinux running on the host device (Pi 5 or PC)

## NIS Protocol Endpoints

```
POST /openclaw/invoke
  { "tool": "nis_xarm", "args": { "command": "status" } }

GET /v4/xarm/status        — via NeuroLinux Core on port 8080
GET /v4/xarm/ports         — list detected serial ports
```

## Commands via OpenClaw Bridge

```json
{ "tool": "nis_xarm", "args": { "command": "status", "port": "/dev/ttyUSB0", "model": "1.6" } }
{ "tool": "nis_xarm", "args": { "command": "home" } }
{ "tool": "nis_xarm", "args": { "command": "open" } }
{ "tool": "nis_xarm", "args": { "command": "close" } }
```

## Direct Python Usage

```python
from src.neurolinux.drivers.hiwonder_xarm import HiwonderXArmDriver

driver = HiwonderXArmDriver(port="/dev/ttyUSB0", model="1.6")
driver.home()
driver.set_gripper(open=True)
```

## Troubleshooting

- **No device found**: Run `ls /dev/ttyUSB*` or `ls /dev/ttyACM*`
- **Permission denied**: `sudo usermod -a -G dialout $USER` (Linux)
- **Wrong baud rate**: xArm 1s = 9600, xArm 1.6 = 115200
- **Arm not responding**: Check power supply (5V 3A for 1.6 model)

## References

- See `references/servo_ids.md` for servo channel mapping
- See `references/protocol.md` for low-level serial protocol
