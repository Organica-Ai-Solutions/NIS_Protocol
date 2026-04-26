---
name: neurolinux-devices
description: Discover and manage devices connected to NeuroLinux (cameras, robotic arms, sensors, serial devices, USB peripherals). Use when the user asks what devices are connected, wants to add a new device, check device status, or configure peripherals on the NeuroLinux system.
metadata:
  openclaw:
    emoji: "🔌"
    always: false
    primaryEnv: NEUROLINUX_URL
    requires:
      env: []
---

# NeuroLinux Device Management

Discover, monitor, and manage hardware devices connected to NeuroLinux.

## NIS Protocol / NeuroLinux Endpoints

```
GET /v4/devices             — list all detected devices
GET /v4/xarm/ports          — serial ports (xArm / USB robots)
GET /v4/xarm/status         — xArm connection status
```

## Device Types Supported

| Type        | Detection Method  | Notes                          |
|-------------|-------------------|--------------------------------|
| Camera      | V4L2 / OpenCV     | `/dev/video*`                  |
| xArm        | Serial scan       | `/dev/ttyUSB*`, `/dev/ttyACM*` |
| Microphone  | ALSA              | arecord -l                     |
| GPIO        | gpiod / RPi.GPIO  | Raspberry Pi 5                 |
| USB HID     | hidapi            | Joysticks, gamepads            |

## Adding a New Device via NeuroLinux Dashboard

1. Open Flutter dashboard → **Add Device** wizard
2. Select device type (Camera, Robot Arm, Sensor, etc.)
3. Follow connection steps
4. Device appears in `/v4/devices`

## Device Query via OpenClaw Bridge

```json
POST /openclaw/invoke
{ "tool": "nis_chat", "args": { "message": "What devices are connected to NeuroLinux?" } }
```

The NIS Protocol chat agent will call `/v4/devices` automatically when asked.

## Programmatic Access

```python
import httpx
r = httpx.get("http://localhost:8080/v4/devices")
devices = r.json()["devices"]
for d in devices:
    print(d["type"], d["id"], d.get("status"))
```
