# NeuroLinux xArm — Full Migration Summary

> Updated Feb 28 2026 — covers ALL sessions (Windsurf Feb 26 + Cursor Feb 27-28).
> **This is the single source of truth. Read this first in every new session.**

## 1. What This Project Is

A FastAPI Python agent running on a **Raspberry Pi** that controls a **Hiwonder xArm** robotic arm via USB HID. The Pi also serves a **live camera feed** (Logitech C270) and calls a **NVIDIA H100** running Cosmos Reason2 for visual spatial reasoning. The whole system is called **NeuroLinux Agent**.

**Goal:** Pick-and-place operation for a cookoff demo. Arm picks a lighter from a table, places it in a bin.

## 2. Network / Infrastructure

| Machine | IP / Alias | Role |
|---|---|---|
| Windows dev machine | `localhost` | Where you code, push files, run scripts |
| Raspberry Pi | `192.168.1.163:8085` | Runs the agent (`neurolinux-agent` systemd service) |
| H100 GPU server | `172.16.1.83:8100` | Runs Cosmos Reason2 (`/reason`, `/health`) |
| H100 SSH alias | `awesome-gpu-name` | SSH alias for the H100 |

**CRITICAL:** The H100's real LAN IP is `172.16.1.83`. The Pi must use `172.16.1.83` directly. All `R2_URL` / `H100_REASON_URL` in the agent use `172.16.1.83:8100`.

**Pi credentials:** User: `neurolinux` / Pass: `neurolinux`
- Agent file on Pi: `/opt/neurolinux/neurolinux_agent.py`
- Dev file on Windows: `C:\Users\DiegoTorres\Desktop\neurolinux_agent_read.py`

## 3. Key Files

| File | Location | Purpose |
|---|---|---|
| `neurolinux_agent_read.py` | `C:\Users\DiegoTorres\Desktop\` | **Main agent source** (edit this, push to Pi) |
| `push_both.py` | `C:\Users\DiegoTorres\Desktop\` | Push+restart script (paramiko SSH) |
| `do_all.py` | `C:\Users\DiegoTorres\Desktop\` | Interactive touch calibration + Cosmos depth cal |
| `main_pi.py` | `C:\Users\DiegoTorres\Desktop\NIS_Protocol\` | NIS Protocol main |
| `cookoff.py` | `C:\Users\DiegoTorres\Desktop\NIS_Protocol\routes\` | Cookoff demo routes |

## 4. Push to Pi

```bash
# Fast: just agent
scp "C:\Users\DiegoTorres\Desktop\neurolinux_agent_read.py" neurolinux@192.168.1.163:/opt/neurolinux/neurolinux_agent.py

# Full: agent + NIS + restart + verify
python C:\Users\DiegoTorres\Desktop\push_both.py

# Restart service
ssh neurolinux@192.168.1.163 "sudo -S systemctl restart neurolinux-agent <<< 'neurolinux'"
```

## 5. Agent Endpoints (http://192.168.1.163:8085)

- `GET /health` — `{status, xarm, xarm_simulation, camera}`
- `POST /arm/move` — `{servo_id, position, duration_ms}`
- `POST /arm/group_move` — `{positions: {"1":500,...}, duration_ms}`
- `POST /arm/goto` — `{name}`
- `GET /camera/snapshot` — returns `{image_base64}`
- `GET /camera/stream` — MJPEG stream
- `POST /cosmos/depth_calibrate` — Cosmos Reason2 spatial scan
- `WS ws://192.168.1.163:8085/ws/cosmos` — live frames + Cosmos scene text

## 6. Servo Map (confirmed Feb 26 2026)

```
NEUTRAL = all 500 → arm folded compact toward FRONT
s1 = gripper        100=open, 500=close
s2 = base pan       500=center, 200=left, 800=right
s3 = secondary shoulder  500=neutral, 800=arm flat horizontal
s4 = elbow bend     500=neutral
s5 = shoulder rot   500=neutral (keep at 500)
s6 = MAIN reach     500=compact, 800=extends to BACK (pick direction)
```

## 7. NAMED_POSITIONS

```python
NAMED_POSITIONS = {
    "home":          {5:500, 4:500, 3:500, 2:500, 1:350, 6:500},
    "ready":         {5:500, 4:500, 3:500, 2:500, 1:100, 6:500},
    "pick_table":    {5:500, 4:200, 3:500, 2:500, 1:100, 6:800},
    "place_bin":     {5:500, 4:800, 3:500, 2:500, 1:100, 6:800},
    "reach_forward": {5:500, 4:500, 3:800, 2:500, 1:350, 6:500},
    "wave_up":       {5:500, 4:500, 3:500, 2:500, 1:350, 6:700},
    "wave_side":     {5:500, 4:500, 3:500, 2:800, 1:350, 6:600},
    "grip_open":     {1:100},
    "grip_close":    {1:500},
}
```
> `pick_blue`, `pick_yellow`, `pick_green` NOT yet saved — **#1 pending task**.

## 8. HID Protocol

```
Device: VID=0x0483, PID=0x5750
SEND group move:
  [0x00, 0x55, 0x55, n*3+5, 0x03, n, time_lo, time_hi,
   id1, pos_lo1, pos_hi1, ...]  — 65 bytes total

READ positions (cmd 0x1C):
  Send: [0x00, 0x55, 0x55, n+3, 0x1C, n, id1, id2, ...]
  Response: pos = resp[base+1] | (resp[base+2] << 8)  range: 0-1000
```

## 9. H100 / AI Stack

- **Cosmos Reason2**: `http://172.16.1.83:8100/reason` — `{"image":"<b64>","query":"...","max_tokens":512}`
- **Qwen3.5 vLLM**: `http://localhost:8701/v1` (via SSH tunnel, GPU 4, 16384 ctx)
- **Qwen proxy**: `http://localhost:8702/v1` — injects `enable_thinking=False`, rewrites model to `qwen35-nis`
- **Continue.dev**: configured at `~/.continue/config.json` — uses proxy at :8702

## 10. H100 Active Training Jobs

| tmux | GPU | Job |
|---|---|---|
| train_gpu1 | 1 | train_vla_bimanual.py |
| train_gpu3 | 3 | train_world_model_v4.py |
| train_gpu5c | 5 | train_sim2real_heavy_v2.py |
| train_gpu6 | 6 | train_diffusion_policy_v2.py |
| (pid) | 7 | train_vla_realdata.py --dataset aloha |

When aloha finishes → start `train_vla_heavy_v3.py` on GPU 7.
When world_model_v4 finishes → start `train_kan_v3_bspline.py` on GPU 3.

## 11. #1 Pending Task

Save physical pick poses for colored objects on the Pi:
```bash
# With arm physically positioned over blue object:
curl -X POST http://192.168.1.163:8085/arm/save_touch_pose -H "Content-Type: application/json" -d '{"name":"pick_blue"}'
# Repeat for pick_yellow, pick_green
```
