#!/usr/bin/env python3
"""Run debug calls on H100 directly to get full 500 error bodies."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=90):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:1500])
    return r.returncode, out

print("=== Predict2.5 /video2world full error ===")
ssh_one(r"""/data/organica-ai/NIS_Protocol/venv/bin/python -c "
import urllib.request, json, base64, io
from PIL import Image
import numpy as np
arr = np.zeros((480, 848, 3), dtype=np.uint8)
arr[:,:] = [80, 100, 120]
buf = io.BytesIO()
Image.fromarray(arr).save(buf, format='JPEG', quality=85)
seed_b64 = base64.b64encode(buf.getvalue()).decode()
payload = json.dumps({
    'prompt': 'A robot arm picks up a red cube',
    'image_b64': seed_b64,
    'num_frames': 25, 'fps': 10,
    'height': 480, 'width': 848,
    'num_inference_steps': 20,
    'guidance_scale': 7.0, 'seed': 42
}).encode()
req = urllib.request.Request('http://localhost:8200/video2world', data=payload,
    headers={'Content-Type': 'application/json'}, method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=60)
    d = json.loads(resp.read())
    vid = d.get('video_b64','')
    print('OK video_b64 len:', len(vid), 'latency_ms:', d.get('latency_ms'))
except urllib.request.HTTPError as e:
    print('HTTP', e.code, ':', e.read().decode(errors='replace')[:600])
except Exception as e:
    print('error:', str(e)[:200])
" 2>/dev/null""", timeout=90)

print()
print("=== Transfer2.5 /transfer full error ===")
ssh_one(r"""/data/organica-ai/NIS_Protocol/venv/bin/python -c "
import urllib.request, json
payload = json.dumps({'demo': 'car_edge', 'control_type': 'edge', 'guidance': 3.0}).encode()
req = urllib.request.Request('http://localhost:8300/transfer', data=payload,
    headers={'Content-Type': 'application/json'}, method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=60)
    d = json.loads(resp.read())
    print('OK:', str(d)[:300])
except urllib.request.HTTPError as e:
    print('HTTP', e.code, ':', e.read().decode(errors='replace')[:600])
except Exception as e:
    print('error:', str(e)[:200])
" 2>/dev/null""", timeout=90)
