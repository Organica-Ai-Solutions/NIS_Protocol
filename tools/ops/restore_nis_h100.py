#!/usr/bin/env python3
"""Restore NIS Protocol on H100 - restart in correct tmux/venv."""
import subprocess, time

SSH_HOST = "awesome-gpu-name"

def ssh(cmd, timeout=45):
    r = subprocess.run(
        ["ssh", SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    out = r.stdout.strip()
    if out: print(f"  {out[:400]}")
    return r.returncode, out

print("=== Check if NIS is already back up ===")
rc, out = ssh("curl -s --max-time 5 http://localhost:8000/health 2>/dev/null || echo unreachable")

if "healthy" in out:
    print("  NIS is already running!")
    ssh("curl -s http://localhost:8000/health | python3 -c \"import sys,json; d=json.load(sys.stdin); print('version:', d.get('version'), 'routes:', d.get('modular_routes'))\"")
else:
    print("  NIS is down - starting in tmux session 'nis'...")
    # Kill any leftover process on 8000
    ssh("fuser -k 8000/tcp 2>/dev/null || true")
    time.sleep(1)
    # Start in a new tmux window so it persists
    ssh(
        "tmux new-window -t kan2 -n nis 'cd ~/organica-ai/NIS_Protocol && "
        "source ~/organica-ai/venv/bin/activate && "
        "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 2>&1 | tee /tmp/nis.log' 2>/dev/null || "
        "cd ~/organica-ai/NIS_Protocol && "
        "source ~/organica-ai/venv/bin/activate && "
        "nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/nis.log 2>&1 &"
    )
    time.sleep(6)
    ssh("curl -s --max-time 8 http://localhost:8000/health 2>/dev/null || echo still-down")

print()
print("=== Verify cosmos.py fix is live ===")
# Test the /cosmos/reason endpoint with a real (tiny) base64 image
ssh(r"""python3 -c "
import urllib.request, json, base64
from PIL import Image
import io

img = Image.new('RGB', (32, 32), color=(100, 150, 200))
buf = io.BytesIO()
img.save(buf, format='JPEG')
b64 = base64.b64encode(buf.getvalue()).decode()

payload = json.dumps({'image_data': b64, 'task': 'test', 'constraints': []}).encode()
req = urllib.request.Request(
    'http://localhost:8000/cosmos/reason',
    data=payload,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    resp = urllib.request.urlopen(req, timeout=15)
    d = json.loads(resp.read())
    print('cosmos/reason OK - keys:', list(d.keys())[:6])
except Exception as e:
    print('cosmos/reason error:', str(e)[:120])
" 2>/dev/null || echo "PIL not available on H100 path" """)
