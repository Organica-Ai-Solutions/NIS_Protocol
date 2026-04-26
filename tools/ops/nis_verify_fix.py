#!/usr/bin/env python3
"""Verify NIS cosmos/reason fix is live on H100."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    if out: print(f"  {out[:800]}")
    if err and "Warning" not in err and "known hosts" not in err:
        print(f"  ERR: {err[:200]}")
    return r.returncode, out

print("=== NIS health ===")
ssh_one("curl -s http://localhost:8000/health", timeout=15)

print()
print("=== cosmos/reason fix verification ===")
ssh_one(
    r"""/data/organica-ai/NIS_Protocol/venv/bin/python -c "
import urllib.request, json, base64, io
from PIL import Image
img = Image.new('RGB', (64,64), (100,150,200))
buf = io.BytesIO(); img.save(buf, 'JPEG')
b64 = base64.b64encode(buf.getvalue()).decode()
print('image b64 len:', len(b64))
payload = json.dumps({'image_data': b64, 'task': 'describe scene', 'constraints': []}).encode()
req = urllib.request.Request(
    'http://localhost:8000/cosmos/reason',
    data=payload,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    resp = urllib.request.urlopen(req, timeout=20)
    d = json.loads(resp.read())
    print('OK - keys:', list(d.keys())[:8])
    print('source:', d.get('source'), 'error:', d.get('error'))
except Exception as e:
    print('error:', str(e)[:150])
" 2>/dev/null""",
    timeout=40
)

print()
print("=== Confirm cosmos.py fix is in place ===")
ssh_one(
    "grep -n 'b64decode\\|PIL\\|blank' /data/organica-ai/NIS_Protocol/routes/cosmos.py | head -10",
    timeout=15
)
