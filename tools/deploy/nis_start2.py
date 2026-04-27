#!/usr/bin/env python3
"""Start NIS using the correct venv uvicorn binary directly."""
import subprocess, time

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_batch(commands, timeout=120):
    script = "; ".join(commands)
    r = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(out[:1200])
    if err and "Warning" not in err and "known hosts" not in err:
        print("ERR:", err[:200])
    return r.returncode, out

print("[1] Verify venv uvicorn binary...")
ssh_batch([
    "ls -la /data/organica-ai/NIS_Protocol/venv/bin/uvicorn",
    "/data/organica-ai/NIS_Protocol/venv/bin/uvicorn --version",
    "ls /data/organica-ai/NIS_Protocol/main.py",
])

print()
print("[2] Start NIS using venv uvicorn binary directly...")
ssh_batch([
    "fuser -k 8000/tcp 2>/dev/null || true",
    "sleep 1",
    "cd /data/organica-ai/NIS_Protocol && nohup /data/organica-ai/NIS_Protocol/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/nis.log 2>&1 &",
    "echo pid=$!",
    "sleep 10",
    "curl -s --max-time 8 http://localhost:8000/health 2>/dev/null || echo still-down",
], timeout=90)

print()
print("[3] Log tail...")
ssh_batch(["tail -20 /tmp/nis.log 2>/dev/null"], timeout=30)

print()
print("[4] Test cosmos/reason endpoint...")
ssh_batch([
    r"""/data/organica-ai/NIS_Protocol/venv/bin/python -c "
import urllib.request, json, base64, io
from PIL import Image
img = Image.new('RGB', (64,64), (100,150,200))
buf = io.BytesIO(); img.save(buf, 'JPEG')
b64 = base64.b64encode(buf.getvalue()).decode()
payload = json.dumps({'image_data': b64, 'task': 'describe scene', 'constraints': []}).encode()
req = urllib.request.Request('http://localhost:8000/cosmos/reason', data=payload, headers={'Content-Type':'application/json'}, method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=20)
    d = json.loads(resp.read())
    print('OK - keys:', list(d.keys())[:8], 'source:', d.get('source'))
except Exception as e:
    print('error:', str(e)[:150])
" 2>/dev/null""",
], timeout=40)
