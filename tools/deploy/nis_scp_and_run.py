#!/usr/bin/env python3
"""SCP startup script to H100 then execute it."""
import subprocess, time

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]
SCP = ["scp", "-o", "ConnectTimeout=60", "-o", "StrictHostKeyChecking=no"]

def ssh_one(cmd, timeout=60):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(f"  {out[:800]}")
    if err and "Warning" not in err and "known hosts" not in err:
        print(f"  ERR: {err[:200]}")
    return r.returncode, out

print("[1] Upload startup script...")
r = subprocess.run(
    SCP + [r"C:\Users\DiegoTorres\Desktop\NIS_Protocol\nis_start_remote.sh",
           "awesome-gpu-name:/tmp/nis_start.sh"],
    capture_output=True, text=True, timeout=30
)
print(f"  SCP: {'OK' if r.returncode == 0 else 'FAILED - ' + r.stderr[:100]}")

print()
print("[2] Kill port 8000 and run startup script...")
ssh_one("fuser -k 8000/tcp 2>/dev/null || true", timeout=15)
time.sleep(2)
ssh_one("bash /tmp/nis_start.sh", timeout=15)

print()
print("[3] Wait for NIS to start...")
time.sleep(12)
ssh_one("curl -s --max-time 8 http://localhost:8000/health 2>/dev/null || echo down", timeout=25)

print()
print("[4] Log...")
ssh_one("tail -20 /tmp/nis.log 2>/dev/null || echo no-log", timeout=15)

print()
print("[5] Test cosmos/reason with real image...")
ssh_one(
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
    print('cosmos/reason OK - keys:', list(d.keys())[:8])
    print('source:', d.get('source'), 'error:', d.get('error'))
except Exception as e:
    print('error:', str(e)[:150])
" 2>/dev/null""",
    timeout=40
)
