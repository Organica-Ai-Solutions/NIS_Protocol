#!/usr/bin/env python3
"""Upload cosmos.py fix to /data path and start NIS with correct venv."""
import subprocess, time, os

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]
SCP = ["scp", "-o", "ConnectTimeout=60", "-o", "StrictHostKeyChecking=no"]

LOCAL_COSMOS = r"C:\Users\DiegoTorres\Desktop\NIS_Protocol\routes\cosmos.py"
REMOTE_DATA_COSMOS = "awesome-gpu-name:/data/organica-ai/NIS_Protocol/routes/cosmos.py"
REMOTE_HOME_COSMOS = "awesome-gpu-name:~/organica-ai/NIS_Protocol/routes/cosmos.py"

VENV_PYTHON = "/data/organica-ai/NIS_Protocol/venv/bin/python"
NIS_DIR = "/data/organica-ai/NIS_Protocol"

def ssh_batch(commands, timeout=120):
    script = "; ".join(commands)
    r = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(out[:1000])
    if err and "Warning" not in err and "known hosts" not in err:
        print("ERR:", err[:200])
    return r.returncode, out

def scp_upload(local, remote, timeout=30):
    r = subprocess.run(SCP + [local, remote], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        err = r.stderr.strip()
        if err and "Warning" not in err: print(f"  SCP ERR: {err[:200]}")
        return False
    return True

print("[1] Upload cosmos.py fix to both NIS locations...")
ok1 = scp_upload(LOCAL_COSMOS, REMOTE_DATA_COSMOS)
ok2 = scp_upload(LOCAL_COSMOS, REMOTE_HOME_COSMOS)
print(f"  /data path: {'OK' if ok1 else 'FAILED'}")
print(f"  ~/organica-ai path: {'OK' if ok2 else 'FAILED'}")

print()
print("[2] Kill any existing process on port 8000...")
ssh_batch([
    "fuser -k 8000/tcp 2>/dev/null || true",
    "sleep 2",
])

print()
print("[3] Start NIS Protocol with /data venv...")
ssh_batch([
    f"cd {NIS_DIR} && nohup {VENV_PYTHON} -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/nis.log 2>&1 &",
    "echo NIS started pid=$!",
    "sleep 10",
    "curl -s --max-time 8 http://localhost:8000/health 2>/dev/null || echo still-down",
], timeout=90)

print()
print("[4] Startup log...")
ssh_batch(["tail -20 /tmp/nis.log 2>/dev/null || echo no-log"], timeout=30)

print()
print("[5] Test cosmos/reason with real image...")
test_script = r"""python3 -c "
import urllib.request, json, base64, io
try:
    from PIL import Image
    img = Image.new('RGB', (64,64), (100,150,200))
    buf = io.BytesIO(); img.save(buf, 'JPEG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    print('image_b64_len:', len(b64))
except Exception as e:
    b64 = ''; print('PIL error:', e)
payload = json.dumps({'image_data': b64, 'task': 'describe the scene', 'constraints': []}).encode()
req = urllib.request.Request('http://localhost:8000/cosmos/reason', data=payload, headers={'Content-Type':'application/json'}, method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=20)
    d = json.loads(resp.read())
    print('OK keys:', list(d.keys())[:8])
    print('error:', d.get('error'), 'source:', d.get('source'))
except Exception as e:
    print('error:', str(e)[:150])
" 2>/dev/null"""
ssh_batch([test_script], timeout=40)
