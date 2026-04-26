#!/usr/bin/env python3
"""Start NIS Protocol using the correct .venv on H100."""
import subprocess, time

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_batch(commands, timeout=120):
    script = "; ".join(commands)
    r = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(out[:1000])
    if err and "Warning" not in err and "known hosts" not in err:
        print("ERR:", err[:200])
    return r.returncode, out

print("[1] Verify .venv has uvicorn...")
ssh_batch([
    "/home/nvidia/.venv/bin/python -m uvicorn --version 2>/dev/null || echo no-uvicorn-in-venv",
    "/home/nvidia/.venv/bin/pip show uvicorn 2>/dev/null | head -2 || echo not-found",
])

print()
print("[2] Start NIS Protocol with .venv...")
ssh_batch([
    "fuser -k 8000/tcp 2>/dev/null || true",
    "sleep 1",
    "cd ~/organica-ai/NIS_Protocol && nohup /home/nvidia/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/nis.log 2>&1 &",
    "echo started pid=$!",
    "sleep 8",
    "curl -s --max-time 6 http://localhost:8000/health 2>/dev/null || echo still-down",
], timeout=90)

print()
print("[3] Check startup log...")
ssh_batch([
    "tail -15 /tmp/nis.log 2>/dev/null || echo no-log",
], timeout=30)

print()
print("[4] Verify cosmos/reason fix with real image...")
ssh_batch([
    r"""python3 -c "
import urllib.request, json, base64, io
try:
    from PIL import Image
    img = Image.new('RGB', (32,32), (100,150,200))
    buf = io.BytesIO(); img.save(buf, 'JPEG')
    b64 = base64.b64encode(buf.getvalue()).decode()
except Exception as e:
    b64 = ''
    print('PIL error:', e)
payload = json.dumps({'image_data': b64, 'task': 'describe scene', 'constraints': []}).encode()
req = urllib.request.Request('http://localhost:8000/cosmos/reason', data=payload, headers={'Content-Type':'application/json'}, method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=15)
    d = json.loads(resp.read())
    print('cosmos/reason OK - keys:', list(d.keys())[:8])
    print('has_error:', d.get('error'), 'source:', d.get('source'))
except Exception as e:
    print('cosmos/reason error:', str(e)[:150])
" 2>/dev/null""",
], timeout=30)
