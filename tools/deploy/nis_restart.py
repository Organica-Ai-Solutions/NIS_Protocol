#!/usr/bin/env python3
"""Restart NIS Protocol on H100 in tmux, then verify."""
import subprocess, time

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_batch(commands, timeout=120):
    script = "; ".join(commands)
    r = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(out[:800])
    if err and "Warning" not in err and "known hosts" not in err:
        print("ERR:", err[:200])
    return r.returncode, out

print("[1] Find pip/uvicorn on H100...")
ssh_batch([
    "pip3 show uvicorn 2>/dev/null | head -2 || echo no-uvicorn-pip3",
    "pip show uvicorn 2>/dev/null | head -2 || echo no-uvicorn-pip",
    "python3 -m uvicorn --version 2>/dev/null || echo no-uvicorn-module",
    "ls ~/organica-ai/NIS_Protocol/requirements*.txt 2>/dev/null | head -3",
])

print()
print("[2] Install uvicorn + fastapi if missing, then start NIS...")
ssh_batch([
    "pip3 install uvicorn fastapi 2>/dev/null | tail -3 || true",
    "tmux send-keys -t kan2 '' Enter",  # wake session
    "tmux send-keys -t kan2 'cd ~/organica-ai/NIS_Protocol && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/nis.log 2>&1 &' Enter",
    "sleep 8",
    "curl -s --max-time 6 http://localhost:8000/health 2>/dev/null || echo still-down",
    "tail -10 /tmp/nis.log 2>/dev/null || echo no-log",
], timeout=90)

print()
print("[3] Verify cosmos/reason fix...")
ssh_batch([
    r"""python3 -c "
import urllib.request, json, base64, io
try:
    from PIL import Image
    img = Image.new('RGB', (32,32), (100,150,200))
    buf = io.BytesIO(); img.save(buf, 'JPEG')
    b64 = base64.b64encode(buf.getvalue()).decode()
except:
    b64 = ''
payload = json.dumps({'image_data': b64, 'task': 'describe scene', 'constraints': []}).encode()
req = urllib.request.Request('http://localhost:8000/cosmos/reason', data=payload, headers={'Content-Type':'application/json'}, method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=15)
    d = json.loads(resp.read())
    print('cosmos/reason OK keys:', list(d.keys())[:8])
except Exception as e:
    print('cosmos/reason:', str(e)[:120])
" 2>/dev/null"""
], timeout=30)
