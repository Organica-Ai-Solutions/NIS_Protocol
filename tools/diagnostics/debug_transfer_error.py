#!/usr/bin/env python3
"""Get full Transfer2.5 500 error body and check server log."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=90):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:2000])
    return r.returncode, out

print("=== Transfer2.5 /transfer error body ===")
ssh_one(r"""/data/organica-ai/NIS_Protocol/venv/bin/python -c "
import urllib.request, json
payload = json.dumps({'demo': 'car_edge', 'control_type': 'edge', 'guidance': 3.0}).encode()
req = urllib.request.Request('http://localhost:8300/transfer', data=payload,
    headers={'Content-Type': 'application/json'}, method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=30)
    d = json.loads(resp.read())
    print('OK:', str(d)[:300])
except urllib.request.HTTPError as e:
    print('HTTP', e.code, ':', e.read().decode(errors='replace')[:800])
except Exception as e:
    print('error:', str(e)[:200])
" 2>/dev/null""")

print()
print("=== Transfer server log tail ===")
ssh_one("tail -40 /tmp/transfer_server.log 2>/dev/null | grep -v 'GET /health' | tail -25")

print()
print("=== Check spec file exists ===")
ssh_one("ls -la /data/organica-ai/cosmos-transfer2.5/assets/car_example/edge/ 2>/dev/null || echo missing")
ssh_one("ls -la /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge/ 2>/dev/null | head -5 || echo checkpoint-missing")
