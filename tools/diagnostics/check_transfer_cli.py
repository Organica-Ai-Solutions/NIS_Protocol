#!/usr/bin/env python3
"""Check Transfer2.5 inference.py CLI args and run a quick test."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:2000])
    return r.returncode, out

print("=== inference.py --help ===")
ssh_one("/data/organica-ai/cosmos-transfer2.5/.venv/bin/python3 /data/organica-ai/cosmos-transfer2.5/examples/inference.py --help 2>&1 | head -40")

print()
print("=== spec file contents ===")
ssh_one("cat /data/organica-ai/cosmos-transfer2.5/assets/car_example/edge/car_edge_spec.json")

print()
print("=== Check what args the server is actually passing ===")
# Reconstruct the command from the server code
ssh_one(r"""python3 -c "
import json
spec = '/data/organica-ai/cosmos-transfer2.5/assets/car_example/edge/car_edge_spec.json'
checkpoint = '/data/organica-ai/models/cosmos/transfer2.5/2B/general/edge'
venv_py = '/data/organica-ai/cosmos-transfer2.5/.venv/bin/python3'
cmd = [venv_py, 'examples/inference.py', '-i', spec, '-o', '/tmp/test_out',
       '--model', 'edge', '--checkpoint-path', checkpoint,
       '--disable-guardrails', 'control:edge']
print('CMD:', ' '.join(cmd))
" 2>/dev/null""")
