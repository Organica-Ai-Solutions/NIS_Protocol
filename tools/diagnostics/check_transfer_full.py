#!/usr/bin/env python3
"""Get full stderr from Transfer2.5 inference to find the real error."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=90):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:3000])
    return r.returncode, out

print("=== Run inference, capture full output, 60s timeout ===")
ssh_one(r"""cd /data/organica-ai/cosmos-transfer2.5 && timeout 60 \
  .venv/bin/python3 examples/inference.py \
  -i assets/car_example/edge/car_edge_spec.json \
  -o /tmp/transfer_full_test \
  --model edge \
  --checkpoint-path /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge \
  --disable-guardrails \
  control:edge 2>&1 | grep -v 'Fetching\|Downloading\|100%\|0%\|it/s' | tail -30 || echo "exit:$?" """, timeout=75)

print()
print("=== debug.log tail ===")
ssh_one("tail -30 /tmp/transfer_full_test/debug.log 2>/dev/null || echo no-debug-log")
