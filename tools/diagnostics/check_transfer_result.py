#!/usr/bin/env python3
"""Check if transfer inference completed and what the actual failure is."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=120):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:3000])
    return r.returncode, out

print("=== Run full inference (120s), capture last 40 lines ===")
ssh_one(r"""cd /data/organica-ai/cosmos-transfer2.5 && timeout 120 \
  .venv/bin/python3 examples/inference.py \
  -i assets/car_example/edge/car_edge_spec.json \
  -o /tmp/transfer_result_test \
  --model edge \
  --checkpoint-path /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge \
  --disable-guardrails \
  control:edge 2>&1 | grep -v 'Fetching\|it/s\|0%\|100%' | tail -40; echo "exit:$?" """, timeout=130)

print()
print("=== Output files ===")
ssh_one("ls -la /tmp/transfer_result_test/ 2>/dev/null || echo empty")
