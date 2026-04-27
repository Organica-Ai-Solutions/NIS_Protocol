#!/usr/bin/env python3
"""Get the actual stderr from the transfer subprocess that causes 500."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=60):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:2500])
    return r.returncode, out

# Run with --disable-guardrails and capture full output for 25s
print("=== Run with --disable-guardrails, 25s timeout ===")
ssh_one(r"""cd /data/organica-ai/cosmos-transfer2.5 && timeout 25 \
  .venv/bin/python3 examples/inference.py \
  -i assets/car_example/edge/car_edge_spec.json \
  -o /tmp/transfer_test2 \
  --model edge \
  --checkpoint-path /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge \
  --disable-guardrails \
  control:edge 2>&1 | tail -20 || echo "exit:$?" """)

print()
print("=== Check console.log ===")
ssh_one("cat /tmp/transfer_test2/console.log 2>/dev/null | tail -20 || echo no-log")

print()
print("=== Check output dir ===")
ssh_one("ls -la /tmp/transfer_test2/ 2>/dev/null || echo empty")
