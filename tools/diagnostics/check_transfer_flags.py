#!/usr/bin/env python3
"""Check full inference.py help and get the actual 500 stderr."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:2000])
    return r.returncode, out

print("=== Full inference.py help ===")
ssh_one("/data/organica-ai/cosmos-transfer2.5/.venv/bin/python3 /data/organica-ai/cosmos-transfer2.5/examples/inference.py --help 2>&1 | grep -E 'guardrail|safety|disable|checkpoint' | head -20")

print()
print("=== Run inference directly and capture stderr ===")
ssh_one(r"""cd /data/organica-ai/cosmos-transfer2.5 && timeout 20 \
  .venv/bin/python3 examples/inference.py \
  -i assets/car_example/edge/car_edge_spec.json \
  -o /tmp/transfer_test_out \
  --model edge \
  --checkpoint-path /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge \
  control:edge 2>&1 | head -30 || echo "exit code: $?" """)
