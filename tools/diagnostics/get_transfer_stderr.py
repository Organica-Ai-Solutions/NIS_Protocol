#!/usr/bin/env python3
"""Get the actual stderr the server sees when Transfer2.5 fails."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=90):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:3000])
    return r.returncode, out

# Run exactly what the server runs, capture stderr, 30s only to see early failure
print("=== Exact server command, 30s, full stderr ===")
ssh_one(r"""cd /data/organica-ai/cosmos-transfer2.5 && timeout 30 \
  .venv/bin/python3 examples/inference.py \
  -i assets/car_example/edge/car_edge_spec.json \
  -o /tmp/srv_test \
  --model edge \
  --checkpoint-path /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge \
  --disable-guardrails \
  control:edge > /tmp/srv_stdout.txt 2> /tmp/srv_stderr.txt; \
  echo "exit:$?"; \
  echo "=STDOUT="; tail -10 /tmp/srv_stdout.txt; \
  echo "=STDERR="; tail -20 /tmp/srv_stderr.txt""", timeout=40)
