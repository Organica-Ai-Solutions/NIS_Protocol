#!/usr/bin/env python3
"""Run Transfer2.5 inference to completion on H100 (up to 15 min)."""
import subprocess, time

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=30",
       "-o", "TCPKeepAlive=yes", "awesome-gpu-name"]

def ssh_one(cmd, timeout=960):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:3000])
    return r.returncode, out

print("Running Transfer2.5 inference (up to 15 min)...")
t0 = time.time()
ssh_one(r"""cd /data/organica-ai/cosmos-transfer2.5 && \
  .venv/bin/python3 examples/inference.py \
  -i assets/car_example/edge/car_edge_spec.json \
  -o /tmp/transfer_final \
  --model edge \
  --checkpoint-path /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge \
  --disable-guardrails \
  control:edge 2>&1 | grep -v 'Fetching\|it/s\|0%\|100%\|Downloading checkpoint file' | tail -30; \
  echo "exit:$?" """, timeout=960)

print(f"\nElapsed: {time.time()-t0:.0f}s")
print()
ssh_one("ls -lh /tmp/transfer_final/*.mp4 2>/dev/null || echo no-mp4-output")
