#!/usr/bin/env python3
"""Deploy fixed cosmos_transfer25_server.py to H100 and restart."""
import subprocess, time

SCP = ["scp", "-o", "ConnectTimeout=60", "-o", "StrictHostKeyChecking=no"]
SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:800])
    return r.returncode, out

print("[1] Upload fixed cosmos_transfer25_server.py...")
r = subprocess.run(
    SCP + [r"S:\CascadeProjects\cosmos-setup\cosmos_transfer25_server.py",
           "awesome-gpu-name:/data/organica-ai/cosmos_transfer25_server.py"],
    capture_output=True, text=True, timeout=30
)
print(f"  SCP: {'OK' if r.returncode == 0 else 'FAILED - ' + r.stderr[:100]}")

print()
print("[2] Kill old Transfer2.5 server and restart...")
ssh_one("pkill -f cosmos_transfer25_server 2>/dev/null || true")
time.sleep(2)
ssh_one(
    "CUDA_VISIBLE_DEVICES=3 nohup /home/nvidia/organica-ai/venv/bin/python3 "
    "/data/organica-ai/cosmos_transfer25_server.py > /tmp/transfer_server.log 2>&1 & "
    "echo pid=$!"
)
time.sleep(5)
ssh_one("curl -s --max-time 5 http://localhost:8300/health 2>/dev/null || echo still-starting")

print()
print("[3] Test with HF_HUB_OFFLINE=1 (30s timeout to see if download skipped)...")
ssh_one(r"""cd /data/organica-ai/cosmos-transfer2.5 && timeout 30 env \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=3 \
  .venv/bin/python3 examples/inference.py \
  -i assets/car_example/edge/car_edge_spec.json \
  -o /tmp/transfer_offline_test \
  --model edge \
  --checkpoint-path /data/organica-ai/models/cosmos/transfer2.5/2B/general/edge \
  --disable-guardrails \
  control:edge 2>&1 | head -20 || echo "exit:$?" """, timeout=40)
