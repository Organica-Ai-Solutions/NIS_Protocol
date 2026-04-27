#!/usr/bin/env python3
"""Check server logs for Predict2.5 and Transfer2.5 500 errors."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:1200])
    return r.returncode, out

print("=== Predict2.5 log (last 30 lines) ===")
ssh_one("tail -30 /tmp/predict_server.log 2>/dev/null || echo no-log")

print()
print("=== Transfer2.5 log (last 30 lines) ===")
ssh_one("tail -30 /tmp/transfer_server.log 2>/dev/null || echo no-log")
