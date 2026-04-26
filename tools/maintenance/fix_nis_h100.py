#!/usr/bin/env python3
"""Find correct venv and restart NIS Protocol on H100."""
import subprocess, time

SSH_HOST = "awesome-gpu-name"

def ssh(cmd, timeout=45):
    r = subprocess.run(["ssh", SSH_HOST, cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(f"  {out[:500]}")
    return r.returncode, out

print("=== Find Python / venv ===")
ssh("which python3 && python3 --version")
ssh("ls ~/organica-ai/ 2>/dev/null || echo no-dir")
ssh("find ~/organica-ai -name 'uvicorn' 2>/dev/null | head -5")
ssh("find /home/nvidia -maxdepth 4 -name 'uvicorn' 2>/dev/null | head -5")

print()
print("=== Find main.py ===")
ssh("find ~/organica-ai/NIS_Protocol -name 'main.py' -maxdepth 2 2>/dev/null | head -5")

print()
print("=== Check port 8000 ===")
ssh("ss -tlnp 2>/dev/null | grep 8000 || echo port-8000-free")
