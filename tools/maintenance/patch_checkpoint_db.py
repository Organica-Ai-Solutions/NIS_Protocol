#!/usr/bin/env python3
"""Patch checkpoint_db.py to skip HF metadata fetch when file already exists locally."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:2000])
    return r.returncode, out

print("=== Read path() method in checkpoint_db.py ===")
ssh_one("grep -n 'def path\|def _download\|hf_hub_download\|snapshot_download\|local_path\|cache_dir\|exists\|return' "
        "/data/organica-ai/cosmos-transfer2.5/cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py "
        "2>/dev/null | head -50")

print()
print("=== Read lines around path() method ===")
ssh_one("sed -n '160,220p' "
        "/data/organica-ai/cosmos-transfer2.5/cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py "
        "2>/dev/null")
