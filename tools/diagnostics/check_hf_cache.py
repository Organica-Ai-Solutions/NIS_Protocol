#!/usr/bin/env python3
"""Check HF cache and guardrail checkpoint status on H100."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:2000])
    return r.returncode, out

print("=== HF cache locations ===")
ssh_one("ls ~/.cache/huggingface/hub/ 2>/dev/null | head -20 || echo no-hf-cache")
ssh_one("ls /data/organica-ai/.cache/huggingface/hub/ 2>/dev/null | head -20 || echo no-data-cache")
ssh_one("echo HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE")

print()
print("=== checkpoint_db.py path logic ===")
ssh_one("grep -n 'path\|cache\|download\|HF_HUB\|OFFLINE' /data/organica-ai/cosmos-transfer2.5/cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py 2>/dev/null | head -30")

print()
print("=== Check if guardrail already cached ===")
ssh_one("find ~/.cache /data -name '*Guardrail*' -o -name '*guardrail*' 2>/dev/null | head -10")
ssh_one("find ~/.cache /data -path '*Cosmos-Guardrail*' 2>/dev/null | head -5")
