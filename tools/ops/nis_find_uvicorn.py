#!/usr/bin/env python3
"""Find uvicorn and start NIS Protocol on H100."""
import subprocess, time

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_batch(commands, timeout=120):
    script = "; ".join(commands)
    r = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(out[:1500])
    if err and "Warning" not in err and "known hosts" not in err:
        print("ERR:", err[:200])
    return r.returncode, out

print("=== Find uvicorn everywhere ===")
ssh_batch([
    "find / -name 'uvicorn' -not -path '*/proc/*' 2>/dev/null | head -10",
    "/home/nvidia/.local/bin/uv run python -m uvicorn --version 2>/dev/null || echo uv-no-uvicorn",
    "ls /home/nvidia/.local/bin/ 2>/dev/null",
    # Check if uv has a project venv
    "ls ~/organica-ai/NIS_Protocol/.venv/bin/ 2>/dev/null | head -10 || echo no-project-venv",
    "cat ~/organica-ai/NIS_Protocol/pyproject.toml 2>/dev/null | head -20 || echo no-pyproject",
], timeout=120)
