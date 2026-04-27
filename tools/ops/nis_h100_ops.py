#!/usr/bin/env python3
"""NIS Protocol H100 operations via paramiko over SSH config."""
import subprocess, time, sys

# Use the SSH config alias - run commands via ssh directly but with longer timeouts
# and batch them to minimize round trips

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_batch(commands, timeout=120):
    """Run multiple commands in one SSH session."""
    script = " && ".join(f"({c})" for c in commands)
    r = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    return r.returncode, out, err

print("=== Batch probe H100 ===")
rc, out, err = ssh_batch([
    "which python3",
    "find ~/organica-ai -maxdepth 3 -name 'uvicorn' 2>/dev/null | head -3",
    "find ~/organica-ai/NIS_Protocol -name 'main.py' -maxdepth 2 2>/dev/null",
    "ss -tlnp 2>/dev/null | grep 8000 || echo port-8000-free",
    "curl -s --max-time 3 http://localhost:8000/health 2>/dev/null || echo nis-down",
    "tmux list-sessions 2>/dev/null | head -8",
], timeout=120)
print(out[:1000] if out else "(no output)")
if err and "Warning" not in err: print("ERR:", err[:200])
