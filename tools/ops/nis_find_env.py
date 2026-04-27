#!/usr/bin/env python3
"""Find the correct Python env NIS Protocol was using on H100."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_batch(commands, timeout=120):
    script = "; ".join(commands)
    r = subprocess.run(SSH + [script], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(out[:1200])
    if err and "Warning" not in err and "known hosts" not in err:
        print("ERR:", err[:300])
    return r.returncode, out

print("=== Find all uvicorn installs ===")
ssh_batch([
    "find /home/nvidia /usr /opt /root -name 'uvicorn' -type f 2>/dev/null | head -10",
    "find /home/nvidia -name 'activate' -path '*/bin/activate' 2>/dev/null | head -10",
    "ls /home/nvidia/ 2>/dev/null",
    "ls /data/ 2>/dev/null | head -10",
])

print()
print("=== Check conda ===")
ssh_batch([
    "conda env list 2>/dev/null || echo no-conda",
    "which conda 2>/dev/null || echo no-conda-bin",
    "ls ~/miniconda3/envs/ 2>/dev/null || ls ~/anaconda3/envs/ 2>/dev/null || echo no-conda-envs",
])

print()
print("=== Check pip locations ===")
ssh_batch([
    "find /home/nvidia -name 'pip' -o -name 'pip3' 2>/dev/null | head -10",
    "ls /home/nvidia/.local/bin/ 2>/dev/null | head -20",
])

print()
print("=== Check NIS start.sh for clues ===")
ssh_batch([
    "cat ~/organica-ai/NIS_Protocol/start.sh 2>/dev/null | head -30",
])
