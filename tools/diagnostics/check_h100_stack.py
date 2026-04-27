#!/usr/bin/env python3
"""Check full Cosmos stack status on H100."""
import subprocess

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=45):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(f"  {out[:1000]}")
    return r.returncode, out

print("=== Port status (8000/8100/8200/8300/8400) ===")
ssh_one("ss -tlnp 2>/dev/null | grep -E '8000|8100|8200|8300|8400' || echo none-listening")

print()
print("=== Health checks ===")
for port, name in [(8000,"NIS"),(8100,"Reason2"),(8200,"Predict2.5"),(8300,"Transfer2.5"),(8400,"Demo")]:
    ssh_one(f"curl -s --max-time 4 http://localhost:{port}/health 2>/dev/null | head -c 150 || echo {name}:{port}-down")

print()
print("=== GPU usage ===")
ssh_one("nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | head -8")

print()
print("=== Running cosmos processes ===")
ssh_one("ps aux | grep -E 'cosmos|uvicorn|NIS_Protocol' | grep -v grep | awk '{print $1,$2,$11,$12}' | head -15")

print()
print("=== Tmux sessions ===")
ssh_one("tmux list-sessions 2>/dev/null || echo no-tmux")
