#!/usr/bin/env python3
"""Find and restart NIS Protocol on H100."""
import subprocess, time

SSH_HOST = "awesome-gpu-name"

def ssh(cmd, timeout=30):
    r = subprocess.run(
        ["ssh", SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(f"  {out[:400]}")
    return r.returncode, out, err

print("=== Finding NIS Protocol process ===")
ssh("ps aux | grep -E 'uvicorn|main:app|nis' | grep -v grep | head -10")

print()
print("=== Checking systemd services ===")
ssh("systemctl list-units --type=service --state=running 2>/dev/null | grep -iE 'nis|protocol|uvicorn|fastapi' | head -10 || echo none")

print()
print("=== Checking tmux sessions ===")
ssh("tmux list-sessions 2>/dev/null || echo no-tmux")

print()
print("=== Checking screen sessions ===")
ssh("screen -ls 2>/dev/null || echo no-screen")

print()
print("=== Checking /tmp/nis.log ===")
ssh("tail -5 /tmp/nis.log 2>/dev/null || echo no-log")

print()
print("=== Current NIS health ===")
rc, out, _ = ssh("curl -s --max-time 5 http://localhost:8000/health 2>/dev/null || echo unreachable")
print()

print("=== Attempting restart ===")
# Kill existing uvicorn on port 8000
ssh("fuser -k 8000/tcp 2>/dev/null || true")
time.sleep(2)

# Start NIS in background via nohup
rc, out, err = ssh(
    "cd ~/organica-ai/NIS_Protocol && "
    "nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 "
    "> /tmp/nis_restart.log 2>&1 & echo pid=$!",
    timeout=15
)
print()
time.sleep(5)

print("=== Health after restart ===")
ssh("curl -s --max-time 8 http://localhost:8000/health 2>/dev/null || echo still-unreachable")

print()
print("=== Startup log ===")
ssh("tail -15 /tmp/nis_restart.log 2>/dev/null || echo no-log")
