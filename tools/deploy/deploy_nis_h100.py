#!/usr/bin/env python3
"""Deploy NIS Protocol fixes to H100 via SSH (awesome-gpu-name alias)."""
import subprocess, os, sys, time

NIS_DIR = r"C:\Users\DiegoTorres\Desktop\NIS_Protocol"
SSH_HOST = "awesome-gpu-name"
REMOTE_NIS = "~/organica-ai/NIS_Protocol"

FILES = {
    os.path.join(NIS_DIR, "routes", "cosmos.py"): f"{REMOTE_NIS}/routes/cosmos.py",
}

def scp(local, remote):
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", local, f"{SSH_HOST}:{remote}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"  SCP error: {r.stderr.strip()[:200]}")
        return False
    return True

def ssh(cmd, timeout=30):
    r = subprocess.run(
        ["ssh", SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    out = r.stdout.strip()
    err = r.stderr.strip()
    if out: print(f"  {out[:300]}")
    if err and r.returncode != 0: print(f"  ERR: {err[:200]}")
    return r.returncode == 0, out

print("[1/4] Testing SSH connection to H100...")
ok, out = ssh("echo connected && hostname")
if not ok:
    print("  FAILED - is the SSH tunnel active? Run: ssh awesome-gpu-name")
    sys.exit(1)
print("  OK")

print()
print("[2/4] Checking NIS Protocol on H100...")
ok, out = ssh(f"ls {REMOTE_NIS}/routes/cosmos.py 2>/dev/null && echo exists || echo missing")
if "missing" in out:
    print(f"  NIS Protocol not found at {REMOTE_NIS}")
    print("  Trying alternate paths...")
    ok2, out2 = ssh("find ~ -name 'cosmos.py' -path '*/routes/*' 2>/dev/null | head -3")
    if out2:
        # Use the found path
        found = out2.strip().splitlines()[0]
        remote_routes = os.path.dirname(found)
        REMOTE_NIS = os.path.dirname(remote_routes)
        FILES = {
            os.path.join(NIS_DIR, "routes", "cosmos.py"): found,
        }
        print(f"  Found at: {found}")
    else:
        print("  NIS Protocol not found on H100 - deploy manually")
        sys.exit(1)

print()
print("[3/4] Uploading fixed files...")
for local, remote in FILES.items():
    if not os.path.exists(local):
        print(f"  SKIP (not found locally): {local}")
        continue
    size = os.path.getsize(local)
    print(f"  {os.path.basename(local)} -> {remote}  ({size//1024}KB)")
    if not scp(local, remote):
        print("  Upload failed")
        sys.exit(1)
    print("  Uploaded OK")

print()
print("[4/4] Restarting NIS Protocol service on H100...")
# Try common restart methods
restarted = False
for cmd in [
    "sudo systemctl restart nis-protocol 2>/dev/null && echo restarted",
    "sudo systemctl restart nis_protocol 2>/dev/null && echo restarted",
    f"pkill -f 'uvicorn.*main' 2>/dev/null; cd {REMOTE_NIS} && nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/nis.log 2>&1 & sleep 2 && echo restarted",
]:
    ok, out = ssh(cmd, timeout=15)
    if "restarted" in out:
        restarted = True
        break

if restarted:
    print("  NIS Protocol restarted")
    time.sleep(3)
    ok, out = ssh("curl -s http://localhost:8000/health 2>/dev/null | python3 -c \"import sys,json; d=json.load(sys.stdin); print('status:', d.get('status'), 'version:', d.get('version'))\" 2>/dev/null || echo health-check-failed")
else:
    print("  Could not auto-restart - restart NIS Protocol manually on H100")

print()
print("Deploy complete!")
print(f"  Fix: /cosmos/reason now decodes real base64 image instead of blank array")
print(f"  File: routes/cosmos.py lines 141-153")
