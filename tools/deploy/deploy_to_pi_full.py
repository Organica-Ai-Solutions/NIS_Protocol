#!/usr/bin/env python3
"""
Full deploy to Pi — pushes all fixed files and restarts agent.
Run this whenever Pi comes online: python deploy_to_pi_full.py
"""
import subprocess, sys, time, os

PI = "pi@neurolinux.local"
OVERLAY = r"C:\Users\DiegoTorres\Desktop\NeuroLinux\neurolinux-os\buildroot\board\neurolinux\overlay\opt\neurolinux"
NIS_DIR = r"C:\Users\DiegoTorres\Desktop\NIS_Protocol"

SSH_OPTS = ["-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
            "-o", "StrictHostKeyChecking=no"]

def scp(local, remote):
    r = subprocess.run(
        ["scp"] + SSH_OPTS + [local, f"{PI}:{remote}"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  SCP FAIL: {r.stderr.strip()[:100]}")
        return False
    return True

def ssh(cmd):
    r = subprocess.run(
        ["ssh"] + SSH_OPTS + [PI, cmd],
        capture_output=True, text=True, timeout=30
    )
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0, out

# ── 1. Check Pi reachable ─────────────────────────────────────────────────────
print("[1] Checking Pi connectivity...")
ok, out = ssh("echo alive && hostname")
if not ok or "alive" not in out:
    print(f"  Pi unreachable: {out}")
    sys.exit(1)
print(f"  Connected: {out.strip()}")

# ── 2. Deploy files ───────────────────────────────────────────────────────────
print("\n[2] Deploying files...")

files = [
    (os.path.join(OVERLAY, "cosmos_cookoff.py"),   "/opt/neurolinux/cosmos_cookoff.py"),
    (os.path.join(OVERLAY, "static", "index.html"), "/opt/neurolinux/static/index.html"),
]

for local, remote in files:
    size = os.path.getsize(local) // 1024
    print(f"  {os.path.basename(local)} ({size}KB)...", end=" ")
    if scp(local, remote):
        print("OK")
    else:
        print("FAILED")
        sys.exit(1)

# ── 3. Restart agent ──────────────────────────────────────────────────────────
print("\n[3] Restarting neurolinux-agent...")
ok, out = ssh("sudo systemctl restart neurolinux-agent && sleep 4 && systemctl is-active neurolinux-agent")
print(f"  Status: {out.strip()}")

# ── 4. Verify ─────────────────────────────────────────────────────────────────
print("\n[4] Verifying...")
time.sleep(2)
ok, out = ssh("curl -s http://localhost:8085/health")
if ok and "version" in out:
    import json
    try:
        d = json.loads(out)
        xarm = d.get("xarm")
        cam = d.get("camera", {}).get("available")
        nis = d.get("nis_url")
        print(f"  Agent v{d.get('version')} — xarm={xarm} cam={cam} nis={nis}")
    except Exception:
        print(f"  {out[:120]}")
else:
    print(f"  Health check failed: {out[:80]}")

# ── 5. Quick arm test ─────────────────────────────────────────────────────────
print("\n[5] xArm quick test (home position)...")
ok, out = ssh("curl -s -X POST http://localhost:8085/agent/chat -H 'Content-Type: application/json' -d '{\"message\":\"xarm home\"}'")
if ok:
    try:
        d = json.loads(out)
        print(f"  {d.get('response','?')} — ok={d.get('ok')}")
    except Exception:
        print(f"  {out[:120]}")
else:
    print(f"  {out[:80]}")

print("\n✅ Deploy complete. Pi is ready.")
print("   To run full demo: python test_full_cookoff_demo.py (from Pi)")
print("   Keep start_tunnel.ps1 running for H100 Cosmos access.")
