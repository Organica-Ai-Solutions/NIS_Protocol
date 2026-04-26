"""
push_cookoff_fix.py — Push updated routes/cookoff.py to Pi and restart NIS Protocol.
Uses direct local-network SSH (pi@neurolinux.local or pi@192.168.1.163).
"""
import subprocess, sys, os, time

PI_HOSTS = ["pi@192.168.1.163", "pi@neurolinux.local"]
SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes"]  # key-based auth only

LOCAL_COOKOFF = r"C:\Users\DiegoTorres\Desktop\NIS_Protocol\routes\cookoff.py"

# Try to find the NIS Protocol routes dir on Pi
PI_NIS_PATHS = [
    "/opt/nis-protocol/routes/cookoff.py",
    "/home/pi/NIS_Protocol/routes/cookoff.py",
    "/home/pi/nis_protocol/routes/cookoff.py",
]

def run_ssh(host, cmd, timeout=15):
    r = subprocess.run(["ssh"] + SSH_OPTS + [host, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, (r.stdout + r.stderr).strip()

def run_scp(host, local, remote, timeout=20):
    r = subprocess.run(["scp"] + SSH_OPTS + [local, f"{host}:{remote}"],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stderr.strip()

# --- 1. Find reachable Pi host ---
pi = None
for host in PI_HOSTS:
    print(f"Trying {host}...", end=" ")
    try:
        ok, out = run_ssh(host, "echo alive && hostname", timeout=12)
        if ok and "alive" in out:
            print(f"OK ({out.strip()})")
            pi = host
            break
        else:
            print(f"FAILED: {out[:60]}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    except Exception as e:
        print(f"ERROR: {e}")

if not pi:
    print("\nCannot reach Pi via SSH. Make sure SSH keys are configured.")
    print("Manual option: copy routes/cookoff.py via USB drive or samba share")
    sys.exit(1)

# --- 2. Find where NIS Protocol routes live on Pi ---
print(f"\nFinding NIS Protocol routes on {pi}...")
pi_routes_dir = None
for path in PI_NIS_PATHS:
    ok, out = run_ssh(pi, f"test -f {path} && echo found")
    if ok and "found" in out:
        pi_routes_dir = path
        print(f"  Found: {path}")
        break

if not pi_routes_dir:
    # Try to find it
    ok, out = run_ssh(pi, "find /opt /home -name 'cookoff.py' 2>/dev/null | head -3")
    if ok and out:
        pi_routes_dir = out.strip().split('\n')[0]
        print(f"  Found via search: {pi_routes_dir}")
    else:
        print("  cookoff.py not found on Pi.")
        print("  Trying to find NIS Protocol directory...")
        ok, out = run_ssh(pi, "find /opt /home -name 'main_pi.py' 2>/dev/null | head -3")
        if ok and out:
            pi_dir = os.path.dirname(out.strip().split('\n')[0])
            pi_routes_dir = f"{pi_dir}/routes/cookoff.py"
            print(f"  Using: {pi_routes_dir}")
        else:
            print("  NIS Protocol not found. Check Pi manually.")
            sys.exit(1)

# --- 3. Backup and push ---
print(f"\nBacking up Pi's cookoff.py...")
backup = pi_routes_dir + ".bak"
ok, out = run_ssh(pi, f"cp {pi_routes_dir} {backup} && echo backed_up")
print(f"  Backup: {'OK' if 'backed_up' in out else out}")

print(f"Pushing {LOCAL_COOKOFF}...")
ok, err = run_scp(pi, LOCAL_COOKOFF, pi_routes_dir)
if ok:
    print("  SCP: OK")
else:
    print(f"  SCP FAILED: {err}")
    sys.exit(1)

# --- 4. Restart NIS Protocol ---
print("\nRestarting nis-protocol service on Pi...")
for svc in ["nis-protocol", "nis_protocol", "nis"]:
    ok, out = run_ssh(pi, f"sudo systemctl restart {svc} 2>&1 | head -5")
    if ok:
        print(f"  Restarted {svc}: OK")
        break
    elif "not found" not in out.lower() and "no such" not in out.lower():
        print(f"  {svc}: {out[:80]}")
        break

time.sleep(5)

# --- 5. Verify ---
print("\nVerifying NIS Protocol is running...")
import urllib.request, json
try:
    r = urllib.request.urlopen("http://192.168.1.163:8000/health", timeout=8)
    d = json.loads(r.read())
    print(f"  Health: {d.get('status')} routes={d.get('routes_loaded')} v={d.get('version')}")
    print("\nDeploy complete! cookoff.py updated on Pi.")
except Exception as e:
    print(f"  Health check: {e}")
    print("  NIS Protocol may still be starting up.")
