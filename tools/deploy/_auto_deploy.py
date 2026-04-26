#!/usr/bin/env python3
"""
Auto-deploy all Pi updates when NIS comes back online.

Runs in the background and waits. When NIS is alive:
  1. Deploy routes/cookoff.py        -> /opt/nis-protocol/routes/cookoff.py
  2. Deploy routes/cosmos_dance.py   -> /opt/nis-protocol/routes/cosmos_dance.py
  3. Deploy _dashboard_new.html      -> /opt/neurolinux/dashboard.html
  4. Patch neurolinux_agent.py       -> serve dashboard from file
  5. Restart neurolinux-agent service

Usage:
  python _auto_deploy.py          # runs until NIS is up, then deploys
  python _auto_deploy.py --now    # assume NIS is already up, deploy immediately
"""
import urllib.request, json, base64, pathlib, sys, time, argparse

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NIS   = 'http://192.168.1.163:8000'
AGENT = 'http://192.168.1.163:8085'

LOCAL = pathlib.Path(r'C:\Users\DiegoTorres\Desktop\NIS_Protocol')
PI_NIS_ROUTES = '/opt/nis-protocol/routes'
PI_AGENT_DIR  = '/opt/neurolinux'

FILES_TO_DEPLOY = [
    (LOCAL / 'routes' / 'cookoff.py',      f'{PI_NIS_ROUTES}/cookoff.py'),
    (LOCAL / 'routes' / 'cosmos_dance.py', f'{PI_NIS_ROUTES}/cosmos_dance.py'),
    (LOCAL / 'routes' / 'openclaw.py',     f'{PI_NIS_ROUTES}/openclaw.py'),
    (LOCAL / '_dashboard_new.html',         f'{PI_AGENT_DIR}/dashboard.html'),
]

DASHBOARD_PATCHER = '''
import re, os

AGENT = '/opt/neurolinux/neurolinux_agent.py'
DASH  = '/opt/neurolinux/dashboard.html'

if not os.path.exists(DASH):
    print("ERROR: dashboard.html not found at " + DASH)
    exit(1)

with open(AGENT, 'r', encoding='utf-8', errors='replace') as f:
    src = f.read()

# Check if already patched
if "open(DASH" in src or "dashboard.html" in src:
    print("Already patched — skipping agent patch")
    exit(0)

# Find the embedded HTML return in the cosmos/dashboard endpoint
pattern = re.compile(
    r'([ \\t]*return\\s+(?:f)?(?:"""|\\x27\\x27\\x27)<!DOCTYPE html>.*?(?:"""|\\x27\\x27\\x27))',
    re.DOTALL
)
m = pattern.search(src)
if m:
    old  = m.group(0)
    indent = old[:len(old) - len(old.lstrip())]
    new_code = (
        indent + "DASH = \\'/opt/neurolinux/dashboard.html\\'\\n" +
        indent + "if __import__(\\'os\\').path.exists(DASH):\\n" +
        indent + "    from fastapi.responses import HTMLResponse\\n" +
        indent + "    return HTMLResponse(open(DASH,\\'r\\',encoding=\\'utf-8\\').read())\\n" +
        old  # keep original as fallback if file missing
    )
    with open(AGENT + '.bak_dashboard', 'w', encoding='utf-8') as f:
        f.write(src)
    with open(AGENT, 'w', encoding='utf-8') as f:
        f.write(src.replace(old, new_code, 1))
    print("Dashboard patch applied!")
else:
    print("Could not find HTML return — checking route definition...")
    idx = src.find("/cosmos/dashboard")
    print("Route found at char", idx, "context:", repr(src[max(0,idx-50):idx+200]))
    exit(1)
'''


# ── Helpers ───────────────────────────────────────────────────────────────────

def nis_alive():
    try:
        urllib.request.urlopen(NIS + '/health', timeout=4)
        return True
    except Exception:
        return False


def agent_alive():
    try:
        urllib.request.urlopen(AGENT + '/health', timeout=4)
        return True
    except Exception:
        return False


def shell(cmd, timeout=25):
    d = json.dumps({'cmd': cmd}).encode()
    r = urllib.request.Request(NIS + '/system/shell', data=d,
                               headers={'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    return (resp.get('stdout') or resp.get('output') or resp.get('result') or '').strip()


def push_bytes(data: bytes, remote: str, timeout=35) -> bool:
    b64 = base64.b64encode(data).decode('ascii')
    cmd = (f"python3 -c \"import base64; "
           f"open('{remote}','wb').write(base64.b64decode('{b64}'))\"")
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(NIS + '/system/shell',
                               data=json.dumps({'cmd': cmd}).encode(),
                               headers={'Content-Type': 'application/json'}),
        timeout=timeout).read())
    return r.get('returncode', 1) == 0


def push_file(local: pathlib.Path, remote: str) -> bool:
    data = local.read_bytes()
    shell(f'cp {remote} {remote}.bak 2>/dev/null')
    ok = push_bytes(data, remote)
    size = shell(f'wc -c {remote}')
    print(f"  {'OK' if ok else 'FAIL'}: {local.name} -> {remote} ({size})")
    return ok


# ── Main deploy ───────────────────────────────────────────────────────────────

def deploy():
    print("\n=== Deploying NIS Protocol + Dashboard updates ===\n")

    # Step 1: Push route files
    print("[1] Pushing NIS Protocol route files...")
    all_ok = True
    for local, remote in FILES_TO_DEPLOY:
        if not local.exists():
            print(f"  SKIP: {local.name} not found locally")
            continue
        ok = push_file(local, remote)
        all_ok = all_ok and ok

    if not all_ok:
        print("\n  Some files failed — check NIS shell endpoint")
        return False

    # Step 2: Patch neurolinux_agent.py
    print("\n[2] Patching neurolinux_agent.py for dashboard file serving...")
    ok = push_bytes(DASHBOARD_PATCHER.encode('utf-8'), '/tmp/_patch_agent.py')
    print(f"  Patcher pushed: {'OK' if ok else 'FAIL'}")
    if ok:
        result = shell('python3 /tmp/_patch_agent.py', timeout=20)
        print(f"  Patcher result: {result}")

    # Step 3: Restart NIS service (files updated, needs reload)
    print("\n[3] Restarting nis-protocol service...")
    r1 = shell('sudo systemctl restart nis-protocol 2>&1 || echo "sudo failed"', timeout=20)
    print(f"  NIS restart: {r1 or '(sent)'}")

    # Step 4: Restart neurolinux-agent (dashboard patch)
    print("\n[4] Restarting neurolinux-agent...")
    r2 = shell('sudo systemctl restart neurolinux-agent 2>&1 || '
               'kill -HUP $(lsof -i :8085 -t 2>/dev/null | head -1) 2>&1', timeout=15)
    print(f"  Agent restart: {r2 or '(sent)'}")

    # Step 5: Verify
    print("\n[5] Verifying services...")
    time.sleep(8)

    nis_up = nis_alive()
    agent_up = agent_alive()
    print(f"  NIS   (8000): {'UP' if nis_up else 'DOWN'}")
    print(f"  Agent (8085): {'UP' if agent_up else 'DOWN (may still be restarting)'}")

    if nis_up:
        # Verify new endpoints
        try:
            r = urllib.request.urlopen(NIS + '/openapi.json', timeout=8)
            spec = json.loads(r.read())
            new_eps = [p for p in spec['paths'] if 'dance' in p or '/pick' in p]
            print(f"  New endpoints: {new_eps}")
        except Exception as e:
            print(f"  OpenAPI check: {e}")

    print("\n=== Deploy complete! ===")
    print(f"  Dashboard:  http://192.168.1.163:8085/cosmos/dashboard")
    print(f"  NIS API:    http://192.168.1.163:8000/docs")
    print(f"  Pick:       POST http://192.168.1.163:8000/cookoff/pick")
    print(f"  Dance:      POST http://192.168.1.163:8000/cookoff/dance")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Auto-deploy to Pi when NIS is available')
    parser.add_argument('--now', action='store_true', help='Deploy immediately without waiting')
    parser.add_argument('--wait', type=int, default=3600, help='Max seconds to wait for NIS (default 3600=1hr)')
    args = parser.parse_args()

    if args.now:
        if not nis_alive():
            print("NIS is not up. Use --now only when NIS is already running.")
            print("Run on Pi first: sudo systemctl restart nis-protocol")
            sys.exit(1)
        deploy()
        return

    print(f"Waiting for NIS to come up (max {args.wait}s)...")
    print("On the Pi terminal, run: sudo systemctl restart nis-protocol\n")

    start = time.time()
    dots = 0
    while time.time() - start < args.wait:
        if nis_alive():
            print(f"\nNIS is UP after {time.time()-start:.0f}s!")
            time.sleep(2)  # let it finish starting
            deploy()
            return
        time.sleep(5)
        dots += 1
        print(f"  Waiting... {time.time()-start:.0f}s", end='\r', flush=True)

    print(f"\nNIS did not come up in {args.wait}s.")
    print("Once NIS is running, run: python _auto_deploy.py --now")
    print()
    print("To restart NIS on the Pi:")
    print("  ssh neurolinux@192.168.1.163")
    print("  sudo systemctl restart nis-protocol")


if __name__ == '__main__':
    main()
