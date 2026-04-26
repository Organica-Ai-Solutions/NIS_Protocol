#!/usr/bin/env python3
"""
Deploy Autonomous Challenge to Pi
===================================
Pushes:
  1. static/_dashboard_new.html -> /opt/neurolinux/dashboard.html
  2. main_pi.py          -> /opt/neurolinux/neurolinux_agent.py
  3. Restarts neurolinux-agent
  4. Verifies /autonomous/run + /autonomous/stream endpoints

Usage:
  python _deploy_autonomous.py
  python _deploy_autonomous.py --verify-only
"""
import argparse, base64, json, pathlib, sys, time, urllib.request, urllib.error

NIS   = 'http://192.168.1.163:8000'
AGENT = 'http://192.168.1.163:8085'
LOCAL = pathlib.Path(r'C:\Users\DiegoTorres\Desktop\NIS_Protocol')

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _post_json(url, data, timeout=40):
    body = json.dumps(data).encode()
    req  = urllib.request.Request(url, data=body,
                                   headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def shell(cmd, timeout=30):
    try:
        r = _post_json(NIS + '/system/shell', {'cmd': cmd}, timeout=timeout)
        return (r.get('stdout') or r.get('output') or r.get('result') or '').strip(), r.get('returncode', -1)
    except Exception as e:
        return str(e), -1


CHUNK = 40_000   # bytes per chunk (safe for /system/shell JSON body)

CHUNK = 40_000   # bytes per chunk (safe for /system/shell JSON body)

def push_file(local: pathlib.Path, remote: str, timeout=60) -> bool:
    """Push a file to the Pi via /system/shell using chunked Python writes."""
    data   = local.read_bytes()
    chunks = [data[i:i+CHUNK] for i in range(0, len(data), CHUNK)]
    print(f"  Pushing {local.name} ({len(data)} bytes, {len(chunks)} chunks)...")

    for idx, chunk in enumerate(chunks):
        b64  = base64.b64encode(chunk).decode('ascii')
        mode = 'wb' if idx == 0 else 'ab'
        cmd  = (f"python3 -c \""
                f"import base64; "
                f"f=open('{remote}','{mode}'); "
                f"f.write(base64.b64decode('{b64}')); "
                f"f.close()\"")
        _, rc = shell(cmd, timeout=timeout)
        if rc != 0:
            print(f"    chunk {idx}/{len(chunks)} failed (rc={rc})")
            return False
        print(f"    chunk {idx+1}/{len(chunks)} ok", end='\r')
    print()

    # Verify size
    size_out, _ = shell(f'wc -c {remote} 2>&1')
    ok = str(len(data)) in size_out
    print(f"  {'OK' if ok else 'SIZE MISMATCH'} -> {remote}  (expected {len(data)}, remote: {size_out})")
    return ok


def get_agent(path, timeout=6):
    try:
        r = urllib.request.urlopen(AGENT + path, timeout=timeout)
        return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)


# ── Main deploy ───────────────────────────────────────────────────────────────

def deploy():
    print("\n=== Deploying Autonomous Challenge to Pi ===\n")

    # Backup
    print("[0] Backing up current agent...")
    out, rc = shell('cp /opt/neurolinux/neurolinux_agent.py /opt/neurolinux/neurolinux_agent.py.bak_auto 2>&1')
    print(f"  backup: {out or 'ok'}")

    # Push files
    print("\n[1] Pushing files...")
    files = [
        (LOCAL / 'static' / '_dashboard_new.html', '/opt/neurolinux/dashboard.html'),
        (LOCAL / 'main_pi.py',          '/opt/neurolinux/neurolinux_agent.py'),
    ]
    all_ok = True
    for local, remote in files:
        if not local.exists():
            print(f"  SKIP: {local.name} not found locally at {local}")
            all_ok = False
            continue
        ok = push_file(local, remote)
        all_ok = all_ok and ok

    if not all_ok:
        print("\n  One or more files failed — aborting restart")
        return False

    # Restart agent — use fuser to avoid pkill matching our own shell cmd
    print("\n[2] Restarting neurolinux-agent...")
    # Step a: kill whatever is on 8085
    shell('fuser -k 8085/tcp 2>/dev/null; sleep 1', timeout=10)
    # Step b: start fresh with correct env
    start_cmd = (
        "python3 -c \""
        "import subprocess,os; "
        "p=subprocess.Popen("
        "['/opt/neurolinux/venv/bin/python3','/opt/neurolinux/neurolinux_agent.py'],"
        "env={**os.environ,"
        "'NEUROLINUX_AGENT_PORT':'8085',"
        "'NEUROLINUX_AGENT_HOST':'0.0.0.0',"
        "'PYTHONPATH':'/opt/neurolinux',"
        "'XARM_PORT':'/dev/ttyAMA10'},"
        "stdout=open('/tmp/agent.log','w'),"
        "stderr=subprocess.STDOUT,"
        "start_new_session=True,"
        "close_fds=True); "
        "print('pid='+str(p.pid))\""
    )
    out, _ = shell(start_cmd, timeout=20)
    print(f"  {out or '(no output)'}")

    # Wait for agent to come back
    print("\n[3] Waiting for agent to restart (8s)...")
    for i in range(8):
        time.sleep(1)
        print(f"  {'.' * (i+1)}", end='\r')
    print()

    return verify()


def verify():
    print("\n[4] Verifying endpoints...")

    # Basic health
    d, err = get_agent('/health')
    if err:
        print(f"  FAIL: /health — {err}")
        return False
    print(f"  OK: /health — v{d.get('version','?')} xarm={d.get('xarm')} sim={d.get('xarm_simulation')}")

    # Check /autonomous/run is registered
    d, err = get_agent('/routes')
    if err:
        print(f"  WARN: /routes not available — {err}")
    else:
        paths = d.get('routes', [])
        has_run    = '/autonomous/run'    in paths
        has_stream = '/autonomous/stream' in paths
        print(f"  /autonomous/run:    {'✓' if has_run    else '✗ MISSING'}")
        print(f"  /autonomous/stream: {'✓' if has_stream else '✗ MISSING'}")

    # Check dashboard is served
    try:
        r = urllib.request.urlopen(AGENT + '/cosmos/dashboard', timeout=6)
        html = r.read().decode('utf-8', errors='replace')
        has_tab = 'tab-auto' in html or 'Autonomous' in html
        print(f"  /cosmos/dashboard:  {'✓ (Autonomous tab present)' if has_tab else '✗ tab not found — dashboard may not have reloaded yet'}")
    except Exception as e:
        print(f"  /cosmos/dashboard:  FAIL — {e}")

    print("\n=== Deploy complete! ===")
    print(f"  Dashboard:       http://192.168.1.163:8085/cosmos/dashboard")
    print(f"  SSE stream:      curl -N http://192.168.1.163:8085/autonomous/stream")
    print(f"  Sim test:        curl -X POST http://192.168.1.163:8085/autonomous/run")
    print(f"                        -H 'Content-Type: application/json'")
    print(f"                        -d '{{\"execute_arm\":false,\"max_picks\":2}}'")
    return True


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()

    if args.verify_only:
        verify()
    else:
        deploy()


if __name__ == '__main__':
    main()
