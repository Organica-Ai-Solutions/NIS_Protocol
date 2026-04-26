#!/usr/bin/env python3
"""
Deploy updated xArm controller dashboard to Pi.

Strategy:
  1. Push _dashboard_new.html to /opt/neurolinux/dashboard.html
  2. Patch neurolinux_agent.py: replace the embedded HTML return
     with open('/opt/neurolinux/dashboard.html').read()
  3. Restart neurolinux-agent

Run AFTER NIS is back online:
  (Pi terminal)  sudo systemctl restart nis-protocol
  (Windows)      python _deploy_dashboard.py
"""
import urllib.request, json, base64, pathlib, sys, time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NIS   = 'http://192.168.1.163:8000'
AGENT = 'http://192.168.1.163:8085'
PI_AGENT   = '/opt/neurolinux/neurolinux_agent.py'
PI_DASH    = '/opt/neurolinux/dashboard.html'
LOCAL_HTML = pathlib.Path(r'C:\Users\DiegoTorres\Desktop\NIS_Protocol\_dashboard_new.html')

# ── Helpers ───────────────────────────────────────────────────────────────────
def shell(cmd, timeout=20):
    d = json.dumps({'cmd': cmd}).encode()
    r = urllib.request.Request(NIS+'/system/shell', data=d,
                               headers={'Content-Type':'application/json'})
    resp = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    return (resp.get('stdout') or resp.get('output') or resp.get('result') or '').strip()

def push_bytes(data: bytes, remote: str, timeout=30) -> bool:
    b64 = base64.b64encode(data).decode('ascii')
    cmd = (f"python3 -c \"import base64; "
           f"open('{remote}','wb').write(base64.b64decode('{b64}'))\"")
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(NIS+'/system/shell',
            data=json.dumps({'cmd': cmd}).encode(),
            headers={'Content-Type':'application/json'}),
        timeout=timeout).read())
    return r.get('returncode', 1) == 0

def nis_alive():
    try:
        urllib.request.urlopen(NIS+'/health', timeout=4)
        return True
    except Exception:
        return False

def agent_alive():
    try:
        urllib.request.urlopen(AGENT+'/health', timeout=4)
        return True
    except Exception:
        return False

# ── Check NIS ─────────────────────────────────────────────────────────────────
print("=== xArm Dashboard Deployer ===\n")
if not nis_alive():
    print("NIS server (port 8000) is DOWN.")
    print("Please restart it on the Pi, then re-run this script:")
    print("  sudo systemctl restart nis-protocol")
    sys.exit(1)
print("NIS is UP")

# ── Step 1: Push HTML file ────────────────────────────────────────────────────
print(f"\n[1] Pushing dashboard HTML -> {PI_DASH}")
html_bytes = LOCAL_HTML.read_bytes()
shell(f'cp {PI_DASH} {PI_DASH}.bak 2>/dev/null')
ok = push_bytes(html_bytes, PI_DASH)
print(f"    {'OK' if ok else 'FAILED'} ({len(html_bytes)} bytes)")
if not ok:
    sys.exit(1)

# Verify it landed correctly
size = shell(f'wc -c {PI_DASH}')
print(f"    Verified: {size}")

# ── Step 2: Patch neurolinux_agent.py ────────────────────────────────────────
print(f"\n[2] Patching {PI_AGENT} to serve from file...")

# Find what the current dashboard return looks like
context = shell(f"grep -n 'cosmos/dashboard\\|<!DOCTYPE' {PI_AGENT} | head -10")
print(f"    Current refs:\n    {context}")

PATCHER = r"""
import re, sys

AGENT   = '/opt/neurolinux/neurolinux_agent.py'
DASH    = '/opt/neurolinux/dashboard.html'
NEW_RET = "    return open(DASH, 'r', encoding='utf-8').read()"

with open(AGENT, 'r', encoding='utf-8', errors='replace') as f:
    src = f.read()

# Find the dashboard endpoint function and its return statement
# Look for the pattern:  return """<!DOCTYPE html>..."""  or  return f"""..."""
# Use a pattern that captures everything from return to closing triple-quote
pattern = re.compile(
    r'([ \t]*return\s+(?:f)?(?:"""|\x27\x27\x27)<!DOCTYPE html>.*?(?:"""|\x27\x27\x27))',
    re.DOTALL
)
m = pattern.search(src)
if m:
    old = m.group(0)
    print(f'Found embedded HTML return ({len(old)} chars)')
    # Add DASH constant before the function if not present
    if "DASH = '/opt/neurolinux/dashboard.html'" not in src:
        # Insert DASH = ... near the top (after first import block)
        insert_after = 'import os\n'
        idx = src.find(insert_after)
        if idx >= 0:
            insert_pt = idx + len(insert_after)
            src = src[:insert_pt] + "DASH = '/opt/neurolinux/dashboard.html'\n" + src[insert_pt:]
    new_src = src.replace(old, NEW_RET, 1)
    # Backup
    with open(AGENT+'.bak_dash', 'w', encoding='utf-8') as f:
        f.write(src)
    with open(AGENT, 'w', encoding='utf-8') as f:
        f.write(new_src)
    print('Patch applied successfully!')
    sys.exit(0)
else:
    # Maybe dashboard returns HTML differently - try to find and show context
    idx = src.find('cosmos/dashboard')
    if idx < 0:
        idx = src.find('Cosmos Cookoff Dashboard')
    if idx >= 0:
        print('Context around dashboard:')
        print(repr(src[max(0,idx-100):idx+400]))
    else:
        print('Dashboard endpoint not found in agent file')
    sys.exit(1)
"""

patcher_bytes = PATCHER.encode('utf-8')
ok = push_bytes(patcher_bytes, '/tmp/_patch_dash.py')
print(f"    Patcher pushed: {'OK' if ok else 'FAIL'}")
if not ok:
    sys.exit(1)

result = shell('python3 /tmp/_patch_dash.py', timeout=20)
print(f"    Patcher output: {result}")

if 'Patch applied' not in result:
    print("\n    Patch may not have applied cleanly.")
    print("    Trying manual HTML injection approach...")

    # Fallback: directly write a small shim into the agent
    SHIM = r"""
import re
AGENT = '/opt/neurolinux/neurolinux_agent.py'
DASH  = '/opt/neurolinux/dashboard.html'

with open(AGENT, 'r', encoding='utf-8', errors='replace') as f:
    src = f.read()

# Try simpler approach: find @app.get("/cosmos/dashboard") endpoint
idx = src.find('"/cosmos/dashboard"')
if idx < 0:
    idx = src.find("'/cosmos/dashboard'")
if idx >= 0:
    # Find the def that follows
    def_idx = src.find('def ', idx)
    if def_idx >= 0:
        # Find the return statement in the function
        ret_idx = src.find('return ', def_idx)
        if ret_idx >= 0:
            # Find end of return value (next non-indented line or next function)
            # Insert a read-file return before the current return
            indent = '    '
            new_code = f'\n{indent}DASH = "{DASH}"\n{indent}if __import__("os").path.exists(DASH):\n{indent}    from fastapi.responses import HTMLResponse\n{indent}    return HTMLResponse(open(DASH,"r",encoding="utf-8").read())\n'
            src_new = src[:ret_idx] + new_code + src[ret_idx:]
            with open(AGENT+'.bak2', 'w', encoding='utf-8') as f:
                f.write(src)
            with open(AGENT, 'w', encoding='utf-8') as f:
                f.write(src_new)
            print('Shim injected at cosmos/dashboard endpoint')
        else:
            print('No return found after def')
    else:
        print('No def found after route')
else:
    print('Route /cosmos/dashboard not found')
"""
    shim_bytes = SHIM.encode('utf-8')
    push_bytes(shim_bytes, '/tmp/_shim_dash.py')
    result2 = shell('python3 /tmp/_shim_dash.py', timeout=20)
    print(f"    Shim output: {result2}")

# ── Step 3: Restart neurolinux-agent ─────────────────────────────────────────
print("\n[3] Restarting neurolinux-agent...")
restart = shell('sudo systemctl restart neurolinux-agent 2>&1 || '
                'kill -HUP $(lsof -i :8085 -t 2>/dev/null | head -1) 2>&1', timeout=15)
print(f"    restart: {restart or '(sent)'}")
time.sleep(5)

if agent_alive():
    print("    Agent is UP!")
    print("\nVerify dashboard: http://192.168.1.163:8085/cosmos/dashboard")
else:
    print("    Agent not responding (may be restarting - wait ~10s and check)")

print("\n=== Done ===")
