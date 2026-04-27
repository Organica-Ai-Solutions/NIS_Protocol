#!/usr/bin/env python3
"""
Full Pi agent fix in one script:
1. Remove all 'if not b64: 503' blocks in cookoff section
2. Restart agent
3. Run smoke test
"""
import subprocess, time, urllib.request, json

AGENT = "/opt/neurolinux/neurolinux_agent.py"

# ── Step 1: Patch ──────────────────────────────────────────────────────────
with open(AGENT) as f:
    lines = f.readlines()

# Find cookoff section start
cookoff_start = None
for i, l in enumerate(lines):
    if '@app.get("/cookoff/status")' in l or "# ── Cosmos Cookoff" in l:
        cookoff_start = i
        break

if cookoff_start is None:
    print("ERROR: cookoff section not found"); exit(1)

print(f"Cookoff section at line {cookoff_start+1}")

removed = 0
i = cookoff_start
while i < len(lines) - 1:
    line = lines[i]
    next_line = lines[i + 1]
    if (line.strip() == "if not b64:" and
            "503" in next_line and "amera" in next_line):
        indent = len(line) - len(line.lstrip())
        lines[i]     = " " * indent + "# camera optional — proceed with empty frame\n"
        lines[i + 1] = ""
        removed += 1
        i += 2
        continue
    i += 1

print(f"Removed {removed} 'if not b64: 503' blocks")

with open(AGENT, "w") as f:
    f.writelines(lines)

# Syntax check
r = subprocess.run(["python3", "-m", "py_compile", AGENT],
                   capture_output=True, text=True)
if r.returncode != 0:
    print("❌ Syntax error:", r.stderr[:300]); exit(1)
print("✅ Syntax OK")

# ── Step 2: Restart agent ─────────────────────────────────────────────────
print("Restarting neurolinux-agent...")
subprocess.run(["sudo", "systemctl", "restart", "neurolinux-agent"])
time.sleep(5)

r = subprocess.run(["systemctl", "is-active", "neurolinux-agent"],
                   capture_output=True, text=True)
print(f"Agent status: {r.stdout.strip()}")

# ── Step 3: Smoke test ────────────────────────────────────────────────────
BASE = "http://localhost:8085"

def post(path, body=None, timeout=60):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:100]}"
    except Exception as e:
        return None, str(e)[:100]

print("\n── Smoke Test ──")

# Health
try:
    r = urllib.request.urlopen(BASE + "/health", timeout=8)
    d = json.loads(r.read())
    print(f"[health]  v{d.get('version')} ✅")
except Exception as e:
    print(f"[health]  FAIL: {e}")

# Cookoff status
d, err = post("/cookoff/status", timeout=10)
if d:
    print(f"[status]  mode={d.get('mode')} ✅")
else:
    print(f"[status]  FAIL: {err}")

# cosmos/reason — no camera needed now
print("[reason]  calling /cookoff/cosmos/reason ...")
t0 = time.time()
d, err = post("/cookoff/cosmos/reason",
              {"query": "What should the robot arm do next?"})
elapsed = time.time() - t0
if d:
    src   = d.get("source", "?")
    ok    = d.get("ok", "?")
    scene = str(d.get("scene", d.get("scene_description", "")))[:80]
    print(f"  ✅ ok={ok} source={src} {elapsed:.1f}s")
    if scene: print(f"  scene: {scene}")
else:
    print(f"  ❌ FAIL: {err} ({elapsed:.1f}s)")

print("\nDone.")
