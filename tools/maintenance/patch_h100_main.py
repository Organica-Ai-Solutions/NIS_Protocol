#!/usr/bin/env python3
"""Patch H100 NIS main.py to include cookoff_router, then restart NIS."""
import os, signal, time, urllib.request, json

MAIN = "/data/organica-ai/NIS_Protocol/main.py"

with open(MAIN) as f:
    src = f.read()

# Already patched?
if "cookoff_router" in src:
    print("main.py already has cookoff_router — no patch needed")
else:
    # Insert cookoff import+registration alongside cosmos
    old = (
        "    from routes.cosmos import router as cosmos_router\n"
        "    from routes.humanoid import router as humanoid_router\n"
        "    from routes.isaac_lab import router as isaac_lab_router\n"
        "    app.include_router(cosmos_router)\n"
        "    app.include_router(humanoid_router)\n"
        "    app.include_router(isaac_lab_router)\n"
        "    logger.info(\"✅ NVIDIA Stack integrated (Cosmos, GR00T, Isaac Lab)\")"
    )
    new = (
        "    from routes.cosmos import router as cosmos_router\n"
        "    from routes.cookoff import router as cookoff_router\n"
        "    from routes.humanoid import router as humanoid_router\n"
        "    from routes.isaac_lab import router as isaac_lab_router\n"
        "    app.include_router(cosmos_router)\n"
        "    app.include_router(cookoff_router)\n"
        "    app.include_router(humanoid_router)\n"
        "    app.include_router(isaac_lab_router)\n"
        "    logger.info(\"✅ NVIDIA Stack integrated (Cosmos, Cookoff, GR00T, Isaac Lab)\")"
    )
    if old not in src:
        print("ERROR: could not find expected block in main.py — manual patch needed")
        print("Looking for:\n", old)
    else:
        patched = src.replace(old, new)
        with open(MAIN, "w") as f:
            f.write(patched)
        print("✅ main.py patched — cookoff_router added")

# Restart NIS uvicorn (find PID)
import subprocess
result = subprocess.run(
    ["pgrep", "-f", "uvicorn main:app"],
    capture_output=True, text=True
)
pids = result.stdout.strip().split()
if pids:
    pid = int(pids[0])
    print(f"Restarting NIS uvicorn PID {pid}...")
    # Kill and let systemd/tmux restart it, or send SIGTERM and relaunch
    os.kill(pid, signal.SIGTERM)
    time.sleep(3)
    # Relaunch in background
    subprocess.Popen(
        ["/data/organica-ai/NIS_Protocol/venv/bin/uvicorn",
         "main:app", "--host", "0.0.0.0", "--port", "8000",
         "--workers", "1", "--timeout-keep-alive", "120"],
        cwd="/data/organica-ai/NIS_Protocol",
        stdout=open("/tmp/nis_restart.log", "w"),
        stderr=subprocess.STDOUT,
    )
    print("NIS relaunch initiated — waiting 8s...")
    time.sleep(8)
else:
    print("No uvicorn PID found — NIS may not be running")

# Verify
try:
    r = urllib.request.urlopen("http://localhost:8000/health", timeout=10)
    d = json.loads(r.read())
    print(f"NIS alive: v{d.get('version')} · {d.get('modular_routes')} routes")
except Exception as e:
    print(f"Health check: {e}")

try:
    r = urllib.request.urlopen("http://localhost:8000/cookoff/status", timeout=10)
    d = json.loads(r.read())
    print(f"✅ /cookoff/status: {d.get('status')} h100={list(d.get('h100_services',{}).keys())}")
except Exception as e:
    print(f"❌ /cookoff/status: {e}")

try:
    r = urllib.request.urlopen("http://localhost:8000/cosmos/reason", timeout=5)
except Exception as e:
    # 405 Method Not Allowed = route exists
    if "405" in str(e) or "400" in str(e):
        print("✅ /cosmos/reason route exists")
    else:
        print(f"❌ /cosmos/reason: {e}")
