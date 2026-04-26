#!/usr/bin/env python3
"""Send HUP to NIS uvicorn to reload routes, then verify."""
import os, time, urllib.request, json

pid = 1833642
print(f"Sending SIGHUP to NIS uvicorn PID {pid}...")
os.kill(pid, 1)  # SIGHUP = graceful reload
time.sleep(4)

try:
    r = urllib.request.urlopen("http://localhost:8000/health", timeout=8)
    d = json.loads(r.read())
    print(f"NIS alive: v{d.get('version')} · {d.get('modular_routes')} routes")
except Exception as e:
    print(f"Health check failed: {e}")

# Check cookoff route exists
try:
    r = urllib.request.urlopen("http://localhost:8000/cookoff/status", timeout=8)
    d = json.loads(r.read())
    print(f"cookoff/status: {d.get('status')} h100={list(d.get('h100_services',{}).keys())}")
except Exception as e:
    print(f"cookoff/status: {e}")
