#!/usr/bin/env python3
"""Check which routes NIS has registered and whether cookoff is included."""
import urllib.request, json

r = urllib.request.urlopen("http://localhost:8000/health", timeout=8)
d = json.loads(r.read())
print("NIS version:", d.get("version"))
print("Modular routes:", d.get("modular_routes"))

# Try openapi to see all routes
try:
    r2 = urllib.request.urlopen("http://localhost:8000/openapi.json", timeout=8)
    spec = json.loads(r2.read())
    paths = sorted(spec.get("paths", {}).keys())
    cookoff = [p for p in paths if "cookoff" in p]
    cosmos  = [p for p in paths if "cosmos" in p]
    print(f"\nTotal paths: {len(paths)}")
    print(f"\nCookoff routes ({len(cookoff)}):")
    for p in cookoff:
        print(f"  {p}")
    print(f"\nCosmos routes ({len(cosmos)}):")
    for p in cosmos:
        print(f"  {p}")
except Exception as e:
    print(f"openapi.json error: {e}")

# Check if routes/__init__.py imports cookoff
import os
init_path = "/data/organica-ai/NIS_Protocol/routes/__init__.py"
if os.path.exists(init_path):
    with open(init_path) as f:
        content = f.read()
    has_cookoff = "cookoff" in content
    print(f"\nroutes/__init__.py imports cookoff: {has_cookoff}")
    if not has_cookoff:
        print("  --> cookoff router NOT registered in __init__.py!")
        # Show relevant import lines
        for line in content.split("\n"):
            if "router" in line.lower() or "import" in line.lower():
                print(f"  {line}")
