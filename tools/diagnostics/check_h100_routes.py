#!/usr/bin/env python3
"""Check H100 Reason2 available routes."""
import urllib.request, json

for port, name in [(8100, "Reason2"), (8200, "Predict2.5"), (8300, "Transfer2.5")]:
    try:
        r = urllib.request.urlopen(f"http://localhost:{port}/openapi.json", timeout=5)
        spec = json.loads(r.read())
        paths = sorted(spec.get("paths", {}).keys())
        print(f"\n{name} (:{port}) — {len(paths)} routes:")
        for p in paths:
            print(f"  {p}")
    except Exception as e:
        try:
            r2 = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            print(f"\n{name} (:{port}) — health OK, no openapi: {e}")
        except Exception as e2:
            print(f"\n{name} (:{port}) — unreachable: {e2}")
