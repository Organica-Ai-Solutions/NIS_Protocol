#!/usr/bin/env python3
"""Check camera pipeline routes and snapshot availability."""
import urllib.request, json, base64

BASE = "http://127.0.0.1:8009"

# Health
try:
    r = urllib.request.urlopen(BASE + "/health", timeout=3)
    print("health:", json.loads(r.read()))
except Exception as e:
    print("health error:", e)

# Try all known snapshot paths
for path in ["/snapshot", "/snapshot/base64", "/frame", "/capture", "/jpeg", "/image"]:
    try:
        r = urllib.request.urlopen(BASE + path, timeout=3)
        data = r.read()
        ct = r.headers.get("Content-Type", "?")
        print(f"  {path}: {r.status} {ct} {len(data)} bytes")
    except urllib.error.HTTPError as e:
        print(f"  {path}: HTTP {e.code}")
    except Exception as e:
        print(f"  {path}: {e}")

# Try openapi
try:
    r = urllib.request.urlopen(BASE + "/openapi.json", timeout=3)
    spec = json.loads(r.read())
    paths = sorted(spec.get("paths", {}).keys())
    print("\nAll routes:", paths)
except Exception as e:
    print("openapi:", e)
