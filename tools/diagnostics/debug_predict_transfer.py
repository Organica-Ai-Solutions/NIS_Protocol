#!/usr/bin/env python3
"""Debug Predict2.5 and Transfer2.5 schema issues."""
import urllib.request, json

def get(url, timeout=8):
    try:
        r = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def post(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.request.HTTPError as e:
        body = e.read().decode()
        return None, f"HTTP {e.code}: {body[:400]}"
    except Exception as e:
        return None, str(e)

print("=== Predict2.5 /docs schema ===")
d = get("http://localhost:8200/openapi.json")
if "paths" in d:
    for path, methods in d["paths"].items():
        for method, info in methods.items():
            body = info.get("requestBody", {})
            schema_ref = body.get("content", {}).get("application/json", {}).get("schema", {})
            print(f"  {method.upper()} {path}: {schema_ref}")
else:
    print(" ", d.get("error", str(d))[:200])

print()
print("=== Predict2.5 /status ===")
d = get("http://localhost:8200/status")
print(" ", str(d)[:300])

print()
print("=== Predict2.5 test with minimal payload ===")
for payload in [
    {"prompt": "robot arm", "num_frames": 16, "fps": 8},
    {"prompt": "robot arm"},
    {"text": "robot arm"},
    {"query": "robot arm"},
]:
    d, err = post("http://localhost:8200/text2video", payload, timeout=10)
    if err:
        print(f"  text2video {list(payload.keys())}: {err[:100]}")
    else:
        print(f"  text2video OK: {str(d)[:100]}")

print()
print("=== Transfer2.5 /status ===")
d = get("http://localhost:8300/status")
print(" ", str(d)[:300])

print()
print("=== Transfer2.5 openapi ===")
d = get("http://localhost:8300/openapi.json")
if "paths" in d:
    for path, methods in d["paths"].items():
        for method, info in methods.items():
            print(f"  {method.upper()} {path}")
else:
    print(" ", str(d)[:200])

print()
print("=== Transfer2.5 /demos ===")
d = get("http://localhost:8300/demos")
print(" ", str(d)[:300])

print()
print("=== Transfer2.5 transfer with car_edge ===")
d, err = post("http://localhost:8300/transfer", {"demo": "car_edge", "control_type": "edge", "guidance": 3.0}, timeout=20)
if err:
    print(f"  error: {err[:300]}")
else:
    print(f"  OK: {str(d)[:200]}")
