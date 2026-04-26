#!/usr/bin/env python3
"""Quick health check of all Cosmos stack ports via SSH tunnel."""
import urllib.request, json

for port, name in [(8000,"NIS"),(8100,"Reason2"),(8200,"Predict2.5"),(8300,"Transfer2.5"),(8400,"Demo")]:
    try:
        r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
        d = json.loads(r.read())
        print(f"  :{port} {name}: {d.get('status','?')} - {str(d)[:120]}")
    except Exception as e:
        print(f"  :{port} {name}: FAIL - {str(e)[:80]}")
