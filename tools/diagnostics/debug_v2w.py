#!/usr/bin/env python3
"""Debug Predict2.5 /video2world 500 error."""
import urllib.request, json

def post_verbose(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read()), None
    except urllib.request.HTTPError as e:
        body = e.read().decode(errors="replace")
        return None, f"HTTP {e.code}: {body[:600]}"
    except Exception as e:
        return None, str(e)[:200]

print("=== Predict2.5 /video2world error detail ===")
d, err = post_verbose("http://localhost:8200/video2world", {
    "prompt": "A robot arm picks up a red cube",
    "num_frames": 25,
    "fps": 10,
    "height": 480,
    "width": 848,
    "num_inference_steps": 20,
    "guidance_scale": 7.0,
    "seed": 42,
}, timeout=30)
if err:
    print(err)
else:
    print(d)
