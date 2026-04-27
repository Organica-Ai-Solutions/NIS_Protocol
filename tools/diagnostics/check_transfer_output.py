#!/usr/bin/env python3
"""Check Transfer2.5 inference output and test Predict2.5 via tunnel."""
import urllib.request, json, subprocess, time

SSH = ["ssh", "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=10", "awesome-gpu-name"]

def ssh_one(cmd, timeout=30):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if out: print(out[:800])
    return r.returncode, out

def post(url, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), time.time() - start, None
    except urllib.request.HTTPError as e:
        return None, time.time() - start, f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:
        return None, time.time() - start, str(e)[:200]

print("=== Transfer2.5 output files ===")
ssh_one("ls -lh /tmp/transfer_final/ 2>/dev/null")

print()
print("=== Predict2.5 health ===")
try:
    r = urllib.request.urlopen("http://localhost:8200/health", timeout=5)
    print(" ", json.loads(r.read()).get("status"))
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("=== Predict2.5 video2world (seed from H100) ===")
r = subprocess.run(SSH + ["/data/organica-ai/NIS_Protocol/venv/bin/python /tmp/gen_seed.py"],
                   capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace")
seed = r.stdout.strip()
print(f"  Seed: {len(seed)} chars")

if seed:
    d, elapsed, err = post("http://localhost:8200/video2world", {
        "prompt": "A robot arm picks up a red cube from a table",
        "image_b64": seed,
        "num_frames": 25, "fps": 10,
        "height": 480, "width": 848,
        "num_inference_steps": 20,
        "guidance_scale": 7.0, "seed": 42,
    }, timeout=90)
    if err:
        print(f"  FAIL: {err}")
    else:
        vid = d.get("video_b64", "")
        print(f"  Video: {len(vid)*3//4//1024} KB  latency: {d.get('latency_ms')}ms  total: {elapsed:.1f}s")
        print(f"  RESULT: PASS")
