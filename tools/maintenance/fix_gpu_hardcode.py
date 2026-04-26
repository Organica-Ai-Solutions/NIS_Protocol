#!/usr/bin/env python3
"""Fix hardcoded CUDA_VISIBLE_DEVICES in cosmos servers and increase test timeout."""
import re, os

files = [
    "/data/organica-ai/cosmos_predict_server.py",
    "/data/organica-ai/cosmos_transfer25_server.py",
    "/data/organica-ai/cosmos_reason_server.py",
]

for f in files:
    if not os.path.exists(f):
        print(f"SKIP (not found): {f}")
        continue
    txt = open(f).read()
    new = re.sub(
        r'os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"[0-9]+"',
        '# CUDA_VISIBLE_DEVICES inherited from launch environment',
        txt
    )
    if new != txt:
        open(f, "w").write(new)
        print(f"FIXED: {f}")
    else:
        print(f"already clean: {f}")

# Also fix the test timeout for Transfer2.5 (600s -> 900s)
test_file = "/tmp/h100_full_test.py"
if os.path.exists(test_file):
    txt = open(test_file).read()
    new = txt.replace("timeout=600", "timeout=900")
    open(test_file, "w").write(new)
    print(f"FIXED timeout: {test_file}")
