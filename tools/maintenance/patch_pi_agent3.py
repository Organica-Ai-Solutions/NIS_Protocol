#!/usr/bin/env python3
"""
Patch neurolinux_agent.py: remove all remaining 'if not b64: return 503' 
in cookoff endpoints so reasoning proceeds without a camera frame.
"""

AGENT = "/opt/neurolinux/neurolinux_agent.py"

with open(AGENT) as f:
    lines = f.readlines()

# Find and remove the 'if not b64: return JSONResponse({"error": "Camera unavailable"}, status_code=503)'
# pattern inside cookoff section only (after line 1817)
cookoff_start = None
for i, l in enumerate(lines):
    if "# ── Cosmos Cookoff Endpoints" in l or '@app.get("/cookoff/status")' in l:
        cookoff_start = i
        break

if cookoff_start is None:
    print("ERROR: could not find cookoff section start")
    exit(1)

print(f"Cookoff section starts at line {cookoff_start+1}")

removed = []
i = cookoff_start
while i < len(lines):
    line = lines[i]
    # Pattern: 4-space indent, if not b64:, next line returns 503
    if line.strip() == "if not b64:" and i + 1 < len(lines):
        next_line = lines[i + 1]
        if "503" in next_line and ("Camera unavailable" in next_line or "camera" in next_line.lower()):
            removed.append(i + 1)  # 1-indexed
            # Replace both lines with a no-op comment
            indent = len(line) - len(line.lstrip())
            lines[i]     = " " * indent + "# camera optional — proceed with empty frame\n"
            lines[i + 1] = ""
            i += 2
            continue
    i += 1

print(f"Removed {len(removed)} 'if not b64: 503' blocks at lines: {removed}")

with open(AGENT, "w") as f:
    f.writelines(lines)

import subprocess
r = subprocess.run(["python3", "-m", "py_compile", AGENT], capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
else:
    print("❌ Syntax error:", r.stderr[:300])
