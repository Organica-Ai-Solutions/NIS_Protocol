#!/usr/bin/env python3
"""
Patch neurolinux_agent.py: remove hard 503 on camera unavailable in cookoff endpoints.
Let reasoning proceed with empty image — H100 Cosmos Reason2 handles no-image queries fine.
"""

AGENT = "/opt/neurolinux/neurolinux_agent.py"

with open(AGENT) as f:
    src = f.read()

patches = [
    # cookoff/cosmos/reason — line ~1997
    (
        '    b64 = _capture_b64_with_fallback(quality=75)\n'
        '    if not b64:\n'
        '        return JSONResponse({"error": "Camera unavailable"}, status_code=503)',
        '    b64 = _capture_b64_with_fallback(quality=75) or ""',
    ),
    # cookoff/cosmos/trajectory — same pattern
    (
        '    b64 = capture_b64(quality=75)\n'
        '    result = await cosmos_cookoff.cosmos.trajectory(task, b64)',
        '    b64 = _capture_b64_with_fallback(quality=75) or ""\n'
        '    result = await cosmos_cookoff.cosmos.trajectory(task, b64)',
    ),
    # cookoff/cosmos/goal-verify
    (
        '    b64 = capture_b64(quality=75)\n'
        '    result = await cosmos_cookoff.cosmos.goal_verify(goal, b64, last_action)',
        '    b64 = _capture_b64_with_fallback(quality=75) or ""\n'
        '    result = await cosmos_cookoff.cosmos.goal_verify(goal, b64, last_action)',
    ),
    # cookoff/cosmos/plausibility
    (
        '    b64 = capture_b64(quality=75)\n'
        '    # Call H100 /plausibility directly',
        '    b64 = _capture_b64_with_fallback(quality=75) or ""\n'
        '    # Call H100 /plausibility directly',
    ),
]

applied = 0
for old, new in patches:
    if old in src:
        src = src.replace(old, new)
        applied += 1
    else:
        # Try without the _capture_b64_with_fallback prefix (already patched version)
        print(f"  ⚠ pattern not found (may already be patched): {old[:60]!r}")

print(f"✅ Applied {applied}/{len(patches)} patches")

with open(AGENT, "w") as f:
    f.write(src)

# Syntax check
import subprocess
r = subprocess.run(["python3", "-m", "py_compile", AGENT], capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
else:
    print("❌ Syntax error:", r.stderr[:300])
