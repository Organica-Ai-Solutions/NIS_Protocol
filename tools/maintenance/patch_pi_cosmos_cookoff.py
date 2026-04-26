#!/usr/bin/env python3
"""
Patch cosmos_cookoff.py on Pi:
- trajectory(): route through NIS /cookoff/robot-plan instead of H100 directly
- goal_verify(): route through NIS /cosmos/reason instead of H100 directly
Both H100 direct IPs are unreachable from Pi; NIS proxies correctly.
"""
import subprocess

COOKOFF = "/opt/neurolinux/cosmos_cookoff.py"

with open(COOKOFF) as f:
    src = f.read()

patches = []

# ── Fix 1: trajectory() — use NIS /cookoff/robot-plan ────────────────────
OLD_TRAJ = '''\
    async def trajectory(self, task: str, image_b64: str = None) -> dict:
        """2D gripper trajectory prediction — calls H100 /trajectory (cookbook Action CoT).
        Returns list of {point_2d: [x,y], label: 'gripper trajectory'} in 0-1000 pixel space.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.post(f"{H100_REASON_URL}/trajectory", json={
                    "task": task,
                    "image_base64": image_b64,
                    "robot_type": "xarm",
                })
                if r.status_code == 200:
                    return r.json()
        except Exception as e:
            log.warning("Trajectory endpoint failed: %s", e)
        return {"error": "trajectory_unavailable", "trajectory": []}'''

NEW_TRAJ = '''\
    async def trajectory(self, task: str, image_b64: str = None) -> dict:
        """2D gripper trajectory prediction — routes through NIS /cookoff/robot-plan.
        Returns list of {point_2d: [x,y], label: 'gripper trajectory'} in 0-1000 pixel space.
        """
        # Try NIS /cookoff/robot-plan (NIS proxies to H100 Reason2 /trajectory)
        try:
            body = {"query": task}
            if image_b64:
                body["image_base64"] = image_b64
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(f"{NIS_URL}/cookoff/robot-plan", json=body)
                if r.status_code == 200:
                    d = r.json()
                    traj = (d.get("cosmos_reasoning", {}).get("trajectory")
                            or d.get("trajectory", []))
                    return {"trajectory": traj, "source": "nis_robot_plan",
                            "ok": True, **d}
        except Exception as e:
            log.warning("NIS robot-plan for trajectory failed: %s", e)
        # Fallback: use reason to get a text plan
        result = await self.reason(image_b64 or "", f"Plan trajectory for: {task}")
        return {"trajectory": [], "source": "reason_fallback",
                "ok": not result.get("error"), **result}'''

if OLD_TRAJ in src:
    src = src.replace(OLD_TRAJ, NEW_TRAJ)
    patches.append("trajectory() → NIS /cookoff/robot-plan")
else:
    print("⚠  trajectory() pattern not found — checking for partial match")
    if "H100_REASON_URL}/trajectory" in src:
        # Simpler targeted replace
        src = src.replace(
            'f"{H100_REASON_URL}/trajectory"',
            'f"{NIS_URL}/cookoff/robot-plan"'
        )
        src = src.replace(
            '"task": task,\n                    "image_base64": image_b64,\n                    "robot_type": "xarm",',
            '"query": task,\n                    "image_base64": image_b64,'
        )
        patches.append("trajectory() URL patched (partial)")

# ── Fix 2: goal_verify() — use NIS /cosmos/reason ────────────────────────
OLD_GOAL = '                r = await c.post(f"{H100_REASON_URL}/goal-verify", json={'
NEW_GOAL = '                r = await c.post(f"{NIS_URL}/cosmos/reason", json={'

if OLD_GOAL in src:
    src = src.replace(OLD_GOAL, NEW_GOAL)
    patches.append("goal_verify() → NIS /cosmos/reason")

# ── Fix 3: plausibility() if it exists ───────────────────────────────────
if 'f"{H100_REASON_URL}/plausibility"' in src:
    src = src.replace(
        'f"{H100_REASON_URL}/plausibility"',
        'f"{NIS_URL}/cosmos/reason"'
    )
    patches.append("plausibility() → NIS /cosmos/reason")

print(f"Applied {len(patches)} patches:")
for p in patches:
    print(f"  ✅ {p}")

with open(COOKOFF, "w") as f:
    f.write(src)

r = subprocess.run(["python3", "-m", "py_compile", COOKOFF],
                   capture_output=True, text=True)
if r.returncode == 0:
    print("✅ Syntax OK")
else:
    print("❌ Syntax error:", r.stderr[:300])
