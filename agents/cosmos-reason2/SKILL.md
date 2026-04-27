# SKILL: Cosmos Reason2 Visual-Spatial Reasoning Agent
# =====================================================
# NIS Protocol Skill Injection — H100 VLM Edition
# Model: Cosmos Reason2 on NVIDIA H100 GPU 0

## AGENT IDENTITY
Name: cosmos-reason2
Role: Visual-language reasoning for robotic workspace analysis
Hardware: NVIDIA H100, GPU 0
Endpoint: http://172.16.1.83:8100 (primary) | http://localhost:8100 (relay)
Status: Always-on inference server — DO NOT KILL GPU 0 process

## WHAT COSMOS REASON2 DOES
Cosmos Reason2 is a vision-language model (VLM) that analyzes camera frames
from the Pi Camera Module 3 and returns structured spatial analysis to guide
the xArm pick-and-place pipeline.

Primary functions:
1. **Pre-pick inspection** — detect object, estimate position in arm coordinates
2. **Post-lift grip verify** — confirm lighter is in gripper after GRIP step
3. **Post-place verify** — confirm object landed in drop zone after RELEASE
4. **Goal verification** — after full demo, confirm task was completed

## COORDINATE SYSTEM
```
Origin: base of xArm servo platform
Y+: forward from arm's front (toward camera subject)
X+: right (arm's first-person view)
Z+: upward
Units: centimeters (cm)
Camera: Pi Camera Module 3, elevated 35-45° side view
```

## API INTERFACE
```bash
# Health check
GET http://172.16.1.83:8100/health
→ {"status": "healthy", "model": "cosmos-reason2", "gpu": 0}

# Reasoning (with or without image)
POST http://172.16.1.83:8100/reason
{
  "query": "...",
  "image_base64": "<JPEG base64 or omit>",
  "max_tokens": 150,
  "use_think": false
}
→ {"response": "...", "confidence": 0.92, "model": "reason2"}
```

## PROMPT TEMPLATES

### Pre-pick inspection (arm at HOVER z≈6cm):
```
Arm coordinate system: origin at base, Y+=forward, X+=right, Z+=up.
Gripper is at HOVER position (z≈6cm) looking down at workspace.
Workspace is 17cm × 20.5cm wooden table surface.
Standard pick target: center-front at (x=0, y=17cm).

Q1: Is the lighter/object visible in the camera frame?
Q2: Estimate object position — x_cm from center, y_cm from base.
Q3: Is workspace clear of obstacles?
Reply JSON only: { "object_visible": bool, "object_x_cm": number, "object_y_cm": number, "safe": bool, "confidence": 0-1 }
```

### Post-lift grip verify (arm at LIFT, S1=700):
```
Arm is at LIFT position (S1=700 firm grip, home height).
Q: Is the yellow lighter visible between/near the gripper fingers?
Q: Is the grip secure — lighter not dangling or off-center?
Reply JSON only: { "gripped": bool, "confidence": 0-1, "grip_secure": bool, "notes": "..." }
```

### Post-place verify (arm at RELEASE):
```
Arm just released the lighter at the drop zone (left90 position).
Q: Is the lighter now in the drop bin/zone?
Q: Did the lighter land correctly or did it fall outside?
Reply JSON only: { "object_placed": bool, "in_bin": bool, "confidence": 0-1 }
```

### Goal verification (after full pipeline):
```
Task was: {task_description}
Look at the current scene. Is the task complete?
Reply: YES or NO with one sentence explanation.
```

## X-AXIS CORRECTION FORMULA
If Cosmos reports object offset from arm centerline (x≠0):
```python
if abs(object_x_cm) > 1.5:
    # S6 correction: 375/90 pulses per degree, center=500
    S6_correction = round(object_x_cm * (375/90))
    new_S6 = 500 - S6_correction
    # X+ = right in arm coords = S6 decreases
    # Apply to HOVER, MID, PICK, GRIP, LIFT steps
```

## CONFIDENCE THRESHOLDS
```
confidence >= 0.85 : proceed with pick
confidence 0.60-0.84 : proceed with caution (log warning)
confidence < 0.60 : abort, return to HOME, alert user
object_x_cm > ±3cm : apply S6 correction before picking
```

## FALLBACK BEHAVIOR
- If H100 unreachable: return {"object_visible": true, "object_x_cm": 0, "confidence": 0.0, "safe": true}
- confidence=0.0 signals IK-only mode (no visual correction)
- Never block arm movement for >5s waiting for Cosmos
- Always have timeout=15s on /reason calls

## INTEGRATION POINTS
- `routes/cookoff.py` — calls /reason at demo step 2 (planning) and step 5 (goal verify)
- `_run_ik_pick()` — optionally calls /reason at hover step for X correction
- `nis_console.py` — `camera cosmos` command calls /reason with snapshot
- `routes/cosmos.py` — proxies /reason endpoint to H100

## TIMING
```
/reason with image:    8-15 seconds (GPU inference)
/reason text-only:     2-5 seconds
/health:               <1 second
Camera snapshot:       1-2 seconds (Pi Camera warmup)
Full pre-pick cycle:   ~15 seconds (snapshot + reason + correction)
```

## KNOWN ISSUES
- Camera needs 3 warmup snapshots before first useful frame
- Pi Camera side-view angle causes depth ambiguity at y<10cm
- Cosmos may hallucinate object at edge of frame — trust confidence < 0.7 less
- H100 GPU 0 must stay running: DO NOT KILL cosmos_reason_server.py
