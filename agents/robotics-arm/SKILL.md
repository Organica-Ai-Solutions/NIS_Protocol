# SKILL: Hiwonder xArm Robotic Arm — Pick-and-Place Expert
# =========================================================
# NIS Protocol Skill Injection v3 — IK Edition
# Source: Official Hiwonder xArm AI Documentation (section 5.2.x)

## ARM MODEL
Hiwonder xArm AI / ArmPi FPV
- Servos: S1=gripper, S2=wrist-roll, S3=wrist-pitch, S4=elbow, S5=shoulder, S6=base-rotation
- Gripper: S1=100 OPEN, S1=700 FIRM GRIP (confirmed pick), S1=900 FULLY CLOSED
- Control: HTTP API on Raspberry Pi at 192.168.1.163:8085

## COORDINATE SYSTEM (OFFICIAL)
From Hiwonder documentation section 5.2.2:
```
Origin: bottom of servo at base platform
Y+: forward from the arm's front
X+: right (arm's first-person view)  
Z+: upward
Units: centimeters (cm)
```

## LINK LENGTHS (DOCUMENTED)
```
L1 = 6.9 cm   base to servo5 output shaft (shoulder)
L2 = 9.5 cm   servo5 to servo4 output shaft (upper arm)
L3 = 9.5 cm   servo4 to servo3 output shaft (forearm)
L4 = 16.9 cm  servo3 to gripper tip (wrist+gripper)
```

## IK FORMULA (from Hiwonder docs section 5.2.1)
```
Given target (x, y, z, pitch_deg):
  reach = sqrt(x² + y²)                     # 2D horizontal reach
  theta_base = atan2(x, y)                   # base rotation angle
  m = reach - L4*cos(pitch)                  # wrist forward component
  n = z - L1 - L4*sin(pitch)                # wrist height component
  theta2 = acos((m²+n² - L2²-L3²)/(2*L2*L3))   # elbow
  theta1 = atan2(n,m) - atan2(L3*sin(theta2), L2+L3*cos(theta2))  # shoulder
  theta3 = pitch - theta1 - theta2           # wrist pitch
```

## CORRECT HOME POSITION
```
ki_move(0, 17, 20.5, 0)
  x=0:     arm facing STRAIGHT FORWARD (S6=500)
  y=17cm:  standard forward reach
  z=20.5:  raised position
  pitch=0: gripper LEVEL (horizontal)
```
**WARNING: Old stored home had S6=350 (arm rotated LEFT). That is WRONG.**
**The correct S6 at home is 500 (facing forward).**

## CONFIRMED SERVO POSITIONS (IK verified 2026-02-27, tested 5x reliable)
```
Position       Servo values (S1=gripper, S2=wrist-roll, S3-S5=arm, S6=base)
HOME           S1=100 S2=500 S3=310 S4=870 S5=680 S6=500
HOVER          S1=100 S2=500 S3=222 S4=697 S5=604 S6=500  (z≈6cm)
MID            S1=100 S2=500 S3=158 S4=798 S5=502 S6=500  (z≈3.5cm)
PICK           S1=100 S2=500 S3=142 S4=856 S5=430 S6=500  (z=1.5cm ← CONFIRMED)
GRIP           S1=700 S2=500 S3=142 S4=856 S5=430 S6=500  (firm grip)
LIFT           S1=700 S2=500 S3=310 S4=870 S5=680 S6=500  (home height+grip)
PLACE_LEFT90   S1=700 S2=500 S3=220 S4=827 S5=425 S6=875  (drop zone)
RELEASE_LEFT90 S1=100 S2=500 S3=220 S4=827 S5=425 S6=875  (open grip)
```

NOTE: IK params that work: x=0, y=17cm, z=1.5cm, alpha=-65°
Previous alpha=-71 caused arm to collapse near singularity (S5≈536 unstable).
alpha=-65 confirmed stable: S5=430, S3=142, S4=856.

## LIGHTER POSITION
- Lighter placed at center-front: x=0cm, y=17cm (S6=500, facing straight forward)
- Pick height: z=1.5cm (NOT 1.2cm — 1.2 caused arm to press table), pitch=-65°
- IK command: `ki_move(0, 17, 1.5, -65)` → S3=142 S4=856 S5=430 S6=500

## CONFIRMED 10-STEP PICK SEQUENCE (reliable, tested 2026-02-27)
```python
# Confirmed servo positions — direct group_move, no IK needed
HOME  = {"1":100,"2":500,"3":310,"4":870,"5":680,"6":500}
HOVER = {"1":100,"2":500,"3":222,"4":697,"5":604,"6":500}  # z≈6cm
MID   = {"1":100,"2":500,"3":158,"4":798,"5":502,"6":500}  # z≈3.5cm
PICK  = {"1":100,"2":500,"3":142,"4":856,"5":430,"6":500}  # z=1.5cm
GRIP  = {"1":700,"2":500,"3":142,"4":856,"5":430,"6":500}  # S1=700!
LIFT  = {"1":700,"2":500,"3":310,"4":870,"5":680,"6":500}
PLACE = {"1":700,"2":500,"3":220,"4":827,"5":425,"6":875}  # left90
RELAX = {"1":100,"2":500,"3":220,"4":827,"5":425,"6":875}  # release

# Step 1: HOME + open gripper
group_move(HOME, 1000)
# Step 2: Hover over object (safe height z≈6cm)
group_move(HOVER, 900)
# Step 3: Mid descent (z≈3.5cm)
group_move(MID, 700)
# Step 4: Pick height (z=1.5cm)
group_move(PICK, 600)
# Step 5: GRIP — S1=700 (firm, confirmed)
group_move(GRIP, 500)
# Step 6: LIFT to home height (hold grip)
group_move(LIFT, 800)
# Step 7: SWING to drop zone (left 90°)
group_move(PLACE, 900)
# Step 8: RELEASE
group_move(RELAX, 600)
# Step 9: HOME
group_move(HOME, 1000)
```

## GRIPPER RULES (CONFIRMED)
- Open:       S1=100  (fully open — use for travel)
- Firm grip:  S1=700  (CONFIRMED for lighter — was 500, too loose)
- Full close: S1=900  (empty-hand transport, no object)
- Critical: S1=500 is NOT enough grip — lighter will fall during lift!

## COSMOS REASON2 PROMPTS

### Pre-pick inspection:
```
Arm coordinate system: origin at base, Y+=forward, X+=right, Z+=up.
Gripper is at (x=0, y=17cm, z=6cm) looking down at workspace (HOVER position).
Target: yellow/green LIGHTER at center-front of workspace.
Confirmed pick position: x=0, y=17cm, z=1.5cm, alpha=-65 degrees.
Q1: Is the lighter visible in the camera frame?
Q2: Is the lighter near center (x≈0, y≈17cm)?
Q3: Any obstacles? Is the workspace clear?
Reply: { "lighter_visible": bool, "lighter_x_cm": number, "lighter_y_cm": number, "safe": bool }
```

### Post-lift grip verify:
```
Arm is at LIFT position (S1=700, S3=310, S4=870, S5=680 — home height with grip).
Q: Is the yellow lighter visible in/near the gripper fingers?
Q: Is the grip secure (lighter not dangling)?
Reply: { "gripped": bool, "confidence": 0-1, "notes": "..." }
```

### Cosmos X correction formula:
```python
# If Cosmos reports lighter offset from center:
if abs(lighter_x_cm) > 1.5:
    # Adjust S6: S6_scale = 375/90 pulses/degree, S6_center = 500
    S6_correction = round(lighter_x_cm * (375/90))
    new_S6 = 500 - S6_correction  # minus because X+ = right = S6 decreases
```

## EMERGENCY ABORT
- On grip failure: return to HOME immediately
- Home sequence: ki_move(0, 17, 20.5, 0) — takes 1.5s
- DO NOT use stored S6=350 home — it is incorrect

## CALIBRATION PROCEDURE
1. `python guided_pick.py` → manual guided pick (arm hovers, place lighter under gripper)
2. `python cam_calibrate.py` → arm-guided camera calibration (builds pixel→arm map)
3. `python vision_pick.py` → autonomous camera-closed-loop pick
4. NIS API: `POST /cookoff/pick` → full IK sequence via NIS Protocol
5. NIS API: `POST /cookoff/dance` → Latino arm dance (8 genres)

## TIMING REFERENCE
```
Step          Duration   Wait
HOME          1500ms     1.5s
INSPECT       1200ms     1.5s + snapshot
PICK          1200ms     1.2s + snapshot
GRIP          600ms      0.8s
LIFT          1000ms     1.0s
SWING         1200ms     1.2s
DROP          1200ms     1.2s + snapshot
RELEASE       500ms      0.8s
HOME return   1500ms     1.5s
```

## REACHABILITY LIMITS (from Hiwonder docs)
```
X: -17 to +17 cm (±90° rotation)
Y: 5 to 25 cm forward
Z: 1.2 to 25.8 cm (table to max height)
IK has solution when: sqrt((m²+n²)) <= L2+L3 = 19cm
```
