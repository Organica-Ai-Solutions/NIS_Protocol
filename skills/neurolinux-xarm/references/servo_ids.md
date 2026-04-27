# Hiwonder xArm Servo ID Map

## xArm 1.6 (6-DOF)

| Servo ID | Joint       | Range (°) | Notes             |
|----------|-------------|-----------|-------------------|
| 1        | Base rotate | 0–240     | Full 360 support with 1.6 |
| 2        | Shoulder    | 0–240     |                   |
| 3        | Elbow       | 0–240     |                   |
| 4        | Wrist pitch | 0–240     |                   |
| 5        | Wrist roll  | 0–240     |                   |
| 6        | Gripper     | 0–240     | 0 = open, 240 = closed |

## xArm 1s (5-DOF)

| Servo ID | Joint       | Range (°) |
|----------|-------------|-----------|
| 1        | Base rotate | 0–240     |
| 2        | Shoulder    | 0–240     |
| 3        | Elbow       | 0–240     |
| 4        | Wrist       | 0–240     |
| 5        | Gripper     | 0–240     |

## Home Position

All servos → 120° (center). Use `driver.home()` to reset.
