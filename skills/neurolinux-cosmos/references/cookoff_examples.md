# Cosmos Cookoff Example Queries

## Pick and Place

```json
{
  "tool": "nis_cosmos_plan",
  "args": {
    "query": "Pick up the red cube on the left side of the table and place it in the blue bin",
    "robot_state": { "gripper": "open" }
  }
}
```

Expected response:
```json
{
  "action_recommendations": [
    "Move arm to pre-grasp position above red cube",
    "Lower gripper to cube level",
    "Close gripper on red cube",
    "Lift cube 10cm",
    "Translate arm to blue bin position",
    "Open gripper to release cube"
  ],
  "combined_confidence": 0.91
}
```

## Sort by Color

```
"Sort all cubes by color: red to left bin, blue to right bin, green to center"
```

## Stack Objects

```
"Stack the three wooden blocks in order of size, largest at bottom"
```

## Assembly Task

```
"Attach the blue cap to the white bottle"
```

## Safety Check

```
"Is the workspace clear? Check for obstacles before moving"
```

## Trajectory Only (no execution)

Pass `robot_state.dry_run: true` to get the plan without xArm execution.
