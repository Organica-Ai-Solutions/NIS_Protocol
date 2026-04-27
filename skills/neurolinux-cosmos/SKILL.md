---
name: neurolinux-cosmos
description: NVIDIA Cosmos Cookoff and robot action planning for NeuroLinux. Use when the user wants to plan robot actions using vision-language reasoning, control the xArm with natural language, predict robot trajectories, or run sim-to-real transfer. Integrates Cosmos Reason 2 (port 8100), Cosmos Predict 2.5 (port 8200), and Cosmos Transfer 2.5 (port 8300) with Hiwonder xArm 1.6.
metadata:
  openclaw:
    emoji: "🤖"
    always: false
    primaryEnv: COSMOS_REASON_URL
    homepage: https://www.nvidia.com/en-us/ai/cosmos/
    requires:
      env: []
      anyBins: [python3, python]
---

# NeuroLinux Cosmos Cookoff

Vision-language robot action planning using NVIDIA Cosmos with NeuroLinux and xArm.

## When to Use

- "Pick up the red cube" → Cosmos plan → xArm execution
- Natural language robot commands
- Scene understanding + trajectory prediction
- Cosmos Cookoff challenge workflows

## NIS Protocol Endpoints

```
POST /cookoff/robot-plan            — Full Cosmos plan from image + text query
POST /openclaw/invoke               — via nis_cosmos_plan tool
GET  /openclaw/status               — check Cosmos stack availability
```

## OpenClaw Bridge Usage

```json
POST /openclaw/invoke
{
  "tool": "nis_cosmos_plan",
  "args": {
    "query": "Pick up the red cube and place it on the shelf",
    "robot_state": { "gripper": "open", "joints": [120, 120, 120, 120, 120, 120] },
    "image_base64": "<base64_scene_image>"
  }
}
```

## Cosmos Stack Configuration (H100)

```bash
# Set these env vars for direct H100 access:
COSMOS_REASON_URL=http://<h100-ip>:8100
COSMOS_PREDICT_URL=http://<h100-ip>:8200
COSMOS_TRANSFER_URL=http://<h100-ip>:8300
NIS_PROTOCOL_URL=http://localhost:8000
```

Falls back to NIS Protocol simulation if H100 not reachable.

## Integration Flow

```
User query
  → Cosmos Reason 2    (spatial understanding, action plan)
  → Cosmos Predict 2.5 (trajectory prediction — optional)
  → Cosmos Transfer 2.5(sim-to-real — optional)
  → HiwonderXArmDriver (execution on xArm 1.6)
```

## Team

| Name | Role | Focus |
|------|------|-------|
| Diego Torres | Founder / Lead Engineer | NIS Protocol, H100 stack, xArm integration, NeuroLinux OS |
| Camrin Neiss | Co-founder / Marketing + Frontend Dev | Dashboard UX, demo videos, growth strategy, React frontend |

**Camrin Neiss** — Marketing Analyst · Web Developer · Growth Generalist  
Built Plinza.com (React travel app, 500+ student pilot) and SpadesFitness.com (100+ sales).  
SDSU BS Marketing · Meta Front-End Developer Certificate · React (Coursera)  
Focused on: Cosmos Cookoff demo presentation, NeuroHub UI polish, investor pitch materials.

## References

- See `references/cosmos_setup.md` for H100 stack deployment
- See `references/cookoff_examples.md` for example queries and results
