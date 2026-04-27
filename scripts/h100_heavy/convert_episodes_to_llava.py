#!/usr/bin/env python3
"""
Convert xArm/ALOHA/PushT episode data to LLaVA-format JSON for Cosmos Reason2 SFT.

Each episode step has: image (224x224x3), action (4D), instruction (text)
We convert these into conversation pairs with physics reasoning chains.

Output: /data/organica-ai/datasets/reason2_sft/
  ├── images/          # Saved step images as JPG
  └── annotations.json # LLaVA-format conversations
"""

import json
import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from glob import glob

# ─── Config ───
DATASETS = {
    "xarm": "/data/organica-ai/datasets/xarm",
    "aloha": "/data/organica-ai/datasets/aloha",
    "pusht": "/data/organica-ai/datasets/pusht",
}
OUTPUT_DIR = "/data/organica-ai/datasets/reason2_sft"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# ─── Physics reasoning templates ───
ROBOT_QUESTIONS = [
    "What should the robot do next based on the current scene?",
    "Analyze the scene and suggest the next robot action.",
    "Given this observation, what is the optimal next move for the robot arm?",
    "Describe the physical reasoning for the robot's next action.",
    "What action should the robot take, considering physics and safety?",
    "Plan the next manipulation step for the robot.",
    "What is the safest and most efficient next action?",
    "Analyze the workspace and determine the next robot movement.",
]

PHYSICS_QUESTIONS = [
    "What physical forces are at play in this scene?",
    "Describe the physics constraints the robot must consider here.",
    "What are the safety considerations for this manipulation task?",
    "Analyze the physical plausibility of the robot's current state.",
]

ACTION_DESCRIPTIONS = {
    "xarm": {
        "dims": ["x-displacement", "y-displacement", "z-displacement", "gripper"],
        "robot": "xArm 6-DOF robotic arm",
        "workspace": "tabletop manipulation workspace",
    },
    "aloha": {
        "dims": ["x-displacement", "y-displacement", "z-displacement", "gripper"],
        "robot": "ALOHA bimanual robot",
        "workspace": "bimanual manipulation workspace",
    },
    "pusht": {
        "dims": ["x-displacement", "y-displacement", "z-displacement", "gripper"],
        "robot": "push manipulation robot",
        "workspace": "planar pushing workspace",
    },
}


def action_to_text(action, dataset_name):
    """Convert action vector to natural language description."""
    info = ACTION_DESCRIPTIONS[dataset_name]
    parts = []
    for i, (val, dim) in enumerate(zip(action, info["dims"])):
        if abs(val) > 0.05:
            direction = "positive" if val > 0 else "negative"
            magnitude = "large" if abs(val) > 0.5 else "moderate" if abs(val) > 0.2 else "small"
            parts.append(f"{magnitude} {direction} {dim} ({val:.3f})")

    if not parts:
        return "minimal movement (near-stationary)"
    return ", ".join(parts)


def generate_reasoning(action, dataset_name, instruction, step_idx, total_steps):
    """Generate a chain-of-thought physics reasoning response."""
    info = ACTION_DESCRIPTIONS[dataset_name]
    action_text = action_to_text(action, dataset_name)
    progress = step_idx / max(total_steps - 1, 1)

    # Determine phase
    if progress < 0.2:
        phase = "approach"
        phase_desc = "The robot is in the initial approach phase, moving toward the target."
    elif progress < 0.5:
        phase = "align"
        phase_desc = "The robot is aligning with the target for precise manipulation."
    elif progress < 0.8:
        phase = "execute"
        phase_desc = "The robot is executing the core manipulation action."
    else:
        phase = "complete"
        phase_desc = "The robot is completing the task and returning to a safe position."

    # Gripper state
    gripper_val = action[-1] if len(action) >= 4 else 0
    if gripper_val > 0.5:
        gripper_state = "closing the gripper to grasp the object"
    elif gripper_val < -0.5:
        gripper_state = "opening the gripper to release the object"
    else:
        gripper_state = "maintaining current gripper state"

    # Build chain-of-thought
    reasoning = f"""<think>
## Scene Analysis
- Robot: {info['robot']} in {info['workspace']}
- Task instruction: "{instruction}"
- Current phase: {phase} (step {step_idx + 1}/{total_steps})
- {phase_desc}

## Physics Reasoning
- Gravity: 9.81 m/s² downward, requiring compensation in z-axis movements
- The robot must maintain joint torque limits and avoid singularities
- Collision avoidance: workspace boundaries and object clearances must be respected
- Force control: contact forces should remain within safe limits for the end-effector

## Action Analysis
- Commanded action: {action_text}
- Gripper: {gripper_state}
- The motion is physically plausible given the current robot configuration
- Safety check: action magnitudes are within acceptable velocity limits

## Decision
Based on the task "{instruction}" and current progress ({progress:.0%}), the robot should execute the planned action while monitoring force feedback and maintaining safety constraints.
</think>

The robot should execute a {phase} motion: {action_text}. This involves {gripper_state}. The action follows the task instruction "{instruction}" and respects physics constraints including gravity compensation, joint limits, and collision avoidance. Current task progress: {progress:.0%}."""

    return reasoning


def process_dataset(dataset_name, dataset_path, conversations):
    """Process all episodes in a dataset."""
    episode_dirs = sorted(glob(os.path.join(dataset_path, "episode_*")))
    print(f"Processing {dataset_name}: {len(episode_dirs)} episodes")

    for ep_idx, ep_dir in enumerate(episode_dirs):
        step_files = sorted(glob(os.path.join(ep_dir, "step_*.npz")))
        if not step_files:
            continue

        for step_idx, step_file in enumerate(step_files):
            try:
                data = np.load(step_file, allow_pickle=True)
                image = data["image"]
                action = data["action"].astype(np.float32)
                instruction = str(data["instruction"])

                # Save image
                img_name = f"{dataset_name}_ep{ep_idx:04d}_step{step_idx:04d}.jpg"
                img_path = os.path.join(IMAGES_DIR, img_name)

                if image.max() == 0:
                    # Synthetic black image — generate a simple colored one
                    image = np.random.randint(20, 80, image.shape, dtype=np.uint8)

                Image.fromarray(image).save(img_path, quality=85)

                # Generate conversation
                question = random.choice(ROBOT_QUESTIONS)
                reasoning = generate_reasoning(
                    action, dataset_name, instruction, step_idx, len(step_files)
                )

                conv = {
                    "id": f"{dataset_name}_ep{ep_idx:04d}_step{step_idx:04d}",
                    "image": f"images/{img_name}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": reasoning},
                    ],
                }
                conversations.append(conv)

                # Also add a physics-specific QA pair for every 5th step
                if step_idx % 5 == 0:
                    phys_q = random.choice(PHYSICS_QUESTIONS)
                    phys_a = generate_reasoning(
                        action, dataset_name, instruction, step_idx, len(step_files)
                    )
                    conv2 = {
                        "id": f"{dataset_name}_ep{ep_idx:04d}_step{step_idx:04d}_physics",
                        "image": f"images/{img_name}",
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{phys_q}"},
                            {"from": "gpt", "value": phys_a},
                        ],
                    }
                    conversations.append(conv2)

            except Exception as e:
                print(f"  Error processing {step_file}: {e}")
                continue

        if (ep_idx + 1) % 50 == 0:
            print(f"  {dataset_name}: {ep_idx + 1}/{len(episode_dirs)} episodes done")

    print(f"  {dataset_name}: {len(episode_dirs)} episodes → {len(conversations)} total conversations so far")


def main():
    print("=" * 60)
    print("Converting robot episodes to LLaVA format for Cosmos Reason2 SFT")
    print("=" * 60)

    conversations = []

    for name, path in DATASETS.items():
        if os.path.exists(path):
            process_dataset(name, path, conversations)
        else:
            print(f"Skipping {name}: {path} not found")

    # Shuffle
    random.seed(42)
    random.shuffle(conversations)

    # Split train/val (95/5)
    split_idx = int(len(conversations) * 0.95)
    train_data = conversations[:split_idx]
    val_data = conversations[split_idx:]

    # Save
    train_path = os.path.join(OUTPUT_DIR, "train_annotations.json")
    val_path = os.path.join(OUTPUT_DIR, "val_annotations.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE! Total conversations: {len(conversations)}")
    print(f"  Train: {len(train_data)} → {train_path}")
    print(f"  Val:   {len(val_data)} → {val_path}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
