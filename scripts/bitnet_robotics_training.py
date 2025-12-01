#!/usr/bin/env python3
"""
🤖 BitNet Robotics & CAN Bus Training Script
NIS Protocol v4.0

Generates specialized training data for:
- Robotics (FK/IK, trajectory planning, control systems)
- CAN Bus protocol (automotive, industrial)
- Physics validation
- Autonomous systems

This trains the BitNet model for offline local use.
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict

# Training prompts organized by domain
ROBOTICS_PROMPTS = [
    # Forward/Inverse Kinematics
    "Explain forward kinematics for a 6-DOF robotic arm",
    "How does inverse kinematics work in robotics?",
    "What is the Denavit-Hartenberg convention?",
    "Explain the Jacobian matrix in robotics",
    "How do you solve the inverse kinematics problem?",
    "What are singularities in robot kinematics?",
    "Explain workspace analysis for robotic manipulators",
    "How does redundancy resolution work in robotics?",
    "What is the difference between analytical and numerical IK?",
    "Explain joint space vs task space in robotics",
    
    # Trajectory Planning
    "How does trajectory planning work for robots?",
    "Explain cubic spline interpolation for robot motion",
    "What is the trapezoidal velocity profile?",
    "How do you plan collision-free paths for robots?",
    "Explain RRT and RRT* path planning algorithms",
    "What is the A* algorithm for robot navigation?",
    "How does potential field navigation work?",
    "Explain time-optimal trajectory planning",
    "What is jerk-limited motion planning?",
    "How do you smooth robot trajectories?",
    
    # Control Systems
    "Explain PID control for robotics",
    "What is computed torque control?",
    "How does impedance control work?",
    "Explain force control in robotics",
    "What is adaptive control for robots?",
    "How do you tune a PID controller?",
    "Explain model predictive control (MPC)",
    "What is sliding mode control?",
    "How does feedforward control improve tracking?",
    "Explain the difference between position and velocity control",
    
    # Sensors & Perception
    "How do encoders work in robotics?",
    "Explain LIDAR for robot perception",
    "What is sensor fusion in robotics?",
    "How do IMUs work for robot orientation?",
    "Explain camera calibration for robotics",
    "What is visual servoing?",
    "How does depth sensing work?",
    "Explain SLAM for mobile robots",
    "What are force/torque sensors used for?",
    "How do proximity sensors work?",
    
    # Robot Types
    "Explain the kinematics of a SCARA robot",
    "How do delta robots achieve high speed?",
    "What are the advantages of collaborative robots?",
    "Explain mobile robot kinematics",
    "How do quadruped robots maintain balance?",
    "What is a parallel robot mechanism?",
    "Explain humanoid robot control challenges",
    "How do drone flight controllers work?",
    "What is a Stewart platform?",
    "Explain snake robot locomotion",
]

CAN_BUS_PROMPTS = [
    # CAN Protocol Basics
    "Explain the CAN bus protocol",
    "What is the CAN message frame format?",
    "How does CAN bus arbitration work?",
    "Explain CAN bus bit stuffing",
    "What is the difference between CAN 2.0A and 2.0B?",
    "How does CAN error detection work?",
    "Explain CAN bus termination",
    "What are CAN bus baud rates?",
    "How does CAN bus acknowledge work?",
    "Explain the CAN bus physical layer",
    
    # CAN FD & Advanced
    "What is CAN FD and its advantages?",
    "Explain the differences between CAN and CAN FD",
    "How does CAN FD achieve higher data rates?",
    "What is the CAN FD frame format?",
    "Explain CAN XL protocol",
    "How does J1939 protocol work?",
    "What is CANopen?",
    "Explain DeviceNet protocol",
    "How does OBD-II use CAN bus?",
    "What is UDS protocol over CAN?",
    
    # Automotive Applications
    "How is CAN bus used in vehicles?",
    "Explain engine control unit (ECU) communication",
    "What is the vehicle CAN network architecture?",
    "How do airbag systems use CAN bus?",
    "Explain ABS brake system CAN communication",
    "What is gateway ECU in automotive?",
    "How does CAN bus enable ADAS features?",
    "Explain powertrain CAN network",
    "What is body control module communication?",
    "How do infotainment systems use CAN?",
    
    # Industrial Applications
    "How is CAN bus used in industrial automation?",
    "Explain CAN bus in robotics applications",
    "What is CANopen for motion control?",
    "How do PLCs communicate over CAN?",
    "Explain CAN bus in agricultural machinery",
    "What is ISOBUS for farming equipment?",
    "How is CAN used in medical devices?",
    "Explain CAN bus in elevator systems",
    "What is CAN bus in marine applications?",
    "How do wind turbines use CAN bus?",
    
    # Implementation & Debugging
    "How do you implement a CAN bus driver?",
    "Explain CAN bus message filtering",
    "What tools are used for CAN bus debugging?",
    "How do you analyze CAN bus traffic?",
    "Explain CAN bus error handling strategies",
    "What is bus-off recovery in CAN?",
    "How do you design a CAN bus network?",
    "Explain CAN bus security considerations",
    "What is CAN bus message prioritization?",
    "How do you test CAN bus implementations?",
]

PHYSICS_VALIDATION_PROMPTS = [
    # Classical Mechanics
    "Explain Newton's laws for robotic systems",
    "How does conservation of momentum apply to robots?",
    "What is the Lagrangian formulation for robotics?",
    "Explain torque and angular momentum in manipulators",
    "How do you calculate robot dynamics?",
    "What is the mass matrix in robot dynamics?",
    "Explain Coriolis and centrifugal forces in robots",
    "How does gravity compensation work?",
    "What is the principle of virtual work?",
    "Explain energy-based control methods",
    
    # Dynamics & Control
    "How do you model robot dynamics?",
    "Explain the equations of motion for a robot arm",
    "What is inverse dynamics in robotics?",
    "How does friction affect robot motion?",
    "Explain backlash in gear systems",
    "What is compliance in robotic systems?",
    "How do you model contact dynamics?",
    "Explain impact dynamics for robots",
    "What is the role of inertia in robot control?",
    "How do you handle model uncertainties?",
]

AUTONOMOUS_SYSTEMS_PROMPTS = [
    # Autonomous Navigation
    "How do autonomous vehicles navigate?",
    "Explain path planning for self-driving cars",
    "What is behavior planning in autonomous systems?",
    "How does lane keeping work?",
    "Explain obstacle avoidance algorithms",
    "What is motion prediction for autonomous driving?",
    "How do autonomous drones navigate?",
    "Explain waypoint navigation",
    "What is geofencing for autonomous systems?",
    "How does fleet coordination work?",
    
    # Decision Making
    "How do autonomous systems make decisions?",
    "Explain state machines for robot behavior",
    "What is behavior trees in robotics?",
    "How does reinforcement learning apply to robotics?",
    "Explain mission planning for autonomous systems",
    "What is task allocation in multi-robot systems?",
    "How do robots handle uncertainty?",
    "Explain risk assessment in autonomous systems",
    "What is safety-critical decision making?",
    "How do autonomous systems handle edge cases?",
]

async def generate_training_example(session: aiohttp.ClientSession, prompt: str, domain: str) -> Dict:
    """Generate a single training example"""
    url = "http://localhost/chat"
    payload = {
        "message": prompt,
        "conversation_id": f"bitnet_train_{domain}_{hash(prompt) % 100000}",
        "provider": "anthropic"  # Use best provider for training data
    }
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {
                    "success": True,
                    "prompt": prompt,
                    "response": data.get("response", ""),
                    "provider": data.get("provider", "unknown"),
                    "domain": domain
                }
            else:
                return {"success": False, "prompt": prompt, "error": f"HTTP {resp.status}"}
    except Exception as e:
        return {"success": False, "prompt": prompt, "error": str(e)}

async def train_domain(session: aiohttp.ClientSession, prompts: List[str], domain: str):
    """Train on a specific domain"""
    print(f"\n{'='*60}")
    print(f"🎯 Training Domain: {domain.upper()}")
    print(f"{'='*60}")
    
    success_count = 0
    total = len(prompts)
    
    for i, prompt in enumerate(prompts):
        pct = int((i / total) * 100)
        print(f"[{pct:3d}%] {prompt[:50]}...", end=" ")
        
        result = await generate_training_example(session, prompt, domain)
        
        if result["success"]:
            success_count += 1
            print(f"✅ ({result['provider']})")
        else:
            print(f"⚠️ {result.get('error', 'Unknown error')[:30]}")
        
        await asyncio.sleep(0.3)  # Rate limiting
    
    print(f"\n✅ {domain}: {success_count}/{total} examples generated")
    return success_count

async def check_bitnet_status(session: aiohttp.ClientSession) -> Dict:
    """Check current BitNet training status"""
    try:
        async with session.get("http://localhost/models/bitnet/status", timeout=30) as resp:
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}

async def trigger_training(session: aiohttp.ClientSession):
    """Trigger a BitNet training session"""
    try:
        payload = {"reason": "Robotics and CAN bus training batch completed"}
        async with session.post("http://localhost/models/bitnet/train", json=payload, timeout=300) as resp:
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}

async def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     🤖 BitNet Robotics & CAN Bus Training System               ║
║     NIS Protocol v4.0                                          ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    async with aiohttp.ClientSession() as session:
        # Check initial status
        print("📊 Checking initial BitNet status...")
        status = await check_bitnet_status(session)
        initial_examples = status.get("metrics", {}).get("total_examples_collected", 0)
        print(f"   Current examples: {initial_examples}")
        
        # Train each domain
        total_generated = 0
        
        # Robotics training
        total_generated += await train_domain(session, ROBOTICS_PROMPTS, "robotics")
        
        # CAN Bus training
        total_generated += await train_domain(session, CAN_BUS_PROMPTS, "can_bus")
        
        # Physics validation training
        total_generated += await train_domain(session, PHYSICS_VALIDATION_PROMPTS, "physics")
        
        # Autonomous systems training
        total_generated += await train_domain(session, AUTONOMOUS_SYSTEMS_PROMPTS, "autonomous")
        
        # Check final status
        print("\n" + "="*60)
        print("📊 Final BitNet Status")
        print("="*60)
        
        status = await check_bitnet_status(session)
        final_examples = status.get("metrics", {}).get("total_examples_collected", 0)
        quality = status.get("metrics", {}).get("average_quality_score", 0)
        
        print(f"   Total examples: {final_examples}")
        print(f"   New examples: {final_examples - initial_examples}")
        print(f"   Quality score: {quality:.4f}")
        print(f"   Training available: {status.get('training_available', False)}")
        
        # Trigger training if we have enough examples
        if final_examples >= 400:
            print("\n🎯 Triggering BitNet training session...")
            train_result = await trigger_training(session)
            print(f"   Training result: {train_result.get('status', 'unknown')}")
        
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║     ✅ Training Complete!                                       ║
║     Total examples generated: {total_generated:4d}                            ║
║     BitNet ready for offline use: {final_examples >= 400}                       ║
╚════════════════════════════════════════════════════════════════╝
        """)

if __name__ == "__main__":
    asyncio.run(main())
