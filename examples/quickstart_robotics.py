#!/usr/bin/env python3
"""
NIS Protocol - Robotics Quick Start Example

This example demonstrates how to use the NIS Protocol robotics API for:
- Forward Kinematics (FK)
- Inverse Kinematics (IK)
- Trajectory Planning

Run this after starting the NIS Protocol server:
    docker-compose up -d
    python examples/quickstart_robotics.py
"""

import requests
import json

BASE_URL = "http://localhost"  # Change if running on different host/port


def print_result(title: str, response: requests.Response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"🤖 {title}")
    print(f"{'='*60}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2, default=str)[:1000])  # Truncate long responses
        if len(json.dumps(data)) > 1000:
            print("... (truncated)")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


def example_forward_kinematics():
    """Calculate end-effector position from joint angles"""
    
    # 6-DOF manipulator arm joint angles (degrees)
    payload = {
        "robot_id": "arm_001",
        "robot_type": "manipulator",
        "joint_angles": [0, 30, 60, 0, 30, 0]  # 6 joints
    }
    
    response = requests.post(
        f"{BASE_URL}/robotics/forward_kinematics",
        json=payload
    )
    
    print_result("Forward Kinematics - 6-DOF Arm", response)
    return response


def example_inverse_kinematics():
    """Calculate joint angles to reach target position"""
    
    # Target position in 3D space
    payload = {
        "robot_id": "arm_001",
        "robot_type": "manipulator",
        "target_pose": {
            "position": [0.5, 0.3, 0.4]  # x, y, z in meters
        },
        "initial_guess": [0, 0, 0, 0, 0, 0]  # Starting joint angles
    }
    
    response = requests.post(
        f"{BASE_URL}/robotics/inverse_kinematics",
        json=payload
    )
    
    print_result("Inverse Kinematics - Reach Target", response)
    return response


def example_trajectory_planning():
    """Plan smooth trajectory through waypoints"""
    
    # Drone flight path waypoints
    payload = {
        "robot_id": "drone_001",
        "robot_type": "drone",
        "waypoints": [
            [0, 0, 0],      # Start position
            [1, 0, 1],      # First waypoint
            [2, 1, 1.5],    # Second waypoint
            [3, 0, 0.5]     # End position
        ],
        "duration": 10.0,    # Total flight time in seconds
        "num_points": 50     # Number of trajectory points
    }
    
    response = requests.post(
        f"{BASE_URL}/robotics/plan_trajectory",
        json=payload
    )
    
    print_result("Trajectory Planning - Drone Flight Path", response)
    return response


def example_get_capabilities():
    """Get robotics system capabilities"""
    
    response = requests.get(f"{BASE_URL}/robotics/capabilities")
    print_result("Robotics Capabilities", response)
    return response


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║       NIS Protocol - Robotics Quick Start Example          ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("⚠️  Server may not be fully ready")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to NIS Protocol server")
        print("   Make sure to run: docker-compose up -d")
        return
    
    print("✅ Connected to NIS Protocol server\n")
    
    # Run examples
    example_get_capabilities()
    example_forward_kinematics()
    example_inverse_kinematics()
    example_trajectory_planning()
    
    print(f"\n{'='*60}")
    print("✅ All examples completed!")
    print("📚 See API docs at: http://localhost/docs")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
