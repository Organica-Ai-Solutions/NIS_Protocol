#!/usr/bin/env python3
import sys
import os
import numpy as np
import asyncio
import logging
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent, RobotType
    from src.core.edge_ai_operating_system import create_drone_ai_os
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you are running from the project root or tests directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_robotics")

@pytest.mark.asyncio
async def test_robotics():
    logger.info("🤖 STARTING ROBOTICS TESTS")
    
    # 1. Initialize Agent
    try:
        agent = UnifiedRoboticsAgent(agent_id="test_robot")
        logger.info("✅ Agent Initialized")
    except Exception as e:
        logger.error(f"❌ Agent Initialization Failed: {e}")
        return
    
    # 2. Test Forward Kinematics (Manipulator)
    logger.info("\n🧪 Testing Forward Kinematics (Manipulator)...")
    try:
        joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fk_result = agent.compute_forward_kinematics("arm_1", joint_angles, RobotType.MANIPULATOR)
        
        if fk_result['success']:
            pos = fk_result['end_effector_pose']['position']
            logger.info(f"✅ FK Success. End Effector Pos: {pos}")
        else:
            logger.error(f"❌ FK Failed: {fk_result.get('error')}")
    except Exception as e:
        logger.error(f"❌ FK Exception: {e}")
        
    # 3. Test Inverse Kinematics (Manipulator)
    logger.info("\n🧪 Testing Inverse Kinematics (Manipulator)...")
    try:
        target_pose = {
            'position': np.array([0.5, 0.0, 0.5]), # Target position
            'orientation': np.array([0, 0, 0, 1])
        }
        ik_result = agent.compute_inverse_kinematics("arm_1", target_pose, RobotType.MANIPULATOR)
        
        if ik_result['success']:
            angles = ik_result['joint_angles']
            logger.info(f"✅ IK Success. Joint Angles: {angles}")
            logger.info(f"   Position Error: {ik_result['position_error']:.6f}")
        else:
            logger.error(f"❌ IK Failed: {ik_result.get('error')}")
    except Exception as e:
        logger.error(f"❌ IK Exception: {e}")

    # 4. Test Drone Trajectory
    logger.info("\n🧪 Testing Drone Trajectory Planning...")
    try:
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 5.0]),
            np.array([10.0, 0.0, 5.0]),
            np.array([10.0, 10.0, 5.0])
        ]
        traj_result = agent.plan_trajectory("drone_1", waypoints, RobotType.DRONE, duration=10.0)
        
        if traj_result['success']:
            logger.info(f"✅ Trajectory Planned. Points: {len(traj_result['trajectory'])}")
            logger.info(f"   Physics Valid: {traj_result['physics_valid']}")
            if not traj_result['physics_valid']:
                logger.warning(f"   Warnings: {traj_result['physics_warnings']}")
        else:
            logger.error(f"❌ Trajectory Planning Failed: {traj_result.get('error')}")
    except Exception as e:
        logger.error(f"❌ Trajectory Exception: {e}")

    # 5. Test Edge OS Integration
    logger.info("\n🧪 Testing Edge OS Integration...")
    try:
        drone_os = create_drone_ai_os()
        status = await drone_os.initialize_edge_system()
        
        if drone_os.robotics_agent:
            logger.info("✅ Edge OS initialized Robotics Agent successfully")
            logger.info(f"   OS Status: {status['status']}")
        else:
            logger.error("❌ Edge OS failed to initialize Robotics Agent")
    except Exception as e:
        logger.error(f"❌ Edge OS Exception: {e}")
        
    logger.info("\n🏁 ROBOTICS TESTS COMPLETED")

if __name__ == "__main__":
    asyncio.run(test_robotics())
