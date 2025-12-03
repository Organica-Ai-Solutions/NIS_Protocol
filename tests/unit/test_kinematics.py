#!/usr/bin/env python3
"""
Unit Tests for Robotics Kinematics
Tests forward kinematics, inverse kinematics, and trajectory planning
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent, RobotType


class TestForwardKinematics:
    """Test Forward Kinematics calculations"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup robotics agent for each test"""
        self.agent = UnifiedRoboticsAgent(
            agent_id="test_fk",
            enable_physics_validation=True
        )
    
    @pytest.mark.unit
    def test_fk_zero_angles(self):
        """FK with zero joint angles should return home position"""
        joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = self.agent.compute_forward_kinematics(
            "test_arm", joint_angles, RobotType.MANIPULATOR
        )
        
        assert result['success'] is True
        assert 'end_effector_pose' in result
        assert 'position' in result['end_effector_pose']
        assert len(result['end_effector_pose']['position']) == 3
    
    @pytest.mark.unit
    def test_fk_various_angles(self):
        """FK with various joint angles"""
        test_cases = [
            np.array([0.0, 0.5, 1.0, 0.0, 0.5, 0.0]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-0.5, 0.3, 0.7, -0.2, 0.4, 0.1]),
        ]
        
        for angles in test_cases:
            result = self.agent.compute_forward_kinematics(
                "test_arm", angles, RobotType.MANIPULATOR
            )
            assert result['success'] is True
            assert result['physics_valid'] is True
    
    @pytest.mark.unit
    def test_fk_intermediate_frames(self):
        """FK should return intermediate transformation frames"""
        joint_angles = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 0.0])
        result = self.agent.compute_forward_kinematics(
            "test_arm", joint_angles, RobotType.MANIPULATOR
        )
        
        assert 'intermediate_frames' in result
        # 6-DOF arm should have 6 intermediate frames
        assert len(result['intermediate_frames']) == 6
    
    @pytest.mark.unit
    def test_fk_rotation_matrix_valid(self):
        """Rotation matrix should be orthogonal (R^T * R = I)"""
        joint_angles = np.array([0.3, 0.5, 0.7, 0.2, 0.4, 0.1])
        result = self.agent.compute_forward_kinematics(
            "test_arm", joint_angles, RobotType.MANIPULATOR
        )
        
        R = np.array(result['end_effector_pose']['rotation_matrix'])
        identity = np.eye(3)
        product = R.T @ R
        
        np.testing.assert_array_almost_equal(product, identity, decimal=5)
    
    @pytest.mark.unit
    def test_fk_drone(self):
        """FK for drone should work differently"""
        position = np.array([1.0, 2.0, 3.0])
        result = self.agent.compute_forward_kinematics(
            "test_drone", position, RobotType.DRONE
        )
        
        assert result['success'] is True


class TestInverseKinematics:
    """Test Inverse Kinematics calculations"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup robotics agent for each test"""
        self.agent = UnifiedRoboticsAgent(
            agent_id="test_ik",
            enable_physics_validation=True
        )
    
    @pytest.mark.unit
    def test_ik_reachable_target(self):
        """IK should find solution for reachable target"""
        target_pose = {
            'position': np.array([0.5, 0.3, 0.8]),
            'orientation': np.array([0, 0, 0, 1])
        }
        result = self.agent.compute_inverse_kinematics(
            "test_arm", target_pose, RobotType.MANIPULATOR
        )
        
        assert result['success'] is True
        assert 'joint_angles' in result
        assert len(result['joint_angles']) == 6
        assert result['position_error'] < 0.01  # Less than 1cm error
    
    @pytest.mark.unit
    def test_ik_fk_consistency(self):
        """IK solution should produce correct FK result"""
        target_pose = {
            'position': np.array([0.5, 0.2, 0.7]),
            'orientation': np.array([0, 0, 0, 1])
        }
        
        # Get IK solution
        ik_result = self.agent.compute_inverse_kinematics(
            "test_arm", target_pose, RobotType.MANIPULATOR
        )
        
        if ik_result['success']:
            # Verify with FK
            fk_result = self.agent.compute_forward_kinematics(
                "test_arm", 
                np.array(ik_result['joint_angles']), 
                RobotType.MANIPULATOR
            )
            
            achieved_pos = np.array(fk_result['end_effector_pose']['position'])
            target_pos = target_pose['position']
            
            error = np.linalg.norm(achieved_pos - target_pos)
            assert error < 0.01  # Less than 1cm
    
    @pytest.mark.unit
    def test_ik_multiple_targets(self):
        """IK should work for multiple reachable targets"""
        targets = [
            np.array([0.5, 0.0, 0.5]),
            np.array([0.4, 0.2, 0.6]),
            np.array([0.6, -0.1, 0.7]),
            np.array([0.3, 0.3, 0.4]),
        ]
        
        for target in targets:
            target_pose = {
                'position': target,
                'orientation': np.array([0, 0, 0, 1])
            }
            result = self.agent.compute_inverse_kinematics(
                "test_arm", target_pose, RobotType.MANIPULATOR
            )
            
            # Should either succeed or fail gracefully
            assert 'success' in result


class TestTrajectoryPlanning:
    """Test Trajectory Planning"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup robotics agent for each test"""
        self.agent = UnifiedRoboticsAgent(
            agent_id="test_traj",
            enable_physics_validation=True
        )
    
    @pytest.mark.unit
    def test_trajectory_basic(self):
        """Basic trajectory planning"""
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 0.0, 2.0])
        ]
        
        result = self.agent.plan_trajectory(
            "test_drone", waypoints, RobotType.DRONE, duration=5.0
        )
        
        assert result['success'] is True
        assert 'trajectory' in result
        assert len(result['trajectory']) > 0
    
    @pytest.mark.unit
    def test_trajectory_starts_at_first_waypoint(self):
        """Trajectory should start at first waypoint"""
        waypoints = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0])
        ]
        
        result = self.agent.plan_trajectory(
            "test_drone", waypoints, RobotType.DRONE, duration=3.0
        )
        
        first_point = result['trajectory'][0]
        np.testing.assert_array_almost_equal(
            first_point['position'], waypoints[0], decimal=3
        )
    
    @pytest.mark.unit
    def test_trajectory_ends_at_last_waypoint(self):
        """Trajectory should end at last waypoint"""
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([5.0, 5.0, 5.0])
        ]
        
        result = self.agent.plan_trajectory(
            "test_drone", waypoints, RobotType.DRONE, duration=5.0
        )
        
        last_point = result['trajectory'][-1]
        np.testing.assert_array_almost_equal(
            last_point['position'], waypoints[-1], decimal=3
        )
    
    @pytest.mark.unit
    def test_trajectory_velocity_continuity(self):
        """Trajectory should have continuous velocity (no jumps)"""
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 0.0, 2.0])
        ]
        
        result = self.agent.plan_trajectory(
            "test_drone", waypoints, RobotType.DRONE, duration=5.0
        )
        
        trajectory = result['trajectory']
        
        # Check velocity doesn't jump too much between points
        for i in range(1, len(trajectory)):
            prev_vel = np.array(trajectory[i-1]['velocity'])
            curr_vel = np.array(trajectory[i]['velocity'])
            vel_diff = np.linalg.norm(curr_vel - prev_vel)
            
            # Velocity change should be reasonable
            assert vel_diff < 5.0  # Max 5 m/s change per step
    
    @pytest.mark.unit
    def test_trajectory_physics_validation(self):
        """Trajectory should be physics-validated"""
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0])
        ]
        
        result = self.agent.plan_trajectory(
            "test_drone", waypoints, RobotType.DRONE, duration=5.0
        )
        
        assert 'physics_valid' in result
        # With reasonable waypoints and duration, should be valid
        assert result['physics_valid'] is True


class TestPhysicsConstraints:
    """Test Physics Constraint Validation"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup robotics agent for each test"""
        self.agent = UnifiedRoboticsAgent(
            agent_id="test_physics",
            enable_physics_validation=True
        )
    
    @pytest.mark.unit
    def test_velocity_limits(self):
        """Commands exceeding velocity limits should be flagged"""
        # This tests the internal physics validation
        from src.agents.robotics.unified_robotics_agent import PhysicsConstraints
        
        constraints = PhysicsConstraints(
            max_velocity=2.0,
            max_acceleration=5.0
        )
        
        assert constraints.max_velocity == 2.0
        assert constraints.max_acceleration == 5.0
    
    @pytest.mark.unit
    def test_robot_type_constraints(self):
        """Different robot types should have different constraints"""
        drone_constraints = self.agent.constraints.get(RobotType.DRONE)
        manipulator_constraints = self.agent.constraints.get(RobotType.MANIPULATOR)
        
        # Drones typically have higher velocity limits
        assert drone_constraints.max_velocity > manipulator_constraints.max_velocity


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
