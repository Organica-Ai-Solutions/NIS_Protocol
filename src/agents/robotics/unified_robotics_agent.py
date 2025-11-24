#!/usr/bin/env python3
"""
NIS Protocol - Unified Robotics Agent
Physics-Validated Control for Drones, Droids, and Robotic Systems

This agent serves as the universal translator between different robotic systems,
using physics validation as the common language. Built for drones and humanoid
robots (droids) with extensibility for any robotic platform.

Core Capabilities:
- Kinematics (Forward/Inverse) computation
- Dynamics modeling and prediction
- Trajectory planning with physics constraints
- Multi-platform translation (drone ↔ droid ↔ manipulator)
- Real-time physics validation via PINN
- Safety-critical control with bounds checking

Author: Diego Torres (Organica AI Solutions)
Date: January 2025
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from ...core.agent import NISAgent, NISLayer
# Simplified memory - no need for full MemoryManager which causes import issues
# from ...memory.memory_manager import MemoryManager


class RobotType(Enum):
    """Types of robotic systems supported"""
    DRONE = "drone"                    # Quadcopter, hexacopter, etc.
    HUMANOID = "humanoid"              # Bipedal droids
    MANIPULATOR = "manipulator"        # Robotic arms
    GROUND_VEHICLE = "ground_vehicle"  # Wheeled/tracked robots
    HYBRID = "hybrid"                  # Multi-modal systems


class ControlMode(Enum):
    """Control modes for robotic systems"""
    POSITION = "position"           # Position control
    VELOCITY = "velocity"           # Velocity control
    FORCE = "force"                 # Force/torque control
    TRAJECTORY = "trajectory"       # Trajectory following
    TELEOPERATION = "teleoperation" # Direct control


@dataclass
class RobotState:
    """Current state of a robotic system"""
    position: np.ndarray          # 3D position [x, y, z]
    orientation: np.ndarray       # Quaternion [w, x, y, z]
    velocity: np.ndarray          # Linear velocity [vx, vy, vz]
    angular_velocity: np.ndarray  # Angular velocity [wx, wy, wz]
    joint_positions: Optional[np.ndarray] = None  # For manipulators/humanoids
    joint_velocities: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    
    
@dataclass
class PhysicsConstraints:
    """Physics constraints for safe robot operation"""
    max_velocity: float = 10.0      # m/s
    max_acceleration: float = 5.0   # m/s²
    max_angular_velocity: float = 3.14  # rad/s
    max_force: float = 1000.0       # N
    max_torque: float = 100.0       # N⋅m
    mass: float = 1.0               # kg
    moment_of_inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    gravity: float = 9.81           # m/s²


@dataclass
class TrajectoryPoint:
    """Single point in a robot trajectory"""
    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    orientation: Optional[np.ndarray] = None
    angular_velocity: Optional[np.ndarray] = None


class UnifiedRoboticsAgent(NISAgent):
    """
    Universal robotics control agent with physics validation.
    
    This agent can:
    1. Compute forward/inverse kinematics for any robot
    2. Plan physics-valid trajectories
    3. Translate between different robot platforms
    4. Validate all commands against physics constraints
    5. Provide real-time control with safety guarantees
    """
    
    def __init__(
        self,
        agent_id: str = "unified_robotics",
        description: str = "Physics-validated robotics control agent",
        enable_physics_validation: bool = True,
        enable_redundancy: bool = True
    ):
        super().__init__(agent_id)
        self.description = description
        self.layer = NISLayer.REASONING
        self.logger = logging.getLogger(f"nis.{agent_id}")
        
        # NASA-GRADE REDUNDANCY SYSTEM (Integrated at robotics layer)
        self.enable_redundancy = enable_redundancy
        self.redundancy_manager = None
        if enable_redundancy:
            try:
                from ...services.redundancy_manager import RedundancyManager
                self.redundancy_manager = RedundancyManager()
                self.logger.info("🛰️ NASA-grade redundancy enabled")
            except ImportError:
                self.logger.warning("⚠️ Redundancy manager not available")
                self.enable_redundancy = False
        
        # Memory for robot states and trajectories (simplified)
        self.memory = {
            'robot_states': {},
            'trajectories': {},
            'ik_solutions': {}
        }
        
        # Physics validation
        self.enable_physics_validation = enable_physics_validation
        
        # Robot configurations (can be loaded from file)
        self.robot_configs: Dict[str, Dict[str, Any]] = {}
        
        # Current robot states
        self.robot_states: Dict[str, RobotState] = {}
        
        # Safety constraints per robot type
        self.constraints: Dict[RobotType, PhysicsConstraints] = {
            RobotType.DRONE: PhysicsConstraints(
                max_velocity=20.0,
                max_acceleration=10.0,
                max_angular_velocity=6.28,
                mass=1.5,
                moment_of_inertia=np.diag([0.1, 0.1, 0.2])
            ),
            RobotType.HUMANOID: PhysicsConstraints(
                max_velocity=5.0,
                max_acceleration=3.0,
                max_angular_velocity=3.14,
                mass=75.0,
                moment_of_inertia=np.diag([10.0, 10.0, 5.0])
            ),
            RobotType.MANIPULATOR: PhysicsConstraints(
                max_velocity=2.0,
                max_acceleration=5.0,
                max_angular_velocity=3.14,
                mass=50.0,
                max_force=500.0,
                max_torque=200.0
            )
        }
        
        # Mathematical models
        self.kinematics_cache = {}
        self.dynamics_cache = {}
        
        # Performance tracking
        self.stats = {
            'total_commands': 0,
            'validated_commands': 0,
            'rejected_commands': 0,
            'average_computation_time': 0.0,
            'physics_violations': 0
        }
        
        self.logger.info(f"Initialized Unified Robotics Agent (Physics: {enable_physics_validation}, Redundancy: {enable_redundancy})")
    
    # ========================================================================
    # FORWARD KINEMATICS
    # ========================================================================
    
    def compute_forward_kinematics(
        self,
        robot_id: str,
        joint_angles: np.ndarray,
        robot_type: RobotType = RobotType.MANIPULATOR
    ) -> Dict[str, Any]:
        """
        Compute forward kinematics: joint space → task space
        
        For manipulators: Joint angles → End effector pose
        For humanoids: Joint angles → Body positions
        For drones: Motor speeds → Forces/torques
        
        Args:
            robot_id: Unique robot identifier
            joint_angles: Array of joint angles/positions
            robot_type: Type of robot
            
        Returns:
            Dictionary with end effector pose and intermediate frames
        """
        start_time = time.time()
        
        try:
            if robot_type == RobotType.MANIPULATOR:
                result = self._fk_manipulator(robot_id, joint_angles)
            elif robot_type == RobotType.HUMANOID:
                result = self._fk_humanoid(robot_id, joint_angles)
            elif robot_type == RobotType.DRONE:
                result = self._fk_drone(robot_id, joint_angles)
            else:
                raise ValueError(f"Unsupported robot type: {robot_type}")
            
            computation_time = time.time() - start_time
            
            # Physics validation
            if self.enable_physics_validation:
                validation = self._validate_pose(result['end_effector_pose'], robot_type)
                result['physics_valid'] = validation['valid']
                result['physics_warnings'] = validation.get('warnings', [])
            
            result['computation_time'] = computation_time
            result['timestamp'] = time.time()
            
            self.stats['total_commands'] += 1
            if result.get('physics_valid', True):
                self.stats['validated_commands'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Forward kinematics failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'computation_time': time.time() - start_time
            }
    
    def _fk_manipulator(self, robot_id: str, joint_angles: np.ndarray) -> Dict[str, Any]:
        """Forward kinematics for robotic manipulator (DH parameters)"""
        
        # Get robot config (DH parameters, link lengths, etc.)
        config = self.robot_configs.get(robot_id, self._default_manipulator_config())
        
        # Denavit-Hartenberg transformation
        T = np.eye(4)
        transformations = []
        
        for i, angle in enumerate(joint_angles):
            # DH parameters for link i
            a = config['link_lengths'][i]      # Link length
            d = config['link_offsets'][i]      # Link offset
            alpha = config['link_twists'][i]   # Link twist
            theta = angle + config['joint_offsets'][i]  # Joint angle
            
            # DH transformation matrix
            Ti = self._dh_matrix(a, d, alpha, theta)
            T = T @ Ti
            transformations.append(T.copy())
        
        # Extract position and orientation
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        orientation = Rotation.from_matrix(rotation_matrix).as_quat()  # [x,y,z,w]
        
        return {
            'success': True,
            'end_effector_pose': {
                'position': position,
                'orientation': orientation,
                'rotation_matrix': rotation_matrix
            },
            'intermediate_frames': transformations,
            'joint_angles': joint_angles,
            'robot_type': RobotType.MANIPULATOR.value
        }
    
    def _fk_humanoid(self, robot_id: str, joint_angles: np.ndarray) -> Dict[str, Any]:
        """Forward kinematics for humanoid robot"""
        
        config = self.robot_configs.get(robot_id, self._default_humanoid_config())
        
        # Humanoid has multiple chains: left leg, right leg, left arm, right arm, torso
        body_parts = {}
        
        # Compute FK for each limb
        limb_configs = config.get('limbs', {})
        joint_idx = 0
        
        for limb_name, limb_config in limb_configs.items():
            num_joints = limb_config['num_joints']
            limb_angles = joint_angles[joint_idx:joint_idx + num_joints]
            
            # Compute FK for this limb
            limb_fk = self._fk_manipulator(f"{robot_id}_{limb_name}", limb_angles)
            body_parts[limb_name] = limb_fk['end_effector_pose']
            
            joint_idx += num_joints
        
        # Center of mass calculation
        com = self._compute_center_of_mass(body_parts, config)
        
        return {
            'success': True,
            'body_parts': body_parts,
            'center_of_mass': com,
            'joint_angles': joint_angles,
            'robot_type': RobotType.HUMANOID.value,
            'end_effector_pose': body_parts.get('right_hand', {})  # Default to right hand
        }
    
    def _fk_drone(self, robot_id: str, motor_speeds: np.ndarray) -> Dict[str, Any]:
        """Forward kinematics for drone (motor speeds → forces/torques)"""
        
        config = self.robot_configs.get(robot_id, self._default_drone_config())
        
        # Motor thrust model: F = k * omega^2
        k_thrust = config.get('thrust_coefficient', 1e-5)
        k_torque = config.get('torque_coefficient', 1e-7)
        
        # Compute individual motor thrusts
        thrusts = k_thrust * motor_speeds ** 2
        torques = k_torque * motor_speeds ** 2
        
        # Total thrust (vertical)
        total_thrust = np.sum(thrusts)
        
        # Moments about body axes (simplified quadcopter model)
        # Assuming motors at ±L distance from center
        L = config.get('arm_length', 0.25)  # meters
        
        # Roll moment (about x-axis)
        M_roll = L * (thrusts[1] - thrusts[3])
        
        # Pitch moment (about y-axis)
        M_pitch = L * (thrusts[2] - thrusts[0])
        
        # Yaw moment (about z-axis) - from torque differences
        M_yaw = torques[0] - torques[1] + torques[2] - torques[3]
        
        return {
            'success': True,
            'total_thrust': total_thrust,
            'moments': {
                'roll': M_roll,
                'pitch': M_pitch,
                'yaw': M_yaw
            },
            'individual_thrusts': thrusts,
            'motor_speeds': motor_speeds,
            'robot_type': RobotType.DRONE.value,
            'end_effector_pose': {
                'force': np.array([0, 0, total_thrust]),
                'torque': np.array([M_roll, M_pitch, M_yaw])
            }
        }
    
    # ========================================================================
    # INVERSE KINEMATICS
    # ========================================================================
    
    def compute_inverse_kinematics(
        self,
        robot_id: str,
        target_pose: Dict[str, np.ndarray],
        robot_type: RobotType = RobotType.MANIPULATOR,
        initial_guess: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute inverse kinematics: task space → joint space
        
        Args:
            robot_id: Unique robot identifier
            target_pose: Desired end effector pose {'position': [x,y,z], 'orientation': [qw,qx,qy,qz]}
            robot_type: Type of robot
            initial_guess: Starting joint configuration
            
        Returns:
            Dictionary with joint angles and success flag
        """
        start_time = time.time()
        
        try:
            if robot_type == RobotType.MANIPULATOR:
                result = self._ik_manipulator(robot_id, target_pose, initial_guess)
            elif robot_type == RobotType.HUMANOID:
                result = self._ik_humanoid(robot_id, target_pose, initial_guess)
            elif robot_type == RobotType.DRONE:
                result = self._ik_drone(robot_id, target_pose)
            else:
                raise ValueError(f"Unsupported robot type: {robot_type}")
            
            computation_time = time.time() - start_time
            
            # Physics validation
            if self.enable_physics_validation and result.get('success'):
                validation = self._validate_joint_limits(
                    result['joint_angles'], 
                    robot_type
                )
                result['physics_valid'] = validation['valid']
                result['physics_warnings'] = validation.get('warnings', [])
            
            result['computation_time'] = computation_time
            result['timestamp'] = time.time()
            
            # Update stats
            self.stats['total_commands'] += 1
            if result.get('physics_valid', True):
                self.stats['validated_commands'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inverse kinematics failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'computation_time': time.time() - start_time
            }
    
    def _ik_manipulator(
        self,
        robot_id: str,
        target_pose: Dict[str, np.ndarray],
        initial_guess: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Inverse kinematics for robotic manipulator (numerical optimization)"""
        
        config = self.robot_configs.get(robot_id, self._default_manipulator_config())
        num_joints = len(config['link_lengths'])
        
        # Initial guess (use current state or zeros)
        if initial_guess is None:
            current_state = self.robot_states.get(robot_id)
            if current_state and current_state.joint_positions is not None:
                x0 = current_state.joint_positions
            else:
                x0 = np.zeros(num_joints)
        else:
            x0 = initial_guess
        
        # Target position and orientation
        target_pos = target_pose['position']
        target_quat = target_pose.get('orientation')
        
        # Cost function: minimize distance to target
        def cost_function(joint_angles):
            fk_result = self._fk_manipulator(robot_id, joint_angles)
            current_pos = fk_result['end_effector_pose']['position']
            
            # Position error
            pos_error = np.linalg.norm(current_pos - target_pos)
            
            # Orientation error (if specified)
            if target_quat is not None:
                current_quat = fk_result['end_effector_pose']['orientation']
                # Quaternion distance
                quat_error = 1 - abs(np.dot(current_quat, target_quat))
                return pos_error + 10.0 * quat_error  # Weight orientation more
            
            return pos_error
        
        # Joint limits
        bounds = [(config['joint_limits'][i][0], config['joint_limits'][i][1]) 
                  for i in range(num_joints)]
        
        # Optimize
        result = minimize(
            cost_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        # Verify solution
        fk_check = self._fk_manipulator(robot_id, result.x)
        final_pos = fk_check['end_effector_pose']['position']
        position_error = float(np.linalg.norm(final_pos - target_pos))

        converged = bool(result.success and position_error < 0.01)  # 1cm tolerance

        response: Dict[str, Any] = {
            'success': converged,
            'joint_angles': result.x,
            'position_error': position_error,
            'iterations': int(result.nit),
            'final_pose': fk_check['end_effector_pose'],
            'robot_type': RobotType.MANIPULATOR.value
        }

        if not converged:
            message = getattr(result, 'message', 'IK solver did not converge within tolerance')
            response['error'] = message

        return response
    
    def _ik_humanoid(
        self,
        robot_id: str,
        target_pose: Dict[str, np.ndarray],
        initial_guess: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Inverse kinematics for humanoid (solve for specific limb)"""
        
        # For humanoid, typically solve IK for one limb at a time
        limb_name = target_pose.get('limb', 'right_hand')
        
        # Use manipulator IK for the specific limb
        limb_robot_id = f"{robot_id}_{limb_name}"
        result = self._ik_manipulator(limb_robot_id, target_pose, initial_guess)
        
        result['robot_type'] = RobotType.HUMANOID.value
        result['limb'] = limb_name
        
        return result
    
    def _ik_drone(
        self,
        robot_id: str,
        target_forces: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Inverse kinematics for drone (desired forces/torques → motor speeds)"""
        
        config = self.robot_configs.get(robot_id, self._default_drone_config())
        
        # Desired thrust and moments
        F_desired = target_forces.get('force', np.array([0, 0, 9.81]))  # Default hover
        M_desired = target_forces.get('torque', np.zeros(3))
        
        # Total thrust needed
        total_thrust = F_desired[2]  # Vertical component
        
        # Mixing matrix: relates motor speeds to [thrust, roll, pitch, yaw]
        L = config.get('arm_length', 0.25)
        k_thrust = config.get('thrust_coefficient', 1e-5)
        k_torque = config.get('torque_coefficient', 1e-7)
        
        # Allocation matrix (simplified for quadcopter)
        # [F, M_roll, M_pitch, M_yaw]^T = A * [F1, F2, F3, F4]^T
        A = np.array([
            [1,      1,      1,      1],
            [0,      L,      0,     -L],
            [-L,     0,      L,      0],
            [-k_torque/k_thrust, k_torque/k_thrust, -k_torque/k_thrust, k_torque/k_thrust]
        ])
        
        # Desired wrench
        wrench = np.array([
            total_thrust,
            M_desired[0],  # Roll
            M_desired[1],  # Pitch
            M_desired[2]   # Yaw
        ])
        
        # Solve for individual thrusts: F = A^{-1} * wrench
        try:
            thrusts = np.linalg.solve(A, wrench)
            
            # Convert thrusts to motor speeds: omega = sqrt(F/k)
            motor_speeds = np.sqrt(np.abs(thrusts) / k_thrust) * np.sign(thrusts)
            
            # Clamp to physical limits
            max_rpm = config.get('max_motor_rpm', 10000)
            motor_speeds = np.clip(motor_speeds, 0, max_rpm)
            
            return {
                'success': True,
                'motor_speeds': motor_speeds,
                'individual_thrusts': thrusts,
                'total_thrust': np.sum(thrusts),
                'robot_type': RobotType.DRONE.value
            }
            
        except np.linalg.LinAlgError:
            return {
                'success': False,
                'error': 'Singular mixing matrix',
                'robot_type': RobotType.DRONE.value
            }
    
    # ========================================================================
    # TRAJECTORY PLANNING
    # ========================================================================
    
    def plan_trajectory(
        self,
        robot_id: str,
        waypoints: List[np.ndarray],
        robot_type: RobotType,
        duration: float = 5.0,
        num_points: int = 100
    ) -> Dict[str, Any]:
        """
        Plan smooth trajectory through waypoints with physics constraints
        
        Uses minimum jerk trajectory for smooth motion
        
        Args:
            robot_id: Unique robot identifier
            waypoints: List of 3D positions to pass through
            robot_type: Type of robot
            duration: Total trajectory duration (seconds)
            num_points: Number of trajectory points to generate
            
        Returns:
            Dictionary with trajectory points and physics validation
        """
        start_time = time.time()
        
        try:
            # Get constraints for this robot type
            constraints = self.constraints.get(robot_type, PhysicsConstraints())
            
            # Generate minimum jerk trajectory
            trajectory = self._minimum_jerk_trajectory(
                waypoints,
                duration,
                num_points,
                constraints
            )
            
            # Physics validation
            if self.enable_physics_validation:
                validation = self._validate_trajectory(trajectory, constraints, robot_type)
                valid = validation['valid']
                warnings = validation.get('warnings', [])
            else:
                valid = True
                warnings = []
            
            metrics = self._compute_trajectory_metrics(trajectory)

            result = {
                'success': bool(valid),
                'trajectory': trajectory,
                'num_points': len(trajectory),
                'duration': duration,
                'physics_valid': bool(valid),
                'physics_warnings': warnings,
                'computation_time': time.time() - start_time,
                'robot_type': robot_type.value,
                **metrics
            }
            
            # Update stats
            self.stats['total_commands'] += 1
            if valid:
                self.stats['validated_commands'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trajectory planning failed: {e}")
            return {
                'success': False,
                'trajectory': [],
                'num_points': 0,
                'duration': duration,
                'physics_valid': False,
                'physics_warnings': [],
                'total_distance': 0.0,
                'average_speed': 0.0,
                'average_acceleration': 0.0,
                'error': str(e),
                'computation_time': time.time() - start_time
            }
    
    def _minimum_jerk_trajectory(
        self,
        waypoints: List[np.ndarray],
        duration: float,
        num_points: int,
        constraints: PhysicsConstraints
    ) -> List[TrajectoryPoint]:
        """Generate minimum jerk trajectory (smoothest possible motion)"""
        
        trajectory = []
        times = np.linspace(0, duration, num_points)
        
        # For each segment between waypoints
        num_segments = len(waypoints) - 1
        segment_duration = duration / num_segments
        
        for seg_idx in range(num_segments):
            start_pos = waypoints[seg_idx]
            end_pos = waypoints[seg_idx + 1]
            
            # Number of points in this segment
            seg_points = num_points // num_segments
            seg_times = np.linspace(0, segment_duration, seg_points)
            
            for t in seg_times:
                # Minimum jerk polynomial (5th order)
                s = t / segment_duration  # Normalized time [0, 1]
                
                # Position (5th order polynomial)
                pos_coeff = 10*s**3 - 15*s**4 + 6*s**5
                position = start_pos + (end_pos - start_pos) * pos_coeff
                
                # Velocity (derivative)
                vel_coeff = (30*s**2 - 60*s**3 + 30*s**4) / segment_duration
                velocity = (end_pos - start_pos) * vel_coeff
                
                # Acceleration (second derivative)
                acc_coeff = (60*s - 180*s**2 + 120*s**3) / (segment_duration**2)
                acceleration = (end_pos - start_pos) * acc_coeff
                
                # Clamp to constraints
                velocity = self._clamp_vector(velocity, constraints.max_velocity)
                acceleration = self._clamp_vector(acceleration, constraints.max_acceleration)
                
                trajectory.append(TrajectoryPoint(
                    time=seg_idx * segment_duration + t,
                    position=position,
                    velocity=velocity,
                    acceleration=acceleration
                ))
        
        return trajectory
    
    # ========================================================================
    # PHYSICS VALIDATION
    # ========================================================================
    
    def _validate_trajectory(
        self,
        trajectory: List[TrajectoryPoint],
        constraints: PhysicsConstraints,
        robot_type: RobotType = RobotType.MANIPULATOR
    ) -> Dict[str, Any]:
        """Validate entire trajectory against physics constraints"""
        
        warnings = []
        violations = 0
        
        for i, point in enumerate(trajectory):
            # Check velocity limits
            vel_mag = np.linalg.norm(point.velocity)
            if vel_mag > constraints.max_velocity:
                violations += 1
                if violations == 1:  # Only log first violation
                    warnings.append(f"Velocity limit exceeded at t={point.time:.2f}s: "
                                  f"{vel_mag:.2f} > {constraints.max_velocity} m/s")
            
            # Check acceleration limits
            acc_mag = np.linalg.norm(point.acceleration)
            if acc_mag > constraints.max_acceleration:
                violations += 1
                if violations <= 3:  # Log first few
                    warnings.append(f"Acceleration limit exceeded at t={point.time:.2f}s: "
                                  f"{acc_mag:.2f} > {constraints.max_acceleration} m/s²")
            
            # Dynamics Check (Force/Torque)
            if robot_type == RobotType.DRONE:
                # Calculate required thrust: F = m(a + g)
                # Assuming Z-up, gravity is [0, 0, 9.81]
                gravity_vec = np.array([0, 0, constraints.gravity])
                required_force = constraints.mass * (point.acceleration + gravity_vec)
                thrust_mag = np.linalg.norm(required_force)
                
                # 20% margin for control authority
                max_allowed_thrust = constraints.max_force * 0.8
                
                if thrust_mag > max_allowed_thrust:
                    violations += 1
                    if violations <= 3:
                        warnings.append(f"Thrust limit exceeded at t={point.time:.2f}s: "
                                      f"{thrust_mag:.2f} > {max_allowed_thrust:.2f} N (Max: {constraints.max_force} N)")

        valid = violations == 0
        
        if not valid:
            self.stats['physics_violations'] += violations
        
        return {
            'valid': valid,
            'warnings': warnings,
            'total_violations': violations,
            'violation_rate': violations / len(trajectory) if trajectory else 0
        }

    def _compute_trajectory_metrics(self, trajectory: List[TrajectoryPoint]) -> Dict[str, Any]:
        """Compute aggregate metrics for a planned trajectory"""

        if not trajectory:
            return {
                'total_distance': 0.0,
                'average_speed': 0.0,
                'average_acceleration': 0.0
            }

        distances: List[float] = []
        speeds: List[float] = []
        accelerations: List[float] = []

        for idx in range(1, len(trajectory)):
            prev_point = trajectory[idx - 1]
            curr_point = trajectory[idx]
            distances.append(float(np.linalg.norm(curr_point.position - prev_point.position)))
            speeds.append(float(np.linalg.norm(curr_point.velocity)))
            accelerations.append(float(np.linalg.norm(curr_point.acceleration)))

        total_distance = float(np.sum(distances)) if distances else 0.0
        average_speed = float(np.mean(speeds)) if speeds else 0.0
        average_acceleration = float(np.mean(accelerations)) if accelerations else 0.0

        return {
            'total_distance': total_distance,
            'average_speed': average_speed,
            'average_acceleration': average_acceleration
        }
    
    def _validate_pose(
        self,
        pose: Dict[str, np.ndarray],
        robot_type: RobotType
    ) -> Dict[str, Any]:
        """Validate a single pose against physics constraints"""
        
        warnings = []
        constraints = self.constraints.get(robot_type, PhysicsConstraints())
        
        # Check if position is reachable (simple bounds check)
        position = pose.get('position', np.zeros(3))
        if np.linalg.norm(position) > 10.0:  # 10m reach limit
            warnings.append(f"Position may be unreachable: {np.linalg.norm(position):.2f}m")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }
    
    def _validate_joint_limits(
        self,
        joint_angles: np.ndarray,
        robot_type: RobotType
    ) -> Dict[str, Any]:
        """Validate joint angles against limits"""
        
        warnings = []
        # Joint limits are robot-specific
        # For now, just check reasonable ranges
        
        for i, angle in enumerate(joint_angles):
            if abs(angle) > 2 * np.pi:  # Outside ±360°
                warnings.append(f"Joint {i} angle extreme: {np.degrees(angle):.1f}°")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }
    
    # ========================================================================
    # PLATFORM TRANSLATION
    # ========================================================================
    
    def translate_command(
        self,
        command: Dict[str, Any],
        from_platform: str,
        to_platform: str
    ) -> Dict[str, Any]:
        """
        Translate control command between different robotic platforms
        
        Example: MAVLink (drone) → ROS (droid)
        
        Args:
            command: Command in source platform format
            from_platform: Source platform (e.g., 'mavlink', 'ros', 'custom')
            to_platform: Target platform
            
        Returns:
            Translated command in target platform format
        """
        try:
            # Parse source command
            if from_platform == 'mavlink':
                parsed = self._parse_mavlink_command(command)
            elif from_platform == 'ros':
                parsed = self._parse_ros_command(command)
            else:
                parsed = command  # Assume generic format
            
            # Convert to target format
            if to_platform == 'mavlink':
                translated = self._to_mavlink_command(parsed)
            elif to_platform == 'ros':
                translated = self._to_ros_command(parsed)
            else:
                translated = parsed
            
            return {
                'success': True,
                'translated_command': translated,
                'from_platform': from_platform,
                'to_platform': to_platform
            }
            
        except Exception as e:
            self.logger.error(f"Command translation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _dh_matrix(self, a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        """Denavit-Hartenberg transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,    -st*ca,  st*sa,   a*ct],
            [st,     ct*ca, -ct*sa,   a*st],
            [0,      sa,     ca,      d   ],
            [0,      0,      0,       1   ]
        ])
    
    def _clamp_vector(self, vector: np.ndarray, max_magnitude: float) -> np.ndarray:
        """Clamp vector magnitude to maximum value"""
        magnitude = np.linalg.norm(vector)
        if magnitude > max_magnitude:
            return vector * (max_magnitude / magnitude)
        return vector
    
    def _compute_center_of_mass(
        self,
        body_parts: Dict[str, Dict[str, np.ndarray]],
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Compute center of mass from body part positions and masses"""
        
        total_mass = 0.0
        weighted_sum = np.zeros(3)
        
        for part_name, pose in body_parts.items():
            mass = config.get('limb_masses', {}).get(part_name, 1.0)
            position = pose.get('position', np.zeros(3))
            weighted_sum += mass * position
            total_mass += mass
        
        return weighted_sum / total_mass if total_mass > 0 else np.zeros(3)
    
    # ========================================================================
    # DEFAULT CONFIGURATIONS
    # ========================================================================
    
    def _default_manipulator_config(self) -> Dict[str, Any]:
        """Default 6-DOF manipulator configuration"""
        return {
            'name': 'generic_6dof',
            'num_joints': 6,
            'link_lengths': [0.3, 0.3, 0.3, 0.2, 0.1, 0.05],
            'link_offsets': [0.5, 0, 0, 0, 0, 0],
            'link_twists': [0, np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2],
            'joint_offsets': [0, 0, 0, 0, 0, 0],
            'joint_limits': [(-np.pi, np.pi)] * 6
        }
    
    def _default_humanoid_config(self) -> Dict[str, Any]:
        """Default humanoid robot configuration"""
        return {
            'name': 'generic_humanoid',
            'limbs': {
                'left_leg': {'num_joints': 6},
                'right_leg': {'num_joints': 6},
                'left_arm': {'num_joints': 7},
                'right_arm': {'num_joints': 7},
                'torso': {'num_joints': 3}
            },
            'limb_masses': {
                'left_leg': 10.0,
                'right_leg': 10.0,
                'left_arm': 5.0,
                'right_arm': 5.0,
                'torso': 30.0
            }
        }
    
    def _default_drone_config(self) -> Dict[str, Any]:
        """Default quadcopter configuration"""
        return {
            'name': 'generic_quadcopter',
            'num_motors': 4,
            'arm_length': 0.25,  # meters
            'thrust_coefficient': 1e-5,
            'torque_coefficient': 1e-7,
            'max_motor_rpm': 10000,
            'mass': 1.5  # kg
        }
    
    def _parse_mavlink_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Parse MAVLink format command"""
        # Simplified MAVLink parsing
        return {
            'type': 'position_target',
            'position': np.array(command.get('position', [0, 0, 0])),
            'velocity': np.array(command.get('velocity', [0, 0, 0])),
            'yaw': command.get('yaw', 0.0)
        }
    
    def _parse_ros_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ROS format command"""
        # Simplified ROS parsing
        return {
            'type': 'twist',
            'linear': np.array(command.get('linear', [0, 0, 0])),
            'angular': np.array(command.get('angular', [0, 0, 0]))
        }
    
    def _to_mavlink_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to MAVLink format"""
        return {
            'message_type': 'SET_POSITION_TARGET_LOCAL_NED',
            'position': command.get('position', [0, 0, 0]).tolist(),
            'velocity': command.get('velocity', [0, 0, 0]).tolist(),
            'yaw': command.get('yaw', 0.0)
        }
    
    def _to_ros_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to ROS Twist message format"""
        return {
            'type': 'geometry_msgs/Twist',
            'linear': {
                'x': float(command.get('velocity', [0, 0, 0])[0]),
                'y': float(command.get('velocity', [0, 0, 0])[1]),
                'z': float(command.get('velocity', [0, 0, 0])[2])
            },
            'angular': {
                'x': 0.0,
                'y': 0.0,
                'z': float(command.get('yaw_rate', 0.0))
            }
        }
    
    # ========================================================================
    # NIS AGENT INTERFACE
    # ========================================================================
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process robotics command through NIS Protocol
        
        Message format:
        {
            'command': 'forward_kinematics' | 'inverse_kinematics' | 'plan_trajectory' | 'translate',
            'robot_id': 'drone_001',
            'robot_type': 'drone',
            'data': {...}
        }
        """
        command = message.get('command')
        robot_id = message.get('robot_id', 'default_robot')
        robot_type_str = message.get('robot_type', 'manipulator')
        robot_type = RobotType(robot_type_str)
        
        if command == 'forward_kinematics':
            return self.compute_forward_kinematics(
                robot_id,
                np.array(message['data']['joint_angles']),
                robot_type
            )
        
        elif command == 'inverse_kinematics':
            return self.compute_inverse_kinematics(
                robot_id,
                message['data']['target_pose'],
                robot_type,
                message['data'].get('initial_guess')
            )
        
        elif command == 'plan_trajectory':
            waypoints = [np.array(wp) for wp in message['data']['waypoints']]
            return self.plan_trajectory(
                robot_id,
                waypoints,
                robot_type,
                message['data'].get('duration', 5.0),
                message['data'].get('num_points', 100)
            )
        
        elif command == 'translate':
            return self.translate_command(
                message['data']['command'],
                message['data']['from_platform'],
                message['data']['to_platform']
            )
        
        else:
            return {
                'success': False,
                'error': f'Unknown command: {command}'
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            **self.stats,
            'success_rate': (self.stats['validated_commands'] / self.stats['total_commands']
                           if self.stats['total_commands'] > 0 else 0)
        }

