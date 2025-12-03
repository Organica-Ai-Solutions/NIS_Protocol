#!/usr/bin/env python3
"""
NIS Protocol - Isaac Bridge Agent
ROS 2 communication bridge between NIS Protocol and NVIDIA Isaac

Features:
- ROS 2 topic publishing/subscribing
- Trajectory execution with physics validation
- Real-time robot state monitoring
- Graceful fallback when ROS not available
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("nis.agents.isaac.bridge")

# Try to import ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import Header, Bool, Float64MultiArray
    from nav_msgs.msg import Odometry
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.info("ROS 2 not available, using simulation mode")


class RobotType(Enum):
    """Supported robot types"""
    MANIPULATOR = "manipulator"
    DRONE = "drone"
    HUMANOID = "humanoid"
    MOBILE = "mobile"
    CUSTOM = "custom"


@dataclass
class IsaacConfig:
    """Isaac Bridge configuration"""
    node_name: str = "nis_isaac_bridge"
    namespace: str = "/nis"
    
    # Topic names
    trajectory_topic: str = "/isaac/joint_trajectory"
    joint_state_topic: str = "/isaac/joint_states"
    pose_topic: str = "/isaac/current_pose"
    cmd_vel_topic: str = "/isaac/cmd_vel"
    emergency_stop_topic: str = "/isaac/emergency_stop"
    
    # Timeouts
    trajectory_timeout: float = 30.0
    state_timeout: float = 5.0
    
    # Physics validation
    enable_physics_validation: bool = True
    max_velocity: float = 2.0  # m/s
    max_acceleration: float = 5.0  # m/s^2
    max_jerk: float = 50.0  # m/s^3


@dataclass
class RobotState:
    """Current robot state"""
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    joint_efforts: List[float] = field(default_factory=list)
    pose: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    is_moving: bool = False
    emergency_stopped: bool = False


@dataclass
class TrajectoryCommand:
    """Trajectory command for execution"""
    waypoints: List[List[float]]
    joint_names: List[str] = field(default_factory=list)
    duration: float = 5.0
    robot_type: str = "manipulator"
    validate_physics: bool = True


class IsaacBridgeAgent:
    """
    ROS 2 Bridge Agent for NVIDIA Isaac integration
    
    Connects NIS Protocol's cognitive layer with Isaac's physical layer.
    
    Usage:
        bridge = IsaacBridgeAgent()
        await bridge.initialize()
        
        result = await bridge.execute_trajectory(
            waypoints=[[0, 0, 0], [1, 1, 1]],
            robot_type="drone",
            duration=5.0
        )
    """
    
    def __init__(self, config: IsaacConfig = None):
        self.config = config or IsaacConfig()
        self.initialized = False
        self.simulation_mode = not ROS2_AVAILABLE
        
        # ROS 2 node
        self._node: Optional[Any] = None
        self._executor = None
        self._spin_task = None
        
        # Publishers
        self._trajectory_pub = None
        self._cmd_vel_pub = None
        self._emergency_pub = None
        
        # Subscribers
        self._joint_state_sub = None
        self._pose_sub = None
        
        # State
        self._current_state = RobotState()
        self._state_callbacks: List[Callable] = []
        
        # Robotics agent for physics validation
        self._robotics_agent = None
        
        # Statistics
        self.stats = {
            "trajectories_executed": 0,
            "trajectories_failed": 0,
            "physics_violations": 0,
            "emergency_stops": 0,
            "total_distance": 0.0
        }
        
        logger.info(f"Isaac Bridge Agent created (simulation_mode={self.simulation_mode})")
    
    async def initialize(self) -> bool:
        """Initialize the Isaac bridge"""
        if self.initialized:
            return True
        
        try:
            # Initialize robotics agent for physics validation
            if self.config.enable_physics_validation:
                from ..robotics.unified_robotics_agent import UnifiedRoboticsAgent
                self._robotics_agent = UnifiedRoboticsAgent(
                    enable_physics_validation=True
                )
                logger.info("Physics validation enabled")
            
            if not self.simulation_mode:
                # Initialize ROS 2
                if not rclpy.ok():
                    rclpy.init()
                
                self._node = rclpy.create_node(
                    self.config.node_name,
                    namespace=self.config.namespace
                )
                
                # Create QoS profile
                qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10
                )
                
                # Create publishers
                self._trajectory_pub = self._node.create_publisher(
                    JointTrajectory,
                    self.config.trajectory_topic,
                    qos
                )
                
                self._cmd_vel_pub = self._node.create_publisher(
                    Twist,
                    self.config.cmd_vel_topic,
                    qos
                )
                
                self._emergency_pub = self._node.create_publisher(
                    Bool,
                    self.config.emergency_stop_topic,
                    qos
                )
                
                # Create subscribers
                self._joint_state_sub = self._node.create_subscription(
                    JointState,
                    self.config.joint_state_topic,
                    self._joint_state_callback,
                    qos
                )
                
                self._pose_sub = self._node.create_subscription(
                    PoseStamped,
                    self.config.pose_topic,
                    self._pose_callback,
                    qos
                )
                
                # Start spinning in background
                self._spin_task = asyncio.create_task(self._spin_ros())
                
                logger.info("ROS 2 node initialized")
            
            self.initialized = True
            logger.info("Isaac Bridge Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Isaac Bridge: {e}")
            self.simulation_mode = True
            self.initialized = True
            return True  # Continue in simulation mode
    
    async def _spin_ros(self):
        """Spin ROS 2 node in background"""
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
            await asyncio.sleep(0.01)
    
    def _joint_state_callback(self, msg):
        """Handle joint state updates"""
        self._current_state.joint_positions = list(msg.position)
        self._current_state.joint_velocities = list(msg.velocity)
        self._current_state.joint_efforts = list(msg.effort)
        self._current_state.timestamp = time.time()
        self._current_state.is_moving = any(abs(v) > 0.01 for v in msg.velocity)
        
        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                callback(self._current_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
    
    def _pose_callback(self, msg):
        """Handle pose updates"""
        self._current_state.pose = {
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z,
            "qx": msg.pose.orientation.x,
            "qy": msg.pose.orientation.y,
            "qz": msg.pose.orientation.z,
            "qw": msg.pose.orientation.w
        }
        self._current_state.timestamp = time.time()
    
    async def execute_trajectory(
        self,
        waypoints: List[List[float]],
        robot_type: str = "manipulator",
        duration: float = 5.0,
        joint_names: List[str] = None,
        validate_physics: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a trajectory on the robot
        
        Args:
            waypoints: List of waypoint positions
            robot_type: Type of robot (manipulator, drone, etc.)
            duration: Total trajectory duration in seconds
            joint_names: Names of joints (for manipulators)
            validate_physics: Whether to validate physics constraints
        
        Returns:
            Execution result with success status and details
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # 1. Plan trajectory with NIS Robotics Agent
            if self._robotics_agent and validate_physics:
                plan_result = self._robotics_agent.plan_trajectory(
                    robot_id="isaac_robot",
                    waypoints=waypoints,
                    robot_type=robot_type,
                    duration=duration
                )
                
                if not plan_result.get("success", False):
                    return {
                        "success": False,
                        "error": "Trajectory planning failed",
                        "details": plan_result
                    }
                
                if not plan_result.get("physics_valid", False):
                    self.stats["physics_violations"] += 1
                    return {
                        "success": False,
                        "error": "Physics validation failed",
                        "physics_violations": plan_result.get("physics_violations", [])
                    }
                
                # Use optimized trajectory from planner
                raw_trajectory = plan_result.get("trajectory", waypoints)
                # Convert TrajectoryPoint objects to position lists
                if raw_trajectory and hasattr(raw_trajectory[0], 'position'):
                    trajectory_points = [
                        list(tp.position) if hasattr(tp.position, 'tolist') else list(tp.position)
                        for tp in raw_trajectory
                    ]
                else:
                    trajectory_points = raw_trajectory
            else:
                trajectory_points = waypoints
            
            # 2. Execute trajectory
            if self.simulation_mode:
                # Simulate execution
                result = await self._simulate_trajectory(trajectory_points, duration)
            else:
                # Publish to ROS 2
                result = await self._publish_trajectory(
                    trajectory_points,
                    duration,
                    joint_names or self._default_joint_names(robot_type)
                )
            
            # 3. Update statistics
            execution_time = time.time() - start_time
            
            if result["success"]:
                self.stats["trajectories_executed"] += 1
                self.stats["total_distance"] += self._calculate_distance(waypoints)
            else:
                self.stats["trajectories_failed"] += 1
            
            result["execution_time_ms"] = execution_time * 1000
            result["physics_validated"] = validate_physics
            
            return result
            
        except Exception as e:
            logger.error(f"Trajectory execution error: {e}")
            self.stats["trajectories_failed"] += 1
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _simulate_trajectory(
        self,
        waypoints: List[List[float]],
        duration: float
    ) -> Dict[str, Any]:
        """Simulate trajectory execution"""
        logger.info(f"Simulating trajectory with {len(waypoints)} waypoints over {duration}s")
        
        # Simulate execution time (scaled down)
        await asyncio.sleep(min(duration * 0.1, 1.0))
        
        # Update simulated state
        if waypoints:
            final_pos = waypoints[-1]
            self._current_state.pose = {
                "x": final_pos[0] if len(final_pos) > 0 else 0,
                "y": final_pos[1] if len(final_pos) > 1 else 0,
                "z": final_pos[2] if len(final_pos) > 2 else 0
            }
            self._current_state.timestamp = time.time()
        
        return {
            "success": True,
            "mode": "simulation",
            "waypoints_executed": len(waypoints),
            "final_position": waypoints[-1] if waypoints else []
        }
    
    async def _publish_trajectory(
        self,
        waypoints: List[List[float]],
        duration: float,
        joint_names: List[str]
    ) -> Dict[str, Any]:
        """Publish trajectory to ROS 2"""
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.joint_names = joint_names
        
        # Create trajectory points
        time_step = duration / max(len(waypoints) - 1, 1)
        
        for i, waypoint in enumerate(waypoints):
            point = JointTrajectoryPoint()
            point.positions = waypoint
            point.time_from_start.sec = int(i * time_step)
            point.time_from_start.nanosec = int((i * time_step % 1) * 1e9)
            msg.points.append(point)
        
        # Publish
        self._trajectory_pub.publish(msg)
        logger.info(f"Published trajectory with {len(waypoints)} points")
        
        # Wait for execution (with timeout)
        start = time.time()
        while time.time() - start < duration + 5.0:
            if not self._current_state.is_moving:
                break
            await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "mode": "ros2",
            "waypoints_executed": len(waypoints)
        }
    
    def _default_joint_names(self, robot_type: str) -> List[str]:
        """Get default joint names for robot type"""
        if robot_type == "manipulator":
            return ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        elif robot_type == "drone":
            return ["x", "y", "z", "roll", "pitch", "yaw"]
        elif robot_type == "humanoid":
            return ["hip", "knee", "ankle", "shoulder", "elbow", "wrist"]
        else:
            return ["joint_1", "joint_2", "joint_3"]
    
    def _calculate_distance(self, waypoints: List[List[float]]) -> float:
        """Calculate total trajectory distance"""
        if len(waypoints) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(waypoints)):
            dist = sum((a - b) ** 2 for a, b in zip(waypoints[i], waypoints[i-1])) ** 0.5
            total += dist
        return total
    
    async def send_velocity_command(
        self,
        linear: List[float],
        angular: List[float]
    ) -> Dict[str, Any]:
        """Send velocity command to mobile robot"""
        if not self.initialized:
            await self.initialize()
        
        if self.simulation_mode:
            return {"success": True, "mode": "simulation"}
        
        msg = Twist()
        msg.linear.x = linear[0] if len(linear) > 0 else 0.0
        msg.linear.y = linear[1] if len(linear) > 1 else 0.0
        msg.linear.z = linear[2] if len(linear) > 2 else 0.0
        msg.angular.x = angular[0] if len(angular) > 0 else 0.0
        msg.angular.y = angular[1] if len(angular) > 1 else 0.0
        msg.angular.z = angular[2] if len(angular) > 2 else 0.0
        
        self._cmd_vel_pub.publish(msg)
        
        return {"success": True, "mode": "ros2"}
    
    async def emergency_stop(self) -> Dict[str, Any]:
        """Trigger emergency stop"""
        logger.warning("EMERGENCY STOP triggered!")
        self.stats["emergency_stops"] += 1
        self._current_state.emergency_stopped = True
        
        if not self.simulation_mode and self._emergency_pub:
            msg = Bool()
            msg.data = True
            self._emergency_pub.publish(msg)
        
        return {
            "success": True,
            "emergency_stopped": True,
            "timestamp": time.time()
        }
    
    async def release_emergency_stop(self) -> Dict[str, Any]:
        """Release emergency stop"""
        logger.info("Emergency stop released")
        self._current_state.emergency_stopped = False
        
        if not self.simulation_mode and self._emergency_pub:
            msg = Bool()
            msg.data = False
            self._emergency_pub.publish(msg)
        
        return {
            "success": True,
            "emergency_stopped": False,
            "timestamp": time.time()
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return {
            "joint_positions": self._current_state.joint_positions,
            "joint_velocities": self._current_state.joint_velocities,
            "pose": self._current_state.pose,
            "is_moving": self._current_state.is_moving,
            "emergency_stopped": self._current_state.emergency_stopped,
            "timestamp": self._current_state.timestamp,
            "simulation_mode": self.simulation_mode
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            **self.stats,
            "initialized": self.initialized,
            "simulation_mode": self.simulation_mode,
            "ros2_available": ROS2_AVAILABLE
        }
    
    def register_state_callback(self, callback: Callable):
        """Register callback for state updates"""
        self._state_callbacks.append(callback)
    
    async def shutdown(self):
        """Shutdown the bridge"""
        logger.info("Shutting down Isaac Bridge...")
        
        if self._spin_task:
            self._spin_task.cancel()
        
        if self._node:
            self._node.destroy_node()
        
        if ROS2_AVAILABLE and rclpy.ok():
            rclpy.shutdown()
        
        self.initialized = False
        logger.info("Isaac Bridge shutdown complete")


# Singleton instance
_isaac_bridge: Optional[IsaacBridgeAgent] = None


def get_isaac_bridge() -> IsaacBridgeAgent:
    """Get the Isaac bridge singleton"""
    global _isaac_bridge
    if _isaac_bridge is None:
        _isaac_bridge = IsaacBridgeAgent()
    return _isaac_bridge
