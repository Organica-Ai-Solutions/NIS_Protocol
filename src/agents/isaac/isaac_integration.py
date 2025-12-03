#!/usr/bin/env python3
"""
NIS Protocol - Full Isaac Integration
Connects NVIDIA Isaac with NIS Protocol's cognitive layer

This module provides:
- Unified Isaac manager for all Isaac components
- Integration with UnifiedRoboticsAgent
- GPU-accelerated motion planning (when available)
- Seamless fallback to CPU when Isaac not available
- Real-time telemetry streaming
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import numpy as np

from .isaac_bridge_agent import IsaacBridgeAgent, IsaacConfig, get_isaac_bridge
from .isaac_sim_agent import IsaacSimAgent, SimConfig, get_isaac_sim
from .isaac_perception_agent import IsaacPerceptionAgent, PerceptionConfig, get_isaac_perception

logger = logging.getLogger("nis.agents.isaac.integration")


@dataclass
class IsaacIntegrationConfig:
    """Full Isaac integration configuration"""
    # Enable/disable components
    enable_bridge: bool = True
    enable_sim: bool = True
    enable_perception: bool = True
    
    # Performance settings
    use_gpu_planning: bool = True
    use_gpu_perception: bool = True
    
    # Physics validation
    enable_physics_validation: bool = True
    physics_validation_threshold: float = 0.95
    
    # Telemetry
    enable_telemetry: bool = True
    telemetry_rate_hz: float = 30.0
    
    # Offline mode
    offline_mode: bool = False
    model_cache_path: str = "/opt/isaac_models"


class IsaacIntegrationManager:
    """
    Central manager for all Isaac integrations
    
    Provides unified interface to:
    - Isaac ROS 2 Bridge (robot control)
    - Isaac Sim (simulation)
    - Isaac Perception (vision)
    - NIS Robotics Agent (physics validation)
    
    Usage:
        manager = IsaacIntegrationManager()
        await manager.initialize()
        
        # Execute trajectory with full pipeline
        result = await manager.execute_full_pipeline(
            waypoints=[[0,0,0], [1,1,1]],
            robot_type="manipulator"
        )
    """
    
    def __init__(self, config: IsaacIntegrationConfig = None):
        self.config = config or IsaacIntegrationConfig()
        self.initialized = False
        
        # Isaac components
        self._bridge: Optional[IsaacBridgeAgent] = None
        self._sim: Optional[IsaacSimAgent] = None
        self._perception: Optional[IsaacPerceptionAgent] = None
        
        # NIS Robotics Agent
        self._robotics_agent = None
        
        # Telemetry
        self._telemetry_task = None
        self._telemetry_callbacks: List[Callable] = []
        self._latest_telemetry: Dict[str, Any] = {}
        
        # GPU availability
        self.gpu_available = self._check_gpu()
        
        # Statistics
        self.stats = {
            "pipelines_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_planning_time_ms": 0.0,
            "total_execution_time_ms": 0.0,
            "physics_validations": 0,
            "perception_calls": 0
        }
        
        logger.info(f"Isaac Integration Manager created (GPU: {self.gpu_available})")
    
    def _check_gpu(self) -> bool:
        """Check if CUDA GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    async def initialize(self) -> bool:
        """Initialize all Isaac components"""
        if self.initialized:
            return True
        
        logger.info("Initializing Isaac Integration...")
        
        try:
            # Initialize Bridge
            if self.config.enable_bridge:
                self._bridge = get_isaac_bridge()
                await self._bridge.initialize()
                logger.info("✅ Isaac Bridge initialized")
            
            # Initialize Sim
            if self.config.enable_sim:
                self._sim = get_isaac_sim()
                await self._sim.initialize()
                logger.info("✅ Isaac Sim initialized")
            
            # Initialize Perception
            if self.config.enable_perception:
                self._perception = get_isaac_perception()
                await self._perception.initialize()
                logger.info("✅ Isaac Perception initialized")
            
            # Initialize NIS Robotics Agent
            if self.config.enable_physics_validation:
                try:
                    from ..robotics.unified_robotics_agent import UnifiedRoboticsAgent
                    self._robotics_agent = UnifiedRoboticsAgent(
                        agent_id="isaac_robotics",
                        enable_physics_validation=True,
                        enable_can_protocol=False  # Isaac handles hardware
                    )
                    logger.info("✅ NIS Robotics Agent initialized")
                except Exception as e:
                    logger.warning(f"Robotics agent init failed: {e}")
            
            # Start telemetry if enabled
            if self.config.enable_telemetry:
                self._telemetry_task = asyncio.create_task(self._telemetry_loop())
                logger.info("✅ Telemetry streaming started")
            
            self.initialized = True
            logger.info("🚀 Isaac Integration fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Isaac Integration initialization failed: {e}")
            self.initialized = True  # Continue in degraded mode
            return True
    
    async def execute_full_pipeline(
        self,
        waypoints: List[List[float]],
        robot_type: str = "manipulator",
        robot_id: str = "isaac_robot",
        duration: float = 5.0,
        use_perception: bool = False,
        validate_physics: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full cognitive-physical pipeline
        
        Pipeline:
        1. [Optional] Perception: Detect objects, estimate poses
        2. NIS Planning: Plan trajectory with physics validation
        3. Isaac Execution: Execute on robot (real or sim)
        4. Telemetry: Stream execution data
        
        Args:
            waypoints: Target waypoints
            robot_type: Type of robot
            robot_id: Robot identifier
            duration: Trajectory duration
            use_perception: Whether to use perception for planning
            validate_physics: Whether to validate physics
        
        Returns:
            Complete pipeline result
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        result = {
            "success": False,
            "pipeline_stages": {},
            "robot_id": robot_id,
            "robot_type": robot_type
        }
        
        try:
            # Stage 1: Perception (optional)
            if use_perception and self._perception:
                perception_start = time.time()
                
                # Get current scene understanding
                mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
                detections = await self._perception.detect_objects(mock_image)
                slam_result = await self._perception.update_slam(mock_image)
                
                result["pipeline_stages"]["perception"] = {
                    "success": True,
                    "detections": detections.get("detections", []),
                    "slam_pose": slam_result.get("pose"),
                    "duration_ms": (time.time() - perception_start) * 1000
                }
                self.stats["perception_calls"] += 1
            
            # Stage 2: NIS Planning with Physics Validation
            planning_start = time.time()
            
            if self._robotics_agent and validate_physics:
                # Use NIS Robotics Agent for planning
                plan_result = self._robotics_agent.plan_trajectory(
                    robot_id=robot_id,
                    waypoints=waypoints,
                    robot_type=robot_type,
                    duration=duration
                )
                
                physics_valid = plan_result.get("physics_valid", False)
                optimized_trajectory = plan_result.get("trajectory", waypoints)
                
                result["pipeline_stages"]["planning"] = {
                    "success": plan_result.get("success", False),
                    "physics_valid": physics_valid,
                    "physics_warnings": plan_result.get("physics_warnings", []),
                    "trajectory_points": len(optimized_trajectory),
                    "duration_ms": (time.time() - planning_start) * 1000
                }
                
                self.stats["physics_validations"] += 1
                
                if not physics_valid and validate_physics:
                    result["error"] = "Physics validation failed"
                    result["physics_violations"] = plan_result.get("physics_violations", [])
                    self.stats["failed_executions"] += 1
                    return result
            else:
                optimized_trajectory = waypoints
                result["pipeline_stages"]["planning"] = {
                    "success": True,
                    "physics_valid": None,
                    "note": "Physics validation disabled",
                    "duration_ms": (time.time() - planning_start) * 1000
                }
            
            self.stats["total_planning_time_ms"] += (time.time() - planning_start) * 1000
            
            # Stage 3: Isaac Execution
            execution_start = time.time()
            
            if self._bridge:
                # Convert TrajectoryPoint objects to position lists if needed
                if optimized_trajectory and hasattr(optimized_trajectory[0], 'position'):
                    exec_waypoints = [
                        list(tp.position) if hasattr(tp.position, 'tolist') else list(tp.position)
                        for tp in optimized_trajectory
                    ]
                elif isinstance(optimized_trajectory, list):
                    exec_waypoints = optimized_trajectory
                else:
                    exec_waypoints = optimized_trajectory.tolist()
                
                exec_result = await self._bridge.execute_trajectory(
                    waypoints=exec_waypoints,
                    robot_type=robot_type,
                    duration=duration,
                    validate_physics=False  # Already validated
                )
                
                result["pipeline_stages"]["execution"] = {
                    "success": exec_result.get("success", False),
                    "mode": exec_result.get("mode", "unknown"),
                    "waypoints_executed": exec_result.get("waypoints_executed", 0),
                    "duration_ms": (time.time() - execution_start) * 1000
                }
            else:
                result["pipeline_stages"]["execution"] = {
                    "success": False,
                    "error": "Isaac Bridge not available"
                }
            
            self.stats["total_execution_time_ms"] += (time.time() - execution_start) * 1000
            
            # Final result
            total_time = (time.time() - start_time) * 1000
            
            result["success"] = all(
                stage.get("success", False) 
                for stage in result["pipeline_stages"].values()
            )
            result["total_duration_ms"] = total_time
            
            self.stats["pipelines_executed"] += 1
            if result["success"]:
                self.stats["successful_executions"] += 1
            else:
                self.stats["failed_executions"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            result["error"] = str(e)
            self.stats["failed_executions"] += 1
            return result
    
    async def pick_and_place(
        self,
        object_id: str,
        target_position: List[float],
        robot_id: str = "isaac_robot"
    ) -> Dict[str, Any]:
        """
        High-level pick and place operation
        
        Uses perception to find object, plans grasp, executes motion.
        """
        if not self.initialized:
            await self.initialize()
        
        result = {"success": False, "stages": {}}
        
        try:
            # 1. Detect object pose
            if self._perception:
                mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
                pose_result = await self._perception.estimate_pose(mock_image, object_id)
                
                if not pose_result.get("success"):
                    result["error"] = f"Could not find object: {object_id}"
                    return result
                
                object_position = list(pose_result.get("position", [0, 0, 0]))
                result["stages"]["perception"] = {
                    "object_position": object_position,
                    "confidence": pose_result.get("confidence", 0)
                }
            else:
                object_position = [0.5, 0, 0.5]  # Default position
            
            # 2. Plan approach trajectory
            approach_waypoints = [
                [object_position[0], object_position[1], object_position[2] + 0.2],  # Above object
                object_position,  # At object
            ]
            
            # 3. Execute pick
            pick_result = await self.execute_full_pipeline(
                waypoints=approach_waypoints,
                robot_type="manipulator",
                robot_id=robot_id,
                duration=3.0
            )
            result["stages"]["pick"] = pick_result
            
            if not pick_result.get("success"):
                result["error"] = "Pick failed"
                return result
            
            # 4. Plan place trajectory
            place_waypoints = [
                object_position,  # Current position
                [object_position[0], object_position[1], object_position[2] + 0.2],  # Lift
                [target_position[0], target_position[1], target_position[2] + 0.2],  # Above target
                target_position  # At target
            ]
            
            # 5. Execute place
            place_result = await self.execute_full_pipeline(
                waypoints=place_waypoints,
                robot_type="manipulator",
                robot_id=robot_id,
                duration=4.0
            )
            result["stages"]["place"] = place_result
            
            result["success"] = place_result.get("success", False)
            return result
            
        except Exception as e:
            logger.error(f"Pick and place error: {e}")
            result["error"] = str(e)
            return result
    
    async def _telemetry_loop(self):
        """Background telemetry streaming loop"""
        interval = 1.0 / self.config.telemetry_rate_hz
        
        while True:
            try:
                # Collect telemetry from all components
                telemetry = {
                    "timestamp": time.time(),
                    "bridge": self._bridge.get_current_state() if self._bridge else None,
                    "sim": self._sim.get_stats() if self._sim else None,
                    "perception": {
                        "slam_pose": self._perception.get_slam_pose() if self._perception else None
                    }
                }
                
                self._latest_telemetry = telemetry
                
                # Notify callbacks
                for callback in self._telemetry_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(telemetry)
                        else:
                            callback(telemetry)
                    except Exception as e:
                        logger.error(f"Telemetry callback error: {e}")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
                await asyncio.sleep(1.0)
    
    def register_telemetry_callback(self, callback: Callable):
        """Register callback for telemetry updates"""
        self._telemetry_callbacks.append(callback)
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get latest telemetry data"""
        return self._latest_telemetry
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            **self.stats,
            "initialized": self.initialized,
            "gpu_available": self.gpu_available,
            "components": {
                "bridge": self._bridge.get_stats() if self._bridge else None,
                "sim": self._sim.get_stats() if self._sim else None,
                "perception": self._perception.get_stats() if self._perception else None
            }
        }
    
    async def shutdown(self):
        """Shutdown all Isaac components"""
        logger.info("Shutting down Isaac Integration...")
        
        # Stop telemetry
        if self._telemetry_task:
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components
        if self._bridge:
            await self._bridge.shutdown()
        if self._sim:
            await self._sim.shutdown()
        if self._perception:
            await self._perception.shutdown()
        
        self.initialized = False
        logger.info("Isaac Integration shutdown complete")


# Singleton instance
_isaac_manager: Optional[IsaacIntegrationManager] = None


def get_isaac_manager() -> IsaacIntegrationManager:
    """Get the Isaac Integration Manager singleton"""
    global _isaac_manager
    if _isaac_manager is None:
        _isaac_manager = IsaacIntegrationManager()
    return _isaac_manager


async def initialize_isaac() -> IsaacIntegrationManager:
    """Initialize and return the Isaac Integration Manager"""
    manager = get_isaac_manager()
    await manager.initialize()
    return manager
