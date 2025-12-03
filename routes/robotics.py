"""
NIS Protocol v4.0 - Robotics Routes

This module contains all robotics-related endpoints:
- Forward Kinematics (FK)
- Inverse Kinematics (IK)
- Trajectory Planning
- Capabilities
- WebSocket Control
- Telemetry Streaming

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over
- main.py routes remain active until migration is complete

Usage:
    from routes.robotics import router as robotics_router
    app.include_router(robotics_router, tags=["Robotics"])
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger("nis.routes.robotics")

# Create router
router = APIRouter(prefix="/robotics", tags=["Robotics"])


def _convert_numpy_to_json(obj: Any) -> Any:
    """Convert numpy arrays to JSON-serializable format"""
    try:
        import numpy as np
    except ImportError:
        np = None
    
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    
    if isinstance(obj, dict):
        return {key: _convert_numpy_to_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_to_json(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_numpy_to_json(item) for item in obj)
    
    return obj


@router.post("/forward_kinematics")
async def robotics_forward_kinematics(request: dict):
    """
    🤖 Compute Forward Kinematics (Real Denavit-Hartenberg Transforms)
    
    Calculates end-effector pose from joint angles using actual DH transformations.
    NO MOCKS - Real 4x4 homogeneous matrix computations.
    
    Args:
        robot_id: Unique identifier for the robot
        robot_type: "drone", "manipulator", "humanoid", or "ground_vehicle"
        joint_angles: Array of joint angles (or motor speeds for drones)
        
    Returns:
        Real computed end-effector pose with measured computation time
    """
    try:
        from src.agents.robotics import UnifiedRoboticsAgent, RobotType
        import numpy as np
        
        # Extract request parameters
        robot_id = request.get("robot_id", "robot_001")
        robot_type_str = request.get("robot_type", "manipulator")
        joint_angles = np.array(request.get("joint_angles", []))
        
        if len(joint_angles) == 0:
            raise HTTPException(status_code=400, detail="joint_angles required")
        
        # Map string to enum
        robot_type_map = {
            "drone": RobotType.DRONE,
            "manipulator": RobotType.MANIPULATOR,
            "humanoid": RobotType.HUMANOID,
            "ground_vehicle": RobotType.GROUND_VEHICLE
        }
        
        robot_type = robot_type_map.get(robot_type_str.lower())
        if not robot_type:
            raise HTTPException(status_code=400, detail=f"Invalid robot_type: {robot_type_str}")
        
        # Create agent and compute (REAL implementation)
        agent = UnifiedRoboticsAgent(agent_id="api_robotics_agent")
        result = agent.compute_forward_kinematics(robot_id, joint_angles, robot_type)
        
        # Convert all numpy arrays recursively
        result = _convert_numpy_to_json(result)
        
        logger.info(f"✅ FK computed: {robot_id} ({robot_type_str}) in {result.get('computation_time', 0)*1000:.2f}ms")
        
        response_data = {
            "status": "success",
            "result": result,
            "timestamp": time.time()
        }
        
        json_str = json.dumps(response_data)
        return Response(content=json_str, media_type="application/json")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Robotics FK error: {e}")
        raise HTTPException(status_code=500, detail=f"Forward kinematics failed: {str(e)}")


@router.post("/inverse_kinematics")
async def robotics_inverse_kinematics(request: dict):
    """
    🤖 Compute Inverse Kinematics (Real Scipy Numerical Optimization)
    
    Solves for joint angles to reach target pose using actual scipy.optimize.
    NO MOCKS - Real numerical solver with convergence tracking.
    
    Args:
        robot_id: Unique identifier for the robot
        robot_type: "manipulator", "humanoid", or "ground_vehicle"
        target_pose: Dictionary with 'position' [x, y, z] and optional 'orientation'
        initial_guess: Optional initial joint angles for optimization
        
    Returns:
        Real optimized joint angles with actual iteration count and error
    """
    try:
        from src.agents.robotics import UnifiedRoboticsAgent, RobotType
        import numpy as np
        
        # Extract request parameters
        robot_id = request.get("robot_id", "robot_001")
        robot_type_str = request.get("robot_type", "manipulator")
        target_pose = request.get("target_pose", {})
        initial_guess = request.get("initial_guess")
        
        if "position" not in target_pose:
            raise HTTPException(status_code=400, detail="target_pose.position required")
        
        # Convert to numpy
        target_pose["position"] = np.array(target_pose["position"])
        if "orientation" in target_pose:
            target_pose["orientation"] = np.array(target_pose["orientation"])
        
        if initial_guess is not None:
            initial_guess = np.array(initial_guess)
        
        # Map string to enum
        robot_type_map = {
            "manipulator": RobotType.MANIPULATOR,
            "humanoid": RobotType.HUMANOID,
            "ground_vehicle": RobotType.GROUND_VEHICLE
        }
        
        robot_type = robot_type_map.get(robot_type_str.lower())
        if not robot_type:
            raise HTTPException(status_code=400, detail=f"Invalid robot_type for IK: {robot_type_str}")
        
        # Create agent and compute (REAL scipy optimization)
        agent = UnifiedRoboticsAgent(agent_id="api_robotics_agent")
        result = agent.compute_inverse_kinematics(robot_id, target_pose, robot_type, initial_guess)

        status = "success" if result.get("success", False) else "error"
        message = result.get("error") if status == "error" else None

        # Convert all numpy arrays recursively
        result = _convert_numpy_to_json(result)

        logger.info(f"✅ IK computed: {robot_id} converged in {result.get('iterations', 0)} iterations" if status == "success" else f"⚠️ IK failed for {robot_id}: {message}")

        response_data = {
            "status": status,
            "result": result,
            "timestamp": time.time()
        }

        if message:
            response_data["message"] = message
        
        json_str = json.dumps(response_data)
        return Response(content=json_str, media_type="application/json")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Robotics IK error: {e}")
        raise HTTPException(status_code=500, detail=f"Inverse kinematics failed: {str(e)}")


@router.post("/plan_trajectory")
async def robotics_plan_trajectory(request: dict):
    """
    🤖 Plan Physics-Validated Trajectory (Real Minimum Jerk Polynomial)
    
    Generates smooth trajectory with real physics validation.
    NO MOCKS - Real 5th-order polynomial with actual constraint checking.
    
    Args:
        robot_id: Unique identifier for the robot
        robot_type: "drone", "manipulator", "humanoid", or "ground_vehicle"
        waypoints: List of 3D positions [[x1,y1,z1], [x2,y2,z2], ...]
        duration: Total trajectory duration in seconds
        num_points: Number of trajectory points to generate (default: 50)
        
    Returns:
        Real trajectory with measured velocities/accelerations and physics validation
    """
    try:
        from src.agents.robotics import UnifiedRoboticsAgent, RobotType
        import numpy as np
        
        # Extract request parameters
        robot_id = request.get("robot_id", "robot_001")
        robot_type_str = request.get("robot_type", "drone")
        waypoints_list = request.get("waypoints", [])
        duration = request.get("duration", 5.0)
        num_points = request.get("num_points", 50)
        
        if len(waypoints_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 waypoints required")
        
        # Convert to numpy arrays (handle both dict and list formats)
        waypoints = []
        for wp in waypoints_list:
            if isinstance(wp, dict):
                pos = wp.get("position", wp.get("pos", list(wp.values())[0] if wp else [0,0,0]))
                waypoints.append(np.array(pos))
            else:
                waypoints.append(np.array(wp))
        
        # Map string to enum
        robot_type_map = {
            "drone": RobotType.DRONE,
            "manipulator": RobotType.MANIPULATOR,
            "humanoid": RobotType.HUMANOID,
            "ground_vehicle": RobotType.GROUND_VEHICLE
        }
        
        robot_type = robot_type_map.get(robot_type_str.lower())
        if not robot_type:
            raise HTTPException(status_code=400, detail=f"Invalid robot_type: {robot_type_str}")
        
        # Create agent and compute (REAL trajectory planning)
        agent = UnifiedRoboticsAgent(agent_id="api_robotics_agent", enable_physics_validation=True)
        result = agent.plan_trajectory(robot_id, waypoints, robot_type, duration, num_points)

        status = "success" if result.get("success", False) else "error"
        message = result.get("error") if status == "error" else None

        # Convert trajectory points to serializable format
        if result.get("trajectory"):
            trajectory_list = []
            for point in result["trajectory"]:
                traj_point = {
                    "time": float(getattr(point, 'time', 0.0)),
                    "position": point.position.tolist() if hasattr(point.position, 'tolist') else list(point.position),
                    "velocity": point.velocity.tolist() if hasattr(point.velocity, 'tolist') else list(point.velocity),
                    "acceleration": point.acceleration.tolist() if hasattr(point.acceleration, 'tolist') else list(point.acceleration)
                }
                if hasattr(point, 'orientation') and point.orientation is not None:
                    traj_point["orientation"] = point.orientation.tolist()
                if hasattr(point, 'angular_velocity') and point.angular_velocity is not None:
                    traj_point["angular_velocity"] = point.angular_velocity.tolist()
                trajectory_list.append(traj_point)
            result["trajectory"] = trajectory_list

        # Convert remaining numpy arrays
        result = _convert_numpy_to_json(result)

        logger.info(
            f"✅ Trajectory planned: {robot_id} ({result.get('num_points', 0)} points, physics_valid={result.get('physics_valid')})"
            if status == "success"
            else f"⚠️ Trajectory planning failed for {robot_id}: {message}"
        )

        response_data = {
            "status": status,
            "result": result,
            "timestamp": time.time()
        }

        if message:
            response_data["message"] = message
        
        json_str = json.dumps(response_data)
        return Response(content=json_str, media_type="application/json")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Robotics trajectory planning error: {e}")
        raise HTTPException(status_code=500, detail=f"Trajectory planning failed: {str(e)}")


@router.get("/capabilities")
async def robotics_capabilities():
    """
    🤖 Get Robotics Agent Capabilities (Real Stats Only)
    
    Returns actual agent capabilities and measured performance statistics.
    NO HARDCODED VALUES - All metrics computed from real agent state.
    
    Returns:
        Real-time agent statistics, supported platforms, and capabilities
    """
    try:
        from src.agents.robotics import UnifiedRoboticsAgent, RobotType
        
        # Create agent to get real stats
        agent = UnifiedRoboticsAgent(agent_id="api_robotics_agent")
        stats = agent.get_stats()
        
        capabilities = {
            "agent_info": {
                "agent_id": agent.agent_id,
                "description": agent.description,
                "layer": agent.layer.value,
                "physics_validation_enabled": agent.enable_physics_validation
            },
            "supported_robot_types": [
                {
                    "type": "drone",
                    "description": "Quadcopter/multirotor UAVs",
                    "capabilities": ["forward_kinematics", "trajectory_planning"],
                    "platforms": ["MAVLink", "DJI SDK", "PX4"]
                },
                {
                    "type": "manipulator",
                    "description": "Robotic arms/manipulators",
                    "capabilities": ["forward_kinematics", "inverse_kinematics", "trajectory_planning"],
                    "platforms": ["ROS", "Universal Robots", "Custom"]
                },
                {
                    "type": "humanoid",
                    "description": "Humanoid robots/androids",
                    "capabilities": ["forward_kinematics", "inverse_kinematics", "trajectory_planning"],
                    "platforms": ["ROS", "Custom frameworks"]
                },
                {
                    "type": "ground_vehicle",
                    "description": "Ground-based mobile robots",
                    "capabilities": ["trajectory_planning"],
                    "platforms": ["ROS Navigation", "Custom"]
                }
            ],
            "mathematical_methods": {
                "forward_kinematics": "Denavit-Hartenberg 4x4 transforms",
                "inverse_kinematics": "scipy.optimize numerical solver",
                "trajectory_planning": "Minimum jerk (5th-order polynomial)",
                "physics_validation": "PINN-based constraint checking"
            },
            "real_time_stats": stats,
            "api_endpoints": [
                "POST /robotics/forward_kinematics",
                "POST /robotics/inverse_kinematics",
                "POST /robotics/plan_trajectory",
                "GET /robotics/capabilities",
                "GET /robotics/telemetry/{robot_id}",
                "WS /ws/robotics/control/{robot_id}"
            ]
        }
        
        logger.info(f"✅ Robotics capabilities retrieved: {stats['total_commands']} commands processed")
        
        return {
            "status": "success",
            "capabilities": capabilities,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Robotics capabilities error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")


@router.get("/telemetry/{robot_id}")
async def robotics_telemetry_stream(robot_id: str, update_rate: int = 50):
    """
    📊 Real-time Telemetry Monitoring (Server-Sent Events)
    
    One-way streaming from server to client for monitoring robot state.
    
    Args:
        robot_id: Robot identifier to monitor
        update_rate: Updates per second (default: 50Hz, max: 1000Hz)
    """
    # Limit update rate to prevent system overload
    update_rate = min(update_rate, 1000)
    sleep_time = 1.0 / update_rate
    
    logger.info(f"📊 Starting telemetry stream: robot={robot_id}, rate={update_rate}Hz")
    
    from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent
    telemetry_agent = UnifiedRoboticsAgent(agent_id=f"telemetry_{robot_id}")
    
    async def telemetry_generator():
        """Generate telemetry events"""
        frame_count = 0
        
        try:
            while True:
                frame_count += 1
                stats = telemetry_agent.get_stats()
                
                telemetry = {
                    'robot_id': robot_id,
                    'frame': frame_count,
                    'timestamp': time.time(),
                    'stats': stats,
                    'update_rate': update_rate,
                    'status': 'active'
                }
                
                yield f"data: {json.dumps(telemetry)}\n\n"
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info(f"📊 Telemetry stream cancelled: robot={robot_id}, frames={frame_count}")
            yield f"data: {json.dumps({'status': 'disconnected', 'frame': frame_count})}\n\n"
    
    return StreamingResponse(
        telemetry_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ========================================================================
# CAN PROTOCOL ENDPOINTS
# ========================================================================

@router.get("/can/status")
async def get_can_status():
    """
    🔌 Get CAN Protocol Status
    
    Returns the current status of the CAN bus communication system
    including safety protocols and node status.
    """
    try:
        from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent
        
        agent = UnifiedRoboticsAgent(agent_id="can_status_agent", enable_can_protocol=True)
        can_stats = agent.get_can_statistics()
        
        return {
            "status": "success",
            "can_protocol": {
                "enabled": can_stats.get('enabled', True),
                "simulation_mode": can_stats.get('simulation_mode', True),
                "messages_sent": can_stats.get('messages_sent', 0),
                "messages_received": can_stats.get('messages_received', 0),
                "errors_detected": can_stats.get('errors_detected', 0),
                "emergency_stop_active": can_stats.get('emergency_stop_active', False),
                "uptime": can_stats.get('uptime', 0)
            },
            "safety_protocols": can_stats.get('safety_protocols', {}),
            "node_status": can_stats.get('node_status', {}),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"CAN status error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "can_protocol": {"enabled": False},
            "timestamp": time.time()
        }


@router.post("/can/emergency_stop")
async def trigger_emergency_stop(request: dict):
    """
    🚨 Trigger Emergency Stop via CAN
    
    Sends an emergency stop command to all connected devices.
    
    Args:
        activate: True to activate emergency stop, False to clear
    """
    try:
        from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent
        
        activate = request.get("activate", True)
        
        agent = UnifiedRoboticsAgent(agent_id="emergency_stop_agent", enable_can_protocol=True)
        await agent.initialize_can_protocol()
        
        success = await agent.send_emergency_stop(activate)
        
        action = "activated" if activate else "cleared"
        
        return {
            "status": "success" if success else "error",
            "message": f"Emergency stop {action}" if success else "Failed to send emergency stop",
            "emergency_stop_active": activate,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Emergency stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/can/motor_command")
async def send_motor_command(request: dict):
    """
    🔧 Send Motor Command via CAN
    
    Send a motor control command to a specific motor.
    
    Args:
        motor_id: Motor identifier (1-6)
        command: Command type (stop, enable, disable, position, velocity, torque)
        position: Target position (radians)
        velocity: Target velocity (rad/s)
        torque: Target torque (Nm)
    """
    try:
        from src.agents.robotics.unified_robotics_agent import UnifiedRoboticsAgent
        from src.protocols.robotics_can_definitions import MotorCommand
        
        motor_id = request.get("motor_id", 1)
        command_str = request.get("command", "stop").upper()
        position = request.get("position", 0.0)
        velocity = request.get("velocity", 0.0)
        torque = request.get("torque", 0.0)
        
        # Map command string to enum
        command_map = {
            "STOP": MotorCommand.STOP,
            "ENABLE": MotorCommand.ENABLE,
            "DISABLE": MotorCommand.DISABLE,
            "POSITION": MotorCommand.POSITION_CONTROL,
            "POSITION_CONTROL": MotorCommand.POSITION_CONTROL,
            "VELOCITY": MotorCommand.VELOCITY_CONTROL,
            "VELOCITY_CONTROL": MotorCommand.VELOCITY_CONTROL,
            "TORQUE": MotorCommand.TORQUE_CONTROL,
            "TORQUE_CONTROL": MotorCommand.TORQUE_CONTROL,
            "HOME": MotorCommand.HOME,
            "CALIBRATE": MotorCommand.CALIBRATE
        }
        
        command = command_map.get(command_str, MotorCommand.STOP)
        
        agent = UnifiedRoboticsAgent(agent_id="motor_command_agent", enable_can_protocol=True)
        await agent.initialize_can_protocol()
        
        success = await agent.send_motor_command(
            motor_id=motor_id,
            command=command,
            position=position,
            velocity=velocity,
            torque=torque
        )
        
        return {
            "status": "success" if success else "error",
            "motor_id": motor_id,
            "command": command_str,
            "position": position,
            "velocity": velocity,
            "torque": torque,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Motor command error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/can/safety")
async def get_safety_status():
    """
    🛡️ Get Safety Protocol Status
    
    Returns the current safety protocol status including
    error counters, violations, and safety limits.
    """
    try:
        from src.protocols.robotics_can_definitions import RoboticsSafetyProtocols
        
        safety = RoboticsSafetyProtocols()
        
        return {
            "status": "success",
            "safety_limits": {
                "max_velocity": safety.MAX_VELOCITY,
                "max_acceleration": safety.MAX_ACCELERATION,
                "max_angular_velocity": safety.MAX_ANGULAR_VELOCITY,
                "max_force": safety.MAX_FORCE,
                "max_torque": safety.MAX_TORQUE,
                "max_temperature": safety.MAX_TEMPERATURE,
                "temperature_warning": safety.TEMPERATURE_WARNING,
                "temperature_critical": safety.TEMPERATURE_CRITICAL,
                "error_threshold": safety.ERROR_THRESHOLD
            },
            "current_status": safety.get_safety_status(),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Safety status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# OBD-II (AUTOMOTIVE) ENDPOINTS
# ========================================================================

@router.get("/obd/status")
async def get_obd_status():
    """
    🚗 Get OBD-II Protocol Status
    
    Returns the current status of the OBD-II automotive interface
    including vehicle connection status, statistics, and integration info.
    """
    try:
        # Check infrastructure connection
        infra_connected = False
        try:
            from src.infrastructure.nis_infrastructure import get_nis_infrastructure
            infra = get_nis_infrastructure()
            infra_connected = infra.is_connected
        except:
            pass
        
        return {
            "status": "success",
            "obd_protocol": {
                "is_running": True,
                "simulation_mode": True,
                "vehicle_connected": False,
                "readings_count": 0,
                "errors_count": 0,
                "dtc_count": 0,
                "uptime": 0.0,
                "can_bus_enabled": True,
                "safety_monitoring": True
            },
            "vehicle_state": {
                "engine_rpm": 0.0,
                "vehicle_speed": 0.0,
                "coolant_temp": 0.0,
                "throttle_position": 0.0,
                "fuel_level": 0.0,
                "battery_voltage": 0.0,
                "engine_load": 0.0,
                "intake_temp": 0.0,
                "maf_rate": 0.0,
                "fuel_pressure": 0.0,
                "timing_advance": 0.0
            },
            "supported_pids": {
                "engine": ["ENGINE_RPM", "ENGINE_LOAD", "COOLANT_TEMP", "INTAKE_TEMP", "THROTTLE_POSITION", "TIMING_ADVANCE"],
                "fuel": ["FUEL_LEVEL", "FUEL_PRESSURE", "MAF_RATE", "FUEL_RATE"],
                "motion": ["VEHICLE_SPEED"],
                "electrical": ["BATTERY_VOLTAGE", "CONTROL_MODULE_VOLTAGE"],
                "diagnostics": ["STORED_DTCS", "PENDING_DTCS", "CLEAR_DTCS"]
            },
            "integration": {
                "kafka_streaming": infra_connected,
                "redis_caching": infra_connected,
                "can_protocol": True,
                "safety_protocols": True
            },
            "connection_guide": {
                "hardware": "ELM327 OBD-II adapter (USB/Bluetooth/WiFi)",
                "protocol": "CAN bus (ISO 15765-4)",
                "port": "/dev/ttyUSB0 or COM3",
                "baudrate": 500000
            },
            "note": "Simulation mode - connect OBD-II device for real vehicle data",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"OBD status error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "obd_protocol": {"is_running": False},
            "timestamp": time.time()
        }


@router.get("/obd/vehicle")
async def get_vehicle_data():
    """
    🚗 Get Current Vehicle Data
    
    Returns real-time vehicle telemetry from OBD-II interface.
    In simulation mode, returns default values.
    """
    try:
        return {
            "status": "success",
            "vehicle": {
                "engine": {
                    "rpm": 0.0,
                    "load": 0.0,
                    "coolant_temp": 0.0,
                    "intake_temp": 0.0,
                    "throttle_position": 0.0,
                    "timing_advance": 0.0
                },
                "motion": {
                    "speed_kmh": 0.0,
                    "speed_mph": 0.0
                },
                "fuel": {
                    "level_percent": 0.0,
                    "pressure_kpa": 0.0,
                    "rate_lph": 0.0,
                    "maf_rate_gs": 0.0
                },
                "electrical": {
                    "battery_voltage": 0.0
                },
                "diagnostics": {
                    "mil_on": False,
                    "dtc_count": 0
                },
                "vin": None
            },
            "is_connected": False,
            "simulation_mode": True,
            "note": "Connect OBD-II device for real vehicle data",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Vehicle data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/obd/dtcs")
async def get_diagnostic_codes():
    """
    🔧 Get Diagnostic Trouble Codes
    
    Returns all stored DTCs from the vehicle's ECU.
    In simulation mode, returns empty list.
    """
    try:
        # In simulation mode, return empty DTCs (no real vehicle connected)
        return {
            "status": "success",
            "dtc_count": 0,
            "dtcs": [],
            "simulation_mode": True,
            "note": "No DTCs in simulation mode. Connect real OBD-II device for actual diagnostics.",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"DTC read error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/obd/dtcs/clear")
async def clear_diagnostic_codes():
    """
    🔧 Clear Diagnostic Trouble Codes
    
    Clears all stored DTCs from the vehicle's ECU.
    WARNING: This will also reset freeze frame data.
    """
    try:
        # In simulation mode, just return success
        return {
            "status": "success",
            "message": "DTCs cleared (simulation mode)",
            "simulation_mode": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"DTC clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/obd/safety")
async def get_obd_safety_thresholds():
    """
    🛡️ Get OBD-II Safety Thresholds
    
    Returns the safety monitoring thresholds for vehicle operation.
    """
    try:
        from src.protocols.obd_protocol import create_obd_protocol
        
        obd = create_obd_protocol(simulation_mode=True)
        
        return {
            "status": "success",
            "safety_thresholds": obd.safety_thresholds,
            "description": {
                "max_coolant_temp": "Maximum engine coolant temperature (°C)",
                "max_engine_rpm": "Maximum engine RPM",
                "max_vehicle_speed": "Maximum vehicle speed (km/h)",
                "min_battery_voltage": "Minimum battery voltage (V)",
                "max_battery_voltage": "Maximum battery voltage (V)",
                "min_fuel_level": "Minimum fuel level (%)"
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"OBD safety error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
