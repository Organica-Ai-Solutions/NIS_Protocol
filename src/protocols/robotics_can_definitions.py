#!/usr/bin/env python3
"""
Robotics CAN Message Definitions and Safety Protocols
Standardized CAN message IDs and data formats for robotic systems

Based on:
- CANopen standard (CiA 301)
- ROS-Can bridge conventions
- Industrial robotics safety standards (ISO 13849, IEC 61508)
- NASA robotics protocols
"""

from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import struct
import time
import numpy as np


class CANMessageID(IntEnum):
    """Standard CAN message IDs for robotics"""
    # Emergency and Safety (0x000-0x07F)
    EMERGENCY_STOP = 0x000
    SAFETY_RESET = 0x001
    HEARTBEAT_BASE = 0x700  # + node_id
    
    # System Control (0x080-0x0FF)
    SYSTEM_STATUS = 0x080
    SYSTEM_COMMAND = 0x081
    MODE_CONTROL = 0x082
    DIAGNOSTIC_REQUEST = 0x090
    DIAGNOSTIC_RESPONSE = 0x091
    
    # Motor Control (0x180-0x1FF)
    MOTOR_COMMAND_BASE = 0x200  # + motor_id
    MOTOR_STATUS_BASE = 0x280  # + motor_id
    MOTOR_FEEDBACK_BASE = 0x300  # + motor_id
    
    # Sensor Data (0x200-0x2FF)
    IMU_DATA = 0x200
    ENCODER_BASE = 0x210  # + encoder_id
    FORCE_TORQUE_BASE = 0x220  # + sensor_id
    VISION_DATA = 0x230
    LIDAR_DATA = 0x231
    
    # Kinematics and Motion (0x300-0x37F)
    JOINT_STATE_BASE = 0x300  # + joint_id
    END_EFFECTOR_POSE = 0x380
    TRAJECTORY_WAYPOINT = 0x381
    TRAJECTORY_STATUS = 0x382
    
    # Communication (0x400-0x47F)
    NIS_COMMAND = 0x400
    NIS_RESPONSE = 0x401
    NIS_STATUS = 0x402
    COORDINATION_DATA = 0x410
    
    # Power and Battery (0x500-0x57F)
    BATTERY_STATUS = 0x500
    POWER_DISTRIBUTION = 0x501
    VOLTAGE_MONITOR = 0x502
    CURRENT_MONITOR = 0x503


class MotorCommand(IntEnum):
    """Motor command types"""
    STOP = 0x00
    POSITION_CONTROL = 0x01
    VELOCITY_CONTROL = 0x02
    TORQUE_CONTROL = 0x03
    CALIBRATE = 0x04
    HOME = 0x05
    ENABLE = 0x06
    DISABLE = 0x07


class SystemMode(IntEnum):
    """System operating modes"""
    DISABLED = 0x00
    INITIALIZING = 0x01
    IDLE = 0x02
    AUTOMATIC = 0x03
    MANUAL = 0x04
    TEACHING = 0x05
    EMERGENCY = 0x06
    MAINTENANCE = 0x07
    RECOVERY = 0x08


class SafetyState(IntEnum):
    """Safety system states"""
    SAFE = 0x00
    WARNING = 0x01
    ERROR = 0x02
    FATAL = 0x03
    EMERGENCY_STOP = 0x04


@dataclass
class MotorCommandMessage:
    """Motor command CAN message format"""
    motor_id: int
    command: MotorCommand
    position: float = 0.0  # radians or meters
    velocity: float = 0.0  # rad/s or m/s
    torque: float = 0.0    # Nm or N
    max_effort: float = 0.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_can_data(self) -> bytes:
        """Convert to CAN data bytes"""
        # Format: command(1) + position(4) + velocity(4) + torque(4) + max_effort(4) + timestamp(4)
        return struct.pack('<BffffI',
                          self.command.value,
                          self.position,
                          self.velocity,
                          self.torque,
                          self.max_effort,
                          int(self.timestamp))
    
    @classmethod
    def from_can_data(cls, data: bytes) -> 'MotorCommandMessage':
        """Parse from CAN data bytes"""
        if len(data) < 21:  # Minimum expected size
            raise ValueError("Invalid motor command data length")
        
        command, pos, vel, torque, max_effort, timestamp = struct.unpack('<BffffI', data[:21])
        
        return cls(
            motor_id=0,  # Extract from CAN ID
            command=MotorCommand(command),
            position=pos,
            velocity=vel,
            torque=torque,
            max_effort=max_effort,
            timestamp=float(timestamp)
        )


@dataclass
class MotorStatusMessage:
    """Motor status CAN message format"""
    motor_id: int
    state: MotorCommand
    actual_position: float
    actual_velocity: float
    actual_torque: float
    temperature: float
    error_code: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_can_data(self) -> bytes:
        """Convert to CAN data bytes"""
        return struct.pack('<BfffiI',
                          self.state.value,
                          self.actual_position,
                          self.actual_velocity,
                          self.actual_torque,
                          self.temperature,
                          self.error_code,
                          int(self.timestamp))
    
    @classmethod
    def from_can_data(cls, data: bytes) -> 'MotorStatusMessage':
        """Parse from CAN data bytes"""
        if len(data) < 25:
            raise ValueError("Invalid motor status data length")
        
        state, pos, vel, torque, temp, error, timestamp = struct.unpack('<BfffiI', data[:25])
        
        return cls(
            motor_id=0,  # Extract from CAN ID
            state=MotorCommand(state),
            actual_position=pos,
            actual_velocity=vel,
            actual_torque=torque,
            temperature=temp,
            error_code=error,
            timestamp=float(timestamp)
        )


@dataclass
class IMUDataMessage:
    """IMU sensor data CAN message format"""
    acceleration: Tuple[float, float, float]  # m/s²
    angular_velocity: Tuple[float, float, float]  # rad/s
    orientation: Tuple[float, float, float, float]  # quaternion w,x,y,z
    magnetic_field: Tuple[float, float, float]  # Tesla
    temperature: float
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_can_data(self) -> bytes:
        """Convert to CAN data bytes"""
        # Pack into 8 bytes using scaled integers
        # accel: int16 (3), gyro: int16 (3), temp: int16
        accel_scaled = [int(a * 1000) for a in self.acceleration]
        gyro_scaled = [int(g * 1000) for g in self.angular_velocity]
        temp_scaled = int(self.temperature * 100)
        
        return struct.pack('<hhhhi',
                          *accel_scaled,
                          *gyro_scaled,
                          temp_scaled,
                          int(self.timestamp) & 0xFFFF)
    
    @classmethod
    def from_can_data(cls, data: bytes) -> 'IMUDataMessage':
        """Parse from CAN data bytes"""
        if len(data) < 8:
            raise ValueError("Invalid IMU data length")
        
        ax, ay, az, gx, gy, gz, temp, ts = struct.unpack('<hhhhi', data[:8])
        
        return cls(
            acceleration=(ax/1000.0, ay/1000.0, az/1000.0),
            angular_velocity=(gx/1000.0, gy/1000.0, gz/1000.0),
            orientation=(1.0, 0.0, 0.0, 0.0),  # Not in 8-byte format
            magnetic_field=(0.0, 0.0, 0.0),    # Not in 8-byte format
            temperature=temp/100.0,
            timestamp=float(ts)
        )


@dataclass
class JointStateMessage:
    """Joint state CAN message format"""
    joint_id: int
    position: float
    velocity: float
    effort: float
    temperature: float
    error_flags: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_can_data(self) -> bytes:
        """Convert to CAN data bytes"""
        return struct.pack('<ffffiI',
                          self.position,
                          self.velocity,
                          self.effort,
                          self.temperature,
                          self.error_flags,
                          int(self.timestamp))
    
    @classmethod
    def from_can_data(cls, data: bytes) -> 'JointStateMessage':
        """Parse from CAN data bytes"""
        if len(data) < 24:
            raise ValueError("Invalid joint state data length")
        
        pos, vel, effort, temp, flags, timestamp = struct.unpack('<ffffiI', data[:24])
        
        return cls(
            joint_id=0,  # Extract from CAN ID
            position=pos,
            velocity=vel,
            effort=effort,
            temperature=temp,
            error_flags=flags,
            timestamp=float(timestamp)
        )


@dataclass
class EndEffectorPoseMessage:
    """End effector pose CAN message format"""
    position: Tuple[float, float, float]  # x, y, z in meters
    orientation: Tuple[float, float, float, float]  # quaternion w, x, y, z
    velocity: Tuple[float, float, float]  # linear velocity
    angular_velocity: Tuple[float, float, float]  # angular velocity
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_can_data(self) -> bytes:
        """Convert to CAN data bytes (split across multiple CAN frames)"""
        # This is a simplified version - real implementation would use multi-frame
        pos_data = struct.pack('<fff', *self.position)
        return pos_data
    
    @classmethod
    def from_can_data(cls, data: bytes) -> 'EndEffectorPoseMessage':
        """Parse from CAN data bytes"""
        if len(data) < 12:
            raise ValueError("Invalid pose data length")
        
        x, y, z = struct.unpack('<fff', data[:12])
        
        return cls(
            position=(x, y, z),
            orientation=(1.0, 0.0, 0.0, 0.0),  # Default orientation
            velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            timestamp=time.time()
        )


class RoboticsSafetyProtocols:
    """Safety protocols for robotics CAN communication"""
    
    # Safety limits
    MAX_VELOCITY = 2.0  # m/s
    MAX_ACCELERATION = 10.0  # m/s²
    MAX_ANGULAR_VELOCITY = 3.14  # rad/s
    MAX_FORCE = 500.0  # N
    MAX_TORQUE = 200.0  # Nm
    MAX_TEMPERATURE = 80.0  # °C
    
    # Safety thresholds
    TEMPERATURE_WARNING = 60.0  # °C
    TEMPERATURE_CRITICAL = 75.0  # °C
    ERROR_THRESHOLD = 10  # Consecutive errors before shutdown
    
    def __init__(self):
        self.error_counters: Dict[int, int] = {}
        self.safety_violations: List[Dict[str, Any]] = []
        self.last_safety_check = time.time()
    
    def validate_motor_command(self, cmd: MotorCommandMessage) -> Tuple[bool, str]:
        """Validate motor command for safety"""
        violations = []
        
        # Check velocity limits
        if abs(cmd.velocity) > self.MAX_VELOCITY:
            violations.append(f"Velocity exceeds limit: {cmd.velocity} > {self.MAX_VELOCITY}")
        
        # Check torque limits
        if abs(cmd.torque) > self.MAX_TORQUE:
            violations.append(f"Torque exceeds limit: {cmd.torque} > {self.MAX_TORQUE}")
        
        # Check position limits (example: -π to π)
        if abs(cmd.position) > np.pi:
            violations.append(f"Position exceeds joint limits: {cmd.position}")
        
        # Check for NaN values
        if any(np.isnan([cmd.position, cmd.velocity, cmd.torque])):
            violations.append("NaN values detected in command")
        
        is_safe = len(violations) == 0
        error_msg = "; ".join(violations) if violations else ""
        
        return is_safe, error_msg
    
    def validate_sensor_data(self, sensor_type: str, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate sensor data for safety"""
        violations = []
        
        if sensor_type == "motor_status":
            temp = data.get("temperature", 0)
            if temp > self.TEMPERATURE_CRITICAL:
                violations.append(f"Critical temperature: {temp}°C")
            elif temp > self.TEMPERATURE_WARNING:
                violations.append(f"High temperature warning: {temp}°C")
        
        elif sensor_type == "imu":
            accel = data.get("acceleration", [0, 0, 0])
            gyro = data.get("angular_velocity", [0, 0, 0])
            
            # Check for reasonable values
            if any(abs(a) > 50 for a in accel):  # 50g limit
                violations.append(f"Unreasonable acceleration: {accel}")
            
            if any(abs(g) > 20 for g in gyro):  # 20 rad/s limit
                violations.append(f"Unreasonable angular velocity: {gyro}")
        
        is_safe = len(violations) == 0
        error_msg = "; ".join(violations) if violations else ""
        
        return is_safe, error_msg
    
    def check_error_counters(self, node_id: int) -> bool:
        """Check if node has exceeded error threshold"""
        error_count = self.error_counters.get(node_id, 0)
        return error_count >= self.ERROR_THRESHOLD
    
    def increment_error_counter(self, node_id: int):
        """Increment error counter for node"""
        self.error_counters[node_id] = self.error_counters.get(node_id, 0) + 1
    
    def reset_error_counter(self, node_id: int):
        """Reset error counter for node"""
        self.error_counters[node_id] = 0
    
    def log_safety_violation(self, violation_type: str, details: Dict[str, Any]):
        """Log safety violation"""
        self.safety_violations.append({
            "type": violation_type,
            "details": details,
            "timestamp": time.time()
        })
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        return {
            "error_counters": self.error_counters.copy(),
            "total_violations": len(self.safety_violations),
            "last_check": self.last_safety_check,
            "active_violations": [v for v in self.safety_violations 
                                if time.time() - v["timestamp"] < 60]  # Last minute
        }


# Standard CAN message definitions for robotics
ROBOTICS_CAN_MESSAGES = {
    "emergency_stop": {
        "id": CANMessageID.EMERGENCY_STOP,
        "dlc": 1,
        "description": "Emergency stop command"
    },
    "heartbeat": {
        "id": CANMessageID.HEARTBEAT_BASE,
        "dlc": 8,
        "description": "Node heartbeat message"
    },
    "motor_command": {
        "id": CANMessageID.MOTOR_COMMAND_BASE,
        "dlc": 8,
        "description": "Motor control command"
    },
    "motor_status": {
        "id": CANMessageID.MOTOR_STATUS_BASE,
        "dlc": 8,
        "description": "Motor status feedback"
    },
    "imu_data": {
        "id": CANMessageID.IMU_DATA,
        "dlc": 8,
        "description": "IMU sensor data"
    },
    "joint_state": {
        "id": CANMessageID.JOINT_STATE_BASE,
        "dlc": 8,
        "description": "Joint position/velocity/effort"
    },
    "end_effector_pose": {
        "id": CANMessageID.END_EFFECTOR_POSE,
        "dlc": 8,
        "description": "End effector pose"
    },
    "system_status": {
        "id": CANMessageID.SYSTEM_STATUS,
        "dlc": 8,
        "description": "System status and mode"
    }
}


def get_can_message_id(message_name: str, node_id: int = 0) -> int:
    """Get CAN message ID with optional node offset"""
    if message_name not in ROBOTICS_CAN_MESSAGES:
        raise ValueError(f"Unknown message type: {message_name}")
    
    base_id = ROBOTICS_CAN_MESSAGES[message_name]["id"]
    
    # Add node offset for base messages
    if "BASE" in message_name or message_name in ["heartbeat", "motor_command", "motor_status", "joint_state"]:
        return base_id + node_id
    
    return base_id


def create_motor_command_can_id(motor_id: int) -> int:
    """Create CAN ID for motor command"""
    return CANMessageID.MOTOR_COMMAND_BASE + motor_id


def create_motor_status_can_id(motor_id: int) -> int:
    """Create CAN ID for motor status"""
    return CANMessageID.MOTOR_STATUS_BASE + motor_id


def create_joint_state_can_id(joint_id: int) -> int:
    """Create CAN ID for joint state"""
    return CANMessageID.JOINT_STATE_BASE + joint_id


def create_heartbeat_can_id(node_id: int) -> int:
    """Create CAN ID for heartbeat"""
    return CANMessageID.HEARTBEAT_BASE + node_id
