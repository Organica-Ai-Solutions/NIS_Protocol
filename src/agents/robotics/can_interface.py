#!/usr/bin/env python3
"""
NIS Protocol v4.0 - Lightweight CAN Bus Interface
For robotics and drone control on embedded systems (Raspberry Pi, Jetson, etc.)

This is a simplified CAN interface optimized for:
- Low memory footprint (~50KB)
- SocketCAN on Linux (Raspberry Pi)
- Real-time control loops

Author: Organica AI Solutions
Date: December 2025
"""

import asyncio
import struct
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)

# Optional python-can import
try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False
    logger.info("python-can not installed - CAN bus will run in simulation mode")


class NISMessageID(IntEnum):
    """
    Standard CAN message IDs for NIS Protocol robotics.
    
    ID Ranges:
    - 0x000-0x0FF: System/Emergency (highest priority)
    - 0x100-0x1FF: Sensor data
    - 0x200-0x2FF: Control commands
    - 0x300-0x3FF: Status/Telemetry
    - 0x400-0x4FF: AI/Consciousness (NIS-specific)
    """
    # System messages (highest priority)
    EMERGENCY_STOP = 0x001
    HEARTBEAT = 0x002
    ARM_COMMAND = 0x010
    MODE_COMMAND = 0x011
    
    # Sensor data
    IMU_ACCEL = 0x100
    IMU_GYRO = 0x101
    IMU_RAW = 0x102
    GPS_POSITION = 0x110
    GPS_VELOCITY = 0x111
    BAROMETER = 0x120
    MAGNETOMETER = 0x130
    RANGEFINDER = 0x140
    OPTICAL_FLOW = 0x150
    
    # Control commands
    MOTOR_COMMAND = 0x200
    MOTOR_RAW = 0x201
    ATTITUDE_COMMAND = 0x210
    POSITION_COMMAND = 0x220
    VELOCITY_COMMAND = 0x230
    
    # Status/Telemetry
    ESC_STATUS = 0x300
    BATTERY_STATUS = 0x310
    SAFETY_STATUS = 0x320
    SYSTEM_STATUS = 0x330
    
    # AI/Consciousness (NIS-specific)
    AI_INFERENCE = 0x400
    AI_DECISION = 0x401
    CONSCIOUSNESS_STATE = 0x410
    AGENT_COMMAND = 0x420


@dataclass
class CANMessage:
    """Represents a CAN message"""
    arbitration_id: int
    data: bytes
    timestamp: float = 0.0
    is_extended: bool = False
    
    def __repr__(self):
        hex_data = ' '.join(f'{b:02X}' for b in self.data)
        return f"CAN[0x{self.arbitration_id:03X}] {hex_data}"


class CANBusInterface:
    """
    Lightweight CAN Bus interface for NIS Protocol.
    
    Supports:
    - SocketCAN (Linux/Raspberry Pi) - Primary
    - Virtual CAN (vcan) for testing
    - Simulation mode when no hardware
    
    Example:
        can = CANBusInterface("can0")
        await can.connect()
        await can.send(NISMessageID.HEARTBEAT, struct.pack('<I', uptime))
        can.register_handler(NISMessageID.IMU_RAW, handle_imu)
    """
    
    def __init__(self, channel: str = "can0", bitrate: int = 1000000):
        """
        Initialize CAN interface.
        
        Args:
            channel: CAN interface name (can0, vcan0, etc.)
            bitrate: CAN bus speed in bps (default 1Mbps)
        """
        self.channel = channel
        self.bitrate = bitrate
        self.bus: Optional[Any] = None
        self._handlers: Dict[int, Callable[[bytes], None]] = {}
        self._running = False
        self._simulation = not CAN_AVAILABLE
        
    @property
    def is_connected(self) -> bool:
        return self._running
        
    @property
    def is_simulation(self) -> bool:
        return self._simulation
        
    async def connect(self) -> bool:
        """
        Connect to CAN bus.
        
        Returns:
            True if connected (or simulation mode), False on error
        """
        if self._simulation:
            logger.info(f"CAN simulation mode (no hardware): {self.channel}")
            self._running = True
            return True
            
        try:
            self.bus = can.interface.Bus(
                channel=self.channel,
                interface='socketcan',
                bitrate=self.bitrate
            )
            self._running = True
            
            # Start receive task
            asyncio.create_task(self._receive_loop())
            
            logger.info(f"Connected to CAN bus: {self.channel} @ {self.bitrate}bps")
            return True
            
        except Exception as e:
            logger.error(f"CAN connect failed: {e}")
            logger.info("Falling back to simulation mode")
            self._simulation = True
            self._running = True
            return True
            
    async def disconnect(self):
        """Disconnect from CAN bus"""
        self._running = False
        if self.bus:
            self.bus.shutdown()
            self.bus = None
        logger.info("CAN bus disconnected")
            
    async def send(self, msg_id: int, data: bytes) -> bool:
        """
        Send a CAN message.
        
        Args:
            msg_id: CAN arbitration ID (use NISMessageID enum)
            data: Message data (max 8 bytes for standard CAN)
            
        Returns:
            True if sent successfully
        """
        if len(data) > 8:
            logger.warning(f"CAN data truncated to 8 bytes (was {len(data)})")
            data = data[:8]
            
        if self._simulation:
            # In simulation, just log
            logger.debug(f"CAN TX [SIM]: 0x{msg_id:03X} {data.hex()}")
            return True
            
        if not self.bus:
            return False
            
        try:
            msg = can.Message(
                arbitration_id=msg_id,
                data=data,
                is_extended_id=False
            )
            self.bus.send(msg)
            return True
        except Exception as e:
            logger.error(f"CAN send failed: {e}")
            return False
            
    def register_handler(self, msg_id: int, handler: Callable[[bytes], None]):
        """
        Register a handler for a specific message ID.
        
        Args:
            msg_id: CAN arbitration ID to handle
            handler: Callback function(data: bytes)
        """
        self._handlers[msg_id] = handler
        logger.debug(f"Registered handler for CAN ID 0x{msg_id:03X}")
        
    def unregister_handler(self, msg_id: int):
        """Remove a message handler"""
        self._handlers.pop(msg_id, None)
        
    async def _receive_loop(self):
        """Background task to receive CAN messages"""
        while self._running and self.bus:
            try:
                # Non-blocking receive with short timeout
                msg = self.bus.recv(timeout=0.001)
                if msg and msg.arbitration_id in self._handlers:
                    try:
                        self._handlers[msg.arbitration_id](msg.data)
                    except Exception as e:
                        logger.error(f"Handler error for 0x{msg.arbitration_id:03X}: {e}")
            except Exception:
                pass
            
            # Yield to other tasks
            await asyncio.sleep(0.0001)  # 100μs


# ==================== Message Encoding Helpers ====================

def encode_motor_command(m1: float, m2: float, m3: float, m4: float) -> bytes:
    """
    Encode motor command (4 motors, 0.0-1.0 range).
    
    Returns 8 bytes: 4x uint16 scaled to 0-1000
    """
    return struct.pack('<HHHH',
        int(max(0, min(1, m1)) * 1000),
        int(max(0, min(1, m2)) * 1000),
        int(max(0, min(1, m3)) * 1000),
        int(max(0, min(1, m4)) * 1000)
    )


def decode_motor_command(data: bytes) -> tuple:
    """Decode motor command to (m1, m2, m3, m4) floats"""
    if len(data) < 8:
        return (0.0, 0.0, 0.0, 0.0)
    m1, m2, m3, m4 = struct.unpack('<HHHH', data[:8])
    return (m1/1000.0, m2/1000.0, m3/1000.0, m4/1000.0)


def encode_attitude_command(roll: float, pitch: float, yaw_rate: float, thrust: float) -> bytes:
    """
    Encode attitude command.
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw_rate: Yaw rate in rad/s
        thrust: Thrust 0.0-1.0
        
    Returns 8 bytes: 3x int16 (mrad) + 1x uint16 (0-1000)
    """
    return struct.pack('<hhhH',
        int(roll * 1000),       # mrad
        int(pitch * 1000),      # mrad
        int(yaw_rate * 1000),   # mrad/s
        int(max(0, min(1, thrust)) * 1000)
    )


def encode_imu_data(ax: float, ay: float, az: float, gx: float, gy: float, gz: float) -> bytes:
    """
    Encode IMU data (accelerometer + gyro).
    
    Note: This requires 12 bytes, so we split into 2 messages or use CAN FD.
    This function returns accel only (6 bytes).
    """
    return struct.pack('<hhh',
        int(ax * 1000),  # mg
        int(ay * 1000),
        int(az * 1000)
    )


def decode_imu_accel(data: bytes) -> tuple:
    """Decode IMU accelerometer data to (ax, ay, az) in g"""
    if len(data) < 6:
        return (0.0, 0.0, 0.0)
    ax, ay, az = struct.unpack('<hhh', data[:6])
    return (ax/1000.0, ay/1000.0, az/1000.0)


def encode_gps_position(lat: float, lon: float) -> bytes:
    """
    Encode GPS position.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns 8 bytes: 2x int32 scaled by 1e7
    """
    return struct.pack('<ii',
        int(lat * 1e7),
        int(lon * 1e7)
    )


def decode_gps_position(data: bytes) -> tuple:
    """Decode GPS position to (lat, lon) in degrees"""
    if len(data) < 8:
        return (0.0, 0.0)
    lat, lon = struct.unpack('<ii', data[:8])
    return (lat / 1e7, lon / 1e7)


def encode_heartbeat(uptime_sec: int, status: int, mode: int) -> bytes:
    """
    Encode heartbeat message.
    
    Args:
        uptime_sec: System uptime in seconds
        status: System status (0=OK, 1=WARNING, 2=ERROR)
        mode: Current mode
        
    Returns 6 bytes
    """
    return struct.pack('<IBB', uptime_sec, status, mode)


# ==================== Exports ====================

__all__ = [
    'CANBusInterface',
    'CANMessage', 
    'NISMessageID',
    'CAN_AVAILABLE',
    'encode_motor_command',
    'decode_motor_command',
    'encode_attitude_command',
    'encode_imu_data',
    'decode_imu_accel',
    'encode_gps_position',
    'decode_gps_position',
    'encode_heartbeat',
]
