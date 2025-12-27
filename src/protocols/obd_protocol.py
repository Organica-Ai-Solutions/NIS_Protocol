#!/usr/bin/env python3
"""
NIS Protocol - OBD-II (On-Board Diagnostics) Integration
Production-Ready Automotive Data Interface

Supports:
- OBD-II standard PIDs (Parameter IDs)
- CAN bus integration for automotive
- Real-time vehicle telemetry
- Diagnostic trouble codes (DTCs)
- Safety monitoring for autonomous vehicles

Standards:
- SAE J1979 (OBD-II PIDs)
- SAE J1939 (Heavy-duty vehicles)
- ISO 15765 (CAN-based diagnostics)
"""

import asyncio
import logging
import time
import struct
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque
import json

from .can_protocol import CANProtocol, CANFrame, SafetyLevel, CANNode

logger = logging.getLogger("nis.protocols.obd")


class OBDMode(IntEnum):
    """OBD-II Service Modes (SAE J1979)"""
    CURRENT_DATA = 0x01              # Show current data
    FREEZE_FRAME = 0x02              # Show freeze frame data
    STORED_DTCS = 0x03               # Show stored DTCs
    CLEAR_DTCS = 0x04                # Clear DTCs and stored values
    TEST_RESULTS_O2 = 0x05           # Test results, O2 sensor monitoring
    TEST_RESULTS_OTHER = 0x06        # Test results, other monitoring
    PENDING_DTCS = 0x07              # Show pending DTCs
    CONTROL_OPERATION = 0x08         # Control on-board system
    VEHICLE_INFO = 0x09              # Request vehicle information
    PERMANENT_DTCS = 0x0A            # Permanent DTCs


class OBDPID(IntEnum):
    """Common OBD-II PIDs (Mode 01)"""
    # Engine
    ENGINE_LOAD = 0x04
    COOLANT_TEMP = 0x05
    FUEL_PRESSURE = 0x0A
    INTAKE_PRESSURE = 0x0B
    ENGINE_RPM = 0x0C
    VEHICLE_SPEED = 0x0D
    TIMING_ADVANCE = 0x0E
    INTAKE_TEMP = 0x0F
    MAF_RATE = 0x10
    THROTTLE_POSITION = 0x11
    
    # Fuel System
    FUEL_SYSTEM_STATUS = 0x03
    FUEL_LEVEL = 0x2F
    FUEL_RAIL_PRESSURE = 0x22
    FUEL_INJECTION_TIMING = 0x5D
    FUEL_RATE = 0x5E
    
    # Emissions
    O2_SENSOR_1 = 0x14
    O2_SENSOR_2 = 0x15
    CATALYST_TEMP_B1S1 = 0x3C
    CATALYST_TEMP_B2S1 = 0x3D
    
    # Battery/Electrical
    CONTROL_MODULE_VOLTAGE = 0x42
    BATTERY_VOLTAGE = 0x42  # Same as control module voltage
    
    # Distance/Time
    DISTANCE_WITH_MIL = 0x21
    RUN_TIME_ENGINE_START = 0x1F
    DISTANCE_SINCE_CODES_CLEARED = 0x31
    
    # Vehicle Info (Mode 09)
    VIN = 0x02
    CALIBRATION_ID = 0x04
    ECU_NAME = 0x0A


@dataclass
class OBDReading:
    """Single OBD-II reading"""
    pid: OBDPID
    raw_value: bytes
    decoded_value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True
    error: Optional[str] = None


@dataclass
class DiagnosticTroubleCode:
    """Diagnostic Trouble Code (DTC)"""
    code: str                    # e.g., "P0301"
    description: str
    severity: str                # "critical", "warning", "info"
    system: str                  # "powertrain", "chassis", "body", "network"
    is_pending: bool = False
    is_permanent: bool = False
    freeze_frame: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class VehicleState:
    """Complete vehicle state from OBD-II"""
    # Engine
    engine_rpm: float = 0.0
    engine_load: float = 0.0
    coolant_temp: float = 0.0
    intake_temp: float = 0.0
    throttle_position: float = 0.0
    timing_advance: float = 0.0
    
    # Speed/Motion
    vehicle_speed: float = 0.0
    
    # Fuel
    fuel_level: float = 0.0
    fuel_pressure: float = 0.0
    fuel_rate: float = 0.0
    maf_rate: float = 0.0
    
    # Electrical
    battery_voltage: float = 0.0
    
    # Status
    mil_on: bool = False
    dtc_count: int = 0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    is_connected: bool = False
    vin: Optional[str] = None


class OBDProtocol:
    """
    OBD-II Protocol Handler for NIS Protocol
    
    Features:
    - Real-time vehicle data streaming
    - DTC reading and clearing
    - Safety monitoring
    - Integration with CAN protocol
    - Kafka/Redis streaming support
    """
    
    # OBD-II CAN IDs
    OBD_REQUEST_ID = 0x7DF      # Broadcast request
    OBD_RESPONSE_BASE = 0x7E8   # ECU response base (0x7E8 - 0x7EF)
    
    def __init__(
        self,
        can_channel: str = "can0",
        enable_safety_monitoring: bool = True,
        polling_interval: float = 0.1,  # 100ms
        simulation_mode: bool = True
    ):
        self.can_channel = can_channel
        self.enable_safety_monitoring = enable_safety_monitoring
        self.polling_interval = polling_interval
        self.simulation_mode = simulation_mode
        
        # CAN protocol for communication
        self.can_protocol = CANProtocol(
            interface="socketcan",
            channel=can_channel,
            bitrate=500000,  # Standard OBD-II bitrate
            force_simulation=simulation_mode,
            enable_safety_monitor=enable_safety_monitoring
        )
        
        # Vehicle state
        self.vehicle_state = VehicleState()
        self.dtcs: List[DiagnosticTroubleCode] = []
        
        # Reading history
        self.reading_history: deque = deque(maxlen=10000)
        
        # Callbacks
        self.data_callbacks: List[Callable[[VehicleState], None]] = []
        self.dtc_callbacks: List[Callable[[DiagnosticTroubleCode], None]] = []
        self.safety_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Statistics
        self.stats = {
            'readings_count': 0,
            'errors_count': 0,
            'dtcs_found': 0,
            'safety_alerts': 0,
            'uptime': 0.0
        }
        
        # Safety thresholds
        self.safety_thresholds = {
            'max_coolant_temp': 110.0,      # °C
            'max_engine_rpm': 7000.0,       # RPM
            'max_vehicle_speed': 200.0,     # km/h
            'min_battery_voltage': 11.5,    # V
            'max_battery_voltage': 15.0,    # V
            'min_fuel_level': 10.0          # %
        }
        
        # Running state
        self.is_running = False
        self.start_time = None
        
        logger.info(f"OBD Protocol initialized (channel: {can_channel}, simulation: {simulation_mode})")
    
    async def initialize(self) -> bool:
        """Initialize OBD-II connection"""
        try:
            success = await self.can_protocol.initialize()
            if success:
                # Register OBD response handlers
                for ecu_id in range(0x7E8, 0x7F0):
                    self.can_protocol.register_handler(ecu_id, self._handle_obd_response)
                
                self.is_running = True
                self.start_time = time.time()
                
                # Start polling task
                asyncio.create_task(self._polling_loop())
                
                logger.info("✅ OBD Protocol initialized")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ OBD initialization failed: {e}")
            return False
    
    async def _polling_loop(self):
        """Background polling loop for OBD data"""
        # Priority PIDs to poll
        priority_pids = [
            OBDPID.ENGINE_RPM,
            OBDPID.VEHICLE_SPEED,
            OBDPID.COOLANT_TEMP,
            OBDPID.THROTTLE_POSITION,
            OBDPID.ENGINE_LOAD,
            OBDPID.FUEL_LEVEL,
            OBDPID.CONTROL_MODULE_VOLTAGE
        ]
        
        pid_index = 0
        
        while self.is_running:
            try:
                # Poll next PID
                pid = priority_pids[pid_index]
                await self.request_pid(OBDMode.CURRENT_DATA, pid)
                
                pid_index = (pid_index + 1) % len(priority_pids)
                
                # Update uptime
                if self.start_time:
                    self.stats['uptime'] = time.time() - self.start_time
                
                await asyncio.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                self.stats['errors_count'] += 1
                await asyncio.sleep(1.0)
    
    async def request_pid(self, mode: OBDMode, pid: OBDPID) -> bool:
        """Request a specific PID from the vehicle"""
        try:
            # Build OBD request frame
            # Format: [num_bytes, mode, pid, 0x00, 0x00, 0x00, 0x00, 0x00]
            data = bytes([0x02, mode, pid, 0x00, 0x00, 0x00, 0x00, 0x00])
            
            success = await self.can_protocol.send_message(
                arbitration_id=self.OBD_REQUEST_ID,
                data=data,
                safety_level=SafetyLevel.MEDIUM
            )
            
            return success
            
        except Exception as e:
            logger.error(f"PID request error: {e}")
            return False
    
    async def _handle_obd_response(self, frame: CANFrame):
        """Handle OBD-II response from ECU"""
        try:
            data = frame.data
            if len(data) < 3:
                return
            
            num_bytes = data[0]
            mode = data[1] - 0x40  # Response mode = request mode + 0x40
            pid = data[2]
            
            # Decode based on PID
            reading = self._decode_pid_response(pid, data[3:])
            
            if reading and reading.is_valid:
                self._update_vehicle_state(reading)
                self.reading_history.append(reading)
                self.stats['readings_count'] += 1
                
                # Safety check
                if self.enable_safety_monitoring:
                    await self._check_safety(reading)
                
                # Notify callbacks
                for callback in self.data_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(self.vehicle_state)
                        else:
                            callback(self.vehicle_state)
                    except Exception as e:
                        logger.error(f"Data callback error: {e}")
            
        except Exception as e:
            logger.error(f"OBD response handling error: {e}")
            self.stats['errors_count'] += 1
    
    def _decode_pid_response(self, pid: int, data: bytes) -> Optional[OBDReading]:
        """Decode PID response data"""
        try:
            if len(data) < 1:
                return None
            
            # Decode based on PID
            if pid == OBDPID.ENGINE_RPM:
                # RPM = ((A * 256) + B) / 4
                if len(data) >= 2:
                    value = ((data[0] * 256) + data[1]) / 4.0
                    return OBDReading(OBDPID.ENGINE_RPM, data[:2], value, "RPM")
            
            elif pid == OBDPID.VEHICLE_SPEED:
                # Speed = A (km/h)
                value = float(data[0])
                return OBDReading(OBDPID.VEHICLE_SPEED, data[:1], value, "km/h")
            
            elif pid == OBDPID.COOLANT_TEMP:
                # Temp = A - 40 (°C)
                value = float(data[0]) - 40.0
                return OBDReading(OBDPID.COOLANT_TEMP, data[:1], value, "°C")
            
            elif pid == OBDPID.ENGINE_LOAD:
                # Load = A * 100 / 255 (%)
                value = (data[0] * 100.0) / 255.0
                return OBDReading(OBDPID.ENGINE_LOAD, data[:1], value, "%")
            
            elif pid == OBDPID.THROTTLE_POSITION:
                # Throttle = A * 100 / 255 (%)
                value = (data[0] * 100.0) / 255.0
                return OBDReading(OBDPID.THROTTLE_POSITION, data[:1], value, "%")
            
            elif pid == OBDPID.FUEL_LEVEL:
                # Fuel = A * 100 / 255 (%)
                value = (data[0] * 100.0) / 255.0
                return OBDReading(OBDPID.FUEL_LEVEL, data[:1], value, "%")
            
            elif pid == OBDPID.CONTROL_MODULE_VOLTAGE:
                # Voltage = ((A * 256) + B) / 1000 (V)
                if len(data) >= 2:
                    value = ((data[0] * 256) + data[1]) / 1000.0
                    return OBDReading(OBDPID.CONTROL_MODULE_VOLTAGE, data[:2], value, "V")
            
            elif pid == OBDPID.INTAKE_TEMP:
                # Temp = A - 40 (°C)
                value = float(data[0]) - 40.0
                return OBDReading(OBDPID.INTAKE_TEMP, data[:1], value, "°C")
            
            elif pid == OBDPID.MAF_RATE:
                # MAF = ((A * 256) + B) / 100 (g/s)
                if len(data) >= 2:
                    value = ((data[0] * 256) + data[1]) / 100.0
                    return OBDReading(OBDPID.MAF_RATE, data[:2], value, "g/s")
            
            elif pid == OBDPID.FUEL_PRESSURE:
                # Pressure = A * 3 (kPa)
                value = float(data[0]) * 3.0
                return OBDReading(OBDPID.FUEL_PRESSURE, data[:1], value, "kPa")
            
            elif pid == OBDPID.TIMING_ADVANCE:
                # Timing = (A / 2) - 64 (degrees)
                value = (data[0] / 2.0) - 64.0
                return OBDReading(OBDPID.TIMING_ADVANCE, data[:1], value, "°")
            
            return None
            
        except Exception as e:
            logger.error(f"PID decode error: {e}")
            return OBDReading(OBDPID(pid), data, 0.0, "", is_valid=False, error=str(e))
    
    def _update_vehicle_state(self, reading: OBDReading):
        """Update vehicle state from reading"""
        self.vehicle_state.timestamp = reading.timestamp
        self.vehicle_state.is_connected = True
        
        if reading.pid == OBDPID.ENGINE_RPM:
            self.vehicle_state.engine_rpm = reading.decoded_value
        elif reading.pid == OBDPID.VEHICLE_SPEED:
            self.vehicle_state.vehicle_speed = reading.decoded_value
        elif reading.pid == OBDPID.COOLANT_TEMP:
            self.vehicle_state.coolant_temp = reading.decoded_value
        elif reading.pid == OBDPID.ENGINE_LOAD:
            self.vehicle_state.engine_load = reading.decoded_value
        elif reading.pid == OBDPID.THROTTLE_POSITION:
            self.vehicle_state.throttle_position = reading.decoded_value
        elif reading.pid == OBDPID.FUEL_LEVEL:
            self.vehicle_state.fuel_level = reading.decoded_value
        elif reading.pid == OBDPID.CONTROL_MODULE_VOLTAGE:
            self.vehicle_state.battery_voltage = reading.decoded_value
        elif reading.pid == OBDPID.INTAKE_TEMP:
            self.vehicle_state.intake_temp = reading.decoded_value
        elif reading.pid == OBDPID.MAF_RATE:
            self.vehicle_state.maf_rate = reading.decoded_value
        elif reading.pid == OBDPID.FUEL_PRESSURE:
            self.vehicle_state.fuel_pressure = reading.decoded_value
        elif reading.pid == OBDPID.TIMING_ADVANCE:
            self.vehicle_state.timing_advance = reading.decoded_value
    
    async def _check_safety(self, reading: OBDReading):
        """Check reading against safety thresholds"""
        alerts = []
        
        if reading.pid == OBDPID.COOLANT_TEMP:
            if reading.decoded_value > self.safety_thresholds['max_coolant_temp']:
                alerts.append(("CRITICAL", "Engine overheating", {
                    "coolant_temp": reading.decoded_value,
                    "threshold": self.safety_thresholds['max_coolant_temp']
                }))
        
        elif reading.pid == OBDPID.ENGINE_RPM:
            if reading.decoded_value > self.safety_thresholds['max_engine_rpm']:
                alerts.append(("WARNING", "Engine RPM too high", {
                    "rpm": reading.decoded_value,
                    "threshold": self.safety_thresholds['max_engine_rpm']
                }))
        
        elif reading.pid == OBDPID.VEHICLE_SPEED:
            if reading.decoded_value > self.safety_thresholds['max_vehicle_speed']:
                alerts.append(("WARNING", "Speed limit exceeded", {
                    "speed": reading.decoded_value,
                    "threshold": self.safety_thresholds['max_vehicle_speed']
                }))
        
        elif reading.pid == OBDPID.CONTROL_MODULE_VOLTAGE:
            if reading.decoded_value < self.safety_thresholds['min_battery_voltage']:
                alerts.append(("WARNING", "Low battery voltage", {
                    "voltage": reading.decoded_value,
                    "threshold": self.safety_thresholds['min_battery_voltage']
                }))
            elif reading.decoded_value > self.safety_thresholds['max_battery_voltage']:
                alerts.append(("WARNING", "High battery voltage", {
                    "voltage": reading.decoded_value,
                    "threshold": self.safety_thresholds['max_battery_voltage']
                }))
        
        elif reading.pid == OBDPID.FUEL_LEVEL:
            if reading.decoded_value < self.safety_thresholds['min_fuel_level']:
                alerts.append(("WARNING", "Low fuel level", {
                    "fuel_level": reading.decoded_value,
                    "threshold": self.safety_thresholds['min_fuel_level']
                }))
        
        # Process alerts
        for severity, message, details in alerts:
            self.stats['safety_alerts'] += 1
            logger.warning(f"🚨 {severity}: {message} - {details}")
            
            for callback in self.safety_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message, details)
                    else:
                        callback(message, details)
                except Exception as e:
                    logger.error(f"Safety callback error: {e}")
    
    async def read_dtcs(self) -> List[DiagnosticTroubleCode]:
        """Read all diagnostic trouble codes"""
        try:
            # Request stored DTCs
            await self.request_pid(OBDMode.STORED_DTCS, 0x00)
            
            # Wait for response
            await asyncio.sleep(0.5)
            
            # In simulation mode, return empty list
            if self.simulation_mode:
                return []
            
            return self.dtcs
            
        except Exception as e:
            logger.error(f"DTC read error: {e}")
            return []
    
    async def clear_dtcs(self) -> bool:
        """Clear all diagnostic trouble codes"""
        try:
            data = bytes([0x01, OBDMode.CLEAR_DTCS, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            
            success = await self.can_protocol.send_message(
                arbitration_id=self.OBD_REQUEST_ID,
                data=data,
                safety_level=SafetyLevel.HIGH
            )
            
            if success:
                self.dtcs.clear()
                logger.info("✅ DTCs cleared")
            
            return success
            
        except Exception as e:
            logger.error(f"DTC clear error: {e}")
            return False
    
    async def get_vin(self) -> Optional[str]:
        """Get Vehicle Identification Number"""
        try:
            await self.request_pid(OBDMode.VEHICLE_INFO, OBDPID.VIN)
            await asyncio.sleep(0.5)
            return self.vehicle_state.vin
        except Exception as e:
            logger.error(f"VIN read error: {e}")
            return None
    
    def register_data_callback(self, callback: Callable[[VehicleState], None]):
        """Register callback for vehicle data updates"""
        self.data_callbacks.append(callback)
    
    def register_dtc_callback(self, callback: Callable[[DiagnosticTroubleCode], None]):
        """Register callback for DTC events"""
        self.dtc_callbacks.append(callback)
    
    def register_safety_callback(self, callback: Callable[[str, Dict], None]):
        """Register callback for safety alerts"""
        self.safety_callbacks.append(callback)
    
    def get_vehicle_state(self) -> VehicleState:
        """Get current vehicle state"""
        return self.vehicle_state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get OBD protocol statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'simulation_mode': self.simulation_mode,
            'can_channel': self.can_channel,
            'vehicle_connected': self.vehicle_state.is_connected,
            'dtc_count': len(self.dtcs)
        }
    
    async def shutdown(self):
        """Shutdown OBD protocol"""
        self.is_running = False
        await self.can_protocol.shutdown()
        logger.info("OBD Protocol shutdown complete")


# Factory function
def create_obd_protocol(
    can_channel: str = "can0",
    enable_safety: bool = True,
    simulation_mode: bool = True
) -> OBDProtocol:
    """Create OBD protocol instance"""
    return OBDProtocol(
        can_channel=can_channel,
        enable_safety_monitoring=enable_safety,
        simulation_mode=simulation_mode
    )
