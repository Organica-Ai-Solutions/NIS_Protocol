#!/usr/bin/env python3
"""
Enhanced OBD-II Protocol with Hardware Auto-Detection
Automatically detects real OBD adapters and falls back to simulation
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import IntEnum

from src.core.hardware_detection import OBDHardwareDetector, OperationMode
from .obd_protocol import OBDMode, OBDPID, OBDReading, OBDProtocol

logger = logging.getLogger("nis.protocols.obd_enhanced")


class EnhancedOBDProtocol(OBDProtocol):
    """
    Enhanced OBD-II Protocol with automatic hardware detection
    
    Features:
    - Auto-detects real OBD-II adapters (ELM327, etc.)
    - Graceful fallback to simulation mode
    - Metadata in responses indicating operation mode
    - Support for all standard OBD-II PIDs
    """
    
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 38400,
        protocol: Optional[str] = None,
        force_simulation: bool = False
    ):
        """
        Initialize enhanced OBD protocol
        
        Args:
            port: Serial port for OBD adapter (auto-detect if None)
            baudrate: Serial baudrate (default 38400 for ELM327)
            protocol: OBD protocol to use (auto-detect if None)
            force_simulation: Force simulation mode even if hardware available
        """
        super().__init__()
        
        self.port = port
        self.baudrate = baudrate
        self.protocol = protocol
        
        # Hardware auto-detection
        self.hardware_detector = OBDHardwareDetector(force_simulation=force_simulation)
        self.hardware_detector.detect()
        self.operation_mode = self.hardware_detector.mode
        
        # Connection state
        self.connection = None
        self.is_connected = False
        
        # Statistics
        self.stats = {
            'readings_count': 0,
            'errors_count': 0,
            'dtcs_count': 0,
            'uptime': 0.0
        }
        
        logger.info(f"Enhanced OBD Protocol initialized (Mode: {self.operation_mode.value})")
    
    async def connect(self) -> bool:
        """
        Connect to OBD adapter
        
        Returns:
            True if connected successfully
        """
        try:
            if self.operation_mode == OperationMode.REAL:
                try:
                    import obd
                    
                    # Try to connect to real adapter
                    if self.port:
                        self.connection = obd.OBD(portstr=self.port, baudrate=self.baudrate)
                    else:
                        self.connection = obd.OBD()  # Auto-detect
                    
                    if self.connection.is_connected():
                        self.is_connected = True
                        self.stats['start_time'] = time.time()
                        logger.info(f"Connected to OBD adapter on {self.connection.port_name()}")
                        return True
                    else:
                        logger.warning("Failed to connect to OBD adapter, falling back to simulation")
                        self.operation_mode = OperationMode.SIMULATION
                        
                except Exception as e:
                    logger.error(f"OBD connection error: {e}, falling back to simulation")
                    self.operation_mode = OperationMode.SIMULATION
            
            # Simulation mode
            if self.operation_mode == OperationMode.SIMULATION:
                self.is_connected = True
                self.stats['start_time'] = time.time()
                logger.info("OBD simulation mode enabled")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize OBD connection: {e}")
            return False
    
    async def read_pid(self, pid: OBDPID, mode: OBDMode = OBDMode.CURRENT_DATA) -> Optional[OBDReading]:
        """
        Read a specific PID
        
        Args:
            pid: OBD PID to read
            mode: OBD mode (default: current data)
        
        Returns:
            OBD reading or None if failed
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if self.operation_mode == OperationMode.REAL:
                return await self._read_real_pid(pid, mode)
            else:
                return await self._read_simulated_pid(pid, mode)
                
        except Exception as e:
            logger.error(f"Failed to read PID {pid.name}: {e}")
            self.stats['errors_count'] += 1
            return None
    
    async def _read_real_pid(self, pid: OBDPID, mode: OBDMode) -> Optional[OBDReading]:
        """Read PID from real OBD adapter"""
        import obd
        
        # Map our PID enum to python-obd commands
        pid_map = {
            OBDPID.ENGINE_RPM: obd.commands.RPM,
            OBDPID.VEHICLE_SPEED: obd.commands.SPEED,
            OBDPID.THROTTLE_POSITION: obd.commands.THROTTLE_POS,
            OBDPID.COOLANT_TEMP: obd.commands.COOLANT_TEMP,
            OBDPID.ENGINE_LOAD: obd.commands.ENGINE_LOAD,
            OBDPID.INTAKE_TEMP: obd.commands.INTAKE_TEMP,
            OBDPID.MAF_RATE: obd.commands.MAF,
            OBDPID.FUEL_LEVEL: obd.commands.FUEL_LEVEL,
        }
        
        command = pid_map.get(pid)
        if not command:
            logger.warning(f"PID {pid.name} not mapped to python-obd command")
            return None
        
        response = self.connection.query(command)
        
        if response.value is not None:
            self.stats['readings_count'] += 1
            return OBDReading(
                pid=pid,
                raw_value=bytes(),  # python-obd doesn't expose raw bytes
                decoded_value=float(response.value.magnitude),
                unit=str(response.value.units),
                timestamp=time.time(),
                is_valid=True
            )
        else:
            self.stats['errors_count'] += 1
            return None
    
    async def _read_simulated_pid(self, pid: OBDPID, mode: OBDMode) -> Optional[OBDReading]:
        """Read simulated PID data"""
        # Simulated values for common PIDs
        simulated_data = {
            OBDPID.ENGINE_RPM: (2000.0, "rpm"),
            OBDPID.VEHICLE_SPEED: (60.0, "kph"),
            OBDPID.THROTTLE_POSITION: (45.0, "percent"),
            OBDPID.COOLANT_TEMP: (85.0, "celsius"),
            OBDPID.ENGINE_LOAD: (50.0, "percent"),
            OBDPID.INTAKE_TEMP: (25.0, "celsius"),
            OBDPID.MAF_RATE: (15.5, "gps"),
            OBDPID.FUEL_LEVEL: (75.0, "percent"),
        }
        
        if pid in simulated_data:
            value, unit = simulated_data[pid]
            self.stats['readings_count'] += 1
            return OBDReading(
                pid=pid,
                raw_value=bytes([int(value)]),
                decoded_value=value,
                unit=unit,
                timestamp=time.time(),
                is_valid=True
            )
        else:
            return None
    
    async def get_vehicle_data(self) -> Dict[str, Any]:
        """
        Get comprehensive vehicle data
        
        Returns:
            Dictionary with all available vehicle telemetry
        """
        data = {
            "operation_mode": self.operation_mode.value,
            "hardware_available": self.hardware_detector.status.available,
            "is_connected": self.is_connected,
            "timestamp": time.time()
        }
        
        # Read common PIDs
        pids_to_read = [
            OBDPID.ENGINE_RPM,
            OBDPID.VEHICLE_SPEED,
            OBDPID.THROTTLE_POSITION,
            OBDPID.COOLANT_TEMP,
            OBDPID.ENGINE_LOAD,
            OBDPID.FUEL_LEVEL
        ]
        
        readings = {}
        for pid in pids_to_read:
            reading = await self.read_pid(pid)
            if reading and reading.is_valid:
                readings[pid.name.lower()] = {
                    "value": reading.decoded_value,
                    "unit": reading.unit
                }
        
        data["readings"] = readings
        data["stats"] = self.stats.copy()
        
        return data
    
    async def get_dtcs(self) -> List[Dict[str, Any]]:
        """
        Get diagnostic trouble codes
        
        Returns:
            List of DTCs with descriptions
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if self.operation_mode == OperationMode.REAL:
                import obd
                
                response = self.connection.query(obd.commands.GET_DTC)
                
                if response.value:
                    dtcs = []
                    for code, description in response.value:
                        dtcs.append({
                            "code": code,
                            "description": description,
                            "timestamp": time.time()
                        })
                    self.stats['dtcs_count'] = len(dtcs)
                    return dtcs
                else:
                    return []
            else:
                # Simulation mode - return empty list
                return []
                
        except Exception as e:
            logger.error(f"Failed to read DTCs: {e}")
            return []
    
    async def clear_dtcs(self) -> bool:
        """
        Clear diagnostic trouble codes
        
        Returns:
            True if cleared successfully
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if self.operation_mode == OperationMode.REAL:
                import obd
                
                response = self.connection.query(obd.commands.CLEAR_DTC)
                return response.is_null() == False
            else:
                # Simulation mode - always succeed
                logger.info("DTCs cleared (simulation mode)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear DTCs: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from OBD adapter"""
        if self.connection and self.operation_mode == OperationMode.REAL:
            self.connection.close()
        
        self.is_connected = False
        logger.info("OBD connection closed")


# Factory function
def create_obd_protocol(
    port: Optional[str] = None,
    force_simulation: bool = False
) -> EnhancedOBDProtocol:
    """
    Create enhanced OBD protocol instance
    
    Args:
        port: Serial port for OBD adapter (auto-detect if None)
        force_simulation: Force simulation mode
    
    Returns:
        EnhancedOBDProtocol instance
    """
    return EnhancedOBDProtocol(
        port=port,
        force_simulation=force_simulation
    )
