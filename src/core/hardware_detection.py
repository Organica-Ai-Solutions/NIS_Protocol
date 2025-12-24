"""
Hardware Auto-Detection Base Classes
Provides automatic hardware detection with graceful fallback to simulation
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Operation mode for hardware-dependent components"""
    REAL = "real"
    SIMULATION = "simulation"
    HYBRID = "hybrid"  # Some hardware available, some simulated


class HardwareStatus:
    """Status information for hardware detection"""
    
    def __init__(
        self,
        available: bool,
        mode: OperationMode,
        device_info: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.available = available
        self.mode = mode
        self.device_info = device_info or {}
        self.error = error
        self.detected_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "hardware_available": self.available,
            "operation_mode": self.mode.value,
            "device_info": self.device_info,
            "error": self.error,
            "detected_at": self.detected_at
        }


class HardwareDetector(ABC):
    """
    Base class for hardware auto-detection
    
    Implements the pattern:
    1. Try to detect and connect to real hardware
    2. If successful, use real mode
    3. If failed, fall back to simulation mode
    4. Log which mode is active
    """
    
    def __init__(self, force_simulation: bool = False):
        """
        Initialize hardware detector
        
        Args:
            force_simulation: If True, skip hardware detection and use simulation
        """
        self.force_simulation = force_simulation
        self._status: Optional[HardwareStatus] = None
        self._initialized = False
    
    @property
    def status(self) -> HardwareStatus:
        """Get current hardware status"""
        if not self._initialized:
            self.detect()
        return self._status
    
    @property
    def mode(self) -> OperationMode:
        """Get current operation mode"""
        return self.status.mode
    
    @property
    def is_real(self) -> bool:
        """Check if using real hardware"""
        return self.mode == OperationMode.REAL
    
    @property
    def is_simulation(self) -> bool:
        """Check if using simulation"""
        return self.mode == OperationMode.SIMULATION
    
    def detect(self) -> HardwareStatus:
        """
        Detect hardware availability
        
        Returns:
            HardwareStatus with detection results
        """
        if self.force_simulation:
            logger.info(f"{self.__class__.__name__}: Forced simulation mode")
            self._status = HardwareStatus(
                available=False,
                mode=OperationMode.SIMULATION,
                device_info={"forced": True}
            )
            self._initialized = True
            return self._status
        
        try:
            # Try to detect real hardware
            available, device_info = self._detect_hardware()
            
            if available:
                mode = OperationMode.REAL
                logger.info(
                    f"{self.__class__.__name__}: Real hardware detected - {device_info}"
                )
            else:
                mode = OperationMode.SIMULATION
                logger.warning(
                    f"{self.__class__.__name__}: No hardware detected, using simulation mode"
                )
            
            self._status = HardwareStatus(
                available=available,
                mode=mode,
                device_info=device_info
            )
            
        except Exception as e:
            logger.error(
                f"{self.__class__.__name__}: Hardware detection failed: {e}, "
                "falling back to simulation"
            )
            self._status = HardwareStatus(
                available=False,
                mode=OperationMode.SIMULATION,
                error=str(e)
            )
        
        self._initialized = True
        return self._status
    
    @abstractmethod
    def _detect_hardware(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Implement hardware-specific detection logic
        
        Returns:
            Tuple of (hardware_available, device_info)
        """
        pass
    
    @abstractmethod
    def get_real_data(self, *args, **kwargs) -> Any:
        """Get data from real hardware"""
        pass
    
    @abstractmethod
    def get_simulation_data(self, *args, **kwargs) -> Any:
        """Get simulated data"""
        pass
    
    def get_data(self, *args, **kwargs) -> Any:
        """
        Get data - automatically uses real or simulation based on detection
        
        Returns data with metadata indicating which mode was used
        """
        if not self._initialized:
            self.detect()
        
        if self.is_real:
            try:
                data = self.get_real_data(*args, **kwargs)
                return self._add_metadata(data, OperationMode.REAL)
            except Exception as e:
                logger.error(
                    f"{self.__class__.__name__}: Real hardware failed: {e}, "
                    "falling back to simulation"
                )
                # Fall back to simulation on error
                data = self.get_simulation_data(*args, **kwargs)
                return self._add_metadata(data, OperationMode.SIMULATION, error=str(e))
        else:
            data = self.get_simulation_data(*args, **kwargs)
            return self._add_metadata(data, OperationMode.SIMULATION)
    
    def _add_metadata(
        self,
        data: Any,
        mode: OperationMode,
        error: Optional[str] = None
    ) -> Any:
        """Add operation mode metadata to response"""
        if isinstance(data, dict):
            data["_hardware_status"] = {
                "mode": mode.value,
                "hardware_available": self._status.available if self._status else False,
                "error": error,
                "timestamp": time.time()
            }
        return data


class CANHardwareDetector(HardwareDetector):
    """Hardware detector for CAN bus interfaces"""
    
    def _detect_hardware(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect CAN hardware (SocketCAN on Linux)"""
        try:
            import can
            
            # Try to detect available CAN interfaces
            interfaces = []
            
            # Check for SocketCAN interfaces (Linux)
            try:
                import subprocess
                result = subprocess.run(
                    ['ip', 'link', 'show'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if 'can' in result.stdout.lower():
                    # Parse CAN interfaces
                    for line in result.stdout.split('\n'):
                        if 'can' in line.lower():
                            interfaces.append(line.split(':')[1].strip() if ':' in line else 'can0')
            except:
                pass
            
            if interfaces:
                return True, {
                    "interface_type": "socketcan",
                    "interfaces": interfaces,
                    "platform": "linux"
                }
            
            # If no interfaces found, check if python-can is at least installed
            return False, {
                "interface_type": "none",
                "python_can_available": True,
                "note": "python-can installed but no CAN interfaces detected"
            }
            
        except ImportError:
            return False, {
                "error": "python-can not installed",
                "install_command": "pip install python-can"
            }
    
    def get_real_data(self, *args, **kwargs) -> Dict[str, Any]:
        """Get real CAN bus data"""
        import can
        
        interface = self._status.device_info.get("interfaces", ["can0"])[0]
        
        try:
            bus = can.interface.Bus(channel=interface, bustype='socketcan')
            
            # Read a message with timeout
            msg = bus.recv(timeout=1.0)
            
            if msg:
                return {
                    "arbitration_id": msg.arbitration_id,
                    "data": list(msg.data),
                    "timestamp": msg.timestamp,
                    "is_extended_id": msg.is_extended_id
                }
            else:
                return {"status": "no_message", "timeout": 1.0}
                
        except Exception as e:
            raise RuntimeError(f"Failed to read from CAN bus: {e}")
    
    def get_simulation_data(self, *args, **kwargs) -> Dict[str, Any]:
        """Get simulated CAN data"""
        return {
            "arbitration_id": 0x123,
            "data": [0, 0, 0, 0, 0, 0, 0, 0],
            "timestamp": time.time(),
            "is_extended_id": False,
            "simulated": True
        }


class OBDHardwareDetector(HardwareDetector):
    """Hardware detector for OBD-II interfaces"""
    
    def _detect_hardware(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect OBD-II hardware (ELM327 adapters)"""
        try:
            import obd
            
            # Try to auto-connect to OBD adapter
            connection = obd.OBD()
            
            if connection.is_connected():
                return True, {
                    "adapter_type": "ELM327",
                    "port": connection.port_name(),
                    "protocol": str(connection.protocol_name()),
                    "connected": True
                }
            else:
                return False, {
                    "adapter_type": "none",
                    "python_obd_available": True,
                    "note": "python-obd installed but no OBD adapter detected"
                }
                
        except ImportError:
            return False, {
                "error": "python-obd not installed",
                "install_command": "pip install obd"
            }
        except Exception as e:
            return False, {
                "error": f"OBD detection failed: {str(e)}"
            }
    
    def get_real_data(self, *args, **kwargs) -> Dict[str, Any]:
        """Get real OBD-II data"""
        import obd
        
        connection = obd.OBD()
        
        if not connection.is_connected():
            raise RuntimeError("OBD adapter not connected")
        
        # Query common PIDs
        rpm = connection.query(obd.commands.RPM)
        speed = connection.query(obd.commands.SPEED)
        throttle = connection.query(obd.commands.THROTTLE_POS)
        coolant_temp = connection.query(obd.commands.COOLANT_TEMP)
        
        return {
            "rpm": rpm.value.magnitude if rpm.value else None,
            "speed_kph": speed.value.magnitude if speed.value else None,
            "throttle_percent": throttle.value.magnitude if throttle.value else None,
            "coolant_temp_c": coolant_temp.value.magnitude if coolant_temp.value else None,
            "timestamp": time.time()
        }
    
    def get_simulation_data(self, *args, **kwargs) -> Dict[str, Any]:
        """Get simulated OBD-II data"""
        return {
            "rpm": 2000,
            "speed_kph": 60,
            "throttle_percent": 45,
            "coolant_temp_c": 85,
            "timestamp": time.time(),
            "simulated": True
        }
