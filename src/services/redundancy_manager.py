"""
NASA-Grade Redundancy Manager for NIS Protocol Embodiment
==========================================================

Implements aerospace reliability patterns:
- Triple Modular Redundancy (TMR)
- Watchdog timers
- Graceful degradation
- Self-diagnostics
- Error detection and correction

HONEST NOTE: This is a SIMULATION of redundancy patterns.
In real hardware, you'd have actual redundant sensors/actuators.
But the LOGIC is production-grade and directly applicable.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import logging
import statistics

logger = logging.getLogger("nis.redundancy")


class ComponentHealth(Enum):
    """Component health states"""
    NOMINAL = "nominal"           # Operating normally
    DEGRADED = "degraded"         # Reduced capability
    FAILED = "failed"             # Not operational
    UNKNOWN = "unknown"           # Cannot determine


class RedundancyMode(Enum):
    """Redundancy configurations"""
    TRIPLE_MODULAR = "triple_modular"      # 3 sensors, majority vote
    DUAL_REDUNDANT = "dual_redundant"      # 2 sensors, cross-check
    HOT_SPARE = "hot_spare"                # Active + standby
    SINGLE_POINT = "single_point"          # No redundancy (dangerous)


class WatchdogTimer:
    """
    Watchdog timer to detect hung operations
    If not reset within timeout, triggers failsafe
    """
    def __init__(self, timeout: float, name: str):
        self.timeout = timeout
        self.name = name
        self.last_reset = time.time()
        self.enabled = True
        self.triggered = False
        
    def reset(self):
        """Reset the watchdog - call this regularly in normal operation"""
        self.last_reset = time.time()
        self.triggered = False
        
    def check(self) -> bool:
        """Check if watchdog has timed out"""
        if not self.enabled:
            return False
            
        elapsed = time.time() - self.last_reset
        if elapsed > self.timeout:
            self.triggered = True
            logger.error(f"⏰ WATCHDOG TIMEOUT: {self.name} ({elapsed:.2f}s > {self.timeout}s)")
            return True
        return False


class RedundantSensor:
    """
    Triple Modular Redundancy (TMR) for sensor readings
    
    Reality: In actual hardware, these would be 3 physical sensors.
    Here: We simulate sensor failures and voting logic.
    """
    def __init__(self, sensor_name: str, num_channels: int = 3):
        self.sensor_name = sensor_name
        self.num_channels = num_channels
        self.channel_health = [ComponentHealth.NOMINAL] * num_channels
        self.channel_readings: List[Optional[float]] = [None] * num_channels
        self.failure_counts = [0] * num_channels
        self.MAX_FAILURES = 3  # After 3 failures, mark as failed
        
    def read_all_channels(self, true_value: float) -> List[Optional[float]]:
        """
        Simulate reading from all redundant channels
        In real hardware: actual sensor I/O
        In simulation: add realistic noise/failures
        """
        readings = []
        for i in range(self.num_channels):
            if self.channel_health[i] == ComponentHealth.FAILED:
                readings.append(None)
            elif self.channel_health[i] == ComponentHealth.DEGRADED:
                # Degraded sensor: higher noise
                noise = (time.time() % 1.0 - 0.5) * 0.2
                readings.append(true_value + noise)
            else:
                # Nominal: small noise
                noise = (time.time() % 1.0 - 0.5) * 0.02
                readings.append(true_value + noise)
        
        self.channel_readings = readings
        return readings
    
    def majority_vote(self, readings: List[Optional[float]], tolerance: float = 0.1) -> Tuple[Optional[float], bool]:
        """
        NASA-style majority voting algorithm
        
        Returns: (voted_value, agreement_flag)
        """
        valid_readings = [r for r in readings if r is not None]
        
        if len(valid_readings) < 2:
            # Cannot vote with less than 2 sensors
            logger.warning(f"⚠️ {self.sensor_name}: Insufficient sensors ({len(valid_readings)}/{self.num_channels})")
            return None, False
        
        # Check if readings agree within tolerance
        avg = statistics.mean(valid_readings)
        disagreements = 0
        
        for i, reading in enumerate(readings):
            if reading is not None:
                if abs(reading - avg) > tolerance:
                    disagreements += 1
                    self.failure_counts[i] += 1
                    logger.warning(f"⚠️ {self.sensor_name} channel {i}: Reading {reading:.3f} vs avg {avg:.3f}")
                    
                    # Mark as failed after too many disagreements
                    if self.failure_counts[i] >= self.MAX_FAILURES:
                        self.channel_health[i] = ComponentHealth.FAILED
                        logger.error(f"❌ {self.sensor_name} channel {i}: FAILED after {self.MAX_FAILURES} errors")
        
        # Majority vote result
        if disagreements == 0:
            return avg, True
        elif len(valid_readings) >= 2:
            # Use median (more robust to outliers)
            return statistics.median(valid_readings), False
        else:
            return None, False
    
    def get_health(self) -> ComponentHealth:
        """Overall health of redundant sensor"""
        failed = sum(1 for h in self.channel_health if h == ComponentHealth.FAILED)
        
        if failed >= self.num_channels - 1:
            return ComponentHealth.FAILED
        elif failed > 0:
            return ComponentHealth.DEGRADED
        else:
            return ComponentHealth.NOMINAL


class RedundancyManager:
    """
    Main redundancy manager for embodiment system
    Implements NASA/aerospace reliability patterns
    """
    
    def __init__(self):
        # Redundant sensors (3 channels each)
        self.sensors = {
            "position_x": RedundantSensor("position_x", 3),
            "position_y": RedundantSensor("position_y", 3),
            "position_z": RedundantSensor("position_z", 3),
            "battery": RedundantSensor("battery", 3),
            "temperature": RedundantSensor("temperature", 3),
        }
        
        # Watchdog timers
        self.watchdogs = {
            "motion_execution": WatchdogTimer(5.0, "motion_execution"),
            "safety_check": WatchdogTimer(2.0, "safety_check"),
            "system_heartbeat": WatchdogTimer(10.0, "system_heartbeat"),
        }
        
        # System state
        self.system_health = ComponentHealth.NOMINAL
        self.degraded_components: List[str] = []
        self.failsafe_active = False
        
        # Performance tracking
        self.total_checks = 0
        self.disagreements = 0
        self.sensor_failures = 0
        
        logger.info("🛰️ Redundancy Manager initialized (NASA-grade patterns)")
    
    async def read_sensor_redundant(
        self, 
        sensor_name: str, 
        true_value: float
    ) -> Dict[str, Any]:
        """
        Read sensor with TMR (Triple Modular Redundancy)
        
        Returns:
        {
            "value": voted_value,
            "agreement": bool,
            "health": ComponentHealth,
            "raw_readings": [ch1, ch2, ch3],
            "failed_channels": [indices]
        }
        """
        if sensor_name not in self.sensors:
            logger.error(f"❌ Unknown sensor: {sensor_name}")
            return {"error": "unknown_sensor"}
        
        sensor = self.sensors[sensor_name]
        
        # Read all channels
        readings = sensor.read_all_channels(true_value)
        
        # Majority vote
        voted_value, agreement = sensor.majority_vote(readings)
        
        # Track statistics
        self.total_checks += 1
        if not agreement:
            self.disagreements += 1
        
        # Get health
        health = sensor.get_health()
        if health == ComponentHealth.FAILED:
            self.sensor_failures += 1
        
        failed_channels = [
            i for i, h in enumerate(sensor.channel_health) 
            if h == ComponentHealth.FAILED
        ]
        
        return {
            "sensor": sensor_name,
            "value": voted_value,
            "agreement": agreement,
            "health": health.value,
            "raw_readings": readings,
            "failed_channels": failed_channels,
            "timestamp": datetime.now().isoformat()
        }
    
    async def check_all_sensors(
        self, 
        body_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check all redundant sensors and return voted values
        """
        results = {}
        
        # Check position sensors
        pos = body_state.get("position", {})
        results["position_x"] = await self.read_sensor_redundant("position_x", pos.get("x", 0.0))
        results["position_y"] = await self.read_sensor_redundant("position_y", pos.get("y", 0.0))
        results["position_z"] = await self.read_sensor_redundant("position_z", pos.get("z", 0.0))
        
        # Check system sensors
        results["battery"] = await self.read_sensor_redundant("battery", body_state.get("battery_level", 100.0))
        results["temperature"] = await self.read_sensor_redundant("temperature", body_state.get("temperature", 20.0))
        
        # Overall health assessment
        failed_sensors = [
            name for name, data in results.items()
            if data.get("health") == ComponentHealth.FAILED.value
        ]
        
        degraded_sensors = [
            name for name, data in results.items()
            if data.get("health") == ComponentHealth.DEGRADED.value
        ]
        
        if len(failed_sensors) > 0:
            self.system_health = ComponentHealth.FAILED
        elif len(degraded_sensors) > 0:
            self.system_health = ComponentHealth.DEGRADED
        else:
            self.system_health = ComponentHealth.NOMINAL
        
        self.degraded_components = failed_sensors + degraded_sensors
        
        return {
            "sensor_results": results,
            "system_health": self.system_health.value,
            "degraded_components": self.degraded_components,
            "statistics": {
                "total_checks": self.total_checks,
                "disagreement_rate": self.disagreements / max(self.total_checks, 1),
                "sensor_failures": self.sensor_failures
            }
        }
    
    async def check_watchdogs(self) -> Dict[str, bool]:
        """Check all watchdog timers"""
        triggered = {}
        
        for name, watchdog in self.watchdogs.items():
            if watchdog.check():
                triggered[name] = True
                await self.trigger_failsafe(f"Watchdog timeout: {name}")
        
        return triggered
    
    async def trigger_failsafe(self, reason: str):
        """
        Activate failsafe mode
        In real hardware: emergency stop, safe mode
        """
        if not self.failsafe_active:
            self.failsafe_active = True
            logger.critical(f"🚨 FAILSAFE ACTIVATED: {reason}")
            
            # In real system:
            # - Stop all motion
            # - Lock actuators
            # - Alert operators
            # - Log to black box
    
    def graceful_degradation(self) -> Dict[str, Any]:
        """
        Determine what operations are still safe in degraded mode
        
        This is NASA-style graceful degradation:
        - Lose one sensor: continue with reduced capability
        - Lose two sensors: safe mode only
        - Lose all sensors: emergency stop
        """
        failed = len([s for s in self.sensors.values() if s.get_health() == ComponentHealth.FAILED])
        degraded = len([s for s in self.sensors.values() if s.get_health() == ComponentHealth.DEGRADED])
        
        if self.system_health == ComponentHealth.NOMINAL:
            return {
                "mode": "nominal",
                "allowed_operations": ["full_motion", "high_speed", "autonomous"],
                "restrictions": []
            }
        elif self.system_health == ComponentHealth.DEGRADED:
            return {
                "mode": "degraded",
                "allowed_operations": ["reduced_motion", "low_speed", "supervised"],
                "restrictions": [
                    "Max speed reduced to 50%",
                    "No autonomous operation",
                    "Constant monitoring required"
                ]
            }
        else:  # FAILED
            return {
                "mode": "failsafe",
                "allowed_operations": ["emergency_stop", "status_report"],
                "restrictions": [
                    "All motion prohibited",
                    "Manual recovery required",
                    "System diagnostics needed"
                ]
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive redundancy status"""
        sensor_status = {
            name: {
                "health": sensor.get_health().value,
                "failed_channels": [
                    i for i, h in enumerate(sensor.channel_health)
                    if h == ComponentHealth.FAILED
                ],
                "failure_counts": sensor.failure_counts
            }
            for name, sensor in self.sensors.items()
        }
        
        watchdog_status = {
            name: {
                "enabled": wd.enabled,
                "triggered": wd.triggered,
                "time_since_reset": time.time() - wd.last_reset,
                "timeout": wd.timeout
            }
            for name, wd in self.watchdogs.items()
        }
        
        return {
            "system_health": self.system_health.value,
            "failsafe_active": self.failsafe_active,
            "degraded_components": self.degraded_components,
            "sensors": sensor_status,
            "watchdogs": watchdog_status,
            "degradation_mode": self.graceful_degradation(),
            "statistics": {
                "total_checks": self.total_checks,
                "disagreement_rate": f"{(self.disagreements / max(self.total_checks, 1)) * 100:.2f}%",
                "sensor_failures": self.sensor_failures
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def self_diagnostics(self) -> Dict[str, Any]:
        """
        Run comprehensive self-diagnostics
        Like spacecraft built-in test (BIT)
        """
        logger.info("🔧 Running self-diagnostics...")
        
        diagnostics = {
            "test_time": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "issues_found": []
        }
        
        # Test 1: Sensor agreement check
        diagnostics["tests_run"] += 1
        if self.disagreements / max(self.total_checks, 1) < 0.1:  # <10% disagreement
            diagnostics["tests_passed"] += 1
        else:
            diagnostics["tests_failed"] += 1
            diagnostics["issues_found"].append("High sensor disagreement rate")
        
        # Test 2: No failed sensors
        diagnostics["tests_run"] += 1
        if self.sensor_failures == 0:
            diagnostics["tests_passed"] += 1
        else:
            diagnostics["tests_failed"] += 1
            diagnostics["issues_found"].append(f"{self.sensor_failures} sensor failures detected")
        
        # Test 3: Watchdogs operational
        diagnostics["tests_run"] += 1
        triggered_wds = [name for name, wd in self.watchdogs.items() if wd.triggered]
        if len(triggered_wds) == 0:
            diagnostics["tests_passed"] += 1
        else:
            diagnostics["tests_failed"] += 1
            diagnostics["issues_found"].append(f"Watchdog timeouts: {triggered_wds}")
        
        # Test 4: System health
        diagnostics["tests_run"] += 1
        if self.system_health == ComponentHealth.NOMINAL:
            diagnostics["tests_passed"] += 1
        else:
            diagnostics["tests_failed"] += 1
            diagnostics["issues_found"].append(f"System health: {self.system_health.value}")
        
        diagnostics["overall_health"] = "PASS" if diagnostics["tests_failed"] == 0 else "FAIL"
        
        logger.info(f"🔧 Diagnostics complete: {diagnostics['tests_passed']}/{diagnostics['tests_run']} passed")
        
        return diagnostics


# To integrate with consciousness_service.py:
"""
class ConsciousnessService:
    def __init_embodiment__(self):
        # Add redundancy manager
        self.redundancy_manager = RedundancyManager()
        ...
    
    async def check_motion_safety(self, ...):
        # Use redundant sensors
        sensor_data = await self.redundancy_manager.check_all_sensors(self.body_state)
        
        # Check for degradation
        if sensor_data["system_health"] != "nominal":
            degradation = self.redundancy_manager.graceful_degradation()
            if "full_motion" not in degradation["allowed_operations"]:
                return {"safe": False, "reason": "system_degraded"}
        
        # Reset watchdog
        self.redundancy_manager.watchdogs["safety_check"].reset()
        ...
"""
