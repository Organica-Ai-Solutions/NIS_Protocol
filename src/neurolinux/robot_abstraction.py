"""
NIS Protocol — Robot Abstraction Layer
Universal robot interface used by NeuroLinux drivers.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("neurolinux.robot_abstraction")


class RobotCategory(str, Enum):
    ARM = "arm"
    MOBILE = "mobile"
    HUMANOID = "humanoid"
    GRIPPER = "gripper"
    DUAL_ARM = "dual_arm"
    UNKNOWN = "unknown"


class ControlMode(str, Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"
    HYBRID = "hybrid"


class CommunicationProtocol(str, Enum):
    SERIAL = "serial"
    USB = "usb"
    ETHERNET = "ethernet"
    CAN = "can"
    ROS = "ros"
    SIMULATION = "simulation"


@dataclass
class RobotCapabilities:
    """Describes what a robot can do."""
    dof: int = 6
    has_gripper: bool = True
    has_force_sensor: bool = False
    has_vision: bool = False
    max_payload_kg: float = 0.5
    max_reach_mm: float = 400.0
    control_modes: List[ControlMode] = field(default_factory=lambda: [ControlMode.POSITION])
    communication: CommunicationProtocol = CommunicationProtocol.SERIAL
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    """Current state snapshot of a robot."""
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    joint_torques: List[float] = field(default_factory=list)
    end_effector_pose: Optional[List[float]] = None   # [x, y, z, rx, ry, rz]
    gripper_opening_mm: float = 0.0
    is_moving: bool = False
    is_homed: bool = False
    error_code: int = 0
    error_message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotCommand:
    """A command to send to a robot."""
    command_type: str = "move"                        # move, home, stop, gripper, named
    joint_targets: Optional[List[float]] = None
    pose_target: Optional[List[float]] = None         # [x, y, z, rx, ry, rz]
    named_position: Optional[str] = None              # home, wave, pick_table, place_bin …
    gripper_target_mm: Optional[float] = None
    velocity_scale: float = 0.5
    acceleration_scale: float = 0.5
    wait: bool = True
    timeout_s: float = 30.0
    extra: Dict[str, Any] = field(default_factory=dict)


class RobotInterface(ABC):
    """Abstract base class for all NeuroLinux robot drivers."""

    def __init__(self, name: str, category: RobotCategory, capabilities: RobotCapabilities):
        self.name = name
        self.category = category
        self.capabilities = capabilities
        self.logger = logging.getLogger(f"neurolinux.robot.{name}")
        self._connected = False
        self._state = RobotState()

    # ── Connection lifecycle ──────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """Open connection to the robot hardware."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection cleanly."""

    @property
    def connected(self) -> bool:
        return self._connected

    # ── State ─────────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_state(self) -> RobotState:
        """Read current robot state."""

    # ── Commands ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def execute(self, command: RobotCommand) -> Dict[str, Any]:
        """Execute a RobotCommand. Returns {ok, message, state}."""

    async def home(self) -> Dict[str, Any]:
        return await self.execute(RobotCommand(command_type="named", named_position="home"))

    async def stop(self) -> Dict[str, Any]:
        return await self.execute(RobotCommand(command_type="stop"))

    async def open_gripper(self) -> Dict[str, Any]:
        return await self.execute(RobotCommand(command_type="gripper", gripper_target_mm=35.0))

    async def close_gripper(self) -> Dict[str, Any]:
        return await self.execute(RobotCommand(command_type="gripper", gripper_target_mm=0.0))

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "connected": self._connected,
            "capabilities": {
                "dof": self.capabilities.dof,
                "has_gripper": self.capabilities.has_gripper,
                "max_payload_kg": self.capabilities.max_payload_kg,
                "communication": self.capabilities.communication,
            },
        }


# ── Robot Registry ────────────────────────────────────────────────────────────

class RobotRegistry:
    """
    Global registry of named robot instances.
    Allows any module to look up a robot by name without circular imports.
    """

    _robots: Dict[str, "RobotInterface"] = {}
    _driver_classes: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, robot: "RobotInterface") -> None:
        """Register a connected robot instance."""
        cls._robots[name] = robot
        logger.info(f"Robot registered: {name} ({robot.category})")

    @classmethod
    def register_driver(cls, driver_name: str, driver_class: type) -> None:
        """Register a driver class so create_robot can instantiate it by name."""
        cls._driver_classes[driver_name] = driver_class

    @classmethod
    def get(cls, name: str) -> Optional["RobotInterface"]:
        """Return a registered robot by name, or None."""
        return cls._robots.get(name)

    @classmethod
    def list_robots(cls) -> List[str]:
        """Return names of all registered robots."""
        return list(cls._robots.keys())

    @classmethod
    def list_drivers(cls) -> List[str]:
        """Return names of all registered driver classes."""
        return list(cls._driver_classes.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._robots.clear()
        cls._driver_classes.clear()


# ── Convenience functions ─────────────────────────────────────────────────────

def get_robot(name: str) -> Optional[RobotInterface]:
    """Look up a robot instance by name from the global registry."""
    return RobotRegistry.get(name)


def list_available_drivers() -> List[str]:
    """Return all driver names registered in the global registry."""
    return RobotRegistry.list_drivers()


def create_robot(driver_name: str, robot_name: str, **kwargs: Any) -> RobotInterface:
    """
    Instantiate a robot driver by name using the registry.

    Args:
        driver_name: Key used when the driver was registered via
                     RobotRegistry.register_driver().
        robot_name:  Human-readable name for this robot instance.
        **kwargs:    Forwarded to the driver constructor.

    Raises:
        ValueError: If driver_name is not registered.
    """
    cls = RobotRegistry._driver_classes.get(driver_name)
    if cls is None:
        available = list_available_drivers()
        raise ValueError(
            f"Driver '{driver_name}' not registered. "
            f"Available: {available or ['none — register with RobotRegistry.register_driver()']}"
        )
    instance = cls(name=robot_name, **kwargs)
    RobotRegistry.register(robot_name, instance)
    return instance
