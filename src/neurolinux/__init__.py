"""
NeuroLinux - Edge AI for Robotics
Agentic control system for Pi 5, Jetson, and Drones

Copyright 2026 Organica AI Solutions
Licensed under Apache License 2.0
"""

from .device_client import NeuroLinuxDeviceClient
from .agentic_control import (
    AgentOrchestrator,
    PerceptionAgent,
    ReasoningAgent,
    ActionAgent,
    SafetyAgent,
    AutonomyTier,
    DeviceType,
    ActionType,
    Observation,
    Action,
    Plan,
    create_agentic_controller
)
from .framework_bridges import (
    FrameworkManager,
    RAIBridge,
    ROSABridge,
    LeRobotBridge,
    RAIConfig,
    ROSAConfig,
    LeRobotConfig
)
from .vla_controller import (
    VLAController,
    VLAConfig,
    VLAModelType,
    VLAObservation,
    VLAAction,
    create_vla_controller,
    auto_select_model_for_hardware
)
from .hardware_adapters import (
    HardwareAdapter,
    HardwareConfig,
    HardwareType,
    SensorData,
    MotorCommand,
    RaspberryPi5Adapter,
    JetsonAdapter,
    DroneAdapter,
    RobotArmCANAdapter,
    create_hardware_adapter
)
from .nis_integration import (
    NISLLMClient,
    NISLLMConfig,
    UnifiedRoboticsAgentBridge,
    H100ModelClient,
    NISWebSocketClient,
    NISIntegratedClient,
    create_nis_client
)
from .edge_deployment import (
    EdgeDeploymentManager,
    EdgePlatform,
    EdgeCapabilities,
    HardwareDetector,
    EdgeModelOptimizer,
    OfflineModeManager,
    EdgeHealthMonitor,
    ModelConfig,
    create_edge_deployment_manager,
    auto_setup_edge
)
from .robot_abstraction import (
    RobotInterface,
    RobotRegistry,
    RobotCategory,
    RobotCapabilities,
    RobotState,
    RobotCommand,
    ControlMode,
    CommunicationProtocol,
    create_robot,
    list_available_drivers,
    get_robot
)
from .ros2_bridge import (
    ROS2Bridge,
    ROS2Config,
    ROS2_AVAILABLE,
    create_ros2_bridge
)
from .isaac_ros_bridge import (
    IsaacROSBridge,
    IsaacROSConfig,
    IsaacROSCapability,
    IsaacSimBridge,
    ISAAC_ROS_AVAILABLE,
    create_isaac_ros_bridge,
    create_isaac_sim_bridge
)
from .robot_drivers import (
    UniversalRobotDriver,
    KUKADriver,
    SpotDriver,
    UnitreeDriver,
    FrankaDriver,
    ClearpathDriver,
    ABBDriver,
    FanucDriver,
    list_supported_robots
)
from .pc_bridge import (
    PCBridge,
    PCAgentServer,
    PCType,
    PCProfile,
    ComputeJob,
    JobType,
    JobStatus,
    PeerNode,
    ALIENWARE_AREA51_R2_DEFAULTS,
)
from .robot_config import (
    RobotConfig,
    RobotConfigManager,
    ConnectionConfig,
    GripperConfig,
    CameraConfig,
    SafetyConfig,
    ControlConfig,
    VLAConfig,
    RobotDiscovery,
    load_robot_config,
    save_robot_config,
    create_default_config,
    discover_robots
)

__all__ = [
    # Device Client
    "NeuroLinuxDeviceClient",
    
    # Agentic Control
    "AgentOrchestrator",
    "PerceptionAgent",
    "ReasoningAgent",
    "ActionAgent",
    "SafetyAgent",
    "AutonomyTier",
    "DeviceType",
    "ActionType",
    "Observation",
    "Action",
    "Plan",
    "create_agentic_controller",
    
    # Framework Bridges
    "FrameworkManager",
    "RAIBridge",
    "ROSABridge",
    "LeRobotBridge",
    "RAIConfig",
    "ROSAConfig",
    "LeRobotConfig",
    
    # VLA Controller
    "VLAController",
    "VLAConfig",
    "VLAModelType",
    "VLAObservation",
    "VLAAction",
    "create_vla_controller",
    "auto_select_model_for_hardware",
    
    # Hardware Adapters
    "HardwareAdapter",
    "HardwareConfig",
    "HardwareType",
    "SensorData",
    "MotorCommand",
    "RaspberryPi5Adapter",
    "JetsonAdapter",
    "DroneAdapter",
    "RobotArmCANAdapter",
    "create_hardware_adapter",
    
    # NIS Integration
    "NISLLMClient",
    "NISLLMConfig",
    "UnifiedRoboticsAgentBridge",
    "H100ModelClient",
    "NISWebSocketClient",
    "NISIntegratedClient",
    "create_nis_client",
    
    # Edge Deployment
    "EdgeDeploymentManager",
    "EdgePlatform",
    "EdgeCapabilities",
    "HardwareDetector",
    "EdgeModelOptimizer",
    "OfflineModeManager",
    "EdgeHealthMonitor",
    "ModelConfig",
    "create_edge_deployment_manager",
    "auto_setup_edge",
    
    # Robot Abstraction
    "RobotInterface",
    "RobotRegistry",
    "RobotCategory",
    "RobotCapabilities",
    "RobotState",
    "RobotCommand",
    "ControlMode",
    "CommunicationProtocol",
    "create_robot",
    "list_available_drivers",
    "get_robot",
    
    # ROS 2 Bridge
    "ROS2Bridge",
    "ROS2Config",
    "ROS2_AVAILABLE",
    "create_ros2_bridge",
    
    # Isaac ROS Bridge
    "IsaacROSBridge",
    "IsaacROSConfig",
    "IsaacROSCapability",
    "IsaacSimBridge",
    "ISAAC_ROS_AVAILABLE",
    "create_isaac_ros_bridge",
    "create_isaac_sim_bridge",
    
    # Robot Drivers
    "UniversalRobotDriver",
    "KUKADriver",
    "SpotDriver",
    "UnitreeDriver",
    "FrankaDriver",
    "ClearpathDriver",
    "ABBDriver",
    "FanucDriver",
    "list_supported_robots",
    
    # Robot Configuration
    "RobotConfig",
    "RobotConfigManager",
    "ConnectionConfig",
    "GripperConfig",
    "CameraConfig",
    "SafetyConfig",
    "ControlConfig",
    "VLAConfig",
    "RobotDiscovery",
    "load_robot_config",
    "save_robot_config",
    "create_default_config",
    "discover_robots",
    
    # PC Bridge
    "PCBridge",
    "PCAgentServer",
    "PCType",
    "PCProfile",
    "ComputeJob",
    "JobType",
    "JobStatus",
    "PeerNode",
    "ALIENWARE_AREA51_R2_DEFAULTS",
]

__version__ = "1.0.0"
