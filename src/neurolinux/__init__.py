"""
NeuroLinux - Edge AI for Robotics
Agentic control system for Pi 5, Jetson, and Drones

Copyright 2026 Organica AI Solutions
Licensed under Apache License 2.0
"""

# ── Always-available modules (files confirmed on disk) ────────────────────────
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
    auto_setup_edge,
)
from .robot_abstraction import (
    RobotInterface,
    RobotCategory,
    RobotCapabilities,
    RobotState,
    RobotCommand,
    ControlMode,
    CommunicationProtocol,
    # Registry helpers added by fix
    RobotRegistry,
    create_robot,
    list_available_drivers,
    get_robot,
)
from .pc_bridge import (
    PCBridge,
    PCType,
    PCProfile,
    ComputeJob,
    JobType,
    JobStatus,
    PeerNode,
    ALIENWARE_AREA51_R2_DEFAULTS,
)

# ── Optional modules — guarded imports (files not yet present) ─────────────────
try:
    from .device_client import NeuroLinuxDeviceClient
except ImportError:
    NeuroLinuxDeviceClient = None  # type: ignore

try:
    from .agentic_control import (
        AgentOrchestrator, PerceptionAgent, ReasoningAgent,
        ActionAgent, SafetyAgent, AutonomyTier, DeviceType,
        ActionType, Observation, Action, Plan, create_agentic_controller,
    )
except ImportError:
    AgentOrchestrator = PerceptionAgent = ReasoningAgent = ActionAgent = None  # type: ignore
    SafetyAgent = AutonomyTier = DeviceType = ActionType = None  # type: ignore
    Observation = Action = Plan = create_agentic_controller = None  # type: ignore

try:
    from .framework_bridges import (
        FrameworkManager, RAIBridge, ROSABridge, LeRobotBridge,
        RAIConfig, ROSAConfig, LeRobotConfig,
    )
except ImportError:
    FrameworkManager = RAIBridge = ROSABridge = LeRobotBridge = None  # type: ignore
    RAIConfig = ROSAConfig = LeRobotConfig = None  # type: ignore

try:
    from .vla_controller import (
        VLAController, VLAConfig, VLAModelType, VLAObservation,
        VLAAction, create_vla_controller, auto_select_model_for_hardware,
    )
except ImportError:
    VLAController = VLAConfig = VLAModelType = VLAObservation = None  # type: ignore
    VLAAction = create_vla_controller = auto_select_model_for_hardware = None  # type: ignore

try:
    from .hardware_adapters import (
        HardwareAdapter, HardwareConfig, HardwareType, SensorData, MotorCommand,
        RaspberryPi5Adapter, JetsonAdapter, DroneAdapter,
        RobotArmCANAdapter, create_hardware_adapter,
    )
except ImportError:
    HardwareAdapter = HardwareConfig = HardwareType = SensorData = MotorCommand = None  # type: ignore
    RaspberryPi5Adapter = JetsonAdapter = DroneAdapter = None  # type: ignore
    RobotArmCANAdapter = create_hardware_adapter = None  # type: ignore

try:
    from .nis_integration import (
        NISLLMClient, NISLLMConfig, UnifiedRoboticsAgentBridge,
        H100ModelClient, NISWebSocketClient, NISIntegratedClient, create_nis_client,
    )
except ImportError:
    NISLLMClient = NISLLMConfig = UnifiedRoboticsAgentBridge = None  # type: ignore
    H100ModelClient = NISWebSocketClient = NISIntegratedClient = create_nis_client = None  # type: ignore

try:
    from .ros2_bridge import (ROS2Bridge, ROS2Config, ROS2_AVAILABLE, create_ros2_bridge)
except ImportError:
    ROS2Bridge = ROS2Config = create_ros2_bridge = None  # type: ignore
    ROS2_AVAILABLE = False

try:
    from .isaac_ros_bridge import (
        IsaacROSBridge, IsaacROSConfig, IsaacROSCapability,
        IsaacSimBridge, ISAAC_ROS_AVAILABLE,
        create_isaac_ros_bridge, create_isaac_sim_bridge,
    )
except ImportError:
    IsaacROSBridge = IsaacROSConfig = IsaacROSCapability = None  # type: ignore
    IsaacSimBridge = create_isaac_ros_bridge = create_isaac_sim_bridge = None  # type: ignore
    ISAAC_ROS_AVAILABLE = False

try:
    from .robot_drivers import (
        UniversalRobotDriver, KUKADriver, SpotDriver, UnitreeDriver,
        FrankaDriver, ClearpathDriver, ABBDriver, FanucDriver, list_supported_robots,
    )
except ImportError:
    UniversalRobotDriver = KUKADriver = SpotDriver = UnitreeDriver = None  # type: ignore
    FrankaDriver = ClearpathDriver = ABBDriver = FanucDriver = None  # type: ignore
    list_supported_robots = None  # type: ignore

try:
    from .pc_bridge import PCAgentServer
except ImportError:
    PCAgentServer = None  # type: ignore

try:
    from .robot_config import (
        RobotConfig, RobotConfigManager, ConnectionConfig, GripperConfig,
        CameraConfig, SafetyConfig, ControlConfig, VLAConfig,
        RobotDiscovery, load_robot_config, save_robot_config,
        create_default_config, discover_robots,
    )
except ImportError:
    RobotConfig = RobotConfigManager = ConnectionConfig = GripperConfig = None  # type: ignore
    CameraConfig = SafetyConfig = ControlConfig = VLAConfig = None  # type: ignore
    RobotDiscovery = load_robot_config = save_robot_config = None  # type: ignore
    create_default_config = discover_robots = None  # type: ignore

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
