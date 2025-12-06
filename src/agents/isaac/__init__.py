"""
NIS Protocol - NVIDIA Isaac Integration

Bridges NIS Protocol (cognitive layer) with NVIDIA Isaac (physical layer)

Components:
- IsaacBridgeAgent: ROS 2 communication bridge
- IsaacSimAgent: Isaac Sim integration for simulation
- IsaacPerceptionAgent: FoundationPose, SyntheticaDETR integration
- IsaacIntegrationManager: Unified manager for full pipeline

Architecture:
    NIS Protocol (Cognitive Layer)
    ├── Reasoning Agent → Query Router → Physics/KAN/Laplace
    ├── NIS Robotics Agent (FK/IK/Trajectory + Physics Validation)
    └── Isaac Integration Manager ↓
    
    NVIDIA Isaac (Physical Layer)
    ├── Isaac ROS (cuMotion, cuVSLAM, nvblox, FoundationPose)
    ├── Isaac Sim + Newton (Simulation)
    └── Jetson/Thor (Edge Deployment)
"""

from .isaac_bridge_agent import (
    IsaacBridgeAgent,
    get_isaac_bridge,
    IsaacConfig,
    RobotState,
    TrajectoryCommand
)

from .isaac_sim_agent import (
    IsaacSimAgent,
    get_isaac_sim,
    SimConfig,
    SyntheticDataConfig
)

from .isaac_perception_agent import (
    IsaacPerceptionAgent,
    get_isaac_perception,
    PerceptionConfig
)

from .isaac_integration import (
    IsaacIntegrationManager,
    IsaacIntegrationConfig,
    get_isaac_manager,
    initialize_isaac
)

__all__ = [
    # Bridge
    'IsaacBridgeAgent',
    'get_isaac_bridge',
    'IsaacConfig',
    'RobotState',
    'TrajectoryCommand',
    # Sim
    'IsaacSimAgent',
    'get_isaac_sim',
    'SimConfig',
    'SyntheticDataConfig',
    # Perception
    'IsaacPerceptionAgent',
    'get_isaac_perception',
    'PerceptionConfig',
    # Integration Manager
    'IsaacIntegrationManager',
    'IsaacIntegrationConfig',
    'get_isaac_manager',
    'initialize_isaac'
]
