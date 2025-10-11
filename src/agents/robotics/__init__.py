"""
NIS Protocol Robotics Agents

Physics-validated robotics control system supporting:
- Drones (quadcopters, hexacopters)
- Humanoid robots (droids)
- Robotic manipulators
- Ground vehicles

Universal translator between robotic platforms using physics as common language.
"""

from .unified_robotics_agent import (
    UnifiedRoboticsAgent,
    RobotType,
    ControlMode,
    RobotState,
    PhysicsConstraints,
    TrajectoryPoint
)

__all__ = [
    'UnifiedRoboticsAgent',
    'RobotType',
    'ControlMode',
    'RobotState',
    'PhysicsConstraints',
    'TrajectoryPoint'
]

