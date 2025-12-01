"""
Base Agent
Abstract base class for all simulated agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class AgentState:
    """Agent state"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float64))  # Quaternion
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    
    def to_dict(self) -> Dict:
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "orientation": self.orientation.tolist(),
            "angular_velocity": self.angular_velocity.tolist()
        }


class BaseAgent(ABC):
    """
    Base class for simulated agents
    Subclass for drones, vehicles, satellites, etc.
    """
    
    def __init__(self, agent_id: str, initial_position: Tuple[float, float, float] = (0, 0, 0)):
        self.agent_id = agent_id
        self.state = AgentState(position=np.array(initial_position))
        self.body_id: Optional[int] = None
        self.physics_client: Optional[int] = None
        self.command_queue: list = []
        self.sensors: Dict[str, Any] = {}
        
    @abstractmethod
    def spawn(self, physics_client: int):
        """Spawn agent in physics simulation"""
        pass
    
    @abstractmethod
    def update(self, dt: float):
        """Update agent state"""
        pass
    
    @abstractmethod
    def apply_command(self, command: Dict[str, Any]):
        """Apply control command to agent"""
        pass
    
    def get_state(self) -> Dict:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.to_dict(),
            "sensors": self.sensors
        }
    
    def reset(self):
        """Reset agent to initial state"""
        self.state = AgentState()
        self.command_queue.clear()
    
    def despawn(self):
        """Remove agent from simulation"""
        if self.body_id is not None and self.physics_client is not None:
            try:
                import pybullet as p
                p.removeBody(self.body_id)
            except:
                pass
        self.body_id = None
    
    def queue_command(self, command: Dict[str, Any]):
        """Queue a command for execution"""
        self.command_queue.append(command)
    
    def _process_commands(self):
        """Process queued commands"""
        while self.command_queue:
            command = self.command_queue.pop(0)
            self.apply_command(command)
