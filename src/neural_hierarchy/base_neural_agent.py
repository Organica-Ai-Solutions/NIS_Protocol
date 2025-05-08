from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class NeuralLayer(Enum):
    SENSORY = "sensory"          # Input processing
    PERCEPTION = "perception"    # Pattern recognition and interpretation
    EXECUTIVE = "executive"      # Decision making and control
    MEMORY = "memory"           # Information storage and retrieval
    EMOTIONAL = "emotional"     # Emotional processing and regulation
    MOTOR = "motor"            # Output generation and action

@dataclass
class NeuralSignal:
    """Base class for neural signals passed between agents"""
    source_layer: NeuralLayer
    target_layer: NeuralLayer
    content: Any
    timestamp: datetime = datetime.now()
    priority: float = 1.0  # Signal priority/strength

class NeuralAgent(ABC):
    """Base class for all neural agents"""
    
    def __init__(
        self,
        agent_id: str,
        layer: NeuralLayer,
        description: str = "",
        activation_threshold: float = 0.5
    ):
        self.agent_id = agent_id
        self.layer = layer
        self.description = description
        self.activation_threshold = activation_threshold
        self.activation_level = 0.0
        self.connected_agents = {}
        self.state = {}
        
    @abstractmethod
    def process_signal(self, signal: NeuralSignal) -> Optional[NeuralSignal]:
        """Process incoming neural signal and optionally generate response"""
        pass
    
    def connect_to(self, other_agent: 'NeuralAgent', connection_strength: float = 1.0):
        """Connect this agent to another agent"""
        self.connected_agents[other_agent.agent_id] = {
            'agent': other_agent,
            'strength': connection_strength
        }
    
    def send_signal(self, target_agent_id: str, content: Any) -> Optional[NeuralSignal]:
        """Send a signal to a connected agent"""
        if target_agent_id not in self.connected_agents:
            return None
            
        target = self.connected_agents[target_agent_id]['agent']
        strength = self.connected_agents[target_agent_id]['strength']
        
        signal = NeuralSignal(
            source_layer=self.layer,
            target_layer=target.layer,
            content=content,
            priority=strength * self.activation_level
        )
        
        return target.process_signal(signal)
    
    def update_activation(self, signal_strength: float):
        """Update agent's activation level based on input strength"""
        self.activation_level = min(
            1.0,
            self.activation_level + signal_strength
        )
        
        # Decay activation over time
        self.activation_level *= 0.9
    
    def is_active(self) -> bool:
        """Check if agent is active based on activation threshold"""
        return self.activation_level >= self.activation_threshold
    
    def update_state(self, updates: Dict[str, Any]):
        """Update agent's internal state"""
        self.state.update(updates)
    
    def reset(self):
        """Reset agent to initial state"""
        self.activation_level = 0.0
        self.state = {} 