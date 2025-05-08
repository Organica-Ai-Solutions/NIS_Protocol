"""
Learning Agent Base Class

Provides the foundation for learning and adaptation in the NIS Protocol.
"""

from typing import Dict, Any, Optional
import time

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState

class LearningAgent(NISAgent):
    """
    Base class for learning agents in the NIS Protocol.
    
    This agent provides the foundation for implementing different learning
    strategies and mechanisms for system adaptation.
    """
    
    def __init__(
        self,
        agent_id: str,
        description: str = "Base learning agent",
        emotional_state: Optional[EmotionalState] = None,
        learning_rate: float = 0.1
    ):
        """
        Initialize the learning agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            learning_rate: Base learning rate for parameter updates
        """
        super().__init__(agent_id, NISLayer.LEARNING, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.learning_rate = learning_rate
        
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a learning-related request.
        
        Args:
            message: Message containing learning operation
                'operation': Operation to perform
                + Additional parameters based on operation
                
        Returns:
            Result of the learning operation
        """
        operation = message.get("operation", "").lower()
        
        if operation == "update":
            return self._update_parameters(message)
        elif operation == "get_params":
            return self._get_parameters(message)
        elif operation == "reset":
            return self._reset_parameters(message)
        else:
            return {
                "status": "error",
                "error": f"Unknown operation: {operation}",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
    
    def _update_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update learning parameters based on feedback.
        
        Args:
            message: Message with update parameters
            
        Returns:
            Update operation result
        """
        raise NotImplementedError("Subclasses must implement _update_parameters()")
    
    def _get_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get current learning parameters.
        
        Args:
            message: Message with parameter query
            
        Returns:
            Current parameter values
        """
        raise NotImplementedError("Subclasses must implement _get_parameters()")
    
    def _reset_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset learning parameters to initial values.
        
        Args:
            message: Message with reset parameters
            
        Returns:
            Reset operation result
        """
        raise NotImplementedError("Subclasses must implement _reset_parameters()")
    
    def adjust_learning_rate(self, factor: float) -> None:
        """
        Adjust the learning rate by a factor.
        
        Args:
            factor: Multiplier for the learning rate
        """
        self.learning_rate *= factor 