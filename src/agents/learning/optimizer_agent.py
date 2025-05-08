"""
Optimizer Agent

Handles parameter optimization for learning in the NIS Protocol.
"""

from typing import Dict, Any, Optional, List
import time
import numpy as np

from src.core.registry import NISAgent, NISLayer
from src.emotion.emotional_state import EmotionalState
from .learning_agent import LearningAgent

class OptimizerAgent(LearningAgent):
    """
    Agent responsible for optimizing learning parameters.
    
    This agent implements various optimization strategies to improve
    learning performance across the system.
    """
    
    def __init__(
        self,
        agent_id: str = "optimizer",
        description: str = "Optimizes learning parameters",
        emotional_state: Optional[EmotionalState] = None,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        decay: float = 0.0001
    ):
        """
        Initialize the optimizer agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            learning_rate: Base learning rate
            momentum: Momentum factor for optimization
            decay: Learning rate decay factor
        """
        super().__init__(agent_id, description, emotional_state, learning_rate)
        self.momentum = momentum
        self.decay = decay
        self.velocity = {}  # Stores momentum updates
        self.iteration = 0
        
    def _update_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update parameters using optimization strategy.
        
        Args:
            message: Message with update parameters
                'params': Dictionary of parameters to update
                'gradients': Dictionary of parameter gradients
                
        Returns:
            Update operation result
        """
        params = message.get("params", {})
        gradients = message.get("gradients", {})
        
        if not params or not gradients:
            return {
                "status": "error",
                "error": "Missing params or gradients",
                "agent_id": self.agent_id,
                "timestamp": time.time()
            }
        
        # Apply momentum optimization
        updated_params = {}
        for param_name, param_value in params.items():
            if param_name not in gradients:
                continue
                
            # Initialize velocity if not exists
            if param_name not in self.velocity:
                self.velocity[param_name] = np.zeros_like(param_value)
            
            # Update velocity
            self.velocity[param_name] = (
                self.momentum * self.velocity[param_name] +
                self.learning_rate * gradients[param_name]
            )
            
            # Update parameter
            updated_params[param_name] = param_value - self.velocity[param_name]
        
        # Update iteration count and decay learning rate
        self.iteration += 1
        self.learning_rate *= (1.0 / (1.0 + self.decay * self.iteration))
        
        return {
            "status": "success",
            "updated_params": updated_params,
            "current_learning_rate": self.learning_rate,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _get_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get current optimization parameters.
        
        Args:
            message: Message with parameter query
            
        Returns:
            Current parameter values
        """
        return {
            "status": "success",
            "parameters": {
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "decay": self.decay,
                "iteration": self.iteration
            },
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def _reset_parameters(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset optimization parameters.
        
        Args:
            message: Message with reset parameters
            
        Returns:
            Reset operation result
        """
        # Reset to initial values
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.decay = 0.0001
        self.velocity = {}
        self.iteration = 0
        
        return {
            "status": "success",
            "message": "Parameters reset to initial values",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
    
    def suggest_learning_rate(self, loss_history: List[float]) -> float:
        """
        Suggest an optimal learning rate based on loss history.
        
        Args:
            loss_history: List of recent loss values
            
        Returns:
            Suggested learning rate
        """
        if len(loss_history) < 2:
            return self.learning_rate
            
        # Calculate loss trend
        loss_diff = np.diff(loss_history)
        avg_diff = np.mean(loss_diff)
        
        if avg_diff > 0:  # Loss increasing
            return self.learning_rate * 0.5
        elif avg_diff < 0:  # Loss decreasing
            return self.learning_rate * 1.1
        else:
            return self.learning_rate 