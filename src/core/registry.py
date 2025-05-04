"""
NIS Protocol Registry Implementation

This module provides the registry class that manages all NIS Protocol agents.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
import time
from .agent import NISAgent, NISLayer


class NISLayer(Enum):
    """Layers in the NIS Protocol cognitive hierarchy."""
    
    PERCEPTION = "perception"
    INTERPRETATION = "interpretation"
    REASONING = "reasoning"
    MEMORY = "memory"
    ACTION = "action"
    LEARNING = "learning"
    COORDINATION = "coordination"


class NISAgent:
    """Base class for all NIS Protocol agents."""
    
    def __init__(
        self,
        agent_id: str,
        layer: NISLayer,
        description: str
    ):
        """
        Initialize a NIS Protocol agent.
        
        Args:
            agent_id: Unique identifier for the agent
            layer: Cognitive layer this agent belongs to
            description: Human-readable description of the agent's role
        """
        self.agent_id = agent_id
        self.layer = layer
        self.description = description
        self.active = True
        
        # Register with the global registry
        NISRegistry().register(self)
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
            
        Returns:
            Response message
        """
        raise NotImplementedError("Subclasses must implement process()")


class NISRegistry:
    """Central registry for all NIS Protocol agents.
    
    The registry is responsible for tracking all agents, their capabilities,
    and their current status. It facilitates communication between agents
    and manages their lifecycle.
    
    This class follows the Singleton pattern to ensure there is only one
    registry instance throughout the application.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NISRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.agents: Dict[str, NISAgent] = {}
        self.emotional_state = None
        self._initialized = True
    
    def register(self, agent: NISAgent) -> None:
        """Register an agent with the registry.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.agent_id] = agent
    
    def get_agents_by_layer(self, layer: NISLayer) -> List[NISAgent]:
        """Get all agents in a specific layer.
        
        Args:
            layer: The layer to filter by
            
        Returns:
            A list of active agents in the specified layer
        """
        return [
            agent for agent in self.agents.values()
            if agent.layer == layer and agent.active
        ]
    
    def get_agent_by_id(self, agent_id: str) -> Optional[NISAgent]:
        """Get an agent by its ID.
        
        Args:
            agent_id: The ID of the agent to get
            
        Returns:
            The agent with the specified ID, or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_emotional_state(self):
        """Get the current emotional state.
        
        Returns:
            The current emotional state object
        
        Raises:
            ValueError: If the emotional state has not been set
        """
        if self.emotional_state is None:
            raise ValueError("Emotional state has not been set")
        return self.emotional_state
    
    def set_emotional_state(self, emotional_state) -> None:
        """Set the emotional state object.
        
        Args:
            emotional_state: The emotional state object to set
        """
        self.emotional_state = emotional_state
    
    def process_message(self, message: Dict[str, Any], target_layer: NISLayer) -> List[Dict[str, Any]]:
        """Process a message through all agents in a specific layer.
        
        Args:
            message: The message to process
            target_layer: The layer to process the message through
            
        Returns:
            A list of processed messages from each agent in the target layer
        """
        responses = []
        
        for agent in self.get_agents_by_layer(target_layer):
            try:
                response = agent.process(message)
                responses.append(response)
            except Exception as e:
                # Create an error response
                error_response = {
                    "agent_id": agent.agent_id,
                    "timestamp": time.time(),
                    "status": "error",
                    "payload": {"error": str(e)},
                    "metadata": {"exception_type": type(e).__name__},
                    "emotional_state": message.get("emotional_state", {})
                }
                responses.append(error_response)
        
        return responses
    
    def shutdown(self) -> None:
        """Shut down all agents.
        
        This method sets all agents to inactive, effectively shutting down
        the system.
        """
        for agent in self.agents.values():
            agent.set_active(False)
            
    def reset(self) -> None:
        """Reset the registry.
        
        This method clears all registered agents and resets the emotional state.
        """
        self.agents = {}
        self.emotional_state = None 